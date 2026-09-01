"""Stage orchestration: ingest -> condition -> detect -> isolate -> measure -> report.

Implements the CLAUDE.md §3 architecture diagram end to end. Each stage is its own
package and each still runs alone from the CLI (§3: "that matters when debugging in front
of judges"); this module is only the wiring between them.

Stage 5 -- Classify runs the §5 ensemble: the §5.2 feature classifier, the §5.3 CNN, the
§5.4 deterministic rules, and the §5.5 referee, with §5.6 evidence sentences. Both learned
models work untrained and abstain with a reason, so on a fresh clone with no checkpoints a
detection still gets a physically-grounded label where a rule fires (OFDM, chirp, CW,
noise, SSB) and ``unclassified`` at confidence 0 where none does -- never a guess.

Two conventions worth stating once, because every number downstream depends on them:

* **Conditioning scale.** §2 Stage 2 normalises to unit average power, so every power
  measured on the conditioned signal is relative to that. ``_dbfs_offset`` converts back
  to the original full-scale reference so ``power_dbfs`` and ``noise_floor_dbfs`` mean what
  their names say.
* **Frequency axis.** Estimators measure offsets from the capture's own centre. Absolute
  RF is ``capture.center_freq + offset``, and is reported as ``null`` when the centre
  frequency is unknown -- §3 forbids inventing one. When the *sample rate* is unknown too,
  ``CaptureMeta.frequencies_are_normalised`` is set and every "Hz" is a fraction of the
  sample rate, which the report says out loud.

Whole-file analysis only: §2 Stage 2's overlapped block processing for multi-gigabyte
captures is Phase 8 hardening, and a capture large enough to need it gets a warning rather
than silent truncation.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from sigscope.dsp.condition import condition
from sigscope.dsp.detect import DetectorConfig, detect_bursts
from sigscope.dsp.estimators import (
    EstimatorConfig,
    estimate_am_depth,
    estimate_bandwidth,
    estimate_center_freq,
    estimate_chirp,
    estimate_cw_keying,
    estimate_fm_deviation,
    estimate_fsk_params,
    estimate_ofdm_params,
    estimate_psk_order,
    estimate_snr,
    estimate_spectral_asymmetry,
    estimate_symbol_rate,
    isolate_burst,
)
from sigscope.dsp.noise import NoiseEstimate
from sigscope.dsp.spectrogram import Spectrogram
from sigscope.features.cumulants import cumulants
from sigscope.io import read_capture
from sigscope.models import Classifiers, classify, load_classifiers
from sigscope.types import (
    Bandwidth,
    Burst,
    CaptureMeta,
    Detection,
    Estimate,
    FileInfo,
    Modulation,
    Report,
)

__all__ = ["AnalysisConfig", "Analysis", "analyse_file", "analyse_iq", "file_info"]

UNCLASSIFIED = "unclassified"

# a capture bigger than this is analysed whole anyway, but warned about (§2 Stage 2)
LARGE_CAPTURE_SAMPLES = 1 << 26  # 64 M samples = 512 MB as complex64


@dataclass
class AnalysisConfig:
    """Everything the pipeline can be tuned by, in one place."""

    detector: DetectorConfig = field(default_factory=DetectorConfig)
    estimators: EstimatorConfig = field(default_factory=EstimatorConfig)
    # a pathological file can label thousands of specks; cap the work and say so
    max_detections: int = 200
    # Stage 5 -- Classify. Set False to run measurement only.
    classify: bool = True


@dataclass
class Analysis:
    """The report plus the intermediate products the writers need.

    ``spectrogram`` and ``noise`` are kept because the HTML report renders the same
    spectrogram the detector actually used -- drawing a differently-parameterised one
    would put the boxes in visibly wrong places.
    """

    report: Report
    spectrogram: Spectrogram | None
    noise: NoiseEstimate | None
    bursts: list[Burst] = field(default_factory=list)


def file_info(path: Path) -> FileInfo:
    """SHA-256 and size of the input, streamed so a 2 GB capture does not land in RAM."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return FileInfo(name=path.name, sha256=digest.hexdigest(), bytes=path.stat().st_size)


def _dbfs_offset(scale: float) -> float:
    """dB to add to a conditioned-domain power to reference it to the original full scale.

    ``condition`` divides by the signal's RMS, so the original mean power is ``scale**2``
    and the correction is ``20*log10(scale)``.
    """
    return 20.0 * math.log10(scale) if scale > 0 else 0.0


def _absolute_freq(meta: CaptureMeta, offset_hz: float | None) -> float | None:
    """Capture-centre offset -> absolute RF, or ``None`` when the centre is unknown."""
    if offset_hz is None or meta.center_freq is None:
        return None
    return float(meta.center_freq + offset_hz)


def _finite(value: float | None) -> float | None:
    """Keep only real, finite numbers -- a NaN in the JSON is a fabricated result."""
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


# --------------------------------------------------------------------------------------
# Stage 4/5 -- measure one burst and turn it into a Detection
# --------------------------------------------------------------------------------------


def _evidence_for(
    snr_db: float | None,
    am: Estimate,
    extra: dict,
    notes: list[str],
) -> list[str]:
    """Plain sentences describing what was measured (CLAUDE.md §5.6).

    §3 requires at least two, always. These come from the §4 estimators rather than from a
    classifier, so every one of them is already a true, checkable statement about the
    signal even while the modulation label is ``unclassified``.
    """
    evidence: list[str] = []

    if extra.get("ofdm_subcarrier_spacing_hz") is not None:
        evidence.append(
            f"Autocorrelation peaks at lag {extra['ofdm_useful_symbol_len']:.0f} samples, "
            f"consistent with a cyclic prefix; subcarrier spacing "
            f"{extra['ofdm_subcarrier_spacing_hz']:.1f} Hz."
        )
    if extra.get("chirp_rate_hz_per_s") is not None:
        evidence.append(
            f"The spectrogram ridge fits a straight line, sweeping "
            f"{extra['chirp_rate_hz_per_s']:.3e} Hz per second."
        )
    if extra.get("n_tones") is not None and extra.get("fsk_tone_spacing_hz") is not None:
        evidence.append(
            f"The instantaneous frequency histogram has {extra['n_tones']:.0f} distinct "
            f"peaks spaced {extra['fsk_tone_spacing_hz']:.0f} Hz apart."
        )
    if extra.get("psk_order") is not None:
        order = extra["psk_order"]
        evidence.append(
            f"Raising to the power {order:.0f} produced the sharpest single spectral line, "
            f"which is consistent with {order:.0f} phase states."
        )
    if extra.get("cumulant_c40_ratio") is not None:
        evidence.append(
            f"The fourth-order cumulant ratio |C40|/C21^2 measures "
            f"{extra['cumulant_c40_ratio']:.2f}; the reference values are BPSK 2.00, "
            f"QPSK 1.00, 8PSK 0.00."
        )
    if am.value is not None and any("nearly constant" in n for n in am.notes):
        evidence.append(
            f"Envelope is nearly constant (modulation depth {am.value:.3f}), "
            "so this is not an amplitude scheme."
        )
    if extra.get("morse_dash_dot_ratio") is not None:
        evidence.append(
            f"On-durations cluster into two groups with a ratio of "
            f"{extra['morse_dash_dot_ratio']:.2f}, which is Morse timing."
        )

    # §5.6: "the single most credibility-building line in the product"
    if snr_db is not None and snr_db < 5.0:
        evidence.append(
            f"Measured SNR is {snr_db:.1f} dB. Below 5 dB several of our estimators lose "
            "accuracy, so treat this result with caution."
        )
    elif snr_db is not None:
        evidence.append(f"Measured signal-to-noise ratio is {snr_db:.1f} dB.")

    for note in notes:
        if len(evidence) >= 4:
            break
        sentence = (note[0].upper() + note[1:]) if note else note
        if sentence and sentence not in evidence:
            evidence.append(sentence)

    if len(evidence) < 2:
        evidence.append(
            "Modulation classification is not wired in yet (Phase 6), so no label is "
            "claimed for this detection."
        )
    return evidence[:5]


def _unmeasurable(
    burst: Burst,
    index: int,
    meta: CaptureMeta,
    offset_hz: float | None,
    snr_db: float | None,
    reason: str,
) -> Detection:
    """A detection we found but could not measure -- reported, never dropped."""
    return Detection(
        id=index,
        time_start_s=float(burst.t0),
        time_stop_s=float(burst.t1),
        duration_s=float(burst.duration),
        center_freq_hz=_absolute_freq(meta, offset_hz),
        center_freq_offset_hz=offset_hz,
        bandwidth_hz=Bandwidth(),
        snr_db=snr_db,
        power_dbfs=None,
        modulation=Modulation(
            label=UNCLASSIFIED,
            confidence=0.0,
            votes={"feature_clf": None, "cnn": None, "rules": None},
            evidence=[
                f"This detection could not be measured: {reason}.",
                "It is reported anyway so the count of signals found stays honest.",
            ],
        ),
    )


def _measure_burst(
    burst: Burst,
    index: int,
    x: np.ndarray,
    meta: CaptureMeta,
    spec: Spectrogram,
    cfg: AnalysisConfig,
    dbfs_offset: float,
    warnings: list[str],
    classifiers: Classifiers | None = None,
) -> Detection:
    """Run §4.4-§4.13 over one box, classify it (§5), and assemble its ``Detection``."""
    fs = meta.sample_rate
    extra: dict = {}
    notes: list[str] = []

    # §4.7 SNR first -- §4.6 wants it, and §5.6 quotes it in the evidence
    snr_result = estimate_snr(spec, burst, cfg=cfg.estimators)
    snr_db = _finite(snr_result.snr.value)
    if snr_result.below_floor:
        warnings.append(
            f"detection {index}: signal power did not exceed the in-band noise; "
            "SNR reported as below 0 dB"
        )

    # §4.5 centre frequency, measured on the capture's own axis
    centre = estimate_center_freq(x, fs, burst, cfg=cfg.estimators)
    offset_hz = _finite(centre.centroid.value)
    if centre.n_humps > 1:
        warnings.append(
            f"detection {index}: the in-box spectrum has {centre.n_humps} humps and may be "
            "a blend of more than one signal"
        )
    if centre.peak.value is not None:
        extra["center_freq_peak_hz"] = _finite(centre.peak.value)
    extra["spectrum_humps"] = centre.n_humps

    # §4.4 isolation -- everything below runs on the clean burst
    try:
        iso = isolate_burst(x, fs, burst, cfg=cfg.estimators)
    except ValueError as exc:
        warnings.append(f"detection {index}: could not be isolated ({exc})")
        return _unmeasurable(burst, index, meta, offset_hz, snr_db, str(exc))

    extra["decimation"] = iso.decimation
    extra["isolated_sample_rate_hz"] = iso.fs_b
    notes += iso.notes

    # §4.6 bandwidth, told the SNR so its confidence can fall with it
    bandwidth = estimate_bandwidth(iso.y, iso.fs_b, snr_db=snr_db, cfg=cfg.estimators)

    # §4.8 symbol rate -- three methods, harmonic check, reconciliation
    rate = estimate_symbol_rate(iso.y, iso.fs_b, cfg=cfg.estimators)
    warnings.extend(f"detection {index}: {w}" for w in rate.warnings)
    symbol_rate = _finite(rate.estimate.value)

    # §4.9 PSK order and carrier offset
    mth = estimate_psk_order(iso.y, iso.fs_b, cfg=cfg.estimators)
    if mth.psk_order.value is not None:
        extra["psk_order"] = _finite(mth.psk_order.value)
        extra["mth_power_sharpness_db"] = {
            str(k): round(v, 2) for k, v in mth.sharpness_db.items()
        }
    if mth.carrier_offset.value is not None:
        extra["carrier_offset_hz"] = _finite(mth.carrier_offset.value)

    # §4.10 FSK
    fsk = estimate_fsk_params(iso.y, iso.fs_b, symbol_rate=symbol_rate, cfg=cfg.estimators)
    if fsk.n_tones.value is not None:
        extra["n_tones"] = _finite(fsk.n_tones.value)
        extra["fsk_deviation_hz"] = _finite(fsk.deviation_hz.value)
        extra["fsk_tone_spacing_hz"] = _finite(fsk.tone_spacing_hz.value)
        extra["fsk_modulation_index"] = _finite(fsk.modulation_index.value)

    # §4.11 OFDM. Only the raw correlation numbers are recorded here; the interpretive
    # keys (useful symbol length, subcarrier spacing) are written by the §5.4 rule when it
    # fires. §4.11's peak test alone still fires on pulse-shaped single carriers, and an
    # evidence sentence reading "consistent with a cyclic prefix" under a label that is not
    # OFDM is precisely the confident-wrong-number §2 forbids.
    ofdm = estimate_ofdm_params(iso.y, iso.fs_b, cfg=cfg.estimators)
    if ofdm.r_peak is not None:
        extra["cp_autocorr_peak"] = _finite(ofdm.r_peak)
        extra["cp_autocorr_ratio"] = _finite(ofdm.peak_ratio)

    # §4.12 chirp -- on the raw time slice; isolation would filter the sweep away
    chirp = estimate_chirp(
        x[iso.n0 : iso.n1], fs, f_lo=burst.f_lo, f_hi=burst.f_hi, cfg=cfg.estimators
    )
    if chirp.is_chirp:
        extra["chirp_rate_hz_per_s"] = _finite(chirp.chirp_rate_hz_per_s.value)
        extra["chirp_sweep_bandwidth_hz"] = _finite(chirp.sweep_bandwidth_hz.value)
        extra["chirp_nonlinear"] = bool(chirp.nonlinear)

    # §4.13 analogue
    am = estimate_am_depth(iso.y, cfg=cfg.estimators)
    fm = estimate_fm_deviation(iso.y, iso.fs_b, cfg=cfg.estimators)
    asymmetry = estimate_spectral_asymmetry(iso.y, iso.fs_b, cfg=cfg.estimators)
    cw = estimate_cw_keying(iso.y, iso.fs_b, cfg=cfg.estimators)
    extra["am_depth"] = _finite(am.value)
    extra["fm_deviation_hz"] = _finite(fm.value)
    extra["sideband_asymmetry_db"] = _finite(asymmetry.value)
    if cw.is_morse:
        extra["morse_dash_dot_ratio"] = _finite(cw.dash_dot_ratio.value)
        extra["morse_dot_s"] = _finite(cw.dot_s.value)
        extra["morse_wpm"] = _finite(cw.wpm)

    # §5.2 cumulants, symbol-sampled when the symbol rate is known
    if symbol_rate and symbol_rate > 0 and iso.y.size >= 64:
        sps = iso.fs_b / symbol_rate
        try:
            cums = cumulants(iso.y, sps=int(round(sps)) if sps >= 2 else None)
            extra["cumulant_c40_ratio"] = _finite(cums.ratio_c40)
            extra["cumulant_c42_ratio"] = _finite(cums.ratio_c42)
            extra["cumulant_symbol_sampled"] = bool(cums.symbol_sampled)
        except ValueError:
            pass

    if burst.wideband:
        extra["wideband"] = True
        warnings.append(
            f"detection {index}: covers most of the capture in time and frequency; "
            "flagged as continuous/wideband"
        )

    power_dbfs = (
        snr_result.power_dbfs + dbfs_offset if snr_result.power_dbfs is not None else None
    )
    snr_field: float | str | None = snr_db
    if snr_db is None and snr_result.below_floor:
        snr_field = "< 0 dB"

    # ---- Stage 5: classify (§5.2 -> §5.6) ----
    if cfg.classify:
        sps = None
        if symbol_rate and symbol_rate > 0:
            candidate = iso.fs_b / symbol_rate
            if 2.0 <= candidate <= 64.0:
                sps = int(round(candidate))
        result, evidence = classify(
            iso.y,
            iso.fs_b,
            snr_db=snr_db,
            sps=sps,
            raw_slice=x[iso.n0 : iso.n1],
            fs_raw=fs,
            box_f_lo=burst.f_lo,
            box_f_hi=burst.f_hi,
            extra=extra,
            symbol_rate=symbol_rate,
            bandwidth_hz=bandwidth.bandwidth.occupied_99,
            classifiers=classifiers,
        )
        warnings.extend(f"detection {index}: {w}" for w in result.warnings)
        extra.update(result.extra)
        extra["snr_band"] = result.band
        if result.weights_are_prior and result.rule is None:
            extra["ensemble_weights_are_prior"] = True
        modulation = Modulation(
            label=result.label,
            confidence=round(float(result.confidence), 4),
            runner_up=result.runner_up,
            runner_up_confidence=(
                round(float(result.runner_up_confidence), 4)
                if result.runner_up_confidence is not None
                else None
            ),
            votes=result.votes,
            evidence=evidence,
        )
    else:
        modulation = Modulation(
            label=UNCLASSIFIED,
            confidence=0.0,
            votes={"feature_clf": None, "cnn": None, "rules": None},
            evidence=_evidence_for(snr_db, am, extra, notes),
        )

    return Detection(
        id=index,
        time_start_s=float(burst.t0),
        time_stop_s=float(burst.t1),
        duration_s=float(burst.duration),
        center_freq_hz=_absolute_freq(meta, offset_hz),
        center_freq_offset_hz=offset_hz,
        bandwidth_hz=bandwidth.bandwidth,
        snr_db=snr_field,
        power_dbfs=_finite(power_dbfs),
        symbol_rate_hz=rate.estimate,
        modulation=modulation,
        extra=extra,
    )


# --------------------------------------------------------------------------------------
# The pipeline itself
# --------------------------------------------------------------------------------------


def analyse_iq(
    iq: np.ndarray,
    meta: CaptureMeta,
    *,
    info: FileInfo | None = None,
    cfg: AnalysisConfig | None = None,
    on_progress: Callable[[str, float], None] | None = None,
) -> Analysis:
    """Condition -> detect -> isolate -> measure -> report, for an in-memory capture.

    Split out from :func:`analyse_file` so the pipeline can be driven from the scene
    composer and the tests without touching the filesystem.

    ``on_progress(stage, fraction)`` is called as each stage completes and once per
    measured burst. The API surfaces it as §7's 0-1 progress; without the per-burst calls
    the bar would sit still for the whole measurement phase, which is most of the runtime
    on a busy capture.
    """
    report_progress = on_progress or (lambda *_: None)
    cfg = cfg or AnalysisConfig()
    started = time.perf_counter()
    warnings: list[str] = []

    iq = np.ascontiguousarray(iq, dtype=np.complex64)
    info = info or FileInfo(name="<memory>", sha256="", bytes=int(iq.nbytes))

    if meta.frequencies_are_normalised:
        warnings.append(
            "sample rate is unknown; every frequency below is a fraction of the sample "
            "rate, not Hz"
        )
    if meta.center_freq is None:
        warnings.append(
            "centre frequency is unknown; detections carry an offset from the capture "
            "centre but no absolute RF frequency"
        )
    if iq.size >= LARGE_CAPTURE_SAMPLES:
        warnings.append(
            f"capture is {iq.size} samples and was analysed whole; overlapped block "
            "processing for very large files is not wired in yet"
        )

    if iq.size < 256:
        warnings.append(f"capture is only {iq.size} samples; too short to analyse")
        report = Report(
            file=info,
            capture=meta,
            noise_floor_dbfs=None,
            detections=[],
            warnings=warnings,
            runtime_s=round(time.perf_counter() - started, 3),
        )
        return Analysis(report=report, spectrogram=None, noise=None)

    # ---- Stage 2: condition (DC removal + unit-power normalisation) ----
    report_progress("conditioning", 0.15)
    x, cond = condition(iq)
    dbfs_offset = _dbfs_offset(cond.scale)
    if cond.dc_notched:
        warnings.append(
            "a strong DC spike was notched out before detection so it would not be "
            "reported as a signal"
        )

    # ---- Stage 3: detect. Already conditioned, so do not condition twice. ----
    report_progress("detecting", 0.30)
    detection = detect_bursts(x, meta.sample_rate, replace(cfg.detector, precondition=False))
    warnings.extend(detection.warnings)

    bursts = detection.bursts
    if len(bursts) > cfg.max_detections:
        warnings.append(
            f"{len(bursts)} candidate signals found; reporting the {cfg.max_detections} "
            "strongest by bandwidth-duration area"
        )
        bursts = sorted(bursts, key=lambda b: b.bandwidth * b.duration, reverse=True)
        bursts = sorted(bursts[: cfg.max_detections], key=lambda b: (b.t0, b.f_lo))

    # ---- Stages 4 and 5: isolate, measure, classify ----
    # loaded once and reused across every burst -- deserialising a checkpoint per
    # detection would dominate the runtime on a busy capture
    classifiers = load_classifiers() if cfg.classify else None
    if classifiers is not None and not classifiers.any_trained:
        warnings.append(
            "no trained classifier checkpoint found, so modulation comes from the §5.4 "
            "deterministic rules alone; run `sigscope fetch-data` then `sigscope train`"
        )
    detections = []
    for i, burst in enumerate(bursts, start=1):
        detections.append(
            _measure_burst(
                burst, i, x, meta, detection.spectrogram, cfg, dbfs_offset, warnings,
                classifiers,
            )
        )
        # measurement plus classification is the bulk of the runtime, so this is where the
        # progress bar earns its place
        report_progress("measuring", 0.45 + 0.45 * i / max(len(bursts), 1))

    report_progress("reporting", 0.90)

    report = Report(
        file=info,
        capture=meta,
        noise_floor_dbfs=round(detection.noise.floor_db + dbfs_offset, 2),
        detections=detections,
        warnings=warnings,
        runtime_s=round(time.perf_counter() - started, 3),
    )
    return Analysis(
        report=report,
        spectrogram=detection.spectrogram,
        noise=detection.noise,
        bursts=bursts,
    )


def analyse_file(
    path: str | Path,
    *,
    fs: float | None = None,
    fc: float | None = None,
    dtype: str | None = None,
    cfg: AnalysisConfig | None = None,
    on_progress: Callable[[str, float], None] | None = None,
) -> Analysis:
    """Analyse one capture file end to end (CLAUDE.md §3).

    ``fs`` / ``fc`` / ``dtype`` are caller overrides that beat anything guessed from the
    file. Raises :class:`sigscope.io.CaptureError` on an unreadable input -- the CLI turns
    that into a message naming the problem, never a traceback (§9 B).
    """
    path = Path(path)
    started = time.perf_counter()
    iq, meta = read_capture(path, fs=fs, fc=fc, dtype=dtype)
    analysis = analyse_iq(
        iq, meta, info=file_info(path), cfg=cfg, on_progress=on_progress
    )
    # count ingest time too -- §9 E budgets the whole run, not just the DSP
    analysis.report.runtime_s = round(time.perf_counter() - started, 3)
    return analysis
