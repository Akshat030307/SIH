"""Acceptance test C -- detection against known ground truth (CLAUDE.md §9 C).

§9 C's list, each as its own test: precision and recall >= 0.90 above 10 dB and >= 0.70 in
3-10 dB; two signals one bandwidth apart reported as two; a bursty signal with sub-3-hop
gaps merged into one; pure noise giving zero detections and saying so; a continuous
full-band signal reported once and flagged wideband; a strong DC spike not reported as a
signal.

The zero-on-noise test is the one that matters most and the one that used to fail. With
§4.3's suggested fixed 8 dB threshold the detector returned 9,245 detections on one second
of noise, because that margin sits only 2.59 dB over the mean of an exponential
distribution and passes 16% of all cells. :func:`cfar_threshold_db` derives the margin from
a target false-alarm rate instead.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from sigscope import testsignals as ts
from sigscope.dsp.detect import DetectorConfig, cfar_threshold_db, detect_bursts
from sigscope.types import Burst

FS = 1_000_000.0


def _noise(n: int, seed: int, power: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (
        np.sqrt(power / 2) * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    ).astype(np.complex64)


def _place(
    canvas: np.ndarray,
    signal: np.ndarray,
    f_offset: float,
    t0: float,
    bandwidth: float,
    snr_db: float,
    fs: float = FS,
    noise_power: float = 1.0,
) -> tuple[float, float, float, float]:
    """Add ``signal`` at a known time/frequency with a known in-band SNR.

    Noise power inside the signal's own bandwidth is ``noise_power * bandwidth / fs``, so
    scaling to a target in-band SNR is closed form -- which is what makes the recall
    numbers below mean something.
    """
    n0 = int(t0 * fs)
    n1 = min(canvas.size, n0 + signal.size)
    segment = signal[: n1 - n0]
    in_band_noise = noise_power * bandwidth / fs
    amplitude = np.sqrt(
        in_band_noise * 10 ** (snr_db / 10) / float(np.mean(np.abs(segment) ** 2))
    )
    t = np.arange(n0, n1) / fs
    canvas[n0:n1] += (amplitude * segment * np.exp(2j * np.pi * f_offset * t)).astype(
        np.complex64
    )
    return (n0 / fs, n1 / fs, f_offset - bandwidth / 2, f_offset + bandwidth / 2)


def _scene(snr_db: float, seed: int) -> tuple[np.ndarray, list[tuple]]:
    """Four signals of different kinds at known places, on a 1 s / 1 MHz canvas."""
    rng = np.random.default_rng(seed)
    canvas = _noise(int(FS), seed)
    truth = [
        _place(canvas, ts.psk(4, 25_000.0, FS, 8000, rng=int(rng.integers(1e6))),
               +300_000.0, 0.10, 33_750.0, snr_db),
        _place(canvas, ts.fsk(2, 15_000.0, 10_000.0, FS, 4000, rng=int(rng.integers(1e6))),
               -250_000.0, 0.30, 50_000.0, snr_db),
        _place(canvas, ts.psk(2, 12_500.0, FS, 6000, rng=int(rng.integers(1e6))),
               +120_000.0, 0.50, 16_875.0, snr_db),
        _place(canvas, ts.tone(0.0, FS, int(0.3 * FS)),
               -400_000.0, 0.15, 2_000.0, snr_db),
    ]
    return canvas, truth


def _overlaps(burst: Burst, truth: tuple) -> bool:
    t0, t1, f_lo, f_hi = truth
    return (burst.t0 <= t1 and t0 <= burst.t1) and (burst.f_lo <= f_hi and f_lo <= burst.f_hi)


def _score(snr_db: float, seeds: range) -> tuple[float, float]:
    """Mean recall and precision over several noise seeds."""
    recalls, precisions = [], []
    for seed in seeds:
        canvas, truth = _scene(snr_db, seed)
        bursts = detect_bursts(canvas, FS).bursts
        hits = [t for t in truth if any(_overlaps(b, t) for b in bursts)]
        matched = [b for b in bursts if any(_overlaps(b, t) for t in truth)]
        recalls.append(len(hits) / len(truth))
        precisions.append(len(matched) / len(bursts) if bursts else 0.0)
    return float(np.mean(recalls)), float(np.mean(precisions))


# --------------------------------------------------------------------------------------
# the threshold itself
# --------------------------------------------------------------------------------------


def test_cfar_threshold_matches_the_closed_form():
    """``m = 10*log10(-ln(p_fa) / q)`` with ``q`` the reference quantile of Exp(1)."""
    import math

    for p_fa in (1e-2, 1e-3, 1e-5):
        q = -math.log(1.0 - 0.25)
        assert cfar_threshold_db(p_fa, 25.0) == pytest.approx(
            10.0 * math.log10(-math.log(p_fa) / q)
        )


def test_cfar_threshold_is_well_above_the_suggested_eight_db():
    """§4.3 suggests 8 dB. That passes 16% of pure-noise cells; the derived margin is
    higher, and this test is here so nobody quietly puts 8 dB back."""
    assert cfar_threshold_db(1e-2) > 11.0


def test_cfar_threshold_rejects_nonsense():
    for bad in (0.0, 1.0, -1.0):
        with pytest.raises(ValueError):
            cfar_threshold_db(bad)


# --------------------------------------------------------------------------------------
# §9 C -- pure noise gives zero detections and says so plainly
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5])
def test_pure_noise_gives_zero_detections(seed):
    result = detect_bursts(_noise(500_000, seed), FS)
    assert result.bursts == []


@pytest.mark.parametrize("n", [100_000, 1_000_000])
def test_pure_noise_at_several_lengths(n):
    """The spectrogram picks ``nfft`` from the capture length, so the cell count -- and
    therefore the number of chances to throw a false alarm -- changes with duration."""
    assert detect_bursts(_noise(n, seed=11), FS).bursts == []


def test_empty_band_says_what_happened_and_what_to_do():
    """§7: "Empty and error states say what happened and what to do"."""
    result = detect_bursts(_noise(500_000, seed=12), FS)
    assert result.warnings
    message = " ".join(result.warnings).lower()
    assert "no signals found" in message
    assert "threshold" in message


# --------------------------------------------------------------------------------------
# §9 C -- precision and recall by SNR band
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("snr_db", [20.0, 15.0, 10.0])
def test_precision_and_recall_above_10db(snr_db):
    """§9 C: "Precision and recall >= 0.90 above 10 dB"."""
    recall, precision = _score(snr_db, range(4))
    assert recall >= 0.90, f"recall {recall:.2f} at {snr_db} dB"
    assert precision >= 0.90, f"precision {precision:.2f} at {snr_db} dB"


@pytest.mark.parametrize("snr_db", [5.0, 3.0])
def test_precision_and_recall_between_3_and_10db(snr_db):
    """§9 C: ">= 0.70 in 3-10 dB"."""
    recall, precision = _score(snr_db, range(4))
    assert recall >= 0.70, f"recall {recall:.2f} at {snr_db} dB"
    assert precision >= 0.70, f"precision {precision:.2f} at {snr_db} dB"


# --------------------------------------------------------------------------------------
# §9 C -- resolution, merging, wideband, DC
# --------------------------------------------------------------------------------------


def test_two_signals_one_bandwidth_apart_are_two_detections():
    """§9 C: "Two signals one bandwidth apart reported as two"."""
    bandwidth = 40_000.0
    canvas = _noise(int(FS), seed=20)
    _place(canvas, ts.psk(4, 25_000.0, FS, 8000, rng=1), -bandwidth, 0.1, bandwidth, 25.0)
    _place(canvas, ts.psk(4, 25_000.0, FS, 8000, rng=2), +bandwidth, 0.1, bandwidth, 25.0)

    bursts = detect_bursts(canvas, FS).bursts

    assert len(bursts) == 2, f"expected 2 detections, got {len(bursts)}"
    centres = sorted(b.f_center for b in bursts)
    assert centres[0] == pytest.approx(-bandwidth, abs=bandwidth / 2)
    assert centres[1] == pytest.approx(+bandwidth, abs=bandwidth / 2)


def test_bursty_transmission_merges_into_one_detection():
    """§9 C: "A bursty signal with sub-3-hop gaps merged into one"."""
    canvas = _noise(int(FS), seed=21)
    burst = ts.psk(4, 25_000.0, FS, 1000, rng=3)  # 40 ms of signal
    # the STFT hop here is ~512 us, so the gaps have to be under ~1.5 ms to be the
    # "sub-3-hop gaps" §9 C is talking about; 1 ms of silence between 40 ms bursts
    for k in range(6):
        _place(canvas, burst, +200_000.0, 0.1 + k * 0.041, 33_750.0, 25.0)

    bursts = [b for b in detect_bursts(canvas, FS).bursts if abs(b.f_center - 200_000.0) < 60_000]

    assert len(bursts) == 1, f"expected the bursts stitched into one, got {len(bursts)}"


def test_signal_with_a_spectral_null_is_one_detection_not_two():
    """The frequency-axis half of the merge rule.

    A 2-FSK burst has two tones with a dip between them, which the connected-component
    step sees as two side-by-side blobs at the same instant. §4.3's merge only joins along
    time, so without the frequency clause this is reported as two signals.
    """
    canvas = _noise(int(FS), seed=22)
    _place(canvas, ts.fsk(2, 40_000.0, 10_000.0, FS, 4000, rng=4),
           +200_000.0, 0.2, 100_000.0, 25.0)

    bursts = [b for b in detect_bursts(canvas, FS).bursts if abs(b.f_center - 200_000.0) < 150_000]

    assert len(bursts) == 1, f"the two FSK tones should be one detection, got {len(bursts)}"


def test_continuous_wideband_signal_is_one_flagged_detection():
    """§9 C: "A continuous full-band signal is one detection flagged wideband"."""
    # §4.3 calls a component wideband at >90% of the duration and >60% of the bandwidth,
    # so the signal has to actually span the capture: 500 kBd RRC is 675 kHz of a 1 MHz
    # band, and 500k symbols at 2 samples each fills the whole second.
    canvas = _noise(int(FS), seed=23, power=0.01)
    wide = ts.psk(4, 500_000.0, FS, 500_000, rng=5)
    canvas[: wide.size] += (wide[: canvas.size] * 3.0).astype(np.complex64)

    bursts = detect_bursts(canvas, FS).bursts

    assert len(bursts) == 1, f"expected one wideband detection, got {len(bursts)}"
    assert bursts[0].wideband


def test_strong_dc_spike_is_not_reported_as_a_signal():
    """§9 C: "A strong DC spike is not reported as a signal".

    The Stage 2 conditioner notches it; ``detect_bursts`` runs that by default.
    """
    canvas = _noise(int(FS), seed=24)
    canvas += np.complex64(12.0)  # a large constant offset is a DC spike

    bursts = detect_bursts(canvas, FS).bursts

    assert not any(abs(b.f_center) < 5_000.0 for b in bursts), (
        "the DC spike was reported as a signal"
    )


def test_leakage_skirts_of_a_strong_signal_are_suppressed():
    """One strong emitter must not become a dozen detections.

    Window sidelobes and splatter either side of a strong burst clear the threshold on
    their own; before suppression a single 25 dB FSK burst produced ten extra boxes.
    """
    canvas = _noise(int(FS), seed=25)
    truth = _place(canvas, ts.fsk(2, 15_000.0, 10_000.0, FS, 4000, rng=6),
                   -250_000.0, 0.3, 50_000.0, 30.0)

    result = detect_bursts(canvas, FS)

    assert len(result.bursts) == 1, (
        f"expected one detection, got {len(result.bursts)}: "
        f"{[(round(b.f_lo), round(b.f_hi)) for b in result.bursts]}"
    )
    assert _overlaps(result.bursts[0], truth)


def test_explicit_threshold_overrides_the_derivation():
    """§4.3's literal behaviour stays reachable, and ``--threshold-db`` depends on it."""
    canvas = _noise(200_000, seed=26)
    permissive = replace(DetectorConfig(), threshold_db=8.0)

    assert detect_bursts(canvas, FS).bursts == []
    assert len(detect_bursts(canvas, FS, permissive).bursts) > 0
