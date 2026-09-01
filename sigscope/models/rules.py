"""Deterministic rules (CLAUDE.md §5.4).

The §5.4 table, one function each:

===========  ===============================================  ==============
Rule         Fires when                                       Output
===========  ===============================================  ==============
OFDM         CP autocorrelation peak at a consistent lag      ``ofdm``
Chirp        linear ridge fit R^2 > 0.9 and a large sweep     ``chirp-lfm``
CW           on/off envelope with dot/dash ratio near 3:1     ``cw``
Noise        SNR < 3 dB and spectral flatness > 0.9           ``noise``
SSB          spectral asymmetry > 10 dB                       ``AM-SSB``
===========  ===============================================  ==============

These **override the learned models when they fire, because they are physics** (§5.4). The
ensemble takes a fired rule at confidence 0.9 and records the other votes anyway (§5.5).
That is also why a false positive here is expensive: a rule that fires wrongly beats a
correct CNN. Each rule therefore delegates to the corresponding §4 estimator, which has
already been measured against known truth and has its own abstention behaviour, rather
than re-deriving the test with looser arithmetic.

One departure, inherited from §4.11 and documented there: the OFDM rule keys on the
correlation peak's ratio to its own background rather than §5.4's absolute ``> 0.3``, which
with the §4.11 normalisation can never fire for any real cyclic-prefix fraction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from sigscope.dsp.estimators import (
    EstimatorConfig,
    estimate_am_depth,
    estimate_chirp,
    estimate_cw_keying,
    estimate_ofdm_params,
    estimate_spectral_asymmetry,
)
from sigscope.features.spectral import spectral_stats

__all__ = ["RuleHit", "RuleConfig", "apply_rules", "RULE_LABELS"]

# §5.1: rule-only labels. Never trained, and the UI must say so.
RULE_LABELS: tuple[str, ...] = ("ofdm", "chirp-lfm", "cw", "noise", "AM-SSB")

RULE_CONFIDENCE = 0.9  # §5.5 step 1


@dataclass(frozen=True)
class RuleConfig:
    """Thresholds for the §5.4 table."""

    noise_snr_db: float = 3.0
    noise_flatness: float = 0.9
    ssb_asymmetry_db: float = 10.0
    # OFDM is a cyclic prefix *and* a flat spectrum. §4.11 measures the prefix; requiring
    # flatness too is what stops a pulse-shaped single carrier, whose autocorrelation has
    # its own structure at short lags, from firing a rule that overrides both models.
    ofdm_min_flatness: float = 0.50
    # An unmodulated carrier sitting off-centre is spectrally asymmetric too, so asymmetry
    # alone would label every offset CW tone as AM-SSB. Single-sideband voice varies in
    # amplitude; a carrier does not, so the envelope has to move for this rule to fire.
    ssb_min_am_depth: float = 0.10


@dataclass
class RuleHit:
    """A fired rule: the label it forces, why, and at what confidence."""

    label: str
    confidence: float
    rule: str
    evidence: list[str] = field(default_factory=list)
    extra: dict = field(default_factory=dict)


def apply_rules(
    y: np.ndarray,
    fs_b: float,
    *,
    snr_db: float | None = None,
    raw_slice: np.ndarray | None = None,
    fs_raw: float | None = None,
    box_f_lo: float | None = None,
    box_f_hi: float | None = None,
    chirp: Any = None,
    cfg: EstimatorConfig | None = None,
    rules: RuleConfig | None = None,
) -> RuleHit | None:
    """Run the §5.4 rules in priority order and return the first that fires.

    ``y`` is the isolated burst at ``fs_b``. ``raw_slice`` / ``fs_raw`` are the *un-isolated*
    time slice, needed only by the chirp rule -- isolation lowpasses to the detection box
    and would filter away the very sweep the rule looks for (§4.12). Pass ``chirp`` when
    the caller has already run :func:`~sigscope.dsp.estimators.estimate_chirp`; the
    pipeline does, and computing it twice per detection cost a spectrogram of the whole
    burst each time.

    Order matters. Noise is tested first: on a burst that is mostly noise the other tests
    are measuring nothing, and a spurious "cw" or "AM-SSB" on an empty channel would
    override a correct model vote. OFDM and chirp come next because they are the two
    physical structures §5.4 trusts over any learned opinion. SSB is last, being the
    weakest of the five.
    """
    cfg = cfg or EstimatorConfig()
    rules = rules or RuleConfig()
    y = np.ascontiguousarray(y, dtype=np.complex64)

    # ---- noise: SNR < 3 dB and spectral flatness > 0.9 ----
    if snr_db is not None and snr_db < rules.noise_snr_db:
        flatness = spectral_stats(y).spectral_flatness
        if flatness > rules.noise_flatness:
            return RuleHit(
                label="noise",
                confidence=RULE_CONFIDENCE,
                rule="noise/§5.4",
                evidence=[
                    f"Measured SNR is {snr_db:.1f} dB and the spectrum is almost perfectly "
                    f"flat (flatness {flatness:.2f}), which is what an empty channel looks "
                    "like.",
                ],
                extra={"spectral_flatness": float(flatness)},
            )

    # ---- OFDM: cyclic-prefix autocorrelation peak at a consistent lag ----
    ofdm = estimate_ofdm_params(y, fs_b, cfg=cfg)
    ofdm_flatness = spectral_stats(y).spectral_flatness if ofdm.is_ofdm else 0.0
    if (
        ofdm.is_ofdm
        and ofdm.subcarrier_spacing_hz.value is not None
        and ofdm_flatness >= rules.ofdm_min_flatness
    ):
        return RuleHit(
            label="ofdm",
            confidence=RULE_CONFIDENCE,
            rule="ofdm/§5.4",
            evidence=[
                f"Autocorrelation peaks at lag {ofdm.useful_symbol_len.value:.0f} samples "
                f"({ofdm.peak_ratio:.0f}x the background), which is a cyclic prefix; that "
                "makes this OFDM as a matter of physics, not statistics.",
                f"Useful symbol duration {ofdm.symbol_duration_s.value * 1e6:.1f} us gives "
                f"a subcarrier spacing of {ofdm.subcarrier_spacing_hz.value:.1f} Hz.",
                f"The spectrum is flat (flatness {ofdm_flatness:.2f}), as a bank of equal "
                "subcarriers should be, rather than the rounded shape of a single carrier.",
            ],
            extra={
                "ofdm_useful_symbol_len": ofdm.useful_symbol_len.value,
                "ofdm_subcarrier_spacing_hz": ofdm.subcarrier_spacing_hz.value,
                "ofdm_cp_length": ofdm.cp_length.value,
            },
        )

    # ---- chirp: linear ridge fit R^2 > 0.9 and a large sweep ----
    if chirp is None and raw_slice is not None and fs_raw:
        chirp = estimate_chirp(raw_slice, fs_raw, f_lo=box_f_lo, f_hi=box_f_hi, cfg=cfg)
    if chirp is not None and chirp.is_chirp and chirp.chirp_rate_hz_per_s.value is not None:
            return RuleHit(
                label="chirp-lfm",
                confidence=RULE_CONFIDENCE,
                rule="chirp/§5.4",
                evidence=[
                    f"The spectrogram ridge fits a straight line with R^2 = "
                    f"{chirp.r2_linear:.3f}, sweeping "
                    f"{chirp.chirp_rate_hz_per_s.value:.3e} Hz per second.",
                    f"That sweep covers {chirp.sweep_bandwidth_hz.value:.0f} Hz, which is a "
                    "linear FM chirp rather than a modulated carrier.",
                ],
                extra={
                    "chirp_rate_hz_per_s": chirp.chirp_rate_hz_per_s.value,
                    "chirp_sweep_bandwidth_hz": chirp.sweep_bandwidth_hz.value,
                    "chirp_nonlinear": chirp.nonlinear,
                },
            )

    # ---- CW: on/off envelope with a dot/dash ratio near 3:1 ----
    cw = estimate_cw_keying(y, fs_b, cfg=cfg)
    if cw.is_morse and cw.dash_dot_ratio.value is not None:
        return RuleHit(
            label="cw",
            confidence=RULE_CONFIDENCE,
            rule="cw/§5.4",
            evidence=[
                f"The envelope is on/off keyed across {cw.n_marks} marks whose lengths "
                f"cluster into two groups in a {cw.dash_dot_ratio.value:.2f} : 1 ratio, "
                "which is Morse dot and dash timing.",
                f"The dot length of {cw.dot_s.value * 1000:.0f} ms corresponds to about "
                f"{cw.wpm:.0f} words per minute.",
            ],
            extra={
                "morse_dash_dot_ratio": cw.dash_dot_ratio.value,
                "morse_dot_s": cw.dot_s.value,
                "morse_wpm": cw.wpm,
            },
        )

    # ---- SSB: spectral asymmetry > 10 dB, and the sign says which sideband ----
    asymmetry = estimate_spectral_asymmetry(y, fs_b, cfg=cfg)
    depth = estimate_am_depth(y, cfg=cfg)
    varies = depth.value is not None and depth.value > rules.ssb_min_am_depth
    if (
        asymmetry.value is not None
        and abs(asymmetry.value) > rules.ssb_asymmetry_db
        and varies
    ):
        side = "upper (USB)" if asymmetry.value > 0 else "lower (LSB)"
        return RuleHit(
            label="AM-SSB",
            confidence=RULE_CONFIDENCE,
            rule="ssb/§5.4",
            evidence=[
                f"The spectrum is {abs(asymmetry.value):.1f} dB stronger on the {side} side "
                "of centre; a double-sideband signal is symmetric, so this is "
                "single-sideband.",
                f"The envelope varies (modulation depth {depth.value:.2f}), so this is a "
                "modulated sideband rather than an unmodulated carrier sitting off centre.",
            ],
            extra={"sideband_asymmetry_db": asymmetry.value, "sideband": side},
        )

    return None
