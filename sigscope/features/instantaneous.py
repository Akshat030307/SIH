"""Instantaneous amplitude / phase / frequency statistics (CLAUDE.md §5.2).

The classical Azzouz-Nandi feature set that §5.2 names: ``gamma_max``, ``sigma_ap``,
``sigma_dp``, ``sigma_aa``, ``sigma_af``, kurtosis of amplitude, kurtosis of instantaneous
frequency, and PAPR in dB. All cheap, all explainable to a judge -- which is the whole
reason §5 pairs them with the CNN rather than trusting the network alone.

Everything is computed on the power-normalised burst and is dimensionless or in dB, so the
same numbers mean the same thing whether they came from a RadioML example at an unstated
sample rate or from a real burst at 2 MHz.

Two details that matter and that a naive implementation gets wrong:

* The phase statistics are measured only on samples whose amplitude clears a threshold.
  Where the envelope passes near zero the phase is numerically meaningless, and including
  those samples turns ``sigma_dp`` into a measure of the noise rather than of the
  modulation. §5.2 does not say this; Azzouz-Nandi does, and without it the feature is
  worthless on any amplitude-varying scheme.
* Nothing here may return NaN. A single NaN propagates into the scaler and then into every
  tree in the classifier, so each statistic is guarded and falls back to 0.0.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["InstantaneousStats", "instantaneous_stats", "AMPLITUDE_THRESHOLD"]

# samples below this fraction of the mean amplitude have meaningless phase (§5.2 note)
AMPLITUDE_THRESHOLD = 0.5


@dataclass
class InstantaneousStats:
    """The §5.2 instantaneous feature block."""

    gamma_max: float
    sigma_ap: float
    sigma_dp: float
    sigma_aa: float
    sigma_af: float
    kurtosis_amplitude: float
    kurtosis_freq: float
    papr_db: float
    mean_amplitude: float = 0.0
    amplitude_variance: float = 0.0
    names: tuple[str, ...] = field(default=(), repr=False)

    def as_dict(self) -> dict[str, float]:
        return {
            "gamma_max": self.gamma_max,
            "sigma_ap": self.sigma_ap,
            "sigma_dp": self.sigma_dp,
            "sigma_aa": self.sigma_aa,
            "sigma_af": self.sigma_af,
            "kurtosis_amplitude": self.kurtosis_amplitude,
            "kurtosis_freq": self.kurtosis_freq,
            "papr_db": self.papr_db,
            "amplitude_variance": self.amplitude_variance,
        }


def _safe(value: float, fallback: float = 0.0) -> float:
    """Never let a NaN or inf reach the classifier."""
    value = float(value)
    return value if np.isfinite(value) else fallback


def _kurtosis(v: np.ndarray) -> float:
    """Fisher kurtosis (normal = 0). Returns 0 for a degenerate sample."""
    if v.size < 4:
        return 0.0
    centred = v - v.mean()
    variance = float(np.mean(centred**2))
    if variance <= 1e-30:
        return 0.0
    return _safe(float(np.mean(centred**4) / variance**2) - 3.0)


def instantaneous_stats(
    y: np.ndarray, *, amplitude_threshold: float = AMPLITUDE_THRESHOLD
) -> InstantaneousStats:
    """Compute the §5.2 instantaneous statistics of a burst.

    ``y`` is power-normalised internally. ``fs`` is deliberately absent: every statistic
    here is either dimensionless or a ratio, so the feature vector transfers unchanged
    between a RadioML example and a real decimated burst.
    """
    y = np.asarray(y, dtype=np.complex128).ravel()
    if y.size < 8:
        return InstantaneousStats(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    power = float(np.mean(np.abs(y) ** 2))
    if power <= 1e-30:
        return InstantaneousStats(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    y = y / np.sqrt(power)

    amplitude = np.abs(y)
    mean_amplitude = float(np.mean(amplitude))
    if mean_amplitude <= 1e-30:
        return InstantaneousStats(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    # normalised-centred amplitude, the basis of the amplitude features
    a_n = amplitude / mean_amplitude
    a_cn = a_n - 1.0

    # gamma_max: peak of the amplitude spectrum, normalised by length. Large for a scheme
    # that carries information in its envelope, near zero for a constant-modulus one.
    spectrum = np.abs(np.fft.fft(a_cn)) ** 2
    gamma_max = _safe(float(np.max(spectrum)) / a_cn.size)

    # phase features, on the samples whose amplitude makes the phase meaningful
    strong = a_n > amplitude_threshold
    if int(np.count_nonzero(strong)) >= 8:
        phase = np.unwrap(np.angle(y))[strong]
        # remove the linear trend: a carrier offset is not modulation
        index = np.arange(phase.size, dtype=np.float64)
        slope, intercept = np.polyfit(index, phase, 1)
        centred_phase = phase - (slope * index + intercept)
        sigma_dp = _safe(float(np.std(centred_phase)))
        mean_square = float(np.mean(centred_phase**2))
        mean_abs = float(np.mean(np.abs(centred_phase)))
        sigma_ap = _safe(float(np.sqrt(max(mean_square - mean_abs**2, 0.0))))
    else:
        sigma_dp = sigma_ap = 0.0

    sigma_aa = _safe(float(np.std(np.abs(a_cn))))

    # instantaneous frequency, in cycles per sample (fs cancels out)
    if y.size >= 3:
        f_inst = np.angle(y[1:] * np.conj(y[:-1])) / (2.0 * np.pi)
        f_strong = f_inst[strong[1:]] if int(np.count_nonzero(strong[1:])) >= 8 else f_inst
        spread = float(np.std(f_strong))
        sigma_af = _safe(spread)
        kurtosis_freq = _kurtosis(f_strong)
    else:
        sigma_af = 0.0
        kurtosis_freq = 0.0

    papr_db = _safe(10.0 * np.log10(float(np.max(np.abs(y) ** 2)) + 1e-30))

    return InstantaneousStats(
        gamma_max=gamma_max,
        sigma_ap=sigma_ap,
        sigma_dp=sigma_dp,
        sigma_aa=sigma_aa,
        sigma_af=sigma_af,
        kurtosis_amplitude=_kurtosis(amplitude),
        kurtosis_freq=kurtosis_freq,
        papr_db=papr_db,
        mean_amplitude=mean_amplitude,
        amplitude_variance=_safe(float(np.var(a_n))),
    )
