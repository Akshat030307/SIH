"""Spectral features for the classifier (CLAUDE.md §5.2 "Spectral features").

§5.2's list, verbatim: upper/lower power asymmetry in dB (catches SSB); peak count in the
``f_inst`` histogram (catches M-FSK order); squared- and 4th-power spectral line strengths
(§4.9); cyclic-prefix autocorrelation peak (§4.11); spectral flatness (OFDM is flat-topped,
PSK is raised-cosine rounded).

These are the same quantities the §4 estimators measure, computed here in a cheap
fixed-cost form suitable for a 128-sample RadioML window rather than a full burst. The
estimators stay the authority for anything that gets *reported*; this module only produces
classifier input.

Everything is scale-free and sample-rate-free, so a feature vector computed on a RadioML
example and one computed on a decimated real burst live in the same space -- which is the
only reason a model trained on the former can be applied to the latter at all.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks

__all__ = ["SpectralStats", "spectral_stats"]


@dataclass
class SpectralStats:
    """The §5.2 spectral feature block."""

    asymmetry_db: float
    spectral_flatness: float
    n_freq_peaks: float
    line_strength_m2: float
    line_strength_m4: float
    line_strength_m8: float
    cp_autocorr_peak: float
    cp_autocorr_ratio: float
    occupied_fraction: float

    def as_dict(self) -> dict[str, float]:
        return {
            "asymmetry_db": self.asymmetry_db,
            "spectral_flatness": self.spectral_flatness,
            "n_freq_peaks": self.n_freq_peaks,
            "line_strength_m2": self.line_strength_m2,
            "line_strength_m4": self.line_strength_m4,
            "line_strength_m8": self.line_strength_m8,
            "cp_autocorr_peak": self.cp_autocorr_peak,
            "cp_autocorr_ratio": self.cp_autocorr_ratio,
            "occupied_fraction": self.occupied_fraction,
        }


def _safe(value: float, fallback: float = 0.0) -> float:
    value = float(value)
    return value if np.isfinite(value) else fallback


def _averaged_psd(y: np.ndarray, min_segments: int = 8) -> np.ndarray:
    """Welch-style averaged periodogram, for a stable spectral-flatness estimate.

    Falls back to a single periodogram for a burst too short to segment, in which case
    flatness is biased low and the noise rule simply will not fire -- which is the safe
    direction (§5.4 rules override the learned models).
    """
    n = y.size
    segment = n // min_segments
    if segment < 16:
        return np.abs(np.fft.fft(y)) ** 2
    window = np.hanning(segment)
    total = np.zeros(segment)
    count = 0
    for start in range(0, n - segment + 1, segment // 2):
        total += np.abs(np.fft.fft(y[start : start + segment] * window)) ** 2
        count += 1
    return total / max(count, 1)


def _line_strength(y: np.ndarray, power: int) -> float:
    """Peak-to-median of ``|FFT(y**power)|`` in dB -- the §4.9 sharpness, as a feature."""
    if y.size < 16:
        return 0.0
    spectrum = np.abs(np.fft.fft(y**power))
    median = float(np.median(spectrum))
    if median <= 1e-30:
        return 0.0
    return _safe(20.0 * np.log10(float(np.max(spectrum)) / median))


def spectral_stats(y: np.ndarray) -> SpectralStats:
    """Compute the §5.2 spectral statistics of a burst (power-normalised internally)."""
    y = np.asarray(y, dtype=np.complex128).ravel()
    zero = SpectralStats(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    if y.size < 16:
        return zero

    power = float(np.mean(np.abs(y) ** 2))
    if power <= 1e-30:
        return zero
    y = y / np.sqrt(power)

    spectrum = np.fft.fftshift(np.abs(np.fft.fft(y)) ** 2)
    total = float(np.sum(spectrum))
    if total <= 1e-30:
        return zero

    # Flatness needs an *averaged* periodogram, not a raw one. A single periodogram of
    # white noise has exponentially distributed bins, whose geometric-to-arithmetic mean
    # ratio is exp(-gamma) = 0.56 rather than 1 -- so the §5.4 noise rule's
    # "flatness > 0.9" could never fire against a raw periodogram, no matter how pure the
    # noise. Averaging K segments tightens the bins toward their expectation and pushes
    # white noise to ~0.94 at K=8, which is what §5.4's threshold is written for.
    averaged = _averaged_psd(y)

    # upper / lower sideband asymmetry (§4.13 SSB, as a feature)
    half = spectrum.size // 2
    lower = float(np.sum(spectrum[:half])) + 1e-30
    upper = float(np.sum(spectrum[half:])) + 1e-30
    asymmetry_db = _safe(10.0 * np.log10(upper / lower))

    # spectral flatness: geometric mean / arithmetic mean. Near 1.0 for white noise; a
    # flat-topped OFDM block sits high, a rounded raised-cosine PSK lower.
    positive = averaged + 1e-30
    log_mean = float(np.mean(np.log(positive)))
    arithmetic = float(np.mean(positive))
    flatness = _safe(float(np.exp(log_mean) / arithmetic)) if arithmetic > 0 else 0.0

    # fraction of the band holding 99% of the power -- a coarse occupied-bandwidth feature
    order = np.sort(spectrum)[::-1]
    cumulative = np.cumsum(order) / total
    occupied_fraction = _safe(float(np.searchsorted(cumulative, 0.99) + 1) / spectrum.size)

    # instantaneous-frequency histogram peak count (§4.10 tone count, as a feature)
    if y.size >= 32:
        f_inst = np.angle(y[1:] * np.conj(y[:-1]))
        lo, hi = np.percentile(f_inst, [1.0, 99.0])
        trimmed = f_inst[(f_inst >= lo) & (f_inst <= hi)]
        if trimmed.size >= 16 and hi > lo:
            histogram, _ = np.histogram(trimmed, bins=32)
            smooth = np.convolve(histogram.astype(float), np.ones(3) / 3.0, mode="same")
            peaks, _ = find_peaks(smooth, prominence=0.2 * float(smooth.max()))
            n_freq_peaks = float(peaks.size)
        else:
            n_freq_peaks = 0.0
    else:
        n_freq_peaks = 0.0

    # cyclic-prefix autocorrelation (§4.11), the OFDM tell
    cp_peak, cp_ratio = 0.0, 0.0
    if y.size >= 64:
        nfft = 1 << int(np.ceil(np.log2(2 * y.size)))
        autocorr = np.abs(np.fft.ifft(np.abs(np.fft.fft(y, nfft)) ** 2))
        energy = float(autocorr[0])
        if energy > 1e-30:
            normalised = autocorr / energy
            lag_hi = min(y.size // 2, 512)
            if lag_hi > 8:
                band = normalised[8:lag_hi]
                cp_peak = _safe(float(np.max(band)))
                background = float(np.median(band))
                cp_ratio = _safe(cp_peak / background) if background > 1e-12 else 0.0

    return SpectralStats(
        asymmetry_db=asymmetry_db,
        spectral_flatness=flatness,
        n_freq_peaks=n_freq_peaks,
        line_strength_m2=_line_strength(y, 2),
        line_strength_m4=_line_strength(y, 4),
        line_strength_m8=_line_strength(y, 8),
        cp_autocorr_peak=cp_peak,
        cp_autocorr_ratio=cp_ratio,
        occupied_fraction=occupied_fraction,
    )
