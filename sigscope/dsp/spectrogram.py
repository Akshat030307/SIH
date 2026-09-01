"""Short-time Fourier transform for detection and measurement (CLAUDE.md §4.1).

Two-sided always -- the signal is complex baseband and the negative frequencies carry
real information. ``nfft`` is picked adaptively for ~2000 time columns and clamped to a
power of two in [256, 8192].
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.signal import stft


@dataclass
class Spectrogram:
    """Result of :func:`compute_spectrogram`.

    ``f`` (Hz) is monotonic ascending over ``[-fs/2, fs/2)``; ``t`` (s) is the column
    time; ``S_db`` has shape ``(len(f), len(t))``. ``hop`` is the STFT stride in samples.
    """

    f: np.ndarray
    t: np.ndarray
    S_db: np.ndarray
    fs: float
    nfft: int
    hop: int

    @property
    def shape(self) -> tuple[int, int]:
        return self.S_db.shape

    def mean_power_per_bin(self, chunk: int = 512) -> np.ndarray:
        """Time-averaged **linear** power in each frequency bin, computed once.

        ``10 ** (S_db / 10)`` over a whole spectrogram is the single most expensive thing
        in the pipeline: on a 10 s / 2 MHz capture ``S_db`` is 8192 x 9766, so the linear
        copy is 640 MB of float64. §4.7 needs that array for every detection, and
        recomputing it per burst cost 74 of 105 seconds and most of the peak RSS on a
        17-detection scene.

        The result depends only on the spectrogram, so it is computed once and memoised,
        and accumulated over column chunks so the 640 MB temporary is never materialised.
        """
        cached = getattr(self, "_mean_power_per_bin", None)
        if cached is not None:
            return cached
        n_freq, n_time = self.S_db.shape
        total = np.zeros(n_freq, dtype=np.float64)
        for start in range(0, n_time, chunk):
            block = self.S_db[:, start : start + chunk]
            total += np.power(10.0, block.astype(np.float64) / 10.0).sum(axis=1)
        result = total / max(n_time, 1)
        object.__setattr__(self, "_mean_power_per_bin", result)
        return result


def choose_nfft(n_samples: int, *, target_cols: int = 2000, lo: int = 256, hi: int = 8192) -> int:
    """Power-of-two ``nfft`` aiming for ``target_cols`` STFT columns (hop = nfft/4)."""
    target = max(4.0 * n_samples / max(target_cols, 1), 1.0)
    nfft = 1 << max(0, round(math.log2(target)))
    return int(np.clip(nfft, lo, hi))


def compute_spectrogram(
    x: np.ndarray,
    fs: float,
    *,
    nfft: int | None = None,
    overlap_frac: float = 0.75,
    window: str = "hann",
    target_cols: int = 2000,
    nfft_lo: int = 256,
    nfft_hi: int = 8192,
) -> Spectrogram:
    """STFT magnitude in dB (CLAUDE.md §4.1). Pure function: array in, dataclass out."""
    x = np.ascontiguousarray(x, dtype=np.complex64)
    if x.size == 0:
        raise ValueError("compute_spectrogram: empty input")
    if nfft is None:
        nfft = choose_nfft(x.size, target_cols=target_cols, lo=nfft_lo, hi=nfft_hi)
    nfft = int(min(nfft, x.size)) if x.size < nfft else int(nfft)
    noverlap = int(nfft * overlap_frac)

    f, t, Zxx = stft(
        x, fs=fs, nperseg=nfft, noverlap=noverlap, window=window, return_onesided=False
    )
    Zxx = np.fft.fftshift(Zxx, axes=0)
    f = np.fft.fftshift(f)
    S_db = (20.0 * np.log10(np.abs(Zxx) + 1e-12)).astype(np.float32)
    return Spectrogram(f=f, t=t, S_db=S_db, fs=float(fs), nfft=int(nfft), hop=int(nfft - noverlap))
