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
