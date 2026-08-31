"""Tiny known-truth signal generators -- for tests ONLY (CLAUDE.md §6.3).

Exists so estimators can be checked against signals whose parameters are known in
closed form. It is **never imported by the pipeline** and is **not** a training-data
generator (that is why §6 removed the generator). Six public functions, kept small.

All return ``np.complex64`` baseband at sample rate ``fs`` (Hz). ``rng`` accepts an int
seed or a ``numpy.random.Generator`` for reproducibility.
"""

from __future__ import annotations

import numpy as np

TWO_PI = 2.0 * np.pi


def _rng(rng: int | np.random.Generator | None) -> np.random.Generator:
    return rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)


def _rrc(beta: float, sps: int, span: int) -> np.ndarray:
    """Unit-energy root-raised-cosine FIR, roll-off ``beta``, ``span`` symbols per side."""
    tau = np.arange(-span * sps, span * sps + 1, dtype=float) / sps
    h = np.empty_like(tau)
    at_zero = np.isclose(tau, 0.0)
    at_sing = np.isclose(np.abs(tau), 1.0 / (4.0 * beta)) if beta > 0 else np.zeros_like(at_zero)
    h[at_zero] = 1.0 + beta * (4.0 / np.pi - 1.0)
    if beta > 0:
        h[at_sing] = (beta / np.sqrt(2.0)) * (
            (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
            + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))
        )
    rest = ~(at_zero | at_sing)
    t = tau[rest]
    h[rest] = (
        np.sin(np.pi * t * (1 - beta)) + 4 * beta * t * np.cos(np.pi * t * (1 + beta))
    ) / (np.pi * t * (1 - (4 * beta * t) ** 2))
    return h / np.sqrt(np.sum(h**2))


def tone(f: float, fs: float, n: int) -> np.ndarray:
    """Complex exponential at ``f`` Hz."""
    return np.exp(2j * np.pi * f * np.arange(n) / fs).astype(np.complex64)


def psk(
    order: int,
    symbol_rate: float,
    fs: float,
    n_sym: int,
    *,
    beta: float = 0.35,
    span: int = 8,
    rng: int | np.random.Generator | None = None,
) -> np.ndarray:
    """M-PSK with RRC pulse shaping. ``fs`` must be an integer multiple of ``symbol_rate``."""
    sps = fs / symbol_rate
    if not float(sps).is_integer():
        raise ValueError("fs must be an integer multiple of symbol_rate")
    sps = int(sps)
    m = _rng(rng).integers(0, order, n_sym)
    up = np.zeros(n_sym * sps, dtype=complex)
    up[::sps] = np.exp(2j * np.pi * m / order)
    return np.convolve(up, _rrc(beta, sps, span), mode="same").astype(np.complex64)


def fsk(
    order: int,
    deviation: float,
    symbol_rate: float,
    fs: float,
    n_sym: int,
    *,
    rng: int | np.random.Generator | None = None,
) -> np.ndarray:
    """Continuous-phase M-FSK. Tone levels are symmetric about 0; for 2-FSK they are
    ``+/-deviation`` (so ``deviation`` is half the outer spread, per §4.10)."""
    sps = fs / symbol_rate
    if not float(sps).is_integer():
        raise ValueError("fs must be an integer multiple of symbol_rate")
    sps = int(sps)
    m = _rng(rng).integers(0, order, n_sym)
    if order == 1:
        levels = np.zeros(1)
    else:
        levels = (np.arange(order) - (order - 1) / 2) * (2 * deviation / (order - 1))
    f_inst = np.repeat(levels[m], sps)
    return np.exp(2j * np.pi * np.cumsum(f_inst) / fs).astype(np.complex64)


def lfm_chirp(f0: float, f1: float, duration: float, fs: float) -> np.ndarray:
    """Linear FM sweep from ``f0`` to ``f1`` Hz over ``duration`` s."""
    t = np.arange(int(round(duration * fs))) / fs
    k = (f1 - f0) / duration
    return np.exp(2j * np.pi * (f0 * t + 0.5 * k * t**2)).astype(np.complex64)


def ofdm(
    n_sc: int,
    cp_len: int,
    n_sym: int,
    fs: float,
    *,
    rng: int | np.random.Generator | None = None,
) -> np.ndarray:
    """QPSK-loaded OFDM: IFFT of ``n_sc`` subcarriers with a ``cp_len``-sample cyclic prefix.
    ``fs`` is accepted for interface symmetry; sample spacing is 1 subcarrier."""
    r = _rng(rng)
    bits = r.integers(0, 2, (2, n_sym, n_sc)) * 2 - 1
    data = (bits[0] + 1j * bits[1]) / np.sqrt(2)
    body = np.fft.ifft(data, axis=1) * np.sqrt(n_sc)
    frame = np.concatenate([body[:, n_sc - cp_len :], body], axis=1)
    return frame.reshape(-1).astype(np.complex64)


def add_awgn(
    x: np.ndarray,
    snr_db: float,
    *,
    rng: int | np.random.Generator | None = None,
) -> np.ndarray:
    """Add complex AWGN for a target SNR measured over the whole of ``x``."""
    x = np.asarray(x)
    r = _rng(rng)
    p_signal = float(np.mean(np.abs(x) ** 2))
    p_noise = p_signal / (10.0 ** (snr_db / 10.0))
    noise = np.sqrt(p_noise / 2.0) * (r.standard_normal(x.shape) + 1j * r.standard_normal(x.shape))
    return (x + noise).astype(np.complex64)
