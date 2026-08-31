"""Sample-format table, interleaved-IQ decoding, and the dtype-guessing heuristic.

Implements the "Dtype guessing" paragraph of CLAUDE.md §3 Stage 1: try int8, int16-LE and
float32-LE and score how much each decoding looks like real IQ. A wrong guess is spiky,
clipped, non-zero-mean, or -- most tellingly -- not smooth sample-to-sample (oversampled
IQ correlates; mis-typed bytes do not). The little-endian low byte separates int8 from
int16: a real 16-bit ADC's is white, an 8-bit I signal widened to 16 stays correlated.
All three scores are kept and surfaced so the user can override.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats

# name -> (numpy dtype, full-scale divisor to reach ~unit amplitude)
_SPEC: dict[str, tuple[np.dtype, float]] = {
    "int8": (np.dtype("<i1"), 128.0),
    "int16": (np.dtype("<i2"), 32768.0),
    "float32": (np.dtype("<f4"), 1.0),
}
GUESS_ORDER: tuple[str, ...] = ("int8", "int16", "float32")


def _ramp(x: float, lo: float, hi: float) -> float:
    """Linear 0->1 ramp: 0 at/below ``lo``, 1 at/above ``hi``."""
    return min(1.0, max(0.0, (x - lo) / (hi - lo)))

# headerless extensions that pin the sample format (CLAUDE.md §3 Stage 1)
EXT_DTYPE: dict[str, str] = {
    ".cs8": "int8",
    ".cs16": "int16",
    ".cf32": "float32",
    ".cfile": "float32",  # GNU Radio complex float
    ".fc32": "float32",
}


def numpy_dtype(name: str) -> np.dtype:
    return _SPEC[name][0]


def itemsize(name: str) -> int:
    return _SPEC[name][0].itemsize


def decode_interleaved(raw: np.ndarray, name: str) -> np.ndarray:
    """Interleaved I,Q,I,Q... of the given sample format -> ``complex64`` unit-scaled."""
    divisor = _SPEC[name][1]
    flat = np.ascontiguousarray(raw).view(_SPEC[name][0])
    flat = flat[: (flat.size // 2) * 2]
    iq = flat[0::2].astype(np.float32) + 1j * flat[1::2].astype(np.float32)
    if divisor != 1.0:
        iq /= divisor
    return iq.astype(np.complex64)


@dataclass
class DtypeGuess:
    """Result of :func:`guess_dtype` -- the pick plus every candidate's score."""

    name: str
    confidence: float
    scores: dict[str, float]
    notes: list[str]


def _stream_corr(s: np.ndarray) -> float:
    """Lag-1/lag-2 autocorrelation *coefficient* magnitude (0..1) of one stream."""
    if s.size < 8:
        return 0.0
    s = s.astype(np.float64) - float(np.mean(s))
    var = float(np.mean(s * s))
    if var == 0.0:
        return 0.0
    ac1 = float(np.mean(s[1:] * s[:-1]))
    ac2 = float(np.mean(s[2:] * s[:-2]))
    return max(abs(ac1), abs(ac2)) / var


def _lowbyte_autocorr(buf: bytes) -> float:
    """Lag-1/2 autocorrelation magnitude of the even-offset bytes.

    If the buffer is int16, those bytes are the little-endian LOW bytes: a real ADC's
    low bits are dither/quantisation noise and decorrelate (~ 0). If the buffer is
    really 8-bit IQ, those same bytes are the I signal itself and stay correlated
    (oversampled) -- which is the tell that disambiguates int8 from int16.
    """
    b = np.frombuffer(buf, dtype=np.int8)[: 2_000_000]
    if b.size < 512:
        return 0.0
    return _stream_corr(b[0::2].astype(np.float64))


def _score(values: np.ndarray, name: str, lowbyte_ac: float) -> float:
    """Plausibility in 0..1 that ``values`` (decoded as ``name``) are real IQ samples.

    Multiplicative cues: near-zero mean; not spiky (heavy positive kurtosis); little hard
    clipping; no tall spike at zero; both the I and Q stream are smooth (oversampled IQ
    correlates sample-to-sample, a mis-typed stream does not); and a low-byte correlation
    term that pushes correlated low bytes towards int8 and white ones towards int16.
    """
    if name == "float32" and not np.isfinite(values).all():
        return 0.0
    if values.size < 512:
        return 0.0
    with np.errstate(all="ignore"):
        v = values.astype(np.float64)[:1_000_000]
    std = float(v.std())
    if not np.isfinite(std) or std == 0.0:
        return 0.0
    vc = (v - v.mean()) / std

    s_mean = math.exp(-3.0 * abs(float(v.mean())) / std)

    # excess kurtosis: 0 Gaussian, ~-1.2 uniform, ~-2 constant-modulus (BPSK/CW), >0 spiky.
    # Punish spiky hard; be lenient about platykurtic since real CW/FSK/BPSK live there
    # (the smoothness cue below is what rejects uniform "byte soup").
    k = float(stats.kurtosis(vc, fisher=True, bias=False))
    s_shape = math.exp(-max(k, 0.0) / 2.0 - max(-k - 0.5, 0.0) / 3.0)

    if name == "int8":
        clip = float(np.mean(np.abs(values) >= 127))
    elif name == "int16":
        clip = float(np.mean(np.abs(values) >= 32767))
    else:
        clip = float(np.mean(np.abs(v) >= 1.0e4))  # implausibly large for a float capture
    s_clip = 1.0 - min(clip * 5.0, 1.0)

    # small-amplitude captures legitimately round many samples to 0; only a large
    # zero spike (structured / non-signal file) should count against a format.
    zero_frac = float(np.mean(values == 0))
    s_zero = 1.0 - min(max(zero_frac - 0.10, 0.0) * 3.0, 1.0)

    both = min(_stream_corr(vc[0::2]), _stream_corr(vc[1::2]))
    s_corr = min(1.0, 0.15 + 3.0 * both)

    # low-byte correlation disambiguates int8 vs int16 (see _lowbyte_autocorr):
    # a real 16-bit LSByte is white (~0), an 8-bit I signal is strongly correlated
    # (~0.8). Pure-noise captures sit near 0 for both and fall to int16 by default.
    if name == "int16":
        s_low = 1.0 - _ramp(lowbyte_ac, 0.10, 0.40)
    elif name == "int8":
        s_low = _ramp(lowbyte_ac, 0.15, 0.45)
    else:
        s_low = 1.0

    return float(s_mean * s_shape * s_clip * s_zero * s_corr * s_low)


def guess_dtype(sample_bytes: bytes, file_bytes: int) -> DtypeGuess:
    """Score int8 / int16 / float32 against a byte sample and pick the most plausible."""
    lowbyte_ac = _lowbyte_autocorr(sample_bytes)
    scores: dict[str, float] = {}
    for name in GUESS_ORDER:
        step = itemsize(name) * 2  # one complex sample
        if file_bytes % step != 0 or len(sample_bytes) < step * 64:
            scores[name] = 0.0
            continue
        usable = (len(sample_bytes) // step) * step
        raw = np.frombuffer(sample_bytes[:usable], dtype=numpy_dtype(name))
        scores[name] = _score(raw, name, lowbyte_ac)

    best = max(scores, key=lambda n: scores[n])
    ordered = sorted(scores.values(), reverse=True)
    top, second = ordered[0], (ordered[1] if len(ordered) > 1 else 0.0)
    total = float(sum(scores.values()))
    # confident when the winner is both plausible on its own and clearly ahead
    separation = top / total if total > 0.0 else 0.0
    confidence = round(min(0.98, separation * _ramp(top, 0.02, 0.15)), 3)

    notes = [
        "dtype plausibility (int8/int16/float32): "
        + ", ".join(f"{n}={scores[n]:.2f}" for n in GUESS_ORDER)
        + f" -> {best}"
    ]
    if second > 0.0 and second / top > 0.8:
        notes.append(f"runner-up dtype scored close ({second:.2f}); pass dtype= to override")
    return DtypeGuess(name=best, confidence=confidence, scores=scores, notes=notes)
