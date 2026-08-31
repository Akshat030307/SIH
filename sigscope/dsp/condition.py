"""Stage 2 -- Condition (CLAUDE.md §2 "Stage 2 -- Condition").

Minimal for now: remove the DC offset (subtract the mean, then a very narrow 0 Hz notch
if a spike remains -- SDR front ends leave a strong centre spike that would otherwise be
detected as a signal), and normalise to unit average power keeping the scale factor.
Overlapped block processing for very large files lands with the pipeline wiring (Phase 5).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import lfilter


@dataclass
class ConditionInfo:
    mean_removed: bool
    dc_notched: bool
    scale: float


def condition(
    x: np.ndarray,
    *,
    dc_spike_ratio: float = 20.0,
    dc_block_pole: float = 0.9995,
    normalise: bool = True,
) -> tuple[np.ndarray, ConditionInfo]:
    """DC removal + unit-power normalisation (CLAUDE.md §2 Stage 2). Pure function."""
    x = np.ascontiguousarray(x, dtype=np.complex64)
    x = x - np.complex64(x.mean())

    probe = np.abs(np.fft.fft(x[: min(x.size, 1 << 16)]))
    notched = False
    if probe.size and probe[0] > dc_spike_ratio * (np.median(probe) + 1e-12):
        x = lfilter([1.0, -1.0], [1.0, -dc_block_pole], x).astype(np.complex64)
        notched = True

    scale = float(np.sqrt(np.mean(np.abs(x) ** 2)))
    if normalise and scale > 0.0:
        x = (x / scale).astype(np.complex64)
    return x, ConditionInfo(mean_removed=True, dc_notched=notched, scale=scale)
