"""Robust noise-floor estimate (CLAUDE.md §4.2).

Never the mean -- one strong signal drags it up. Per frequency bin take a low percentile
over time, then the median across bins for the scalar floor. ``sigma_db`` from the median
absolute deviation gives a robust spread for later confidence scoring.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class NoiseEstimate:
    """Per-bin noise floor (dB), the scalar floor (dB), and a robust spread (dB)."""

    per_bin_db: np.ndarray
    floor_db: float
    sigma_db: float


def estimate_noise(S_db: np.ndarray, *, percentile: float = 25.0) -> NoiseEstimate:
    """Per-bin percentile over time -> median floor + MAD spread (CLAUDE.md §4.2)."""
    per_bin = np.percentile(S_db, percentile, axis=1).astype(np.float64)
    floor = float(np.median(per_bin))
    mad = float(np.median(np.abs(per_bin - floor)))
    return NoiseEstimate(per_bin_db=per_bin, floor_db=floor, sigma_db=1.4826 * mad)
