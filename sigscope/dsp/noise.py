"""Robust noise-floor estimate (CLAUDE.md §4.2).

Never the mean -- one strong signal drags it up. Per frequency bin take a low percentile
over time, then a low percentile *across* bins for the scalar floor. ``sigma_db`` from the
median absolute deviation gives a robust spread for later confidence scoring.

§4.2 says to take the **median** across bins. That holds while signals occupy less than
half the band and breaks when they occupy more: a continuous transmission covering 68% of
the band puts the median inside the signal, so the floor is reported as the signal level
and the detector then finds nothing at all -- the §4.3 "continuous/wideband" case fails to
detect the very thing it exists to describe. Taking the 25th percentile across bins keeps
§4.2's intent (a robust statistic, immune to strong signals) while tolerating up to ~75%
band occupancy. For a capture with only a few narrow signals the two agree to a fraction
of a dB, since both land in noise either way.

A signal filling 100% of the band for 100% of the duration remains indistinguishable from
a raised noise floor by this statistic alone -- there is no in-band reference left to
compare against. That is a real limit, not an oversight, and the detector reports it rather
than guessing.
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


def estimate_noise(
    S_db: np.ndarray,
    *,
    percentile: float = 25.0,
    bin_percentile: float = 25.0,
    max_columns: int = 4096,
) -> NoiseEstimate:
    """Per-bin percentile over time -> robust scalar floor + MAD spread (CLAUDE.md §4.2).

    ``percentile`` is taken over time within each bin; ``bin_percentile`` is then taken
    across bins for the scalar floor. Set ``bin_percentile=50`` for §4.2's literal median --
    see the module docstring for why 25 is the default.

    ``max_columns`` subsamples the time axis before taking the percentile. A percentile is
    a distribution statistic and 4096 evenly-spaced columns estimate it as well as 9766 do,
    while the sort underneath is the single largest allocation in the pipeline -- on a
    10 s / 2 MHz capture the full-width version copied 320 MB and cost 5 s of a 30 s budget.
    """
    columns = S_db
    if max_columns and S_db.shape[1] > max_columns:
        step = int(np.ceil(S_db.shape[1] / max_columns))
        columns = S_db[:, ::step]
    per_bin = np.percentile(columns, percentile, axis=1).astype(np.float64)
    floor = float(np.percentile(per_bin, bin_percentile))
    mad = float(np.median(np.abs(per_bin - floor)))
    return NoiseEstimate(per_bin_db=per_bin, floor_db=floor, sigma_db=1.4826 * mad)
