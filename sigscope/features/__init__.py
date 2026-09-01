"""Handcrafted classification features (CLAUDE.md §5.2).

Three blocks, assembled by :func:`extract_features` into one frozen-order vector:

- :mod:`~sigscope.features.cumulants`     -- C20/C21/C40/C41/C42/C63 and the §5.2 ratios
- :mod:`~sigscope.features.instantaneous` -- gamma_max, sigma_ap/dp/aa/af, kurtoses, PAPR
- :mod:`~sigscope.features.spectral`      -- asymmetry, flatness, M-th power lines, CP peak

About 30 features, all cheap and all explainable to a judge -- which is what lets §5.6
write an evidence sentence for each one and what lets an analyst check the answer by hand.
"""

from __future__ import annotations

from sigscope.features.cumulants import (
    THEORETICAL_RATIOS,
    Cumulants,
    cumulants,
    symbol_sample,
)
from sigscope.features.instantaneous import InstantaneousStats, instantaneous_stats
from sigscope.features.spectral import SpectralStats, spectral_stats
from sigscope.features.vector import (
    FEATURE_NAMES,
    N_FEATURES,
    extract_feature_matrix,
    extract_features,
)

__all__ = [
    "Cumulants",
    "THEORETICAL_RATIOS",
    "cumulants",
    "symbol_sample",
    "InstantaneousStats",
    "instantaneous_stats",
    "SpectralStats",
    "spectral_stats",
    "FEATURE_NAMES",
    "N_FEATURES",
    "extract_features",
    "extract_feature_matrix",
]
