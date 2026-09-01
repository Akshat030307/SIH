"""The assembled feature vector (CLAUDE.md §5.2 "About 30 features").

Joins the three §5.2 blocks -- cumulants, instantaneous statistics, spectral statistics --
into one fixed-length, fixed-order vector with a matching list of names.

The order is frozen by :data:`FEATURE_NAMES`. A model is trained against a specific column
order; if the order ever changed under a saved checkpoint every prediction would silently
become nonsense, so the classifier records the names it was fitted on and refuses to
predict against a mismatch.

Every feature is finite by construction: :func:`extract_features` sweeps the vector at the
end and replaces anything non-finite with zero. Sklearn's histogram gradient boosting
tolerates NaN, but the scaler does not, and a NaN reaching the CNN poisons an entire batch.
"""

from __future__ import annotations

import numpy as np

from sigscope.features.cumulants import cumulants
from sigscope.features.instantaneous import instantaneous_stats
from sigscope.features.spectral import spectral_stats

__all__ = ["FEATURE_NAMES", "N_FEATURES", "extract_features", "extract_feature_matrix"]

# Frozen column order. Adding a feature means appending here and retraining.
FEATURE_NAMES: tuple[str, ...] = (
    # §5.2 higher-order cumulants -- the classic tell
    "c20_abs",
    "c21",
    "c40_abs",
    "c41_abs",
    "c42_abs",
    "c63_abs",
    "ratio_c40",
    "ratio_c42",
    # §5.2 instantaneous statistics
    "gamma_max",
    "sigma_ap",
    "sigma_dp",
    "sigma_aa",
    "sigma_af",
    "kurtosis_amplitude",
    "kurtosis_freq",
    "papr_db",
    "amplitude_variance",
    # §5.2 spectral features
    "asymmetry_db",
    "spectral_flatness",
    "n_freq_peaks",
    "line_strength_m2",
    "line_strength_m4",
    "line_strength_m8",
    "cp_autocorr_peak",
    "cp_autocorr_ratio",
    "occupied_fraction",
    # derived combinations that separate classes the raw features leave adjacent
    "c40_minus_c42",
    "line_m4_minus_m2",
    "line_m8_minus_m4",
)

N_FEATURES = len(FEATURE_NAMES)


def extract_features(y: np.ndarray, *, sps: int | None = None) -> np.ndarray:
    """One burst -> the §5.2 feature vector, in :data:`FEATURE_NAMES` order.

    ``sps`` (samples per symbol) is passed through to the cumulants so they can be
    symbol-sampled where the symbol rate is known -- that is what makes them match the
    §5.2 theoretical table rather than sitting 20% low. It is left ``None`` for RadioML
    training windows, where the symbol rate is not given, so that training and inference
    see the feature computed the same way.
    """
    y = np.asarray(y, dtype=np.complex128).ravel()

    try:
        cums = cumulants(y, sps=sps)
        cumulant_block = [
            abs(cums.c20), cums.c21, abs(cums.c40), abs(cums.c41),
            abs(cums.c42), abs(cums.c63), cums.ratio_c40, cums.ratio_c42,
        ]
    except ValueError:
        cumulant_block = [0.0] * 8

    inst = instantaneous_stats(y)
    spec = spectral_stats(y)

    values = np.array(
        [
            *cumulant_block,
            inst.gamma_max, inst.sigma_ap, inst.sigma_dp, inst.sigma_aa, inst.sigma_af,
            inst.kurtosis_amplitude, inst.kurtosis_freq, inst.papr_db,
            inst.amplitude_variance,
            spec.asymmetry_db, spec.spectral_flatness, spec.n_freq_peaks,
            spec.line_strength_m2, spec.line_strength_m4, spec.line_strength_m8,
            spec.cp_autocorr_peak, spec.cp_autocorr_ratio, spec.occupied_fraction,
            # derived
            cumulant_block[6] - cumulant_block[7],
            spec.line_strength_m4 - spec.line_strength_m2,
            spec.line_strength_m8 - spec.line_strength_m4,
        ],
        dtype=np.float64,
    )

    if values.size != N_FEATURES:  # pragma: no cover -- guards the frozen order
        raise RuntimeError(
            f"feature vector is {values.size} long but FEATURE_NAMES has {N_FEATURES}"
        )
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)


def extract_feature_matrix(
    bursts: np.ndarray, *, sps: int | None = None, progress: int = 0
) -> np.ndarray:
    """Feature matrix for a stack of bursts, shape ``(n, N_FEATURES)``.

    ``bursts`` is either an ``(n, samples)`` complex array or an ``(n, 2, samples)`` real
    array in the RadioML layout, which is converted to complex here.
    """
    bursts = np.asarray(bursts)
    if bursts.ndim == 3 and bursts.shape[1] == 2:
        bursts = bursts[:, 0, :] + 1j * bursts[:, 1, :]
    if bursts.ndim != 2:
        raise ValueError(f"expected (n, samples) or (n, 2, samples), got {bursts.shape}")

    out = np.empty((bursts.shape[0], N_FEATURES), dtype=np.float64)
    for i, burst in enumerate(bursts):
        out[i] = extract_features(burst, sps=sps)
        if progress and i and i % progress == 0:
            print(f"    features {i:,}/{bursts.shape[0]:,}", flush=True)
    return out
