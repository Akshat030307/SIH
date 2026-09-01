"""DSP stages -- Condition, Detect, Isolate, Measure (CLAUDE.md §2, §4).

Landed so far: spectrogram (§4.1), robust noise floor (§4.2), detection (§4.3), a minimal
Stage-2 conditioner (DC removal, power normalisation), and the Phase 4 classical
estimators (§4.4-§4.13) in :mod:`sigscope.dsp.estimators`.

Every function here is pure: numpy array in, dataclass out, no I/O, no global state,
``fs`` always explicit.
"""

from __future__ import annotations

from sigscope.dsp.condition import ConditionInfo, condition
from sigscope.dsp.detect import DetectionResult, DetectorConfig, detect_bursts
from sigscope.dsp.estimators import (
    BandwidthResult,
    CenterFreq,
    ChirpParams,
    CwKeying,
    EstimatorConfig,
    FskParams,
    IsolatedBurst,
    MthPowerResult,
    OfdmParams,
    SnrResult,
    SymbolRateResult,
    estimate_am_depth,
    estimate_bandwidth,
    estimate_center_freq,
    estimate_chirp,
    estimate_cw_keying,
    estimate_fm_deviation,
    estimate_fsk_params,
    estimate_ofdm_params,
    estimate_psk_order,
    estimate_snr,
    estimate_spectral_asymmetry,
    estimate_symbol_rate,
    instantaneous_frequency,
    isolate_burst,
    symbol_rate_cyclostationary,
    symbol_rate_envelope_autocorr,
    symbol_rate_freq_transitions,
)
from sigscope.dsp.noise import NoiseEstimate, estimate_noise
from sigscope.dsp.spectrogram import Spectrogram, choose_nfft, compute_spectrogram

__all__ = [
    # Stage 2 -- condition
    "condition",
    "ConditionInfo",
    # §4.1-§4.3 -- spectrogram, noise, detection
    "compute_spectrogram",
    "choose_nfft",
    "Spectrogram",
    "estimate_noise",
    "NoiseEstimate",
    "detect_bursts",
    "DetectorConfig",
    "DetectionResult",
    # §4.4-§4.13 -- isolate and measure
    "EstimatorConfig",
    "IsolatedBurst",
    "isolate_burst",
    "CenterFreq",
    "estimate_center_freq",
    "BandwidthResult",
    "estimate_bandwidth",
    "SnrResult",
    "estimate_snr",
    "SymbolRateResult",
    "estimate_symbol_rate",
    "symbol_rate_cyclostationary",
    "symbol_rate_freq_transitions",
    "symbol_rate_envelope_autocorr",
    "MthPowerResult",
    "estimate_psk_order",
    "FskParams",
    "estimate_fsk_params",
    "OfdmParams",
    "estimate_ofdm_params",
    "ChirpParams",
    "estimate_chirp",
    "CwKeying",
    "estimate_cw_keying",
    "estimate_am_depth",
    "estimate_fm_deviation",
    "estimate_spectral_asymmetry",
    "instantaneous_frequency",
]
