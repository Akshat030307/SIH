"""DSP stages -- Condition, Detect, Isolate, Measure (CLAUDE.md §2, §4).

Landed so far (Phase 3): spectrogram (§4.1), robust noise floor (§4.2), detection (§4.3),
plus a minimal Stage-2 conditioner (DC removal, power normalisation). Isolation (§4.4)
and the parameter estimators (§4.5-§4.13) are Phase 4.

Every function here is pure: numpy array in, dataclass out, no I/O, no global state,
``fs`` always explicit.
"""

from __future__ import annotations

from sigscope.dsp.condition import ConditionInfo, condition
from sigscope.dsp.detect import DetectionResult, DetectorConfig, detect_bursts
from sigscope.dsp.noise import NoiseEstimate, estimate_noise
from sigscope.dsp.spectrogram import Spectrogram, choose_nfft, compute_spectrogram

__all__ = [
    "condition",
    "ConditionInfo",
    "compute_spectrogram",
    "choose_nfft",
    "Spectrogram",
    "estimate_noise",
    "NoiseEstimate",
    "detect_bursts",
    "DetectorConfig",
    "DetectionResult",
]
