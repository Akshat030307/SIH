"""Frozen data model for the SIGSCOPE pipeline.

Implements the object graph the pipeline passes around (CLAUDE.md §2 "Coding rules")
and the frozen output schema (CLAUDE.md §3 "Frozen output schema"). ``Report.to_dict``
emits exactly that JSON shape, key for key. This schema does not change again — every
other module is built against it.

Conventions (§2): Hz, seconds, dB at every boundary; sample rate is always explicit;
every estimate carries a value **and** a confidence in 0..1 **and** the method name;
anything unmeasurable is ``None`` plus a ``warnings`` entry, never a placeholder number.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

SCHEMA_VERSION = "1.0"

__all__ = [
    "SCHEMA_VERSION",
    "Estimate",
    "FileInfo",
    "CaptureMeta",
    "Burst",
    "Bandwidth",
    "Modulation",
    "Detection",
    "Report",
]


@dataclass
class Estimate:
    """A single measured quantity: value + confidence + method (§2 "Coding rules").

    ``value`` is ``None`` when the estimator could not get a reliable answer; the reason
    goes in ``notes`` and, at report level, in ``Report.warnings``. Serialises to the
    ``{"value", "confidence", "method"}`` object used by ``symbol_rate_hz`` in §3.
    """

    value: float | None
    confidence: float
    method: str
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"value": self.value, "confidence": self.confidence, "method": self.method}


@dataclass
class FileInfo:
    """The ``file`` block of §3: identity of the input on disk."""

    name: str
    sha256: str
    bytes: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CaptureMeta:
    """Capture-level metadata recovered by Stage 1 — Ingest (§2 "Stage 1").

    Carries the §2 field names internally; ``to_capture_dict`` renames them to the
    ``capture`` block of the §3 schema. ``center_freq_source`` and
    ``frequencies_are_normalised`` are required by that schema and have no other home,
    so they live here too.

    All frequencies in Hz, times in seconds. ``center_freq`` is nullable — a headerless
    file with nothing in its name leaves it unknown. When the sample rate itself is
    unknown, Stage 1 sets ``sample_rate = 1.0`` and ``frequencies_are_normalised = True``
    so downstream frequencies are reported as a fraction of sample rate, never as a
    fabricated Hz value.
    """

    sample_rate: float
    center_freq: float | None
    dtype_guessed: str
    dtype_confidence: float
    source_format: str
    start_time: str | None
    n_samples: int
    metadata_confidence: float
    notes: list[str] = field(default_factory=list)
    center_freq_source: str | None = None
    frequencies_are_normalised: bool = False
    # stereo WAV interpreted as IQ (§3 Stage 1 step 2: "Flag iq_from_stereo=True")
    iq_from_stereo: bool = False
    # every dtype candidate and its score, so the user can override (§3 Stage 1)
    dtype_candidates: dict[str, float] = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        return self.n_samples / self.sample_rate if self.sample_rate else 0.0

    def to_capture_dict(self) -> dict[str, Any]:
        return {
            "sample_rate_hz": self.sample_rate,
            "center_freq_hz": self.center_freq,
            "center_freq_source": self.center_freq_source,
            "duration_s": self.duration_s,
            "dtype": self.dtype_guessed,
            "dtype_confidence": self.dtype_confidence,
            "frequencies_are_normalised": self.frequencies_are_normalised,
            "notes": list(self.notes),
        }


@dataclass
class Burst:
    """A time-frequency rectangle produced by Stage 3 — Detect (§4.3).

    ``t0``/``t1`` in seconds from the start of the capture; ``f_lo``/``f_hi`` in Hz
    (or a normalised fraction of sample rate when the rate is unknown). ``wideband``
    marks the §4.3 special case: a component covering most of the duration and
    bandwidth, i.e. probably the whole channel.
    """

    t0: float
    t1: float
    f_lo: float
    f_hi: float
    wideband: bool = False

    @property
    def duration(self) -> float:
        return self.t1 - self.t0

    @property
    def f_center(self) -> float:
        return 0.5 * (self.f_lo + self.f_hi)

    @property
    def bandwidth(self) -> float:
        return self.f_hi - self.f_lo


@dataclass
class Bandwidth:
    """The three bandwidth numbers of §4.6, all reported side by side.

    ``occupied_99`` is the headline 99% occupied bandwidth (§4.6 OBW99); ``minus_3db``
    and ``minus_20db`` are the widths at 3 dB and 20 dB below the peak. Any value that
    could not be measured is ``None``.
    """

    occupied_99: float | None = None
    minus_3db: float | None = None
    minus_20db: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "occupied_99": self.occupied_99,
            "minus_3db": self.minus_3db,
            "minus_20db": self.minus_20db,
        }


@dataclass
class Modulation:
    """Ensemble classification result and its evidence (§3, §5.5, §5.6).

    ``votes`` maps voter name -> label (``None`` if that voter abstained), e.g.
    ``{"feature_clf": "QPSK", "cnn": "QPSK", "rules": None}``. ``evidence`` is the list
    of plain human sentences from §5.6 — always present, minimum two.
    """

    label: str
    confidence: float
    runner_up: str | None = None
    runner_up_confidence: float | None = None
    votes: dict[str, str | None] = field(default_factory=dict)
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "confidence": self.confidence,
            "runner_up": self.runner_up,
            "runner_up_confidence": self.runner_up_confidence,
            "votes": dict(self.votes),
            "evidence": list(self.evidence),
        }


@dataclass
class Detection:
    """One separate signal found inside the capture — an entry of ``detections`` in §3.

    Frequencies in Hz (or normalised), times in seconds, powers in dB. ``snr_db`` may
    carry the string sentinel ``"< 0 dB"`` when the signal power comes out non-positive
    (§4.7) rather than a NaN. Any unmeasured scalar is ``None`` with a matching
    ``Report.warnings`` entry. ``extra`` holds modulation sub-parameters (§3
    "Sub-parameters"): ``psk_order``, ``fsk_deviation_hz``, ``n_tones``,
    ``ofdm_subcarrier_spacing_hz``, ``chirp_rate_hz_per_s``, ``carrier_offset_hz`` …
    """

    id: int
    time_start_s: float
    time_stop_s: float
    duration_s: float
    center_freq_hz: float | None
    center_freq_offset_hz: float | None
    bandwidth_hz: Bandwidth
    snr_db: float | str | None
    power_dbfs: float | None
    symbol_rate_hz: Estimate | None = None
    modulation: Modulation | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "time_start_s": self.time_start_s,
            "time_stop_s": self.time_stop_s,
            "duration_s": self.duration_s,
            "center_freq_hz": self.center_freq_hz,
            "center_freq_offset_hz": self.center_freq_offset_hz,
            "bandwidth_hz": self.bandwidth_hz.to_dict(),
            "snr_db": self.snr_db,
            "power_dbfs": self.power_dbfs,
            "symbol_rate_hz": self.symbol_rate_hz.to_dict() if self.symbol_rate_hz else None,
            "modulation": self.modulation.to_dict() if self.modulation else None,
            "extra": dict(self.extra),
        }


@dataclass
class Report:
    """The single internal object behind all three outputs (§3 "Stage 6 — Report").

    ``to_dict`` emits the frozen §3 schema exactly; ``report.json`` is that dict
    serialised. The SigMF and HTML writers consume the same object.
    """

    file: FileInfo
    capture: CaptureMeta
    noise_floor_dbfs: float | None
    detections: list[Detection] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    runtime_s: float = 0.0
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "file": self.file.to_dict(),
            "capture": self.capture.to_capture_dict(),
            "noise_floor_dbfs": self.noise_floor_dbfs,
            "detections": [d.to_dict() for d in self.detections],
            "warnings": list(self.warnings),
            "runtime_s": self.runtime_s,
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
