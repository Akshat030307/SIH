"""``report.json`` writer (CLAUDE.md §3 "Frozen output schema").

Serialises a :class:`sigscope.types.Report` via ``Report.to_dict``, which *is* the frozen
schema -- this module adds only the file handling and a validator.

The validator exists because §3 froze the schema on purpose ("Everyone builds against it.
Half of all hackathon failures are integration failures on the last night"), and a frozen
schema nobody checks is just a comment. :func:`validate_report_dict` is what the API, the
dashboard and the tests all assert against.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sigscope.types import SCHEMA_VERSION, Report

__all__ = ["write_json", "validate_report_dict", "SchemaError"]


class SchemaError(ValueError):
    """A report dict does not match the frozen §3 schema."""


_TOP_LEVEL = {
    "schema_version",
    "file",
    "capture",
    "noise_floor_dbfs",
    "detections",
    "warnings",
    "runtime_s",
}
_FILE_KEYS = {"name", "sha256", "bytes"}
_CAPTURE_KEYS = {
    "sample_rate_hz",
    "center_freq_hz",
    "center_freq_source",
    "duration_s",
    "dtype",
    "dtype_confidence",
    "frequencies_are_normalised",
    "notes",
}
_DETECTION_KEYS = {
    "id",
    "time_start_s",
    "time_stop_s",
    "duration_s",
    "center_freq_hz",
    "center_freq_offset_hz",
    "bandwidth_hz",
    "snr_db",
    "power_dbfs",
    "symbol_rate_hz",
    "modulation",
    "extra",
}
_BANDWIDTH_KEYS = {"occupied_99", "minus_3db", "minus_20db"}
_MODULATION_KEYS = {
    "label",
    "confidence",
    "runner_up",
    "runner_up_confidence",
    "votes",
    "evidence",
}
_ESTIMATE_KEYS = {"value", "confidence", "method"}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SchemaError(message)


def _check_keys(obj: Any, expected: set[str], where: str) -> None:
    _require(isinstance(obj, dict), f"{where}: expected an object, got {type(obj).__name__}")
    missing = expected - set(obj)
    extra = set(obj) - expected
    _require(not missing, f"{where}: missing keys {sorted(missing)}")
    _require(not extra, f"{where}: unexpected keys {sorted(extra)}")


def _check_number(value: Any, where: str, *, nullable: bool = True) -> None:
    if value is None:
        _require(nullable, f"{where}: must not be null")
        return
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{where}: expected a number, got {value!r}",
    )
    _require(value == value and value not in (float("inf"), float("-inf")),
             f"{where}: must be finite, got {value!r}")


def validate_report_dict(data: dict[str, Any]) -> None:
    """Raise :class:`SchemaError` unless ``data`` matches the frozen §3 schema exactly.

    Checks structure and types, and enforces the two §3 rules that are easy to break and
    expensive to discover late: anything unmeasurable is ``null`` rather than a placeholder
    number, and every detection carries at least two evidence sentences.
    """
    _check_keys(data, _TOP_LEVEL, "report")
    _require(
        data["schema_version"] == SCHEMA_VERSION,
        f"report.schema_version: expected {SCHEMA_VERSION!r}, got {data['schema_version']!r}",
    )

    _check_keys(data["file"], _FILE_KEYS, "report.file")
    _check_keys(data["capture"], _CAPTURE_KEYS, "report.capture")
    _check_number(data["capture"]["sample_rate_hz"], "capture.sample_rate_hz", nullable=False)
    _check_number(data["capture"]["center_freq_hz"], "capture.center_freq_hz")
    _check_number(data["noise_floor_dbfs"], "report.noise_floor_dbfs")
    _check_number(data["runtime_s"], "report.runtime_s", nullable=False)
    _require(isinstance(data["warnings"], list), "report.warnings: expected a list")
    _require(
        all(isinstance(w, str) for w in data["warnings"]),
        "report.warnings: every entry must be a string",
    )
    _require(isinstance(data["detections"], list), "report.detections: expected a list")

    for detection in data["detections"]:
        where = f"detection {detection.get('id', '?')}"
        _check_keys(detection, _DETECTION_KEYS, where)
        for key in ("time_start_s", "time_stop_s", "duration_s"):
            _check_number(detection[key], f"{where}.{key}", nullable=False)
        for key in ("center_freq_hz", "center_freq_offset_hz", "power_dbfs"):
            _check_number(detection[key], f"{where}.{key}")

        # §4.7 allows the string sentinel "< 0 dB" here, and nothing else non-numeric
        snr = detection["snr_db"]
        if isinstance(snr, str):
            _require(snr == "< 0 dB", f"{where}.snr_db: unexpected string {snr!r}")
        else:
            _check_number(snr, f"{where}.snr_db")

        _check_keys(detection["bandwidth_hz"], _BANDWIDTH_KEYS, f"{where}.bandwidth_hz")
        for key in _BANDWIDTH_KEYS:
            _check_number(detection["bandwidth_hz"][key], f"{where}.bandwidth_hz.{key}")

        rate = detection["symbol_rate_hz"]
        if rate is not None:
            _check_keys(rate, _ESTIMATE_KEYS, f"{where}.symbol_rate_hz")
            _check_number(rate["value"], f"{where}.symbol_rate_hz.value")
            _check_number(rate["confidence"], f"{where}.symbol_rate_hz.confidence",
                          nullable=False)
            _require(
                0.0 <= rate["confidence"] <= 1.0,
                f"{where}.symbol_rate_hz.confidence: must be in 0..1",
            )
            _require(
                isinstance(rate["method"], str) and rate["method"],
                f"{where}.symbol_rate_hz.method: must name the method",
            )

        modulation = detection["modulation"]
        if modulation is not None:
            _check_keys(modulation, _MODULATION_KEYS, f"{where}.modulation")
            _require(
                isinstance(modulation["label"], str) and modulation["label"],
                f"{where}.modulation.label: must be a non-empty string",
            )
            _check_number(modulation["confidence"], f"{where}.modulation.confidence",
                          nullable=False)
            _require(
                0.0 <= modulation["confidence"] <= 1.0,
                f"{where}.modulation.confidence: must be in 0..1",
            )
            _require(
                isinstance(modulation["evidence"], list) and len(modulation["evidence"]) >= 2,
                f"{where}.modulation.evidence: §3 requires at least two sentences, got "
                f"{len(modulation.get('evidence') or [])}",
            )
        _require(isinstance(detection["extra"], dict), f"{where}.extra: expected an object")


def write_json(report: Report, path: str | Path, *, indent: int | None = 2) -> Path:
    """Write ``report.json`` and validate it on the way out.

    Validating here rather than only in tests means a schema violation surfaces as a clear
    error at the moment the file is written, not as a puzzling failure in the dashboard.
    """
    path = Path(path)
    data = report.to_dict()
    validate_report_dict(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=indent) + "\n", encoding="utf-8")
    return path
