"""SigMF annotation writer (CLAUDE.md §3, §10 "SigMF export").

Emits a ``.sigmf-meta`` with one annotation per detection so NTRO tooling that already
reads SigMF can ingest our results directly -- §10's line is "We are not asking anyone to
adopt our format". §9 E requires the output to load in the ``sigmf`` library, which
:func:`write_sigmf` checks before it returns.

Frequency edges are the one place this has to be careful. SigMF's ``core:freq_lower_edge``
and ``core:freq_upper_edge`` are **absolute RF**, so they are written only when the capture
centre frequency is actually known. When it is not, they are omitted and the baseband
offsets go in the annotation description instead -- §3 forbids inventing a centre
frequency, and silently writing baseband offsets into fields that mean absolute RF would
be exactly that.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sigscope.types import Detection, Report

__all__ = ["build_sigmf_meta", "write_sigmf"]

# SigMF core keys (module-level constants; SigMFFile.* aliases are deprecated in sigmf 1.13)
_GLOBAL = "global"
_CAPTURES = "captures"
_ANNOTATIONS = "annotations"
_SIGMF_VERSION = "1.0.0"

# our own namespace for the measurements SigMF core has no field for
_NS = "sigscope"


def _dtype_to_sigmf(dtype: str, *, normalised: bool) -> str:
    """Best-effort map from our on-disk dtype label to a SigMF ``core:datatype``.

    We always hand the pipeline ``complex64``, so ``cf32_le`` is the honest description of
    what our annotations refer to regardless of how the file was stored on disk.
    """
    del dtype, normalised
    return "cf32_le"


def _annotation(detection: Detection, report: Report) -> dict[str, Any]:
    """One SigMF annotation for one detection."""
    fs = report.capture.sample_rate
    start = int(round(detection.time_start_s * fs))
    count = max(1, int(round(detection.duration_s * fs)))

    label = detection.modulation.label if detection.modulation else "unclassified"
    annotation: dict[str, Any] = {
        "core:sample_start": start,
        "core:sample_count": count,
        "core:label": label,
        "core:generator": "sigscope",
    }

    centre = detection.center_freq_hz
    half = (detection.bandwidth_hz.occupied_99 or 0.0) / 2.0
    if centre is not None and half > 0 and not report.capture.frequencies_are_normalised:
        annotation["core:freq_lower_edge"] = centre - half
        annotation["core:freq_upper_edge"] = centre + half

    description = [f"detection {detection.id}"]
    if detection.center_freq_offset_hz is not None:
        unit = "x fs" if report.capture.frequencies_are_normalised else "Hz"
        description.append(f"offset {detection.center_freq_offset_hz:+.1f} {unit} from centre")
    if detection.bandwidth_hz.occupied_99 is not None:
        description.append(f"OBW99 {detection.bandwidth_hz.occupied_99:.1f}")
    if isinstance(detection.snr_db, (int, float)):
        description.append(f"SNR {detection.snr_db:.1f} dB")
    elif isinstance(detection.snr_db, str):
        description.append(f"SNR {detection.snr_db}")
    annotation["core:description"] = "; ".join(description)

    # everything SigMF core has no home for goes under our own namespace, which is what
    # the SigMF spec says a custom extension should do
    measured: dict[str, Any] = {
        "center_freq_offset_hz": detection.center_freq_offset_hz,
        "bandwidth_hz": detection.bandwidth_hz.to_dict(),
        "snr_db": detection.snr_db,
        "power_dbfs": detection.power_dbfs,
    }
    if detection.symbol_rate_hz is not None:
        measured["symbol_rate_hz"] = detection.symbol_rate_hz.to_dict()
    if detection.modulation is not None:
        measured["modulation_confidence"] = detection.modulation.confidence
        measured["evidence"] = list(detection.modulation.evidence)
    if detection.extra:
        measured["extra"] = detection.extra
    annotation[f"{_NS}:measured"] = measured
    return annotation


def build_sigmf_meta(report: Report) -> dict[str, Any]:
    """Build the ``.sigmf-meta`` dictionary for a report (CLAUDE.md §3 Stage 6)."""
    capture = report.capture

    global_block: dict[str, Any] = {
        "core:datatype": _dtype_to_sigmf(
            capture.dtype_guessed, normalised=capture.frequencies_are_normalised
        ),
        "core:version": _SIGMF_VERSION,
        "core:num_channels": 1,
        "core:recorder": "sigscope",
        "core:description": (
            f"sigscope analysis of {report.file.name}: "
            f"{len(report.detections)} detection(s)"
        ),
    }
    if not capture.frequencies_are_normalised:
        global_block["core:sample_rate"] = float(capture.sample_rate)
    if report.file.sha256:
        global_block[f"{_NS}:source_sha256"] = report.file.sha256

    global_block[f"{_NS}:analysis"] = {
        "schema_version": report.schema_version,
        "noise_floor_dbfs": report.noise_floor_dbfs,
        "frequencies_are_normalised": capture.frequencies_are_normalised,
        "sample_rate_is_assumed": capture.frequencies_are_normalised,
        "dtype_guessed": capture.dtype_guessed,
        "dtype_confidence": capture.dtype_confidence,
        "metadata_confidence": capture.metadata_confidence,
        "warnings": list(report.warnings),
        "notes": list(capture.notes),
        "runtime_s": report.runtime_s,
    }

    capture_block: dict[str, Any] = {"core:sample_start": 0}
    if capture.center_freq is not None and not capture.frequencies_are_normalised:
        capture_block["core:frequency"] = float(capture.center_freq)
    if capture.start_time:
        capture_block["core:datetime"] = capture.start_time

    return {
        _GLOBAL: global_block,
        _CAPTURES: [capture_block],
        _ANNOTATIONS: [_annotation(d, report) for d in report.detections],
    }


def write_sigmf(report: Report, path: str | Path, *, validate: bool = True) -> Path:
    """Write ``capture.sigmf-meta`` and confirm the ``sigmf`` library can load it (§9 E).

    The round-trip check is the point: §10 promises the output "drops into existing
    tooling", so a file that our own writer produces but the reference library rejects is
    a failed promise, and better caught here than on stage.
    """
    path = Path(path)
    if path.suffix != ".sigmf-meta":
        path = path.with_suffix(".sigmf-meta")
    meta = build_sigmf_meta(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    if validate:
        _assert_loads(path)
    return path


def _assert_loads(path: Path) -> None:
    """Load the written file back through the ``sigmf`` library (§9 acceptance test E)."""
    from sigmf import SigMFFile

    data = json.loads(path.read_text(encoding="utf-8"))
    handle = SigMFFile(metadata=data)
    # metadata-only: there is no .sigmf-data next to an annotation sidecar we produced,
    # so validate the metadata rather than the (absent) recording
    handle.validate()
