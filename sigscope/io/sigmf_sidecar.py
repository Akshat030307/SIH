"""SigMF sidecar reader (CLAUDE.md §3 Stage 1, step 1).

If a ``.sigmf-meta`` sidecar is present it is authoritative: sample rate, centre
frequency, start time and sample format are all read from it with no guessing.
The ``.sigmf-meta`` is parsed as plain JSON (the ``sigmf`` library is used for *writing*
annotations later, §3 Stage 6).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import hilbert

from sigscope.io.errors import CaptureError

# SigMF core:datatype component -> (numpy base dtype, full-scale divisor, bias)
_BASE: dict[tuple[str, int], tuple[str, float, float]] = {
    ("i", 8): ("i1", 128.0, 0.0),
    ("u", 8): ("u1", 128.0, 127.5),
    ("i", 16): ("i2", 32768.0, 0.0),
    ("u", 16): ("u2", 32768.0, 32767.5),
    ("i", 32): ("i4", 2147483648.0, 0.0),
    ("f", 32): ("f4", 1.0, 0.0),
    ("f", 64): ("f8", 1.0, 0.0),
}


@dataclass
class SigmfCapture:
    iq: np.ndarray
    sample_rate: float | None
    center_freq: float | None
    start_time: str | None
    datatype: str
    notes: list[str]


def find_sidecar(path: Path) -> tuple[Path, Path] | None:
    """Return ``(meta_path, data_path)`` if ``path`` is part of a SigMF pair, else None."""
    p = Path(path)
    if p.suffix == ".sigmf-meta":
        return p, p.with_suffix(".sigmf-data")
    if p.suffix == ".sigmf-data":
        return p.with_suffix(".sigmf-meta"), p
    for meta in (p.with_suffix(".sigmf-meta"), Path(str(p) + ".sigmf-meta")):
        if meta.is_file():
            return meta, p
    return None


def _parse_datatype(spec: str) -> tuple[np.dtype, bool, float, float]:
    """``'ci16_le'`` -> (numpy dtype, is_complex, divisor, bias)."""
    s = spec.strip().lower()
    if s[:1] not in ("c", "r"):
        raise CaptureError(f"SigMF core:datatype {spec!r} must start with 'c' or 'r'")
    is_complex = s[0] == "c"
    body = s[1:]
    endian = "<"
    if body.endswith("_le"):
        body = body[:-3]
    elif body.endswith("_be"):
        body, endian = body[:-3], ">"
    kind = body[0]
    try:
        bits = int(body[1:])
    except ValueError as exc:
        raise CaptureError(f"SigMF core:datatype {spec!r} is not understood") from exc
    if (kind, bits) not in _BASE:
        raise CaptureError(f"SigMF core:datatype {spec!r} is not a supported format")
    base, divisor, bias = _BASE[(kind, bits)]
    return np.dtype(endian + base), is_complex, divisor, bias


def read_sigmf(meta_path: Path, data_path: Path) -> SigmfCapture:
    """Read a SigMF meta/data pair exactly as specified (no guessing)."""
    try:
        meta = json.loads(Path(meta_path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise CaptureError(f"{meta_path}: unreadable SigMF metadata ({exc})") from exc
    if not isinstance(meta, dict) or "global" not in meta:
        raise CaptureError(f"{meta_path}: not a SigMF metadata document")
    if not Path(data_path).is_file():
        raise CaptureError(f"{data_path} not found next to {Path(meta_path).name}")

    g = meta["global"]
    spec = g.get("core:datatype")
    if not spec:
        raise CaptureError(f"{meta_path}: global.'core:datatype' is missing")
    np_dtype, is_complex, divisor, bias = _parse_datatype(spec)

    raw = np.fromfile(data_path, dtype=np_dtype)
    if raw.size == 0:
        raise CaptureError(f"{data_path} is empty")
    samples = (raw.astype(np.float64) - bias) / divisor
    if is_complex:
        samples = samples[: (samples.size // 2) * 2]
        iq = (samples[0::2] + 1j * samples[1::2]).astype(np.complex64)
    else:
        iq = hilbert(samples).astype(np.complex64)

    caps = meta.get("captures") or [{}]
    first = caps[0] if isinstance(caps[0], dict) else {}
    sample_rate = g.get("core:sample_rate")
    freq = first.get("core:frequency")
    notes = [f"SigMF sidecar {Path(meta_path).name}: datatype {spec}, trusted verbatim"]
    if not is_complex:
        notes.append("SigMF real recording converted to analytic baseband via Hilbert transform")
    return SigmfCapture(
        iq=iq,
        sample_rate=float(sample_rate) if sample_rate else None,
        center_freq=float(freq) if freq is not None else None,
        start_time=first.get("core:datetime"),
        datatype=spec,
        notes=notes,
    )
