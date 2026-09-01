"""Headerless raw-IQ reader (CLAUDE.md §3 Stage 1, step 3).

Interleaved I,Q samples with no header. The sample format comes from the extension
(``.cs8`` / ``.cs16`` / ``.cf32`` / ``.cfile``) when present, otherwise from the
histogram heuristic in :mod:`sigscope.io._dtypes`. Large files are read in overlapping
blocks (CLAUDE.md §2 Stage 2) via :func:`iter_raw_blocks` so peak memory stays bounded.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from sigscope.io._dtypes import (
    EXT_DTYPE,
    decode_interleaved,
    guess_dtype,
    itemsize,
    numpy_dtype,
)
from sigscope.io.errors import CaptureError

_SAMPLE_BUDGET = 4 << 20  # bytes inspected by the dtype guesser


@dataclass
class RawCapture:
    iq: np.ndarray
    dtype: str
    dtype_confidence: float
    dtype_scores: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


# (dtype name, confidence, per-candidate scores, notes)
_Resolved = tuple[str, float, dict[str, float], list[str]]


def _resolve_dtype(path: Path, override: str | None) -> _Resolved:
    file_bytes = path.stat().st_size
    if file_bytes == 0:
        raise CaptureError(f"{path}: file is empty")

    if override is not None:
        if override not in ("int8", "int16", "float32"):
            raise CaptureError(f"unknown dtype override {override!r}; use int8/int16/float32")
        if file_bytes % (itemsize(override) * 2) != 0:
            raise CaptureError(
                f"{path}: size {file_bytes} B is not a whole number of {override} IQ samples"
            )
        return override, 1.0, {}, [f"dtype {override} forced by caller"]

    ext_hint = EXT_DTYPE.get(path.suffix.lower())
    if ext_hint is not None:
        if file_bytes % (itemsize(ext_hint) * 2) != 0:
            raise CaptureError(
                f"{path}: {path.suffix} implies interleaved {ext_hint}, but size {file_bytes} B "
                "is not a multiple of one complex sample"
            )
        return ext_hint, 0.95, {}, [f"dtype {ext_hint} from '{path.suffix}' extension convention"]

    with open(path, "rb") as fh:
        sample = fh.read(_SAMPLE_BUDGET)
    guess = guess_dtype(sample, file_bytes)
    if max(guess.scores.values(), default=0.0) < 0.01:
        pretty = {k: round(v, 3) for k, v in guess.scores.items()}
        raise CaptureError(
            f"{path}: does not look like raw interleaved IQ "
            f"(tried int8/int16/float32; plausibility {pretty}). "
            "If this is text or another container, convert it first."
        )
    return guess.name, guess.confidence, guess.scores, guess.notes


def read_raw(path: Path, *, dtype: str | None = None) -> RawCapture:
    """Read a whole headerless file into ``complex64``."""
    path = Path(path)
    name, conf, scores, notes = _resolve_dtype(path, dtype)
    raw = np.fromfile(path, dtype=numpy_dtype(name))
    if raw.size < 2:
        raise CaptureError(f"{path}: fewer than one complete IQ sample")
    return RawCapture(
        iq=decode_interleaved(raw, name),
        dtype=name,
        dtype_confidence=conf,
        dtype_scores=scores,
        notes=notes,
    )


def iter_raw_blocks(
    path: Path,
    *,
    dtype: str | None = None,
    block_samples: int = 1 << 20,
    overlap: float = 0.25,
) -> tuple[str, int, Iterator[np.ndarray]]:
    """Return ``(dtype, n_samples, block_iterator)``; blocks are overlapping ``complex64``.

    Each block is **read** from the file rather than sliced out of a mapping (CLAUDE.md
    §9 B, "2 GB file processed in blocks with peak RSS under 2 GB").

    An earlier version mapped the file with ``np.memmap`` and copied a window per block.
    That keeps the heap bounded, but on Windows every page the mapping touches joins the
    process working set and stays there: walking a 2 GB capture pushed measured RSS to
    1.98 GB against a 2 GB ceiling, with barely 1 GB of it actually on the heap. Explicit
    reads give the same bounded heap with no file-backed residency to accumulate, so the
    figure §9 B measures is the figure the pipeline actually needs.
    """
    path = Path(path)
    name, _conf, _scores, _notes = _resolve_dtype(path, dtype)
    element = np.dtype(numpy_dtype(name))
    total_elements = path.stat().st_size // element.itemsize
    n_samples = int(total_elements // 2)
    if n_samples < 1:
        raise CaptureError(f"{path}: fewer than one complete IQ sample")
    step = max(1, int(block_samples * (1.0 - overlap)))

    def _blocks() -> Iterator[np.ndarray]:
        with open(path, "rb") as handle:
            for start in range(0, n_samples, step):
                stop = min(start + block_samples, n_samples)
                handle.seek(2 * start * element.itemsize)
                seg = np.fromfile(handle, dtype=element, count=2 * (stop - start))
                if seg.size == 0:
                    break
                yield decode_interleaved(seg, name)
                if stop == n_samples:
                    break

    return name, n_samples, _blocks()
