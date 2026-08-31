"""Stage 1 — Ingest (CLAUDE.md §2 "Stage 1", §3).

Public API:
  - :func:`read_capture`  -- file -> ``(iq: complex64[N], CaptureMeta)``
  - :func:`iter_blocks`   -- same, streamed as overlapping blocks for large files
  - :class:`CaptureError` -- raised (never a bare traceback) on any unreadable input

Complex baseband is ``np.complex64``, converted once here at the edge (§2 "Coding rules").
"""

from __future__ import annotations

from sigscope.io.core import iter_blocks, read_capture
from sigscope.io.errors import CaptureError

__all__ = ["read_capture", "iter_blocks", "CaptureError"]
