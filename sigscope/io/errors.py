"""Ingest error type (CLAUDE.md §3 "Stage 1", §9 acceptance test B).

Every unreadable / unrecognised input raises :class:`CaptureError` with a message that
names the problem. Callers never see a bare traceback.
"""

from __future__ import annotations


class CaptureError(Exception):
    """Raised when a file cannot be turned into ``(iq, CaptureMeta)``."""
