"""SIGSCOPE — automated analysis of .iq and .wav radio captures (SIH26147 / NTRO).

The full build spec is in CLAUDE.md. Pipeline stages (§3):
INGEST -> CONDITION -> DETECT -> ISOLATE -> MEASURE -> CLASSIFY -> REPORT.
"""

from __future__ import annotations

__version__ = "0.1.0"
