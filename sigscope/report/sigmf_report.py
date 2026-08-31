"""SigMF annotation writer (CLAUDE.md §3, §10 "SigMF export").

Emits ``capture.sigmf-meta`` with one annotation per detection so NTRO tooling that
already reads SigMF can ingest the results directly. Output must load in the ``sigmf``
library (§9 acceptance test E).

Not implemented yet (Phase 5, §8).
"""

from __future__ import annotations
