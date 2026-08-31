"""Local FastAPI application (CLAUDE.md §7 "API").

Started by ``sigscope serve``. Endpoints for upload/analyse, job polling, report JSON,
spectrogram PNG/JSON, demodulated audio, SigMF download, batch, and health. Background
jobs run in a thread with an in-memory job table — no Celery, no Redis. Local only;
a comment marks where auth would go.

Not implemented yet (Phase 7, §8).
"""

from __future__ import annotations
