"""Local FastAPI application (CLAUDE.md §7 "API").

Started by ``sigscope serve``. Endpoints for upload/analyse, job polling, report JSON,
spectrogram PNG/JSON, demodulated audio, SigMF download, batch, and health. Background
jobs run in a thread with an in-memory job table -- no Celery, no Redis. Local only;
:func:`~sigscope.api.app.create_app` carries the note marking where auth would go.
"""

from __future__ import annotations

from sigscope.api.app import MAX_UPLOAD_BYTES, create_app, serve
from sigscope.api.audio import AudioResult, demodulate
from sigscope.api.jobs import Job, JobState, JobStore

__all__ = [
    "create_app",
    "serve",
    "MAX_UPLOAD_BYTES",
    "JobStore",
    "Job",
    "JobState",
    "demodulate",
    "AudioResult",
]
