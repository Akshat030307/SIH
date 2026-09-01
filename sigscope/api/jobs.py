"""Background jobs in a thread with an in-memory job table (CLAUDE.md §7 "API").

§7 is explicit about the shape: "Background jobs in a thread with an in-memory job table.
No Celery, no Redis -- one more thing to fail on the demo laptop." So this is a bounded
thread pool, a dict, and a lock. Nothing is persisted; restarting the server clears the
table, which is correct for a local analyst tool and is what keeps the dependency list at
zero.

Two things this has to get right that a naive job table does not:

* **Cancellation and cleanup.** Every job owns temp files (the streamed upload, the
  rendered PNG). :meth:`JobStore.discard` removes them, and the store evicts the oldest
  finished jobs once it passes ``max_jobs`` so a long demo session cannot fill the disk.
* **Progress that means something.** §7 wants progress 0-1. The pipeline stages are
  weighted by roughly how long they actually take, so the bar does not sit at 0.1 for
  twenty seconds and then jump to done.
"""

from __future__ import annotations

import contextlib
import threading
import time
import traceback
import uuid
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

__all__ = ["JobState", "Job", "JobStore", "STAGE_PROGRESS"]


class JobState(StrEnum):
    """§7: ``queued`` / ``running`` / ``done`` / ``failed``."""

    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


# Rough share of total runtime per stage, so the progress bar tracks reality. Measured on
# a 2 s / 2 MHz four-signal capture: ingest and detection are fast, per-burst measurement
# and classification dominate.
STAGE_PROGRESS: dict[str, float] = {
    "queued": 0.0,
    "reading": 0.05,
    "conditioning": 0.15,
    "detecting": 0.30,
    "measuring": 0.45,
    "reporting": 0.90,
    "done": 1.0,
}


@dataclass
class Job:
    """One unit of work and everything the API needs to answer questions about it."""

    id: str
    kind: str  # "analyse" or "batch"
    name: str
    state: JobState = JobState.QUEUED
    progress: float = 0.0
    stage: str = "queued"
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    # results, populated by the worker
    result: Any = None
    artefacts: dict[str, Path] = field(default_factory=dict)
    temp_paths: list[Path] = field(default_factory=list)

    @property
    def runtime_s(self) -> float | None:
        if self.started_at is None:
            return None
        return (self.finished_at or time.time()) - self.started_at

    def to_dict(self) -> dict[str, Any]:
        """The §7 ``GET /api/jobs/{id}`` payload."""
        return {
            "id": self.id,
            "kind": self.kind,
            "name": self.name,
            "state": self.state.value,
            "progress": round(self.progress, 3),
            "stage": self.stage,
            "error": self.error,
            "created_at": self.created_at,
            "runtime_s": round(self.runtime_s, 3) if self.runtime_s is not None else None,
            "artefacts": sorted(self.artefacts),
        }


class JobStore:
    """Thread-safe job table with a bounded worker pool.

    ``max_workers`` defaults to 2 rather than the CPU count: the DSP underneath is already
    numpy-threaded, and letting several large captures run at once on a demo laptop makes
    every one of them slower and the progress bars misleading.
    """

    def __init__(self, *, max_workers: int = 2, max_jobs: int = 64) -> None:
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._lock = threading.Lock()
        self._semaphore = threading.Semaphore(max_workers)
        self._max_jobs = max_jobs
        self._threads: list[threading.Thread] = []

    # ------------------------------------------------------------------ table
    def create(self, kind: str, name: str, *, temp_paths: list[Path] | None = None) -> Job:
        job = Job(id=uuid.uuid4().hex[:16], kind=kind, name=name,
                  temp_paths=list(temp_paths or []))
        with self._lock:
            self._jobs[job.id] = job
            self._evict_locked()
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        with self._lock:
            return list(self._jobs.values())

    def discard(self, job_id: str) -> bool:
        """Forget a job and delete everything it owns on disk."""
        with self._lock:
            job = self._jobs.pop(job_id, None)
        if job is None:
            return False
        _cleanup(job)
        return True

    def _evict_locked(self) -> None:
        """Drop the oldest finished jobs once the table is over budget."""
        while len(self._jobs) > self._max_jobs:
            for job_id, job in list(self._jobs.items()):
                if job.state in (JobState.DONE, JobState.FAILED):
                    self._jobs.pop(job_id, None)
                    _cleanup(job)
                    break
            else:
                return  # nothing finished yet; let the table run over rather than kill work

    # ------------------------------------------------------------------ running
    def submit(self, job: Job, work: Callable[[Job], Any]) -> Job:
        """Run ``work(job)`` on a daemon thread, recording state transitions."""

        def runner() -> None:
            with self._semaphore:
                job.state = JobState.RUNNING
                job.started_at = time.time()
                try:
                    job.result = work(job)
                    job.state = JobState.DONE
                    job.progress = 1.0
                    job.stage = "done"
                except Exception as exc:  # noqa: BLE001 -- a failed job is data, not a crash
                    job.state = JobState.FAILED
                    job.error = f"{type(exc).__name__}: {exc}"
                    job.stage = "failed"
                    # keep the traceback server-side for debugging; the API returns the
                    # one-line message, since §7's error states are for an analyst
                    job.result = {"traceback": traceback.format_exc()}
                finally:
                    job.finished_at = time.time()

        thread = threading.Thread(target=runner, name=f"sigscope-{job.id}", daemon=True)
        self._threads.append(thread)
        thread.start()
        return job

    def advance(self, job: Job, stage: str, fraction: float | None = None) -> None:
        """Move a job to ``stage``, or to an explicit progress fraction."""
        job.stage = stage
        job.progress = (
            float(fraction) if fraction is not None else STAGE_PROGRESS.get(stage, job.progress)
        )

    def shutdown(self, timeout: float = 5.0) -> None:
        """Wait briefly for running jobs, then drop every temp file we own."""
        for thread in list(self._threads):
            thread.join(timeout=timeout)
        for job in self.list():
            _cleanup(job)


def _cleanup(job: Job) -> None:
    """Delete a job's temp files. Never raises -- cleanup failure must not break a request."""
    for path in list(job.temp_paths) + list(job.artefacts.values()):
        with contextlib.suppress(OSError):
            Path(path).unlink(missing_ok=True)
