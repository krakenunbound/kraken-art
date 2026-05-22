"""In-process job manager.

One worker thread drains a single FIFO queue. Single-job-at-a-time keeps VRAM
predictable and lets us swap pipelines/models cleanly between jobs. Multi-queue
fan-out can come later if VRAM headroom allows.
"""
from __future__ import annotations
import asyncio
import logging
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from queue import Queue
from typing import Any, Callable

log = logging.getLogger("kraken.jobs")

JobFn = Callable[["Job"], dict]  # returns the final result payload


@dataclass
class JobProgress:
    step: int = 0
    total_steps: int = 0
    image_index: int = 0
    total_images: int = 0
    message: str = ""
    preview_b64: str | None = None


@dataclass
class Job:
    id: str
    kind: str
    params: dict
    status: str = "queued"        # queued | running | succeeded | failed | cancelled
    progress: JobProgress = field(default_factory=JobProgress)
    result: dict | None = None
    error: str | None = None
    cancel: threading.Event = field(default_factory=threading.Event)
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    # asyncio.Queue cannot live across threads; subscribers register loops + queues.
    _subs: list[tuple[asyncio.AbstractEventLoop, asyncio.Queue]] = field(default_factory=list)

    def emit(self, event: dict) -> None:
        """Push a JSON event to every WebSocket subscriber. Safe to call from worker thread."""
        for loop, q in list(self._subs):
            try:
                loop.call_soon_threadsafe(q.put_nowait, event)
            except Exception:
                pass

    def snapshot(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "progress": {
                "step": self.progress.step,
                "total_steps": self.progress.total_steps,
                "image_index": self.progress.image_index,
                "total_images": self.progress.total_images,
                "message": self.progress.message,
            },
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


class JobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._queue: Queue[tuple[Job, JobFn]] = Queue()
        self._lock = threading.Lock()
        self._worker = threading.Thread(target=self._loop, daemon=True, name="kraken-jobs")
        self._worker.start()

    # ---------- public API ----------

    def submit(self, kind: str, params: dict, fn: JobFn) -> Job:
        job = Job(id=uuid.uuid4().hex, kind=kind, params=params)
        with self._lock:
            self._jobs[job.id] = job
        self._queue.put((job, fn))
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        with self._lock:
            return list(self._jobs.values())

    def cancel(self, job_id: str) -> bool:
        job = self.get(job_id)
        if not job or job.status in ("succeeded", "failed", "cancelled"):
            return False
        job.cancel.set()
        if job.status == "queued":
            job.status = "cancelled"
            job.finished_at = time.time()
            job.emit({"type": "status", "status": "cancelled"})
        return True

    def subscribe(self, job_id: str) -> tuple[asyncio.Queue, Callable[[], None]] | None:
        job = self.get(job_id)
        if not job:
            return None
        loop = asyncio.get_event_loop()
        q: asyncio.Queue = asyncio.Queue(maxsize=256)
        entry = (loop, q)
        job._subs.append(entry)

        def unsubscribe() -> None:
            try:
                job._subs.remove(entry)
            except ValueError:
                pass

        # Replay the current snapshot so late subscribers see current state.
        q.put_nowait({"type": "snapshot", **job.snapshot()})
        return q, unsubscribe

    # ---------- worker ----------

    def _loop(self) -> None:
        while True:
            job, fn = self._queue.get()
            if job.cancel.is_set():
                continue
            job.status = "running"
            job.started_at = time.time()
            job.emit({"type": "status", "status": "running"})
            log.info("job %s (%s) running — params=%s", job.id, job.kind, {k: v for k, v in job.params.items() if k != "negative"})
            try:
                job.result = fn(job)
                if job.cancel.is_set():
                    job.status = "cancelled"
                    job.emit({"type": "status", "status": "cancelled"})
                    log.info("job %s cancelled", job.id)
                else:
                    job.status = "succeeded"
                    job.emit({"type": "status", "status": "succeeded", "result": job.result})
                    log.info("job %s succeeded in %.1fs", job.id, time.time() - job.started_at)
            except Exception as e:
                tb = traceback.format_exc()
                job.error = f"{type(e).__name__}: {e}"
                job.status = "failed"
                job.emit({"type": "status", "status": "failed", "error": job.error})
                log.error("job %s FAILED: %s\n%s", job.id, job.error, tb)
            finally:
                job.finished_at = time.time()


# Module-level singleton.
manager = JobManager()
