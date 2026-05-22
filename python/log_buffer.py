"""In-memory ring buffer log handler + disk fallback.

Endpoints under /api/logs read from this. Last 2000 entries kept in RAM,
indexed monotonically so the UI can poll incrementally.
"""
from __future__ import annotations
import logging
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

MAX_LINES = 2000


class RingHandler(logging.Handler):
    def __init__(self, capacity: int) -> None:
        super().__init__()
        self._buf: deque[dict[str, Any]] = deque(maxlen=capacity)
        self._lock = threading.Lock()
        self._next_id = 0

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            with self._lock:
                self._buf.append({
                    "id":      self._next_id,
                    "ts":      record.created,
                    "level":   record.levelname,
                    "logger":  record.name,
                    "message": msg,
                })
                self._next_id += 1
        except Exception:
            self.handleError(record)

    def snapshot(self, limit: int | None = None, since_id: int | None = None) -> list[dict[str, Any]]:
        with self._lock:
            items = list(self._buf)
        if since_id is not None:
            items = [it for it in items if it["id"] > since_id]
        if limit:
            items = items[-limit:]
        return items

    def clear(self) -> None:
        with self._lock:
            self._buf.clear()


# Module-level singleton
ring = RingHandler(MAX_LINES)
ring.setLevel(logging.DEBUG)
# Multi-line tracebacks: log records include exception info via the format string.
ring.setFormatter(logging.Formatter("%(message)s"))


def install(log_dir: Path | None = None) -> RingHandler:
    """Attach ring + stdout + file handlers to the root logger. Idempotent."""
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # Avoid duplicate handlers across reloads.
    has_ring = any(isinstance(h, RingHandler) for h in root.handlers)
    if not has_ring:
        root.addHandler(ring)

    has_stream = any(isinstance(h, logging.StreamHandler) and getattr(h, "stream", None) is sys.stdout for h in root.handlers)
    if not has_stream:
        # Windows consoles often default to cp1252; reconfigure so log lines
        # with arrows / unicode don't throw UnicodeEncodeError mid-job.
        stream = sys.stdout
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
        sh = logging.StreamHandler(stream)
        sh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        root.addHandler(sh)

    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        fname = log_dir / f"sidecar-{time.strftime('%Y-%m-%d')}.log"
        has_file = any(isinstance(h, logging.FileHandler) and Path(getattr(h, "baseFilename", "")) == fname for h in root.handlers)
        if not has_file:
            fh = logging.FileHandler(fname, encoding="utf-8")
            fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
            root.addHandler(fh)

    # Quiet down access-log noise; we want app logs prominent.
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    return ring
