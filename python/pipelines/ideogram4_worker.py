"""Standalone Ideogram worker process.

The sidecar launches this as a child process and relays JSON-lines events back
to the existing job WebSocket. If CUDA or Triton hard-crashes, only this child
dies; the FastAPI sidecar stays alive.
"""
from __future__ import annotations

import json
import logging
import faulthandler
import sys
import threading
import traceback
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
PY_ROOT = ROOT / "python"
if str(PY_ROOT) not in sys.path:
    sys.path.insert(0, str(PY_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
faulthandler.enable(all_threads=True)


class WorkerJob:
    def __init__(self, job_id: str, params: dict):
        self.id = job_id
        self.params = params
        self.cancel = threading.Event()
        self.progress = SimpleNamespace(
            step=0,
            total_steps=0,
            image_index=0,
            total_images=0,
            message="",
            preview_b64=None,
        )

    def emit(self, event: dict) -> None:
        print(json.dumps(event, ensure_ascii=False, separators=(",", ":")), flush=True)


def main() -> int:
    if len(sys.argv) == 2 and sys.argv[1] == "--server":
        from pipelines import ideogram4

        for line in sys.stdin:
            if not line.strip():
                continue
            try:
                spec = json.loads(line)
                job = WorkerJob(str(spec["id"]), dict(spec["params"]))
                result = ideogram4._run_inprocess(job)
                print(json.dumps({"type": "done", "result": result}, ensure_ascii=False, separators=(",", ":")), flush=True)
            except Exception as e:
                print(
                    json.dumps(
                        {
                            "type": "error",
                            "error": f"{type(e).__name__}: {e}",
                            "traceback": traceback.format_exc(),
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    flush=True,
                )
        return 0

    if len(sys.argv) != 2:
        print(json.dumps({"type": "error", "error": "Usage: ideogram4_worker.py <job.json>"}), flush=True)
        return 2
    spec_path = Path(sys.argv[1])
    try:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        job = WorkerJob(str(spec["id"]), dict(spec["params"]))
        from pipelines import ideogram4

        result = ideogram4._run_inprocess(job)
        print(json.dumps({"type": "done", "result": result}, ensure_ascii=False, separators=(",", ":")), flush=True)
        return 0
    except Exception as e:
        print(
            json.dumps(
                {
                    "type": "error",
                    "error": f"{type(e).__name__}: {e}",
                    "traceback": traceback.format_exc(),
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            flush=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
