"""Standalone end-to-end smoke test for the Krea 2 pipeline.

Runs the real wrapper run() with a fake job so we exercise encode -> stream-load
-> sample -> decode -> save without the FastAPI/job-manager stack. Usage:

    venv/Scripts/python.exe scripts/krea2_smoke.py <raw|turbo> [steps]
"""
from __future__ import annotations
import logging
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")


class _Progress:
    step = 0
    total_steps = 0
    image_index = 0
    total_images = 0


class FakeJob:
    def __init__(self, params: dict):
        self.id = "smoke-test-0000"
        self.params = params
        self.cancel = threading.Event()
        self.progress = _Progress()

    def emit(self, msg: dict):
        if msg.get("type") == "progress":
            print(f"  step {msg['step']}/{msg['total_steps']}", flush=True)
        elif msg.get("type") == "image":
            print(f"  image {msg['image_index']} -> {msg['filename']}", flush=True)


def main():
    variant = (sys.argv[1] if len(sys.argv) > 1 else "turbo").lower()
    model = "Krea-2-Turbo" if variant == "turbo" else "Krea-2-Raw"
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else (8 if variant == "turbo" else 28)

    from pipelines import krea2

    params = {
        "arch": "krea2",
        "diffusion_model": model,
        "prompt": "a red fox sitting in fresh snow at golden hour, photorealistic, sharp detail",
        "negative": "",
        "width": 1024,
        "height": 1024,
        "steps": steps,
        "cfg": 4.5,
        "count": 1,
        "seed": 1234,
    }
    job = FakeJob(params)
    t0 = time.time()
    result = krea2.run(job)
    dt = time.time() - t0
    print(f"\nDONE in {dt:.1f}s")
    print("result:", result)
    for o in result["outputs"]:
        print("saved:", o["path"])


if __name__ == "__main__":
    main()
