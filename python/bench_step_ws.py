"""Per-step bench using the kraken sidecar's WebSocket progress feed.

Replaces `bench_step_times.sh`'s HTTP polling (which hits FastAPI's GIL every
250 ms and may contend with the diffusion thread). Subscribes to /ws/jobs/{id}
and records timestamps on each push event. Matches the methodology used in
`bench_comfyui.py` so the two are apples-to-apples.

Usage: python bench_step_ws.py [seed] [steps] [diffusion_model]

  diffusion_model (optional): overrides the transformer checkpoint. Accepts a
  path relative to models/diffusion_models OR an absolute path (so you can point
  it at an all-in-one FP8 checkpoint under models/checkpoints for the
  resident-vs-streaming A/B). VAE + text encoders stay fixed so the ONLY
  variable is the transformer.

  FP32 (streaming) baseline:
    python bench_step_ws.py 10001 28
  FP8 (fully resident) comparison:
    python bench_step_ws.py 10001 28 "F:/Kraken Art/models/checkpoints/Flux1DFP8/flux1CompactCLIPAnd_Flux1DevFp8.safetensors"

IMPORTANT:
  The Kraken sidecar MUST be running on port 7780 before you run this script.
  The script will now give a clear error message + instructions if it isn't.
"""
from __future__ import annotations
import json
import sys
import time

import requests
import websockets.sync.client as ws_client


KRAKEN = "http://127.0.0.1:7780"
KRAKEN_WS = "ws://127.0.0.1:7780/ws/jobs"


def main() -> None:
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 10001
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 28
    diffusion_model = sys.argv[3] if len(sys.argv) > 3 else "Flux 1D FP32/flux_dev.safetensors"

    # Quick health check first so we give a friendly error instead of a traceback
    try:
        health = requests.get(f"{KRAKEN}/health", timeout=5)
        if health.status_code != 200:
            raise requests.exceptions.ConnectionError()
    except requests.exceptions.ConnectionError:
        print("\n" + "="*70)
        print("ERROR: Cannot connect to the Kraken sidecar on port 7780.")
        print("="*70)
        print("\nThe sidecar is not running.")
        print("\nTo fix this, do ONE of the following:")
        print("\n  Option A (Recommended for clean benchmarks):")
        print("    .\\python\\venv\\Scripts\\python.exe python\\main.py")
        print("\n  Option B (Full app):")
        print("    npm run tauri dev")
        print("\nWait until you see the sidecar is up (it will say it's listening on 7780),")
        print("then run this benchmark script again in a separate terminal.")
        print("\n" + "="*70 + "\n")
        sys.exit(1)

    payload = {
        "arch": "flux1",
        "diffusion_model": diffusion_model,
        "vae": "FLUX1/fluxVaeSft_aeSft.sft",
        "text_encoders": ["clip_l.safetensors", "t5/t5xxl_fp16.safetensors"],
        "prompt": "a kraken in deep sea bioluminescence",
        "width": 1024, "height": 1024,
        "steps": steps, "cfg": 1.0,
        "sampler": "euler", "scheduler": "normal",
        "count": 1, "seed": seed,
    }

    try:
        r = requests.post(f"{KRAKEN}/api/generate", json=payload, timeout=30)
        r.raise_for_status()
    except requests.exceptions.ConnectionError:
        print("\n" + "="*70)
        print("ERROR: Cannot connect to the Kraken sidecar on port 7780.")
        print("="*70)
        print("\nThe sidecar is not running.")
        print("\nTo fix this, do ONE of the following:")
        print("\n  Option A (Recommended for clean benchmarks):")
        print("    .\\python\\venv\\Scripts\\python.exe python\\main.py")
        print("\n  Option B (Full app):")
        print("    npm run tauri dev")
        print("\nWait until you see the sidecar is up (it will say it's listening on 7780),")
        print("then run this benchmark script again in a separate terminal.")
        print("\n" + "="*70 + "\n")
        sys.exit(1)

    jid = r.json()["job_id"]
    print(f"model={diffusion_model}")
    print(f"seed={seed} steps={steps} job={jid}")

    submit_ms = int(time.time() * 1000)
    ws = ws_client.connect(f"{KRAKEN_WS}/{jid}")

    last_step = 0
    last_step_ms: int | None = None
    durs_ms: list[int] = []

    while True:
        try:
            raw = ws.recv(timeout=300)
        except TimeoutError:
            print("WS timeout"); break
        if isinstance(raw, bytes):
            continue
        try:
            evt = json.loads(raw)
        except json.JSONDecodeError:
            continue
        typ = evt.get("type")
        if typ == "progress":
            step = int(evt.get("step", 0))
            if step <= last_step:
                continue
            now_ms = int(time.time() * 1000)
            dur = (now_ms - last_step_ms) if last_step_ms else 0
            durs_ms.append(dur)
            elapsed = now_ms - submit_ms
            print(f"  step {step:2d}/{evt.get('total_steps', steps)} @ t+{elapsed:5d}ms  step_dur={dur:5d}ms")
            last_step = step
            last_step_ms = now_ms
        elif typ == "status":
            st = evt.get("status", "")
            if st in ("succeeded", "failed", "cancelled"):
                print(f"final status: {st}  error: {evt.get('error') or 'none'}")
                break
        elif typ == "snapshot":
            if evt.get("status") in ("succeeded", "failed", "cancelled"):
                print(f"snapshot terminal: {evt.get('status')}")
                break

    total_ms = int(time.time() * 1000) - submit_ms
    ws.close()

    # Skip the first 3 dur samples (warmup + measurement bias).
    durs = durs_ms[1:]
    steady = durs[3:] if len(durs) > 5 else durs
    if not steady:
        print("no per-step data captured")
        return

    sorted_d = sorted(steady)
    median = sorted_d[len(sorted_d) // 2]
    mean = sum(steady) / len(steady)
    print()
    print("--- WS-probe per-step summary (ms) ---")
    print(f"  total wall clock: {total_ms} ms")
    print(f"  all steps       : {durs}")
    print(f"  steady (>3)     : {steady}")
    print(f"  median          : {median} ms")
    print(f"  mean            : {mean:.0f} ms")
    print(f"  min             : {min(steady)} ms")


if __name__ == "__main__":
    main()


# === FLUX warm-start investigation notes (2026-05-28) ===
# When running this for the current performance gap analysis:
#
# 1. Start the sidecar fresh (or after a full app restart).
# 2. Run this script twice in a row with the same seed for cold vs warm.
# 3. During/after the run, capture:
#    - The exact line from the sidecar log containing "streaming partition"
#      (it prints number of blocks on CPU vs GPU and the budget used).
#    - `nvidia-smi dmon -s pucm` or MSI Afterburner during the generation.
#    - Whether "DynamicVRAM" or similar messages appear.
#
# This bench already uses WebSocket (matching ComfyUI's push model) so
# the per-step numbers are directly comparable to the ComfyUI reference
# the user provided from D:\AI_Art\ComfyUI.
