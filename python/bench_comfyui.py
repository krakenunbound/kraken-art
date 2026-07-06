"""Measure ComfyUI's per-step time on the SAME workflow we benchmark against.

Posts the Flux Basic Workflow to ComfyUI's /prompt API, attaches to /ws for
progress events, and prints per-step wall-clock + summary stats so we can
compare apples-to-apples with `python/bench_step_times.sh`.

ComfyUI must be running on http://127.0.0.1:8188.

Usage:
    python bench_comfyui.py [seed]
"""
from __future__ import annotations
import json
import sys
import time
import uuid

import requests
import websockets.sync.client as ws_client


COMFY = "http://127.0.0.1:8188"
COMFY_WS = "ws://127.0.0.1:8188/ws"


def build_prompt(seed: int) -> dict:
    """Convert the editor-format workflow into ComfyUI's API format.

    Same FLUX-dev FP32 + t5xxl_fp16 + clip_l + fluxVaeSft VAE that Kraken Art
    benches against. 1024×1024, 28 steps, CFG 1.0, euler/simple, batch 1.
    """
    prompt_text = (
        "A beautiful cyborg figure sitting at night, wearing black turtle "
        "neck sweater, resting chin in palm of hand, relaxed pose. She has a "
        "cybernetic face. Extreme close-up facial shot, tilted angled face "
        "pose. Tilted head. Half of her face is covered with her robotic "
        "hand. The face is redacted out entirely using a hollowed hole "
        "effect. Resembling a ripped dimensional rift. A flower meadow is "
        "displayed in the rift and colorful flowers are spilling out of the "
        "rift. The flowers illuminate the scene in warm colorful hues. Bold "
        "vibrant colors and photorealistic background. Extremely detailed. "
        "Professional photography. bokeh lighting effects, Stunning visuals, "
        "high-quality"
    )
    return {
        "8": {  # UNETLoader
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": "Flux 1D FP32\\flux_dev.safetensors",
                "weight_dtype": "default",
            },
        },
        "10": {  # VAELoader
            "class_type": "VAELoader",
            "inputs": {"vae_name": "FLUX1\\fluxVaeSft_aeSft.sft"},
        },
        "9": {  # DualCLIPLoader
            "class_type": "DualCLIPLoader",
            "inputs": {
                "clip_name1": "clip_l.safetensors",
                "clip_name2": "t5\\t5xxl_fp16.safetensors",
                "type": "flux",
                "device": "default",
            },
        },
        "3": {  # Positive prompt
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["9", 0], "text": prompt_text},
        },
        "4": {  # Negative (empty for FLUX — uses CFG=1 anyway)
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["9", 0], "text": ""},
        },
        "5": {  # EmptyLatentImage
            "class_type": "EmptyLatentImage",
            "inputs": {"width": 1024, "height": 1024, "batch_size": 1},
        },
        "2": {  # KSampler
            "class_type": "KSampler",
            "inputs": {
                "model": ["8", 0],
                "positive": ["3", 0],
                "negative": ["4", 0],
                "latent_image": ["5", 0],
                "seed": seed,
                "steps": 28,
                "cfg": 1.0,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1.0,
            },
        },
        "6": {  # VAEDecode
            "class_type": "VAEDecode",
            "inputs": {"samples": ["2", 0], "vae": ["10", 0]},
        },
        "7": {  # PreviewImage
            "class_type": "PreviewImage",
            "inputs": {"images": ["6", 0]},
        },
    }


def main() -> None:
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 999
    client_id = str(uuid.uuid4())

    # Connect WS first so we don't miss early events.
    ws = ws_client.connect(f"{COMFY_WS}?clientId={client_id}")

    payload = {"prompt": build_prompt(seed), "client_id": client_id}
    r = requests.post(f"{COMFY}/prompt", json=payload, timeout=30)
    r.raise_for_status()
    prompt_id = r.json()["prompt_id"]
    print(f"seed={seed}  prompt_id={prompt_id}")

    submit_ms = int(time.time() * 1000)
    sampling_start_ms: int | None = None
    last_step_ms: int | None = None
    step_durations_ms: list[int] = []
    last_step: int = 0
    total_steps: int = 28

    # Drain WS until 'executing' returns with node=None (which signals "done").
    while True:
        msg = ws.recv()
        if isinstance(msg, bytes):
            continue  # binary previews; skip
        evt = json.loads(msg)
        kind = evt.get("type")
        data = evt.get("data", {}) or {}
        if kind == "progress":
            # Per-step events from KSampler. data = {value, max, prompt_id, node}
            now_ms = int(time.time() * 1000)
            step = int(data.get("value", 0))
            total_steps = int(data.get("max", total_steps))
            if step <= last_step:
                continue
            if sampling_start_ms is None:
                sampling_start_ms = now_ms
            dur = (now_ms - last_step_ms) if last_step_ms else 0
            step_durations_ms.append(dur)
            elapsed = now_ms - submit_ms
            print(f"  step {step:2d}/{total_steps} @ t+{elapsed:5d}ms  step_dur={dur:5d}ms")
            last_step_ms = now_ms
            last_step = step
        elif kind == "executing" and data.get("node") is None and data.get("prompt_id") == prompt_id:
            break
        elif kind == "execution_error" and data.get("prompt_id") == prompt_id:
            print("EXECUTION ERROR:")
            print(json.dumps(data, indent=2)[:2000])
            ws.close()
            sys.exit(1)

    total_ms = int(time.time() * 1000) - submit_ms
    ws.close()

    # Drop the first 3 step deltas (warmup / first-step bias).
    durs = step_durations_ms[1:]  # the first 'dur' is 0 (no prior step)
    steady = durs[3:] if len(durs) > 5 else durs
    if not steady:
        print("\nno per-step data captured")
        return

    sorted_d = sorted(steady)
    median = sorted_d[len(sorted_d) // 2]
    mean = sum(steady) / len(steady)
    print()
    print("--- ComfyUI per-step summary (ms) ---")
    print(f"  total gen wall-clock : {total_ms} ms")
    print(f"  all steps            : {durs}")
    print(f"  steady (>3)          : {steady}")
    print(f"  median               : {median} ms")
    print(f"  mean                 : {mean:.0f} ms")
    print(f"  min                  : {min(steady)} ms")


if __name__ == "__main__":
    main()
