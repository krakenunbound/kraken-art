"""Full FLUX generate smoke test (load + 4-step inference)."""
from __future__ import annotations
import gc
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from pipelines import flux

def main() -> int:
    flux.unload()
    gc.collect()
    pipe = flux._ensure_pipeline(
        "Flux 1D FP16/fluxmania_kreamania.safetensors",
        "FLUX1/fluxVaeSft_aeSft.sft",
        "clip_l.safetensors",
        "t5/t5xxl_fp16.safetensors",
    )
    print("loaded; running 4 steps @ 512x512...")
    t0 = time.time()
    exec_dev = pipe._execution_device
    gen_device = exec_dev.type if isinstance(exec_dev, torch.device) else str(exec_dev)
    gen = torch.Generator(device=gen_device).manual_seed(42)
    out = pipe(
        prompt="a single bright red apple on a wooden table, studio lighting",
        width=512,
        height=512,
        num_inference_steps=4,
        guidance_scale=1.0,
        generator=gen,
    )
    path = Path(__file__).resolve().parent.parent.parent / "outputs" / "test-flux-smoke.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.images[0].save(path)
    print(f"OK in {time.time() - t0:.1f}s -> {path}")
    flux.unload()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
