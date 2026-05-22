"""Full FLUX pipeline load smoke test (no inference)."""
from __future__ import annotations
import gc
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipelines import flux

def main() -> int:
    flux.unload()
    gc.collect()
    t0 = time.time()
    pipe = flux._ensure_pipeline(
        "Flux 1D FP16/fluxmania_kreamania.safetensors",
        "FLUX1/fluxVaeSft_aeSft.sft",
        "clip_l.safetensors",
        "t5/t5xxl_fp16.safetensors",
    )
    print(f"pipeline OK in {time.time() - t0:.1f}s")
    pe, pooled, ids = pipe.encode_prompt("test prompt", device="cpu")
    print(f"prompt_embeds={pe.dtype} pooled={pooled.dtype} transformer={pipe.transformer.dtype}")
    flux.unload()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
