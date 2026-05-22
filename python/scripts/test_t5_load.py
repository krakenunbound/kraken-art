"""Minimal T5-XXL load smoke test — run standalone to bisect native crashes."""
from __future__ import annotations
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from config import MODELS_ROOT

def main() -> int:
    path = MODELS_ROOT / "text_encoders" / "t5" / "t5xxl_fp16.safetensors"
    if not path.exists():
        print("missing", path)
        return 1
    print("loading", path, f"({path.stat().st_size / 1024**3:.1f} GB)")
    from pipelines.flux import _load_t5xxl, _dtype
    dtype = _dtype()
    t0 = __import__("time").time()
    model = _load_t5xxl(path, dtype)
    print(f"OK in {__import__('time').time() - t0:.1f}s, dtype={next(model.parameters()).dtype}")
    del model
    gc.collect()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
