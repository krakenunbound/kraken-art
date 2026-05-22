"""ESRGAN / Real-ESRGAN / SwinIR / etc. upscaling via spandrel.

spandrel auto-detects the model architecture from the safetensors/pth keys so
any of the user's upscale_models/*.pth files work. We cache loaded models by
path; unload() drops them.
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from config import MODELS_ROOT

log = logging.getLogger("kraken.upscale")

_cache: dict[str, Any] = {}


def unload() -> dict:
    """Drop cached upscale models from VRAM."""
    n = len(_cache)
    _cache.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"upscalers_freed": n}


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve(name: str) -> Path:
    folder = MODELS_ROOT / "upscale_models"
    p = folder / name
    if p.exists():
        return p
    for f in folder.rglob(name):
        return f
    raise FileNotFoundError(f"upscale model not found: {name}")


def _load(path: Path):
    key = str(path)
    if key in _cache:
        return _cache[key]
    from spandrel import ModelLoader
    log.info("loading upscaler %s", path)
    model = ModelLoader().load_from_file(str(path))
    model = model.to(_device()).eval()
    _cache[key] = model
    log.info("  scale=%dx in_channels=%s", model.scale, model.input_channels)
    return model


def upscale(image: Image.Image, model_name: str, factor: float = 2.0) -> Image.Image:
    """Run `image` through the named spandrel-loadable upscaler.

    The model has an intrinsic scale (e.g. 4x). If `factor` differs, we resize
    the result to match exactly so the user gets the dimensions they asked for.
    """
    path = _resolve(model_name)
    model = _load(path)

    arr = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(_device())

    with torch.no_grad():
        out = model(t)

    target_w = max(1, int(round(image.width  * factor)))
    target_h = max(1, int(round(image.height * factor)))
    if out.shape[-1] != target_w or out.shape[-2] != target_h:
        out = torch.nn.functional.interpolate(
            out, size=(target_h, target_w), mode="bilinear", align_corners=False
        )

    arr = (out.squeeze(0).clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)
