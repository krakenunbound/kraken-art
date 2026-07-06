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


def _model_scale(model: Any) -> int:
    try:
        return max(1, int(getattr(model, "scale", 1) or 1))
    except Exception:
        return 1


def _infer_array(model: Any, image: Image.Image) -> np.ndarray:
    arr = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(_device())
    with torch.no_grad():
        out = model(t)
    return out.squeeze(0).clamp(0, 1).permute(1, 2, 0).cpu().numpy()


def _resize_to_factor(arr: np.ndarray, source_size: tuple[int, int], factor: float) -> Image.Image:
    img = Image.fromarray((arr * 255).round().clip(0, 255).astype(np.uint8))
    target_w = max(1, int(round(source_size[0] * factor)))
    target_h = max(1, int(round(source_size[1] * factor)))
    if img.size != (target_w, target_h):
        img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
    return img


def upscale(image: Image.Image, model_name: str, factor: float = 2.0) -> Image.Image:
    """Run `image` through the named spandrel-loadable upscaler.

    The model has an intrinsic scale (e.g. 4x). If `factor` differs, we resize
    the result to match exactly so the user gets the dimensions they asked for.
    """
    path = _resolve(model_name)
    model = _load(path)
    return _resize_to_factor(_infer_array(model, image), image.size, factor)


def _starts(length: int, tile_size: int, overlap: int) -> list[int]:
    if length <= tile_size:
        return [0]
    step = max(1, tile_size - overlap)
    values: list[int] = []
    pos = 0
    while True:
        values.append(pos)
        if pos + tile_size >= length:
            break
        pos = min(pos + step, length - tile_size)
    return values


def _tile_mask(width: int, height: int, overlap: int, edges: tuple[bool, bool, bool, bool]) -> np.ndarray:
    left_edge, right_edge, top_edge, bottom_edge = edges
    wx = np.ones(width, dtype=np.float32)
    wy = np.ones(height, dtype=np.float32)
    ramp_x = min(max(1, overlap), max(1, width // 2))
    ramp_y = min(max(1, overlap), max(1, height // 2))
    if not left_edge:
        wx[:ramp_x] *= np.linspace(0.05, 1.0, ramp_x, dtype=np.float32)
    if not right_edge:
        wx[-ramp_x:] *= np.linspace(1.0, 0.05, ramp_x, dtype=np.float32)
    if not top_edge:
        wy[:ramp_y] *= np.linspace(0.05, 1.0, ramp_y, dtype=np.float32)
    if not bottom_edge:
        wy[-ramp_y:] *= np.linspace(1.0, 0.05, ramp_y, dtype=np.float32)
    return (wy[:, None] * wx[None, :])[:, :, None]


def upscale_tiled(
    image: Image.Image,
    model_name: str,
    factor: float = 2.0,
    tile_size: int | None = 768,
    overlap: int = 64,
) -> Image.Image:
    """Upscale with overlapping input tiles, then resize to the requested factor."""
    tile_size = int(tile_size or 0)
    if tile_size <= 0 or max(image.size) <= tile_size:
        return upscale(image, model_name, factor)

    path = _resolve(model_name)
    model = _load(path)
    scale = _model_scale(model)
    src = image.convert("RGB")
    w, h = src.size
    intrinsic_w = w * scale
    intrinsic_h = h * scale
    acc = np.zeros((intrinsic_h, intrinsic_w, 3), dtype=np.float32)
    weights = np.zeros((intrinsic_h, intrinsic_w, 1), dtype=np.float32)
    overlap = max(0, min(int(overlap), tile_size // 2))

    for y0 in _starts(h, tile_size, overlap):
        for x0 in _starts(w, tile_size, overlap):
            x1 = min(w, x0 + tile_size)
            y1 = min(h, y0 + tile_size)
            tile = src.crop((x0, y0, x1, y1))
            out = _infer_array(model, tile)
            ox0 = x0 * scale
            oy0 = y0 * scale
            oh, ow = out.shape[:2]
            mask = _tile_mask(
                ow,
                oh,
                overlap * scale,
                (x0 == 0, x1 >= w, y0 == 0, y1 >= h),
            )
            acc[oy0:oy0 + oh, ox0:ox0 + ow] += out * mask
            weights[oy0:oy0 + oh, ox0:ox0 + ow] += mask

    arr = acc / np.maximum(weights, 1e-6)
    return _resize_to_factor(arr.clip(0, 1), image.size, factor)
