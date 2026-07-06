from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from PIL import Image
from PIL.PngImagePlugin import PngInfo

# tEXt chunk keys embedded in every generated PNG.
#   "parameters"      — A1111 / Civitai-readable text (what external sites display).
#   "kraken_settings" — our full settings JSON, the lossless source for "reuse settings".
A1111_KEY = "parameters"
KRAKEN_KEY = "kraken_settings"


def _safe_filename_part(value: str, *, max_len: int = 96) -> str:
    text = value.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    if not text:
        return "model"
    return text[:max_len].strip("-") or "model"


def model_filename_prefix(params: dict[str, Any]) -> str:
    arch = _safe_filename_part(str(params.get("arch") or "image"), max_len=32)
    model = params.get("checkpoint") or params.get("diffusion_model") or arch
    model_stem = Path(str(model)).stem
    model_part = _safe_filename_part(model_stem)
    if model_part == arch or model_part.startswith(f"{arch}-"):
        return model_part
    return f"{arch}-{model_part}"


def build_output_stem(params: dict[str, Any], job_id: str, timestamp: str | None = None) -> str:
    ts = timestamp or time.strftime("%H%M%S")
    manual_raw = str(params.get("output_name") or "").strip()
    manual = _safe_filename_part(manual_raw, max_len=64) if manual_raw else ""
    model = model_filename_prefix(params)
    prefix = f"{manual}_{model}" if manual else model
    return f"{prefix}_{ts}-{job_id[:8]}"


def build_settings_dict(params: dict[str, Any], seed: int) -> dict[str, Any]:
    """Canonical reusable-settings shape for one generated image."""
    return {
        "arch": params.get("arch"),
        "output_name": params.get("output_name"),
        "checkpoint": params.get("checkpoint"),
        "diffusion_model": params.get("diffusion_model"),
        "vae": params.get("vae"),
        "text_encoders": params.get("text_encoders") or [],
        "loras": params.get("loras") or [],
        "embeddings": params.get("embeddings") or [],
        "prompt": params.get("prompt") or "",
        "negative": params.get("negative") or "",
        "ideogram_magic": params.get("ideogram_magic"),
        "ideogram_magic_mode": params.get("ideogram_magic_mode"),
        "width": params.get("width"),
        "height": params.get("height"),
        "steps": params.get("steps"),
        "cfg": params.get("cfg"),
        "sampler": params.get("sampler"),
        "scheduler": params.get("scheduler"),
        "clip_skip": params.get("clip_skip"),
        "count": 1,
        "seed": int(seed),
        "upscale_enabled": params.get("upscale_enabled"),
        "upscale_mode": params.get("upscale_mode"),
        "upscale_model": params.get("upscale_model"),
        "upscale_factor": params.get("upscale_factor"),
        "upscale_denoise": params.get("upscale_denoise"),
        "upscale_tile_size": params.get("upscale_tile_size"),
    }


def _model_name(s: dict[str, Any]) -> str | None:
    model = s.get("checkpoint") or s.get("diffusion_model")
    return Path(model).stem if model else None


def _lora_tags(s: dict[str, Any]) -> str:
    """Inline `<lora:name:weight>` tags — the A1111 convention Civitai recognises."""
    tags = []
    for l in s.get("loras") or []:
        if not isinstance(l, dict):
            continue
        name = Path(str(l.get("name", ""))).stem
        if not name:
            continue
        weight = l.get("model_weight")
        if weight is None:
            weight = l.get("weight", 1.0)
        tags.append(f"<lora:{name}:{weight}>")
    return " ".join(tags)


def format_a1111_parameters(s: dict[str, Any]) -> str:
    """Render settings as the A1111 `parameters` text block Civitai parses."""
    prompt = (s.get("prompt") or "").strip()
    lora_tags = _lora_tags(s)
    if lora_tags:
        prompt = f"{prompt} {lora_tags}".strip()

    lines: list[str] = [prompt]
    neg = (s.get("negative") or "").strip()
    if neg:
        lines.append(f"Negative prompt: {neg}")

    fields: list[str] = []
    if s.get("steps") is not None:
        fields.append(f"Steps: {s['steps']}")
    if s.get("sampler"):
        fields.append(f"Sampler: {s['sampler']}")
    if s.get("scheduler"):
        fields.append(f"Schedule type: {s['scheduler']}")
    if s.get("cfg") is not None:
        fields.append(f"CFG scale: {s['cfg']}")
    fields.append(f"Seed: {s['seed']}")
    if s.get("width") and s.get("height"):
        fields.append(f"Size: {s['width']}x{s['height']}")
    model = _model_name(s)
    if model:
        fields.append(f"Model: {model}")
    if s.get("clip_skip"):
        fields.append(f"Clip skip: {s['clip_skip']}")
    if s.get("arch"):
        fields.append(f"Arch: {s['arch']}")
    lines.append(", ".join(fields))
    return "\n".join(lines)


def build_pnginfo(settings: dict[str, Any]) -> PngInfo:
    info = PngInfo()
    info.add_text(A1111_KEY, format_a1111_parameters(settings))
    info.add_text(KRAKEN_KEY, json.dumps(settings, ensure_ascii=False))
    return info


def save_png_with_metadata(
    img: Image.Image, path: Path, params: dict[str, Any], seed: int
) -> dict[str, Any]:
    """Save `img` as PNG with generation metadata embedded; return the settings dict."""
    settings = build_settings_dict(params, seed)
    img.save(path, format="PNG", pnginfo=build_pnginfo(settings))
    return settings


def read_settings_from_png(path: Path) -> dict[str, Any] | None:
    """Read the embedded `kraken_settings` JSON from a generated PNG, or None."""
    try:
        with Image.open(path) as im:
            raw = (im.text or {}).get(KRAKEN_KEY)
    except Exception:
        return None
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except Exception:
        return None
    return data if isinstance(data, dict) else None
