"""Kraken Art sidecar config — paths, ports, model-category mapping."""
from __future__ import annotations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_ROOT = ROOT / "models"
OUTPUTS_ROOT = ROOT / "outputs"

SIDECAR_HOST = "127.0.0.1"
SIDECAR_PORT = 7780

MODEL_EXTS = {".safetensors", ".sft", ".ckpt", ".pt", ".pth", ".bin", ".gguf"}

# Display name -> folder name(s) under MODELS_ROOT.
# Multiple folders fold into one category (e.g. text_encoders + clip).
MODEL_CATEGORIES: dict[str, list[str]] = {
    "checkpoints":      ["checkpoints"],
    "diffusion_models": ["diffusion_models", "unet"],
    "vae":              ["vae"],
    "text_encoders":    ["text_encoders", "clip"],
    "clip_vision":      ["clip_vision"],
    "loras":            ["loras"],
    "controlnet":       ["controlnet"],
    "upscale_models":   ["upscale_models", "latent_upscale_models"],
    "embeddings":       ["embeddings"],
    "style_models":     ["style_models"],
    "ip_adapter":       ["ipadapter", "ip_adapter"],
}
