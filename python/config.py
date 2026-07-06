"""Kraken Art sidecar config — paths, ports, model-category mapping."""
from __future__ import annotations
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_ROOT = ROOT / "models"
OUTPUTS_ROOT = ROOT / "outputs"

SIDECAR_HOST = "127.0.0.1"
SIDECAR_PORT = 7780

MODEL_EXTS = {".safetensors", ".sft", ".ckpt", ".pt", ".pth", ".bin", ".gguf"}

# External audio project root (the finished Kraken_Audio product).
# We do NOT copy the heavy audio models/venvs here. We point at the existing installation.
DEFAULT_AUDIO_ACE_ROOT = ROOT.parent / "Kraken_Audio" / "ACE-Step-1.5"
AUDIO_ACE_ROOT = Path(os.environ.get("KRAKEN_AUDIO_ACE_ROOT", str(DEFAULT_AUDIO_ACE_ROOT)))
DEFAULT_AUDIO_TTS_ROOT = ROOT.parent / "Kraken_Audio" / "VoxtralTTS"
AUDIO_TTS_ROOT = Path(os.environ.get("KRAKEN_AUDIO_TTS_ROOT", str(DEFAULT_AUDIO_TTS_ROOT)))
DEFAULT_AUDIO_PROVIDERS_ROOT = ROOT.parent / "Kraken_Audio" / "providers"
AUDIO_PROVIDERS_ROOT = Path(os.environ.get("KRAKEN_AUDIO_PROVIDERS_ROOT", str(DEFAULT_AUDIO_PROVIDERS_ROOT)))
DEFAULT_AUDIO_MODELS_ROOT = ROOT.parent / "Kraken_Audio" / "models"
AUDIO_MODELS_ROOT = Path(os.environ.get("KRAKEN_AUDIO_MODELS_ROOT", str(DEFAULT_AUDIO_MODELS_ROOT)))

# Display name -> folder name(s) under MODELS_ROOT.
# Multiple folders fold into one category (e.g. text_encoders + clip).
# "audio" is special: we also scan the external ACE-Step-1.5/checkpoints structure.
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
    "audio":            ["audio", "audio_models"],   # user can symlink or copy small files here; heavy DiT/LM live in AUDIO_ACE_ROOT
}
