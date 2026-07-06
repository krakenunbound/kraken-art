"""Audio generation support for Kraken Art.

This package provides a thin client and orchestration layer for the
ACE-Step (and future Stable Audio / LuxTTS) engines that live in the
sibling "Kraken_Audio" installation.

Design goals:
- Never import the heavy audio training/inference code into the main image venv
  (transformers version conflict).
- Talk to a running ACE-Step API (started by the existing Kraken_Audio launcher
  or manually) over HTTP.
- When generating music, optionally first produce album art by calling back
  into Kraken Art's own image generation pipelines (the key "redirect ComfyUI
  calls to internal" requirement).
"""

from .ace_client import AceClient, get_default_ace_client

__all__ = ["AceClient", "get_default_ace_client"]
