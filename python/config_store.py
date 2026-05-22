"""User settings persisted to F:\\Kraken Art\\config\\settings.json.

Deep-merged with sensible defaults on every load so adding a new key in code
doesn't require deleting the user's file.
"""
from __future__ import annotations
import json
import logging
import threading
from copy import deepcopy
from pathlib import Path
from typing import Any

from config import ROOT

log = logging.getLogger("kraken.settings")

SETTINGS_PATH = ROOT / "config" / "settings.json"

DEFAULTS: dict[str, Any] = {
    "civitai": {
        "api_token": "",
        "nsfw_visible": False,
        "default_sort": "Most Downloaded",
    },
    "downloads": {
        # Per-Civitai-type folder overrides; missing/None = use the built-in mapping
        "category_paths": {},
    },
    "performance": {
        # Pre-cast FP8 weights to bf16 at load (uses ~2× transformer VRAM,
        # ~30% faster per step). Mirrors ComfyUI's behaviour when VRAM allows.
        # `auto` = enable when measured free VRAM > model size + safety buffer.
        # `on` = always try (may OOM on tight VRAM cards).
        # `off` = always use the per-forward cast path (safest, slowest).
        "flux_fast_inference": "auto",
        # Headroom required above the bf16 model size before we'll pre-cast in
        # `auto` mode. 2 GB covers activations + allocator fragmentation on
        # most setups.
        "flux_fast_inference_buffer_gb": 2.0,
        # torch.compile the FLUX transformer's repeated blocks. Measured
        # 2026-05-22: NET NEGATIVE on our arch (median 3445 ms vs 2061 ms
        # baseline) — our StreamingLinear's dynamic `__dict__.get` dispatch
        # forces per-step Dynamo guard misses + recompiles. Off by default.
        # `on` = enable (only on fast-mode path; needs triton-windows installed).
        # `off` / `auto` (default) = skip compile.
        # See Documentation/FLUX-SPEED-WIP.md.
        "flux_compile": "off",
    },
}

_lock = threading.Lock()
_cache: dict[str, Any] | None = None


def _deep_merge(base: dict, overlay: dict) -> dict:
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def load() -> dict[str, Any]:
    global _cache
    with _lock:
        if _cache is not None:
            return _cache
        SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        stored: dict = {}
        if SETTINGS_PATH.exists():
            try:
                stored = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
            except Exception as e:
                log.warning("settings.json parse failed (%s) — using defaults", e)
        merged = deepcopy(DEFAULTS)
        _deep_merge(merged, stored)
        _cache = merged
        return _cache


def save(updates: dict[str, Any]) -> dict[str, Any]:
    """Patch settings with `updates` (deep merge), persist, return new state."""
    global _cache
    with _lock:
        current = load()
        _deep_merge(current, updates)
        SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        SETTINGS_PATH.write_text(json.dumps(current, indent=2), encoding="utf-8")
        _cache = current
        return _cache


def get(dotted: str, default: Any = None) -> Any:
    cur: Any = load()
    for part in dotted.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


def public_view() -> dict[str, Any]:
    """Settings snapshot safe to send to the UI — masks the API token."""
    s = deepcopy(load())
    token = s.get("civitai", {}).get("api_token") or ""
    s.setdefault("civitai", {})["api_token_set"] = bool(token)
    s["civitai"]["api_token"] = ""  # never echo back
    return s
