"""Model folder scanner. Walks F:\\Kraken Art\\models\\ and returns a categorized listing."""
from __future__ import annotations
from pathlib import Path
from fastapi import APIRouter

from config import MODELS_ROOT, MODEL_EXTS, MODEL_CATEGORIES, AUDIO_ACE_ROOT

router = APIRouter()

# Cached listing. Rebuilt on POST /models/refresh and lazily on first GET.
_cache: dict | None = None


def _scan_folder(folder: Path) -> list[dict]:
    out: list[dict] = []
    if not folder.exists():
        return out
    for p in folder.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in MODEL_EXTS:
            continue
        try:
            size = p.stat().st_size
        except OSError:
            size = 0
        rel = p.relative_to(folder)
        out.append({
            "name": str(rel).replace("\\", "/"),
            "filename": p.name,
            "subdir": str(rel.parent).replace("\\", "/") if rel.parent.parts else "",
            "abs_path": str(p),
            "size_bytes": size,
            "ext": p.suffix.lower(),
        })
    out.sort(key=lambda x: x["name"].lower())
    return out


def _scan_audio_external() -> list[dict]:
    """Scan the external Kraken_Audio / ACE-Step-1.5 installation for music models.

    We deliberately do not require the user to copy the multi-GB DiT/LM checkpoints
    into the Kraken Art tree. We just surface what is already there.
    """
    items: list[dict] = []
    if not AUDIO_ACE_ROOT.exists():
        return items
    # Typical ACE layout: ACE-Step-1.5/checkpoints/<model-dir>/*.safetensors (and lm models)
    ckpt_dir = AUDIO_ACE_ROOT / "checkpoints"
    if ckpt_dir.exists():
        for p in ckpt_dir.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() not in MODEL_EXTS:
                continue
            try:
                size = p.stat().st_size
            except OSError:
                size = 0
            rel = p.relative_to(ckpt_dir)
            items.append({
                "name": f"[ace] {str(rel).replace(chr(92), '/')}",
                "filename": p.name,
                "subdir": str(rel.parent).replace(chr(92), "/") if rel.parent.parts else "",
                "abs_path": str(p),
                "size_bytes": size,
                "ext": p.suffix.lower(),
                "source": "external_ace",
            })
    # Also pick up any .safetensors directly under the ACE root (some installs)
    for p in AUDIO_ACE_ROOT.glob("*.safetensors"):
        if p.is_file():
            try:
                size = p.stat().st_size
            except OSError:
                size = 0
            items.append({
                "name": f"[ace] {p.name}",
                "filename": p.name,
                "subdir": "",
                "abs_path": str(p),
                "size_bytes": size,
                "ext": p.suffix.lower(),
                "source": "external_ace",
            })
    items.sort(key=lambda x: x["name"].lower())
    return items


def _build_listing() -> dict:
    cats: dict[str, list[dict]] = {}
    counts: dict[str, int] = {}
    for cat, folders in MODEL_CATEGORIES.items():
        items: list[dict] = []
        for sub in folders:
            items.extend(_scan_folder(MODELS_ROOT / sub))
        if cat == "audio":
            items.extend(_scan_audio_external())
        cats[cat] = items
        counts[cat] = len(items)
    return {
        "root": str(MODELS_ROOT),
        "exists": MODELS_ROOT.exists(),
        "audio_ace_root": str(AUDIO_ACE_ROOT),
        "audio_ace_exists": AUDIO_ACE_ROOT.exists(),
        "categories": cats,
        "counts": counts,
    }


@router.get("/models")
def list_models() -> dict:
    global _cache
    if _cache is None:
        _cache = _build_listing()
    return _cache


@router.post("/models/refresh")
def refresh_models() -> dict:
    global _cache
    _cache = _build_listing()
    return {"refreshed": True, "counts": _cache["counts"]}
