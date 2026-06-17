"""Model folder scanner. Walks F:\\Kraken Art\\models\\ and returns a categorized listing."""
from __future__ import annotations
from pathlib import Path
from typing import Any
from fastapi import APIRouter
import html
import re

from config import ROOT, MODELS_ROOT, MODEL_EXTS, MODEL_CATEGORIES, AUDIO_ACE_ROOT

router = APIRouter()

# Cached listing. Rebuilt on POST /models/refresh and lazily on first GET.
_cache: dict | None = None


def _model_metadata(path: Path) -> dict:
    """Read sidecar metadata written by Civitai downloads, when present."""
    import json

    for suffix in (".civitai.json", ".metadata.json"):
        meta_path = path.with_name(path.stem + suffix)
        if not meta_path.exists():
            continue
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}
    return {}


def _detected_arch(path: Path, metadata: dict) -> str | None:
    haystack = " ".join(
        str(x or "")
        for x in (
            path.as_posix(),
            metadata.get("baseModel"),
            metadata.get("name"),
            metadata.get("version_name"),
            metadata.get("type"),
        )
    ).lower()
    if any(s in haystack for s in ("illustrious", "ilust", "illu xl")):
        return "illustrious"
    if any(s in haystack for s in ("flux-2", "flux2", "flux_2")):
        return "flux2"
    if "flux" in haystack:
        return "flux1"
    if any(s in haystack for s in ("qwen_image", "qwen-image", "qwenimage")):
        return "qwen_image"
    if any(s in haystack for s in ("z_image", "zimage", "z-image")):
        return "z_image"
    if any(s in haystack for s in ("ideogram", "ideogram-4")):
        return "ideogram4"
    if "hunyuan" in haystack:
        return "hunyuan"
    if "wan" in haystack:
        return "wan"
    if "ltx" in haystack:
        return "ltx"
    if any(s in haystack for s in ("sdxl", "sdxl 1.0", "pony", "xl", "juggernaut", "epicrealism", "perfectdeliberate")):
        return "sdxl"
    return None


def _plain_text(value: Any) -> str:
    text = html.unescape(str(value or ""))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _parse_recommended_settings(metadata: dict) -> dict | None:
    text = _plain_text(
        metadata.get("description")
        or metadata.get("modelDescription")
        or metadata.get("notes")
        or ""
    )
    if not text:
        return None
    lower = text.lower()
    if "recommended" not in lower and not any(k in lower for k in ("cfg", "steps", "sampling")):
        return None

    out: dict[str, Any] = {"source": "civitai_description"}

    cfg_match = re.search(r"cfg(?:\s*scale)?\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*(?:[-–]\s*(\d+(?:\.\d+)?))?", text, re.I)
    if cfg_match:
        lo = float(cfg_match.group(1))
        hi = float(cfg_match.group(2)) if cfg_match.group(2) else lo
        out["cfg"] = round((lo + hi) / 2, 2)
        out["cfg_range"] = [lo, hi]

    steps_match = re.search(r"steps?\s*[:\-]?\s*(\d+)\s*(?:[-–]\s*(\d+))?", text, re.I)
    if steps_match:
        lo = int(steps_match.group(1))
        hi = int(steps_match.group(2)) if steps_match.group(2) else lo
        out["steps"] = round((lo + hi) / 2)
        out["steps_range"] = [lo, hi]

    res_matches = re.findall(r"(\d{3,4})\s*x\s*(\d{3,4})", text, re.I)
    if res_matches:
        resolutions = [[int(w), int(h)] for w, h in res_matches]
        out["resolutions"] = resolutions
        # Prefer the familiar square size if the model author lists it;
        # otherwise use the first listed recommendation.
        preferred = next((r for r in resolutions if r == [1024, 1024]), resolutions[0])
        out["width"], out["height"] = preferred

    sampler_candidates = [
        ("euler a", "euler_a", "normal"),
        ("euler_a", "euler_a", "normal"),
        ("euler", "euler", "normal"),
        ("dpm++ 2m", "dpmpp_2m", "normal"),
        ("dpmpp_2m", "dpmpp_2m", "normal"),
        ("ddim", "ddim", "normal"),
        ("unipc", "unipc", "normal"),
    ]
    hits = [(lower.find(label), sampler, scheduler) for label, sampler, scheduler in sampler_candidates if lower.find(label) >= 0]
    if hits:
        _, sampler, scheduler = min(hits, key=lambda h: h[0])
        out["sampler"] = sampler
        out["scheduler"] = "karras" if "karras" in lower and sampler == "dpmpp_2m" else scheduler

    return out if len(out) > 1 else None


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
        metadata = _model_metadata(p)
        out.append({
            "name": str(rel).replace("\\", "/"),
            "filename": p.name,
            "subdir": str(rel.parent).replace("\\", "/") if rel.parent.parts else "",
            "abs_path": str(p),
            "size_bytes": size,
            "ext": p.suffix.lower(),
            "base_model": metadata.get("baseModel"),
            "source_type": metadata.get("type"),
            "detected_arch": _detected_arch(rel, metadata),
            "recommended_settings": _parse_recommended_settings(metadata),
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


def _virtual_ideogram_models() -> list[dict]:
    """Expose gated Hugging Face Ideogram repos in the normal model selector."""
    def entry(quant: str, repo_id: str, local_dir: Path) -> dict:
        local_ready = (local_dir / "model_index.json").exists()
        label_source = "local" if local_ready else "Hugging Face gated"
        abs_path = str(local_dir) if local_ready else repo_id
        try:
            size = sum(p.stat().st_size for p in local_dir.rglob("*") if p.is_file()) if local_ready else 0
        except OSError:
            size = 0
        return {
            "name": f"Ideogram 4 {quant.upper()} ({label_source})",
            "filename": f"ideogram-4-{quant}",
            "subdir": "ideogram4",
            "abs_path": abs_path,
            "size_bytes": size,
            "ext": "hf",
            "base_model": "Ideogram 4",
            "source_type": "local_snapshot" if local_ready else "huggingface_gated",
            "detected_arch": "ideogram4",
            "warning": (
                "Use Turbo 12 for iteration. Quality 48 is available but will take minutes per 1024px image."
                if quant == "nf4"
                else "FP8 is not the preferred RTX 3090 path; use NF4 unless you are doing a controlled comparison."
            ),
            "recommended_settings": {
                "cfg": 7.0,
                "steps": 12,
                "width": 1024,
                "height": 1024,
                "source": f"kraken_ideogram4_{quant}",
            },
        }

    def q4k_entry() -> dict:
        local_dir = ROOT / "Ideogram" / "ideogram-4-gguf-q4_k"
        gguf = local_dir / "ideogram4-q4_k.gguf"
        local_ready = gguf.exists()
        label_source = "local" if local_ready else "missing GGUF"
        try:
            size = gguf.stat().st_size if local_ready else 0
        except OSError:
            size = 0
        return {
            "name": f"Ideogram 4 Q4_K GGUF ({label_source}, experimental)",
            "filename": "ideogram-4-gguf-q4_k",
            "subdir": "ideogram4",
            "abs_path": str(local_dir),
            "size_bytes": size,
            "ext": "gguf",
            "base_model": "Ideogram 4",
            "source_type": "local_gguf_q4k" if local_ready else "missing",
            "detected_arch": "ideogram4",
            "experimental": True,
            "disabled_by_default": True,
            "warning": (
                "Guarded experimental: the current in-process Q4_K loader crashed during cold load. "
                "It is a quality/memory experiment, not the speed path. Requires KRAKEN_IDEOGRAM4_ALLOW_Q4K=1."
            ),
            "recommended_settings": {
                "cfg": 7.0,
                "steps": 12,
                "width": 1024,
                "height": 1024,
                "source": "kraken_ideogram4_gguf_q4k",
            },
        }

    def fused_int8_entry() -> dict:
        local_dir = ROOT / "Ideogram" / "ideogram-4-int8-fused"
        weights = ROOT / "Ideogram" / "ideogram-4-int8-w8a8" / "ideogram4-int8-w8a8.safetensors"
        ready = local_dir.exists() and weights.exists()
        try:
            size = weights.stat().st_size if weights.exists() else 0
        except OSError:
            size = 0
        return {
            "name": f"Ideogram 4 INT8 Fused ({'local, guarded experimental' if ready else 'missing weights'})",
            "filename": "ideogram-4-int8-fused",
            "subdir": "ideogram4",
            "abs_path": str(weights if weights.exists() else local_dir),
            "size_bytes": size,
            "ext": "safetensors",
            "base_model": "Ideogram 4",
            "source_type": "local_int8_fused" if ready else "missing",
            "detected_arch": "ideogram4",
            "experimental": True,
            "disabled_by_default": True,
            "warning": (
                "Guarded experimental: fused INT8 is the RTX 3090 speed/quality target, "
                "but it needs isolated-worker testing before normal use."
            ),
            "recommended_settings": {
                "cfg": 7.0,
                "steps": 12,
                "width": 1024,
                "height": 1024,
                "source": "kraken_ideogram4_int8_fused",
            },
        }

    return [
        entry("nf4", "ideogram-ai/ideogram-4-nf4", MODELS_ROOT / "ideogram4" / "nf4"),
        fused_int8_entry(),
        q4k_entry(),
        entry("fp8", "ideogram-ai/ideogram-4-fp8", MODELS_ROOT / "ideogram4" / "fp8"),
    ]


def _build_listing() -> dict:
    cats: dict[str, list[dict]] = {}
    counts: dict[str, int] = {}
    for cat, folders in MODEL_CATEGORIES.items():
        items: list[dict] = []
        for sub in folders:
            items.extend(_scan_folder(MODELS_ROOT / sub))
        if cat == "audio":
            items.extend(_scan_audio_external())
        if cat == "diffusion_models":
            items.extend(_virtual_ideogram_models())
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
