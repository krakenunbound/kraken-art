"""Civitai API integration — search, model details, download with progress.

Public API: https://civitai.com/api/v1/
Auth: optional Bearer token from settings (higher rate limits + private/NSFW
gated content).

All downloads go through the existing JobManager so the UI can subscribe to
progress via the same /ws/jobs/{id} WebSocket as image generation.
"""
from __future__ import annotations
import hashlib
import json
import logging
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import config_store
from config import MODELS_ROOT
from jobs import manager

log = logging.getLogger("kraken.civitai")
router = APIRouter()

CIVITAI_BASE = "https://civitai.com/api/v1"

# Civitai type → our `models/<folder>/` mapping. None = not auto-downloadable.
TYPE_TO_FOLDER: dict[str, str | None] = {
    "Checkpoint":        "checkpoints",
    "LORA":              "loras",
    "LoCon":             "loras",
    "DoRA":              "loras",
    "TextualInversion":  "embeddings",
    "VAE":               "vae",
    "Controlnet":        "controlnet",
    "Upscaler":          "upscale_models",
    "AestheticGradient": "embeddings",
    "Hypernetwork":      "hypernetworks",  # folder may not exist yet
    "Poses":             None,
    "Wildcards":         None,
    "Workflows":         None,
    "Other":             None,
}

_SAFE_NAME_RE = re.compile(r'[<>:"|?*\\/]')


def _safe_filename(name: str) -> str:
    return _SAFE_NAME_RE.sub("_", name).strip(". ")


def _auth_headers() -> dict[str, str]:
    token = config_store.get("civitai.api_token", "")
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {}


def _category_for_type(civitai_type: str) -> str | None:
    # Honour user override first, then built-in mapping.
    overrides = config_store.get("downloads.category_paths", {}) or {}
    return overrides.get(civitai_type, TYPE_TO_FOLDER.get(civitai_type))


# Arch → canonical subfolder name. Chosen so the model scanner's
# `_detected_arch` (path-substring based, see api/models.py) tags the downloaded
# file with the right architecture, which in turn drives the Generate-tab
# model/LoRA filtering. Folder names may differ slightly from any pre-existing
# manual folders, but detection is substring-based so filtering still works.
_ARCH_SUBFOLDER: dict[str, str] = {
    "illustrious": "Illustrious",
    "pony":        "Pony",
    "sdxl":        "SDXL",
    "sd15":        "SD1.5",
    "sd35":        "SD3.5",
    "flux1":       "Flux1",
    "flux2":       "Flux2",
    "z_image":     "Z-Image",
    "qwen_image":  "Qwen-Image",
    "chroma":      "Chroma",
    "hidream":     "HiDream",
    "hunyuan":     "HunYuan",
    "wan":         "WAN",
    "ltx":         "LTX",
}

# Component archs read their main weight file from `diffusion_models/`, not
# `checkpoints/`. A Civitai "Checkpoint"-type download for one of these — whether
# an all-in-one bundle (TE+VAE baked in) or a bare transformer — is redirected
# there so it appears in the diffusion-model picker the matching pipeline reads
# from. The component pipelines tolerate AIO files (bundled TE/VAE keys are
# dropped before conversion), so both the user's workflows work: AIO and split.
_COMPONENT_ARCHS = {"flux1", "flux2", "z_image", "qwen_image", "chroma", "hidream"}


def _detect_arch_from_version(version: dict[str, Any], model_info: dict[str, Any]) -> str | None:
    """Best-effort architecture from Civitai version metadata. Mirrors the
    ordering in `api/models._detected_arch` so a downloaded file and a
    locally-scanned one resolve to the same arch."""
    haystack = " ".join(
        str(x or "")
        for x in (
            version.get("baseModel"),
            version.get("name"),
            version.get("model", {}).get("name"),
            model_info.get("name"),
        )
    ).lower()
    if "illustrious" in haystack or "ilust" in haystack:
        return "illustrious"
    if "pony" in haystack:
        return "pony"
    if any(s in haystack for s in ("flux-2", "flux2", "flux_2", "flux.2")):
        return "flux2"
    if "flux" in haystack:
        return "flux1"
    if any(s in haystack for s in ("qwen_image", "qwen-image", "qwenimage", "qwen image", "qwen")):
        return "qwen_image"
    if any(s in haystack for s in ("z_image", "zimage", "z-image", "z image")):
        return "z_image"
    if "hunyuan" in haystack:
        return "hunyuan"
    if "wan" in haystack:
        return "wan"
    if "ltx" in haystack:
        return "ltx"
    if any(s in haystack for s in ("sd 3.5", "sd3.5", "sd 3", "sd3", "stable diffusion 3")):
        return "sd35"
    if "chroma" in haystack:
        return "chroma"
    if "hidream" in haystack or "hi-dream" in haystack:
        return "hidream"
    if any(s in haystack for s in ("sd 1.5", "sd1.5", "stable diffusion 1.5")):
        return "sd15"
    if "sdxl" in haystack or "sd xl" in haystack or " xl" in haystack:
        return "sdxl"
    return None


def _route_by_arch(
    version: dict[str, Any], model_info: dict[str, Any], base_category: str
) -> tuple[str, str | None]:
    """Apply arch subfoldering to a download destination.

    Returns `(category_path, detected_arch)`. For component archs downloaded as
    a "Checkpoint", redirects `checkpoints` → `diffusion_models` so the file
    lands where the pipeline that consumes it actually looks.
    """
    arch = _detect_arch_from_version(version, model_info)
    if not arch:
        return base_category, None
    category = base_category
    if base_category == "checkpoints" and arch in _COMPONENT_ARCHS:
        category = "diffusion_models"
    sub = _ARCH_SUBFOLDER.get(arch)
    if sub:
        category = f"{category}/{sub}"
    return category, arch


# ---------- search / details ----------

_search_cache: dict[str, tuple[float, Any]] = {}
_SEARCH_TTL_SEC = 60.0
_SEARCH_AGGREGATE_MAX_PAGES = 8


@router.get("/civitai/search")
async def search(
    query: str = "",
    types: str | None = None,        # comma-separated Civitai types
    baseModels: str | None = None,   # comma-separated baseModel names
    nsfw: bool | None = None,
    limit: int = 20,
    page: int = 1,
    sort: str = "Most Downloaded",
    period: str = "AllTime",
    cursor: str | None = None,
) -> dict[str, Any]:
    params: dict[str, Any] = {"limit": str(limit), "sort": sort}
    if period:
        params["period"] = period
    if cursor:
        params["cursor"] = cursor
    # Civitai rejects `page` when query search is used and cursor pagination is
    # now the reliable path for large model result sets. Keep accepting `page`
    # from our UI for local display, but do not forward it upstream.
    if query:
        params["query"] = query
    if types:
        params["types"] = types
    if baseModels:
        params["baseModels"] = baseModels
    if nsfw is False:
        params["nsfw"] = "false"

    url = CIVITAI_BASE + "/models?" + urlencode(params, doseq=True)
    cached = _search_cache.get(url)
    if cached and (time.time() - cached[0]) < _SEARCH_TTL_SEC:
        return cached[1]

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            r = await client.get(url, headers=_auth_headers())
        except httpx.RequestError as e:
            raise HTTPException(502, f"Civitai unreachable: {e}")
        if r.status_code != 200:
            raise HTTPException(r.status_code, f"Civitai: {r.text[:300]}")
        data = r.json()

        # Civitai cursor pages can be sparse when query + type/baseModel filters
        # are combined. Pull a few cursors ahead so one UI page is filled with
        # actual matches instead of showing "no results" while more cursors exist.
        pages_fetched = 1
        while len(data.get("items") or []) < limit and pages_fetched < _SEARCH_AGGREGATE_MAX_PAGES:
            next_cursor = (data.get("metadata") or {}).get("nextCursor")
            if not next_cursor:
                break
            next_params = dict(params)
            next_params["cursor"] = next_cursor
            next_url = CIVITAI_BASE + "/models?" + urlencode(next_params, doseq=True)
            try:
                next_r = await client.get(next_url, headers=_auth_headers())
            except httpx.RequestError as e:
                raise HTTPException(502, f"Civitai unreachable: {e}")
            if next_r.status_code != 200:
                break
            next_data = next_r.json()
            data["items"] = (data.get("items") or []) + (next_data.get("items") or [])
            data["metadata"] = next_data.get("metadata") or data.get("metadata") or {}
            pages_fetched += 1
            if not (next_data.get("metadata") or {}).get("nextCursor"):
                break

        if len(data.get("items") or []) > limit:
            data["items"] = data["items"][:limit]
    _search_cache[url] = (time.time(), data)
    return data


@router.get("/civitai/model/{model_id}")
async def get_model(model_id: int) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.get(f"{CIVITAI_BASE}/models/{model_id}", headers=_auth_headers())
    if r.status_code != 200:
        raise HTTPException(r.status_code, f"Civitai: {r.text[:300]}")
    return r.json()


@router.get("/civitai/version/{version_id}")
async def get_version(version_id: int) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.get(f"{CIVITAI_BASE}/model-versions/{version_id}", headers=_auth_headers())
    if r.status_code != 200:
        raise HTTPException(r.status_code, f"Civitai: {r.text[:300]}")
    return r.json()


# ---------- download ----------

def _download_job(job) -> dict[str, Any]:
    """Run in the JobManager worker thread. httpx sync streaming, sha256 verify,
    .part resume, progress events on the standard job event bus."""
    p = job.params
    url: str = p["url"]
    dest = Path(p["dest"])
    expected_sha: str | None = p.get("sha256")
    metadata: dict = p.get("metadata", {})

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")

    headers = dict(_auth_headers())
    resume_from = tmp.stat().st_size if tmp.exists() else 0
    if resume_from:
        headers["Range"] = f"bytes={resume_from}-"
        log.info("resuming download of %s from %d bytes", dest.name, resume_from)

    t0 = time.time()
    last_emit = 0.0
    bytes_done = resume_from
    # Hash only computable if we start at 0; on resume we skip integrity check.
    hasher = hashlib.sha256() if (expected_sha and resume_from == 0) else None

    log.info("downloading %s → %s", url, dest)
    with httpx.Client(timeout=httpx.Timeout(60.0, read=300.0), follow_redirects=True) as client:
        with client.stream("GET", url, headers=headers) as r:
            if r.status_code not in (200, 206):
                raise RuntimeError(f"HTTP {r.status_code} from Civitai: {r.text[:200]}")
            total_remaining = int(r.headers.get("content-length", 0))
            total = total_remaining + resume_from if total_remaining else None
            mode = "ab" if resume_from > 0 else "wb"
            with open(tmp, mode) as f:
                for chunk in r.iter_bytes(chunk_size=1024 * 1024):
                    if job.cancel.is_set():
                        log.info("download cancelled by user")
                        return {"cancelled": True, "partial_bytes": bytes_done}
                    if not chunk:
                        continue
                    f.write(chunk)
                    if hasher:
                        hasher.update(chunk)
                    bytes_done += len(chunk)
                    now = time.time()
                    if now - last_emit > 0.5:
                        last_emit = now
                        elapsed = now - t0
                        speed_mbps = (bytes_done - resume_from) / max(elapsed, 0.001) / 1024 / 1024
                        pct = int(bytes_done / total * 100) if total else 0
                        job.emit({
                            "type": "progress",
                            "step": pct,
                            "total_steps": 100,
                            "bytes_done": bytes_done,
                            "bytes_total": total,
                            "speed_mbps": round(speed_mbps, 2),
                        })

    # Verify hash if we computed one
    if expected_sha and hasher:
        actual = hasher.hexdigest()
        if actual.lower() != expected_sha.lower():
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"SHA256 mismatch: expected {expected_sha[:16]}…, got {actual[:16]}…")
        log.info("sha256 verified for %s", dest.name)
    elif expected_sha:
        log.info("skipping sha256 check (resumed download)")

    tmp.rename(dest)
    meta_path = dest.parent / (dest.stem + ".civitai.json")
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("download complete: %s (%.1fs)", dest.name, time.time() - t0)

    # Bust the model-scan cache so the UI sees the new file on next /api/models.
    try:
        from api import models as models_api  # late import to avoid circular
        models_api._cache = None
    except Exception:
        pass

    return {
        "kind": "download",
        "path": str(dest),
        "metadata_path": str(meta_path),
        "size_bytes": bytes_done,
        "elapsed_s": round(time.time() - t0, 1),
    }


class DownloadRequest(BaseModel):
    modelVersionId: int
    fileId: int | None = None  # None = pick primary file
    category_override: str | None = None  # force destination folder by name


@router.post("/civitai/download")
async def start_download(req: DownloadRequest) -> dict[str, Any]:
    # Fetch version + file info
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.get(f"{CIVITAI_BASE}/model-versions/{req.modelVersionId}",
                             headers=_auth_headers())
    if r.status_code != 200:
        raise HTTPException(r.status_code, f"Civitai version: {r.text[:300]}")
    version = r.json()
    model_info: dict[str, Any] = {}
    model_id = version.get("modelId")
    if model_id:
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                model_resp = await client.get(f"{CIVITAI_BASE}/models/{model_id}", headers=_auth_headers())
                if model_resp.status_code == 200:
                    model_info = model_resp.json()
            except httpx.RequestError:
                model_info = {}

    files = version.get("files") or []
    if not files:
        raise HTTPException(404, "Version has no downloadable files")
    if req.fileId:
        f_obj = next((f for f in files if f.get("id") == req.fileId), None)
        if not f_obj:
            raise HTTPException(404, "fileId not in version")
    else:
        f_obj = next((f for f in files if f.get("primary")), files[0])

    civitai_type = version.get("model", {}).get("type", "Other")
    category = req.category_override or _category_for_type(civitai_type)
    if not category:
        raise HTTPException(
            400, f"Civitai type {civitai_type!r} has no folder mapping (override with category_override)."
        )
    detected_arch: str | None = None
    if not req.category_override and category in ("checkpoints", "loras", "embeddings"):
        category, detected_arch = _route_by_arch(version, model_info, category)

    target_dir = MODELS_ROOT / category
    filename = _safe_filename(f_obj.get("name") or f"civitai_{f_obj.get('id')}.safetensors")
    dest = target_dir / filename

    if dest.exists():
        raise HTTPException(409, f"{dest.name} already exists in {category}/")

    download_url = f_obj.get("downloadUrl")
    if not download_url:
        raise HTTPException(400, "Civitai file has no downloadUrl")

    metadata = {
        "civitai_model_id":   version.get("modelId"),
        "civitai_version_id": version.get("id"),
        "name":               version.get("model", {}).get("name") or model_info.get("name"),
        "version_name":       version.get("name"),
        "baseModel":          version.get("baseModel"),
        "trainedWords":       version.get("trainedWords", []),
        "type":               civitai_type,
        "detected_arch":      detected_arch,
        "description":        version.get("description") or model_info.get("description") or version.get("model", {}).get("description"),
        "nsfw":               version.get("model", {}).get("nsfw", False),
        "creator":            version.get("creator") or model_info.get("creator") or {},
        "downloadUrl":        download_url,
        "fileName":           f_obj.get("name"),
        "fileId":             f_obj.get("id"),
        "sha256":             (f_obj.get("hashes") or {}).get("SHA256"),
        "size_kb":            f_obj.get("sizeKB"),
        "images":             [
            {"url": img.get("url"), "nsfw": img.get("nsfw")}
            for img in (version.get("images") or [])[:4]
        ],
    }

    job = manager.submit(
        kind="download",
        params={
            "url": download_url,
            "dest": str(dest),
            "sha256": metadata["sha256"],
            "metadata": metadata,
        },
        fn=_download_job,
    )
    return {"job_id": job.id, "dest": str(dest), "category": category, "detected_arch": detected_arch}
