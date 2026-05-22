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


# ---------- search / details ----------

_search_cache: dict[str, tuple[float, Any]] = {}
_SEARCH_TTL_SEC = 60.0


@router.get("/civitai/search")
async def search(
    query: str = "",
    types: str | None = None,        # comma-separated Civitai types
    baseModels: str | None = None,   # comma-separated baseModel names
    nsfw: bool | None = None,
    limit: int = 20,
    page: int = 1,
    sort: str = "Most Downloaded",
) -> dict[str, Any]:
    params: dict[str, Any] = {"limit": str(limit), "page": str(page), "sort": sort}
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
        "name":               version.get("model", {}).get("name"),
        "version_name":       version.get("name"),
        "baseModel":          version.get("baseModel"),
        "trainedWords":       version.get("trainedWords", []),
        "type":               civitai_type,
        "nsfw":               version.get("model", {}).get("nsfw", False),
        "creator":            version.get("creator") or {},
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
    return {"job_id": job.id, "dest": str(dest), "category": category}
