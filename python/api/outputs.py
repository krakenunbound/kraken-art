"""Serve and manage generated output files.

Endpoints:
  GET    /api/outputs/file/{path}    serve an image by path relative to outputs/
  POST   /api/outputs/delete         delete files by relative path(s)
  GET    /api/outputs/list           browse outputs/ (paged, date-organised)

All paths are validated to live inside OUTPUTS_ROOT to refuse traversal.
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from config import OUTPUTS_ROOT

log = logging.getLogger("kraken.outputs")
router = APIRouter()


def _resolve_safe(rel: str) -> Path:
    """Resolve `rel` under OUTPUTS_ROOT, refuse path traversal."""
    if not rel:
        raise HTTPException(400, "empty path")
    candidate = (OUTPUTS_ROOT / rel).resolve()
    root = OUTPUTS_ROOT.resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        raise HTTPException(400, f"path outside outputs/: {rel!r}")
    return candidate


@router.get("/outputs/file/{path:path}")
def serve_file(path: str):
    p = _resolve_safe(path)
    if not p.exists() or not p.is_file():
        raise HTTPException(404, "not found")
    # Browsers cache via the URL — since each generated file is uniquely named
    # we can serve with no-store off, letting cache do its job.
    return FileResponse(p)


class DeleteRequest(BaseModel):
    paths: list[str]


@router.post("/outputs/delete")
def delete_files(req: DeleteRequest) -> dict[str, Any]:
    deleted: list[str] = []
    errors: list[dict[str, str]] = []
    for rel in req.paths:
        try:
            p = _resolve_safe(rel)
            if p.exists() and p.is_file():
                p.unlink()
                deleted.append(rel)
                # Also drop sidecar metadata if any
                meta = p.with_name(p.stem + ".civitai.json")
                if meta.exists():
                    meta.unlink()
            else:
                errors.append({"path": rel, "error": "not found"})
        except HTTPException:
            raise
        except Exception as e:
            errors.append({"path": rel, "error": str(e)})
    log.info("delete: %d removed, %d errors", len(deleted), len(errors))
    return {"deleted": deleted, "errors": errors}


@router.get("/outputs/list")
def list_files(limit: int = 200) -> dict[str, Any]:
    """Walk outputs/ recursively, return up to `limit` recent files (newest first)."""
    if not OUTPUTS_ROOT.exists():
        return {"root": str(OUTPUTS_ROOT), "items": []}
    items: list[dict[str, Any]] = []
    for p in OUTPUTS_ROOT.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".mp4", ".webm", ".wav", ".mp3"}:
            rel = p.relative_to(OUTPUTS_ROOT).as_posix()
            items.append({
                "rel_path": rel,
                "filename": p.name,
                "size_bytes": p.stat().st_size,
                "mtime": p.stat().st_mtime,
            })
    items.sort(key=lambda x: x["mtime"], reverse=True)
    return {"root": str(OUTPUTS_ROOT), "items": items[:limit]}
