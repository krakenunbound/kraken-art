"""Serve and manage generated output files.

Endpoints:
  GET    /api/outputs/file/{path}    serve an image by path relative to outputs/
  POST   /api/outputs/import-video   import an uploaded video into outputs/
  POST   /api/outputs/import-video-path copy a local video path into outputs/
  POST   /api/outputs/delete         delete files by relative path(s)
  GET    /api/outputs/list           browse outputs/ (paged, date-organised)
  GET    /api/outputs/latest-images  newest generated images with model labels
  GET    /api/outputs/latest-videos  newest generated videos

All paths are validated to live inside OUTPUTS_ROOT to refuse traversal.
"""
from __future__ import annotations
import json
import logging
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

from config import OUTPUTS_ROOT
from pipelines.output_metadata import read_settings_from_png

log = logging.getLogger("kraken.outputs")
router = APIRouter()

VIDEO_EXTS = {".mp4", ".webm", ".mov", ".mkv", ".avi", ".m4v"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


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


def _model_label(settings: dict[str, Any] | None) -> str:
    if not settings:
        return "Unknown model"
    model = settings.get("checkpoint") or settings.get("diffusion_model")
    arch = settings.get("arch")
    if model:
        label = Path(str(model)).stem
    elif arch:
        label = str(arch)
    else:
        label = "Unknown model"
    arch_key = re.sub(r"[^a-z0-9]+", "", str(arch or "").lower())
    label_key = re.sub(r"[^a-z0-9]+", "", label.lower())
    if arch and arch_key not in label_key:
        label = f"{label} ({arch})"
    return label


def _image_item(p: Path) -> dict[str, Any]:
    rel = p.relative_to(OUTPUTS_ROOT).as_posix()
    settings = read_settings_from_png(p) if p.suffix.lower() == ".png" else None
    return {
        "rel_path": rel,
        "filename": p.name,
        "size_bytes": p.stat().st_size,
        "mtime": p.stat().st_mtime,
        "model_label": _model_label(settings),
        "arch": settings.get("arch") if settings else None,
        "seed": settings.get("seed") if settings else None,
        "width": settings.get("width") if settings else None,
        "height": settings.get("height") if settings else None,
    }


def _video_item(p: Path) -> dict[str, Any]:
    return {
        "rel_path": p.relative_to(OUTPUTS_ROOT).as_posix(),
        "filename": p.name,
        "path": str(p),
        "size_bytes": p.stat().st_size,
        "mtime": p.stat().st_mtime,
    }


def _safe_video_filename(filename: str) -> str:
    src = Path(filename or "video.mp4").name
    ext = Path(src).suffix.lower()
    if ext not in VIDEO_EXTS:
        raise HTTPException(400, f"unsupported video type: {ext or '(none)'}")
    stem = Path(src).stem
    stem = re.sub(r"[^A-Za-z0-9._ -]+", "_", stem).strip(" ._-") or "video"
    return f"{stem[:90]}{ext}"


def _import_dest(filename: str) -> Path:
    safe = _safe_video_filename(filename)
    dest_dir = OUTPUTS_ROOT / time.strftime("%Y-%m-%d") / "imports"
    dest_dir.mkdir(parents=True, exist_ok=True)
    p = Path(safe)
    stem = f"{time.strftime('%H%M%S')}-source-{p.stem}"
    candidate = dest_dir / f"{stem}{p.suffix}"
    i = 2
    while candidate.exists():
        candidate = dest_dir / f"{stem}-{i}{p.suffix}"
        i += 1
    return candidate


@router.get("/outputs/file/{path:path}")
def serve_file(path: str):
    p = _resolve_safe(path)
    if not p.exists() or not p.is_file():
        raise HTTPException(404, "not found")
    # Browsers cache via the URL — since each generated file is uniquely named
    # we can serve with no-store off, letting cache do its job.
    return FileResponse(p)


@router.get("/outputs/settings/{path:path}")
def output_settings(path: str) -> dict[str, Any]:
    p = _resolve_safe(path)
    if not p.exists() or not p.is_file():
        raise HTTPException(404, "output not found")

    # Primary source: metadata embedded in the PNG itself (Civitai-readable too).
    data = read_settings_from_png(p)
    if data is not None:
        return data

    # Fallback: legacy `.settings.json` sidecar from before metadata was embedded.
    meta = p.with_name(p.stem + ".settings.json")
    if meta.exists() and meta.is_file():
        try:
            data = json.loads(meta.read_text(encoding="utf-8"))
        except Exception as e:
            raise HTTPException(500, f"settings unreadable: {e}")
        if isinstance(data, dict):
            return data

    raise HTTPException(404, "no embedded settings for output")


class DeleteRequest(BaseModel):
    paths: list[str]


class ImportVideoPathRequest(BaseModel):
    path: str


class CaptureVideoFrameRequest(BaseModel):
    source_rel_path: str
    offset_seconds: float = 0.12


def _frame_dest(src: Path) -> Path:
    dest_dir = OUTPUTS_ROOT / time.strftime("%Y-%m-%d") / "frames"
    dest_dir.mkdir(parents=True, exist_ok=True)
    stem = re.sub(r"[^A-Za-z0-9._ -]+", "_", src.stem).strip(" ._-") or "video"
    candidate = dest_dir / f"{time.strftime('%H%M%S')}-lastframe-{stem[:70]}.png"
    i = 2
    while candidate.exists():
        candidate = dest_dir / f"{time.strftime('%H%M%S')}-lastframe-{stem[:70]}-{i}.png"
        i += 1
    return candidate


def _video_frame_count(src: Path) -> int | None:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_frames",
        "-of",
        "default=nokey=1:noprint_wrappers=1",
        str(src),
    ]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, timeout=30).strip()
        return int(out) if out.isdigit() else None
    except Exception:
        return None


@router.post("/outputs/import-video")
async def import_video(filename: str, request: Request) -> dict[str, Any]:
    dest = _import_dest(filename)
    tmp = dest.with_name(dest.name + ".part")
    size = 0
    try:
        with tmp.open("wb") as f:
            async for chunk in request.stream():
                if not chunk:
                    continue
                size += len(chunk)
                f.write(chunk)
        if size <= 0:
            raise HTTPException(400, "uploaded video is empty")
        tmp.replace(dest)
    except HTTPException:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise
    except Exception as e:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise HTTPException(500, f"video import failed: {e}")
    log.info("imported uploaded video: %s (%d bytes)", dest, size)
    return {"item": _video_item(dest)}


@router.post("/outputs/import-video-path")
def import_video_path(req: ImportVideoPathRequest) -> dict[str, Any]:
    raw = (req.path or "").strip().strip('"').strip("'")
    if not raw:
        raise HTTPException(400, "empty video path")
    src = Path(raw).expanduser()
    if not src.exists() or not src.is_file():
        raise HTTPException(404, f"video not found: {raw}")
    if src.suffix.lower() not in VIDEO_EXTS:
        raise HTTPException(400, f"unsupported video type: {src.suffix.lower() or '(none)'}")
    src = src.resolve()
    try:
        src.relative_to(OUTPUTS_ROOT.resolve())
        return {"item": _video_item(src)}
    except ValueError:
        pass

    dest = _import_dest(src.name)
    try:
        shutil.copy2(src, dest)
    except Exception as e:
        raise HTTPException(500, f"video import failed: {e}")
    log.info("imported video path: %s -> %s", src, dest)
    return {"item": _video_item(dest)}


@router.post("/outputs/capture-last-frame")
def capture_last_frame(req: CaptureVideoFrameRequest) -> dict[str, Any]:
    src = _resolve_safe(req.source_rel_path)
    if not src.exists() or not src.is_file():
        raise HTTPException(404, f"source video not found: {req.source_rel_path}")
    if src.suffix.lower() not in VIDEO_EXTS:
        raise HTTPException(400, f"unsupported video type: {src.suffix.lower() or '(none)'}")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise HTTPException(500, "ffmpeg not found on PATH; cannot capture video frame")

    dest = _frame_dest(src)
    offset = max(0.01, min(float(req.offset_seconds or 0.12), 2.0))
    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-sseof",
        f"-{offset:.3f}",
        "-i",
        str(src),
        "-frames:v",
        "1",
        str(dest),
    ]
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=60)
    except Exception as e:
        raise HTTPException(500, f"ffmpeg capture failed: {e}") from e
    if proc.returncode != 0:
        raise HTTPException(500, f"ffmpeg capture failed: {proc.stderr.strip()[:600]}")
    if not dest.exists() or dest.stat().st_size == 0:
        dest.unlink(missing_ok=True)
        frame_count = _video_frame_count(src)
        if frame_count and frame_count > 0:
            idx = frame_count - 1
            exact_cmd = [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(src),
                "-vf",
                f"select=eq(n\\,{idx})",
                "-vsync",
                "0",
                "-frames:v",
                "1",
                str(dest),
            ]
            try:
                proc = subprocess.run(exact_cmd, text=True, capture_output=True, timeout=60)
            except Exception as e:
                raise HTTPException(500, f"ffmpeg exact-frame capture failed: {e}") from e
            if proc.returncode != 0:
                raise HTTPException(500, f"ffmpeg exact-frame capture failed: {proc.stderr.strip()[:600]}")
    if not dest.exists() or dest.stat().st_size == 0:
        raise HTTPException(500, "ffmpeg completed but no frame was captured")
    log.info("captured last frame: %s -> %s", src, dest)
    return {"item": _image_item(dest), "source": _video_item(src)}


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
                for suffix in (".civitai.json", ".settings.json"):
                    meta = p.with_name(p.stem + suffix)
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
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS | VIDEO_EXTS | {".wav", ".mp3"}:
            rel = p.relative_to(OUTPUTS_ROOT).as_posix()
            items.append({
                "rel_path": rel,
                "filename": p.name,
                "size_bytes": p.stat().st_size,
                "mtime": p.stat().st_mtime,
            })
    items.sort(key=lambda x: x["mtime"], reverse=True)
    return {"root": str(OUTPUTS_ROOT), "items": items[:limit]}


@router.get("/outputs/latest-images")
def latest_images(limit: int = 9) -> dict[str, Any]:
    """Return newest saved image outputs, newest first, including model labels."""
    if not OUTPUTS_ROOT.exists():
        return {"root": str(OUTPUTS_ROOT), "items": []}
    limit = max(1, min(int(limit or 9), 50))
    files = [
        p
        for p in OUTPUTS_ROOT.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return {"root": str(OUTPUTS_ROOT), "items": [_image_item(p) for p in files[:limit]]}


@router.get("/outputs/latest-videos")
def latest_videos(limit: int = 20) -> dict[str, Any]:
    """Return newest saved video outputs, newest first."""
    if not OUTPUTS_ROOT.exists():
        return {"root": str(OUTPUTS_ROOT), "items": []}
    limit = max(1, min(int(limit or 20), 100))
    files = [
        p
        for p in OUTPUTS_ROOT.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    ]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return {
        "root": str(OUTPUTS_ROOT),
        "items": [_video_item(p) for p in files[:limit]],
    }
