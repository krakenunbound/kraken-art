"""Standalone image-upscale endpoint.

Powers the Image Upscale tab: take any image (a gallery output or a loaded
file), run ESRGAN or full USDU (tile + img2img refine) at a chosen model and
target size, and save the result back into outputs/. Runs through the shared
job manager so the minutes-long USDU streams live per-tile progress over the
same /ws/jobs/{id} WebSocket the Generate tab uses.
"""
from __future__ import annotations

import gc
import io
import logging
import time
from base64 import b64encode
from pathlib import Path

from fastapi import APIRouter, HTTPException
from PIL import Image
from pydantic import BaseModel

from config import OUTPUTS_ROOT
from jobs import manager
from pipelines.output_metadata import save_png_with_metadata

log = logging.getLogger("kraken.api.upscale")
router = APIRouter()


class UpscaleRequest(BaseModel):
    # source — one of these
    source_rel_path: str | None = None   # relative to outputs/ (gallery pick)
    source_path: str | None = None       # absolute path (loaded file)

    mode: str = "usdu"                    # "esrgan" | "usdu"
    upscale_model: str | None = None     # spandrel ESRGAN model (pre-upscale)

    size_mode: str = "factor"            # "factor" | "resolution"
    factor: float = 2.0
    target_w: int | None = None
    target_h: int | None = None

    # USDU refine (ignored for esrgan mode)
    refine_arch: str = "sdxl"            # "sdxl" (flux pending)
    refine_checkpoint: str | None = None
    refine_vae: str | None = None
    steps: int = 20
    denoise: float = 0.2
    tile_size: int | None = None
    cfg: float = 6.0
    sampler: str = "dpmpp_2m"
    scheduler: str = "karras"
    clip_skip: int | None = None
    prompt: str = ""
    negative: str = ""
    seed: int = 0

    # snap (stage 3)
    snap_mode: str = "fill / crop"
    anchor_x: str = "center"
    anchor_y: str = "center"


def _resolve_source(req: UpscaleRequest) -> Path:
    if req.source_rel_path:
        p = (OUTPUTS_ROOT / req.source_rel_path).resolve()
        try:
            p.relative_to(OUTPUTS_ROOT.resolve())
        except ValueError:
            raise HTTPException(400, "source_rel_path escapes outputs root")
        if not p.exists():
            raise HTTPException(404, f"source not found: {req.source_rel_path}")
        return p
    if req.source_path:
        p = Path(req.source_path)
        if not p.exists():
            raise HTTPException(404, f"source not found: {req.source_path}")
        return p
    raise HTTPException(400, "no source image given")


def _thumb_b64(img: Image.Image, max_side: int = 384) -> str:
    t = img.copy()
    t.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    t.save(buf, format="JPEG", quality=82)
    return b64encode(buf.getvalue()).decode("ascii")


def _free_other_pipelines(keep: str) -> None:
    """Release VRAM held by archs we won't use this job."""
    for mod_name in ("flux", "ideogram4", "sdxl"):
        if mod_name == keep:
            continue
        try:
            mod = __import__(f"pipelines.{mod_name}", fromlist=["unload"])
            if hasattr(mod, "unload"):
                mod.unload()
        except Exception:
            pass
    gc.collect()


def _run(job) -> dict:
    from pipelines import usdu

    p = job.params
    req = UpscaleRequest(**p)
    src_path = _resolve_source(req)
    img = Image.open(src_path).convert("RGB")
    wi, hi = img.size

    if req.size_mode == "resolution" and req.target_w and req.target_h:
        target_w, target_h = int(req.target_w), int(req.target_h)
    else:
        target_w = max(8, int(round(wi * float(req.factor))))
        target_h = max(8, int(round(hi * float(req.factor))))

    mode = (req.mode or "usdu").lower()
    refine_fn = None
    keep = "sdxl"

    if mode == "usdu":
        if req.refine_arch == "sdxl":
            if not req.refine_checkpoint:
                raise HTTPException(400, "usdu mode needs a refine_checkpoint (SDXL)")
            _free_other_pipelines(keep="sdxl")
            from pipelines import sdxl
            refine_fn = sdxl.make_refiner(
                req.refine_checkpoint, req.refine_vae,
                sampler=req.sampler, scheduler=req.scheduler,
                prompt=req.prompt, negative=req.negative,
                cfg=req.cfg, clip_skip=req.clip_skip,
            )
        else:
            # FLUX/other img2img refine not wired yet — degrade to ESRGAN-only
            # so the job still produces a real upscale instead of failing.
            log.warning("USDU refine_arch=%s not supported yet; running ESRGAN-only", req.refine_arch)
            mode = "esrgan"
    if mode != "usdu":
        _free_other_pipelines(keep="")

    def progress(done: int, total: int, message: str) -> None:
        if job.cancel.is_set():
            raise RuntimeError("cancelled")
        job.progress.step = done
        job.progress.total_steps = max(1, total)
        job.progress.message = message
        job.emit({
            "type": "progress",
            "step": done,
            "total_steps": max(1, total),
            "image_index": 0,
            "total_images": 1,
            "message": message,
        })

    t0 = time.time()
    result = usdu.run_usdu(
        img, target_w, target_h,
        upscale_model=req.upscale_model,
        refine_fn=refine_fn,
        steps=int(req.steps),
        denoise=float(req.denoise),
        seed=int(req.seed),
        tile_size=req.tile_size,
        progress=progress,
        snap_mode=req.snap_mode,
        anchor_x=req.anchor_x,
        anchor_y=req.anchor_y,
    )
    out_img: Image.Image = result["image"]
    elapsed = time.time() - t0

    out_dir = OUTPUTS_ROOT / time.strftime("%Y-%m-%d")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = time.strftime("%H%M%S") + "-" + job.id[:8]
    suffix = "usdu" if (mode == "usdu") else "up"
    fname = f"{stem}-{suffix}.png"
    fpath = out_dir / fname

    meta = {
        "prompt": req.prompt or f"upscale ({suffix}) of {src_path.name}",
        "negative": req.negative,
        "upscale_mode": mode,
        "upscale_model": req.upscale_model,
        "refine_checkpoint": req.refine_checkpoint,
        "steps": req.steps,
        "upscale_denoise": req.denoise,
        "width": out_img.width,
        "height": out_img.height,
        "source": str(src_path),
    }
    save_png_with_metadata(out_img, fpath, meta, int(req.seed))
    rel_path = fpath.relative_to(OUTPUTS_ROOT).as_posix()

    job.emit({
        "type": "image",
        "image_index": 0,
        "path": str(fpath),
        "rel_path": rel_path,
        "filename": fname,
        "seed": int(req.seed),
        "preview_b64": _thumb_b64(out_img),
    })
    log.info("upscale (%s) %dx%d -> %dx%d in %.1fs : %s",
             mode, wi, hi, out_img.width, out_img.height, elapsed, fname)

    return {
        "kind": "upscale",
        "mode": mode,
        "source": str(src_path),
        "output": {
            "path": str(fpath),
            "rel_path": rel_path,
            "filename": fname,
            "width": out_img.width,
            "height": out_img.height,
        },
        "predicted_resolution": f"{result['pred_w']} x {result['pred_h']}",
        "target_resolution": f"{target_w} x {target_h}",
        "tiles": result["tiles"],
        "elapsed_sec": round(elapsed, 1),
        "output_dir": str(out_dir),
    }


@router.post("/upscale")
def upscale(req: UpscaleRequest) -> dict:
    if not (req.source_rel_path or req.source_path):
        raise HTTPException(400, "no source image given")
    job = manager.submit("upscale", req.model_dump(), _run)
    return {"job_id": job.id, "status": job.status}
