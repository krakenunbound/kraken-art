"""Generation endpoint — submits a job to the manager and returns a job_id."""
from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from jobs import manager
from pipelines import flux, sdxl

router = APIRouter()


class LoraEntry(BaseModel):
    name: str
    weight: float = 1.0


class GenerateRequest(BaseModel):
    # Architecture — picks the pipeline. Only `sdxl` works in Phase 1.
    arch: str = "sdxl"

    # All-in-one checkpoint (SDXL/Illustrious/Juggernaut). For component-style
    # archs (FLUX, Qwen, etc.) `checkpoint` is omitted and these are set instead.
    checkpoint: str | None = None
    diffusion_model: str | None = None
    vae: str | None = None
    text_encoders: list[str] = Field(default_factory=list)
    clip_vision: str | None = None

    loras: list[LoraEntry] = Field(default_factory=list)
    embeddings: list[str] = Field(default_factory=list)

    # Prompt
    prompt: str = ""
    negative: str = ""

    # Image params
    width: int = 1024
    height: int = 1024
    steps: int = 30
    cfg: float = 7.0
    sampler: str = "dpmpp_2m"
    scheduler: str = "normal"          # normal | karras | exponential | sgm_uniform | beta | simple
    count: int = 1
    seed: int | None = None

    # Upscale (Phase 2 — backend accepts, actual upscale lands with tasks #10/#15)
    upscale_enabled: bool = False
    upscale_mode: str = "esrgan"          # esrgan | usdu | iterative
    upscale_model: str | None = None
    upscale_factor: float = 2.0
    upscale_denoise: float = 0.35         # USDU/iterative only
    upscale_tile_size: int = 512          # USDU only


_ARCH_HINTS = {
    "flux2":      "FLUX2 pipeline lands in Phase 3 (task #11).",
    "qwen_image": "Qwen-Image pipeline lands in Phase 3 (task #11). Picks: diffusion_model + qwen_image_vae + qwen text encoder.",
    "hunyuan":    "HunYuan pipeline lands in Phase 3 (task #11).",
    "wan":        "WAN video pipeline lands in Phase 4 (task #12).",
    "ltx":        "LTX video pipeline lands in Phase 4 (task #12).",
}


@router.post("/generate")
def generate(req: GenerateRequest) -> dict:
    arch = req.arch.lower()
    if arch in ("sdxl", "illustrious"):
        if not req.checkpoint:
            raise HTTPException(400, "SDXL/Illustrious requires a `checkpoint` (all-in-one .safetensors).")
        job = manager.submit("image", req.model_dump(), sdxl.run)
        return {"job_id": job.id, "status": job.status}
    if arch == "flux1":
        if not req.diffusion_model:
            raise HTTPException(400, "FLUX1 requires a `diffusion_model` (the transformer .safetensors).")
        if not req.vae:
            raise HTTPException(400, "FLUX1 requires a VAE — pick `QWEN/ae.safetensors` from the VAE dropdown.")
        if len(req.text_encoders) < 2 or not req.text_encoders[0] or not req.text_encoders[1]:
            raise HTTPException(400, "FLUX1 requires two text encoders: slot 1 = clip_l.safetensors, slot 2 = t5xxl_fp16 (or fp8).")
        job = manager.submit("image", req.model_dump(), flux.run)
        return {"job_id": job.id, "status": job.status}
    hint = _ARCH_HINTS.get(arch, "")
    raise HTTPException(400, f"architecture {req.arch!r} not yet supported. {hint}")


@router.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    job = manager.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    return job.snapshot()


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict:
    ok = manager.cancel(job_id)
    if not ok:
        raise HTTPException(409, "job not cancellable")
    return {"job_id": job_id, "status": "cancel_requested"}


@router.get("/jobs")
def list_jobs() -> dict:
    return {"jobs": [j.snapshot() for j in manager.list()]}
