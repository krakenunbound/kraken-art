"""Generation endpoint — submits a job to the manager and returns a job_id."""
from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from jobs import manager
from pipelines import flux, ideogram4, sdxl, wan_video, z_image
from pipelines.audio.engine_manager import manager as _audio_engines

router = APIRouter()


def _release_audio_for_visual() -> None:
    """VRAM arbiter (direction 2): stop any running audio engine before an
    image/video job so its model isn't holding VRAM the visual pipeline needs.
    Safe under the single-FIFO job worker — by the time a visual request lands,
    any audio job has finished and only the (idle, resident) engine remains."""
    try:
        _audio_engines.stop_all()
    except Exception:
        pass


class LoraEntry(BaseModel):
    name: str
    weight: float | None = None
    model_weight: float | None = None


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
    ideogram_magic: bool = True
    ideogram_magic_mode: str = "local"  # local | api | raw
    # Adaptive velocity-cache speed/quality knob (Ideogram only):
    #   max  = every step computed (reference quality, ~4.3 min on a 3090)
    #   high = near-identical quality, ~2x faster (~2.0 min)  [default]
    #   fast = great drafts, ~2.7x faster (~1.6 min)
    ideogram_speed_mode: str = "high"

    # Image params
    width: int = 1024
    height: int = 1024
    steps: int = 30
    cfg: float = 7.0
    sampler: str = "dpmpp_2m"
    scheduler: str = "normal"          # normal | karras | exponential | sgm_uniform | beta | simple
    clip_skip: int | None = None        # SDXL/Illustrious only; None/0 = diffusers default
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
    "wan":        "WAN is a video model — use POST /generate/video (the Video tab), not /generate.",
    "ltx":        "LTX video pipeline lands in Phase 4 (task #12).",
}


class VideoGenerateRequest(BaseModel):
    # Only WAN i2v in Phase 4. Dual-expert: high-noise + low-noise transformers.
    arch: str = "wan"

    diffusion_model: str | None = None     # high-noise expert
    diffusion_model_2: str | None = None   # low-noise expert
    vae: str | None = None
    text_encoders: list[str] = Field(default_factory=list)  # slot 1 = UMT5-XXL

    loras: list[LoraEntry] = Field(default_factory=list)

    prompt: str = ""
    negative: str = ""

    # The i2v conditioning first frame: base64 (data URL or raw) or a path
    # (absolute, or relative to the outputs root).
    input_image: str | None = None

    width: int = 832
    height: int = 480
    num_frames: int = 81
    fps: int = 16
    steps: int = 6
    cfg: float = 1.0
    seed: int | None = None


class IdeogramMagicPromptRequest(BaseModel):
    prompt: str = ""
    negative: str = ""
    width: int = 1024
    height: int = 1024
    mode: str = "local"


@router.post("/generate/video")
def generate_video(req: VideoGenerateRequest) -> dict:
    arch = req.arch.lower()
    if arch != "wan":
        raise HTTPException(400, f"video architecture {req.arch!r} not supported (only `wan` i2v).")
    if not req.diffusion_model or not req.diffusion_model_2:
        raise HTTPException(400, "WAN i2v requires both `diffusion_model` (high-noise) and `diffusion_model_2` (low-noise) experts.")
    if not req.vae:
        raise HTTPException(400, "WAN i2v requires the WAN VAE — pick it from the VAE dropdown.")
    if not req.text_encoders or not req.text_encoders[0]:
        raise HTTPException(400, "WAN i2v requires the UMT5-XXL text encoder in slot 1.")
    if not req.input_image:
        raise HTTPException(400, "WAN i2v requires an input image (the first frame).")
    _release_audio_for_visual()
    ideogram4.unload()
    job = manager.submit("video", req.model_dump(), wan_video.run)
    return {"job_id": job.id, "status": job.status}


@router.post("/generate")
def generate(req: GenerateRequest) -> dict:
    arch = req.arch.lower()
    _release_audio_for_visual()
    if arch in ("sdxl", "illustrious"):
        if not req.checkpoint:
            raise HTTPException(400, "SDXL/Illustrious requires a `checkpoint` (all-in-one .safetensors).")
        ideogram4.unload()
        job = manager.submit("image", req.model_dump(), sdxl.run)
        return {"job_id": job.id, "status": job.status}
    if arch == "flux1":
        if not req.diffusion_model:
            raise HTTPException(400, "FLUX1 requires a `diffusion_model` (the transformer .safetensors).")
        if not req.vae:
            raise HTTPException(400, "FLUX1 requires a VAE — pick `QWEN/ae.safetensors` from the VAE dropdown.")
        if len(req.text_encoders) < 2 or not req.text_encoders[0] or not req.text_encoders[1]:
            raise HTTPException(400, "FLUX1 requires two text encoders: slot 1 = clip_l.safetensors, slot 2 = t5xxl_fp16 (or fp8).")
        ideogram4.unload()
        job = manager.submit("image", req.model_dump(), flux.run)
        return {"job_id": job.id, "status": job.status}
    if arch == "z_image":
        if not req.diffusion_model:
            raise HTTPException(400, "Z-Image requires a `diffusion_model` (the transformer .safetensors).")
        if not req.vae:
            raise HTTPException(400, "Z-Image requires a VAE — pick the Z-Image / Flux.1-AE VAE from the VAE dropdown.")
        if not req.text_encoders or not req.text_encoders[0]:
            raise HTTPException(400, "Z-Image requires a Qwen3 text encoder in slot 1 (e.g. QWEN/qwen_3_4b.safetensors).")
        ideogram4.unload()
        job = manager.submit("image", req.model_dump(), z_image.run)
        return {"job_id": job.id, "status": job.status}
    if arch == "ideogram4":
        if not req.diffusion_model:
            raise HTTPException(400, "Ideogram 4 requires selecting the virtual Ideogram 4 model.")
        job = manager.submit("image", req.model_dump(), ideogram4.run)
        return {"job_id": job.id, "status": job.status}
    hint = _ARCH_HINTS.get(arch, "")
    raise HTTPException(400, f"architecture {req.arch!r} not yet supported. {hint}")


@router.post("/ideogram4/magic-prompt")
def ideogram_magic_prompt(req: IdeogramMagicPromptRequest) -> dict:
    try:
        return ideogram4.expand_prompt(
            req.prompt,
            req.negative,
            req.width,
            req.height,
            req.mode,
            pretty=True,
        )
    except Exception as e:
        raise HTTPException(500, f"{type(e).__name__}: {e}") from e


# ---- Deterministic prompt builder (shared across image archs) --------------
# One "scene" (subject + style/lighting/camera/mood picks + in-image text)
# renders into each model's native dialect: Ideogram -> structured JSON,
# FLUX/Z-Image -> natural-language paragraph, SDXL -> concise tags. No LLM.

class PromptBuilderRequest(BaseModel):
    arch: str = "ideogram4"
    subject: str = ""
    texts: list[str] = Field(default_factory=list)
    style: str = "auto"
    lighting: str = "auto"
    camera: str = "auto"
    mood: str = "auto"
    negative: str = ""
    width: int = 1024
    height: int = 1024


@router.get("/prompt-builder/options")
def prompt_builder_options() -> dict:
    from pipelines import prompt_builder
    return prompt_builder.options()


@router.post("/prompt-builder/build")
def prompt_builder_build(req: PromptBuilderRequest) -> dict:
    from pipelines import prompt_builder
    try:
        spec = prompt_builder.spec_from_payload(req.model_dump())
        return prompt_builder.build(spec, req.arch)
    except Exception as e:
        raise HTTPException(500, f"{type(e).__name__}: {e}") from e


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
