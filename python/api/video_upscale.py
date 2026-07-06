"""Video upscaling endpoint.

Runs SeedVR2 through its standalone CLI, then optionally converts the result to
60fps with ffmpeg motion interpolation. The job is queued through the shared
Kraken job manager so it does not race image/video generation for VRAM.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from jobs import manager
from pipelines import video_upscale
from pipelines.audio.engine_manager import manager as _audio_engines

router = APIRouter()


class VideoUpscaleRequest(BaseModel):
    source_rel_path: str | None = None
    source_path: str | None = None

    engine: str = "esrgan"  # seedvr2 | esrgan
    dit_model: str | None = None
    upscale_model: str | None = None
    scale_mode: str = "resolution"  # resolution | factor
    upscale_factor: float = 2.0
    resolution: int = 1080
    max_resolution: int = 1920
    batch_size: int = 33
    temporal_overlap: int = 3
    uniform_batch_size: bool = True
    chunk_size: int = 0
    color_correction: str = "lab"
    ai_detail_strength: float = 0.15
    input_noise_scale: float = 0.0
    latent_noise_scale: float = 0.0
    ten_bit: bool = False
    vae_encode_tiled: bool = True
    vae_encode_tile_size: int = 1024
    vae_encode_tile_overlap: int = 128
    vae_decode_tiled: bool = True
    vae_decode_tile_size: int = 1024
    vae_decode_tile_overlap: int = 128
    blocks_to_swap: int = 0
    swap_io_components: bool = True
    attention_mode: str = "sdpa"
    compile_dit: bool = False
    compile_vae: bool = False
    compile_mode: str = "default"
    seed: int = 42
    esrgan_tile_size: int = 0
    esrgan_tile_overlap: int = 64
    keep_audio: bool = True

    target_fps: int = 60
    interpolation: str = "rife_ncnn"  # none | rife_ncnn | ffmpeg_motion


@router.get("/video-upscale/status")
def get_status() -> dict:
    return video_upscale.status()


@router.post("/video-upscale")
def submit(req: VideoUpscaleRequest) -> dict:
    if req.engine not in {"seedvr2", "esrgan"}:
        raise HTTPException(400, "engine must be 'seedvr2' or 'esrgan'.")
    if not (req.source_rel_path or req.source_path):
        raise HTTPException(400, "Pick a source video or paste a source_path.")
    if req.scale_mode not in {"resolution", "factor"}:
        raise HTTPException(400, "scale_mode must be 'resolution' or 'factor'.")
    if req.scale_mode == "factor" and not (1.0 <= req.upscale_factor <= 8.0):
        raise HTTPException(400, "upscale_factor must be between 1.0 and 8.0.")
    if req.resolution < 240 or req.max_resolution < 240:
        raise HTTPException(400, "resolution and max_resolution must be at least 240.")
    if not (0.0 <= req.ai_detail_strength <= 1.0):
        raise HTTPException(400, "ai_detail_strength must be between 0.0 and 1.0.")
    if not (0.0 <= req.input_noise_scale <= 1.0):
        raise HTTPException(400, "input_noise_scale must be between 0.0 and 1.0.")
    if not (0.0 <= req.latent_noise_scale <= 1.0):
        raise HTTPException(400, "latent_noise_scale must be between 0.0 and 1.0.")
    if req.batch_size < 1:
        raise HTTPException(400, "batch_size must be at least 1.")
    if (req.batch_size - 1) % 4 != 0:
        raise HTTPException(400, "SeedVR2 batch_size must follow 4n+1: 1, 5, 9, 13, 17, 21, ...")
    if req.interpolation not in {"none", "rife_ncnn", "ffmpeg_motion"}:
        raise HTTPException(400, "interpolation must be 'none', 'rife_ncnn', or 'ffmpeg_motion'.")
    if req.attention_mode not in {"sdpa", "flash_attn_2", "flash_attn_3", "sageattn_2", "sageattn_3"}:
        raise HTTPException(400, "attention_mode must be sdpa, flash_attn_2/3, or sageattn_2/3.")
    if req.compile_mode not in {"default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"}:
        raise HTTPException(400, "compile_mode is invalid.")
    for name, value in {
        "vae_encode_tile_size": req.vae_encode_tile_size,
        "vae_encode_tile_overlap": req.vae_encode_tile_overlap,
        "vae_decode_tile_size": req.vae_decode_tile_size,
        "vae_decode_tile_overlap": req.vae_decode_tile_overlap,
    }.items():
        if value < 0:
            raise HTTPException(400, f"{name} must be 0 or greater.")
    if req.vae_encode_tiled and req.vae_encode_tile_size <= 0:
        raise HTTPException(400, "vae_encode_tile_size must be greater than 0 when encode tiling is enabled.")
    if req.vae_decode_tiled and req.vae_decode_tile_size <= 0:
        raise HTTPException(400, "vae_decode_tile_size must be greater than 0 when decode tiling is enabled.")
    if req.vae_encode_tiled and req.vae_encode_tile_overlap >= req.vae_encode_tile_size:
        raise HTTPException(400, "vae_encode_tile_overlap must be smaller than vae_encode_tile_size.")
    if req.vae_decode_tiled and req.vae_decode_tile_overlap >= req.vae_decode_tile_size:
        raise HTTPException(400, "vae_decode_tile_overlap must be smaller than vae_decode_tile_size.")
    if req.engine == "esrgan" and req.esrgan_tile_size < 0:
        raise HTTPException(400, "esrgan_tile_size must be 0 or greater.")
    if req.engine == "esrgan" and req.esrgan_tile_overlap < 0:
        raise HTTPException(400, "esrgan_tile_overlap must be 0 or greater.")
    try:
        _audio_engines.stop_all()
    except Exception:
        pass
    job = manager.submit("video_upscale", req.model_dump(), video_upscale.run)
    return {"job_id": job.id, "status": job.status}
