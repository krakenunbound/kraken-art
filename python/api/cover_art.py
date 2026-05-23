"""POST /api/cover-art — synchronous album-cover generation for the Music tab
and for external callers like Kraken_Audio's Song Studio (Phase C of the audio
integration).

Why sync, not async-with-job-id:
  The original ComfyUI cover-art worker in Song Studio is itself a thread on the
  caller's side that does its own polling. Wrapping a second polling layer
  around our existing JobManager would just add latency for zero functional
  gain. A blocking ~10–30 s call matches the caller's existing mental model
  exactly.

Architecture dispatch:
  arch_hint="flux1" (default)      → use the locally-installed FLUX1 stack
                                     (diffusion_model + ae VAE + clip_l + t5xxl)
  arch_hint="z_image"              → 501 for now; lights up once task #57 lands.
                                     The downloaded gonzalomoZpop_v40.safetensors
                                     becomes the recommended transformer at
                                     that point.
  arch_hint="sdxl"                 → first SDXL/Juggernaut checkpoint on disk
                                     (cheap fallback if FLUX is OOM).

Aspect-ratio buckets match Civitai/Suno conventions and resolve to FLUX-
friendly resolutions:
  square    → 1024 × 1024
  portrait  → 832  × 1216
  landscape → 1216 × 832

Side effect for Song Studio compatibility:
  When the caller passes `song_dir`, we *also* write the PNG into
  `<song_dir>/cover_art/cover.png` and emit `<song_dir>/cover_art/cover.json`
  with the same shape Song Studio's own ComfyUI worker would have written —
  so the existing `latest_cover_art_image_path()` + `cover_art_state()`
  readers in codex_song_studio.py see the result transparently.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from config import MODELS_ROOT, OUTPUTS_ROOT

log = logging.getLogger("kraken.cover_art")
router = APIRouter()


# ----- aspect ratio → (width, height) ------------------------------------------------

ASPECT_RATIOS: dict[str, tuple[int, int]] = {
    "square":    (1024, 1024),
    "portrait":  (832,  1216),
    "landscape": (1216, 832),
}


# ----- default model picks (best-effort, no hard requirement) -------------------------
# These mirror what most users will have if they followed the README + ran the
# Civitai library tab at least once. We never *fail* if a default is missing —
# the request can override each slot explicitly.

def _first_existing(category: str, *candidates: str) -> str | None:
    """Return the first candidate relative path that exists under MODELS_ROOT/category."""
    base = MODELS_ROOT / category
    for c in candidates:
        if (base / c).exists():
            return c
    # Fallback: walk and pick the largest .safetensors in the category.
    if base.exists():
        files = sorted(
            (p for p in base.rglob("*.safetensors") if p.is_file()),
            key=lambda p: p.stat().st_size,
            reverse=True,
        )
        if files:
            return files[0].relative_to(base).as_posix()
    return None


def _flux_defaults() -> dict[str, Any]:
    """Pick a sensible FLUX1 cover-art configuration from what's on disk."""
    diffusion = _first_existing(
        "diffusion_models",
        "Flux 1D FP16/fluxmania_kreamania.safetensors",
        "Flux 1D FP16/project0PJ0Krea_pj0KREAFP16.safetensors",
    )
    vae = _first_existing(
        "vae",
        "FLUX1/fluxVaeSft_aeSft.sft",
        "QWEN/ae.safetensors",
    )
    clip_l = _first_existing("text_encoders", "clip_l.safetensors")
    t5xxl = _first_existing(
        "text_encoders",
        "t5/t5xxl_fp16.safetensors",
        "t5/t5xxl_fp8_e4m3fn.safetensors",
    )
    return {
        "arch":            "flux1",
        "diffusion_model": diffusion,
        "vae":             vae,
        "text_encoders":   [clip_l, t5xxl] if (clip_l and t5xxl) else [],
        "steps":           20,
        "cfg":             1.0,
    }


def _sdxl_defaults() -> dict[str, Any]:
    """Pick a sensible SDXL fallback if FLUX is unavailable."""
    ckpt = _first_existing(
        "checkpoints",
        "juggernautXL_versionX.safetensors",
        "sd_xl_base_1.0.safetensors",
    )
    return {
        "arch":       "sdxl",
        "checkpoint": ckpt,
        "steps":      28,
        "cfg":        7.0,
    }


# ----- request / response shapes -----------------------------------------------------

class CoverArtRequest(BaseModel):
    prompt: str = Field(..., description="The cover-art prompt — full text the diffusion model sees.")
    aspect_ratio: str = Field(
        default="square",
        description=f"One of {sorted(ASPECT_RATIOS.keys())}.",
    )
    seed: int | None = Field(default=None, description="Optional fixed seed for reproducibility.")
    arch_hint: str = Field(default="flux1", description="flux1 | sdxl | z_image (z_image lands later, returns 501 today).")
    # Allow per-call overrides for advanced callers (Song Studio rarely uses these).
    diffusion_model: str | None = None
    vae: str | None = None
    text_encoders: list[str] | None = None
    checkpoint: str | None = None
    steps: int | None = None
    cfg: float | None = None
    negative: str = ""

    # Side-effect: also drop a copy + manifest inside this Song Studio song
    # directory so the existing `latest_cover_art_image_path()` reader picks it
    # up without any further plumbing.
    song_dir: str | None = Field(default=None, description="If set, also write cover into <song_dir>/cover_art/.")
    song_id: str | None = None
    song_title: str | None = None


class CoverArtResponse(BaseModel):
    ok: bool
    image_path: str
    image_url: str
    width: int
    height: int
    arch: str
    model: str | None
    elapsed_s: float
    song_dir_cover_path: str | None = None
    song_dir_manifest_path: str | None = None


# ----- the synthetic Job (we don't go through JobManager; this is sync) --------------

class _InlineJob:
    """Just enough of `jobs.Job` for `flux.run()` / `sdxl.run()` to operate.

    The image pipelines emit progress events via `job.emit(...)`. We log them
    so the user can tail the sidecar log to see cover-art progress in real
    time, but we don't expose a WebSocket for this sync path — that would
    contradict the "blocking call" design choice.
    """

    def __init__(self, params: dict[str, Any]) -> None:
        self.id = uuid.uuid4().hex
        self.params = params
        # `flux.run` / `sdxl.run` consult cancel.is_set() between images. A
        # never-set Event keeps the loop running to completion.
        import threading
        self.cancel = threading.Event()
        # `flux.run` also references `job.progress` (rarely). A tiny stub works.
        class _P:
            step = 0; total_steps = 0; image_index = 0; total_images = 1; message = ""
        self.progress = _P()

    def emit(self, event: dict[str, Any]) -> None:
        kind = event.get("type", "?")
        if kind in {"image", "audio_cover_progress"}:
            log.info("cover-art %s: image %s", self.id[:8], event.get("image_index", 0))
        elif kind == "status":
            log.info("cover-art %s: %s", self.id[:8], event.get("message") or event.get("status"))
        elif kind in {"warning", "error"}:
            log.warning("cover-art %s: %s — %s", self.id[:8], kind, event.get("message"))


# ----- Song Studio cover_art/ side-effect ---------------------------------------------

def _write_song_studio_cover(
    req: CoverArtRequest,
    image_path: Path,
    arch: str,
    model_used: str | None,
) -> tuple[str | None, str | None]:
    """Drop a copy into <song_dir>/cover_art/cover.png + write Song Studio's
    expected `cover.json` manifest so its readers see it.

    Returns (cover_path, manifest_path) or (None, None) when no song_dir.
    """
    if not req.song_dir:
        return None, None
    try:
        song_dir = Path(req.song_dir).resolve()
        cover_dir = song_dir / "cover_art"
        cover_dir.mkdir(parents=True, exist_ok=True)
        suffix = image_path.suffix or ".png"
        dest = cover_dir / f"cover{suffix}"
        shutil.copy2(image_path, dest)
        manifest = {
            "job_id": f"krakenart-{uuid.uuid4().hex[:12]}",
            "song_id": req.song_id or "",
            "song_title": req.song_title or "",
            "status": "completed",
            "progress_text": "Generated cover art saved (Kraken Art).",
            "error": "",
            "image_path": str(dest),
            "prompt_used": req.prompt,
            # Helpful provenance fields so future debugging can tell at a glance
            # that this manifest came from Kraken Art, not ComfyUI.
            "workflow_name": f"kraken-art:{arch}",
            "aspect_ratio": req.aspect_ratio,
            "comfy_prompt_id": "",  # intentionally blank — we are not ComfyUI
            "kraken_art_arch": arch,
            "kraken_art_model": model_used or "",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        manifest_path = cover_dir / "cover.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        log.info("wrote Song Studio cover manifest: %s", manifest_path)
        return str(dest), str(manifest_path)
    except Exception as e:
        log.warning("Song Studio cover side-effect failed (%s); the API caller still got the main file.", e)
        return None, None


# ----- main endpoint -----------------------------------------------------------------

@router.post("/cover-art", response_model=CoverArtResponse)
def generate_cover_art(req: CoverArtRequest) -> CoverArtResponse:
    t0 = time.time()
    aspect = req.aspect_ratio.lower()
    if aspect not in ASPECT_RATIOS:
        raise HTTPException(400, f"aspect_ratio must be one of {sorted(ASPECT_RATIOS.keys())}, got {req.aspect_ratio!r}.")
    width, height = ASPECT_RATIOS[aspect]

    arch = req.arch_hint.lower()
    if arch == "z_image":
        # Honest scope: Z-Image text-encoder loader (Qwen3-4B with the custom
        # 2560-dim projection) is task #57. The Civitai gonzalomoZpop_v40
        # transformer is on disk and verified, but the pipeline wiring isn't
        # ready yet. We refuse rather than silently down-grade to FLUX so the
        # caller knows.
        raise HTTPException(
            501,
            "Z-Image cover backend lands with task #57 (Phase C0 follow-up). "
            "Use arch_hint='flux1' or 'sdxl' today.",
        )

    # Resolve model defaults + apply per-call overrides.
    if arch == "sdxl":
        defaults = _sdxl_defaults()
        from pipelines import sdxl as image_pipeline
        params = {
            "arch":        "sdxl",
            "checkpoint":  req.checkpoint or defaults["checkpoint"],
            "vae":         req.vae,
            "prompt":      req.prompt,
            "negative":    req.negative,
            "width":       width,
            "height":      height,
            "steps":       req.steps or defaults["steps"],
            "cfg":         req.cfg or defaults["cfg"],
            "count":       1,
            "seed":        req.seed,
            "sampler":     "dpmpp_2m",
            "scheduler":   "karras",
        }
        if not params["checkpoint"]:
            raise HTTPException(
                500,
                "SDXL cover requested but no checkpoint found under models/checkpoints/. "
                "Install one or fall back to arch_hint='flux1'.",
            )
        model_used = params["checkpoint"]
    else:
        # flux1 default
        defaults = _flux_defaults()
        from pipelines import flux as image_pipeline
        params = {
            "arch":            "flux1",
            "diffusion_model": req.diffusion_model or defaults["diffusion_model"],
            "vae":             req.vae or defaults["vae"],
            "text_encoders":   req.text_encoders or defaults["text_encoders"],
            "prompt":          req.prompt,
            "negative":        req.negative,
            "width":           width,
            "height":          height,
            "steps":           req.steps or defaults["steps"],
            "cfg":             req.cfg or defaults["cfg"],
            "count":           1,
            "seed":            req.seed,
        }
        missing = [k for k in ("diffusion_model", "vae") if not params[k]]
        if missing or len(params["text_encoders"]) < 2:
            raise HTTPException(
                500,
                "FLUX1 cover requested but the local model picks are incomplete "
                f"(missing: {missing or 'text_encoders[CLIP-L + T5-XXL]'}). "
                "Install the FLUX1 stack first, or send explicit diffusion_model / vae / text_encoders "
                "in the request body.",
            )
        model_used = params["diffusion_model"]

    log.info("cover-art: arch=%s prompt=%r aspect=%s %dx%d", arch, req.prompt[:60], aspect, width, height)

    job = _InlineJob(params)
    try:
        result = image_pipeline.run(job)
    except Exception as e:
        log.error("cover-art generation failed: %s\n%s", e, traceback.format_exc())
        raise HTTPException(500, f"Cover generation failed: {type(e).__name__}: {e}")

    outputs = result.get("outputs") or []
    if not outputs:
        raise HTTPException(500, "Pipeline returned no outputs.")
    first = outputs[0]
    image_path = Path(first["path"])

    # URL the caller can fetch the image at. Goes through the existing
    # /api/outputs/{relative_path} static route the OUTPUTS_ROOT router exposes.
    try:
        rel = image_path.relative_to(OUTPUTS_ROOT).as_posix()
        image_url = f"/api/outputs/{rel}"
    except ValueError:
        image_url = ""

    song_dir_cover, song_dir_manifest = _write_song_studio_cover(req, image_path, arch, model_used)

    return CoverArtResponse(
        ok=True,
        image_path=str(image_path),
        image_url=image_url,
        width=width,
        height=height,
        arch=arch,
        model=model_used,
        elapsed_s=round(time.time() - t0, 1),
        song_dir_cover_path=song_dir_cover,
        song_dir_manifest_path=song_dir_manifest,
    )


@router.get("/cover-art/defaults")
def cover_art_defaults() -> dict[str, Any]:
    """Show what the endpoint would pick if called without overrides. Useful
    for the Music tab settings panel + for the Song Studio integration to
    print on launch ('cover-art backend: Kraken Art FLUX1 — fluxmania...').
    """
    return {
        "aspect_ratios": ASPECT_RATIOS,
        "flux1":         _flux_defaults(),
        "sdxl":          _sdxl_defaults(),
        "z_image": {
            "status": "deferred",
            "task":   "#57",
            "note":   "gonzalomoZpop_v40.safetensors is downloaded and SHA-verified; pipeline wiring (Qwen3-4B text encoder) is the follow-up.",
        },
    }
