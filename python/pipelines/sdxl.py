"""SDXL pipeline wrapper. One loaded checkpoint cached at a time.

Sampler/scheduler aliasing mirrors common conventions (A1111/ComfyUI names →
diffusers scheduler classes). VAE swap supported. LoRAs deferred to Phase 2.
"""
from __future__ import annotations
import gc
import io
import logging
import time
from base64 import b64encode
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from config import MODELS_ROOT, OUTPUTS_ROOT
from pipelines.load_utils import unload_pipeline

log = logging.getLogger("kraken.sdxl")


_pipeline: Any | None = None
_pipeline_key: tuple | None = None  # (checkpoint_path, vae_path)
_loaded_loras: list[str] = []        # adapter names currently registered on _pipeline
_loaded_embeddings: set[str] = set()


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _dtype() -> torch.dtype:
    return torch.float16 if torch.cuda.is_available() else torch.float32


def _resolve(category: str, name: str | None) -> Path | None:
    if not name:
        return None
    # `name` may be a relative path inside the category folder (subdir/file.safetensors)
    # or just a filename. Try both.
    base = MODELS_ROOT / category
    p = base / name
    if p.exists():
        return p
    for f in base.rglob(name):
        return f
    return None


def _get_scheduler(name: str):
    """Map our sampler names to diffusers scheduler classes."""
    from diffusers import (
        DDIMScheduler,
        DPMSolverMultistepScheduler,
        EulerAncestralDiscreteScheduler,
        EulerDiscreteScheduler,
        UniPCMultistepScheduler,
    )
    n = name.lower()
    if n in ("euler",):                          return EulerDiscreteScheduler
    if n in ("euler_a", "euler_ancestral"):      return EulerAncestralDiscreteScheduler
    if n in ("ddim",):                           return DDIMScheduler
    if n in ("unipc",):                          return UniPCMultistepScheduler
    if n in ("dpmpp_2m", "dpm++_2m"):            return DPMSolverMultistepScheduler
    return EulerDiscreteScheduler


def _set_scheduler(pipe, sampler: str, scheduler: str = "normal") -> None:
    """Sampler picks the integration algorithm; scheduler picks the sigma spacing."""
    sched_cls = _get_scheduler(sampler)
    config = dict(pipe.scheduler.config)
    # Strip any prior schedule flags so we apply fresh ones.
    for k in ("use_karras_sigmas", "use_exponential_sigmas", "use_beta_sigmas"):
        config.pop(k, None)

    s = (scheduler or "normal").lower()
    if s == "karras":
        config["use_karras_sigmas"] = True
    elif s == "exponential":
        config["use_exponential_sigmas"] = True
    elif s == "beta":
        config["use_beta_sigmas"] = True
    elif s in ("sgm_uniform", "trailing"):
        config["timestep_spacing"] = "trailing"
    elif s == "ddim_uniform":
        config["timestep_spacing"] = "linspace"
    # else "normal" / "simple": leave defaults

    pipe.scheduler = sched_cls.from_config(config)


def _ensure_pipeline(checkpoint: str, vae: str | None):
    """Load or reuse the SDXL pipeline. Re-loads only when checkpoint or VAE changes."""
    global _pipeline, _pipeline_key

    ckpt_path = _resolve("checkpoints", checkpoint)
    if not ckpt_path:
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    vae_path = _resolve("vae", vae) if vae else None

    key = (str(ckpt_path), str(vae_path) if vae_path else None)
    if _pipeline is not None and _pipeline_key == key:
        return _pipeline

    if _pipeline is not None:
        global _loaded_loras, _loaded_embeddings
        _loaded_loras = []
        _loaded_embeddings = set()
        del _pipeline
        _pipeline = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    from diffusers import AutoencoderKL, StableDiffusionXLPipeline

    log.info("loading SDXL checkpoint: %s", ckpt_path)
    t0 = time.time()
    try:
        pipe = StableDiffusionXLPipeline.from_single_file(
            str(ckpt_path),
            torch_dtype=_dtype(),
            use_safetensors=ckpt_path.suffix.lower() == ".safetensors",
            add_watermarker=False,
        )
    except Exception as e:
        # The most common failure here is "this isn't actually an SDXL checkpoint" —
        # FLUX, Qwen-Image, HunYuan, LTX, etc. live in the same folder. Give the user
        # a clear hint instead of the raw diffusers error.
        msg = str(e)
        hint = ""
        if "Qwen" in msg or "qwen" in msg:
            hint = " — this checkpoint looks like a Qwen-Image variant, not SDXL. Try a SDXL/Illustrious/Juggernaut/EpicRealismXL file."
        elif "flux" in msg.lower() or "FluxTransformer" in msg:
            hint = " — this checkpoint looks like FLUX. FLUX support is in Phase 2 (task #10)."
        elif "Hunyuan" in msg or "HunYuan" in msg:
            hint = " — this checkpoint looks like HunYuan. Multi-arch support is Phase 3 (task #11)."
        raise RuntimeError(f"SDXL load failed for {ckpt_path.name}: {msg}{hint}") from e
    log.info("checkpoint loaded in %.1fs", time.time() - t0)

    if vae_path:
        try:
            vae_model = AutoencoderKL.from_single_file(str(vae_path), torch_dtype=_dtype())
            pipe.vae = vae_model
        except Exception:
            # If single_file VAE load fails (some VAE files are non-standard), keep the bundled one.
            pass

    # Dynamic offload: picker measures free VRAM vs checkpoint size and chooses
    # the fastest strategy that still fits.
    from pipelines.offload import apply as apply_offload, pick_strategy
    ckpt_size_gb = ckpt_path.stat().st_size / 1024**3
    strategy, info = pick_strategy(ckpt_size_gb, has_text_encoders=False, arch="sdxl")
    actual = apply_offload(pipe, strategy, device=_device())
    log.info("SDXL offload: chose=%s applied=%s · %s", strategy, actual, info)

    pipe.set_progress_bar_config(disable=True)
    try:
        pipe.enable_vae_tiling()
    except Exception:
        pass

    _pipeline = pipe
    _pipeline_key = key
    return _pipeline


def _make_callback(job, total_steps: int, image_index: int, total_images: int):
    """diffusers `callback_on_step_end` — fired after each denoise step."""
    def cb(pipe, step: int, timestep, callback_kwargs):
        if job.cancel.is_set():
            raise RuntimeError("cancelled")
        job.progress.step = step + 1
        job.progress.total_steps = total_steps
        job.progress.image_index = image_index
        job.progress.total_images = total_images
        job.emit({
            "type": "progress",
            "step": step + 1,
            "total_steps": total_steps,
            "image_index": image_index,
            "total_images": total_images,
        })
        return callback_kwargs
    return cb


def _thumbnail_b64(img: Image.Image, max_side: int = 320) -> str:
    thumb = img.copy()
    thumb.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    thumb.save(buf, format="JPEG", quality=80)
    return b64encode(buf.getvalue()).decode("ascii")


def unload() -> dict:
    """Tear down the cached pipeline + free CUDA cache. Returns counts for logging."""
    global _pipeline, _pipeline_key, _loaded_loras, _loaded_embeddings
    had_pipe = _pipeline is not None
    n_loras = len(_loaded_loras)
    n_emb = len(_loaded_embeddings)
    if _pipeline is not None:
        pipe = _pipeline
        _pipeline = None
        _pipeline_key = None
        _loaded_loras = []
        _loaded_embeddings = set()
        unload_pipeline(pipe)
    else:
        _pipeline_key = None
        _loaded_loras = []
        _loaded_embeddings = set()
    return {"had_pipeline": had_pipe, "loras_freed": n_loras, "embeddings_freed": n_emb}


def _adapter_name(lora_name: str) -> str:
    # Adapter names must be safe identifiers for diffusers' adapter registry.
    return "lora_" + "".join(ch if ch.isalnum() else "_" for ch in lora_name)[:48]


def _apply_loras(pipe, loras: list[dict]) -> None:
    """Sync the pipeline's LoRA adapters to match `loras`.

    diffusers caches loaded weights by adapter_name; we unload all old ones, load
    the requested set, then activate them with their respective weights.
    """
    global _loaded_loras

    # Unload any currently-loaded adapters.
    try:
        if _loaded_loras:
            pipe.unload_lora_weights()
    except Exception as e:
        log.warning("unload_lora_weights raised: %s", e)
    _loaded_loras = []

    if not loras:
        return

    names: list[str] = []
    weights: list[float] = []
    for entry in loras:
        path = _resolve("loras", entry["name"])
        if not path:
            log.warning("lora not found, skipping: %s", entry["name"])
            continue
        adapter = _adapter_name(entry["name"])
        try:
            pipe.load_lora_weights(str(path.parent), weight_name=path.name, adapter_name=adapter)
            names.append(adapter)
            weights.append(float(entry.get("weight", 1.0)))
            log.info("loaded LoRA %s @ %.2f", entry["name"], weights[-1])
        except Exception as e:
            log.warning("failed to load LoRA %s: %s", entry["name"], e)

    if names:
        try:
            pipe.set_adapters(names, adapter_weights=weights)
            _loaded_loras = names
        except Exception as e:
            log.warning("set_adapters failed: %s", e)


def _apply_embeddings(pipe, embeddings: list[str]) -> None:
    """Load textual-inversion files. Each becomes a token referenceable in prompts."""
    global _loaded_embeddings
    for name in embeddings:
        if name in _loaded_embeddings:
            continue
        path = _resolve("embeddings", name)
        if not path:
            log.warning("embedding not found, skipping: %s", name)
            continue
        try:
            token = "<" + path.stem + ">"
            pipe.load_textual_inversion(str(path.parent), weight_name=path.name, token=token)
            _loaded_embeddings.add(name)
            log.info("loaded embedding %s as token %s", name, token)
        except Exception as e:
            log.warning("failed to load embedding %s: %s", name, e)


def run(job) -> dict:
    """Job entry. job.params shape matches the GenerateRequest model."""
    from pipelines import flux as flux_mod
    flux_mod.unload()
    gc.collect()

    p = job.params

    pipe = _ensure_pipeline(p["checkpoint"], p.get("vae"))
    _set_scheduler(pipe, p.get("sampler") or "dpmpp_2m", p.get("scheduler") or "normal")
    _apply_loras(pipe, p.get("loras") or [])
    _apply_embeddings(pipe, p.get("embeddings") or [])

    width  = int(p.get("width") or 1024)
    height = int(p.get("height") or 1024)
    steps  = int(p.get("steps") or 30)
    cfg    = float(p.get("cfg") or 7.0)
    count  = int(p.get("count") or 1)
    seed   = p.get("seed")

    out_dir = OUTPUTS_ROOT / time.strftime("%Y-%m-%d")
    out_dir.mkdir(parents=True, exist_ok=True)
    base_stem = time.strftime("%H%M%S") + "-" + job.id[:8]

    saved: list[dict] = []
    device = _device()

    for i in range(count):
        if job.cancel.is_set():
            break

        per_seed = (seed if seed is not None else torch.seed()) + i
        generator = torch.Generator(device=device).manual_seed(int(per_seed) & 0x7FFFFFFF)

        result = pipe(
            prompt=p.get("prompt", ""),
            negative_prompt=p.get("negative", "") or None,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=cfg,
            generator=generator,
            callback_on_step_end=_make_callback(job, steps, i, count),
        )
        img: Image.Image = result.images[0]

        fname = f"{base_stem}-{i:02d}.png"
        fpath = out_dir / fname
        img.save(fpath, format="PNG")
        entry = {
            "path": str(fpath),
            "filename": fname,
            "seed": int(per_seed),
            "width": width,
            "height": height,
        }

        # Optional ESRGAN upscale (USDU + iterative still pending — task #15, #17)
        if p.get("upscale_enabled") and p.get("upscale_model"):
            mode = (p.get("upscale_mode") or "esrgan").lower()
            if mode == "esrgan":
                try:
                    from pipelines import upscale_esrgan
                    up = upscale_esrgan.upscale(img, p["upscale_model"], float(p.get("upscale_factor", 2.0)))
                    up_path = out_dir / f"{base_stem}-{i:02d}-up.png"
                    up.save(up_path, format="PNG")
                    entry["upscaled_path"] = str(up_path)
                    img = up  # preview the upscaled version
                except Exception as e:
                    log.warning("upscale failed: %s", e)
            else:
                log.info("upscale mode '%s' is pending (task #15/#17), skipping", mode)

        rel_path = fpath.relative_to(OUTPUTS_ROOT).as_posix()
        entry["rel_path"] = rel_path
        saved.append(entry)
        job.emit({
            "type": "image",
            "image_index": i,
            "path": str(fpath),
            "rel_path": rel_path,
            "filename": fname,
            "seed": int(per_seed),
            "preview_b64": _thumbnail_b64(img),
        })

    return {
        "kind": "image",
        "count_requested": count,
        "count_produced": len(saved),
        "outputs": saved,
        "output_dir": str(out_dir),
    }
