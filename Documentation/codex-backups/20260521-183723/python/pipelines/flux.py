"""FLUX1 pipeline. Loads transformer + VAE from local single-file safetensors.

Text encoders (CLIP-L + T5-XXL) load from local safetensors. Their *configs and
tokenizers* come from public HF repos (openai/clip-vit-large-patch14 and
google/t5-v1_1-xxl) — tiny files, no gated-access issues. Heavy weights stay
local.

VRAM is managed dynamically: `pick_strategy()` compares measured component sizes
against free VRAM and selects full_gpu, model_cpu_offload, or sequential.
"""
from __future__ import annotations
import gc
import io
import json
import logging
import time
from base64 import b64encode
from pathlib import Path
from typing import Any

import numpy as np
import safetensors.torch as sft
import torch
from PIL import Image

from config import MODELS_ROOT, OUTPUTS_ROOT
from pipelines.kraken_flux_pipeline import KrakenFluxPipeline
from pipelines.load_utils import (
    ensure_module_dtype,
    file_size_gb,
    materialize_meta_tensors,
    stream_safetensors_into_meta,
    unload_pipeline,
)

log = logging.getLogger("kraken.flux")

_pipeline: Any | None = None
_pipeline_key: tuple | None = None
_loaded_loras: list[str] = []

_OLD_FP8_WEIGHT_SCALE_SUFFIX = ".scale_weight"
_OLD_FP8_INPUT_SCALE_SUFFIX = ".scale_input"
_COMFY_FP8_WEIGHT_SCALE_SUFFIX = ".weight_scale"
_COMFY_FP8_INPUT_SCALE_SUFFIX = ".input_scale"
_COMFY_QUANT_SUFFIX = ".comfy_quant"
_SCALED_FP8_MARKER_SUFFIX = ".scaled_fp8"


def _torch_fp8_dtype(format_name: str) -> torch.dtype:
    if format_name == "float8_e4m3fn":
        return torch.float8_e4m3fn
    if format_name == "float8_e5m2":
        return torch.float8_e5m2
    raise ValueError(f"unsupported FLUX quantization format: {format_name}")


def _decode_comfy_quant_config(tensor: torch.Tensor) -> dict[str, Any]:
    try:
        raw = tensor.detach().cpu().numpy().tobytes().rstrip(b"\x00")
        return json.loads(raw.decode("utf-8"))
    except Exception as e:
        raise ValueError("invalid Comfy quantization metadata in FLUX checkpoint") from e


def _old_scaled_fp8_format(state_dict: dict[str, torch.Tensor]) -> str:
    for key, value in state_dict.items():
        if key == "scaled_fp8" or key.endswith(_SCALED_FP8_MARKER_SUFFIX):
            if value.dtype == torch.float8_e5m2:
                return "float8_e5m2"
            return "float8_e4m3fn"
    return "float8_e4m3fn"


def _dequant_scaled_fp8_weight(
    weight: torch.Tensor,
    scale: torch.Tensor,
    *,
    _storage_dtype: torch.dtype,
    target_dtype: torch.dtype,
) -> torch.Tensor:
    # Comfy old scaled-FP8 checkpoints commonly store values that are already
    # on the FP8 numeric lattice, but in a wider safetensors dtype. Casting those
    # through torch.float8 first only increases peak memory; the dequantized full
    # weight is the numeric value times the saved per-layer scale.
    out = weight.to(target_dtype)
    out.mul_(scale.to(target_dtype))
    return out.contiguous()


def _dequantize_comfy_scaled_fp8(
    state_dict: dict[str, torch.Tensor],
    *,
    target_dtype: torch.dtype,
) -> int:
    old_scale_keys = [k for k in state_dict if k.endswith(_OLD_FP8_WEIGHT_SCALE_SUFFIX)]
    quant_keys = [k for k in state_dict if k.endswith(_COMFY_QUANT_SUFFIX)]
    converted = 0

    if old_scale_keys:
        format_name = _old_scaled_fp8_format(state_dict)
        storage_dtype = _torch_fp8_dtype(format_name)
        missing_weights: list[str] = []
        log.info("  dequantizing %d Comfy old scaled-FP8 FLUX layers (%s -> %s)", len(old_scale_keys), format_name, target_dtype)
        for i, scale_key in enumerate(old_scale_keys):
            layer = scale_key[: -len(_OLD_FP8_WEIGHT_SCALE_SUFFIX)]
            weight_key = f"{layer}.weight"
            weight = state_dict.get(weight_key)
            if weight is None:
                missing_weights.append(weight_key)
                continue
            state_dict[weight_key] = _dequant_scaled_fp8_weight(
                weight,
                state_dict[scale_key],
                _storage_dtype=storage_dtype,
                target_dtype=target_dtype,
            )
            converted += 1
            if (i + 1) % 32 == 0:
                gc.collect()
        if missing_weights:
            raise ValueError(
                "FLUX checkpoint has old scaled-FP8 metadata but missing weights; "
                f"first missing keys: {missing_weights[:3]}"
            )

    if quant_keys:
        log.info("  dequantizing %d Comfy quantized FLUX layers -> %s", len(quant_keys), target_dtype)
        for i, quant_key in enumerate(quant_keys):
            layer = quant_key[: -len(_COMFY_QUANT_SUFFIX)]
            conf = _decode_comfy_quant_config(state_dict[quant_key])
            format_name = conf.get("format")
            if format_name not in {"float8_e4m3fn", "float8_e5m2"}:
                raise ValueError(
                    "FLUX checkpoint uses unsupported Comfy quantization "
                    f"format {format_name!r} in layer {layer}"
                )
            weight_key = f"{layer}.weight"
            scale_key = f"{layer}{_COMFY_FP8_WEIGHT_SCALE_SUFFIX}"
            weight = state_dict.get(weight_key)
            scale = state_dict.get(scale_key)
            if weight is None or scale is None:
                raise ValueError(
                    "FLUX checkpoint has Comfy quant metadata but missing tensor data; "
                    f"layer={layer}, weight_present={weight is not None}, scale_present={scale is not None}"
                )
            state_dict[weight_key] = _dequant_scaled_fp8_weight(
                weight,
                scale,
                _storage_dtype=_torch_fp8_dtype(format_name),
                target_dtype=target_dtype,
            )
            converted += 1
            if (i + 1) % 32 == 0:
                gc.collect()

    if converted:
        for key in list(state_dict.keys()):
            if (
                key == "scaled_fp8"
                or key.endswith(_SCALED_FP8_MARKER_SUFFIX)
                or key.endswith(_OLD_FP8_WEIGHT_SCALE_SUFFIX)
                or key.endswith(_OLD_FP8_INPUT_SCALE_SUFFIX)
                or key.endswith(_COMFY_FP8_WEIGHT_SCALE_SUFFIX)
                or key.endswith(_COMFY_FP8_INPUT_SCALE_SUFFIX)
                or key.endswith(_COMFY_QUANT_SUFFIX)
            ):
                del state_dict[key]
        gc.collect()

    return converted


def _looks_like_bfl_flux(state_dict: dict[str, torch.Tensor]) -> bool:
    return (
        "img_in.weight" in state_dict
        or any(k.startswith("double_blocks.") for k in state_dict)
        or any(k.startswith("single_blocks.") for k in state_dict)
    )


def _looks_like_diffusers_flux(state_dict: dict[str, torch.Tensor]) -> bool:
    return (
        "time_text_embed.timestep_embedder.linear_1.weight" in state_dict
        or any(k.startswith("transformer_blocks.") for k in state_dict)
        or any(k.startswith("single_transformer_blocks.") for k in state_dict)
    )


def _prepare_flux_transformer_state_dict(
    raw_state_dict: dict[str, torch.Tensor],
    *,
    dtype: torch.dtype,
):
    from diffusers.loaders.single_file_utils import convert_flux_transformer_checkpoint_to_diffusers

    state_dict = {k.removeprefix("model.diffusion_model."): v for k, v in raw_state_dict.items()}
    del raw_state_dict
    gc.collect()

    converted_fp8 = _dequantize_comfy_scaled_fp8(state_dict, target_dtype=dtype)
    if converted_fp8:
        log.info("  dequantized %d scaled-FP8 transformer layers before Diffusers conversion", converted_fp8)

    if _looks_like_diffusers_flux(state_dict):
        log.info("  detected Diffusers-format FLUX transformer checkpoint")
        return state_dict

    if _looks_like_bfl_flux(state_dict):
        log.info("  converting %d BFL/Comfy FLUX transformer keys to Diffusers format", len(state_dict))
        return convert_flux_transformer_checkpoint_to_diffusers(state_dict)

    sample_keys = list(state_dict.keys())[:8]
    raise ValueError(
        "unsupported FLUX transformer checkpoint layout. "
        f"Expected BFL/Comfy or Diffusers keys; first keys: {sample_keys}"
    )


# ---------- utilities ----------

def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _dtype() -> torch.dtype:
    return torch.bfloat16 if torch.cuda.is_available() else torch.float32


def _resolve(category: str, name: str | None) -> Path | None:
    if not name:
        return None
    base = MODELS_ROOT / category
    p = base / name
    if p.exists():
        return p
    for f in base.rglob(name):
        return f
    return None


def unload() -> dict:
    global _pipeline, _pipeline_key, _loaded_loras
    had = _pipeline is not None
    n_loras = len(_loaded_loras)
    if _pipeline is not None:
        pipe = _pipeline
        _pipeline = None
        _pipeline_key = None
        _loaded_loras = []
        unload_pipeline(pipe)
    else:
        _pipeline_key = None
        _loaded_loras = []
    return {"had_pipeline": had, "loras_freed": n_loras}


# ---------- component loaders ----------

def _load_clip_l(path: Path, dtype: torch.dtype):
    from transformers import CLIPTextModel, CLIPTextConfig
    config = CLIPTextConfig.from_pretrained("openai/clip-vit-large-patch14")
    return stream_safetensors_into_meta(
        CLIPTextModel, config, path, dtype, "CLIP-L", key_prefix_strip="text_encoder."
    )


def _load_t5xxl(path: Path, dtype: torch.dtype):
    from transformers import T5EncoderModel, T5Config
    config = T5Config.from_pretrained("google/t5-v1_1-xxl")
    return stream_safetensors_into_meta(
        T5EncoderModel, config, path, dtype, "T5-XXL", key_prefix_strip="text_encoder_2."
    )


def _get_tokenizers():
    from transformers import CLIPTokenizer, T5TokenizerFast
    return (
        CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14"),
        T5TokenizerFast.from_pretrained("google/t5-v1_1-xxl"),
    )


# ---------- pipeline lifecycle ----------

def _ensure_pipeline(diffusion_model: str, vae_name: str | None, te1: str | None, te2: str | None):
    global _pipeline, _pipeline_key

    tx_path  = _resolve("diffusion_models", diffusion_model)
    vae_path = _resolve("vae", vae_name) if vae_name else None
    te1_path = _resolve("text_encoders", te1) if te1 else None
    te2_path = _resolve("text_encoders", te2) if te2 else None

    if not tx_path:
        raise FileNotFoundError(f"diffusion model not found: {diffusion_model}")
    if not vae_path:
        raise ValueError("FLUX requires a VAE — pick `QWEN/ae.safetensors` (the FLUX/Qwen-shared VAE) from the VAE dropdown.")
    if not te1_path or not te2_path:
        raise ValueError("FLUX requires two text encoders. Slot 1: clip_l.safetensors. Slot 2: t5xxl_fp16.safetensors (or fp8 for less VRAM).")

    key = (str(tx_path), str(vae_path), str(te1_path), str(te2_path))
    if _pipeline is not None and _pipeline_key == key:
        return _pipeline

    unload()

    log.info("FLUX pipeline load starting (transformer conversion + text encoder load)")

    from diffusers import (
        AutoencoderKL,
        FluxTransformer2DModel,
        FlowMatchEulerDiscreteScheduler,
    )
    from diffusers.loaders.single_file_utils import (
        convert_ldm_vae_checkpoint,
    )

    dtype = _dtype()

    # Inform (don't block) on tight RAM/VRAM. Sequential offload usually squeaks
    # through on modest GPUs, but competing processes can still cause OOM/crash.
    size_gb = file_size_gb(tx_path)
    te1_gb = file_size_gb(te1_path)
    te2_gb = file_size_gb(te2_path)
    vae_gb = file_size_gb(vae_path)
    try:
        import psutil
        free_ram_gb = psutil.virtual_memory().available / 1024**3
        try:
            import pynvml
            pynvml.nvmlInit()
            mem = pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(0))
            free_vram_gb = mem.free / 1024**3
            pynvml.nvmlShutdown()
        except Exception:
            free_vram_gb = None
        log.info(
            "pre-flight: transformer=%.1f GB · free RAM=%.1f GB · free VRAM=%s",
            size_gb, free_ram_gb,
            f"{free_vram_gb:.1f} GB" if free_vram_gb is not None else "unknown",
        )
        if free_ram_gb < 8 or (free_vram_gb is not None and free_vram_gb < 6):
            log.warning(
                "tight resources for FLUX: free RAM %.1f GB / free VRAM %s. "
                "Sequential CPU offload will try anyway but expect either a slow run or a crash. "
                "Most likely culprit if it crashes: another process (ComfyUI especially) holding RAM/VRAM.",
                free_ram_gb,
                f"{free_vram_gb:.1f} GB" if free_vram_gb is not None else "unknown",
            )
    except ImportError:
        log.warning("psutil not installed — pre-flight check skipped.")

    # Local config root — bypasses HF's gated `black-forest-labs/FLUX.1-dev`
    # entirely. Weights stay local; architecture metadata ships with the project.
    flux_cfg_root = Path(__file__).resolve().parent.parent / "model_configs" / "flux1_dev"

    # Comfy/BFL safetensors use keys like `double_blocks.*`; Diffusers expects
    # `transformer_blocks.*`. Some Comfy checkpoints also store old scaled-FP8
    # codes plus scale tensors, so normalize those to real weights before the
    # Diffusers conversion instead of letting them render garbage.
    log.info("loading FLUX transformer from %s (%.1f GB on disk)", tx_path, size_gb)
    t0 = time.time()
    with open(flux_cfg_root / "transformer" / "config.json") as f:
        tx_cfg = json.load(f)

    raw_tx_sd = sft.load_file(str(tx_path))
    tx_sd = _prepare_flux_transformer_state_dict(raw_tx_sd, dtype=dtype)
    gc.collect()

    with torch.device("meta"):
        transformer = FluxTransformer2DModel.from_config(tx_cfg)
    missing, unexpected = transformer.load_state_dict(tx_sd, strict=False, assign=True)
    if missing:
        log.warning("transformer missing %d keys after conversion (first: %s)", len(missing), missing[:3])
    if unexpected:
        log.warning("transformer unexpected %d keys after conversion (first: %s)", len(unexpected), list(unexpected)[:3])
    del tx_sd
    materialize_meta_tensors(transformer, dtype, "transformer")
    transformer = transformer.to("cpu", dtype=dtype)
    ensure_module_dtype(transformer, dtype, "transformer")
    gc.collect()
    log.info("  transformer loaded in %.1fs", time.time() - t0)

    # Diffusers' single_file loader branches on extension and refuses non-.safetensors
    # files (the user's FLUX VAE is `fluxVaeSft_aeSft.sft`). Bypass it: construct the
    # AutoencoderKL from our local config, then load weights from the safetensors-
    # format file directly via the safetensors library (works for any extension).
    # The user's FLUX VAE (`fluxVaeSft_aeSft.sft`) uses LDM-style keys
    # (`encoder.down.0.block.0.conv1.weight`) whereas diffusers' AutoencoderKL
    # expects `encoder.down_blocks.0.resnets.0.conv1.weight`. Run the standard
    # LDM→diffusers VAE converter before loading.
    log.info("loading FLUX VAE from %s (manual safetensors load + LDM->diffusers convert)", vae_path)
    with open(flux_cfg_root / "vae" / "config.json") as f:
        vae_cfg = json.load(f)

    raw_vae_sd = sft.load_file(str(vae_path))
    # Strip optional LDM wrapper prefix.
    stripped_vae_sd = {k.removeprefix("first_stage_model."): v for k, v in raw_vae_sd.items()}
    del raw_vae_sd
    vae_sd = convert_ldm_vae_checkpoint(stripped_vae_sd, vae_cfg)
    del stripped_vae_sd

    with torch.device("meta"):
        vae = AutoencoderKL.from_config(vae_cfg)
    missing, unexpected = vae.load_state_dict(vae_sd, strict=False, assign=True)
    if missing:
        log.warning("VAE missing %d keys after conversion (first: %s)", len(missing), missing[:3])
    if unexpected:
        log.warning("VAE unexpected %d keys after conversion (first: %s)", len(unexpected), list(unexpected)[:3])
    del vae_sd
    materialize_meta_tensors(vae, dtype, "VAE")
    vae = vae.to("cpu", dtype=dtype)
    ensure_module_dtype(vae, dtype, "VAE")
    gc.collect()

    text_encoder   = _load_clip_l(te1_path, dtype)
    text_encoder_2 = _load_t5xxl(te2_path, dtype)
    ensure_module_dtype(text_encoder, dtype, "CLIP-L")
    ensure_module_dtype(text_encoder_2, dtype, "T5-XXL")
    tokenizer, tokenizer_2 = _get_tokenizers()

    # FLUX-dev's flow-match scheduler MUST be configured with dynamic shifting
    # and the dev shift values. The default constructor gives use_dynamic_shifting=False
    # which silently produces noise output at 1024x1024 because the noise schedule
    # is wrong.
    with open(flux_cfg_root / "scheduler" / "scheduler_config.json") as f:
        scheduler_cfg = json.load(f)
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_cfg)

    pipe = KrakenFluxPipeline(
        transformer=transformer,
        vae=vae,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        scheduler=scheduler,
    )

    from pipelines.offload import apply as apply_offload, pick_strategy
    strategy, info = pick_strategy(
        size_gb,
        has_text_encoders=True,
        arch="flux1",
        text_encoder_size_gb=te1_gb,
        secondary_text_encoder_size_gb=te2_gb,
        vae_size_gb=vae_gb,
    )
    actual = apply_offload(pipe, strategy, device=_device())
    pipe._kraken_offload_strategy = actual  # type: ignore[attr-defined]
    log.info("FLUX offload: chose=%s applied=%s · %s", strategy, actual, info)

    pipe.set_progress_bar_config(disable=True)
    # VAE tiling causes checkerboard garbage on FLUX decode at 512/1024 — skip it.
    # FLUX VAE is ~300 MB; with VAE pinned on GPU we do not need tiling for VRAM.
    if actual == "sequential_cpu_offload":
        try: pipe.enable_attention_slicing()
        except Exception: pass

    _pipeline = pipe
    _pipeline_key = key
    return pipe


# ---------- LoRA + run ----------

def _adapter_name(lora_name: str) -> str:
    return "lora_" + "".join(ch if ch.isalnum() else "_" for ch in lora_name)[:48]


def _apply_loras(pipe, loras: list[dict]) -> None:
    global _loaded_loras
    try:
        if _loaded_loras:
            pipe.unload_lora_weights()
    except Exception as e:
        log.warning("unload_lora_weights raised: %s", e)
    _loaded_loras = []
    if not loras:
        return
    names, weights = [], []
    for entry in loras:
        path = _resolve("loras", entry["name"])
        if not path:
            log.warning("lora not found, skipping: %s", entry["name"])
            continue
        adapter = _adapter_name(entry["name"])
        try:
            pipe.load_lora_weights(str(path.parent), weight_name=path.name, adapter_name=adapter)
            names.append(adapter); weights.append(float(entry.get("weight", 1.0)))
            log.info("loaded LoRA %s @ %.2f", entry["name"], weights[-1])
        except Exception as e:
            log.warning("failed to load LoRA %s: %s", entry["name"], e)
    if names:
        try:
            pipe.set_adapters(names, adapter_weights=weights)
            _loaded_loras = names
        except Exception as e:
            log.warning("set_adapters failed: %s", e)


def _flux_embedded_guidance(diffusion_model: str) -> float:
    """ComfyUI 'FluxGuidance' equivalent — NOT the KSampler CFG slider.

    FLUX dev models bake guidance into the transformer (`guidance_embeds=True`).
    diffusers exposes this as `guidance_scale` (~3.5 for dev, ~1 for schnell).
    """
    n = (diffusion_model or "").lower()
    if any(tag in n for tag in ("schnell", "turbo", "lightning", "lcm", "hyper")):
        return 1.0
    return 3.5


def _generator_device(pipe, offload_strategy: str) -> str:
    if offload_strategy in ("sequential_cpu_offload", "model_cpu_offload"):
        return "cpu"
    exec_dev = pipe._execution_device
    return exec_dev.type if isinstance(exec_dev, torch.device) else str(exec_dev)


def _make_callback(job, total_steps: int, image_index: int, total_images: int):
    def cb(pipe, step: int, timestep, callback_kwargs):
        if job.cancel.is_set():
            raise RuntimeError("cancelled")
        job.progress.step = step + 1
        job.progress.total_steps = total_steps
        job.progress.image_index = image_index
        job.progress.total_images = total_images
        job.emit({
            "type": "progress",
            "step": step + 1, "total_steps": total_steps,
            "image_index": image_index, "total_images": total_images,
        })
        return callback_kwargs
    return cb


def _thumbnail_b64(img: Image.Image, max_side: int = 320) -> str:
    thumb = img.copy()
    thumb.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    thumb.save(buf, format="JPEG", quality=80)
    return b64encode(buf.getvalue()).decode("ascii")


def run(job) -> dict:
    from pipelines import sdxl as sdxl_mod
    sdxl_mod.unload()
    gc.collect()

    p = job.params

    te = p.get("text_encoders") or []
    te1 = te[0] if len(te) >= 1 else None
    te2 = te[1] if len(te) >= 2 else None

    pipe = _ensure_pipeline(p["diffusion_model"], p.get("vae"), te1, te2)
    _apply_loras(pipe, p.get("loras") or [])

    width  = int(p.get("width")  or 1024)
    height = int(p.get("height") or 1024)
    steps  = int(p.get("steps")  or 28)
    # UI `cfg` = ComfyUI KSampler CFG (classical negative-prompt guidance). FLUX keeps this at 1.
    # Embedded dev guidance (ComfyUI FluxGuidance node) is separate — see _flux_embedded_guidance().
    true_cfg = float(p.get("cfg") or 1.0)
    embedded_guidance = _flux_embedded_guidance(p.get("diffusion_model", ""))
    count  = int(p.get("count")  or 1)
    seed   = p.get("seed")

    if p.get("negative"):
        log.info("FLUX ignores negative prompts; dropping '%s'", p["negative"][:60])

    out_dir = OUTPUTS_ROOT / time.strftime("%Y-%m-%d")
    out_dir.mkdir(parents=True, exist_ok=True)
    base_stem = time.strftime("%H%M%S") + "-" + job.id[:8]

    saved: list[dict] = []
    offload_strategy = getattr(pipe, "_kraken_offload_strategy", "model_cpu_offload")
    gen_device = _generator_device(pipe, offload_strategy)
    for i in range(count):
        if job.cancel.is_set():
            break
        per_seed = (seed if seed is not None else torch.seed()) + i
        generator = torch.Generator(device=gen_device).manual_seed(int(per_seed) & 0x7FFFFFFF)
        result = pipe(
            prompt=p.get("prompt", ""),
            width=width, height=height,
            num_inference_steps=steps,
            guidance_scale=embedded_guidance,
            true_cfg_scale=true_cfg,
            generator=generator,
            output_type="pil",
            callback_on_step_end=_make_callback(job, steps, i, count),
        )
        img = result.images[0]

        fname = f"{base_stem}-{i:02d}.png"
        fpath = out_dir / fname
        img.save(fpath, format="PNG")
        entry = {
            "path": str(fpath), "filename": fname, "seed": int(per_seed),
            "width": width, "height": height,
        }

        # Optional upscale
        if p.get("upscale_enabled") and p.get("upscale_model"):
            try:
                from pipelines import upscale_esrgan
                up = upscale_esrgan.upscale(img, p["upscale_model"], float(p.get("upscale_factor", 2.0)))
                up_path = out_dir / f"{base_stem}-{i:02d}-up.png"
                up.save(up_path, format="PNG")
                entry["upscaled_path"] = str(up_path)
                img = up  # preview the upscaled version
            except Exception as e:
                log.warning("upscale failed: %s", e)

        saved.append(entry)
        job.emit({
            "type": "image",
            "image_index": i, "path": str(fpath), "filename": fname,
            "seed": int(per_seed), "preview_b64": _thumbnail_b64(img),
        })

    return {
        "kind": "image",
        "count_requested": count,
        "count_produced": len(saved),
        "outputs": saved,
        "output_dir": str(out_dir),
    }
