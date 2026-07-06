"""Z-Image pipeline. Component assembly from local single-file safetensors.

Z-Image (Alibaba's Tongyi-Lab text-to-image) is built from three weight files,
ComfyUI-style:
  - transformer: `ZImageTransformer2DModel`, native (non-diffusers) key layout
    converted on load by diffusers' `convert_z_image_transformer_checkpoint_to_diffusers`.
  - text encoder: a Qwen3-4B base model (`Qwen3Model`). The file ships with a
    `model.` prefix (the ForCausalLM layout); we strip it and load the encoder.
    Conditioning is `hidden_states[-2]` after `apply_chat_template(enable_thinking=True)`,
    masked to each prompt's real token length — a LIST of variable-length tensors.
  - VAE: the Flux.1-AE (16-channel, scaling 0.3611 / shift 0.1159, no quant convs),
    LDM key layout, converted with `convert_ldm_vae_checkpoint`.

Architecture metadata (configs + tokenizer) lives under
`model_configs/z_image_turbo/` so we never touch gated HF repos; only the heavy
weights come from the user's local model folders.

VRAM strategy: encode the prompt first with the Qwen3 encoder, then FREE it
before loading the transformer. That keeps peak resident memory to
transformer (~11.5 GB) + VAE (~0.3 GB) during sampling rather than holding the
7.5 GB encoder at the same time. The Turbo variant runs guidance_scale=0.0
(no CFG); the base variant uses CFG and also needs negative conditioning.
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

import safetensors.torch as sft
import torch
from PIL import Image

from config import MODELS_ROOT, OUTPUTS_ROOT
from pipelines.load_utils import (
    ensure_module_dtype,
    file_size_gb,
    materialize_meta_tensors,
    stream_safetensors_into_meta,
    unload_pipeline,
)
from pipelines.output_metadata import build_output_stem, save_png_with_metadata

log = logging.getLogger("kraken.z_image")

_CFG_ROOT = Path(__file__).resolve().parent.parent / "model_configs" / "z_image_turbo"

_pipeline: Any | None = None
_pipeline_key: tuple | None = None
_loaded_loras: list[str] = []


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


def _free_vram_gb() -> float | None:
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        pynvml.nvmlShutdown()
        return mem.free / 1024**3
    except Exception:
        return None


def unload() -> dict:
    global _pipeline, _pipeline_key, _loaded_loras
    had = _pipeline is not None
    n_loras = len(_loaded_loras)
    pipe = _pipeline
    _pipeline = None
    _pipeline_key = None
    _loaded_loras = []
    if pipe is not None:
        unload_pipeline(pipe)
    return {"had_pipeline": had, "loras_freed": n_loras}


# ---------- text encoding ----------

def _is_turbo(diffusion_model: str) -> bool:
    n = (diffusion_model or "").lower()
    return any(tag in n for tag in ("turbo", "lightning", "lcm", "hyper", "schnell"))


def _encode_z_image_prompt(
    te_path: Path,
    prompts: list[str],
    *,
    dtype: torch.dtype,
    max_sequence_length: int = 512,
) -> list[torch.Tensor]:
    """Encode each prompt with the Qwen3-4B encoder, returning a list of
    variable-length masked embeddings (CPU tensors). The encoder is loaded,
    used, and freed entirely within this call so it never coexists with the
    transformer in VRAM.
    """
    from transformers import AutoTokenizer, Qwen3Model

    t0 = time.time()
    cfg = json.loads((_CFG_ROOT / "text_encoder" / "config.json").read_text())
    from transformers import Qwen3Config
    # The file ships as a Qwen3ForCausalLM; we only need the base encoder.
    cfg.pop("architectures", None)
    qcfg = Qwen3Config(**cfg)
    text_encoder = stream_safetensors_into_meta(
        Qwen3Model, qcfg, te_path, dtype, "Qwen3-4B TE", key_prefix_strip="model."
    )
    ensure_module_dtype(text_encoder, dtype, "Qwen3-4B TE")
    tokenizer = AutoTokenizer.from_pretrained(str(_CFG_ROOT / "tokenizer"))

    device = torch.device(_device())
    text_encoder = text_encoder.to(device=device, dtype=dtype)

    templated: list[str] = []
    for prompt_item in prompts:
        messages = [{"role": "user", "content": prompt_item}]
        templated.append(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
        )

    embeddings_list: list[torch.Tensor] = []
    with torch.inference_mode():
        text_inputs = tokenizer(
            templated,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = text_inputs.input_ids.to(device)
        masks = text_inputs.attention_mask.to(device).bool()
        hidden = text_encoder(
            input_ids=input_ids,
            attention_mask=masks,
            output_hidden_states=True,
        ).hidden_states[-2]
        for i in range(len(hidden)):
            embeddings_list.append(hidden[i][masks[i]].to(dtype=dtype, device="cpu"))

    del text_encoder, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    log.info(
        "encoded %d Z-Image prompt(s) in %.1fs; Qwen3 encoder unloaded before transformer load",
        len(prompts), time.time() - t0,
    )
    return embeddings_list


# ---------- component loaders ----------

def _load_transformer(tx_path: Path, dtype: torch.dtype):
    from diffusers import ZImageTransformer2DModel
    from diffusers.loaders.single_file_utils import (
        convert_z_image_transformer_checkpoint_to_diffusers,
    )

    size_gb = file_size_gb(tx_path)
    log.info("loading Z-Image transformer from %s (%.1f GB on disk)", tx_path, size_gb)
    t0 = time.time()

    cfg = json.loads((_CFG_ROOT / "transformer" / "config.json").read_text())
    cfg.pop("_class_name", None)
    cfg.pop("_diffusers_version", None)

    raw_sd = sft.load_file(str(tx_path))
    # Native Z-Image layout (cap_embedder, context_refiner, x_embedder, …).
    # The converter renames + QKV-splits to diffusers keys and strips any
    # `model.diffusion_model.` prefix from all-in-one checkpoints.
    sd = convert_z_image_transformer_checkpoint_to_diffusers(raw_sd)
    del raw_sd
    gc.collect()

    with torch.device("meta"):
        transformer = ZImageTransformer2DModel.from_config(cfg)
    missing, unexpected = transformer.load_state_dict(sd, strict=False, assign=True)
    if missing:
        log.warning("transformer missing %d keys (first: %s)", len(missing), missing[:3])
    if unexpected:
        log.warning("transformer unexpected %d keys (first: %s)", len(unexpected), list(unexpected)[:3])
    del sd
    materialize_meta_tensors(transformer, dtype, "transformer")
    ensure_module_dtype(transformer, dtype, "transformer")
    gc.collect()
    log.info("  transformer loaded in %.1fs", time.time() - t0)
    return transformer


def _load_vae(vae_path: Path, dtype: torch.dtype):
    from diffusers import AutoencoderKL
    from diffusers.loaders.single_file_utils import convert_ldm_vae_checkpoint

    log.info("loading Z-Image VAE from %s (LDM->diffusers convert)", vae_path)
    cfg = json.loads((_CFG_ROOT / "vae" / "config.json").read_text())
    cfg.pop("_class_name", None)
    cfg.pop("_diffusers_version", None)

    raw_sd = sft.load_file(str(vae_path))
    stripped = {k.removeprefix("first_stage_model."): v for k, v in raw_sd.items()}
    del raw_sd
    vae_sd = convert_ldm_vae_checkpoint(stripped, cfg)
    del stripped

    with torch.device("meta"):
        vae = AutoencoderKL.from_config(cfg)
    missing, unexpected = vae.load_state_dict(vae_sd, strict=False, assign=True)
    if missing:
        log.warning("VAE missing %d keys (first: %s)", len(missing), missing[:3])
    if unexpected:
        log.warning("VAE unexpected %d keys (first: %s)", len(unexpected), list(unexpected)[:3])
    del vae_sd
    materialize_meta_tensors(vae, dtype, "VAE")
    vae = vae.to("cpu", dtype=dtype)
    ensure_module_dtype(vae, dtype, "VAE")
    gc.collect()
    return vae


# ---------- pipeline lifecycle ----------

def _ensure_pipeline(diffusion_model: str, vae_name: str | None):
    global _pipeline, _pipeline_key

    tx_path = _resolve("diffusion_models", diffusion_model)
    vae_path = _resolve("vae", vae_name) if vae_name else None
    if not tx_path:
        raise FileNotFoundError(f"diffusion model not found: {diffusion_model}")
    if not vae_path:
        raise ValueError("Z-Image requires a VAE — pick the Z-Image / Flux.1-AE VAE from the VAE dropdown.")

    key = (str(tx_path), str(vae_path))
    if _pipeline is not None and _pipeline_key == key:
        return _pipeline

    unload()
    log.info("Z-Image pipeline load starting (transformer convert + VAE)")

    from diffusers import FlowMatchEulerDiscreteScheduler, ZImagePipeline

    dtype = _dtype()
    transformer = _load_transformer(tx_path, dtype)
    vae = _load_vae(vae_path, dtype)

    sched_cfg = json.loads((_CFG_ROOT / "scheduler" / "scheduler_config.json").read_text())
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(sched_cfg)

    # Text encoder + tokenizer are handled out-of-band in `_encode_z_image_prompt`
    # and freed before this runs, so the pipeline holds no encoder. We pass
    # prompt_embeds directly at call time.
    pipe = ZImagePipeline(
        scheduler=scheduler,
        vae=vae,
        text_encoder=None,
        tokenizer=None,
        transformer=transformer,
    )
    pipe.set_progress_bar_config(disable=True)

    # Placement: transformer (~11.5 GB) + VAE (~0.3 GB) fit comfortably on a
    # 24 GB card once the encoder is freed. On smaller cards fall back to
    # diffusers' model_cpu_offload so it still runs (slower) rather than OOMing.
    free_vram = _free_vram_gb()
    model_gb = sum(p.numel() * p.element_size() for p in transformer.parameters()) / 1024**3
    needed_gb = model_gb + file_size_gb(vae_path) + 2.0
    if free_vram is None or free_vram >= needed_gb:
        pipe.to(_device())
        pipe._kraken_offload_strategy = "no_offload_full"  # type: ignore[attr-defined]
        log.info(
            "Z-Image: full GPU placement (free=%s GB, need=%.1f GB)",
            f"{free_vram:.1f}" if free_vram is not None else "?", needed_gb,
        )
    else:
        pipe.enable_model_cpu_offload(device=_device())
        pipe._kraken_offload_strategy = "model_cpu_offload"  # type: ignore[attr-defined]
        log.info(
            "Z-Image: model_cpu_offload (free=%s GB < need=%.1f GB)",
            f"{free_vram:.1f}" if free_vram is not None else "?", needed_gb,
        )

    _pipeline = pipe
    _pipeline_key = key
    return pipe


# ---------- LoRA + run ----------

def _adapter_name(lora_name: str) -> str:
    return "lora_" + "".join(ch if ch.isalnum() else "_" for ch in lora_name)[:48]


def _lora_model_weight(entry: dict) -> float:
    value = entry.get("model_weight")
    if value is None:
        value = entry.get("weight")
    if value is None:
        value = 1.0
    return float(value)


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
            names.append(adapter); weights.append(_lora_model_weight(entry))
            log.info("loaded LoRA %s @ %.2f", entry["name"], weights[-1])
        except Exception as e:
            log.warning("failed to load LoRA %s: %s", entry["name"], e)
    if names:
        try:
            pipe.set_adapters(names, adapter_weights=weights)
            _loaded_loras = names
        except Exception as e:
            log.warning("set_adapters failed: %s", e)


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
    # Free the other arch pipelines so we don't double-book VRAM.
    from pipelines import flux as flux_mod, sdxl as sdxl_mod
    sdxl_mod.unload()
    flux_mod.unload()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    p = job.params
    diffusion_model = p["diffusion_model"]
    dtype = _dtype()

    te = p.get("text_encoders") or []
    te_name = te[0] if te else None
    te_path = _resolve("text_encoders", te_name) if te_name else None
    if not te_path:
        raise ValueError("Z-Image requires a Qwen3 text encoder in slot 1 (e.g. QWEN/qwen_3_4b.safetensors).")

    width  = int(p.get("width")  or 1024)
    height = int(p.get("height") or 1024)
    turbo = _is_turbo(diffusion_model)
    steps  = int(p.get("steps") or (8 if turbo else 28))
    # Turbo runs with no CFG (guidance 0). Base honours the UI cfg (default 1.0
    # for distilled, but the field allows higher for the full model).
    guidance = 0.0 if turbo else float(p.get("cfg") or 1.0)
    count  = int(p.get("count") or 1)
    seed   = p.get("seed")

    # Z-Image requires height/width divisible by vae_scale_factor*2 (=16). Snap down.
    width  -= width % 16
    height -= height % 16

    # 1) Encode prompt(s) with the Qwen3 encoder, then free it.
    _t_enc = time.time()
    prompt_embeds_cpu = _encode_z_image_prompt(te_path, [p.get("prompt", "")], dtype=dtype)
    negative_embeds_cpu: list[torch.Tensor] | None = None
    if guidance > 0:
        negative_embeds_cpu = _encode_z_image_prompt(
            te_path, [p.get("negative", "") or ""], dtype=dtype
        )
    _enc_s = time.time() - _t_enc

    # 2) Build transformer + VAE pipeline.
    _t_setup = time.time()
    pipe = _ensure_pipeline(diffusion_model, p.get("vae"))
    _apply_loras(pipe, p.get("loras") or [])
    log.info(
        "Z-Image run stages: text-encode=%.1fs · pipeline-ensure+loras=%.1fs (sampling follows)",
        _enc_s, time.time() - _t_setup,
    )

    exec_device = pipe._execution_device
    prompt_embeds = [t.to(device=exec_device) for t in prompt_embeds_cpu]
    negative_prompt_embeds = (
        [t.to(device=exec_device) for t in negative_embeds_cpu] if negative_embeds_cpu else None
    )
    del prompt_embeds_cpu, negative_embeds_cpu

    out_dir = OUTPUTS_ROOT / time.strftime("%Y-%m-%d")
    out_dir.mkdir(parents=True, exist_ok=True)
    base_stem = build_output_stem(p, job.id)

    saved: list[dict] = []
    for i in range(count):
        if job.cancel.is_set():
            break
        per_seed = (seed if seed is not None else torch.seed()) + i
        generator = torch.Generator(device=exec_device.type).manual_seed(int(per_seed) & 0x7FFFFFFF)
        result = pipe(
            prompt=None,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            width=width, height=height,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
            output_type="pil",
            callback_on_step_end=_make_callback(job, steps, i, count),
        )
        img = result.images[0]

        fname = f"{base_stem}-{i:02d}.png"
        fpath = out_dir / fname
        save_png_with_metadata(img, fpath, p, int(per_seed))
        entry = {
            "path": str(fpath), "filename": fname, "seed": int(per_seed),
            "width": width, "height": height,
        }

        if p.get("upscale_enabled") and p.get("upscale_model"):
            try:
                from pipelines import upscale_esrgan
                up = upscale_esrgan.upscale(img, p["upscale_model"], float(p.get("upscale_factor", 2.0)))
                up_path = out_dir / f"{base_stem}-{i:02d}-up.png"
                save_png_with_metadata(up, up_path, p, int(per_seed))
                entry["upscaled_path"] = str(up_path)
                img = up
            except Exception as e:
                log.warning("upscale failed: %s", e)

        rel_path = fpath.relative_to(OUTPUTS_ROOT).as_posix()
        entry["rel_path"] = rel_path
        saved.append(entry)
        job.emit({
            "type": "image",
            "image_index": i, "path": str(fpath), "rel_path": rel_path, "filename": fname,
            "seed": int(per_seed), "preview_b64": _thumbnail_b64(img),
        })

    return {
        "kind": "image",
        "count_requested": count,
        "count_produced": len(saved),
        "outputs": saved,
        "output_dir": str(out_dir),
    }
