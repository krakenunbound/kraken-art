"""FLUX1 pipeline. Loads transformer + VAE from local single-file safetensors.

Text encoders (CLIP-L + T5-XXL) load from local safetensors. Their *configs and
tokenizers* come from public HF repos (openai/clip-vit-large-patch14 and
google/t5-v1_1-xxl) — tiny files, no gated-access issues. Heavy weights stay
local.

VRAM is managed dynamically: `pick_strategy()` compares measured component sizes
against free VRAM and selects full_gpu, model_cpu_offload, or sequential.
"""
from __future__ import annotations
import collections
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
from pipelines.output_metadata import build_output_stem, save_png_with_metadata

log = logging.getLogger("kraken.flux")

_pipeline: Any | None = None
_pipeline_key: tuple | None = None
_loaded_loras: list[str] = []


def _install_pos_embed_cache() -> None:
    """Monkey-patch diffusers' FluxPosEmbed.forward to cache rotary embeddings
    across the 28 sampling steps of a single pipe(...) call.

    Diffusers recomputes RoPE every step in float64 (`get_1d_rotary_pos_embed`
    in `embeddings.py`), then casts back to bf16. The ids tensor for a fixed
    resolution + prompt length is identical bytes across every step within a
    generation, so a hit-on-shape-and-endpoints cache is a free ~280 ms.

    Cache key uses tensor shape + first / last scalar values rather than a
    byte-hash, which is faster than a full tensor compare and uniquely
    identifies the (height, width, txt_len) triple for FLUX's row-major ids.
    Cross-call collisions (e.g. a different prompt with the same dims) are
    impossible because txt_ids carries the position information.
    """
    try:
        from diffusers.models.transformers.transformer_flux import FluxPosEmbed
    except ImportError:
        return  # diffusers not installed where we expected; skip silently
    if getattr(FluxPosEmbed, "_kraken_cached_forward", False):
        return  # already patched (idempotent)

    original_forward = FluxPosEmbed.forward

    def cached_forward(self, ids: "torch.Tensor"):  # type: ignore[no-redef]
        # ids is tiny (~14 KB for 1024² FLUX with 512-token prompt). The .item()
        # calls do force a small CPU↔GPU sync but the value is preallocated and
        # cheap. We measured ~5 µs total here vs ~10 ms for the original forward.
        try:
            key = (
                tuple(ids.shape),
                ids.dtype,
                ids.device.type,
                float(ids[0, 0].item()),
                float(ids[-1, -1].item()),
            )
        except Exception:
            return original_forward(self, ids)
        cache = getattr(self, "_kraken_pos_cache", None)
        if cache is not None and cache[0] == key:
            return cache[1]
        value = original_forward(self, ids)
        self._kraken_pos_cache = (key, value)
        return value

    FluxPosEmbed.forward = cached_forward
    FluxPosEmbed._kraken_cached_forward = True
    log.info("installed FluxPosEmbed.forward cache (RoPE reused across sampling steps)")


# Install at import time so the patch is in place before any pipe loads.
# Toggled off for per-step A/B comparison 2026-05-22.
# _install_pos_embed_cache()

# Install comfy-kitchen's fused apply_rope kernel as a drop-in for diffusers'
# apply_rotary_emb when the package is available. Apache-2.0 from Comfy-Org —
# the same kernel ComfyUI uses in their FLUX path. Falls back to diffusers'
# pure-Python original if comfy-kitchen isn't installed.
# 2026-05-22: DISABLED while investigating a 12× per-step regression measured
# in our streaming-partition path. Standalone the kernel is faster (1.49 ms vs
# 2.84 ms for diffusers) but called from FluxAttnProcessor with VRAM under
# pressure (~500 MB free during sampling) the gen slows to ~24 s/step. Cause
# unidentified; see Documentation/FLUX-SPEED-WIP.md for details.
# from pipelines.kraken_rope import install_ck_rope_patch
# install_ck_rope_patch()

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


def _extract_comfy_scaled_fp8_scales(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    old_scale_keys = [k for k in state_dict if k.endswith(_OLD_FP8_WEIGHT_SCALE_SUFFIX)]
    quant_keys = [k for k in state_dict if k.endswith(_COMFY_QUANT_SUFFIX)]
    scales: dict[str, torch.Tensor] = {}

    if old_scale_keys:
        format_name = _old_scaled_fp8_format(state_dict)
        storage_dtype = _torch_fp8_dtype(format_name)
        missing_weights: list[str] = []
        log.info("  detected %d Comfy old scaled-FP8 FLUX layers (%s)", len(old_scale_keys), format_name)
        for scale_key in old_scale_keys:
            layer = scale_key[: -len(_OLD_FP8_WEIGHT_SCALE_SUFFIX)]
            weight_key = f"{layer}.weight"
            if weight_key not in state_dict:
                missing_weights.append(weight_key)
                continue
            scales[weight_key] = state_dict[scale_key].detach().cpu()
            if state_dict[weight_key].dtype != storage_dtype:
                state_dict[weight_key] = state_dict[weight_key].to(storage_dtype).contiguous()
        if missing_weights:
            raise ValueError(
                "FLUX checkpoint has old scaled-FP8 metadata but missing weights; "
                f"first missing keys: {missing_weights[:3]}"
            )

    if quant_keys:
        log.info("  detected %d Comfy quantized FLUX layers", len(quant_keys))
        for quant_key in quant_keys:
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
            storage_dtype = _torch_fp8_dtype(format_name)
            if state_dict[weight_key].dtype != storage_dtype:
                state_dict[weight_key] = state_dict[weight_key].to(storage_dtype).contiguous()
            scales[weight_key] = scale.detach().cpu()

    if scales:
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

    return scales


def _convert_flux_weight_scales_to_diffusers(
    source_scales: dict[str, torch.Tensor],
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if not source_scales:
        return {}

    converted: dict[str, torch.Tensor] = {}
    remaining = set(source_scales)

    def add(dst: str, src: str) -> None:
        if src in source_scales:
            converted[dst] = source_scales[src]
            remaining.discard(src)

    def block_count(prefix: str) -> int:
        indexes = [int(k.split(".", 2)[1]) for k in state_dict if k.startswith(prefix)]
        return max(indexes) + 1 if indexes else 0

    add("time_text_embed.timestep_embedder.linear_1.weight", "time_in.in_layer.weight")
    add("time_text_embed.timestep_embedder.linear_2.weight", "time_in.out_layer.weight")
    add("time_text_embed.text_embedder.linear_1.weight", "vector_in.in_layer.weight")
    add("time_text_embed.text_embedder.linear_2.weight", "vector_in.out_layer.weight")
    add("time_text_embed.guidance_embedder.linear_1.weight", "guidance_in.in_layer.weight")
    add("time_text_embed.guidance_embedder.linear_2.weight", "guidance_in.out_layer.weight")
    add("context_embedder.weight", "txt_in.weight")
    add("x_embedder.weight", "img_in.weight")

    for i in range(block_count("double_blocks.")):
        block_prefix = f"transformer_blocks.{i}."
        add(f"{block_prefix}norm1.linear.weight", f"double_blocks.{i}.img_mod.lin.weight")
        add(f"{block_prefix}norm1_context.linear.weight", f"double_blocks.{i}.txt_mod.lin.weight")
        for dst in ("to_q", "to_k", "to_v"):
            add(f"{block_prefix}attn.{dst}.weight", f"double_blocks.{i}.img_attn.qkv.weight")
        for dst in ("add_q_proj", "add_k_proj", "add_v_proj"):
            add(f"{block_prefix}attn.{dst}.weight", f"double_blocks.{i}.txt_attn.qkv.weight")
        add(f"{block_prefix}ff.net.0.proj.weight", f"double_blocks.{i}.img_mlp.0.weight")
        add(f"{block_prefix}ff.net.2.weight", f"double_blocks.{i}.img_mlp.2.weight")
        add(f"{block_prefix}ff_context.net.0.proj.weight", f"double_blocks.{i}.txt_mlp.0.weight")
        add(f"{block_prefix}ff_context.net.2.weight", f"double_blocks.{i}.txt_mlp.2.weight")
        add(f"{block_prefix}attn.to_out.0.weight", f"double_blocks.{i}.img_attn.proj.weight")
        add(f"{block_prefix}attn.to_add_out.weight", f"double_blocks.{i}.txt_attn.proj.weight")

    for i in range(block_count("single_blocks.")):
        block_prefix = f"single_transformer_blocks.{i}."
        add(f"{block_prefix}norm.linear.weight", f"single_blocks.{i}.modulation.lin.weight")
        for dst in ("attn.to_q", "attn.to_k", "attn.to_v", "proj_mlp"):
            add(f"{block_prefix}{dst}.weight", f"single_blocks.{i}.linear1.weight")
        add(f"{block_prefix}proj_out.weight", f"single_blocks.{i}.linear2.weight")

    add("proj_out.weight", "final_layer.linear.weight")
    add("norm_out.linear.weight", "final_layer.adaLN_modulation.1.weight")

    if remaining:
        raise ValueError(
            "unsupported scaled-FP8 FLUX weight scale keys after layout conversion; "
            f"first keys: {list(sorted(remaining))[:5]}"
        )

    return converted


def _model_param_bytes(module) -> int:
    total = 0
    for p in module.parameters():
        total += p.numel() * p.element_size()
    for b in module.buffers():
        total += b.numel() * b.element_size()
    return total


def _fp8_param_bytes(module) -> int:
    """Sum of param bytes that are currently FP8 — these are the ones that grow
    when pre-cast to bf16 (1 → 2 bytes per element)."""
    total = 0
    for p in module.parameters():
        if p.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            total += p.numel() * p.element_size()
    return total


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


def _pre_cast_fp8_linears_to_bf16(transformer, dtype: torch.dtype) -> tuple[int, float]:
    """ComfyUI-style: convert FP8 Linear weights to bf16 in-place, apply scale,
    drop monkey-patched forward. Subsequent forwards use vanilla nn.Linear.forward.

    Returns (modules_converted, gb_added).
    """
    converted = 0
    bytes_added = 0
    for module in transformer.modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        weight = module.weight
        if weight.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
            continue

        device = weight.device
        # FP8 → bf16 (GPU cast, fast). Apply per-layer scale if present.
        bf16_weight = weight.data.to(device=device, dtype=dtype)
        scale = getattr(module, "_kraken_weight_scale", None)
        if scale is not None:
            bf16_weight = bf16_weight * scale.to(device=device, dtype=dtype)

        old_bytes = weight.numel() * weight.element_size()
        new_bytes = bf16_weight.numel() * bf16_weight.element_size()
        bytes_added += new_bytes - old_bytes

        # Replace the parameter (drops the FP8 storage)
        module.weight = torch.nn.Parameter(bf16_weight, requires_grad=False)

        # Drop the scale buffer — it's now baked into the weight.
        if "_kraken_weight_scale" in module._buffers:
            del module._buffers["_kraken_weight_scale"]

        # Remove our instance-level forward override so super().forward() (the
        # fast standard PyTorch path) runs from here on.
        for flag in ("_kraken_scaled_fp8_forward", "_kraken_fp8_forward"):
            if hasattr(module, flag):
                try:
                    delattr(module, flag)
                except Exception:
                    pass
        if "forward" in module.__dict__:
            del module.__dict__["forward"]

        converted += 1

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return converted, bytes_added / 1024**3


def _scaled_linear_forward(module, x):
    weight = module.weight
    if weight.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        weight = weight.to(dtype=x.dtype)
    scale = module._kraken_weight_scale
    # Skip the no-op `.to()` call when the buffer is already on the right device/
    # dtype — accelerate's offload hook moves persistent buffers in lock-step with
    # the module, so this is the common case during sampling.
    if scale.device != weight.device or scale.dtype != weight.dtype:
        scale = scale.to(device=weight.device, dtype=weight.dtype)
    return torch.nn.functional.linear(x, weight * scale, module.bias)


def _fp8_linear_forward(module, x):
    weight = module.weight
    if weight.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        weight = weight.to(dtype=x.dtype)
    bias = module.bias
    if bias is not None and bias.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        bias = bias.to(dtype=x.dtype)
    return torch.nn.functional.linear(x, weight, bias)


def _fp8_linear_weight_names(module) -> set[str]:
    names: set[str] = set()
    for module_name, child in module.named_modules():
        if not isinstance(child, torch.nn.Linear):
            continue
        weight = getattr(child, "weight", None)
        if weight is not None and weight.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            names.add(f"{module_name}.weight" if module_name else "weight")
    return names


def _ensure_module_dtype_preserving_fp8(
    module,
    dtype: torch.dtype,
    label: str,
    *,
    preserve_fp8_linears_only: bool = False,
) -> None:
    changed = 0
    preserve_names = _fp8_linear_weight_names(module) if preserve_fp8_linears_only else None
    for name, tensor in list(module.named_parameters()) + list(module.named_buffers()):
        if not tensor.is_floating_point() or tensor.dtype == dtype:
            continue
        if tensor.dtype in (torch.float8_e4m3fn, torch.float8_e5m2) and (
            preserve_names is None or name in preserve_names
        ):
            continue
        tensor.data = tensor.data.to(dtype=dtype)
        changed += 1
    if changed:
        log.info("%s: coerced %d non-FP8 tensors to %s", label, changed, dtype)


def _patch_fp8_linears(module, label: str) -> int:
    import types

    patched = 0
    for child in module.modules():
        if not isinstance(child, torch.nn.Linear):
            continue
        weight = getattr(child, "weight", None)
        if weight is None or weight.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
            continue
        if getattr(child, "_kraken_fp8_forward", False):
            continue
        child.forward = types.MethodType(_fp8_linear_forward, child)
        child._kraken_fp8_forward = True
        patched += 1
    if patched:
        log.info("%s: patched %d FP8 Linear modules for runtime casting", label, patched)
    return patched


def _apply_flux_weight_scales(transformer, weight_scales: dict[str, torch.Tensor]) -> None:
    if not weight_scales:
        return

    import types

    attached = 0
    for weight_key, scale in weight_scales.items():
        if not weight_key.endswith(".weight"):
            raise ValueError(f"invalid FLUX scale target {weight_key!r}; expected a .weight key")
        module_name = weight_key[: -len(".weight")]
        try:
            module = transformer.get_submodule(module_name)
        except AttributeError as e:
            raise ValueError(f"scaled-FP8 FLUX target module not found: {module_name}") from e
        if not isinstance(module, torch.nn.Linear):
            raise ValueError(f"scaled-FP8 FLUX target is not a Linear module: {module_name}")
        # Pre-cast the scale to bf16 (compute dtype) so the per-forward `.to()` in
        # _scaled_linear_forward is a no-op on the hot path.
        module.register_buffer("_kraken_weight_scale", scale.to(torch.bfloat16), persistent=False)
        if not getattr(module, "_kraken_scaled_fp8_forward", False):
            module.forward = types.MethodType(_scaled_linear_forward, module)
            module._kraken_scaled_fp8_forward = True
        attached += 1

    log.info("  attached scaled-FP8 weight scales to %d FLUX Linear modules", attached)


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

    # All-in-one checkpoints (ComfyUI "checkpoint" layout) bundle the text
    # encoders and VAE alongside the transformer. Kraken loads those components
    # separately from their own files, so after the `model.diffusion_model.`
    # strip these extra keys would otherwise survive into the diffusers
    # conversion as unexpected keys and corrupt the dtype/partition accounting.
    # UNet-only checkpoints carry none of these prefixes, so this is a no-op for
    # the normal FLUX path.
    _non_transformer_prefixes = ("text_encoders.", "vae.", "first_stage_model.", "conditioner.")
    n_before = len(state_dict)
    state_dict = {
        k: v for k, v in state_dict.items()
        if not k.startswith(_non_transformer_prefixes)
    }
    if len(state_dict) != n_before:
        log.info(
            "  dropped %d non-transformer keys from all-in-one checkpoint (loading TE/VAE separately)",
            n_before - len(state_dict),
        )

    source_weight_scales = _extract_comfy_scaled_fp8_scales(state_dict)
    if source_weight_scales:
        log.info("  preserving %d scaled-FP8 transformer scales for runtime Linear scaling", len(source_weight_scales))

    if _looks_like_diffusers_flux(state_dict):
        log.info("  detected Diffusers-format FLUX transformer checkpoint")
        return state_dict, source_weight_scales

    if _looks_like_bfl_flux(state_dict):
        weight_scales = _convert_flux_weight_scales_to_diffusers(source_weight_scales, state_dict)
        log.info("  converting %d BFL/Comfy FLUX transformer keys to Diffusers format", len(state_dict))
        return convert_flux_transformer_checkpoint_to_diffusers(state_dict), weight_scales

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


def unload(*, drop_te: bool = True) -> dict:
    """Tear down the cached pipeline and (optionally) the text-encoder cache.

    `drop_te=True` (default) is the user-facing semantics: clear_memory and
    arch switches that won't reuse the same TEs both want everything gone.
    `drop_te=False` is used INTERNALLY by `_ensure_pipeline` when it's about
    to rebuild the transformer but the next encode call will likely use the
    same text encoders — dropping them would force a wasteful 10 s disk reload.
    """
    global _pipeline, _pipeline_key, _loaded_loras, _img2img_pipe, _img2img_for
    _img2img_pipe = None
    _img2img_for = None
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
    if drop_te:
        _drop_te_cache()
    return {"had_pipeline": had, "loras_freed": n_loras}


# ---------- component loaders ----------

def _load_clip_l(path: Path, dtype: torch.dtype):
    from transformers import CLIPTextModel, CLIPTextConfig
    config = CLIPTextConfig.from_pretrained("openai/clip-vit-large-patch14")
    return stream_safetensors_into_meta(
        CLIPTextModel, config, path, dtype, "CLIP-L", key_prefix_strip="text_encoder."
    )


def _load_t5xxl(path: Path, dtype: torch.dtype, *, preserve_fp8: bool = False):
    from transformers import T5EncoderModel, T5Config
    config = T5Config.from_pretrained("google/t5-v1_1-xxl")
    return stream_safetensors_into_meta(
        T5EncoderModel,
        config,
        path,
        dtype,
        "T5-XXL",
        key_prefix_strip="text_encoder_2.",
        preserve_fp8=preserve_fp8,
        # T5 dequant from FP8 is the single biggest fixed cost on cold loads
        # (~5 GB of weights through the GPU cast loop). Cache the bf16 result
        # on disk so subsequent cold starts skip it entirely.
        use_bf16_cache=True,
    )


def _get_tokenizers():
    from transformers import CLIPTokenizer, T5TokenizerFast
    return (
        CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14"),
        T5TokenizerFast.from_pretrained("google/t5-v1_1-xxl"),
    )


# Persistent text-encoder cache (Phase 4). Earlier attempt failed because
# combining a 10 GB TE cache with a fully-CPU-resident model_cpu_offload
# transformer (22 GB) pushed peak RAM past Windows commit limits during
# transformer reload. With streaming offload (Phase 1-3) the transformer's
# CPU footprint drops to ~1.3 GB (only the streamed weights stay CPU-resident),
# so the TE cache + cloned-streamed-weights total ~11 GB — safe on a 64 GB box.
#
# Cache key is (clip_path, t5_path) so picking a different TE invalidates.
# Models live in CPU RAM between gens; encode_prompt moves them to GPU briefly
# then back. Tokenizers are cheap so we cache them too.
_te_cache: dict[tuple[str, str], dict] = {}

# Conditioning-output cache (the cheap, safe half of "Phase 4"). Caches the
# small ENCODED tensors — prompt_embeds (~4 MB) + pooled (~3 KB) — NOT the
# 10 GB encoder weights. This mirrors ComfyUI's CLIPTextEncode node caching:
# a warm gen with an unchanged prompt + encoders skips the 9.6 GB T5 reload and
# the encode forward pass entirely. Key includes the encoder paths so swapping
# encoders invalidates. Bounded to avoid unbounded RAM growth.
_cond_cache: "collections.OrderedDict[tuple, tuple[torch.Tensor, torch.Tensor]]" = (
    collections.OrderedDict()
)
_COND_CACHE_MAX = 16


def _drop_te_cache() -> None:
    """Clear the persistent text-encoder cache. Called from `unload()`, from
    the /api/clear_memory endpoint, and on arch switches."""
    global _te_cache
    cleared_cond = len(_cond_cache)
    _cond_cache.clear()
    if _te_cache:
        n = len(_te_cache)
        _te_cache.clear()
        gc.collect()
        log.info("dropped %d cached text-encoder set(s) from CPU RAM", n)
    elif cleared_cond:
        gc.collect()
    if cleared_cond:
        log.info("dropped %d cached conditioning tensor(s)", cleared_cond)


def _encode_flux_prompt(
    te1: str | None,
    te2: str | None,
    prompt: str,
    *,
    dtype: torch.dtype,
    max_sequence_length: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    te1_path = _resolve("text_encoders", te1) if te1 else None
    te2_path = _resolve("text_encoders", te2) if te2 else None
    if not te1_path or not te2_path:
        raise ValueError("FLUX requires two text encoders. Slot 1: clip_l.safetensors. Slot 2: t5xxl_fp16.safetensors (or fp8 for less VRAM).")

    # Conditioning-output cache check. On a warm gen with an identical prompt +
    # encoders we return the previously-encoded tensors and skip the entire
    # 9.6 GB T5 reload + encode (the single biggest fixed per-gen cost). This is
    # what gives ComfyUI its big cold->warm drop on a repeated prompt.
    cache_key = (str(te1_path), str(te2_path), prompt or "", max_sequence_length, str(dtype))
    cached = _cond_cache.get(cache_key)
    if cached is not None:
        _cond_cache.move_to_end(cache_key)
        log.info("FLUX conditioning cache HIT — skipping text-encoder load + encode")
        return cached[0], cached[1]

    t_enc = time.time()

    # Phase 4 attempt (kept here for future, currently disabled): cache CLIP+T5
    # in CPU RAM between gens. On this hardware it crashed the sidecar via
    # Windows commit-limit exhaustion (TE cache 10 GB + pinned streamed weights
    # 1.3 GB + transformer mmap 22 GB + Python heap > available commit). The
    # T5 bf16 disk cache (load_utils.bf16_cache_path) already amortizes the
    # heaviest cost; per-gen TE load is ~5–8 s with that on, not worth the
    # OOM risk for a public-release default. Leave the cache infra in place
    # (`_te_cache`, `_drop_te_cache`, the drop_te=False unload flag) for a
    # future opt-in setting once we have a safer commit-aware budget check.
    log.info("encoding FLUX prompt with staged text encoders")
    text_encoder = _load_clip_l(te1_path, dtype)
    text_encoder_2 = _load_t5xxl(te2_path, dtype, preserve_fp8=False)
    ensure_module_dtype(text_encoder, dtype, "CLIP-L")
    ensure_module_dtype(text_encoder_2, dtype, "T5-XXL")
    tokenizer, tokenizer_2 = _get_tokenizers()

    device = torch.device(_device())
    text_encoder = text_encoder.to(device=device, dtype=dtype)
    text_encoder_2 = text_encoder_2.to(device=device, dtype=dtype)

    prompts = [prompt or ""]
    with torch.inference_mode():
        clip_inputs = tokenizer(
            prompts,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        )
        pooled_prompt_embeds = text_encoder(
            clip_inputs.input_ids.to(device),
            output_hidden_states=False,
        ).pooler_output.to(dtype=dtype, device="cpu")

        t5_inputs = tokenizer_2(
            prompts,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        prompt_embeds = text_encoder_2(
            t5_inputs.input_ids.to(device),
            output_hidden_states=False,
        )[0].to(dtype=dtype, device="cpu")

    del text_encoder, text_encoder_2, tokenizer, tokenizer_2
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    log.info(
        "encoded FLUX prompt in %.1fs; text encoders unloaded before transformer load",
        time.time() - t_enc,
    )

    _cond_cache[cache_key] = (prompt_embeds, pooled_prompt_embeds)
    _cond_cache.move_to_end(cache_key)
    while len(_cond_cache) > _COND_CACHE_MAX:
        _cond_cache.popitem(last=False)

    return prompt_embeds, pooled_prompt_embeds


# ---------- pipeline lifecycle ----------

def _ensure_pipeline(diffusion_model: str, vae_name: str | None, te1: str | None, te2: str | None):
    global _pipeline, _pipeline_key

    tx_path  = _resolve("diffusion_models", diffusion_model)
    vae_path = _resolve("vae", vae_name) if vae_name else None

    if not tx_path:
        raise FileNotFoundError(f"diffusion model not found: {diffusion_model}")
    if not vae_path:
        raise ValueError("FLUX requires a VAE — pick `QWEN/ae.safetensors` (the FLUX/Qwen-shared VAE) from the VAE dropdown.")

    key = (str(tx_path), str(vae_path))
    if _pipeline is not None and _pipeline_key == key:
        return _pipeline

    # Rebuilding pipeline (arch switch or first load). Don't drop the TE cache
    # — same gen might use the same text encoders with a different transformer
    # (very common: switch between FP32 and FP8 FLUX builds while keeping the
    # same t5xxl_fp16/clip_l). The TE cache invalidates on (clip_path, t5_path)
    # mismatch anyway, so stale entries can't be used wrongly.
    unload(drop_te=False)

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
    tx_sd, transformer_weight_scales = _prepare_flux_transformer_state_dict(raw_tx_sd, dtype=dtype)
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
    _ensure_module_dtype_preserving_fp8(transformer, dtype, "transformer", preserve_fp8_linears_only=True)
    _apply_flux_weight_scales(transformer, transformer_weight_scales)

    # Swap every nn.Linear in the transformer to a StreamingLinear. This is the
    # foundation for the Forge/Fooocus-style offload that replaces diffusers'
    # `enable_model_cpu_offload`. StreamingLinear's forward unifies:
    #   - GPU-resident plain weights (vanilla fast path, zero overhead)
    #   - GPU-resident FP8 weights (cast per call — Ampere has no FP8 matmul)
    #   - GPU-resident scaled-FP8 weights (cast + multiply by `_kraken_weight_scale`)
    #   - CPU-resident weights of any of the above (H2D copy per forward)
    # Must run AFTER `_apply_flux_weight_scales` so scale buffers carry over via
    # `from_linear`. The legacy `_scaled_linear_forward` monkey-patches on plain
    # nn.Linear instances are obsolete once the swap happens — StreamingLinear
    # handles all four cases natively.
    from pipelines.streaming_linear import swap_linears
    n_swapped = swap_linears(transformer)
    if n_swapped:
        log.info("  swapped %d Linear modules to StreamingLinear", n_swapped)

    # Patch B (QKV fusion + bf16-native RoPE).
    from pipelines.kraken_flux_attn import fuse_attention_qkv, install_kraken_attn_processor
    n_fused = fuse_attention_qkv(transformer)
    install_kraken_attn_processor(transformer)
    if n_fused:
        log.info("  fused QKV in %d attention modules + installed bf16-RoPE processor", n_fused)

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
        text_encoder=None,
        text_encoder_2=None,
        tokenizer=None,
        tokenizer_2=None,
        scheduler=scheduler,
    )
    transformer._kraken_compute_dtype = dtype  # type: ignore[attr-defined]
    pipe._kraken_compute_dtype = dtype  # type: ignore[attr-defined]

    # ---------- Fast Inference: no offload hooks, FP8 weights on GPU ----------
    # ComfyUI's path on Ampere (3090 etc.). They skip accelerate-style offload
    # hooks entirely; per-forward hook dispatch from diffusers' enable_model_cpu_offload
    # adds up over 28 steps × 504 Linears. We keep FP8 weights in place (no
    # pre-cast doubling), push the pipeline onto GPU once, and the patched
    # _scaled_linear_forward still does the cast — but without the hook overhead
    # in the call path.
    #
    # Budget for this mode: actual model in-memory size + VAE + activation buffer.
    # A scaled-FP8 FLUX file that's 22 GB on disk lands at ~11-14 GB in memory.
    import config_store
    fast_mode = config_store.get("performance.flux_fast_inference", "auto")
    buffer_gb = float(config_store.get("performance.flux_fast_inference_buffer_gb", 2.0))

    cur_model_gb = _model_param_bytes(transformer) / 1024**3
    fp8_param_gb = _fp8_param_bytes(transformer) / 1024**3
    free_vram_now = _free_vram_gb()
    needed_gb = cur_model_gb + vae_gb + buffer_gb

    use_fast = False
    if fast_mode == "on":
        use_fast = True
        log.info("FLUX fast-inference: forced ON (setting=on)")
    elif fast_mode == "auto":
        if free_vram_now is not None and free_vram_now >= needed_gb:
            use_fast = True
        log.info(
            "FLUX fast-inference auto-check: free=%s GB · need=%.2f GB "
            "(model %.2f [FP8 portion %.2f] + vae %.2f + activations %.2f) -> %s",
            f"{free_vram_now:.2f}" if free_vram_now is not None else "?",
            needed_gb, cur_model_gb, fp8_param_gb, vae_gb, buffer_gb,
            "USE FAST (no offload)" if use_fast else "FALL BACK to offload",
        )

    if use_fast:
        try:
            pipe.to(_device())
            pipe._kraken_offload_strategy = "no_offload_fast"  # type: ignore[attr-defined]
            log.info("FLUX fast: pipeline on %s, no diffusers offload hooks", _device())

            # torch.compile the transformer's repeated blocks. Published wins:
            # PyTorch blog reports 1.5× on H100, diffusers docs claim 19% over
            # xFormers on Ampere with channels_last + max-autotune.
            #
            # ONLY safe on the fast (fully-resident) path. The streaming path
            # uses StreamingLinear's prefetch handoff via `self.__dict__` which
            # would graph-break the compiler. On fast mode the prefetch slots
            # are never populated so the forward reduces to plain `F.linear`
            # (with optional FP8 cast for scaled checkpoints), which compiles
            # cleanly via diffusers' `compile_repeated_blocks` regional path.
            #
            # User opt-out via settings.performance.flux_compile = "off". Default
            # "auto" enables compile when fast mode is selected. First-gen cost
            # is ~5-10 s; all subsequent gens reap the win.
            # Default "off" — measured net-negative on the StreamingLinear arch.
            # Users with a pure-resident model (no streaming) and patience for the
            # first-gen compile cost can override via Settings → Performance.
            compile_mode = config_store.get("performance.flux_compile", "off")
            if compile_mode == "on":
                # Pre-flight: Triton is required for the Inductor backend to
                # generate GPU kernels. PyTorch on Windows doesn't ship Triton
                # by default — users have to install `triton-windows` (community
                # build) themselves. If it's missing, skip silently rather than
                # crashing the gen with a deferred RuntimeError mid-forward.
                try:
                    import triton  # noqa: F401
                    has_triton = True
                except ImportError:
                    has_triton = False

                if not has_triton:
                    log.info(
                        "FLUX fast: torch.compile skipped — Triton not installed. "
                        "Install `triton-windows` (pip) to enable; expected ~15%% per-step win on Ampere."
                    )
                    pipe._kraken_compiled = False  # type: ignore[attr-defined]
                else:
                    try:
                        # Bump Dynamo's recompile cache. FLUX has 57 transformer
                        # blocks; with our StreamingLinear's dynamic-attribute
                        # dispatch + the mix of bf16/FP8 weights in scaled-FP8
                        # checkpoints, dynamo can hit the default 8-recompile
                        # ceiling fast. 256 covers any reasonable per-block
                        # specialization without false-positives on bugs.
                        torch._dynamo.config.cache_size_limit = 256

                        transformer.to(memory_format=torch.channels_last)
                        # `compile_repeated_blocks` compiles each FluxTransformerBlock
                        # + FluxSingleTransformerBlock independently — regional
                        # compilation. Reduces compile time dramatically (each block
                        # is small) while preserving runtime speedups. `dynamic=False`
                        # because shapes are fixed within a gen. `fullgraph=False`
                        # (default) so any per-block ops that can't be traced fall
                        # back to eager without aborting the whole pipeline.
                        transformer.compile_repeated_blocks(fullgraph=False, dynamic=False)
                        pipe._kraken_compiled = True  # type: ignore[attr-defined]
                        log.info("FLUX fast: torch.compile applied to FluxTransformerBlock / FluxSingleTransformerBlock (~57 blocks)")
                    except Exception as e:
                        pipe._kraken_compiled = False  # type: ignore[attr-defined]
                        log.warning("torch.compile setup failed (%s) — continuing without compile (no perf loss vs baseline)", e)
            else:
                log.info("FLUX fast: torch.compile disabled by settings.performance.flux_compile=off")
        except Exception as e:
            log.warning("FLUX no-offload placement failed (%s); falling back to model_cpu_offload", e)
            from pipelines.offload import apply as apply_offload
            apply_offload(pipe, "model_cpu_offload", device=_device())
            pipe._kraken_offload_strategy = "model_cpu_offload"
    else:
        # ---------- Streaming offload (Forge/Fooocus-style) ---------------
        # Diffusers' enable_model_cpu_offload moves entire submodules CPU↔GPU per
        # forward — fine for SDXL, ~3× too slow for FLUX. We instead partition the
        # transformer at Linear granularity:
        #   - Always-resident on GPU: VAE + small transformer params (embedders,
        #     norms, proj_out — anything not inside a transformer_block).
        #   - On GPU as budget allows: full transformer_blocks (loaded in order
        #     until VRAM budget is exhausted).
        #   - Streamed from CPU: remaining transformer_blocks. Every StreamingLinear
        #     inside gets `_kraken_stream_weights=True`; its forward copies the
        #     weight to GPU on demand with `non_blocking=True`.
        #
        # Activation buffer: empirically ~1.5 GB at 1024² × 28 steps for FLUX.
        # We reserve a bit more (configurable via `flux_fast_inference_buffer_gb`,
        # default 2.0) to cover one-off spikes.
        from pipelines.streaming_linear import apply_streaming, count_streaming
        pipe.vae.to(_device())
        free_vram_now2 = _free_vram_gb() or 24.0
        # Re-measure after VAE move so the budget is based on what's actually free.
        free_vram_after_vae = _free_vram_gb() or (free_vram_now2 - vae_gb)
        import sys
        # On Windows, subtract an additional 1.5 GB to account for WDDM / desktop VRAM paging overhead.
        windows_margin = 1.5 if sys.platform == "win32" else 0.0
        transformer_budget_gb = max(0.5, free_vram_after_vae - buffer_gb - windows_margin)
        transformer_budget_bytes = int(transformer_budget_gb * 1024**3)
        log.info(
            "FLUX streaming setup: free VRAM=%.2f GB after VAE · transformer budget=%.2f GB "
            "(reserved %.2f GB for activations)",
            free_vram_after_vae, transformer_budget_gb, buffer_gb,
        )
        summary = apply_streaming(
            transformer,
            budget_bytes=transformer_budget_bytes,
            device=_device(),
            # Pinned host memory + dedicated mover CUDA stream let each
            # CPU-resident weight's H2D copy overlap with the previous Linear's
            # compute — matches Forge's mechanism, brings warm steps from
            # ~2 s to ~1.7 s on FLUX-dev streaming a few blocks.
            pin_memory=True,
            label="FLUX transformer",
        )
        n_total, n_streamed = count_streaming(transformer)
        strategy_label = "streaming_linear" if summary["blocks_on_cpu"] else "no_offload_full"
        pipe._kraken_offload_strategy = strategy_label  # type: ignore[attr-defined]
        log.info(
            "FLUX %s: %d/%d Linears stream from CPU; %d blocks resident on GPU, %d on CPU",
            strategy_label, n_streamed, n_total,
            summary["blocks_on_gpu"], summary["blocks_on_cpu"],
        )
        # `pipe._execution_device` is a property that walks components looking
        # for the first non-CPU tensor; with our partition, non-block params
        # (including embedders) are on cuda so it correctly returns cuda.

    pipe.set_progress_bar_config(disable=True)
    # VAE tiling causes checkerboard garbage on FLUX decode at 512/1024 — skip it.

    # First Block Cache via para-attn. Diffusion residuals between consecutive
    # sampling steps are typically very similar — when the residual diff is
    # below `flux_fbcache_threshold`, this skips the bulk of the transformer
    # forward and reuses the prior step's residual. Published 1.5-2× effective
    # speedup on FLUX. Threshold default 0.08; lower = more caching = faster
    # but higher quality risk. Disable with performance.flux_fbcache = "off".
    # FBCache default flipped to "off" 2026-05-22 after benchmarking — see
    # config_store comment + Documentation/FLUX-SPEED-WIP.md.
    fbcache_mode = config_store.get("performance.flux_fbcache", "off")
    if fbcache_mode == "on":
        try:
            # Patch para-attn for diffusers 0.38's two-arg single_transformer_block
            # API. para-attn 0.3.38 was written against the older signature; without
            # this patch the cache hits TypeError mid-forward.
            from pipelines.kraken_fbcache import install_fbcache_patch
            install_fbcache_patch()
            from para_attn.first_block_cache.diffusers_adapters.flux import apply_cache_on_pipe
            threshold = float(config_store.get("performance.flux_fbcache_threshold", 0.08))
            apply_cache_on_pipe(pipe, residual_diff_threshold=threshold)
            pipe._kraken_fbcache_threshold = threshold  # type: ignore[attr-defined]
            log.info("FLUX First Block Cache: enabled (threshold=%.3f)", threshold)
        except ImportError:
            log.info("FLUX First Block Cache: para-attn not installed — skipping (pip install para-attn to enable)")
        except Exception as e:
            log.warning("FLUX First Block Cache setup failed (%s) — continuing without it", e)

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

        allocated = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
        reserved = torch.cuda.memory_reserved() / 1024**3 if torch.cuda.is_available() else 0.0
        log.info("Step %d progress: PyTorch VRAM allocated=%.2f GB, reserved=%.2f GB", step + 1, allocated, reserved)

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


_img2img_pipe = None
_img2img_for = None  # id() of the txt2img pipe its components came from


def _build_img2img(pipe):
    """Build/reuse a KrakenFluxImg2ImgPipeline sharing `pipe`'s loaded components.

    Same transformer (StreamingLinear + scaled-FP8 + attn processor), VAE and
    scheduler — zero extra VRAM. Cached until the underlying transformer changes.
    """
    global _img2img_pipe, _img2img_for
    from pipelines.kraken_flux_pipeline import KrakenFluxImg2ImgPipeline

    if _img2img_pipe is not None and _img2img_for == id(pipe):
        return _img2img_pipe

    i2i = KrakenFluxImg2ImgPipeline(
        transformer=pipe.transformer,
        vae=pipe.vae,
        text_encoder=None,
        text_encoder_2=None,
        tokenizer=None,
        tokenizer_2=None,
        scheduler=pipe.scheduler,
    )
    i2i._kraken_compute_dtype = getattr(pipe, "_kraken_compute_dtype", None)
    i2i.set_progress_bar_config(disable=True)
    _img2img_pipe = i2i
    _img2img_for = id(pipe)
    return i2i


def make_refiner(pipe, prompt_embeds, pooled_prompt_embeds, *, guidance: float, true_cfg: float = 1.0):
    """Return a USDU tile-refiner closure for FLUX.

    Standard img2img: VAE-encode the tile, noise to `denoise`, run the same FLUX
    denoise loop. Reuses the gen's prompt embeds — nothing FLUX-specific, exactly
    like ComfyUI's KSampler with a checkpoint loader.
    """
    i2i = _build_img2img(pipe)
    offload_strategy = getattr(pipe, "_kraken_offload_strategy", "model_cpu_offload")
    gen_device = _generator_device(pipe, offload_strategy)
    exec_device = pipe._execution_device
    pe = prompt_embeds.to(device=exec_device)
    ppe = pooled_prompt_embeds.to(device=exec_device)

    def refine(tile: Image.Image, *, steps: int, denoise: float, seed: int) -> Image.Image:
        gen = torch.Generator(device=gen_device).manual_seed(int(seed) & 0x7FFFFFFF)
        out = i2i(
            prompt=None,
            prompt_embeds=pe,
            pooled_prompt_embeds=ppe,
            image=tile,
            strength=float(denoise),
            num_inference_steps=int(steps),
            guidance_scale=guidance,
            true_cfg_scale=true_cfg,
            generator=gen,
            output_type="pil",
        )
        return out.images[0]

    return refine


def run(job) -> dict:
    from pipelines import sdxl as sdxl_mod
    sdxl_mod.unload()
    gc.collect()

    p = job.params

    te = p.get("text_encoders") or []
    te1 = te[0] if len(te) >= 1 else None
    te2 = te[1] if len(te) >= 2 else None

    dtype = _dtype()
    _t_enc = time.time()
    prompt_embeds_cpu, pooled_prompt_embeds_cpu = _encode_flux_prompt(
        te1,
        te2,
        p.get("prompt", ""),
        dtype=dtype,
    )
    _enc_s = time.time() - _t_enc
    _t_setup = time.time()
    pipe = _ensure_pipeline(p["diffusion_model"], p.get("vae"), te1, te2)
    _apply_loras(pipe, p.get("loras") or [])
    log.info(
        "FLUX run stages: text-encode=%.1fs · pipeline-ensure+loras=%.1fs (sampling follows)",
        _enc_s, time.time() - _t_setup,
    )

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
    base_stem = build_output_stem(p, job.id)

    saved: list[dict] = []
    offload_strategy = getattr(pipe, "_kraken_offload_strategy", "model_cpu_offload")
    gen_device = _generator_device(pipe, offload_strategy)
    exec_device = pipe._execution_device
    prompt_embeds = prompt_embeds_cpu.to(device=exec_device)
    pooled_prompt_embeds = pooled_prompt_embeds_cpu.to(device=exec_device)
    del prompt_embeds_cpu, pooled_prompt_embeds_cpu
    for i in range(count):
        if job.cancel.is_set():
            break
        per_seed = (seed if seed is not None else torch.seed()) + i
        generator = torch.Generator(device=gen_device).manual_seed(int(per_seed) & 0x7FFFFFFF)
        result = pipe(
            prompt=None,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
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
        save_png_with_metadata(img, fpath, p, int(per_seed))
        entry = {
            "path": str(fpath), "filename": fname, "seed": int(per_seed),
            "width": width, "height": height,
        }

        # Optional upscale — ESRGAN (fast) or USDU (tile + img2img refine, same model).
        if p.get("upscale_enabled") and p.get("upscale_model"):
            mode = (p.get("upscale_mode") or "esrgan").lower()
            factor = float(p.get("upscale_factor", 2.0))
            try:
                if mode == "usdu":
                    from pipelines import usdu
                    refiner = make_refiner(
                        pipe, prompt_embeds, pooled_prompt_embeds,
                        guidance=embedded_guidance, true_cfg=true_cfg,
                    )
                    res = usdu.run_usdu(
                        img, max(8, round(width * factor)), max(8, round(height * factor)),
                        upscale_model=p["upscale_model"], refine_fn=refiner,
                        steps=int(p.get("upscale_steps") or 20),
                        denoise=float(p.get("upscale_denoise") or 0.2),
                        seed=int(per_seed),
                        tile_size=p.get("upscale_tile_size") or None,
                        progress=usdu.job_progress(job, i, count),
                    )
                    up = res["image"]
                else:
                    from pipelines import upscale_esrgan
                    up = upscale_esrgan.upscale(img, p["upscale_model"], factor)
                up_path = out_dir / f"{base_stem}-{i:02d}-up.png"
                save_png_with_metadata(up, up_path, p, int(per_seed))
                entry["upscaled_path"] = str(up_path)
                img = up  # preview the upscaled version
            except Exception as e:
                log.warning("upscale (%s) failed: %s", mode, e)

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
