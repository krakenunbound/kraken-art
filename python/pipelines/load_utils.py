"""Shared model-loading utilities for all Kraken Art pipelines.

Design goal: work on any GPU/RAM budget without loading full checkpoint dicts
into memory. One tensor at a time via safetensors streaming; strip accelerate
hooks on unload so arch switches don't leak or crash.
"""
from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open

log = logging.getLogger("kraken.load")

_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)
_TORCH_TO_ML_FP8: dict = {}


def file_size_gb(path: Path) -> float:
    return path.stat().st_size / 1024**3


# Process-wide flag: once we've confirmed the CUDA FP8 cast works on this machine,
# we stay on the fast path for the rest of the session. If it ever raises, we
# permanently demote to the numpy fallback to avoid retrying a crashy path.
_GPU_FP8_CAST_OK: bool | None = None  # None = untried, True = works, False = use numpy


def _dequant_fp8_cpu_via_numpy(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Slow but stable: round-trip FP8 → numpy (via ml_dtypes) → torch. Used as
    fallback when the CUDA cast path is unavailable or has failed previously."""
    import ml_dtypes

    if not _TORCH_TO_ML_FP8:
        _TORCH_TO_ML_FP8[torch.float8_e4m3fn] = ml_dtypes.float8_e4m3fn
        _TORCH_TO_ML_FP8[torch.float8_e5m2] = ml_dtypes.float8_e5m2

    ml_dt = _TORCH_TO_ML_FP8[tensor.dtype]
    shape = tensor.shape
    raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().copy()
    del tensor
    fp8_view = raw.view(ml_dt).reshape(tuple(shape))
    f32 = fp8_view.astype(np.float32)
    del raw, fp8_view
    if dtype == torch.bfloat16:
        bf16_np = f32.astype(ml_dtypes.bfloat16)
        del f32
        u16 = bf16_np.view(np.uint16)
        del bf16_np
        return torch.from_numpy(u16.copy()).view(torch.bfloat16)
    if dtype == torch.float16:
        out = torch.from_numpy(f32.astype(np.float16).copy())
        del f32
        return out
    out = torch.from_numpy(f32.copy())
    del f32
    return out


def dequant_fp8_tensor(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """FP8 → `dtype`. Fast path uses PyTorch's CUDA cast kernel (the same one the
    FLUX transformer forward uses thousands of times per gen — proven stable).
    Falls back to a CPU/numpy round-trip via ml_dtypes if CUDA is unavailable
    or if the GPU cast ever raises on this machine."""
    global _GPU_FP8_CAST_OK
    if tensor.dtype not in _FP8_DTYPES:
        return tensor

    if (
        _GPU_FP8_CAST_OK is not False
        and torch.cuda.is_available()
        and dtype in (torch.bfloat16, torch.float16, torch.float32)
    ):
        try:
            # CPU → GPU FP8 → GPU dtype → CPU. The cast itself is the cheap part;
            # the transfers dominate. ~10 ms for a 50 MB tensor on PCIe 4.0.
            gpu_fp8 = tensor.detach().to(device="cuda", non_blocking=True)
            gpu_cast = gpu_fp8.to(dtype=dtype)
            del gpu_fp8
            out = gpu_cast.cpu()
            del gpu_cast
            if _GPU_FP8_CAST_OK is None:
                _GPU_FP8_CAST_OK = True
                log.info("FP8 dequant: using CUDA cast (fast path)")
            return out
        except Exception as e:
            log.warning("CUDA FP8 cast failed (%s); falling back to numpy path for the rest of this session", e)
            _GPU_FP8_CAST_OK = False
            # fall through to CPU path

    return _dequant_fp8_cpu_via_numpy(tensor, dtype)


def ensure_module_dtype(module, dtype: torch.dtype, label: str = "") -> int:
    changed = 0
    for tensor in list(module.parameters()) + list(module.buffers()):
        if tensor.is_floating_point() and tensor.dtype != dtype:
            tensor.data = tensor.data.to(dtype=dtype)
            changed += 1
    if changed and label:
        log.info("%s: coerced %d tensors to %s", label, changed, dtype)
    return changed


def materialize_meta_tensors(module, dtype: torch.dtype, label: str) -> None:
    from accelerate.utils import set_module_tensor_to_device

    leftover_buf = 0
    leftover_param = 0
    for name, buf in list(module.named_buffers()):
        if buf.device.type != "meta":
            continue
        if "position_ids" in name and not buf.is_floating_point():
            value = torch.arange(0, buf.shape[-1], dtype=buf.dtype).expand(buf.shape).contiguous()
        else:
            value = torch.zeros(buf.shape, dtype=buf.dtype if buf.is_floating_point() else dtype)
        set_module_tensor_to_device(module, name, "cpu", value=value)
        leftover_buf += 1
    for name, param in list(module.named_parameters()):
        if param.device.type != "meta":
            continue
        value = torch.zeros(param.shape, dtype=dtype)
        set_module_tensor_to_device(module, name, "cpu", value=value)
        leftover_param += 1
    if leftover_param:
        log.warning("%s: %d params missing from checkpoint (zero-init)", label, leftover_param)
    if leftover_buf:
        log.info("%s: re-initialised %d meta buffers", label, leftover_buf)


def bf16_cache_path(original: Path) -> Path:
    """Where we cache a dequantized-to-bf16 copy of a safetensors file.

    Lives under `models/_kraken_cache/` so we never touch the user's original
    files. Filename embeds a short hash of the absolute source path so two
    different files with the same basename can both be cached.
    """
    from config import MODELS_ROOT
    cache_root = MODELS_ROOT / "_kraken_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    import hashlib
    src_hash = hashlib.sha1(str(original.resolve()).encode("utf-8")).hexdigest()[:8]
    return cache_root / f"{original.stem}.{src_hash}.bf16.safetensors"


def _cache_is_usable(cache_path: Path, original: Path) -> bool:
    if not cache_path.exists():
        return False
    try:
        return cache_path.stat().st_mtime >= original.stat().st_mtime
    except OSError:
        return False


def _write_bf16_cache(model, cache_path: Path, label: str) -> None:
    """Save the model's current state_dict as bf16 safetensors for fast reloads."""
    try:
        from safetensors.torch import save_file
        sd = {}
        for k, v in model.state_dict().items():
            t = v.detach().cpu().contiguous()
            if t.is_floating_point() and t.dtype != torch.bfloat16:
                t = t.to(torch.bfloat16)
            sd[k] = t
        tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
        save_file(sd, str(tmp))
        tmp.replace(cache_path)
        size_mb = cache_path.stat().st_size / 1024**2
        log.info("%s: wrote bf16 cache → %s (%.1f MB)", label, cache_path.name, size_mb)
    except Exception as e:
        log.warning("%s: bf16 cache write failed (%s) — ignoring", label, e)


def stream_safetensors_into_meta(
    model_cls,
    config,
    sd_path: Path,
    dtype: torch.dtype,
    label: str,
    key_prefix_strip: str = "",
    preserve_fp8: bool = False,
    use_bf16_cache: bool = False,
):
    """Build `model_cls(config)` on meta and stream weights from `sd_path`.

    When `use_bf16_cache=True` and `preserve_fp8=False`, the function transparently
    reads a pre-dequantized bf16 cache file next to the original (if fresh enough)
    and writes one for next time after a cold load. Useful for FP8 text encoders
    where the dequant cost is significant.
    """
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device

    cache_path: Path | None = None
    effective_path: Path = sd_path
    if use_bf16_cache and not preserve_fp8 and dtype == torch.bfloat16:
        cache_path = bf16_cache_path(sd_path)
        if _cache_is_usable(cache_path, sd_path):
            log.info("%s: using bf16 cache %s (skipping FP8 dequant)", label, cache_path.name)
            effective_path = cache_path

    log.info("loading %s from %s (streaming)", label, effective_path)
    with init_empty_weights():
        model = model_cls(config)

    placed = 0
    fp8_n = 0
    unmatched: list[str] = []
    with safe_open(str(effective_path), framework="pt", device="cpu") as f:
        keys = f.offset_keys()
        for i, key in enumerate(keys):
            tensor = f.get_tensor(key)
            name = key.removeprefix(key_prefix_strip) if key_prefix_strip else key
            if tensor.dtype in _FP8_DTYPES:
                fp8_n += 1
                if preserve_fp8:
                    if fp8_n == 1:
                        log.info("%s: preserving FP8 tensors for runtime casting", label)
                elif fp8_n == 1:
                    log.info("%s: dequantizing FP8 tensors -> %s (streaming)", label, dtype)
                if not preserve_fp8:
                    tensor = dequant_fp8_tensor(tensor, dtype)
            try:
                if preserve_fp8 and tensor.dtype in _FP8_DTYPES:
                    value = tensor
                    # Don't pass dtype= — accelerate would refuse / re-promote FP8.
                    set_module_tensor_to_device(model, name, "cpu", value=value)
                else:
                    value = tensor.to(dtype) if tensor.is_floating_point() else tensor
                    # Pass dtype= explicitly. Without it accelerate matches the
                    # destination parameter's dtype from the model config — for T5
                    # that's fp32, which silently promotes bf16 values, doubling
                    # peak RAM during load and triggering an OOM SIGKILL on subsequent
                    # generations when the cached transformer is still resident.
                    set_module_tensor_to_device(
                        model, name, "cpu", value=value,
                        dtype=dtype if value.is_floating_point() else None,
                    )
                placed += 1
            except (ValueError, KeyError, AttributeError):
                unmatched.append(name)
            del tensor
            if (i + 1) % 32 == 0:
                gc.collect()

    if fp8_n:
        verb = "preserved" if preserve_fp8 else "dequantized"
        log.info("%s: %s %d FP8 tensors", label, verb, fp8_n)
    if unmatched:
        log.warning("%s: %d keys did not match model (first: %s)", label, len(unmatched), unmatched[:3])

    materialize_meta_tensors(model, dtype, label)
    log.info("%s: placed %d tensors", label, placed)
    gc.collect()

    # Write bf16 cache the first time we cold-load a file that opted in. Skip
    # if we already read from the cache, or if we preserved FP8 (cache is bf16).
    if (
        use_bf16_cache
        and not preserve_fp8
        and dtype == torch.bfloat16
        and cache_path is not None
        and effective_path == sd_path
    ):
        _write_bf16_cache(model, cache_path, label)

    return model


def strip_accelerate_hooks(pipe: Any) -> None:
    try:
        from accelerate.hooks import remove_hook_from_module
    except ImportError:
        return
    components = getattr(pipe, "components", None) or {}
    for _name, comp in components.items():
        if comp is None:
            continue
        try:
            remove_hook_from_module(comp, recurse=True)
        except Exception:
            pass


def unload_pipeline(pipe: Any | None) -> None:
    """Drop a diffusers pipeline and release GPU/CPU memory.

    For streaming pipelines (FLUX with the StreamingLinear partition) we
    intentionally SKIP `pipe.to("cpu")`. Reasons:
      - Most of the transformer is GPU-resident; moving it to CPU allocates
        20+ GB of CPU RAM as a transient peak (Windows commit limit pain).
      - Pinned host weights are already on CPU and will be freed when the
        StreamingLinear modules are GC'd. No move needed.
      - For non-streaming pipelines (SDXL etc.), `pipe.to("cpu")` only saves
        memory if you plan to keep the pipe alive on CPU — we're tearing it
        down, so GC handles it just as well.
    """
    if pipe is None:
        return
    strip_accelerate_hooks(pipe)
    if hasattr(pipe, "components"):
        for name in list(pipe.components.keys()):
            try:
                setattr(pipe, name, None)
            except Exception:
                pass
    del pipe
    # Two collects: first sweep drops the pipe + its strong-ref cycle through
    # diffusers' progress-bar config / scheduler; second sweep catches the
    # parameters released by the first.
    gc.collect()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
