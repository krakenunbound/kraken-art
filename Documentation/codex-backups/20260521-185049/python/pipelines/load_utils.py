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


def dequant_fp8_tensor(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Convert one FP8 tensor to `dtype` without invoking torch FP8 cast kernels."""
    if tensor.dtype not in _FP8_DTYPES:
        return tensor

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


def stream_safetensors_into_meta(
    model_cls,
    config,
    sd_path: Path,
    dtype: torch.dtype,
    label: str,
    key_prefix_strip: str = "",
):
    """Build `model_cls(config)` on meta and stream weights from `sd_path`."""
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device

    log.info("loading %s from %s (streaming)", label, sd_path)
    with init_empty_weights():
        model = model_cls(config)

    placed = 0
    fp8_n = 0
    unmatched: list[str] = []
    with safe_open(str(sd_path), framework="pt", device="cpu") as f:
        keys = f.offset_keys()
        for i, key in enumerate(keys):
            tensor = f.get_tensor(key)
            name = key.removeprefix(key_prefix_strip) if key_prefix_strip else key
            if tensor.dtype in _FP8_DTYPES:
                fp8_n += 1
                if fp8_n == 1:
                    log.info("%s: dequantizing FP8 tensors -> %s (streaming)", label, dtype)
                tensor = dequant_fp8_tensor(tensor, dtype)
            try:
                value = tensor.to(dtype) if tensor.is_floating_point() else tensor
                set_module_tensor_to_device(model, name, "cpu", value=value)
                placed += 1
            except (ValueError, KeyError, AttributeError):
                unmatched.append(name)
            del tensor
            if (i + 1) % 32 == 0:
                gc.collect()

    if fp8_n:
        log.info("%s: dequantized %d FP8 tensors", label, fp8_n)
    if unmatched:
        log.warning("%s: %d keys did not match model (first: %s)", label, len(unmatched), unmatched[:3])

    materialize_meta_tensors(model, dtype, label)
    log.info("%s: placed %d tensors", label, placed)
    gc.collect()
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
    """Drop a diffusers pipeline and release GPU/CPU memory."""
    if pipe is None:
        return
    strip_accelerate_hooks(pipe)
    try:
        pipe.to("cpu")
    except Exception:
        pass
    if hasattr(pipe, "components"):
        for name in list(pipe.components.keys()):
            try:
                setattr(pipe, name, None)
            except Exception:
                pass
    del pipe
    gc.collect()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
