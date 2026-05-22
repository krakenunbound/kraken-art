"""Pick the right offload strategy for the current GPU + model.

Three modes, in increasing safety / decreasing speed:
  full_gpu               — everything stays on GPU. Fastest. Requires VRAM ≥ model + activations.
  model_cpu_offload      — whole components swap CPU↔GPU per pipeline stage. Medium speed.
                           Requires VRAM ≥ largest single component + activations.
  sequential_cpu_offload — submodules swap CPU↔GPU per forward pass (ComfyUI-style).
                           Slowest but works on small free VRAM.

Picker is heuristic, conservative on the safe side: if anything's borderline we'd
rather drop down one tier and run slower than crash.
"""
from __future__ import annotations
import logging
from typing import Any

log = logging.getLogger("kraken.offload")


def _nvml_mem() -> tuple[float, float] | None:
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        pynvml.nvmlShutdown()
        return mem.free / 1024**3, mem.total / 1024**3
    except Exception:
        return None


def free_vram_gb() -> float | None:
    mem = _nvml_mem()
    return mem[0] if mem else None


def total_vram_gb() -> float | None:
    mem = _nvml_mem()
    return mem[1] if mem else None


def pick_strategy(
    model_size_gb: float,
    *,
    has_text_encoders: bool = False,
    arch: str = "",
    text_encoder_size_gb: float = 0.0,
    secondary_text_encoder_size_gb: float = 0.0,
    vae_size_gb: float = 0.0,
) -> tuple[str, dict]:
    """Choose an offload strategy. Returns (strategy, info_dict_for_logging).

    Pass measured on-disk sizes for each component when they load separately
    (e.g. FLUX transformer + CLIP + T5 + VAE). For bundled checkpoints (SDXL),
    `model_size_gb` alone is enough and TE/VAE sizes can stay 0.

    Budgets are conservative: we'd rather run one tier slower than OOM mid-job.
    """
    import torch
    if not torch.cuda.is_available():
        return "full_gpu", {"reason": "no CUDA", "free_vram_gb": None}

    fv = free_vram_gb()
    tv = total_vram_gb()
    if fv is None and tv is None:
        return "sequential_cpu_offload", {"reason": "VRAM unknown", "free_vram_gb": None, "total_vram_gb": None}

    fv = fv if fv is not None else tv or 0.0
    tv = tv if tv is not None else fv

    # Fallback when callers only set has_text_encoders (legacy / bundled SDXL path).
    if has_text_encoders and text_encoder_size_gb == 0 and secondary_text_encoder_size_gb == 0:
        text_encoder_size_gb = 0.3
        secondary_text_encoder_size_gb = 5.0

    te_total = text_encoder_size_gb + secondary_text_encoder_size_gb
    largest_component = max(
        model_size_gb,
        text_encoder_size_gb,
        secondary_text_encoder_size_gb,
        vae_size_gb,
        0.0,
    )

    # full_gpu: every component resident at once + activation headroom.
    full_gpu_activation_buffer = 2.0
    full_gpu_need = (
        model_size_gb * 1.05
        + te_total
        + vae_size_gb
        + full_gpu_activation_buffer
    )

    # model_cpu_offload: only ONE whole component on GPU at a time (ComfyUI-style).
    # Compare against total VRAM — not free VRAM — because TE/VAE swap off during denoise.
    model_offload_activation_buffer = 1.0
    model_offload_need = largest_component + model_offload_activation_buffer
    model_offload_fits_total = largest_component + model_offload_activation_buffer <= tv * 0.98

    info = {
        "arch": arch,
        "free_vram_gb": round(fv, 2),
        "total_vram_gb": round(tv, 2),
        "model_size_gb": round(model_size_gb, 2),
        "text_encoder_size_gb": round(text_encoder_size_gb, 2),
        "secondary_text_encoder_size_gb": round(secondary_text_encoder_size_gb, 2),
        "vae_size_gb": round(vae_size_gb, 2),
        "largest_component_gb": round(largest_component, 2),
        "full_gpu_need_gb": round(full_gpu_need, 2),
        "model_offload_need_gb": round(model_offload_need, 2),
        "model_offload_fits_total": model_offload_fits_total,
    }

    if fv >= full_gpu_need:
        return "full_gpu", info
    if model_offload_fits_total or fv >= model_offload_need:
        return "model_cpu_offload", info
    return "sequential_cpu_offload", info


def _materialize_meta(pipe: Any, device: str = "cpu") -> None:
    """Move any meta-device tensors to real storage before offload hooks attach."""
    import torch
    from accelerate.utils import set_module_tensor_to_device

    for _name, module in pipe.components.items():
        if module is None or not hasattr(module, "named_parameters"):
            continue
        for pname, param in list(module.named_parameters()):
            if param.device.type != "meta":
                continue
            dt = param.dtype if param.is_floating_point() else torch.float32
            value = torch.zeros(param.shape, dtype=dt)
            set_module_tensor_to_device(module, pname, device, value=value)
        for bname, buf in list(module.named_buffers()):
            if buf.device.type != "meta":
                continue
            if "position_ids" in bname and not buf.is_floating_point():
                value = torch.arange(0, buf.shape[-1], dtype=buf.dtype).expand(buf.shape).contiguous()
            else:
                value = torch.zeros(buf.shape, dtype=buf.dtype)
            set_module_tensor_to_device(module, bname, device, value=value)


def apply(pipe: Any, strategy: str, device: str = "cuda") -> str:
    """Apply the chosen strategy. Returns the strategy actually used (may fall back)."""
    _materialize_meta(pipe, device="cpu")
    if strategy == "full_gpu":
        try:
            pipe.to(device)
            return "full_gpu"
        except Exception as e:
            log.warning("full_gpu placement failed (%s); falling back to model_cpu_offload", e)
            strategy = "model_cpu_offload"
    if strategy == "model_cpu_offload":
        try:
            pipe.enable_model_cpu_offload()
            return "model_cpu_offload"
        except Exception as e:
            log.warning("model_cpu_offload failed (%s); falling back to sequential", e)
            strategy = "sequential_cpu_offload"
    try:
        pipe.enable_sequential_cpu_offload()
        return "sequential_cpu_offload"
    except Exception as e:
        log.warning("sequential_cpu_offload also failed (%s); placing on %s as last resort", e, device)
        try:
            pipe.to(device)
            return "full_gpu_fallback"
        except Exception:
            return "failed"
