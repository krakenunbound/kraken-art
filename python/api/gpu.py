"""GPU detection — reports name, VRAM, CUDA availability, driver. Fault-tolerant: never raises."""
from __future__ import annotations
from fastapi import APIRouter

router = APIRouter()


@router.get("/gpu")
def gpu_info() -> dict:
    info: dict = {
        "detected": False,
        "vendor": None,
        "name": None,
        "vram_total_mb": None,
        "vram_free_mb": None,
        "driver": None,
        "cuda_runtime": None,
        "cuda_available": False,
        "torch_version": None,
        "errors": [],
    }

    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if info["cuda_available"]:
            info["cuda_runtime"] = torch.version.cuda
            info["name"] = torch.cuda.get_device_name(0)
            info["detected"] = True
            info["vendor"] = "NVIDIA"
            props = torch.cuda.get_device_properties(0)
            info["vram_total_mb"] = props.total_memory // (1024 * 1024)
    except Exception as e:
        info["errors"].append(f"torch: {e}")

    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        info["vram_total_mb"] = mem.total // (1024 * 1024)
        info["vram_free_mb"] = mem.free // (1024 * 1024)
        info["driver"] = pynvml.nvmlSystemGetDriverVersion()
        if not info["name"]:
            info["name"] = pynvml.nvmlDeviceGetName(h)
            info["detected"] = True
            info["vendor"] = "NVIDIA"
        pynvml.nvmlShutdown()
    except Exception as e:
        info["errors"].append(f"pynvml: {e}")

    return info
