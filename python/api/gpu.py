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
        "gpu_utilization_percent": None,
        "gpu_temperature_c": None,
        "gpu_clock_mhz": None,
        "gpu_clock_max_mhz": None,
        "power_watts": None,
        "power_limit_watts": None,
        "throttled": False,
        "throttle_reasons": [],
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
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(h)
            info["gpu_utilization_percent"] = int(util.gpu)
        except Exception as e:
            info["errors"].append(f"pynvml utilization: {e}")
        try:
            info["gpu_temperature_c"] = int(
                pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
            )
        except Exception as e:
            info["errors"].append(f"pynvml temperature: {e}")
        # Current vs max core clock — a big gap during a 100%-util job means the
        # card is being held back (almost always heat on a 3090).
        try:
            info["gpu_clock_mhz"] = int(pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_GRAPHICS))
            info["gpu_clock_max_mhz"] = int(pynvml.nvmlDeviceGetMaxClockInfo(h, pynvml.NVML_CLOCK_GRAPHICS))
        except Exception as e:
            info["errors"].append(f"pynvml clocks: {e}")
        try:
            info["power_watts"] = round(pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0, 1)
            info["power_limit_watts"] = round(pynvml.nvmlDeviceGetEnforcedPowerLimit(h) / 1000.0, 1)
        except Exception as e:
            info["errors"].append(f"pynvml power: {e}")
        # Throttle reasons (bitmask). Thermal/power slowdown = the card is
        # capping its own clocks. Names vary across pynvml versions, so look
        # each up defensively.
        try:
            reasons = pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(h)
            reason_map = [
                ("nvmlClocksThrottleReasonSwThermalSlowdown", "thermal (soft)"),
                ("nvmlClocksThrottleReasonHwThermalSlowdown", "thermal (hard)"),
                ("nvmlClocksThrottleReasonHwSlowdown", "hardware slowdown"),
                ("nvmlClocksThrottleReasonSwPowerCap", "power limit"),
                ("nvmlClocksThrottleReasonHwPowerBrakeSlowdown", "power brake"),
            ]
            active = []
            for attr, label in reason_map:
                bit = getattr(pynvml, attr, 0)
                if bit and (reasons & bit):
                    active.append(label)
            info["throttle_reasons"] = active
            info["throttled"] = bool(active)
        except Exception as e:
            info["errors"].append(f"pynvml throttle: {e}")
        info["driver"] = pynvml.nvmlSystemGetDriverVersion()
        if not info["name"]:
            info["name"] = pynvml.nvmlDeviceGetName(h)
            info["detected"] = True
            info["vendor"] = "NVIDIA"
        pynvml.nvmlShutdown()
    except Exception as e:
        info["errors"].append(f"pynvml: {e}")

    return info
