"""System actions: clear VRAM, etc. Mirrors the unload/free pattern from Kraken_Audio."""
from __future__ import annotations
import gc
import logging
from fastapi import APIRouter

from pipelines import flux, sdxl, upscale_esrgan

log = logging.getLogger("kraken.system")
router = APIRouter()


def _vram_free_mb() -> int | None:
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        pynvml.nvmlShutdown()
        return int(mem.free // (1024 * 1024))
    except Exception:
        return None


@router.post("/clear_memory")
def clear_memory() -> dict:
    before = _vram_free_mb()

    details = {}
    # Each pipeline owns its unload; orchestrate them here.
    for name, mod in [("sdxl", sdxl), ("flux", flux), ("upscale", upscale_esrgan)]:
        try:
            details[name] = mod.unload()
        except Exception as e:
            log.warning("%s.unload raised: %s", name, e)
            details[name] = {"error": str(e)}

    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass

    after = _vram_free_mb()
    freed_mb = (after - before) if (after is not None and before is not None) else None
    log.info("clear_memory: before=%s MB free, after=%s MB free, freed=%s MB", before, after, freed_mb)

    return {
        "vram_free_mb_before": before,
        "vram_free_mb_after": after,
        "freed_mb": freed_mb,
        "details": details,
    }
