"""System actions: clear VRAM, etc. Mirrors the unload/free pattern from Kraken_Audio."""
from __future__ import annotations
import gc
import logging
from fastapi import APIRouter

from pipelines import flux, ideogram4, sdxl, upscale_esrgan, z_image, wan_video

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
    # Each pipeline owns its unload; orchestrate them all here. Every image/video
    # arch must be listed — a missing one (e.g. z_image) means Clear VRAM leaves
    # that model resident, which is exactly the bug this covers.
    for name, mod in [
        ("sdxl", sdxl),
        ("flux", flux),
        ("z_image", z_image),
        ("ideogram4", ideogram4),
        ("wan_video", wan_video),
        ("upscale", upscale_esrgan),
    ]:
        try:
            details[name] = mod.unload()
        except Exception as e:
            log.warning("%s.unload raised: %s", name, e)
            details[name] = {"error": str(e)}

    # Also stop any running audio engine (its VRAM lives in a child process).
    try:
        from pipelines.audio.engine_manager import manager as audio_engines
        audio_engines.stop_all()
        details["audio_engines"] = "stopped"
    except Exception as e:
        log.warning("audio engine stop_all raised: %s", e)
        details["audio_engines"] = {"error": str(e)}

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
