"""Kraken Art sidecar entry point — FastAPI on localhost:7780.

Started as a subprocess by the Tauri host. Single-process; the host handles
restart/health. All endpoints are scoped under /api/* except /health.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

# Switch PyTorch's CUDA allocator to cudaMallocAsync BEFORE torch is imported.
# This matches ComfyUI's backend choice — async alloc has lower per-call overhead
# which adds up across 14k Linear forwards per FLUX gen. Must be set pre-import.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")

# Disabled 2026-05-22: setting DIFFUSERS_ATTN_BACKEND=_native_cudnn caused a
# catastrophic regression (~60 s/step vs ~2 s/step baseline). Suspected cause:
# on torch 2.6 + cu124, the cuDNN backend's _native_cudnn_attention path falls
# back to a much slower kernel for FLUX's [B, S, H, D] tensor layout. The
# heuristic default works better on this hardware. Keeping the comment so
# future bench runs know to leave this alone unless we upgrade torch.
# os.environ.setdefault("DIFFUSERS_ATTN_BACKEND", "_native_cudnn")

# Ensure local imports work no matter how python is invoked.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import SIDECAR_HOST, SIDECAR_PORT, MODELS_ROOT, OUTPUTS_ROOT
import log_buffer
log_buffer.install(log_dir=MODELS_ROOT.parent / "logs")
log = logging.getLogger("kraken")
log.info("sidecar starting — models_root=%s outputs_root=%s", MODELS_ROOT, OUTPUTS_ROOT)

from api.gpu import router as gpu_router
from api.deps import router as deps_router
from api.models import router as models_router
from api.generate import router as generate_router
from api.progress import router as progress_router
from api.logs import router as logs_router
from api.system import router as system_router
from api.settings import router as settings_router
from api.civitai import router as civitai_router
from api.outputs import router as outputs_router
from api.audio import router as audio_router
from api.cover_art import router as cover_art_router

app = FastAPI(title="Kraken Art Sidecar", version="0.1.0")

# Tauri webview origins. Loose dev policy; tightened pre-ship.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:1420", "tauri://localhost", "https://tauri.localhost"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(HTTPException)
async def http_exception_logger(request: Request, exc: HTTPException) -> JSONResponse:
    # Surface 4xx/5xx in the log drawer (FastAPI's default just returns the response).
    # 404s are demoted to INFO: they're a normal client-side state (e.g. UI polling
    # a job_id from a previous sidecar process after a restart) and shouldn't drown
    # out real warnings in the log drawer.
    if exc.status_code >= 500:
        level = logging.ERROR
    elif exc.status_code == 404:
        level = logging.INFO
    else:
        level = logging.WARNING
    log.log(level, "%s %s -> %d: %s", request.method, request.url.path, exc.status_code, exc.detail)
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail}, headers=getattr(exc, "headers", None) or {})


@app.exception_handler(RequestValidationError)
async def validation_logger(request: Request, exc: RequestValidationError) -> JSONResponse:
    errs = exc.errors()
    log.warning("%s %s -> 422 validation: %s", request.method, request.url.path, errs)
    return JSONResponse(status_code=422, content={"detail": errs})


@app.exception_handler(Exception)
async def unexpected_logger(request: Request, exc: Exception) -> JSONResponse:
    import traceback
    log.error("%s %s -> 500 unhandled: %s\n%s", request.method, request.url.path, exc, traceback.format_exc())
    return JSONResponse(status_code=500, content={"detail": f"{type(exc).__name__}: {exc}"})


@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "service": "kraken-art-sidecar",
        "models_root": str(MODELS_ROOT),
        "models_root_exists": MODELS_ROOT.exists(),
        "outputs_root": str(OUTPUTS_ROOT),
    }


app.include_router(gpu_router,      prefix="/api")
app.include_router(deps_router,     prefix="/api")
app.include_router(models_router,   prefix="/api")
app.include_router(generate_router, prefix="/api")
app.include_router(logs_router,     prefix="/api")
app.include_router(system_router,   prefix="/api")
app.include_router(settings_router, prefix="/api")
app.include_router(civitai_router,  prefix="/api")
app.include_router(outputs_router,  prefix="/api")
app.include_router(audio_router,    prefix="/api")   # Music / ACE-Step bridge
app.include_router(cover_art_router, prefix="/api")  # Phase C: cover-art for Music tab + Song Studio
app.include_router(progress_router)  # WebSocket path uses /ws/...


@app.on_event("shutdown")
def _stop_audio_engines_on_shutdown() -> None:
    """Kill any audio engine subprocess the sidecar spawned so they don't orphan
    when the app exits (the sidecar owns their lifecycle)."""
    try:
        from pipelines.audio.engine_manager import manager as audio_engines
        audio_engines.stop_all()
    except Exception:
        pass
    try:
        from pipelines.audio import song_studio_service
        song_studio_service.stop()
    except Exception:
        pass


if __name__ == "__main__":
    import uvicorn
    OUTPUTS_ROOT.mkdir(parents=True, exist_ok=True)
    # Pass the app OBJECT, not the "main:app" import string. The string form makes
    # uvicorn re-import this module (it's already running as __main__), which
    # re-runs startup and logs "sidecar starting" a second time — the confusing
    # double line in the logs. No reload/workers here, so the object is correct.
    uvicorn.run(app, host=SIDECAR_HOST, port=SIDECAR_PORT, log_level="info")
