# Architecture

Three processes, one window, one local network.

```
┌─────────────────────────────────────────────────────────────────┐
│  Kraken Art (Tauri window)                                      │
│                                                                 │
│  React UI (Vite) ── fetch / WebSocket ──┐                       │
│       │                                  │                      │
│       │ invoke (Tauri IPC)              │                      │
│       ▼                                  │                      │
│  Rust host                               │                      │
│   • spawns Python sidecar on startup     │                      │
│   • exposes `sidecar_url` / `sidecar_ws_url` to React            │
│   • kills child on app exit              │                      │
│                                          ▼                      │
│                                  127.0.0.1:7780                 │
│                                  ┌────────────────────┐         │
│                                  │ Python sidecar     │         │
│                                  │ (FastAPI/uvicorn)  │         │
│                                  │  • /api/gpu        │         │
│                                  │  • /api/deps       │         │
│                                  │  • /api/models     │         │
│                                  │  • /api/generate   │         │
│                                  │  • /api/jobs/{id}  │         │
│                                  │  • /api/clear_memory│        │
│                                  │  • /api/logs       │         │
│                                  │  • /ws/jobs/{id}   │         │
│                                  └─────────┬──────────┘         │
│                                            │                    │
│                              diffusers / torch / spandrel       │
│                                            │                    │
│                                       CUDA · 3090               │
└─────────────────────────────────────────────────────────────────┘
```

## Why these pieces

**Tauri 2 (Rust)** — desktop shell. Native window, real binary, ~10 MB instead of Electron's 150. The Rust side does the bare minimum: own the window, spawn/kill the Python child, hand the URL to the frontend.

**React + Vite + TypeScript** — frontend. Vite gives instant HMR; React handles the dropdown/state explosion gracefully. Tauri IPC is only used for `sidecar_url`; everything else is plain `fetch` / `WebSocket` to the sidecar.

**Python FastAPI sidecar** — anything that touches the GPU. Heavy ML code stays in Python where diffusers, transformers, spandrel, and the eventual ACE-Step / LuxTTS / Demucs ecosystem live. Single process, async-with-threadpool for blocking inference.

**`diffusers` foundation** — we treat HuggingFace's `diffusers` library the way we treat PyTorch: a foundational library, not someone else's app. All pipeline orchestration, model loading, sampler choice, scheduler routing, LoRA merging, and UI is ours. (See [`feedback-no-existing-frameworks`](../../C:/Users/The Kraken/.claude/projects/F--Kraken-Art/memory/feedback_no_existing_frameworks.md) in agent memory — no forks of ComfyUI / A1111 / Forge.)

## Job lifecycle

1. UI builds a `GenerateParams` object from the selectors and POSTs to `/api/generate`.
2. The endpoint validates inputs (per architecture), then submits to the in-process **JobManager** (one FIFO queue, one worker thread — keeps VRAM predictable, lets us swap models cleanly between jobs).
3. The endpoint returns immediately with `{job_id, status: "queued"}`.
4. The UI opens a WebSocket to `/ws/jobs/{job_id}`.
5. The worker thread loads/reuses the pipeline, runs inference. Per-step events are pushed onto a per-job event bus, which the WebSocket subscriber drains and forwards as JSON.
6. On terminal status (`succeeded` / `failed` / `cancelled`), the WS closes.

## VRAM discipline

The 3090's 24 GB doesn't hold two large models at once. Two safeguards:
- `/api/generate` calls `unload()` on the *other* arches before submitting (SDXL → FLUX unloads SDXL first, and vice versa).
- A user-facing **Clear VRAM** button posts `/api/clear_memory`, which unloads every pipeline + flushes the CUDA cache. The GPU card in the left pane shows VRAM-free in near-real-time so you can watch it work.

## Logging

Three layers, all flowing through one ring buffer + a daily disk log:
- `kraken.*` loggers — our app code,
- `httpx` / `transformers` / `diffusers` — third-party,
- Global FastAPI exception handler — every 4xx, 422, and unhandled 5xx logs the path + detail + traceback.

The UI's bottom drawer reads from `/api/logs?since_id=N` incrementally. ERROR/CRITICAL auto-pops the drawer and ticks a red badge.
