# Kraken Art

A local desktop generative-AI workstation. Forge/A1111-style UX — dropdown selectors, hidden pipeline — for image, video, and audio generation, with first-class support for a local AI agent driving it via MCP.

**Stack:** Tauri 2 (Rust shell) + React/Vite/TS frontend + Python FastAPI sidecar + HuggingFace `diffusers`.

**Status (Phase 1+2 in progress):** image generation works end-to-end for SDXL and FLUX1. Video, music/voice, MCP agent surface, USDU, and face detailer are scoped tasks in progress. See [CHANGELOG.md](CHANGELOG.md) for what's landed.

## Repository layout

```
F:\Kraken Art\
  src/                    React + TypeScript UI
  src-tauri/              Rust host (Tauri 2)
  python/                 FastAPI sidecar, model pipelines
    venv/                 Python 3.11 venv with torch CUDA, diffusers
    pipelines/            One module per architecture (sdxl.py, flux.py, ...)
    api/                  Route handlers (gpu, deps, models, generate, logs, system)
  models/                 ~462 GB local model collection (see MODELS.md)
  outputs/                Generated images/videos/audio, organized by YYYY-MM-DD
  logs/                   Sidecar logs (rotated daily)
  Logo/                   Kraken logo (PNG)
  Documentation/          This folder
```

## Running

Day-to-day dev workflow:

```powershell
cd 'F:\Kraken Art'
npm run tauri dev
```

That launches the Tauri shell, which:
1. Compiles the Rust host (incremental — ~1-2 s after first build),
2. Starts Vite on `:1420` for HMR,
3. Spawns the Python sidecar (`python\venv\Scripts\python.exe python\main.py`) on `:7780`,
4. Opens the Kraken Art window.

The window auto-closes the sidecar on exit. If the sidecar dies mid-session, currently you have to close + relaunch — auto-restart watchdog is a tracked task.

See [DEV.md](DEV.md) for sidecar-only testing, manual launch, and other ops.

## Quick links

- [ARCHITECTURE.md](ARCHITECTURE.md) — how Tauri / React / Python / diffusers fit together
- [PIPELINES.md](PIPELINES.md) — which model architectures are supported and **what files to pick for each**
- [WAN_VIDEO.md](WAN_VIDEO.md) — WAN 2.2 T2V/I2V setup, LoRAs, last-frame extension, and test baseline
- [VIDEO_UPSCALE.md](VIDEO_UPSCALE.md) — video upscale engines, before/after bins, audio preservation, and recommended presets
- [MODELS.md](MODELS.md) — what each `models/<category>` folder is for; recognized extensions
- [DEV.md](DEV.md) — running, restarting, debugging
- [CHANGELOG.md](CHANGELOG.md) — what's been built so far
