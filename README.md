# Kraken Art

A desktop image / video / audio generation app for diffusion models, built around HuggingFace `diffusers`. Tauri + React frontend, Python sidecar, runs locally on Windows / macOS / Linux.

Designed in the spirit of [AUTOMATIC1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui) and [Forge](https://github.com/lllyasviel/stable-diffusion-webui-forge) — a clean GUI, not a node graph — and aimed at users who want one tool that handles their whole creative pipeline.

> ⚠️ Pre-release / actively developed. APIs and UI move quickly. Set expectations accordingly.

## What works today

- **SDXL** / **Illustrious** (all-in-one checkpoints)
- **FLUX1-dev** (component checkpoints: transformer + VAE + CLIP-L + T5-XXL)
- **LoRA** stacking with per-LoRA weight sliders
- **Textual inversion** embeddings
- **ESRGAN upscale** (via spandrel — any RealESRGAN / SwinIR / DAT / OmniSR model)
- **Civitai integration**: browse, preview, download with hash verification
- **Resilient sidecar**: Rust watchdog auto-respawns the Python process on crash
- **Live log drawer** with copy/pause/clear

## In progress / planned

See [`Documentation/TODO.md`](Documentation/TODO.md) for the full backlog. Highlights:
- Phase 3 image archs: FLUX2, Qwen-Image, Z-Image, HunYuan
- Phase 4 video: WAN, LTX
- Phase 5 audio: ACE-Step (music), LuxTTS (voice) — ported from sister project
- Phase 6 MCP server: drive the app from a local AI agent
- Ultimate SD Upscale, face/region detailer, iterative upscale

## Repository layout

```
src/                  React + TypeScript frontend (Vite)
src-tauri/            Tauri Rust shell (spawns + supervises the sidecar)
python/               FastAPI sidecar — diffusion pipelines, model scanner, job queue
  pipelines/          One module per architecture (sdxl.py, flux.py, …)
  api/                FastAPI routers
  bench_step_times.sh Per-step timing harness for any future speed work
config/               User-local settings (gitignored)
models/               Local model store (gitignored — bring your own)
outputs/              Generated images / videos (gitignored)
Documentation/        Architecture notes, pipeline guide, changelog, bugs, TODO
Launch Kraken Art.bat One-click launcher (Windows)
```

## Build from source

### Prerequisites

- **Node 20+** with `npm`
- **Rust 1.78+** (install via `rustup`)
- **Python 3.11** (3.10 may work; nothing earlier)
- **NVIDIA GPU with ≥ 8 GB VRAM** for SDXL, ≥ 12 GB for FLUX (24 GB recommended for FP32 FLUX-dev)
- CUDA 12.4 toolkit OR PyTorch will pull its own runtime libs

### One-time setup

```bash
git clone https://github.com/krakenunbound/kraken-art.git
cd kraken-art

# Frontend
npm install

# Python sidecar
python -m venv python/venv
python/venv/Scripts/activate            # Windows
# source python/venv/bin/activate       # macOS/Linux
pip install -r python/requirements.txt
```

### Run

**Windows**: double-click `Launch Kraken Art.bat`.

**Anywhere**: `npm run tauri dev`.

The launcher kills any stale sidecar on port 7780, then starts the Tauri shell, which spawns the Python sidecar.

### Models

Drop models into `models/<category>/` under the project root. Categories:

- `checkpoints/` — all-in-one SDXL / Illustrious / similar
- `diffusion_models/` — standalone FLUX / Qwen transformers
- `vae/` — VAE files (`.safetensors` or `.sft`)
- `text_encoders/` — CLIP-L, T5, Gemma, etc.
- `loras/`, `embeddings/`, `upscalers/`

Or use the in-app Civitai Library tab to browse and download.

## Performance notes

Performance work on the diffusion path is heavily documented under `Documentation/`:

- [`PIPELINES.md`](Documentation/PIPELINES.md) — the FLUX streaming-offload mechanism, why we chose it over diffusers' `enable_model_cpu_offload`
- [`FLUX-SPEED-WIP.md`](Documentation/FLUX-SPEED-WIP.md) — in-tree experiments with measured numbers
- [`FLUX-SPEED-RESEARCH.md`](Documentation/FLUX-SPEED-RESEARCH.md) — external research with URLs

## License

MIT — see [`LICENSE`](LICENSE).

## Credits

- The Forge / Fooocus / ComfyUI / A1111 communities for everything they've published on diffusion-model memory management and inference.
- The HuggingFace `diffusers` team.
- The `spandrel` project for one-line upscaler loading.
- `lllyasviel` specifically for the streaming-offload pattern this project's FLUX path is built on.
