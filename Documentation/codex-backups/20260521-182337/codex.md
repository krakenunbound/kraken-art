# Codex Work Log

## 2026-05-21

### Mission

Make Kraken Art generate FLUX images as fast and as good as ComfyUI, without using ComfyUI.

### Ground Rules

- Keep this file updated with findings, changes, and verification.
- Before editing any existing file, copy it into a timestamped backup folder under `Documentation/codex-backups/`.
- Do not rewrite broad parts of the app unless the current code path requires it.

### Initial Findings

- Project path: `F:\Kraken Art`.
- This folder is not currently a Git repository, so filesystem backups matter.
- Stack: Tauri 2 + React/Vite/TypeScript frontend + Python FastAPI sidecar + diffusers.
- Current docs say SDXL and FLUX1 both generate end-to-end.
- The major open FLUX problem is speed: FLUX jobs are slower than ComfyUI when the app falls back to diffusers CPU offload.
- Current machine/env observed:
  - GPU: NVIDIA GeForce RTX 3090, 24 GB VRAM.
  - Python venv: Python 3.11.9.
  - Torch: 2.6.0+cu124.
  - diffusers: 0.38.0.
  - transformers: 4.57.6, despite `requirements.txt` documenting `<5`.
  - accelerate: 1.13.0.
- A sidecar is running from `F:\Kraken Art\python\main.py`.
- There appear to be two Python sidecar processes, one from the venv and one from the system Python path; this needs cleanup before reliable benchmarking.
- Latest logs show recent FLUX jobs succeeded, but took about 90 seconds for small 512x512 runs including model load.

### Current Technical Hypothesis

- The slow path is related to offload selection and/or forced fallback from `model_cpu_offload` to `sequential_cpu_offload`.
- `python/pipelines/flux.py` contains a Windows-specific warning that `model_cpu_offload` can corrupt FLUX latents, and forces sequential offload in that case.
- Logs show two different recent behaviors:
  - One run applied `model_cpu_offload`.
  - A later run forced `sequential_cpu_offload`.
- Need to separate load time from cached inference time, then verify which offload path gives correct image output.

### Files Inspected

- `Documentation/README.md`
- `Documentation/ARCHITECTURE.md`
- `Documentation/PIPELINES.md`
- `Documentation/BUGS.md`
- `Documentation/MODELS.md`
- `cursor.md`
- `python/pipelines/flux.py`
- `python/pipelines/offload.py`
- `python/pipelines/kraken_flux_pipeline.py`
- `python/pipelines/load_utils.py`
- `python/scripts/test_flux_generate.py`
- `python/requirements.txt`

### Next Steps

- Back up any existing files before edits.
- Identify and stop duplicate/orphan sidecar processes if needed for clean benchmarks.
- Add targeted timing around FLUX load, prompt encoding, denoise, and decode if missing.
- Test cached FLUX inference with the current offload path.
- Only then patch the speed path.
