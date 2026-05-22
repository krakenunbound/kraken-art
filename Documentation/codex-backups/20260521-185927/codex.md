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

### Research Update

- User confirmed the same FLUX models and settings work correctly and quickly in ComfyUI, so the working assumption is loader/runtime code, not bad models.
- External docs checked:
  - Diffusers FLUX docs for the expected `FluxPipeline` component layout and supported local transformer loading.
  - Diffusers `pipeline_flux.py` for latent unpack/decode behavior.
  - ComfyUI `UNETLoader` / `DualCLIPLoader` docs for the component split used by successful FLUX workflows.
- Local ComfyUI install checked only as reference, not as a runtime dependency:
  - Path: `D:\AI_Art\ComfyUI`
  - Recent Comfy logs show successful `fluxmania_kreamania.safetensors` runs with weight dtype `torch.float8_e4m3fn` and manual cast to `torch.bfloat16`.
  - Successful workflows use FLUX guidance around 3.5 and sampler CFG 1.0.
- Direct Kraken API test before the fix produced a completed job, but the image was blue/green checkerboard noise. The failed test output I created was deleted.

### Root Cause Found

- `Flux 1D FP16/fluxmania_kreamania.safetensors` is not a plain FP16 transformer even though the folder name says FP16.
- It contains Comfy-style old scaled-FP8 metadata:
  - `model.diffusion_model.scaled_fp8`
  - hundreds of `*.scale_weight` and `*.scale_input` tensors
- Kraken previously stripped `model.diffusion_model.` and sent those tensors directly through Diffusers conversion as ordinary weights.
- Comfy does not treat those tensors as ordinary weights. It converts old scaled-FP8 metadata to quantized layer metadata, casts the stored values to FP8, applies per-layer weight scale, and runs with manual cast.
- Treating FP8 code values as real BF16/FP16 weights explains the checkerboard output while the same model works in ComfyUI.

### Files Backed Up

- Backup folder: `F:\Kraken Art\Documentation\codex-backups\20260521-182337`
- Backed up before edits:
  - `codex.md`
  - `python\pipelines\flux.py`

### Code Changes In Progress

- `python\pipelines\flux.py`
  - Added FLUX transformer checkpoint normalization before Diffusers conversion.
  - Added support for Comfy old scaled-FP8 metadata (`*.scale_weight`, `*.scale_input`, `scaled_fp8`).
  - Added support for newer Comfy quant metadata (`*.comfy_quant`, `*.weight_scale`, `*.input_scale`) for FP8 formats.
  - Added explicit layout detection for BFL/Comfy FLUX keys versus Diffusers FLUX keys.
  - Added fail-loud behavior for unsupported FLUX transformer layouts or unsupported quant formats instead of silently producing garbage.
  - Removed the unconditional forced fallback from `model_cpu_offload` to `sequential_cpu_offload`; after the weight-format fix, the app should be allowed to use the faster offload strategy selected for the machine.
