# Cursor bug-squash session — 2026-05-21 (FINAL)

Agent investigation and fix pass for Kraken Art generation failures.

---

## Status: FIXED (refactored for public release)

FLUX and SDXL generation both work end-to-end via the sidecar API. Sidecar survives FLUX→SDXL arch switches.

**Resumed 2026-05-21:** Window crash interrupted verification. Re-ran all smoke tests + sidecar API after refactor — all passed.

---

## Root causes found

### 1. T5 load native crash (B-001) — **FIXED**

**Symptom:** Sidecar exit `3221225477` / SIGKILL during T5-XXL load, no Python traceback.

**Cause:** `_load_into_meta()` called `safetensors.load_file()`, which materialised the **entire** checkpoint dict in RAM while simultaneously filling the meta-init model. Peak commit for T5 (~9 GB dict + ~9 GB weights) stacked on top of an already-loaded 22 GB FLUX transformer exceeded Windows' tolerance → native crash. Not an FP8-specific bug (FP16 T5 crashed the same way).

**Fix:** Stream via `safe_open` — one tensor at a time, place with `set_module_tensor_to_device`, delete immediately. FP8 dequant per-tensor via `_dequant_fp8_tensor()`.

### 2. FLUX dtype mismatch (B-009) — **FIXED**

**Symptom:** `mat1 Float vs mat2 BFloat16` in `transformer.x_embedder`.

**Cause:** diffusers seeds latents from `prompt_embeds.dtype` (float32 from TE LayerNorm) while transformer weights are bf16.

**Fix:** `KrakenFluxPipeline` subclass overrides `prepare_latents` to use `transformer.dtype` (matches `from_pretrained` behavior). Shared `ensure_module_dtype()` after load. Generator device follows `pipe._execution_device` after offload is applied.

### Public-release refactor (no system-specific band-aids)

| Before (band-aid) | After (proper) |
|-------------------|----------------|
| Monkey-patched `pipe.prepare_latents` | `KrakenFluxPipeline` subclass in `python/pipelines/kraken_flux_pipeline.py` |
| Hardcoded `gen_device = "cpu"` | `torch.Generator(device=pipe._execution_device)` |
| Magic `te_extra = 5.0` in offload picker | Measured component file sizes passed from `flux.py` |
| Duplicate load/unload helpers in `flux.py` | Shared `python/pipelines/load_utils.py` |
| Comments referencing 3090 / 24 GB | General VRAM/RAM language |

### 3. FLUX→SDXL arch switch crash — **FIXED**

**Symptom:** Sidecar died loading SDXL immediately after a successful FLUX job.

**Cause:** Accelerate CPU-offload hooks not stripped before dropping pipeline refs; unload at job-submit time could also break queued jobs.

**Fix:** `unload()` strips accelerate hooks + double `gc.collect()`; cross-arch unload moved to start of each pipeline's `run()` (not submit time).

### 4. Unicode logging (B-010) — **FIXED**

ASCII arrows in log strings; UTF-8 stdout in `log_buffer.py`.

---

## Verification (all passed)

| Test | Result |
|------|--------|
| `python/scripts/test_t5_load.py` (FP16 T5) | OK 5.4s |
| `python/scripts/test_flux_load.py` (full pipeline) | OK 24.7s, all bf16 |
| `python/scripts/test_flux_generate.py` (4 steps) | OK 13.3s → `outputs/test-flux-smoke.png` |
| FP8 T5 standalone inference | OK 12.6s → `outputs/test-flux-fp8.png` |
| Sidecar API: FLUX (refactored code) | **succeeded** → `outputs/2026-05-21/164940-726c66a3-00.png` |
| Sidecar API: SDXL after FLUX | **succeeded**, sidecar stayed up |
| `test_flux_load.py` (post-refactor) | OK ~22s |
| `test_flux_generate.py` (post-refactor) | OK ~14s inference |

---

## Files changed

| File | Change |
|------|--------|
| `python/pipelines/load_utils.py` | Shared streaming safetensors loader, FP8 dequant, hook-aware unload |
| `python/pipelines/kraken_flux_pipeline.py` | `KrakenFluxPipeline` — correct latent dtype for manual assembly |
| `python/pipelines/flux.py` | Uses shared utils + subclass; component-size offload picker |
| `python/pipelines/sdxl.py` | Shared `unload_pipeline`, generalized comments |
| `python/pipelines/offload.py` | `pick_strategy()` accepts measured TE/VAE sizes |
| `python/log_buffer.py` | UTF-8 stdout on Windows |
| `python/api/generate.py` | Removed premature unload at submit (now in `run()`) |
| `python/scripts/test_*.py` | Smoke test scripts (new) |
| `Documentation/BUGS.md` | B-001/B-009/B-010 marked resolved |
| `cursor.md` | This document |

---

## Remaining (not band-aids — tracked for release)

| Item | Notes |
|------|-------|
| Transformer bulk `load_file` | 22 GB checkpoint still materialises a full dict during ComfyUI→diffusers conversion. T5/CLIP use streaming loader. Next step: `FluxTransformer2DModel.from_single_file(..., low_cpu_mem_usage=True)` or a streaming converter. |
| B-002 | FLUX sequential offload slower than ComfyUI |
| B-003 | Cancel during model load is no-op |
| B-004 | No sidecar auto-restart watchdog |

## Smoke test commands

```powershell
# Start sidecar
& 'F:\Kraken Art\python\venv\Scripts\python.exe' 'F:\Kraken Art\python\main.py'

# Or full app
cd 'F:\Kraken Art'; npm run tauri dev

# Quick script tests
& 'F:\Kraken Art\python\venv\Scripts\python.exe' 'F:\Kraken Art\python\scripts\test_flux_generate.py'

# API FLUX
$body = @{ arch='flux1'; diffusion_model='Flux 1D FP16/fluxmania_kreamania.safetensors'; vae='FLUX1/fluxVaeSft_aeSft.sft'; text_encoders=@('clip_l.safetensors','t5/t5xxl_fp8_e4m3fn.safetensors'); prompt='red apple'; width=512; height=512; steps=4; cfg=1.0; seed=42 } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:7780/api/generate' -ContentType 'application/json' -Body $body
```

---

## Still open (not blockers)

- **B-002** FLUX ~3× slower than ComfyUI (sequential CPU offload, no prefetch)
- **B-003** Cancel during model load is a no-op
- **B-004** Sidecar auto-restart watchdog (task #25)

Sidecar is currently **running** from the last successful test session.
