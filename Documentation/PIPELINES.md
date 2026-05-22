# Pipelines

Which architectures work today, what files to pick for each, and what's pending.

## At-a-glance

| Architecture     | Status   | Mode        | Where it lands |
|------------------|----------|-------------|----------------|
| **SDXL**         | ✅ works | checkpoint  | `pipelines/sdxl.py` |
| **Illustrious**  | ✅ works | checkpoint  | uses `sdxl.py` (Illustrious is an SDXL fine-tune) |
| **FLUX1**        | ✅ works | components  | `pipelines/flux.py` |
| **ESRGAN upscale** | ✅ works | post-process | `pipelines/upscale_esrgan.py` (spandrel) |
| Ultimate SD Upscale | ⏳ task #15 | post-process | tile + img2img refine |
| Iterative upscale  | ⏳ task #17 | post-process | progressive scale steps |
| Face / region detailer | ⏳ task #16 | post-process | detect → crop → refine → paste |
| FLUX2            | ⏳ task #11 | components  | needs Gemma text encoder |
| Qwen-Image       | ⏳ task #11 | components  | uses Qwen text encoder |
| HunYuan          | ⏳ task #11 | checkpoint or components | TBD |
| Z-Image          | ⏳ task #11 (implied) | components | uses Qwen-derived TE |
| WAN (video)      | ⏳ task #12 | components  | i2v / t2v |
| LTX (video)      | ⏳ task #12 | checkpoint  | |
| ACE-Step (music) | ⏳ task #13 | port from `F:\Kraken_Audio` | |
| LuxTTS (voice)   | ⏳ task #18 | port from `F:\Kraken_Audio` | |

---

## SDXL / Illustrious

**Pipeline:** `StableDiffusionXLPipeline.from_single_file`. All-in-one checkpoint contains UNet + bundled VAE + two text encoders. No separate component picks needed.

| Selector | Pick |
|---|---|
| Architecture | `SDXL (all-in-one)` or `Illustrious (SDXL variant)` |
| Checkpoint | Anything from `models/checkpoints/` that's actually SDXL-arch. Confirmed working: `SDXL/epicrealismXL_*.safetensors`, `juggernautZ_v10ByRundiffusion.safetensors`, `Illustrious/ilustreal_v50VAE.safetensors`. |
| VAE | Leave **(bundled)** unless you have a specific override. The checkpoint includes one. |
| CFG | 7.0 (default) |
| Steps | 25–35 typical |
| Sampler | `dpmpp_2m` |
| Scheduler | `karras` (Karras sigmas) gives the cleanest results for SDXL |

**LoRAs:** any SDXL-trained LoRA works via `pipe.load_lora_weights` + `set_adapters`. Multi-LoRA with per-LoRA weight slider supported.

**Don't pick:** any FLUX / Qwen / HunYuan / WAN / LTX / stable-audio file. The SDXL single-file loader will fail with a `'CLIPTextModel' object has no attribute 'text_model'` or similar (we catch this and surface a friendly hint).

---

## FLUX1

**Pipeline:** custom assembly. We load each component locally:
- `FluxTransformer2DModel.from_config(...)` on meta, then state-dict load from local safetensors
- `AutoencoderKL` from meta + local weights via `convert_ldm_vae_checkpoint`
- CLIP-L + T5-XXL streamed in tensor-by-tensor (`stream_safetensors_into_meta`)
- `FlowMatchEulerDiscreteScheduler` with FLUX-dev shift values
- `KrakenFluxPipeline` (subclass of `FluxPipeline`) assembled

| Selector | Pick |
|---|---|
| Architecture | `FLUX1 (components)` |
| Diffusion model | One of the 22 GB FLUX transformers: `flux_dev.safetensors` (any subfolder — the `Flux 1D FP32` folder name is misleading, the file inside is actually FP16/bf16), `fluxmania_kreamania.safetensors`, `project0PJ0Krea_pj0KREAFP16.safetensors`. |
| VAE | `FLUX1/fluxVaeSft_aeSft.sft` **or** `QWEN/ae.safetensors` — they're the same 16-channel FLUX VAE (also used by Qwen-Image), just different filenames. Both are 319.8 MB. |
| Text encoder 1 | `clip_l.safetensors` (234 MB). |
| Text encoder 2 | `t5xxl_fp8_e4m3fn.safetensors` (4.6 GB) is recommended on the 3090 — saves ~5 GB vs FP16. Use `t5xxl_fp16.safetensors` (9.3 GB) if you want max quality and have RAM headroom. |
| CFG | **1.0**. Auto-set when you pick a FLUX model. (FLUX is distilled; CFG=1 effectively disables classical guidance and lets the model's internal distilled-guidance embedding drive composition.) |
| Steps | **28**. Auto-set when you pick a FLUX model. |
| Sampler / Scheduler | Ignored — FLUX uses its own flow-match scheduler internally. The UI fields don't apply. |
| Negative prompt | **Ignored** by FLUX. Logged as info if supplied. |

**Don't pick:**
- `t5-base.safetensors` for slot 2 — that's a small, generic T5 (850 MB). FLUX needs T5-XXL. Loading t5-base will silently produce key-mismatch warnings and bad output.
- `gemma_3_12B_it_fp4_mixed.safetensors` — that's for FLUX2, not FLUX1.

**LoRAs:** FLUX-trained LoRAs work. SDXL LoRAs do **not** — different transformer architecture, different layer names. See README → "Are FLUX and Z-Image LoRAs interchangeable?" (short answer: no).

### Memory & speed: the streaming offload mechanism

A bf16 FLUX-dev transformer is ~22 GB in memory. On a 24 GB GPU there's not enough headroom for that *plus* the VAE *plus* activations, so something has to give. Three approaches we considered, and what we landed on:

1. **`enable_model_cpu_offload` (diffusers default)** — moves entire submodules CPU↔GPU per forward. Stable but slow: ~138 s for one 1024² × 28-step gen on a 3090 (FP32 flux_dev).
2. **Fully-resident GPU** — only works if model + VAE + activations fit, otherwise OOM mid-step.
3. **StreamingLinear partition** *(what we use)* — port of Forge/Fooocus's mechanism. Every `nn.Linear` in the transformer is swapped with `StreamingLinear`, a subclass whose `forward()` knows how to pull its weight from CPU when needed. At load time, `apply_streaming()` runs a budget-aware partition:

   - **Always-resident on GPU**: VAE, embedders, norms, proj_out, x_embedder, context_embedder, time_text_embed (small modules, would be slower to stream than to keep).
   - **GPU as budget allows**: full `transformer_blocks.N` / `single_transformer_blocks.N` loaded onto GPU until the VRAM budget is exhausted.
   - **Streamed from CPU**: remaining blocks. Their Linear weights stay in pinned host memory; each forward, the weight is copied to GPU on a dedicated **mover CUDA stream** with `non_blocking=True`, overlapping with the compute stream's prior matmul.

   On a 24 GB 3090 with FP32 flux_dev, this lands at ~52 blocks fully resident on GPU and ~5 blocks streamed. The 1.3 GB of streamed weights cross PCIe per step.

| Mode (1024² × 28 steps) | Cold gen | Warm gen | Notes |
|---|---|---|---|
| diffusers `enable_model_cpu_offload` (old) | ~138 s | ~138 s | crashes on follow-up gen (RAM blowup) |
| **StreamingLinear (current)** | **~75 s** | **~73 s** | stable across gens; arch-switch safe |
| For reference: ComfyUI same machine | ~107 s | ~48 s | their PyTorch 2.11 + comfy_kitchen C++ ext + finer prefetch |
| Best case — model fits on GPU (e.g. fluxmania FP8 ~11 GB) | ~75 s | ~65–70 s | `fast` path, no streaming needed |

The `auto` heuristic is in `config_store.performance.flux_fast_inference`. Override to `"on"` to force fully-resident (will OOM if you cheated the budget) or `"off"` to force streaming (useful for memory-tight machines).

#### How to read the FLUX startup log

```
FLUX fast-inference auto-check: free=22.99 GB · need=24.48 GB ... -> FALL BACK to offload
FLUX streaming setup: free VRAM=22.99 GB after VAE · transformer budget=20.99 GB (reserved 2.00 GB for activations)
FLUX transformer: streaming partition — total=22.17 GB · always-GPU=0.12 GB · blocks on GPU=52 · streamed from CPU=5 (1.32 GB) · budget left=0.14 GB
FLUX streaming_linear: 30/504 Linears stream from CPU; 52 blocks resident on GPU, 5 on CPU
```

- **fast-inference auto-check**: did fully-resident fit? If not, fall to streaming.
- **streaming setup**: VAE goes to GPU first; remaining VRAM minus a 2 GB activations reserve = the transformer budget.
- **streaming partition**: byte-level summary — how much GPU was used, how many blocks didn't fit.
- **streaming_linear**: count of `StreamingLinear` instances that flipped to CPU-streaming mode (the actual hot path during sampling).

#### Where this lives

- `python/pipelines/streaming_linear.py` — `StreamingLinear`, `swap_linears`, `apply_streaming`, plus the mover-stream singleton (`_get_mover_stream`).
- `python/pipelines/flux.py` — calls `swap_linears(transformer)` after weight scales are applied, then either fully-resident GPU (`pipe.to('cuda')`) or `apply_streaming(transformer, ...)` depending on the budget check.
- `python/pipelines/load_utils.py` — `stream_safetensors_into_meta()` handles tensor-by-tensor loading + the bf16 disk cache for FP8 text encoders (avoids paying FP8→bf16 dequant on every cold load).

#### Known limits

- **First gen of the session is full cold cost** (~75 s on FP32). Subsequent gens of the same model reuse the partitioned pipeline (~73 s — almost all of the savings come from skipping the transformer rebuild, plus the T5 bf16 cache hit).
- **Switching models** triggers a full unload + reload. The new `unload_pipeline()` deliberately skips `pipe.to('cpu')` to avoid a 20+ GB transient RAM peak — instead it relies on garbage collection to release the GPU storage once the pipeline reference drops.
- **Windows commit limit**: keeping the source safetensors mmap'd would hold ~22 GB of virtual address space. `apply_streaming` `.clone()`s every streamed weight into private heap memory so the source file can close. Without this, the next large mmap (T5 cache, etc.) would hit `OSError 1455: paging file too small`.
- **Phase 3 (future)**: prefetch the *next* streamed Linear's weight while the *current* one's matmul runs. Today the mover stream gives async H2D but compute still waits on the copy of the same layer. True prefetch needs a pre-forward hook on the containing block that knows the order of children. Estimated 5-10 s additional savings on FP32 — would close most of the remaining gap to ComfyUI.

---

## Upscale (ESRGAN)

Powered by **spandrel** — a universal loader supporting Real-ESRGAN, ESRGAN, SwinIR, DAT, OmniSR, and more. Drops in any `.pth` / `.safetensors` upscale model and figures out the architecture.

| Selector | Pick |
|---|---|
| Upscale | check **enabled** |
| Mode | `ESRGAN (single-pass)` — the only mode wired today. USDU and Iterative are stubbed. |
| Model | depends on content (see below) |
| Factor | the model has an intrinsic scale (e.g. 4x); if `factor` differs we resize the result to your exact target |

Model picks by content:
- General photo: `4x-UltraSharp.pth`, `4x_NMKD-Siax_200k.pth`, `4x_foolhardy_Remacri.pth`, `RealESRGAN_x4.pth`
- Anime/illustration: `4x-AnimeSharp.pth`
- Faces specifically: `8x_NMKD-Faces_160000_G.pth` (better at facial detail)
- Conservative classic: `ESRGAN_4x.pth`

Outputs save **two** files per image: the original at `outputs/YYYY-MM-DD/HHMMSS-jobid-NN.png` and the upscaled version at `…-NN-up.png`. The thumbnail in the UI shows the upscaled one.

USDU (Ultimate SD Upscale) and Iterative upscale are stubbed — the backend accepts the params and logs a skip-message. They land in tasks #15 and #17.

---

## What to watch in the log drawer

Friendly messages we surface (in `kraken.sdxl` / `kraken.flux` / `kraken.system` loggers):
- `'CLIPTextModel' object has no attribute 'text_model'` → transformers version mismatch (we pin <5).
- `SDXL load failed for <file>: …` with arch hints if the user picked a non-SDXL file.
- `enable_model_cpu_offload failed; falling back to full GPU` → likely no CUDA available.
- `clear_memory: before=X MB free, after=Y MB free, freed=Z MB` → confirms unload worked.

Any HTTP 4xx / 422 / unhandled 5xx is logged with the request path + detail. Tracebacks for 5xx and job failures are full.
