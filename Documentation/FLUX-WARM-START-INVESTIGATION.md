# FLUX Warm-Start Performance Investigation

**Started:** 2026-05-28 (on branch `feature/flux-warm-start-investigation`)
**Trigger:** User observation — ComfyUI achieves ~48s warm generations on identical FLUX + text encoders, while Kraken Art remains ~60s+ even on warm starts. Cold starts are actually competitive or better in Kraken, but there is no "warm-up benefit."

**Goal:** Understand exactly what ComfyUI does differently that produces a large, repeatable warm-start speedup, then close the gap in Kraken Art without destroying the streaming offload that makes FP32 flux_dev usable on 24 GB cards.

**Backups for this investigation:**
- Git tag: `checkpoint/flux-perf-2026-05-27-...` (on `feature/flux-warm-start-investigation`)
- Filesystem: `backups/2026-05-28-1300-flux-warm-start-investigation-start/`

---

## Current Known Gap (from May 2026 research)

From `FLUX-SPEED-WIP.md`, `FLUX-PERFORMANCE-EXPERIMENTS.md`, and `grok_findings.md`:

| Metric                    | ComfyUI (torch 2.11 + comfy-kitchen) | Kraken Art (current) | Gap          |
|---------------------------|--------------------------------------|----------------------|--------------|
| Warm generation (28 steps, 1024², FP32 flux_dev) | ~47–48 s                            | ~60–64 s            | 12–17 s     |
| Median step time          | ~1.58–1.73 s                        | ~2.0–2.06 s         | ~0.3–0.4 s  |
| Min step time             | ~1.29 s                             | ~2.01 s             | ~0.7 s      |

**Important observation from prior work:**
- Kraken's **cold** start is sometimes *faster* than ComfyUI's cold because of disk caching of T5.
- The problem is **lack of improvement on subsequent generations**.

---

## Leading Hypotheses (from prior research + code)

### 1. Model is not becoming "warm" in the way ComfyUI makes it warm (Highest probability)

ComfyUI's big wins come when the model fits **fully GPU-resident**. Once it does:
- No more PCIe transfers per step.
- All the advanced techniques become additive: First Block Cache / TeaCache, comfy-kitchen fused kernels, better torch.compile behavior, dynamic shape handling, etc.

In Kraken we are still **streaming** a significant number of transformer blocks on every step (even on "warm" runs). The streaming partition itself is working correctly for memory, but it fundamentally prevents the model from reaching the "fully hot" state that ComfyUI reaches on the second generation.

**Evidence:**
- All attempts to layer FBCache, comfy-kitchen RoPE, torch.compile on top of streaming were net-negative due to VRAM contention and stream synchronization.
- The research docs explicitly conclude: "The real path to ComfyUI's 1.71 s/step: stop streaming" (via FP8, NF4, etc.).

### 2. Python / diffusers overhead per transformer block

ComfyUI uses hand-written `DoubleStreamBlock` / `SingleStreamBlock` in C++/CUDA (via comfy-kitchen and their own modeling code). Diffusers' `FluxTransformerBlock` has more Python-level dispatch, attribute lookups, and intermediate tensor creation.

This overhead is paid on **every** step and does not go away with "warm" runs.

### 3. Benchmarking / measurement overhead

Our current warm numbers may include more overhead than ComfyUI's:
- HTTP polling in `bench_step_times.sh`
- Full Tauri + React + WebSocket stack when measuring "in the app"
- More logging / progress emission

Isolated sidecar runs (`bench_step_ws.py`) got closer to 51-56 s, suggesting some of the gap is app overhead.

### 4. PyTorch version + kernel differences

ComfyUI on this box was running torch 2.11 + cu130 + comfy-kitchen. Kraken was on torch 2.6 at the time of the last deep measurements.

### 5. Missing overlapping of weight prefetch with compute (the "true prefetch" mentioned in BUGS.md)

The current `StreamingLinear` does synchronous H2D per layer in the forward. ComfyUI's async weight offloading (2 streams) + careful scheduling can hide more of the transfer behind compute.

---

## Current Implementation Snapshot (to be updated live)

Key files:
- `python/pipelines/streaming_linear.py` — the core `StreamingLinear` + `apply_streaming`
- `python/pipelines/flux.py` — `KrakenFluxPipeline`, loading, `run()`, partition logic
- `python/pipelines/offload.py`
- Various `kraken_*.py` patches

**Action items for this investigation:**
1. Re-measure the exact current warm vs cold gap with the best available tooling (after the ecosystem unification skeleton landed).
2. Instrument true per-layer timing inside the streaming path (H2D time vs compute time).
3. Determine how many blocks are actually being streamed on a "warm" run today.
4. Test whether simply forcing a larger resident budget (or using a smaller model) suddenly produces the "warm speedup" behavior.

---

## Investigation Plan (Living)

**Phase 1 — Re-baseline (do this first)**
- [ ] Run clean warm-start measurement using the best current tooling (`bench_step_ws.py` recommended in prior docs).
- [ ] Capture nvidia-smi, GPU clock, thermal throttling, VRAM usage, and number of streamed blocks during the run.
- [ ] Do the same measurement with the **exact same prompt** run twice in a row inside the full Tauri app.
- [ ] Record results here.

**Phase 2 — Where is the time actually going?**
- Add fine-grained timers inside `StreamingLinear.forward` (time spent in H2D vs actual matmul).
- Measure time spent in Python dispatch vs CUDA kernels.
- Determine if the "warm" run is still doing significant PCIe traffic.

**Phase 3 — Test the "fully resident" hypothesis**
- Try forcing a configuration where almost the entire transformer stays on GPU (smaller buffer_gb, or switch to a model that fits).
- If warm speedup suddenly appears, we have strong confirmation of Hypothesis #1.

**Phase 4 — Decide on strategy**
- Option A: Double down on making streaming as perfect as possible (better prefetch, 2-stream improvements that actually work on single-GPU PCIe, etc.).
- Option B: Make "fast path" (fully resident FP8/NF4) first-class and excellent, accept that big FP32 flux_dev will always be slower than ComfyUI on warm.
- Option C: Hybrid — detect when a model fits and disable streaming + enable caches/kernels.

---

## Notes / Open Questions

- Has anything changed in the streaming / offload code since the May 22 measurements?
- What is the current default `buffer_gb` and how many blocks does it typically keep resident on a 3090 with flux_dev?
- Can we make the partition decision smarter on the second generation (once activations from the first run give us better memory profiling)?

---

**This document will be updated after every measurement or code change in this investigation.** All changes are on the dedicated branch with the backups listed at the top.

---

## 2026-05-28 Analysis — Code + Prior Research Synthesis

### What the current streaming implementation actually does (as of this branch)

After reading `streaming_linear.py`:

- We have implemented **Phase 3** of the streaming design:
  - 2 CUDA mover streams (round-robin, ComfyUI pattern)
  - Pre-allocated flat GPU destination buffers (to avoid per-call allocator pressure)
  - **Block-level pre-forward hooks** that kick off prefetch of the *next* streamed block's weights
  - The `StreamingLinear.forward` has a fast path that waits on a CUDA event for a prefetched weight instead of doing the copy itself

This is significantly more sophisticated than the original "just copy in forward" version.

The prefetch chain is:
- Block N's `pre_forward` hook → starts async H2D for Block N+1's weights on a mover stream
- By the time Block N+1's `forward` runs, the weights should already be on GPU (or the event wait is very short)

### Current Diagnosis (Updated)

The fundamental reason we are not seeing the big "warm" improvement that ComfyUI gets is almost certainly still **Hypothesis #1** from the prior research:

> As long as a non-trivial number of transformer blocks remain on CPU and must be streamed every step (even with excellent prefetch), the generation never reaches the "fully hot, zero PCIe traffic" state that ComfyUI reaches on the second generation when the model fits in VRAM.

ComfyUI's ~48s warm number is achieved when the entire transformer + encoders are GPU-resident. At that point:
- No H2D traffic at all after the first forward
- First Block Cache / TeaCache becomes highly effective (no cache buffer contention)
- comfy-kitchen fused kernels and other low-level wins become pure wins instead of fighting the streaming machinery

Our current 2-stream + prefetch hook implementation is doing the best possible job *within the constraint of having to stream blocks*. It is not a bad implementation of streaming — it is an excellent implementation of something that is inherently more expensive than "the model is just sitting on the GPU."

### Other Contributing Factors (lower probability but worth measuring)

1. **Python overhead in diffusers FluxTransformerBlock** vs ComfyUI's hand-written blocks (still likely 200-400 ms/step)
2. **Benchmarking methodology differences** (our numbers may include more app + polling overhead)
3. **PyTorch version** (2.6 vs 2.11 in the original comparison)

---

## Recommended Immediate Next Actions

1. **Re-baseline the numbers on today's code**
   - Use `python/bench_step_ws.py` (the WebSocket version) for cleanest measurements
   - Run the exact same prompt twice in a row
   - Capture: total time, per-step times, nvidia-smi during the run, how many blocks are actually being streamed (log from `apply_streaming`)

2. **Instrument to answer "how much PCIe traffic is still happening on generation #2?"**
   - Add lightweight timing around the actual `copy_` / `to(non_blocking=True)` calls inside the prefetch hooks and the fallback path in `StreamingLinear.forward`
   - Or use Nsight / PyTorch profiler for one warm run

3. **Quick experiment: Force maximum residency**
   - Temporarily crank the budget extremely high (or use a smaller FLUX variant) and see if warm suddenly drops toward the 48-52 s range. This is the fastest way to confirm or refute the "must be fully resident" theory.

I recommend we start with #1 and #3 in the next session. They will give us very high-signal data with minimal code changes.

---

**Status:** Analysis phase complete. Ready to move into measurement when you are.

---

## Fresh Evidence from Live ComfyUI Instance (2026-05-28)

User provided path: `D:\AI_Art\ComfyUI` (confirmed via docs and direct filesystem).

**Startup log excerpt from `D:\AI_Art\ComfyUI\user\comfyui_8188.log` (relevant first ~80 lines):**

```
[2026-05-28 08:03:57.065] Found comfy_kitchen backend triton: {'available': True, ...}
[2026-05-28 08:03:57.065] Found comfy_kitchen backend eager: ...
[2026-05-28 08:03:57.065] Found comfy_kitchen backend cuda: {'available': True, 'disabled': False, ..., 'capabilities': ['apply_rope', 'apply_rope1', 'dequantize_nvfp4', 'dequantize_per_tensor_fp8', 'quantize_mxfp8', 'quantize_nvfp4', 'quantize_per_tensor_fp8', 'scaled_mm_nvfp4']}
[2026-05-28 08:03:57.175] pytorch version: 2.11.0+cu130
[2026-05-28 08:03:57.175] Set vram state to: NORMAL_VRAM
[2026-05-28 08:03:57.175] Device: cuda:0 NVIDIA GeForce RTX 3090 : cudaMallocAsync
[2026-05-28 08:03:57.187] Using async weight offloading with 2 streams
[2026-05-28 08:03:57.187] Enabled pinned memory 26119.0
[2026-05-28 08:03:57.478] Using pytorch attention
[2026-05-28 08:04:13.428] DynamicVRAM support detected and enabled
...
[2026-05-28 08:04:13.428] comfy-kitchen version: 0.2.8
```

**Key confirmed differences vs current Kraken:**
- PyTorch **2.11.0+cu130**
- Explicit: `Using async weight offloading with 2 streams`
- `DynamicVRAM support detected and enabled`
- Full `comfy_kitchen` cuda backend active (including `scaled_mm_*`)
- `comfy-kitchen version: 0.2.8`

This matches the environment the user originally compared against.

**Source reference (live on disk):**
`D:\AI_Art\ComfyUI\comfy\model_management.py` (the `get_offload_stream`, cast buffer, and 2-stream logic we partially ported).

**Note on our side:**
We have a good-faith port of the 2-stream + flat prefetch buffers + block pre-forward hooks in `python/pipelines/streaming_linear.py` (see `_NUM_MOVER_STREAMS = 2`, `_install_block_prefetch_hooks`, etc.). The gap persists even with that.

---

## Immediate Next Execution Step (Agreed Plan)

User approved moving to measurement.

**Clean baseline command (run after sidecar is up on the new branch):**

```powershell
cd 'F:\Kraken Art'
python\venv\Scripts\python.exe python\bench_step_ws.py 10001 28
```

This uses WebSocket progress (avoids the old HTTP polling noise) and gives per-step + total wall time. We can run it twice in a row for cold vs warm.

I will also add a small enhancement to the bench script to log the streaming partition summary (number of blocks on CPU vs GPU) if it's not too intrusive.

Ready to either:
- Enhance the bench script slightly for better telemetry, or
- Just run/document the baseline command for you to execute while the above ComfyUI instance is the reference.

Let me know the preference and I'll execute the next code change / doc update immediately (with backup first, of course).

---

**Update 2026-05-28 (continued execution)**

- User ran the benchmark script multiple times and kept hitting the raw, ugly `ConnectionRefusedError` traceback because the sidecar wasn't running.
- I immediately improved `python/bench_step_ws.py` with two layers of friendly error handling:
  1. A health check against `/health` right at the start of `main()`.
  2. A clear try/except around the job submission POST.
- Now when the sidecar is down, the user gets a clean, helpful message with the exact two commands to start it (standalone sidecar vs full `npm run tauri dev`), instead of a massive traceback.
- Updated the module docstring to document the new behavior.
- The fresh ComfyUI startup evidence (from `D:\AI_Art\ComfyUI\user\comfyui_8188.log`) has been recorded above.

**Current recommendation to user:**

Run this on the `feature/flux-warm-start-investigation` branch (after ensuring the sidecar is freshly started):

```powershell
cd 'F:\Kraken Art'
python\venv\Scripts\python.exe python\bench_step_ws.py 10001 28
```

Run it at least twice in succession (first = cold, second = warm). While it runs, also grab:
- The sidecar log line containing "streaming partition"
- GPU telemetry

This will give us the cleanest apples-to-apples comparison against the ComfyUI instance whose logs we just pulled.

All changes are documented, backed up, and reversible. Ready for the actual numbers.

---

## 2026-05-28 — ROOT CAUSE FOUND + FIXED: per-gen T5 re-encode

Golden config under test: `Flux 1D FP32/flux_dev.safetensors` (23.5 GB, bf16 weights)
+ `t5xxl_fp16` (9.6 GB) + clip_l + fluxVaeSft, 1024² × 28 steps, seed 10001.

### The actual warm-run differentiator

`_encode_flux_prompt` reloaded CLIP-L **and the 9.6 GB T5-XXL from disk**, moved
them to GPU, encoded, then `del`'d + `empty_cache()`'d them — **on every single
generation**. ComfyUI's `CLIPTextEncode` node caches its output, so a warm gen
with an unchanged prompt does ZERO text-encoding. That ~9 s of T5 reload+encode
was the bulk of the cold≈warm-but-still-slow behavior the user observed.

### Fix shipped (this branch)

Added a small conditioning-output cache (`_cond_cache`, OrderedDict, max 16) in
`_encode_flux_prompt`, keyed by `(clip_path, t5_path, prompt, max_seq_len, dtype)`.
Caches the ~4 MB encoded tensors, NOT the 10 GB encoder weights (that was the old
Phase-4 idea that OOM'd). Cleared by `_drop_te_cache()`. Also added per-stage
timing to `run()` (`FLUX run stages: text-encode=Xs · pipeline-ensure+loras=Ys`).

### Measured result (clean state, RTX 3090, this branch)

| Stage | Gen #1 (cold) | Gen #2 (warm, same prompt) |
|---|---|---|
| text-encode | 9.2 s | **0.0 s** (`conditioning cache HIT`) |
| pipeline-ensure + loras | 21.7 s | **0.0 s** (cached pipeline) |
| setup before step 1 | ~31 s | **~1 s** |
| per-step median / min | 1867 / 1790 ms | 1893 / 1816 ms |
| **total wall clock** | **83.5 s** | **53.0 s** |

Streaming partition on this run: total 22.17 GB · 43 blocks on GPU · 14 streamed
from CPU (3.69 GB) · per-step ~1.87 s. Note the per-step is now well under the
old docs' 2061 ms — streaming offload is in good shape.

**Warm went from the user's reported ~60–62 s to 53.0 s.** That 9 s drop is
exactly the eliminated T5 re-encode.

### Remaining gap to ComfyUI's ~48 s warm

~5 s, and it is now **purely per-step compute**: 1.89 s vs ComfyUI's 1.71 s over
28 steps ≈ 5 s. That is the FP8-resident + comfy-kitchen-kernel + torch-2.11
territory documented in FLUX-SPEED-RESEARCH.md — a much harder, higher-risk lever.

### Next lever (not yet done): changed-prompt path

The conditioning cache only helps a REPEATED prompt. When the user changes the
prompt, Kraken still pays the full 9.2 s T5 disk reload, whereas ComfyUI keeps
T5 resident and pays only the ~1 s encode forward. Keeping CLIP+T5 resident in
**CPU RAM** between gens (skip disk reload, pay only H2D + encode) is the next
high-value win for real iteration. The partition now streams only 3.69 GB, so
CPU-RAM headroom for a resident T5 is plausible — revisit the `_te_cache`
(encoder-weights) path with a commit-aware budget check.