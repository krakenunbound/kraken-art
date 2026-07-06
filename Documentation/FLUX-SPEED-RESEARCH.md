# FLUX speed — external research notes

What other people have actually tried (with URLs), so future sessions don't
re-research these. Companion to `FLUX-SPEED-WIP.md` (which tracks our
in-tree experiments and results).

**Last updated:** 2026-05-22 afternoon, after running an A/B against the
user's ComfyUI install.

---

## comfy-kitchen apply_rope patch — in-context regression (2026-05-22)

Standalone bench (CUDA backend on this 3090 with FLUX-shaped tensors `[1, 4608, 24, 128]`):
- diffusers' `apply_rotary_emb` (original): **2.84 ms/call**
- our `_ck_apply_rotary_emb` (with freqs_cis build + cache hit): **1.49 ms/call**
- raw `ck.apply_rope1` (pre-built freqs): 6.22 ms/call

The patched path IS faster in isolation. **But wired into `FluxAttnProcessor` running inside our streaming-partition pipeline, per-step time exploded from 2 s to ~24 s.** Patch disabled in `flux.py` (`# install_ck_rope_patch()` line), code preserved in `pipelines/kraken_rope.py`.

**Theories not yet validated:**
1. Sync between our mover CUDA stream (streaming offload) and the comfy-kitchen kernel call. The streamed-block prefetch and the apply_rope kernel may be serializing in a way diffusers' tensor-op-only original doesn't.
2. Output tensor stride differs from diffusers' original; downstream `dispatch_attention_fn` then materializes a copy each call.
3. VRAM pressure — we saw 213 MB / 486 MB free during these gens. Each new kernel invocation could be triggering allocator thrash.

**To validate**: run the patch on a model that fits the FAST path (no streaming) — fluxmania FP8 at default settings. If it's fast there, the regression is streaming-stream interaction. If it's still slow, the regression is dispatch/allocator. Logged as a follow-up; deferred for now in favor of higher-impact research.

## Grok independent analysis (2026-05-22)

The user ran an independent analysis via Grok ([grok_findings.md](grok_findings.md)). Grok arrived at the same per-step measurements and conclusions, AND flagged one major lever I'd missed:

**First Block Cache via `para-attn`** — Grok calls this *"the single biggest reason ComfyUI feels dramatically faster"* and recommends:

```python
from para_attn.first_block_cache.diffusers_adapters import apply_cache_on_pipe
apply_cache_on_pipe(pipe, residual_diff_threshold=0.08)  # tune 0.06–0.12
```

Claimed: **1.5–2× effective speedup on FLUX with minimal quality impact.** This is a residual-diff-based step-skipping technique (TeaCache / FBCache family). For a 28-step gen, if the FBC kicks in 14 times, that's effectively 14 steps of compute — huge.

**Tried 2026-05-22, NET NEGATIVE in our pipeline (10-50 s/step vs 2 s/step baseline).** Reasons:

1. **API mismatch:** para-attn 0.3.38's `call_remaining_transformer_blocks` was written against an older diffusers where `FluxSingleTransformerBlock.forward` took a single pre-concatenated `hidden_states`. Diffusers 0.38 refactored these blocks to take `(hidden_states, encoder_hidden_states, ...)` separately and return a tuple. We patched para-attn (`pipelines/kraken_fbcache.py`) to use the new signature, which fixes the TypeError but didn't make it fast.

2. **Streaming-offload contention:** FBCache stores `first_hidden_states_residual`, `hidden_states_residual`, and `encoder_hidden_states_residual` as cache buffers. With our streaming partition already using the activation budget for streamed weights (1.85 GB streamed + 2 GB activations on a 24 GB card), the additional ~1-2 GB of FBCache buffers pushes us into allocator thrash. ComfyUI ships with FBCache WITHOUT streaming — their FLUX-dev auto-FP8 fits fully resident, so no contention.

**Where this could still help in our code:** if we have a model that fits the fast path (no streaming), FBCache should be a real win there. e.g. fluxmania FP8 (~11 GB) on a 24 GB card. Settings knob preserved: `performance.flux_fbcache=on` opt-in (default off). Threshold 0.06-0.12 configurable. Tracked as task #50 → revisit after fast-mode-only models are in scope.

---

## Direct A/B vs ComfyUI on the same hardware (2026-05-22)

Used `python/bench_comfyui.py` to queue the user's "Flux Basic Workflow" through
ComfyUI's `/prompt` API and capture per-step times from their `/ws` progress
stream. SAME model, SAME text encoders, SAME resolution, SAME sampler — the
ONLY difference is the inference stack.

| | ComfyUI 0.19.4 + torch 2.11 + comfy_kitchen | Kraken Art 0.1 + torch 2.6 + diffusers |
|---|---|---|
| Median step | **1731 ms** | 2061 ms |
| Min step | **1286 ms** | 2012 ms |
| Mean step | **1751 ms** | 2135 ms |
| Total cold | 79 s | ~75 s warm (we have T5 disk cache they don't) |
| Workflow | FP32 flux_dev + t5xxl_fp16 + CLIP-L + fluxVae | identical |
| Steps × CFG | 28 × 1.0 | 28 × 1.0 |
| Resolution | 1024² | 1024² |

ComfyUI's **min** step is 36 % faster than ours. **Median** is 16 % faster.
The gap is real and reproducible.

## What ComfyUI is doing that we aren't (read from their source)

ComfyUI's startup log on this box shows:

```
Found comfy_kitchen backend cuda: {capabilities: [apply_rope, apply_rope1,
  dequantize_per_tensor_fp8, quantize_mxfp8, quantize_nvfp4,
  quantize_per_tensor_fp8, scaled_mm_nvfp4]}
Using async weight offloading with 2 streams
Using pytorch attention
```

Three distinct mechanisms, in order of likely impact:

### 1. `comfy-kitchen` is a public pip package with fused C++ kernels

**This is the smoking gun.** `comfy-kitchen` 0.2.8 is an open-source
(Apache-2.0) library from Comfy-Org. Provides CUDA + Triton backends for:
- `apply_rope` / `apply_rope1` — fused rotary embedding (no fp32 round-trip)
- `quantize_per_tensor_fp8`, `dequantize_per_tensor_fp8`
- `quantize_mxfp8`, `quantize_nvfp4`, matching `dequantize_*`
- `scaled_mm_mxfp8`, `scaled_mm_nvfp4` — low-precision matmul

How ComfyUI uses it (from `comfy/ldm/flux/math.py:42-63`):

```python
def _apply_rope(xq, xk, freqs_cis):
    # pure-Python implementation — equivalent to diffusers' apply_rotary_emb
    ...

try:
    q_apply_rope = comfy.quant_ops.ck.apply_rope
    def apply_rope(xq, xk, freqs_cis):
        if pe.shape[3] == 1:
            return _apply_rope(xq, xk, freqs_cis)
        return comfy.quant_ops.ck.apply_rope(xq, xk, freqs_cis)
except ImportError:
    logging.warning("No comfy kitchen, using old apply_rope functions.")
    apply_rope = _apply_rope
```

The fact that they EXPLICITLY check for it and warn if absent tells us they
measured the kernel is meaningfully faster than the pure-Python path.

**Pip-installable** — `pip install comfy-kitchen`. Source:
[github.com/Comfy-Org/comfy-kitchen](https://github.com/Comfy-Org/comfy-kitchen).
Tracked as task #49.

### 2. Async weight offload with TWO CUDA streams — THIS IS THE MAIN GAP

After actually reading `comfy/model_management.py:1155-1261` + `comfy/ops.py:210-258` (2026-05-22):

**ComfyUI's mechanism:**

```python
# 1. Default 2 streams on NVIDIA/AMD (CLI: --async-offload to tune)
NUM_STREAMS = 2

# 2. Pre-create both as torch.cuda.Stream with priority=0
ss = [torch.cuda.Stream(device, priority=0) for _ in range(2)]

# 3. Round-robin selection — get_offload_stream() returns ss[counter],
#    increments counter, and (critically) has the chosen stream WAIT for
#    the compute stream before it does anything:
def get_offload_stream(device):
    ss[counter].wait_stream(current_stream(device))      # offload waits for compute
    counter = (counter + 1) % 2
    return ss[counter]

# 4. Preallocated cast buffer per stream — int8 of size=largest_weight.
#    Reused across all Linear forwards. NO per-call allocation.
def get_cast_buffer(stream, device, size, ref):
    buf = STREAM_CAST_BUFFERS.get(stream)
    if buf is None or buf.numel() < size:
        buf = torch.empty(size, dtype=torch.int8, device=device)
        STREAM_CAST_BUFFERS[stream] = buf
    return buf

# 5. Per-Linear forward (comfy/ops.py:230-258):
offload_stream = get_offload_stream(device)
cast_buffer = get_cast_buffer(offload_stream, device, weight.nbytes + bias.nbytes, layer)
[weight_view, bias_view] = interpret_gathered_like([s.weight, s.bias], cast_buffer)
weight = cast_to(s.weight, dtype, device, non_blocking=True, stream=offload_stream, r=weight_view)
bias   = cast_to(s.bias,   dtype, device, non_blocking=True, stream=offload_stream, r=bias_view)
sync_stream(device, offload_stream)   # compute waits for this stream
```

**Our mechanism (current `pipelines/streaming_linear.py`):**

```python
# 1 stream, 1 buffer-allocation per call, no round-robin:
weight = self.weight.to(device, non_blocking=True)   # allocates new GPU tensor each call
torch.cuda.current_stream(device).wait_event(self._kraken_pending_event)
```

**Why ComfyUI wins ~330 ms/step:**
- Their pipeline depth is **2** (one stream copies layer N's weight while the other can already be queuing layer N+1's).
- Their copies write into a **preallocated reusable buffer** — no allocator pressure, no GPU memory churn.
- Each `get_offload_stream` rotates AND inserts a wait — so the compute stream and offload streams form a producer/consumer pipeline.

**Port plan (task #51):**
1. Module-level `STREAMS = []` of 2 CUDA streams + `STREAM_BUFFERS = {}` of preallocated buffers.
2. Replace `_get_mover_stream` (current single-stream singleton) with `get_offload_stream` that round-robins.
3. Replace `_kraken_pending_w` per-call allocation with a `copy_` into the preallocated buffer.
4. Compute stream waits for the picked offload_stream after each copy — same `sync_stream` pattern.
5. KEEP our per-block prefetch hook on top — it's complementary (queues N+1 while N is still computing).

Source: `D:\AI_Art\ComfyUI\comfy\model_management.py:1155-1318`, `comfy/ops.py:210-280`.

### 3. PyTorch 2.11 vs our 2.6

Five major versions difference. Improvements between 2.6 and 2.11:
- SDPA backend rewrite (PyTorch 2.7)
- New `flash_attention` integrated path (PyTorch 2.8)
- Compile fixes for fp16/bf16 inductor codegen (PyTorch 2.9)
- Better memory_format handling (PyTorch 2.10)

Hard to quantify the gain from torch upgrade alone vs comfy_kitchen, but
ComfyUI bundles 2.11 and that's part of why their stock SDPA path is faster.
Upgrading our venv to torch 2.7 (matches CUDA 12.4 binary wheels) would let
us A/B without major surgery.

## Things to skip / already-tried dead ends

(see following sections — these still apply)

---

## torch.compile — TESTED 2026-05-22, NET NEGATIVE on our arch

**Status: implemented, gated behind `performance.flux_compile=on` (default "off"),
not recommended.** Measured median **3445 ms/step (1.67× slower)** vs 2061 ms
baseline. Per-step variance was bimodal (~3000 / ~3450 ms alternating), classic
sign of Dynamo guard-miss → recompile + eager-fallback flapping.

**Why compile lost on our stack:**

1. **Triton missing on Windows.** PyTorch on Windows ships without Triton.
   `pip install "triton-windows>=3.2,<3.3"` installs a community build that
   pairs with torch 2.6+cu124 (triton-windows 3.7 is for torch 2.7+ — using
   it on 2.6 fails with `AttrsDescriptor` import error). With Triton 3.2.0,
   `torch.compile` works for trivial cases (`@torch.compile def f(x): return
   x + 1` round-trips fine).

2. **`RecompileLimitExceeded`** on first cold gen with default `cache_size_limit=8`.
   Bumped to 256 via `torch._dynamo.config.cache_size_limit = 256` — that
   silenced the error, but didn't fix the underlying recompile churn.

3. **StreamingLinear's dynamic dispatch is fundamentally compile-hostile.**
   The forward checks `self.__dict__.get("_kraken_pending_w")` and several
   conditional branches around FP8 cast + scaled-FP8 multiply. Even on the
   fast path (where all weights are GPU-resident and the dynamic checks
   resolve to None), each guard fires per call. Dynamo can't prove the
   guards are stable, so it recompiles or falls back to eager every few
   steps. Net result: spends more time on guard checks + recompiles than
   it saves on the fused matmul.

4. **`channels_last` memory format is wrong for FLUX.** It's a `NHWC` layout
   intended for convolutional models. FLUX transformer blocks operate on
   `(batch, sequence, features)` tensors — channels_last forces unnecessary
   format conversions on every Linear. Removing it didn't help measurably
   either; the recompile churn dominates.

**What would make compile work:** drop `StreamingLinear` entirely on the
fast path and use vanilla `nn.Linear`. That's a real refactor — two code
paths to maintain (compile-friendly fast, streaming-only otherwise). The
expected win even then is ~15–20 % per step (research-claimed). For a
22 GB FP32 model that requires forcing `flux_fast_inference_buffer_gb=0.3`
to fit; activation OOM risk is real. We'd ship the option but it can't be
the default.

**Decision:** ship `flux_compile=off` as default. Leave `flux_compile=on`
as an opt-in setting for users who:
- Have triton-windows installed
- Are using a model small enough that streaming doesn't kick in
- Are willing to refactor away StreamingLinear on the fast path
- Want to experiment at their own perf risk

Sources:
- [PyTorch blog: torch.compile + diffusers](https://pytorch.org/blog/torch-compile-and-diffusers-a-hands-on-guide-to-peak-performance/)
- [diffusers fp16+compile docs](https://huggingface.co/docs/diffusers/main/optimization/fp16)
- [triton-windows on PyPI](https://pypi.org/project/triton-windows/)
- our measurements: Documentation/FLUX-SPEED-WIP.md (median table)

---

## SageAttention — skip on FLUX

flash-attn-2 replacement, reportedly 20–40 % win on Ampere for attention.
Diffusers has the dispatcher wired
([attention_backends docs](https://huggingface.co/docs/diffusers/main/optimization/attention_backends)):
backends `sage`, `sage_hub`, `_sage_qk_int8_pv_fp16_cuda`,
`_sage_qk_int8_pv_fp16_triton`.

**Known FLUX-specific bug:** [thu-ml/SageAttention #324](https://github.com/thu-ml/SageAttention/issues/324)
— `_sage_qk_int8_pv_fp16_cuda` runs without error on FLUX's attn processor and
produces pure noise. Open, no resolution.

[diffusers integration tracking #11168](https://github.com/huggingface/diffusers/issues/11168)
reports only ~16 % speedup when it does work. On Ampere we can't use the FP8
PV variants (need SM 8.9+; 3090 is 8.6).

**Skip until FLUX noise bug closed upstream.** If revisited, try
`_sage_qk_int8_pv_fp16_triton` first.

---

## NF4 quantization via bitsandbytes — promising alternate path

Forge measured **3.86× speedup vs FP8 on a 3070 Ti** ([Forge #981](https://github.com/lllyasviel/stable-diffusion-webui-forge/discussions/981))
because NF4 uses `bnb.matmul_4bit` (native low-bit kernel) instead of
cast-then-fp16-matmul.

**On a 3090 with no VRAM pressure** the speedup is much smaller (1.2–1.4×),
BUT — and this is the interesting part — **NF4 makes the whole model fit in
~7 GB**, eliminating our streaming entirely. We currently move 1.58 GB/step
across PCIe. Killing that traffic could matter more than the per-step compute
win.

**Should be shipped as an opt-in 4th tier in our offload picker:** Auto / Force
fully resident / Force streaming / **NF4 (smallest VRAM, slight quality cost)**.

Sources:
- [Forge BnB/NF4 guidelines #981](https://github.com/lllyasviel/stable-diffusion-webui-forge/discussions/981)
- [HF diffusers/FLUX.1-dev-torchao-int8 model card](https://huggingface.co/diffusers/FLUX.1-dev-torchao-int8)
- [bitsandbytes quantization docs](https://huggingface.co/docs/diffusers/en/quantization/bitsandbytes)

---

## torchao INT8 — limited on Ampere

`Int8WeightOnlyConfig` works on 3090. Saves VRAM (19.7 → 10.3 GB on FLUX-dev)
but **speedup is modest without compile fused with it**. The diffusers-torchao
README explicitly says FP8 PTQ needs compute capability ≥ 8.9 — 3090 is 8.6,
so FP8 is unavailable.

Path: INT8 weight-only + torch.compile **together** — the sayakpaul recipe.
A100/H100-validated; Ampere should work but unmeasured publicly.

Sources:
- [sayakpaul/diffusers-torchao](https://github.com/sayakpaul/diffusers-torchao)
- [diffusers torchao docs](https://huggingface.co/docs/diffusers/en/quantization/torchao)

---

## Things that don't help on bf16 FLUX

- **stable-fast** ([chengzeyi/stable-fast](https://github.com/chengzeyi/stable-fast)) —
  dormant, no FLUX path.
- **TensorRT** — recipe exists ([Torch-TensorRT FLUX-dev](https://docs.pytorch.org/TensorRT/tutorials/_rendered_examples/dynamo/torch_export_flux_dev.html))
  with 1.5× / 2.4× claims, but **export requires the full graph to fit** —
  incompatible with our streaming offload arch.
- **OneFlow / AITemplate** — no maintained FLUX path.
- **`torch.backends.cuda.matmul.allow_tf32 = True`** — only affects fp32 matmuls,
  bf16 ignores it.
- **`torch.set_float32_matmul_precision('high')`** — same.
- **`cudnn.benchmark = True`** — helps convs (VAE decode), not transformer.
- **`memory_format=torch.channels_last`** without torch.compile — no measurable
  effect.

---

## Recommended priority for future speed work

1. **`compile_repeated_blocks` on the fast-mode path only.** Drop the
   monkey-patches first (already disabled today). Bench cold + warm; cold should
   be +5–10 s due to compile time, warm should drop ~15 %. → Task #47.
2. **NF4 path as an opt-in 4th tier** — major UX win because it eliminates the
   FLUX-fits-tight worry for any user with < 24 GB. Quality cost is real
   (Forge's #981 has examples) but minor for distilled FLUX. → Task #48.
3. **Skip SageAttention** until FLUX noise bug resolved.
4. **Skip TensorRT** — incompatible with streaming.
5. **Skip torchao on Ampere** unless paired with compile; the compile path
   alone is bigger.
