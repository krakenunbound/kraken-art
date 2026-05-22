# FLUX speed — external research notes

What other people have actually tried (with URLs), so future sessions don't
re-research these. Companion to `FLUX-SPEED-WIP.md` (which tracks our
in-tree experiments and results).

**Last updated:** 2026-05-22 midday.

---

## Cross-check: are we at floor or behind?

| Reporter | Hardware | Config | Step time |
|---|---|---|---|
| FurkanGozukara wiki | 3090 Ti | ComfyUI, FLUX-dev **FP8**, 1024² | 1.27 s/it |
| Comfy-Org #9002 | 3090 | ComfyUI, FLUX-dev **FP8**, 20-step preset | ~1.3 s/it |
| (us, last night) | 3090 | ComfyUI on this same box, bf16 | 1.71 s/it |
| **(us, today)** | **3090** | **diffusers 0.38, bf16, streaming** | **2.06 s/it** |

Two facts to internalize:
1. **ComfyUI's published-fast numbers are FP8, not bf16.** ComfyUI auto-casts to
   `fp8_e4m3fn` on Ampere for matmul throughput. Their bf16 number on our box is
   1.71 s/it (matches research).
2. **No public diffusers-FLUX recipe sub-2 s/step on a vanilla 3090 in bf16
   has been posted.** All the sayakpaul/torchao recipes are A100/H100 only.

So our 2.06 s baseline is **~20 % behind a tuned-comfy bf16** and ~60 %
behind FP8. The 20 % is reachable with torch.compile. The full 60 % requires
quantization (NF4 or FP8) — Ampere can't do native FP8 but **NF4 via
bitsandbytes is on the table**.

Sources:
- [FurkanGozukara wiki — RTX 3090 Ti FLUX-dev benchmark](https://github.com/FurkanGozukara/Stable-Diffusion/wiki/RTX-5090-Tested-Against-FLUX-DEV-SD-35-Large-SD-35-Medium-SDXL-SD-15-AMD-9950X-RTX-3090-TI)
- [Comfy-Org #9002 — FLUX DEV fp8 3090 benchmark](https://github.com/Comfy-Org/ComfyUI/discussions/9002)

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
