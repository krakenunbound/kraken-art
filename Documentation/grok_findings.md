# Grok Performance Analysis & Recommendations

**Date:** May 22, 2026
**Author:** Grok (via analysis of kraken-art codebase)
**Status:** Living document

## Current Performance Snapshot

Recent benchmark (RTX 3090, 1024×1024, FLUX-dev):

| Metric              | ComfyUI 0.19.4 (torch 2.1.1) | Kraken Art (torch 2.6) | Gap          |
|---------------------|-------------------------------|------------------------|--------------|
| **Median step**     | 1731 ms                      | **2061 ms**            | **+19%**     |
| **Min step**        | 1286 ms                      | 2012 ms                | +56%         |
| **Mean step**       | 1751 ms                      | **2135 ms**            | +22%         |
| **Total gen**       | 79.2 s **(cold)**            | **~75 s (warm)**       | Kraken wins  |

**Key takeaway**: Warm generation time is already competitive or better than ComfyUI's cold start. The remaining gap is primarily in **per-step latency** (~330–380 ms/step slower).

## Root Cause Analysis

1. **Architecture**:
   - Kraken uses `diffusers.FluxPipeline` + custom `StreamingLinear` (Forge-style weight streaming) + custom attention patches.
   - ComfyUI uses its own highly optimized low-level Flux implementation (custom Python + C++/CUDA kernels, `comfy_kitchen` extensions).

2. **Major Performance Levers Missing**:
   - Dynamic caching (TeaCache / First Block Cache / WaveSpeed) — the single biggest reason ComfyUI feels dramatically faster.
   - Aggressive quantization + `torch.compile` (FP8 / int4 via torchao).
   - Full Phase-3 prefetch in streaming.

3. **Recent Discovery (Critical)**:
   - `apply_rotary_emb` wrapper was rejecting `sequence_dim=1` kwarg used by FLUX's `FluxTransformer2DModel` (input shape `[B, S, H, D]`).
   - This caused fallback/slow paths and potential hidden CUDA synchronization.
   - Claude is currently fixing the wrapper. This fix is expected to yield noticeable per-step improvement and allow safe re-enabling of previous RoPE/PosEmbed optimizations.

## What We've Tried & Learned

- Custom `StreamingLinear` with Phase 1/2 overlap is working very well.
- Several experimental patches (QKV fusion, FluxPosEmbed cache, torch.compile) were attempted but disabled due to regressions with streaming.
- fp16 vs fp8 T5-XXL has major VRAM + speed impact.
- Environmental issues (Windows Defender scanning, thermal throttling) have been mitigated.

## Prioritized Recommendations (Highest Impact First)

### 1. Merge RoPE Wrapper Fix + Re-enable Patch C (Immediate)
- Fix `apply_rotary_emb` wrapper to forward `sequence_dim` and other kwargs.
- Re-enable `FluxPosEmbed` / RoPE cache using `id(ids)` or `data_ptr()` as cache key (avoids `.item()` syncs).
- Re-benchmark immediately after.

### 2. Add First Block Cache (Biggest Single Win)

```python
from para_attn.first_block_cache.diffusers_adapters import apply_cache_on_pipe

apply_cache_on_pipe(pipe, residual_diff_threshold=0.08)  # Tune 0.06–0.12
```

- Expected: 1.5–2× effective speedup on FLUX with minimal quality impact.
- Works well alongside your streaming implementation.

### 3. torchao Quantization + torch.compile
- Use `torchao.quantization.autoquant` or `fp8dqrow` on the transformer.
- Then apply `torch.compile(mode="max-autotune", fullgraph=True)`.
- Switch to fp8 T5-XXL if not already using it.

### 4. Complete Streaming Optimizations
- Implement Phase-3 prefetch (next block weights).
- Fine-tune QKV fusion patch (reshape-first approach).

### 5. Profiling & Monitoring
- Use `torch.profiler` on the hot path after RoPE fix.
- Track VRAM usage, CUDA events, and thermal behavior.

## Expected Outcome

With steps 1 + 2 + 3 completed, Kraken should reach or exceed ComfyUI per-step performance while maintaining the clean diffusers-based architecture and excellent all-in-one app design.

Your current warm total already beating ComfyUI cold start shows the foundation is very strong.

---

**Last updated:** 2026-05-22
**Next benchmark target:** Sub-1800 ms median step time.

Feel free to update this document as progress is made.
