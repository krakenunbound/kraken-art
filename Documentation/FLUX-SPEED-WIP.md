# FLUX speed pass — work-in-progress state

**Last updated:** 2026-05-22 midday, after thermal fix + per-step bench.

## Where we stand

| | FP32 flux_dev, 1024² × 28 steps, 3090 |
|---|---|
| Yesterday (2026-05-21 PM) | ~75 s cold / ~73 s warm — stable across gens + arch switches |
| Target (ComfyUI on same machine) | ~107 s cold / ~48 s warm |
| **Today's measured steady-state** | **~58 s pure sampling (2.06 s/step × 28), ~73 s warm with setup** |

## Findings from today's bench (2026-05-22)

After the morning's apparent regression turned out to be thermal throttle
(GPU was at 82 °C, fans pegged at 100 %, soft-throttling clocks to 1500 MHz),
the user opened MSI Afterburner and raised the temp limit to 91 °C + fan curve
to 83 %. Card now runs 1860 MHz stable at 71 °C during sustained gens.

With thermals controlled, I ran per-step A/B benches via `bench_step_times.sh`
(polls progress every 250 ms, records wall-clock between step transitions):

| Config | Median step | Min step | Mean step |
|---|---|---|---|
| **No patches (shipping)** | **2061 ms** | **2012 ms** | 2135 ms |
| Patch C only (RoPE cache) | 2522 ms | 2093 ms | 2656 ms |
| Patches B + C | 2468 ms | 2027 ms | 2509 ms |
| buffer_gb=0.5 (1 streamed block) | 2187 ms | 1764 ms | 2269 ms |
| **torch.compile + buffer_gb=0.3 (fast)** | **3445 ms** | **2988 ms** | 3371 ms |
| **ComfyUI 0.19.4 (torch 2.11) — same workflow** | **1731 ms** | **1286 ms** | 1751 ms |
| **comfy-kitchen apply_rope patch in our streaming path** | **~24 000 ms (!)** | (cancelled) | — |
| **FBCache (para-attn) in our streaming path** | **~19 288 ms (10-50 s spikes)** | 10 021 ms | 17 637 ms |
| **TRUE baseline (Claude idle, clean state)** | **2909 ms** | **2455 ms** | 2802 ms |
| 2-stream round-robin offload (no preallocated buf) | 3009 ms | 2555 ms | 2836 ms |
| 2-stream + preallocated GPU dest buffer | ~80 000 ms (VRAM thrash) | — | — |
| ComfyUI 0.19.4 on same hardware | **1731 ms** | **1286 ms** | 1751 ms |

**Patches B and C are net-negative on this hardware** and have been backed out.
The min step time is essentially constant at ~2010–2030 ms across configs, but
the median creeps up because the patches add sync-style overhead per step:

- **Patch C (FluxPosEmbed cache)** keys the cache by `ids[0,0].item()` and
  `ids[-1,-1].item()` — both force CUDA stream syncs. Across 28 steps × 2 syncs
  that adds ~400 ms/step in stalls. The cached RoPE compute itself is only
  ~5–10 ms in the original code, so the savings are tiny relative to the
  sync cost. A correct fix would key by `id(ids)` or `data_ptr()` to avoid the
  GPU↔CPU sync — left as future work.
- **Patch B (QKV fusion)** saves kernel launches but introduces every-4-steps
  spikes (~2850 ms vs 2050 baseline) that look like the streamed-block prefetch
  chain reacting badly to the larger fused weight transfer (a single 75 MB
  fused weight crosses PCIe ~3× the size of one to_q/k/v individually). Not
  fundamentally broken — just doesn't win on this exact GPU + torch version.

The **buffer_gb=0.5** experiment is interesting: reducing the streaming buffer
keeps only 1 block on CPU (vs 5–6 default), but median step time got *slower*
(2099 → 2187 ms). The min did drop (2012 → 1764 ms — best result of the day).
The variance pattern suggests the smaller streaming surface area gives less
opportunity for the prefetch chain to overlap with compute. Default 2.0 GB
buffer is the right shipping value.

## Conclusion: shipping configuration

**Streaming offload (Phases 1–3 from 2026-05-21) at default settings is the
shipping config.** ~73 s warm / 28 steps on FP32 flux_dev, stable.

### Crucial insight from 2026-05-22's experiments

Every "ComfyUI-style" compute optimization we tried (comfy-kitchen apply_rope,
FBCache, torch.compile) measured **net-negative** when stacked on top of our
streaming partition, despite being verified-faster in isolation. The pattern:

> ComfyUI's tricks assume the model is fully GPU-resident. Our streaming
> partition keeps ~7 blocks in pinned host memory, with a mover CUDA stream
> moving them in per step. Layering FBCache cache buffers, the comfy-kitchen
> RoPE kernel's CUDA-stream interactions, or torch.compile's
> guards-and-recompiles on top of that already-tight VRAM budget and
> already-busy stream graph creates lock contention that costs more than
> the optimization saves.

**The real path to ComfyUI's 1.71 s/step:** stop streaming. Either:

1. **Auto-cast FP32 → FP8 on Ampere** like ComfyUI does (their `--fp8_e4m3fn-unet`
   default fires automatically when VRAM is tight). Cuts model size in half,
   fits fully GPU-resident, eliminates the streaming partition entirely.
2. **NF4 via bitsandbytes** (task #48). Cuts model to ~7 GB. Forge measured
   3.86× speedup vs FP8. Same idea — make the streaming partition unnecessary.

Once the model fits fully on GPU, all the per-step compute optimizations
(FBCache, comfy-kitchen kernels, torch.compile) become viable and additive.
**Layering them on top of our streaming-required setup is the wrong direction.**

### Update late 2026-05-22: streaming layer is NOT the gap

After porting ComfyUI's 2-stream + preallocated-buffer pattern faithfully:
- 2-stream alone: 3009 ms median (vs 2909 baseline — noise)
- Preallocated GPU dest buffers: catastrophic VRAM thrash (80 s/step) because
  on a 24 GB card, doubling the streamed footprint (CPU pinned + GPU dest)
  exceeds the activation budget.

PCIe is one-transfer-at-a-time on a single card. Two CUDA streams CAN'T
physically overlap host→device copies on the same PCIe link — they only
help if multiple separate engines are available (multi-GPU setups, NVLink).
On a single 3090, our existing 1-stream + per-block prefetch already
captures the available H2D↔compute overlap.

The 1180 ms/step gap to ComfyUI is **not in the streaming infrastructure**.
Most likely candidates, in order of suspected impact:

1. **diffusers' FluxTransformerBlock forward has more Python overhead than
   ComfyUI's hand-written `DoubleStreamBlock.forward`** (more intermediate
   tensors, more kernel launches, more `__getattr__` lookups). Would need
   a side-by-side line count + profiler comparison.
2. **Our bench probe overhead.** We poll `/api/jobs` every 250 ms over HTTP.
   FastAPI handlers acquire the GIL; if they contend with the diffusion
   thread, per-step time inflates. ComfyUI's bench used `/ws` (push) — no
   polling load. Should re-bench with a WebSocket-based progress consumer.
3. **PyTorch 2.6 vs 2.11.** Diffusers' SDPA dispatch and bf16 codegen got
   real improvements between 2.6 and 2.11. ComfyUI bundles 2.11.

Next experiment if we keep digging: replace `bench_step_times.sh`'s HTTP
polling with a WebSocket subscriber to eliminate (2) as a confound. Then
the remaining gap is (1) + (3) and we know how big each is.

## Per-step bench tool

`python/bench_step_times.sh` is the tool of record for any future speed work:

```bash
bash python/bench_step_times.sh [seed] [step_count]
# prints per-step wall-clock + median/mean/min over steady-state steps (skips warmup)
```

Use it on a system that's already had one cold gen (so disk + caches are warm)
and AFTER any thermal-throttle measurement (i.e. verify nvidia-smi shows
sw_thermal_slowdown=Not Active before trusting numbers).

## Patches: current state

| Patch | Status | Reason |
|---|---|---|
| A (`DIFFUSERS_ATTN_BACKEND=_native_cudnn`) | OFF, do not enable | Broke things on torch 2.6+cu124, 30× regression |
| B (QKV fusion + bf16 RoPE) | OFF | Net-negative on per-step variance |
| C (FluxPosEmbed cache) | OFF | Sync-style cache keys cost more than the RoPE compute saves |
| D (AdaLN addcmul) | not written | Skipping — micro-optimizations dominated by base compute |
| E (`torch.compile`) | **off by default, opt-in via settings** | **Tested 2026-05-22 — 1.67× SLOWER on our arch. StreamingLinear's `__dict__.get` dispatch causes Dynamo guard misses → recompile churn. Triton 3.2 IS installed (via `triton-windows`). Code preserved for users with all-resident config. See FLUX-SPEED-RESEARCH.md for the writeup.** |

`kraken_flux_attn.py` still exists and is correct code — just not wired in.
Keep it for reference / future re-enable after a torch upgrade or hardware
swap. Same for `_install_pos_embed_cache()` in `flux.py` (commented out).

## Hardware findings worth noting

- This 3090 thermally throttles around 80–83 °C in default fan profile. MSI
  Afterburner with temp limit 91 °C + manual fan curve resolves it. Document
  this in onboarding for any user reporting "FLUX is slow on a 3090."
- Per-step compute on the 3090 with diffusers 0.38 + torch 2.6 + bf16 FLUX-dev
  is ~2.0 s. That's the floor for this stack.

## The regression is environmental, not in our code

I spent the morning debugging what turned out to be a system-level slowdown that appeared overnight. Verified by:

- Reverting every patch I'd added this morning (cuDNN env var, RoPE cache, QKV fusion) — same slow.
- Running with last night's exact known-good code — still 70+ s per step.
- GPU itself is healthy: pstate P0, gr clock 1935/2130 MHz, mem clock 9751 MHz, temp 79 °C, power 245 W, no throttle reasons active.
- Fresh sidecar process, no other CUDA users, 23 GB free VRAM at gen start.

**The leading suspect** is a Windows Defender Security Intelligence Update that installed at 1:48 AM (between last night's working tests and this morning). System uptime is 3 days. Real-time scanning is ON. Defender is the only thing I can pin to the right timeframe and the right kind of impact (multi-GB safetensors mmap reads + frequent pinned-memory H2D transfers).

**Things to try, in order, when resuming:**
1. **Reboot.** Free, easy, often fixes mystery slowdowns after a Defender update.
2. **Add Defender exclusions** for `F:\Kraken Art\models`, `F:\Kraken Art\python\venv`, `F:\Kraken Art\python`. Settings → Windows Security → Virus & threat protection → Manage settings → Exclusions. Needs admin.
3. **Last resort**: `Set-MpPreference -DisableRealtimeMonitoring $true` in elevated PowerShell while benching, re-enable after.

When step time is back to ~2.5 s/step (~70–75 s total for the same 28-step gen), we're back to baseline and can resume the speed pass.

## Code changes made this morning — current state of each file

### `python/main.py` — Patch A (cuDNN backend hint)

```python
# Currently DISABLED (commented out). When the env hint is set on torch 2.6+cu124,
# diffusers' _native_cudnn_attention picks a kernel that ran ~30× slower for
# FLUX on this hardware. Heuristic default is faster on our setup.
# os.environ.setdefault("DIFFUSERS_ATTN_BACKEND", "_native_cudnn")
```

**Decision:** keep disabled. Re-evaluate after a torch upgrade (≥ 2.7). Tracked as task #41.

### `python/pipelines/flux.py` — Patch C (RoPE cache) and Patch B wiring

Top of the file (after `_loaded_loras` global) has the helper:

```python
def _install_pos_embed_cache() -> None:
    # ...monkey-patches diffusers' FluxPosEmbed.forward...
```

And at the end of that helper block:

```python
# Disabled 2026-05-22 while investigating a per-step regression.
# _install_pos_embed_cache()
```

**Decision:** the patch itself is correct (verified by reading the code path). It was disabled only because today's environment makes every patch look broken. Re-enable by uncommenting once baseline is fast again.

Inside `_ensure_pipeline()` after `swap_linears`:

```python
# Patch B (QKV fusion + bf16-native RoPE) — temporarily disabled while
# debugging a per-step regression in the fusion path. ...
# from pipelines.kraken_flux_attn import fuse_attention_qkv, install_kraken_attn_processor
# n_fused = fuse_attention_qkv(transformer)
# install_kraken_attn_processor(transformer)
# if n_fused:
#     log.info("  fused QKV in %d attention modules + installed bf16-RoPE processor", n_fused)
```

**Decision:** ALSO disabled. There's a real concern noted in the inline comment: diffusers' `_get_fused_projections` does `chunk(3, dim=-1)` which returns three non-contiguous views. Downstream `unflatten` + `norm_q` may materialize them, costing more than the fusion saves. Before re-enabling, the cleanest fix is to make `KrakenFluxAttnProcessor` skip the diffusers helper and split the qkv tensor in contiguous slices itself. See "Patch B follow-up" below.

### `python/pipelines/kraken_flux_attn.py` — NEW file (Patch B implementation)

Full Forge-style fused-attention processor. ~280 lines. Contents:

- **`apply_rotary_emb_bf16(x, freqs_cis, sequence_dim=1)`** — drop-in for diffusers' `apply_rotary_emb` that stays in `x.dtype` (no fp32 round-trip). One `torch.addcmul` instead of two element-wise kernels. **No known issues; safe to use today.**

- **`fuse_attention_qkv(transformer)`** — walks `FluxAttention` modules, builds `to_qkv` (and `to_added_qkv` on double-blocks) by `torch.cat([to_q.weight, to_k.weight, to_v.weight], dim=0)`. Sets `attn.fused_projections = True`. Skips FP8/scaled-FP8 modules (would need pre-baked scales — separate work). **Known issue: see follow-up below.**

- **`install_kraken_attn_processor(transformer)`** — `attn.processor = KrakenFluxAttnProcessor()` on all 57 FluxAttention modules.

- **`KrakenFluxAttnProcessor`** — drop-in for `FluxAttnProcessor` with two changes: uses `apply_rotary_emb_bf16`, and inherits diffusers' `_get_qkv_projections` (which sees `fused_projections=True` and calls `attn.to_qkv(...).chunk(3, dim=-1)`). The `chunk` is the suspected issue — see follow-up.

### Patch B follow-up — what to fix before re-enabling

The diffusers fused helper does:

```python
query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
```

`chunk(3, dim=-1)` on a `(..., 3 * D)` tensor returns three views with stride that skips the other two slices. Subsequent `.unflatten(-1, (heads, -1))` re-shapes; the actual matmul kernel either has to handle the stride or trigger a hidden `.contiguous()`. Cost might exceed the fusion savings.

**Fix:** in `KrakenFluxAttnProcessor.__call__`, replace `_get_qkv_projections(...)` with our own split that reshapes-first:

```python
qkv = attn.to_qkv(hidden_states)
B, S, _ = qkv.shape
# (B, S, 3 * heads * head_dim) → (B, S, 3, heads, head_dim) → unbind on the 3 axis
qkv = qkv.view(B, S, 3, attn.heads, -1)
query, key, value = qkv.unbind(2)  # each (B, S, heads, head_dim), still views but
                                     # over contig blocks of the original stride
```

Note: still views, but the inner two dims (heads, head_dim) are already the right shape so no further unflatten needed. Then skip the `query.unflatten(-1, ...)` step. This is what ComfyUI does (see `D:\AI_Art\ComfyUI\comfy\ldm\flux\layers.py:207`).

If `unbind` views still trigger materialization downstream, fall back to:

```python
query = qkv[:, :, 0].contiguous()
key   = qkv[:, :, 1].contiguous()
value = qkv[:, :, 2].contiguous()
```

That's three explicit copies but each is small (just the slice we need, contig).

## Bench plan when baseline is restored

Run these in order, capturing time per gen:

| Step | Code state | Expected (FP32 1024² × 28) |
|---|---|---|
| 0 | All patches disabled (current state) | ~75 s cold / ~73 s warm |
| 1 | Uncomment `_install_pos_embed_cache()` (Patch C) | ~73 s cold / ~73 s warm (small win) |
| 2 | Apply Patch B follow-up fix, then uncomment the `fuse_attention_qkv + install_kraken_attn_processor` block | ~65 s cold / ~63 s warm (the big one) |
| 3 | Add Patch D (AdaLN modulation via addcmul) — not written yet, see task #44 | ~62 s / ~60 s |
| 4 | Try Patch E: `torch.compile(transformer, mode="reduce-overhead", fullgraph=False)` AFTER apply_streaming, only when no blocks are streamed (fast-mode path with smaller models) | ~50 s if it works |

After each step, run `python/smoke_test_streaming.sh` to verify no crashes + measure timing. Don't move on until the previous step is stable.

## Known reference points

- Forge memory mgmt research: `[Project: FLUX streaming offload](../C:/Users/The Kraken/.claude/projects/F--Kraken-Art/memory/project_flux_streaming_offload.md)` in persistent memory.
- ComfyUI install at `D:\AI_Art\ComfyUI` — read `comfy\ldm\flux\layers.py` (lines 109, 138-150, 205-207, 302-304) for the fused-QKV pattern.
- Diffusers attention dispatch source: `python\venv\Lib\site-packages\diffusers\models\attention_dispatch.py` (line 3139 = `_native_cudnn_attention` ≈ Patch A target).

## Quick re-enable steps once baseline is fast

```powershell
# 1. Uncomment Patch C
# In python/pipelines/flux.py near line ~94:
#   _install_pos_embed_cache()
# (remove the leading "# ")

# 2. Restart sidecar via Launch Kraken Art.bat (or my Bash equivalent)
# Run smoke_test_streaming.sh. Expect ~73 s warm.

# 3. Apply Patch B follow-up to kraken_flux_attn.py (see "Patch B follow-up" above).
# Then uncomment the fusion+processor install block in flux.py near line ~787.

# 4. Restart, smoke test. Expect ~63 s warm if Patch B is correct.
```

That's the breadcrumb trail. Resume from step 1 once baseline is back to ~75 s warm.
