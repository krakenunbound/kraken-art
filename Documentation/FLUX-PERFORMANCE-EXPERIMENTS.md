# FLUX Performance – Experiment Backlog & Current Gap

**Date of last session:** 2026-05-22 (evening)
**Goal:** Match or beat ComfyUI warm speed on the same RTX 3090 for the golden config:
- Model: Flux 1D FP32 / flux_dev.safetensors + t5xxl_fp16 + clip_l + fluxVaeSft
- Resolution: 1024×1024
- Steps: 28, CFG 1.0
- Sampler/Scheduler: euler / normal (or user's normal choice)

---

## Current Measured Gap (Clean Runs, 2026-05-22)

| Stack                  | Cold     | Warm      | Median step | Notes |
|------------------------|----------|-----------|-------------|-------|
| **ComfyUI 0.19.4** (torch 2.11 + cu130 + comfy-kitchen + DynamicVRAM) | 71.3 s  | **47.2 s / 47.7 s** | ~1.58–1.61 s | Golden reference (Cursor + everything else closed) |
| **Kraken Art** (full Tauri + sidecar, after all Gemini opts) | 74 s    | **64 s**  | ~1.94–1.97 s | Still ~15–17 s slower warm |

**Isolated tool result (same day):** `bench_step_ws.py` on the same workflow got ~51–56 s total (closer, shows full app overhead is significant).

**Conclusion of the day:** The low-level optimizations (2-stream + flat buffers, QKV fusion, bf16 RoPE, prefetch hooks, streaming partition) are **working** and active in the logs, but we are not yet capturing the full win that ComfyUI gets.

---

## What Has Been Implemented & Verified (as of 2026-05-22)

### Core Inference Improvements (all active)
- 2-stream round-robin weight offload + preallocated flat GPU destination buffers (ComfyUI `comfy/model_management.py` pattern)
- QKV fusion + `KrakenFluxAttnProcessor` (bf16-native RoPE via `addcmul`)
- StreamingLinear with block-level prefetch hooks (Phase 3)
- Correct WDDM safety margin (1.5 GB) + measured per-run partition
- KrakenFluxPipeline subclass (skips crashing `maybe_free_model_hooks`)

### Quality-of-Life / Testing Improvements
- Hardened tester-only `Launch Kraken Art.bat` (aggressive port cleanup + explicit venv sidecar + wait loop + Y/N cleanup)
- Automatic "last used" persistence for the entire Generate tab (restores on next launch via `lastGenerate` in `config/settings.json`)
- Clean shutdown verification protocol now documented

---

## Prioritized Experiment List (Next Session)

Ranked by expected impact / ease of test. Do these in order.

### Tier 1 – Quick Wins (try first, 15–30 min each)

1. **Lower Activation Headroom**
   - Settings → FLUX Performance → move slider from 2.0 GB → **1.5 GB** (then 1.3 GB).
   - Re-run the exact same 1024×1028 warm prompt.
   - Expected: pulls 1–3 more blocks onto GPU → 150–400 ms/step improvement.
   - Log the new "budget" line from `kraken.flux`.

2. **Force the partition to keep more blocks**
   - Temporarily edit the budget calculation in `python/pipelines/flux.py` (or add a debug "Force X GB budget" setting) and re-test.
   - Goal: see exactly how many blocks we can keep before we hit OOM or allocator thrash.

3. **Measure pure inference time vs full app time**
   - Run the same job via `bench_step_ws.py` (raw sidecar, no Tauri) while the full app is also open.
   - Compare the two numbers. This will tell us how much of the 15 s gap is "Tauri / React / WebSocket / logging" overhead vs actual inference.

### Tier 2 – Higher Impact (bigger changes)

4. **FP8 flux_dev / fluxmania path**
   - Add an FP8 checkpoint to the test matrix.
   - The auto-check in `flux.py` already does the right thing (falls back to offload only when needed). On FP8 the model is ~11 GB → should mostly stay resident.
   - This is the single biggest lever we have for closing the gap without rewriting the whole stack.

5. **Torch 2.11 + cuDNN / SDPA improvements**
   - Comfy is on 2.11 + cu130. We are on 2.6.
   - Evaluate upgrading the venv (or at least running a controlled A/B with the same torch version).

6. **Remove / minimize sidecar logging during the hot path**
   - The per-step `INFO` lines in the log are cheap but not free. Measure the delta when logging is reduced to WARNING during generation.

### Tier 3 – Future / Nice-to-Have

- Named generation presets + "Save as default" UI on the Generate tab (on top of the automatic last-used we just added).
- Full workspace / `.kraken` project files.
- Optional "Fast Mode" that forces lower headroom + disables some logging when the user explicitly wants max speed.

---

## Current Blocker / Mystery

Even with all the correct low-level pieces in place, the full Tauri + sidecar stack is still ~15 s slower than ComfyUI on the same model and settings.

Two leading hypotheses (in order of likelihood):
1. **UI / communication overhead** (the gap between `bench_step_ws.py` and the real app is large).
2. **Partition is still too conservative** (we are streaming more blocks than ComfyUI's DynamicVRAM does).

The next session should focus on distinguishing between (1) and (2) before we do bigger refactors.

---

## Quick Reference Commands (for future sessions)

**Clean launch for testing (from the hardened .bat):**
```powershell
# Or just double-click the updated "Launch Kraken Art.bat"
```

**Isolated benchmark (fastest feedback loop):**
```powershell
cd "F:\Kraken Art\python"
& "F:\Kraken Art\python\venv\Scripts\python.exe" "bench_step_ws.py" 42 28
```

**Full clean-shutdown verification:**
```powershell
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
  Where-Object { $_.CommandLine -like '*main.py*' -and $_.CommandLine -like '*Kraken*' } |
  Select-Object ProcessId, CommandLine | Format-List

netstat -ano | findstr ":7780" | findstr "LISTENING"
```

---

**Status as of 2026-05-22 22:00 UTC-7**
- All Tier 0 work (persistence + tester launcher) is complete and committed.
- Machine was left in a verified clean state (no orphaned sidecars).
- Ready to continue with Tier 1 experiments (headroom test + overhead measurement) on the next session.

Good luck with your D&D game tonight. See you on the other side of the kraken. 🐙
