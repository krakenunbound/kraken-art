# Ideogram 4 Session Log

Date: 2026-06-17

This note records the Kraken Art Ideogram 4 setup and findings from this session. Do not store Hugging Face tokens here.

## Local Assets

- Official Ideogram 4 NF4 repo: `F:\Kraken Art\models\ideogram4\nf4`
- Official Ideogram 4 FP8 repo: `F:\Kraken Art\models\ideogram4\fp8`
- Transformer Lab Q4_K GGUF repo: `F:\Kraken Art\Ideogram\ideogram-4-gguf-q4_k`
- Q4_K file: `F:\Kraken Art\Ideogram\ideogram-4-gguf-q4_k\ideogram4-q4_k.gguf`
- Transformer Lab INT8 fused repo: `F:\Kraken Art\Ideogram\ideogram-4-int8-fused`
- Transformer Lab INT8 W8A8 weights: `F:\Kraken Art\Ideogram\ideogram-4-int8-w8a8\ideogram4-int8-w8a8.safetensors`

## Sources Checked

- Ideogram OSS inference code: https://github.com/ideogram-oss/ideogram4
- Official NF4 model: https://huggingface.co/ideogram-ai/ideogram-4-nf4
- Transformer Lab Q4_K model: https://huggingface.co/transformerlab/ideogram-4-gguf-q4_k
- Transformer Lab quantization writeup: https://lab.cloud/blog/quantizing-ideogram-4/
- Transformer Lab fused INT8 writeup: https://lab.cloud/blog/fused-int8-ideogram-4/
- Transformer Lab app repo: https://github.com/transformerlab/transformerlab-app

## Kraken Art Changes Made

- Added Ideogram 4 as a distinct architecture/profile so it does not overwrite SDXL, Flux, Z-Image, or video behavior.
- Added model-profile entries for NF4, FP8, Q4_K GGUF, and guarded INT8 fused.
- Added Ideogram magic prompt support:
  - `raw` now passes through only valid Ideogram JSON.
  - Raw non-JSON prose is automatically converted to Kraken local JSON before generation.
  - `local` uses Kraken's local JSON-style expansion.
  - `api` can call Ideogram/OpenRouter style hosted expansion when a key is configured, otherwise falls back to local.
- Enlarged prompt and negative-prompt text areas.
- Added a clear prompt button.
- Added gallery scanning so the image area is not empty after app restart.
- Gallery now shows up to the last 20 images and highlights images from the latest batch.
- Added Ideogram progress events for load, generation steps, and save.
- Disabled the local Ideogram caption verifier in the Kraken local generation path by passing `raise_on_caption_issues=None`.
- Added Ideogram pipeline caching. Same-model repeated runs should reuse the resident Ideogram pipeline instead of unloading it between images.
- Added model-profile warnings for Ideogram variants:
  - NF4 defaults to Turbo 12 for iteration.
  - Q4_K and INT8 fused are marked guarded experimental.
- Guarded experimental Ideogram entries remain visible in the model list but are not selectable for normal generation.
- Added `KRAKEN_IDEOGRAM4_ALLOW_Q4K=1` guard around the in-process Q4_K loader after it crashed the sidecar.
- Added a subprocess worker for Ideogram jobs:
  - Sidecar entrypoint remains `ideogram4.run(job)`.
  - Ideogram model load/generation now happens in `python/pipelines/ideogram4_worker.py`.
  - The worker streams JSON progress/image/done/error events over stdout.
  - The sidecar relays those events to the existing job WebSocket.
  - CUDA hard-crashes now kill the worker job, not the FastAPI sidecar.
- Upgraded the Ideogram worker to persistent mode:
  - First Ideogram job cold-loads the pipeline inside the worker.
  - Repeated Ideogram jobs reuse the resident pipeline.
  - Switching to SDXL/Flux/Z-Image/WAN or pressing Clear VRAM terminates the Ideogram worker to release VRAM.
- Added a UI warning when Ideogram raw prompt mode is selected with non-JSON text.
- Added a backend guard so Ideogram raw non-JSON text cannot bypass the local JSON formatter.

## Settings Behavior

For Ideogram 4, Kraken's sampler, scheduler, and CFG controls are not the main controls. The Ideogram OSS pipeline uses its own presets:

- Turbo: 12 steps
- Default: 20 steps
- Quality: 48 steps

The UI should treat these as the meaningful Ideogram speed/quality controls. The SDXL-style sampler/scheduler/CFG fields should not be used to judge Ideogram behavior unless the upstream Ideogram pipeline adds support for them.

## Safety Filter Finding

The black/gray "Image blocked by safety filter" thumbnail was reproduced with a raw non-JSON prompt. The paired same-prompt, same-seed run using Kraken local JSON produced a normal image.

Kraken now skips Ideogram's local caption verifier for local runs by passing `raise_on_caption_issues=None`, and it also converts raw non-JSON prose to Kraken local JSON before generation. Valid Ideogram JSON can still pass through unchanged.

This does not retrain or change the model weights. If the model itself has placeholder behavior baked into its weights or prompt-format expectations, Kraken can avoid the known trigger path but cannot remove that behavior without a different model or fine-tune.

## NF4 Finding

NF4 is currently the most practical working local path inside Kraken Art on the RTX 3090. It is CUDA-only, supported by Diffusers/the official Ideogram package, and can generate 1024px images locally.

Quality is noticeably prompt-sensitive. Ideogram tends to respond better to structured, scene-rich prompts than older Stable Diffusion tag soup. Negative prompts are less central than with SDXL/Flux workflows.

Expected 1024px timing is still minutes, not seconds. Transformer Lab's fused INT8 post lists prior NF4 at about 164.5 seconds per 1024px image at the benchmark settings. Local end-to-end time can be longer due to cold load, prompt expansion, thermal throttling, and Windows/PyTorch overhead.

## Q4_K GGUF Finding

The Q4_K GGUF is not a llama.cpp, stable-diffusion.cpp, or Ollama-style runnable model. The local README is explicit: it is a quantized DiT-only GGUF that must be loaded through PyTorch with the Ideogram 4 inference package plus FP8/base text encoder and VAE components.

Transformer Lab reports Q4_K as a quality-per-memory win over NF4:

- On disk: about 10.4 GB
- Better than NF4 on their 50-prompt slice
- Preliminary latency: about 203 seconds per 1024px image at 48 steps on RTX 3090
- About 23 percent slower than NF4 in their preliminary benchmark

Kraken test job:

- Job id: `88c0eda0694e47da8f8ea2a0e8bfc4e7`
- Model: `Ideogram 4 Q4_K GGUF (local, experimental)`
- Size: 1024 x 1024
- Steps: 12
- Result: sidecar crashed during cold load before generation started.

Observed cause: the initial Kraken wiring loaded the FP8 base transformer on GPU before swapping Q4_K tensors. That defeats the 3090 memory path. The local Q4_K reference loader also dequantizes GGUF tensors to bf16 PyTorch linears, so if used naively the active tensor volume is much larger than the 10.4 GB on-disk file.

Practical conclusion: Q4_K should remain experimental until Kraken uses a true memory-safe load/offload path. It may be valuable for quality, but it is not the speed fix.

## INT8 Fused Finding

Transformer Lab's later fused INT8 post is the real 3090 speed target, not Q4_K. Their claim:

- 1024px image on one RTX 3090
- Peak memory: 23.40 GB
- End-to-end time: 156.5 seconds/image
- Faster than their listed NF4 and FP8 baselines
- Win is consumer-Ampere-specific because RTX 3090 has useful INT8 tensor cores but no native FP8/FP4 tensor cores

The key technical point is that unfused/fake INT8 dequantizes back to bf16 and calls normal `F.linear`, so it does not use the 3090's INT8 hardware. The fused path uses a custom Triton kernel to keep the multiply on integer tensor cores.

Kraken has a guarded INT8 fused profile because earlier load attempts hard-crashed on this Windows stack. The next serious attempt should be isolated in a subprocess or WSL2-style environment so a failed fused-kernel load cannot take down the main sidecar.

## Transformer Lab App Repo Finding

The `transformerlab/transformerlab-app` repo is an orchestration/research application. A sparse local clone was checked under `F:\Kraken Art\tmp\transformerlab-app`. Searching it found the Ideogram blog source and general GGUF/model-management code, but no hidden drop-in Ideogram runner that replaces the HF Q4_K or fused INT8 reference assets.

For Kraken Art, the actionable pieces are still the Hugging Face repos, the included Q4_K loader/recipe, and the fused INT8 kernel package.

## Recommended Current Use

Use NF4 for normal Kraken Art Ideogram work:

- Architecture: `Ideogram 4 (open weights)`
- Model: `Ideogram 4 NF4 (local)`
- First comparison preset: `Turbo 12`
- Quality comparison preset: `Quality 48`
- Minimum size for meaningful tests: 1024 x 1024
- Portrait comparison: 1024 x 1536
- Batch count: 1 while measuring speed or VRAM

Only use Q4_K for controlled testing after the loader is corrected to avoid FP8-GPU cold load and avoid full bf16 expansion when possible.

Only use INT8 fused after isolating it from the main sidecar and verifying the Windows/Triton/CUDA stack can actually load the fused kernel.

## Open Work

- Restore the sidecar after the Q4_K crash before more UI testing.
- Replace the current Q4_K path with a memory-safe implementation or keep it hidden/experimental.
- Consider a separate worker process for Ideogram so hard CUDA failures do not kill the main app.
- Investigate WSL2 or a pinned Linux CUDA environment for Transformer Lab fused INT8, since the blog path is aimed at consumer Ampere but not necessarily this Windows venv.
- Add clearer UI copy for Ideogram controls: presets matter; SDXL sampler/scheduler/CFG do not.
- Add better cold-load progress, because generation-step progress only starts after the model is fully ready.

## Test Checklist

Use this section as the working checklist. Mark each item with date, model, settings, result, output path, elapsed time, and any crash/log detail.

### App And UI

- [x] Restart sidecar after Q4_K crash and verify `/health` returns ok. Result: sidecar healthy on PID `20980` after worker patch.
- [ ] Open Kraken Art and confirm the app reconnects to the restarted sidecar.
- [ ] Confirm Ideogram model profiles appear without hiding SDXL/Flux/Z-Image profiles.
- [ ] Confirm sampler/scheduler/CFG are hidden or clearly de-emphasized for Ideogram.
- [ ] Confirm `Turbo 12`, `Default 20`, and `Quality 48` preset buttons set the intended step count.
- [ ] Confirm prompt and negative prompt boxes are tall enough at full-screen and normal window sizes.
- [ ] Confirm clear prompt button empties the prompt without altering model settings.
- [ ] Confirm magic prompt apply works for `Kraken local JSON`.
- [ ] Confirm gallery loads the last 20 images after app restart.
- [ ] Confirm newest image appears in the top-left gallery slot.
- [ ] Confirm latest batch images get the highlighted border/badge.
- [ ] Confirm clicking/opening gallery images works from the displayed thumbnails.
- [x] Confirm old blocked-placeholder images still display but new local Ideogram jobs do not use the caption safety verifier. Result: caption verifier is bypassed, but raw non-JSON prompts can still generate the placeholder as model output. UI now warns against raw non-JSON prompts.

### NF4 Baseline

- [x] Run NF4 1024 x 1024, Turbo 12, batch 1, simple prompt; record cold-load time and generation time. Result: failed before generation. Worker exited with code `3221225477` (`0xC0000005`, Windows access violation) while loading `F:\Kraken Art\models\ideogram4\nf4`.
- [x] Run NF4 1024 x 1024, Turbo 12, batch 1 again without restarting; confirm pipeline cache hit and record warm-run time. Result after persistent-worker fix: pipeline cache hit, ready in `0.0s`, total job `60.5s`.
- [ ] Run NF4 1024 x 1024, Default 20, batch 1; record time and quality difference.
- [ ] Run NF4 1024 x 1024, Quality 48, batch 1; record time and quality difference.
- [ ] Run NF4 1024 x 1536 portrait, Turbo 12, batch 1 using the gothic character prompt; compare against user-provided Ideogram sample.
- [ ] Run NF4 1024 x 1536 portrait, Quality 48, batch 1 using the gothic character prompt; compare quality and time.
- [ ] Run NF4 1024 x 1536 portrait, Turbo 12, batch 1 using the 70s beach portrait prompt; compare realism and anatomy.
- [ ] Run NF4 1024 x 1024, Turbo 12, batch 1 using the mouse/trolley scene prompt; check object scale and scene adherence.
- [ ] Check VRAM during NF4 cold load, generation, and post-generation idle state.
- [ ] Confirm NF4 remains resident between consecutive Ideogram jobs.
- [ ] Confirm switching from NF4 to SDXL unloads Ideogram cleanly enough for SDXL to run.

### Prompt Expansion

- [x] Compare raw prompt vs Kraken local JSON prompt on the same NF4 seed. Result: raw non-JSON text produced a black "Image blocked by safety filter" image as model output; Kraken local JSON produced real images.
- [ ] Compare short prompt vs deconstructed prompt on the same NF4 seed.
- [ ] Confirm JSON prompt text is passed intact to Ideogram when already present.
- [ ] Confirm local magic prompt does not add SDXL-style low-value tag soup.
- [ ] Confirm negative prompt is not over-weighted for Ideogram output quality.

### Progress And Logs

- [ ] Confirm UI shows "Loading Ideogram 4 ..." immediately on cold load.
- [ ] Confirm UI shows step progress after the first visible Ideogram step.
- [ ] Confirm job logs report first visible step delay.
- [ ] Confirm job logs report periodic step timings.
- [ ] Confirm failed jobs produce a readable UI error instead of leaving the button stuck.
- [ ] Confirm sidecar crash is detected by the UI and recovers after restart.

### Q4_K GGUF

- [x] Move/download Q4_K GGUF into `F:\Kraken Art\Ideogram\ideogram-4-gguf-q4_k`.
- [x] Add Q4_K as an experimental selectable Kraken Art profile.
- [x] Attempt Q4_K 1024 x 1024, Turbo 12, batch 1.
- [x] Record Q4_K result: sidecar crashed during FP8 base cold load before generation.
- [x] Hide Q4_K by default or add warning copy until loader is corrected. Result: UI profile now warns, normal selection disables guarded entries, and backend refuses Q4_K unless `KRAKEN_IDEOGRAM4_ALLOW_Q4K=1`.
- [x] Re-test Q4_K guard through subprocess worker. Result: job `00102237ceed480cbd5823709a6c6c9d` failed cleanly with guard error and `/health` stayed ok.
- [ ] Implement Q4_K memory-safe loading that avoids full FP8 transformer load on GPU first.
- [ ] Re-test Q4_K 1024 x 1024, Turbo 12, batch 1 after loader correction.
- [ ] If Q4_K loads, run warm-run cache test to see if repeat generation avoids cold load.
- [ ] If Q4_K loads, run 1024 x 1024, Quality 48, batch 1 and compare against NF4 Quality 48.
- [ ] If Q4_K loads, record peak VRAM and elapsed time.

### INT8 Fused

- [x] Download/store fused INT8 repo and W8A8 weights locally.
- [x] Add guarded INT8 fused profile.
- [x] Record current status: direct Windows sidecar load attempts have hard-crashed.
- [ ] Decide whether to test fused INT8 in a separate worker process or WSL2 environment.
- [ ] Verify Triton/CUDA compatibility for fused INT8 without loading the full Kraken sidecar.
- [ ] Run fused INT8 kernel import-only smoke test.
- [ ] Run fused INT8 model-load-only smoke test.
- [ ] Run fused INT8 1024 x 1024, Turbo 12, batch 1 if load-only passes.
- [ ] Run fused INT8 1024 x 1024, Quality 48, batch 1 if Turbo passes.
- [ ] Compare fused INT8 timing against NF4 on the same prompt and dimensions.
- [ ] Record peak VRAM and whether it stays under 24 GB.

### Stability And Safety

- [ ] Confirm no Hugging Face token or private credential is stored in project files.
- [x] Confirm failed Ideogram loads do not corrupt gallery metadata or outputs. Result: Q4_K guard failure and NF4 access-violation failure produced no output files and did not break sidecar health.
- [ ] Confirm output PNG metadata records model, preset/steps, prompt mode, seed, and dimensions.
- [ ] Confirm Clear VRAM unloads the active Ideogram pipeline when requested.
- [x] Confirm app can recover after a CUDA OOM or sidecar crash without rebooting Windows. Result: after worker patch, Ideogram worker crash left `/health` true; no Windows reboot needed.

## Test Result — Ideogram Worker Isolation

Date: 2026-06-17

The subprocess worker design has been implemented and smoke-tested.

- Q4_K guarded job: `00102237ceed480cbd5823709a6c6c9d`
  - Result: failed cleanly with the intended guard message.
  - Sidecar health after failure: ok.
- NF4 Turbo 12 job: `37eabe4b280c4a8cb85e30330235fed7`
  - Settings: `1024 x 1024`, 12 steps, batch 1, seed `123456`, Kraken local JSON magic prompt.
  - Result: worker exited with code `3221225477`.
  - Interpretation: `3221225477` is `0xC0000005`, a Windows access violation.
  - Last worker output: loading Ideogram 4 NF4 from `F:\Kraken Art\models\ideogram4\nf4` on CUDA.
  - Sidecar health after failure: ok.

Runtime probe after the NF4 crash:

- torch: `2.6.0+cu124`
- torch CUDA runtime: `12.4`
- CUDA available: yes
- GPU capability: `(8, 6)` / RTX 3090
- bitsandbytes: `0.49.2`
- diffusers: `0.38.0`
- transformers: `4.57.6`
- accelerate: `1.13.0`
- gguf import: ok

Conclusion: basic imports are not the failure. The failure happens during the Ideogram NF4 pipeline weight load and is a native access violation, not a Python exception. Worker isolation is now doing its job; the next investigation should isolate the faulting native module or try a known-good dependency stack.

## Test Result — NF4 Turbo 12 Cold And Warm Cache

Date: 2026-06-17

After implementing persistent worker mode and removing the accidental per-job `unload()` call, NF4 caching now works.

Cold job:

- Job id: `5a7a337f073043f1adeba19d32a1b14a`
- Output: `F:\Kraken Art\outputs\2026-06-17\000849-5a7a337f-00.png`
- Settings: NF4 local, `1024 x 1024`, Turbo 12, batch 1, seed `333001`, Kraken local JSON magic prompt.
- Pipeline cached: false
- Pipeline ready: `104.6s`
- Generation: `60.7s`
- Total job: `169.7s`
- Peak observed VRAM after cold job: about `22.3 GB` resident.

Warm job:

- Job id: `df1dc5aaec3c48b6a89c4deaf8fe8b55`
- Output: `F:\Kraken Art\outputs\2026-06-17\001037-df1dc5aa-00.png`
- Settings: same as cold job, seed `333002`.
- Pipeline cached: true
- Pipeline ready: `0.0s`
- Generation: `60.0s`
- Total job: `60.5s`
- Worker remained resident after completion, so repeated Ideogram runs should now avoid the 104-107 second setup cost.

Prompt-format finding:

- Raw non-JSON prompt output: `F:\Kraken Art\outputs\2026-06-16\234909-6631e841-00.png`
- Result: model produced a black "Image blocked by safety filter" placeholder image.
- Kraken local JSON output: `F:\Kraken Art\outputs\2026-06-16\235358-509172d6-00.png`
- Result: model produced a real image.

Interpretation: for this local Ideogram open-weights path, raw prose is not reliable. Use Kraken local JSON magic prompt or paste valid Ideogram JSON.

Follow-up change: raw non-JSON prose is now forced through Kraken local JSON at generation time, even if the UI is set to raw/off. Raw passthrough remains available only when the prompt text starts as JSON.

---

# Addendum — 2026-06-17 — Research pass on "fast Ideogram 4 on a 3090"

Research only. No pipeline code was changed and no Ideogram job was run to
produce this section. It reconciles the Transformer Lab writeups against the
files already on disk and against the venv, and proposes a concrete fold-in
plan. Sources are listed at the bottom.

## TL;DR

- "Fast Ideogram 4 on a 3090" tops out at about **156.5 s/image** (1024px,
  48 steps, single card) for Transformer Lab's **fused INT8** build. That is
  the *fastest of all Ideogram 4 variants on consumer Ampere* but only about
  **5 percent faster than NF4** (164.5 s), which already runs in Kraken Art
  today. It is not "fast" in the FLUX/seconds sense.
- The single biggest latency lever is **step count**, not quantization.
  Presets are Turbo 12 / Default 20 / Quality 48. The 156.5 s number is at
  48 steps; Turbo 12 is roughly 4x faster (about 40 s/image on NF4) with no
  new code.
- The fused INT8 kernel is a **quality-at-equal-speed** upgrade (FP8-ceiling
  quality, fits on one 24 GB card), not a speed upgrade.
- The blocker to lighting up INT8 fused on this machine is **not Triton** —
  triton 3.2.0 imports fine in the venv. It is **memory during the weight
  swap** (23.4 GB peak on a 24 GB card) plus the fact that a CUDA hard-crash
  during load currently takes down the whole sidecar. The fix is a
  memory-safe load inside an **isolated subprocess worker**.

## The mechanism (what Transformer Lab actually did)

The entire trick is in one file already on disk:
`Ideogram/ideogram-4-int8-fused/triton_int8_gemm.py`.

- The RTX 3090 (sm_86) has native **INT8 tensor cores** but **no FP8/FP4
  tensor cores**. FP8 and NF4 therefore both run as bf16 matmuls on this
  card.
- Standard "INT8" quant is *fake-quant*: it quantizes, then **dequantizes
  back to bf16** and calls `F.linear`, never touching the INT8 hardware — so
  "INT8" ends up *slower* than FP8 (184-185 s vs 172.9 s).
- The fused kernel runs the matmul as `int8 x int8 -> int32` on Ampere
  `mma.s8` units and folds the per-token activation scale x per-channel
  weight scale x bias into the GEMM epilogue, off the int32 accumulator.
  Per-GEMM that is **2.8-4.2x faster than bf16** and bit-exact against
  `torch._int_mm`. Quant recipe = SmoothQuant + per-token dynamic INT8
  activations + per-channel INT8 weights, with ~17 fragility-prone FFN
  down-projections (~8 percent of layers) kept in bf16.

Tricks that do NOT help and should not be chased:
- `torch.compile` — "gives effectively zero gain because the compiler
  graph-breaks at the custom linears." Our `FusedW8A8.forward` is already
  `@torch._dynamo.disable`.
- SageAttention-INT8 — blocked by head-dimension incompatibility.
- No CUDA graphs, no channels_last in their build.

## Measured numbers (RTX 3090, 1024px, 48 steps), verbatim from the source

| Variant | s/image | GPUs | Notes |
|---|---|---|---|
| Fused INT8 (Triton kernel) | 156.5 | 1 | fastest on a 3090; 23.40 GB peak |
| NF4 (published) | 164.5 | 1 | our current working path |
| FP8 (published) | 172.9 | 2 | needs two cards for bf16/FP8 |
| INT8 W8A8, fake-quant (no fused kernel) | 184-185 | 2 | dequant-to-bf16, never uses INT8 cores |
| Q4_K GGUF | ~203 | 1 | quality win over NF4, but SLOWER |

Quality: INT8 holds the FP8 ceiling and beats NF4 by +0.84 Pick / +2.93
CLIP on their 50-prompt slice. Q4_K also beats NF4 on quality. These are
single-run point estimates; small margins are within run-to-run variance.

Scope caveat from the source: the INT8 speed win is **consumer-Ampere
specific**. On A100/B200 a plain bf16 matmul is faster, so the kernel is
autotuned for sm_86 and would need retuning elsewhere. Fine for us — we are
sm_86.

## What is already wired in Kraken Art (verified by reading the code)

`python/pipelines/ideogram4.py` already implements all four quant paths:

- **NF4** — works today via bitsandbytes. Recommended path.
- **Q4_K GGUF** — coded, but `_swap_branch_from_gguf_reader` dequantizes
  each GGUF tensor to a **bf16** `nn.Linear`, so the 10.4 GB on-disk file
  becomes a much larger resident tensor volume. Slower than NF4 (203 s).
  Quality experiment, not a speed path.
- **INT8 fused** — fully coded in `_install_fused_int8`, and actually
  *smarter* than the reference `fused_int8.load_fused_int8`: it CPU-stages
  the text encoder + VAE during the swap and replaces each child with
  `Identity()` + frees before reading the INT8 replacement. Guarded behind
  `KRAKEN_IDEOGRAM4_ALLOW_INT8_FUSED=1` because earlier load attempts hard-
  crashed this Windows stack.
- **Structured-JSON magic prompt** — wired and in active use (the saved
  `lastGenerate` carries a full `high_level_description` +
  `compositional_deconstruction` JSON, which is exactly the schema Ideogram
  4 validates against).

## Venv reality check (read-only probe, no job run)

- `triton` imports: version **3.2.0**. `triton_windows` reports missing as a
  separate module name, which is expected — the Windows-capable Triton
  registers under the `triton` namespace, and it is importable.
- torch **2.6.0+cu124**, CUDA **12.4**, GPU **RTX 3090**, compute capability
  **(8, 6) = sm_86**. INT8 tensor cores present.
- `bitsandbytes` and `gguf` both installed.

Conclusion: every hardware/software prerequisite for the fused kernel is
present. The recorded crashes in this doc ("FP8 base load, NF4 scaffold
swap, NF4 swap with text/VAE CPU staging") are all **memory failures during
the weight swap, before the Triton kernel ever executes**. The kernel itself
is not the suspect.

## The real blocker, stated precisely

The INT8 build peaks at **23.40 GB on a 24 GB card** — almost no headroom.
The load sequence in `usage.py` (and our loader) brings up the full FP8 base
pipeline on CUDA first (text encoder + VAE + both FP8 DiT branches), then
swaps each FP8 linear for a `FusedW8A8` holding the INT8 weight. During the
swap a layer transiently holds both its outgoing FP8 weight and its incoming
INT8 weight. On a card already near its ceiling, one bad transient is an OOM
or an unrecoverable CUDA fault — and because everything runs in the one
FastAPI sidecar process, that fault kills image + audio + the whole app.

So two things stand between us and the fast/quality INT8 path:

1. A **memory-safe load** that never has the full FP8 base co-resident with
   the INT8 weights. Ideally the base DiT branches are created on `meta` /
   CPU and each linear is materialized straight into its INT8 replacement,
   so the FP8 weight for a layer never lands on CUDA at all.
2. **Process isolation** so a CUDA hard-crash during load cannot take the
   sidecar down with it.

## Prioritized fold-in plan (proposed — not yet executed)

1. **Ship "fast Ideogram" = NF4 + Turbo 12 now.** Zero new code. ~40 s/image,
   already working. This is the honest answer to "fast on a 3090." Add UI
   copy that says the preset (not the quant) is the speed dial.
2. **Add a subprocess worker for Ideogram** (design below). Isolates CUDA
   crashes from the sidecar and unblocks safe INT8/Q4_K experimentation.
3. **Inside the worker, make the INT8 load memory-safe**, then drop the
   `KRAKEN_IDEOGRAM4_ALLOW_INT8_FUSED` guard. Payoff: FP8-ceiling quality at
   NF4 speed on a single card.
4. **Demote Q4_K to clearly-experimental** (slower than NF4, not memory-safe
   as written). Keep for quality A/B only.

---

# Design — Ideogram subprocess worker (step 2 of the fold-in plan)

Status: DESIGN ONLY. No files created. For approval before any code.

## Goal

Run every Ideogram 4 job in a short-lived child process so that a CUDA OOM,
an unrecoverable CUDA fault, or a Triton compile failure kills only that
child — never the FastAPI sidecar (which now owns image, audio, and cover
art). Keep the existing `/api/generate` contract and WebSocket progress
unchanged from the UI's point of view.

## Why a subprocess and not a thread

A Python thread shares the process address space and the CUDA context. A
hard CUDA fault (illegal memory access, sticky OOM) is not catchable in
Python and corrupts the context for every other thread — image and audio
included. A separate process has its own CUDA context; when it dies, the OS
reclaims its VRAM and the sidecar is untouched. This is exactly the
"separate worker process" the Open Work section already calls for.

## Shape

- New file `python/pipelines/ideogram_worker.py` — a standalone module with
  a `__main__` entry point. It imports torch + the Ideogram pipeline, reads
  one job spec, generates, writes images, streams progress, exits. It does
  NOT import FastAPI or the job manager — it is a leaf process.
- New thin driver in `python/pipelines/ideogram4.py` (or a sibling
  `ideogram_runner.py`) that the existing `ideogram4.run(job)` delegates to:
  spawn the worker, feed it the job spec, pump its progress events back into
  `job.emit(...)`, collect the result, reap the child.
- The current in-process `ideogram4.py` load/generate code moves into the
  worker mostly verbatim — the loaders (`_ensure_pipeline`, the NF4 / Q4_K /
  INT8 paths) are already written; they just run in the child now.

## IPC

Keep it boring and Windows-friendly:

- **Job spec in:** a single JSON blob on the worker's argv or stdin
  (model name, prompt, width/height, steps->preset, seed, count, output dir,
  prompt mode). No pickling of live objects.
- **Progress + result out:** the worker prints one JSON object per line to
  stdout (`{"type":"progress","step":..,"total":..}`,
  `{"type":"image","path":..,"seed":..}`, final
  `{"type":"done","outputs":[...]}` or `{"type":"error","error":".."}`).
  The driver reads stdout line by line and re-emits each into the existing
  `job.emit(...)` so the WebSocket UI is byte-for-byte the same as today.
- **Cancellation:** the driver watches `job.cancel`; on cancel it terminates
  the child (`Popen.terminate()`, then `kill()` after a grace period). No
  cooperative-cancel plumbing across the boundary needed for v1 — killing
  the process is the cancel.
- Images are written by the worker directly to `OUTPUTS_ROOT/<date>/...`
  (same as now) and only the paths cross the boundary, so no large binary
  IPC. Thumbnails for the live preview can be base64 in the `image` line
  (small) or generated by the driver from the written PNG.

## Lifecycle and VRAM

- **One job = one process.** Spawn on job start, exit on job end. This
  trades the warm-pipeline cache (cold load every job, tens of seconds) for
  crash isolation and guaranteed VRAM reclaim. For a 2.5-minute generation
  the cold-load amortizes acceptably; for rapid Turbo-12 iteration it hurts.
- **Optional v2: a persistent worker** that stays resident across jobs to
  keep the pipeline cached (restoring today's warm-run behavior), with the
  driver health-checking it and respawning on death. More plumbing; defer
  until v1 proves the isolation works.
- Because the child exits, the sidecar's own `Clear VRAM` and engine-switch
  logic no longer has to fight Ideogram for the card — when no Ideogram job
  is running, there is no Ideogram process and no Ideogram VRAM.

## Memory-safe INT8 load (lands inside the worker, step 3)

Independent of the IPC work, the INT8 path needs the base DiT branches to
never fully materialize on CUDA before the swap. Options to evaluate in the
worker, in increasing order of effort:

1. Construct the base pipeline with the two DiT branches on `meta`/CPU, then
   materialize each linear directly as its `FusedW8A8` (or bf16-protected)
   replacement, so a layer's FP8 weight never reaches CUDA. The existing
   `_install_fused_int8` is close; it currently loads the base on-device
   first.
2. If the Ideogram pipeline constructor will not accept a meta/CPU device
   for the DiT, fall back to the current swap-in-place but with the text
   encoder + VAE kept on CPU for the entire swap (we already do this) and a
   tighter per-layer free + `empty_cache` cadence.
3. Confirm peak stays under 24 GB with `torch.cuda.max_memory_allocated`
   logging at each phase before dropping the guard.

## What stays the same

- The `/api/generate` request/response contract.
- The `/ws/jobs/{id}` event stream shape.
- The model selector entries (NF4 / FP8 / Q4_K / INT8 fused virtual models).
- The structured-JSON magic-prompt expansion (runs in the sidecar before the
  worker spawns, or in the worker — either is fine; sidecar-side keeps the
  worker leaner).
- NF4 and Q4_K ride the same worker for free; this is not INT8-specific.

## Risks / open questions for approval

- Cold-load-per-job latency vs. crash isolation — accept for v1, revisit
  with a persistent worker if Turbo-12 iteration feels sluggish.
- Windows `Popen` + line-buffered stdout from a torch child can buffer; the
  worker must `flush=True` every progress line (or run with `-u`).
- Triton's first-call autotune happens on the first denoise step inside the
  child; that one-time cost is per-process, so per-job spawning pays it each
  time. Persistent worker (v2) amortizes it.
- Need to confirm the gated base repo `ideogram-ai/ideogram-4-fp8` is
  present locally (the loaders reference `LOCAL_FP8_REPO`) so the worker
  does not block on a gated download mid-job.

## Sources (this addendum)

- Quantizing Ideogram 4.0 onto a 3090 — https://lab.cloud/blog/quantizing-ideogram-4/
- Fused INT8 Ideogram 4 — https://lab.cloud/blog/fused-int8-ideogram-4/
- transformerlab/ideogram-4-gguf-q4_k — https://huggingface.co/transformerlab/ideogram-4-gguf-q4_k
- transformerlab/transformerlab-app — https://github.com/transformerlab/transformerlab-app
- Local files: `Ideogram/ideogram-4-int8-fused/{fused_int8,triton_int8_gemm,usage}.py`,
  `Ideogram/ideogram-4-int8-fused/README.md`, `python/pipelines/ideogram4.py`

---

# RESULT — 2026-06-17 — Adaptive velocity-cache: 15 min → ~2.0 min at near-full quality

This is the solution. Measured on the actual machine (RTX 3090, Windows, driver
610.47, torch 2.6+cu124), NF4, 1024², seed-matched, structured-JSON prompt.

## The real cause of "15 minutes"

Not the hardware. The user's saved settings had them on the **Q4_K GGUF**, which
unpacks to bf16 at runtime, overflows 24 GB, and spills into Windows shared
memory (10–50× slower). **NF4 fits** (peak 22.3 GB dedicated, no spill) and runs
at full speed. Switching NF4 alone: 15 min → 4.3 min at 48 steps.

## The speedup: adaptive velocity-cache (TeaCache-style)

Ideogram 4 runs TWO transformer passes per step (asymmetric CFG). The velocity it
produces changes a lot in the first/last few steps and barely in the middle. We
recompute the two passes only when the latent has moved more than a relative-L1
threshold since the last compute; always recompute the first `warmup` and last
`cooldown` steps. Flow-matching is remarkably robust to this — **no artifacts** at
any setting tested.

Measured frontier (48-step Quality preset, gen time, warm pipeline):

| Mode | threshold | forwards | gen time | quality |
|---|---|---|---|---|
| Max (full) | 0 | 48/48 | 4.32 min | reference |
| **High (default)** | **0.08** | **21/48** | **~2.0 min** | **≈ full, recommended** |
| Fast (draft) | 0.12 | 17/48 | ~1.6 min | great, slightly softer |

Other strategies measured (all artifact-free): fixed stride-2 (24 fwd, 2.27 min),
stride-3 (16 fwd, 1.51 min), bookend (full first6+last6 + stride mid, 30 fwd,
2.87 min). The **adaptive** scheme was the best quality-per-time and is what
shipped.

Dead ends ruled out by measurement:
- SDPA backend forcing: default already uses the efficient backend; MATH is 6×
  slower; CUDNN has no kernel for the segment mask; TF32 + cudnn.benchmark were
  slightly slower. No free win there.
- torch.compile: graph-breaks on the custom layers (TransformerLab confirmed) —
  ~zero gain.
- INT8 fused kernel: faster AND higher quality, but blocked on Windows by WDDM
  host-commit during the load (three crash experiments, all access violations).
  WSL2/Linux would unlock it. Not needed — adaptive NF4 already beats the goal.

## End-to-end validation (real product path)

Ran `pipelines.ideogram4._run_inprocess` (the exact function the worker calls) at
`speed_mode=high`: 223 s total **including** the ~110 s cold model load → ~1.9 min
of actual generation, 1024×1024, image emitted correctly through the job pipeline.
With Codex's persistent worker the pipeline stays warm, so every image after the
first is just the ~1.9 min.

## What shipped (this session)

- `python/pipelines/ideogram4.py` — new `_generate_cached()` (the adaptive cache,
  mirrors the vendored `__call__`; threshold=0 reproduces the exact full
  trajectory) + `_ideogram_speed_params()`; `_run_inprocess` now calls it instead
  of `pipe(...)`.
- `python/api/generate.py` — `ideogram_speed_mode` field (max | high | fast),
  default `high`.
- `src/Generate.tsx` + `src/api/sidecar.ts` — Speed dropdown for Ideogram,
  persisted in lastGenerate.

## Net for the user

15 min (Q4_K, spilling) → **~2.0 min at near-identical max quality** (NF4 +
adaptive cache, High mode) — a ~7.5× speedup, under the 3-min ideal. Fast mode
~1.6 min for drafts; Max mode 4.3 min for the purist reference. Bigger final
images via generate-at-1MP-then-upscale (native 2K spills; don't).

---

# CAVEAT — shared VRAM (2026-06-17)

The speed numbers above (NF4 fits at 22.3 GB peak; ~2.0 min High mode) were
measured on a near-idle 24 GB card. In normal use the 3090's VRAM is shared
with the Kraken app itself and with other concurrent GPU work (e.g. a Codex
D&D-campaign assistant). If free VRAM drops below ~2.5 GB while an Ideogram
job runs, NF4 can tip into Windows shared-memory spill and slow down
sharply — the same mechanism that caused the original 15 min. Practical
guidance: treat the benchmark numbers as best-case; for guaranteed fast runs,
make sure the card is mostly free (the launcher's engine-wipe + the persistent
worker help). The "fit in 24 GB" work matters MORE under contention, not less.
A future hardening step: have the Ideogram worker check free VRAM before a job
and, if low, fall back to NF4-with-text-encoder-offload (frees ~5–8 GB) so it
still fits.

---

# MAGIC PROMPT — how it works + what we can improve (2026-06-17 research)

## The key nuance most people miss

There are TWO different prompt worlds for Ideogram 4:

1. **Cloud Ideogram (ideogram.ai):** you type plain **natural language**
   (~150–160 words is the sweet spot; quotes for in-image text; no weights,
   no `--ar`, no hex codes). "Magic Prompt = On/Auto" then runs Ideogram's
   server-side LLM that rewrites your short idea into a long, cinematic "hero
   prompt" (subject → environment → camera/lighting → textures). You never see
   the expansion; it just happens.

2. **Local open-weights (what Kraken runs):** the model was trained on a
   **structured JSON caption** — `high_level_description` +
   `compositional_deconstruction` (a `background` plus an `elements` list with
   `bbox` coordinates on a 0–1000 grid, and `text` elements carrying the exact
   in-image string). The local inference package literally validates this
   schema. So locally, the JSON **is** the model's native language; the
   "magic" is converting human intent into a great JSON caption.

So "magic prompt" = automatic prompt expansion. On the cloud it's an LLM.
Locally there's no LLM doing it for you unless you provide one.

## What Kraken has today (python/pipelines/ideogram4_magic.py)

A **deterministic, rule-based** JSON builder (no LLM, no VRAM). It:
- detects photo vs art by keyword lists,
- has a few hardcoded style branches (gothic / beach-70s / mouse-children /
  logo-poster / default),
- pulls quoted text into `text` elements with computed bboxes,
- emits the `high_level_description` + `style_description` +
  `compositional_deconstruction` JSON.
Plus an `api` mode (calls a hosted magic-prompt endpoint if an OpenRouter/
Ideogram key is set) and a `raw` passthrough (used when you paste JSON yourself).

It works, but the expansion is shallow — fixed branches, generic filler. It's a
floor, not a ceiling.

## The honest quality model

For the LOCAL model, **image quality is gated by the quality of the JSON
caption.** A rich, well-structured caption beats a thin one every time. So the
whole game is: produce the best JSON. Three ways, in order of output quality:

1. **A frontier LLM writes the JSON** (Claude / GPT / Gemini / Grok) — best
   results, because these models are far better art directors than any rule
   engine or small local model. This is exactly the user's current habit
   (copy-paste from an external LLM).
2. **A small local LLM writes the JSON** — replicates cloud magic prompt
   offline, but costs VRAM/complexity and is weaker than frontier models.
3. **Deterministic builder** (current) — instant, offline, free, but shallow.

## Recommended improvements (priority order, VRAM-aware)

Given the shared-VRAM constraint, do NOT add a local LLM that competes with the
image model for the 24 GB. Instead:

**P1 — BYO-LLM bridge (highest leverage, zero image-time VRAM).**
Add an optional "expand with LLM" that calls the user's chosen API (Claude /
OpenRouter / or Song Studio's already-running assistant on :8010) with a
curated Ideogram system prompt, returns a validated JSON caption, and fills the
prompt box. This replicates cloud magic prompt using a *frontier* model, runs
before the image model loads, and uses no local VRAM. Reuses the existing
`ideogram_magic_mode = "api"` plumbing.

**P2 — A copyable "Ideogram JSON system prompt" + paste-and-validate.**
Ship a battle-tested system prompt in the UI (one click "Copy") so the user can
paste it into Claude/ChatGPT/Grok, get perfect JSON, paste it back — and a
"Format / validate" button that checks the JSON against the schema and fixes
common issues (missing fields, bad bbox ranges, smart-quotes). Zero infra;
directly upgrades the user's existing copy-paste workflow.

**P3 — Enhance the deterministic builder into a "slot machine."**
Add structured pickers — Style / Lighting / Camera / Mood / Composition — backed
by curated phrase libraries drawn from Ideogram's own prompting guide (concrete
visual grounding: "deep red", "golden hour", "35mm film", "volumetric light",
"shallow depth of field"). Build the JSON from picks + free-text subject, target
~150–160 words. Offline fallback when no LLM is configured; much stronger than
the current 5 hardcoded branches.

**P4 — Template gallery** from EvoLinkAI/awesome-ideogram-4.0-prompts (poster,
product packaging, portrait, typography, brand-system). Curated proven captions
the user picks and fills — instant good results, teaches the format by example.

## Sources
- Ideogram Magic Prompt: https://docs.ideogram.ai/using-ideogram/generation-settings/magic-prompt
- Prompting fundamentals: https://docs.ideogram.ai/using-ideogram/prompting-guide/2-prompting-fundamentals
- Example prompts: https://github.com/EvoLinkAI/awesome-ideogram-4.0-prompts
- User-provided notes: Downloads/"is it possible without A.I. to create a prompt ge....md"
  (deterministic slot-machine builder) + Downloads/"https___docs.ideogram.ai_…md"
  (4-pillar expansion blueprint).
