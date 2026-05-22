# Known issues

Active bugs and limitations as of 2026-05-21. Each entry: what's broken, the symptom, the suspected cause, and the current workaround.

---

## Open

### B-002 · FLUX generation slower than ComfyUI (partially fixed 2026-05-21)
**Status:** mostly resolved — was ~140 s, now ~73 s warm. Remaining gap to ComfyUI's 48 s is true layer prefetch (see CHANGELOG → "FLUX speed pass").
**History:** The first cut used diffusers' `enable_model_cpu_offload`, which moves entire submodules CPU↔GPU per forward — fine for SDXL, devastating for a 22 GB FLUX transformer (~138 s/gen on a 24 GB 3090). The fix was the Forge/Fooocus-style **StreamingLinear partition**: every `nn.Linear` in the transformer becomes a `StreamingLinear`; `apply_streaming` keeps as many transformer blocks fully resident on GPU as the VRAM budget allows, streams the rest from pinned host memory via a dedicated CUDA mover stream.
**Result:** FP32 flux_dev now runs in ~75 s cold / ~73 s warm on a 3090, stable across follow-up gens and arch switches. See `pipelines/streaming_linear.py` + the FLUX section of PIPELINES.md.
**Remaining work:** ComfyUI still wins on warm (48 s) because it prefetches the *next* layer's weight on the mover stream while the *current* layer is computing. Our mover stream is currently sync-waited per layer. Implementing true prefetch needs a pre-forward hook on each transformer block that schedules H2D for child Linears in dependency order. Estimated 15–25 s savings — would close most of the gap.

### B-003 · Cancel during model load is a no-op
**Status:** known limitation.
**Symptom:** Clicking Cancel while a job is loading the transformer/VAE/text-encoders returns `409: job not cancellable`. Only cancels mid-sampling.
**Cause:** Cancel is checked in `callback_on_step_end`, which only fires during diffusion steps. The model-load phase is a single long synchronous call to `set_module_tensor_to_device` / `from_single_file` with no cancel points.
**Path forward:** True load cancel needs either a kill-sidecar fallback (task #25 watchdog will help) or chunked loading with a cancel-check per chunk.

### B-004 · ~~Tauri sidecar doesn't auto-respawn~~ — fixed 2026-05-21 late
**Status:** resolved (task #25).
**Resolution:** Tauri host (`src-tauri/src/lib.rs`) now polls `child.try_wait()` every 2 s. On crash it respawns the sidecar, waits for `/health`, and emits a `sidecar-restarted` event. React listens and shows a 15-s amber banner ("Sidecar restarted after a crash — generate again to retry") while re-bootstrapping model lists and GPU info.
**If you ever need the manual workaround:** click `Launch Kraken Art.bat` in the project root — it kills anything on :7780 and starts a fresh sidecar via the Tauri shell.

### B-005 · CFG auto-detect was inconsistent for FLUX
**Status:** fixed 2026-05-21.
**Symptom:** Picking a FLUX model didn't always set CFG to 1.0; sometimes it stayed at 3.5.
**Cause:** The `useEffect` watching `primaryModel` only fired when the model file actually changed. If the model was already selected when the user switched architectures, the effect didn't re-run.
**Fix in:** `src/Generate.tsx` now has a second `useEffect` watching `archId` directly, which forces `CFG=1.0` / `steps=28` whenever the architecture flips to flux1 / flux2 / z_image.

### B-006 · transformers 5.x breaks SDXL `from_single_file`
**Status:** pinned in `python/requirements.txt` (`transformers>=4.46,<5`).
**Symptom:** `AttributeError: 'CLIPTextModel' object has no attribute 'text_model'` when loading any SDXL checkpoint.
**Cause:** transformers 5 refactored `CLIPTextModel`'s internal attribute layout. diffusers 0.38's single-file loader still references the 4.x path.
**Path forward:** Unpin when diffusers ships a release that supports both layouts.

### B-007 · `.sft` extension not recognized by diffusers single-file loader
**Status:** worked around in `python/pipelines/flux.py`.
**Symptom:** Picking `FLUX1/fluxVaeSft_aeSft.sft` (a safetensors file with the `.sft` alias extension) fails diffusers' `AutoencoderKL.from_single_file` because it branches on file extension and falls back to `torch.load` for unknown extensions.
**Workaround:** Manual safetensors load + diffusers' `convert_ldm_vae_checkpoint`. Same path also handles `.safetensors` so we just use it universally for VAE loading now. Worth applying the same pattern to other component loaders before shipping more arches.

### B-008 · Background-process hygiene
**Status:** known caveat.
**Symptom:** Orphan ComfyUI server (PID held 19 GB RAM after browser closed); orphan polling scripts continue to hammer `/api/jobs/<dead-id>` after sidecar restart, spamming `404 job not found` warnings.
**Cause:** ComfyUI: out of our control, but worth documenting. Polling scripts: my own bash polls from earlier debugging sessions that I didn't always clean up.
**Mitigation:** Sidecar's global exception handler now warns loudly on every 404 with the path, making leaks visible. The Logs drawer rate-limits will help once added (see TODO).

---

## Resolved (since 2026-05-21)

- ✅ **B-001** T5 load native crash — root cause was peak RAM from `load_file()` + model skeleton. Fixed with streaming `safe_open` loader and per-tensor FP8 dequant. Verified: FP8 + FP16 T5, sidecar API.
- ✅ **B-009** FLUX Float/BFloat16 dtype mismatch — fixed `prepare_latents` patch + dtype coercion. Verified end-to-end inference.
- ✅ **B-010** Unicode log crash on Windows — ASCII log strings + UTF-8 stdout handler.
- ✅ Arch-switch crash (FLUX→SDXL) — strip accelerate hooks in `unload()`; cross-arch unload at start of each `run()`.
- ✅ Empty prompt = blank gen — now falls back to placeholder text.
- ✅ Log drawer needed resize — drag handle + localStorage persistence added.
- ✅ Sidecar 4xx/5xx not visible in log drawer — global FastAPI exception handler now routes everything through the ring logger.
- ✅ Frontend showed raw status code instead of error detail — `readError()` parses the response body.
- ✅ Stuck "Generating…" state — WS close handler resets to idle.
- ✅ Sampler-only dropdown — split into Sampler + Scheduler (ComfyUI-style).
- ✅ Architecture auto-detect from model filename — switches arch dropdown + fills VAE/TE/CFG/steps automatically.
- ✅ FLUX FP32 folder confusion — folder name was misleading; file inside is bf16. Documented in PIPELINES.md.
- ✅ FLUX VAE noise output — caused by FlowMatchEulerDiscreteScheduler defaults missing `use_dynamic_shifting=True`. Now loads from `model_configs/flux1_dev/scheduler/scheduler_config.json`.
- ✅ FLUX FP16 OOM at load — caused by T5EncoderModel's default fp32 skeleton (44 GB CPU). Now uses `accelerate.init_empty_weights` + `set_module_tensor_to_device`.
- ✅ Meta tensors leftover after load (e.g. `position_ids`) — explicitly re-initialised after the state_dict assignment.
