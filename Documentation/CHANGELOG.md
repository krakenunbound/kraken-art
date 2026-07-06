# Changelog

Session-by-session record of what's landed.

## 2026-07-05 — WAN 2.2 T2V/I2V, LoRAs, and last-frame extension

WAN video generation moved from an i2v-only path to a fuller local video
creation workflow.

What landed:
- `src/Video.tsx`
  - T2V/I2V mode switch.
  - T2V no longer requires an input image.
  - Explicit high-noise and low-noise expert/LoRA controls.
  - `Use last frame` action on generated videos for I2V continuation.
- `python/pipelines/wan_video.py`
  - Uses `WanPipeline` for T2V and `WanImageToVideoPipeline` for I2V.
  - Applies the T2V transformer config override (`in_channels=16`) and the T2V
    boundary ratio (`0.875`).
  - Keeps I2V image kwargs mode-specific.
- `python/api/generate.py`
  - Adds `mode` validation and requires input images only for I2V.
- `python/api/outputs.py`
  - Adds `/api/outputs/capture-last-frame`, saving the final decodable frame to
    `outputs/<date>/frames/`.
  - Includes an exact-frame fallback for very short clips where `ffmpeg -sseof`
    returns success without writing a PNG.
- `src/api/sidecar.ts`
  - Adds `mode` to video generation params and the last-frame capture helper.
- `Documentation/WAN_VIDEO.md`
  - New operator/developer guide for WAN 2.2 model files, LoRAs, extension
    workflow, and verification baseline.

Model setup:
- Downloaded the Comfy-Org WAN 2.2 T2V FP8 14B high/low experts into
  `models/diffusion_models/WAN22/`.
- Copied the user's WAN 2.2 InterFrame LoRAs into `models/loras/WAN22/`.

Verified:
- `npm run build`
- `python\venv\Scripts\python.exe -m py_compile python\api\generate.py python\api\outputs.py python\pipelines\wan_video.py`
- Real T2V smoke: `outputs/2026-07-05/154131-t2vsmoke.mp4`
- Captured final frame: `outputs/2026-07-05/frames/154558-lastframe-154131-t2vsmoke.png`
- Real I2V continuation smoke: `outputs/2026-07-05/154631-i2vexten.mp4`

## 2026-07-05 — Video Up tab: ESRGAN/SeedVR2 upscale, RIFE 60fps, media bins, audio preservation

Video upscaling is now a first-class workflow in Kraken Art. The new **Video
Up** tab is designed around imported source clips, visible result comparison,
and practical 2K/4K testing on the local RTX 3090.

What landed:
- `python/pipelines/video_upscale.py`
  - ESRGAN/Spandrel frame-upscale path for preservation-first video upscale.
  - SeedVR2 CLI path for diffusion/detail reconstruction.
  - RIFE ncnn Vulkan interpolation path for 60fps output.
  - SeedVR2 **AI detail strength** implemented as a blend between an ESRGAN
    preservation upscale and the SeedVR2 detail result. This gives the video
    workflow a practical strength/denoise-like control even though SeedVR2 does
    not expose an Ultimate-SD-Upscale-style denoise setting.
  - Final-source-audio muxing: intermediate frame/upscale stages are silent, and
    the final MP4 can reattach the original source audio stream when **Keep
    source audio** is enabled.
- `python/api/video_upscale.py`
  - `/api/video-upscale` job endpoint with validation for engine, scale mode,
    fps interpolation mode, SeedVR2 VAE tiling, compile mode, noise scales, and
    audio preservation.
- `python/api/outputs.py`
  - Video import endpoints for file uploads and local path imports.
  - Latest-video items now include absolute local paths for opening results in
    external players.
- `src/VideoUpscale.tsx`
  - Top **Recent / imported videos** lane for source/before clips.
  - Bottom **Upscale results** lane for finished `*-vup.mp4` outputs.
  - Checkbox selection, check-all, delete-checked, delete-all, and per-thumbnail
    delete in both lanes, without browser confirmation prompts.
  - Whole-window drag/drop via Tauri native file-drop events plus browser drop
    fallback.
  - Side-by-side **Compare preview**: source on the left, result on the right.
  - Open result in the system player or VLC when the local path is available.
  - Sliders plus manual number inputs for fps, batch/overlap, SeedVR2 detail
    strength, input/latent noise, VAE tiling, block swap, ESRGAN tile settings,
    and related controls.
- `src/App.tsx`
  - Global **Open outputs** button in the app chrome, opening the real
    `outputs/` folder.
- `src-tauri/tauri.conf.json`
  - Explicit native file-drop support via `dragDropEnabled: true`.
- `Documentation/VIDEO_UPSCALE.md`
  - New operator guide for the feature, recommended presets, audio behavior,
    output layout, and current test baseline.

Test baseline:
- Source test clip: Grok Imagine MP4, `1168×768`, ~24fps.
- ESRGAN/RealESRGAN_x2 1080p60 output was visually preferred by the user over
  the first SeedVR2 1080p60 test.
- 2K ESRGAN/RealESRGAN_x2 runs around `1 fps` during frame upscale on the local
  RTX 3090, before RIFE/encode overhead.

## 2026-05-28 — Video tab: WAN 2.2 i2v A14B pipeline + dedicated UI

Video generation is now a first-class surface, separate from still images. A
new **▶ Video** tab drives a full WAN 2.2 image-to-video pipeline (the A14B
dual-expert model), end-to-end from a still frame to an H.264 mp4.

What landed:
- `python/pipelines/wan_video.py` — full WAN 2.2 i2v pipeline assembled from
  component files (no diffusers `from_pretrained` download):
  - **Dual-expert** transformer: high-noise expert for t ≥ boundary, low-noise
    for t < boundary (`boundary_ratio=0.9`), built into a
    `WanImageToVideoPipeline`. VRAM: both 14B experts can't co-reside on 24 GB,
    so we use `enable_model_cpu_offload` (one-time boundary swap of the active
    expert) rather than FLUX-style per-Linear streaming.
  - ComfyUI **scaled-fp8** handled on both experts and the UMT5 encoder
    (`weight = fp8.to(bf16) * scale_weight`); scale keys remapped through the
    same diffusers rename pass as the weights.
  - UMT5-XXL text encoder streamed in, used to encode prompt (+ negative when
    guidance > 1), then freed before sampling.
  - Frames muxed to mp4 via the system **ffmpeg** (rawvideo pipe → libx264
    yuv420p, crf 18, +faststart) — zero new Python deps.
  - Per-step progress events + a final `video` event (mp4 rel_path, seed,
    first-frame thumbnail).
- `python/api/generate.py` — `POST /generate/video` (`VideoGenerateRequest`)
  validates both experts + WAN VAE + UMT5 encoder + input image, then submits
  `wan_video.run` to the job manager. `outputs.py` already serves/lists mp4.
- `src/Video.tsx` — new panel: input-frame upload with preview, motion prompt +
  negative, high/low expert + WAN VAE + UMT5 dropdowns (auto-picked from the
  scan), lightx2v 4-step LoRA toggle, 480p/720p landscape+portrait presets,
  frame-count/fps/steps/cfg/seed controls, live progress bar, and an HTML5
  `<video>` gallery of results.
- `src/api/sidecar.ts` — `startGenerateVideo` + `VideoGenerateParams`.
- `src/App.tsx` — `▶ Video` tab wired alongside Image / Library / Music.

Verified end-to-end on the 3090: i2v from a 1024² still → `832×480`, 25 frames,
4-step lightx2v, ~70 s wall-clock incl. dual-expert load. Output is a valid
H.264 mp4 (1.56 s clip) served as `video/mp4` over `/api/outputs/file/...`.

Two scaled-FP8 bugs fixed during verification (both surfaced as
`"mul_cuda" not implemented for 'Float8_e4m3fn'`):
- **LoRA adapters landed in FP8.** PEFT creates injected `lora_A/lora_B` tensors
  in the base Linear's dtype; on our FP8 experts that meant the
  `lora_B(lora_A(x)) * scaling` multiply hit FP8. Fix: `_coerce_lora_dtype`
  casts every injected LoRA tensor to bf16 after `set_adapters`.
- **Timestep embedder forced to FP8.** `WanTimeTextImageEmbedding.forward` casts
  the sinusoidal timestep to `next(time_embedder.parameters()).dtype`; with the
  embedder kept in FP8, the timestep itself became FP8 and the matmul died.
  Fix: `_dequantize_fp8_submodule` folds the scale and keeps
  `condition_embedder` / `patch_embedding` in bf16 — matching diffusers' own
  `WanTransformer3DModel._skip_layerwise_casting_patterns`.

## 2026-05-28 — Generation metadata embedded in PNG (reuse settings = standard workflow)

"Reuse generation settings" no longer relies on a separate `.settings.json`
sidecar. Generation metadata is now written *into* the output PNG itself, the
same way ComfyUI / A1111 do — so Civitai (and any other tool) reads prompt,
model, steps, seed, cfg, sampler, scheduler and LoRAs straight out of an
uploaded image, and our own "reuse settings" reads from the very same place.

What landed:
- `python/pipelines/output_metadata.py` rewritten. Two tEXt chunks are embedded
  in every generated PNG:
  - `parameters` — A1111-format text block (prompt, `<lora:name:weight>` inline
    tags, `Negative prompt:`, then `Steps/Sampler/Schedule type/CFG scale/Seed/
    Size/Model/Clip skip/Arch`). This is the format Civitai's "generation data"
    parser recognises.
  - `kraken_settings` — the full settings JSON (arch, diffusion_model/checkpoint,
    vae, text_encoders, loras, embeddings, upscale params, etc.), the lossless
    source for restoring *every* field exactly.
  - New helpers: `build_settings_dict`, `format_a1111_parameters`,
    `build_pnginfo`, `save_png_with_metadata`, `read_settings_from_png`.
- All three pipelines (`flux.py`, `sdxl.py`, `z_image.py`) now save via
  `save_png_with_metadata(...)` instead of `img.save(...)` + a sidecar write.
  Upscaled outputs (`-up.png`) carry the embedded metadata too (previously they
  had no reusable settings at all).
- `/api/outputs/settings/{path}` reads the embedded `kraken_settings` from the
  PNG first, falling back to a legacy `.settings.json` sidecar for images made
  before this change. Same response shape, so the frontend "Reuse generation
  settings" action is unchanged.

## 2026-05-28 — CivitAI download arch-routing

Downloads via the user's CivitAI API key now auto-sort into architecture
subfolders instead of dumping everything into the four legacy buckets.

What landed (`python/api/civitai.py`):
- Replaced the old `_checkpoint_subfolder` (Checkpoint-only,
  Illustrious/Pony/SDXL/Flux1) with arch-aware routing covering 14
  architectures via `_ARCH_SUBFOLDER`, applied to **checkpoints, LoRAs, and
  embeddings** (not just checkpoints).
- `_detect_arch_from_version` infers arch from a lowercased haystack of
  `baseModel` + version name + model name, ordered so specific tokens win
  (illustrious/pony before sdxl, flux2 before flux1, etc.). This mirrors the
  scanner's path-substring `detected_arch` so a downloaded file lands where the
  Generate-tab picker expects it.
- `_route_by_arch` redirects component-architecture checkpoints
  (flux1/flux2/z_image/qwen_image/chroma/hidream) from `checkpoints/` into
  `diffusion_models/<Arch>/`, because those pipelines read the transformer from
  `diffusion_models`. All-in-one checkpoints still work there — the component
  pipelines tolerate AIO files (flux.py drops `text_encoders.`/`vae.` prefixes;
  the z_image converter strips `model.diffusion_model.`), so both the split and
  all-in-one workflows are supported without crippling either.
- `detected_arch` is now recorded in the download metadata sidecar and returned
  in the `/api/civitai/download` response.

## 2026-05-28 — Z-Image pipeline (first Phase-3 architecture)

First new image architecture wired end-to-end since FLUX1, continuing the
"hook up the various models" push. Z-Image is Alibaba Tongyi-Lab's text-to-image
model; it assembles from three local single-file safetensors (ComfyUI-style)
rather than an all-in-one checkpoint.

What landed:
- New `python/pipelines/z_image.py`. Mirrors `flux.py`'s component-assembly
  approach: meta-device load + LDM/native→diffusers conversion + bf16
  materialize. The transformer (`ZImageTransformer2DModel`) loads in native
  Z-Image key layout and is converted via diffusers'
  `convert_z_image_transformer_checkpoint_to_diffusers` (exact 521/521 key
  match against the meta-instantiated model). The VAE is the Flux.1-AE
  (16-channel, scaling 0.3611 / shift 0.1159, no quant convs), loaded with the
  shared `convert_ldm_vae_checkpoint` path. Architecture configs + tokenizer
  ship under `python/model_configs/z_image_turbo/` so no gated HF repo is ever
  touched — only the heavy weights come from the user's local model folders.
- Text encoding is the Qwen3-4B base model (`Qwen3Model`). The on-disk file
  carries a `model.` prefix (the ForCausalLM layout); we strip it and load the
  base encoder. Conditioning is `hidden_states[-2]` after
  `apply_chat_template(enable_thinking=True)`, masked per-prompt to real token
  length — a LIST of variable-length tensors as the pipeline expects. The
  encoder is loaded, used, and freed BEFORE the transformer loads, so peak
  VRAM is transformer (~11.5 GB) + VAE (~0.3 GB) during sampling instead of
  also holding the 7.5 GB encoder.
- Turbo variant runs `guidance_scale=0.0` (no CFG, ~9 steps); the base variant
  honours the UI CFG and additionally encodes a negative prompt. VRAM placement
  is full-GPU when it fits (the 24 GB common case) with a `model_cpu_offload`
  fallback for smaller cards.
- Backend: `/api/generate` now routes `arch == "z_image"` to `z_image.run`
  with diffusion_model + VAE + Qwen3 text-encoder validation.
- Frontend: added a supported `z_image` entry to `ARCH_PROFILES` (components
  mode, 1 encoder, upscale allowed) and a `modelMatchesArch` case so the
  architecture→model→LoRA filtering chain works. Scanner already tagged
  `detected_arch: "z_image"`.
- Verified end-to-end on the RTX 3090: ZImageTurbo_turbo @ 1024², 9 steps,
  seed 12345 produced a clean, coherent image (not noise) in ~29 s cold
  (7.4 s encode + 8 s transformer/VAE load + ~13 s sampling).

## 2026-05-28 — Launcher hygiene, sticky Generate settings, GPU telemetry

Focused cleanup after the FLUX warm-start/audio integration work exposed two
workflow regressions: launching Kraken Art was starting ACE-Step by default,
and Generate-tab selections were not reliably surviving app restarts.

What landed:
- Fixed LoRA loading (B-016). diffusers >=0.30 requires the PEFT backend for
  `load_lora_weights`/`set_adapters`, but `peft` was never installed and was
  absent from `requirements.txt`, so every LoRA failed with "PEFT backend is
  required for this method." Installed `peft` 0.17.1 and pinned
  `peft>=0.13,<0.18`. Requires a sidecar restart to take effect.
- `Launch Kraken Art.bat` no longer starts the Kraken_Audio stack by default.
  Normal launch now starts only the Kraken Art sidecar on `7780` plus the Tauri
  UI. Integrated audio startup is opt-in via `KRAKEN_LAUNCH_AUDIO=1`, which
  starts Song Studio and ACE-Step for explicit audio testing.
- Music tab startup no longer probes Song Studio or ACE-Step automatically.
  Song Studio catalog loading is behind the explicit "Load Song Studio catalog"
  button, and ACE health remains behind the explicit "Check ACE" action. This
  prevents UI navigation from waking ACE or consuming VRAM.
- Generate tab last-used persistence was repaired and expanded. The UI now
  saves/restores architecture, model selections, VAE, text encoders, LoRAs,
  embeddings, prompt fields, dimensions, sampler/scheduler, batch seed/count,
  and upscale settings through `config/settings.json:lastGenerate`.
- Settings PATCH now accepts `lastGenerate`. A `config_store.save()` deadlock
  was fixed by switching the settings lock to `RLock`, so frontend settings
  saves cannot hang while calling `load()` under the same lock.
- GPU panel now shows live GPU utilization percent and GPU temperature in
  Celsius, using the existing NVML path in `/api/gpu`.
- Generate-tab model selectors are now filtered by selected architecture.
  Checkpoint architectures like Illustrious no longer get overwritten by an
  unrelated FLUX selection, and Civitai metadata is used when filenames alone
  are ambiguous.
- Civitai checkpoint downloads now route future checkpoint files into
  architecture subfolders such as `checkpoints/Illustrious` when the model
  version reports a matching `baseModel`. Download metadata now also stores the
  model description so recommended settings can be parsed in a follow-up pass.
- Civitai Library filters now persist between Image/Library tab switches and
  app restarts. Persisted fields include search text, type, base model, sort,
  time period, NSFW toggle, page, and cursor state.
- Added Civitai `period` filtering (`AllTime`, `Year`, `Month`, `Week`, `Day`)
  so searches like "Highest Rated" can explicitly mean all-time rather than a
  recent window.
- Fixed Civitai pagination for current cursor-based API behavior. The backend
  no longer forwards `page` to Civitai when `query` is present, the frontend
  tracks `nextCursor`, and the Library displays open-ended pagination as
  `page N / ...` when Civitai withholds totals.
- Fixed sparse filtered Civitai result pages. The backend now walks several
  cursor pages and aggregates matches before returning a Library page, which
  prevents false "no results" states for searches such as
  `Peach Blossom` + `Checkpoint` + `Illustrious`.
- Expanded SDXL/Illustrious sampler and scheduler choices beyond the original
  small list. The backend maps the added UI names to Diffusers schedulers
  including Euler, Euler A, Heun, LMS, DDIM, UniPC, DEIS, PNDM, LCM, DPM2,
  DPM2 A, DPM++ 2M, DPM++ SDE, and DPM++ 2S A variants.
- LoRA selection is now architecture-filtered. The dropdown uses scanned
  Civitai metadata and filename/base-model heuristics to show only LoRAs that
  match the selected architecture, and removes incompatible selected LoRAs when
  the architecture changes.
- LoRA selection UI now behaves as row slots instead of a select-plus-button
  control. Choosing a LoRA immediately adds it to the active list and creates a
  new empty dropdown below it for adding the next LoRA. The header count now
  reflects the active rows immediately.
- LoRA entries now expose per-LoRA model strength in the UI and request payload.
  Existing saved `weight` values are still accepted; new entries persist both
  `model_weight` and the legacy `weight` field for compatibility.
- Added SDXL/Illustrious `clip_skip` support. The Generate panel exposes a
  `CLIP skip` numeric control near LoRAs (`0` means Diffusers default), and
  the SDXL pipeline forwards it into `StableDiffusionXLPipeline.__call__`.
  FLUX ignores this field.
- Generate and Library tabs are kept mounted and hidden instead of destroyed
  on tab switches. This preserves in-progress image generation, WebSocket
  state, gallery thumbnails, and Library filter state while browsing.
- Generated image outputs now get a sibling `.settings.json` file containing
  the exact reusable generation settings and per-image seed. Right-clicking a
  gallery thumbnail opens a `Reuse generation settings` action that restores
  prompt, model selections, LoRAs, dimensions, sampler/scheduler, seed, and
  upscale settings. Existing images generated before this change do not have
  this sidecar metadata.
- Removed two frontend build blockers in `App.tsx` (`hp` dead state and the
  unused `setLastSeenLogId` setter) while preserving log polling behavior.

Files touched for this cleanup:
- `Launch Kraken Art.bat`
- `python/api/civitai.py`
- `python/api/gpu.py`
- `python/api/models.py`
- `python/api/outputs.py`
- `python/api/settings.py`
- `python/config_store.py`
- `python/pipelines/flux.py`
- `python/pipelines/output_metadata.py`
- `python/pipelines/sdxl.py`
- `src/App.tsx`
- `src/App.css`
- `src/Generate.tsx`
- `src/Library.tsx`
- `src/Music.tsx`
- `src/api/sidecar.ts`

Verification:
- `npm run build` passes.
- `python\venv\Scripts\python.exe -m py_compile` passes for the changed
  backend modules touched during this session.
- `/api/gpu` local check returned live utilization and temperature fields.
- Settings PATCH schema accepts `lastGenerate`.
- Settings save path was tested against a temporary settings file.
- Direct Civitai checks confirmed the new cursor behavior and reproduced the
  sparse-page case before the backend aggregation fix.

---

## 2026-05-27 — Ecosystem Unification Kickoff (Platform Foundations — PR 1 start)

**Branch:** `feature/ecosystem-unification` (branched from `feature/audio-integration`).
**Pre-work tag:** `backup/2026-05-27-2123-pre-ecosystem-pr1`
**Primary undo snapshot:** `backups/2026-05-27-2123-start-ecosystem-unification-pr1/` (full selective source backup)
**Reference:** `Documentation/design-runs/kraken-unified-ecosystem-architecture-6f9f838e.md` (full approved design after writer/reviewer loop with 0 open issues)
**Master log:** `Documentation/IMPLEMENTATION-ECOSYSTEM.md` (append-only record of every backup, decision, todo/havedone)

**Context & Motivation:**
After the complete 2026-05-27 source + documentation audit, the project was assessed as "great progress but chaos barely contained" rather than a smooth ecosystem. The approved design defines the path to a unified Capability/Modality Registry + thin adapters so that images, music, future modalities, manual GUI use, **and** full AI/MCP agent control are first-class peers on the same reliable surface.

**What was done in this session (foundation only — zero behavior change):**
- Two independent backups created **before any source modification** (filesystem snapshot + new git branch + annotated tag).
- Running implementation log + todo/havedone discipline established.
- Strict rule enforced: no functional files touched until backups verified.

This entry marks the official start of the multi-PR unification effort. All subsequent work is documented in `IMPLEMENTATION-ECOSYSTEM.md`.

---

## 2026-05-23 (night) — Phase D: MP3 export (LAME VBR V0 + embedded cover)

**Branch:** `feature/audio-integration`.
**Pre-work tag:** `checkpoint/2026-05-23-2020-pre-phase-d-mp3-export`.
**Post-work tag:** see git tag for hash.

WAV→MP3 export for any song in the Music tab library, single or bulk. User
preference (learned on a sibling project): **LAME `-V 0` VBR (~245 kbps avg)
instead of CBR 320** — sidesteps two CBR pitfalls (wasted bits on silent
frames + downstream mastering tools tripping on the 320 sentinel) while
remaining sonically indistinguishable for ACE-Step-generated music.

What landed:
- `python/pipelines/audio/mp3_export.py` (NEW, ~280 lines) — ffmpeg
  subprocess to `libmp3lame -q:a 0` for encode, mutagen for ID3v2.4 +
  APIC cover embed. Tags: TIT2 (title), TPE1 (artist=workspaceTitle),
  TALB (album), TCON (genre), COMM:prompt, TBPM, TKEY, USLT (full
  lyrics), APIC (cover, type 3 = front).
- `python/api/audio.py` — three new endpoints:
  - `POST /api/audio/songs/{id}/export-mp3` (sync single)
  - `GET /api/audio/songs/{id}/export-mp3/download` (browser download)
  - `POST /api/audio/songs/export-mp3` (async bulk via JobManager + WS)
- `python/requirements.txt` — `mutagen>=1.47` added.
- `src/api/sidecar.ts` — `exportSongMp3`, `exportSongsBulk`, types.
- `src/Music.tsx` — per-card `⬇` button + multi-select `Export MP3`
  toolbar action + live progress banner (subscribes to /ws/jobs/{id}).
- `src/App.css` — `.bulk-export-banner` + button hover states.

Output layout: `outputs/exports/<workspace>/<title>.mp3`.

Smoke-tested live: Salt And Dust (143.8 s WAV) → 6.9 MB MP3 @ 259 kbps
avg, all ID3 frames written, cover embedded. file(1) confirms ID3v2.4 +
MPEG layer III + variable bitrate. All three endpoints return 200 via
TestClient.

System dependency: **ffmpeg with libmp3lame** (already on the user's PATH
from Gyan's Windows build, version 8.0.1). README will document this.

Out of scope:
- FLUX-speed experiments still uncommitted on disk.
- Z-Image arch (task #57) — independent, not a Phase D blocker.

## 2026-05-23 (late evening, refine) — Phase C: lastGenerate as default + no more 501s

**Commit:** (see git log for hash). **Tag:** `checkpoint/2026-05-23-XXXX-phase-c-refine` (to be applied this commit).

User feedback after the Phase C ship: "use the current default Kraken Art
model" — i.e. the endpoint should never return 501 for an unknown arch, it
should fall through to whatever's actually working today and tell the
caller what ran. Surgical refinement:

- `python/api/cover_art.py`:
  - `SUPPORTED_ARCHS = {"flux1", "sdxl"}` — the explicit set of archs with
    a working pipeline today; new archs land here as they ship.
  - New `_user_default_from_settings()` reads `config_store.lastGenerate`
    so the cover endpoint picks exactly the model/VAE/TE combo the user
    last used from the Generate tab.
  - Param-build priority: explicit request field > user's `lastGenerate` >
    auto-discovered default.
  - `CoverArtResponse` now includes `requested_arch` + `arch` +
    `requested_arch_unavailable` so the caller can show "asked for X, got
    Y." Asking for `z_image` today returns a FLUX1 cover with
    `requested_arch_unavailable=True` instead of HTTP 501.
  - `/api/cover-art/defaults` now also returns `user_default` +
    `supported_archs` + `fallback_policy` + `arch_coverage_gap_tracked_in`.

Task #57 reframed: was "[Audio C0] Z-Image Turbo pipeline" (incorrectly
framed as a Phase C dependency); now "[Image arch] Z-Image Turbo support
(any model out of the box)" — sibling to task #11. Phase C never depended
on it.

## 2026-05-23 (late evening) — Phase C: cover-art redirect (ComfyUI → Kraken Art)

**Branch:** `feature/audio-integration`.
**Pre-work tag:** `checkpoint/2026-05-23-1940-pre-phase-c-cover-art` (pushed).
**Post-work tag:** see commit below.

Replaces Song Studio's ComfyUI cover-art call path with a direct round-trip
through Kraken Art's internal FLUX1 pipeline. Strategy 1 of the two C
options the user weighed earlier today: patch Song Studio with a small env-
gated branch, no protocol shim.

User picked Civitai's **GonzaLomo ZPop v4.0** (Z-Image Turbo fine-tune) as
the target cover model. Downloaded BF16 11.7 GB variant via Civitai API,
SHA-256 verified, written to `models/diffusion_models/ZImageTurbo/`
alongside the existing base. **Z-Image pipeline wiring is deferred** (task
#57) because its Qwen3-4B text encoder needs its own loader; FLUX1 is the
working default until then. Flipping to Z-Image once C0 lands is one env
var (`KRAKEN_COVER_ARCH=z_image`).

Civitai token discovery: the user had believed the token was persisted
yesterday but `config/settings.json` actually had no `civitai` block.
Re-entered + saved + verified against the live API (HTTP 307 redirect to
B2 signed storage URL). Settings.json is gitignored (verified at
`.gitignore:39`); token never leaves disk.

What landed in this commit:
- **New** `python/api/cover_art.py` — synchronous `POST /api/cover-art`
  that takes `{prompt, aspect_ratio, arch_hint?, song_dir?, song_id?,
  song_title?}` and returns the saved image + a `cover.png` + `cover.json`
  side-effect drop into the Song Studio song folder. Drives the existing
  `pipelines.flux.run()` via a synthetic in-process `_InlineJob`. Defaults
  auto-discovered from `MODELS_ROOT`: FLUX1 picks `fluxmania_kreamania.safetensors`
  + `ae` VAE + `clip_l` + `t5xxl_fp16`. Also exposes `GET /api/cover-art/defaults`
  for the future settings panel.
- **Edited** `python/main.py` — registered the new router.
- **Edited** `F:/Kraken_Audio/ACE-Step-1.5/acestep/codex_song_studio.py`
  (4887 → 5040 lines). Added `_kraken_cover_url()`, `_kraken_cover_arch_hint()`,
  and `run_cover_via_kraken_art()`. Patched `run_cover_art_job_worker` so
  that when `KRAKEN_COVER_URL` is set, the worker round-trips through
  Kraken Art before falling back to ComfyUI. `COVER_ART_JOBS` updates +
  `cover.json` manifest shape match the existing ComfyUI path verbatim so
  the React UI doesn't know which backend ran. Provenance fields
  (`kraken_art_arch`, `kraken_art_model`, `workflow_name="kraken-art:flux1"`)
  let debugging tell at a glance.
- **New** `models/diffusion_models/ZImageTurbo/gonzalomoZpop_v40.safetensors`
  (12.3 GB, gitignored; metadata sidecar committed).
- **Pre-work backups**: `backups/2026-05-23-1940-pre-phase-c-cover-art/`
  including the original `codex_song_studio.py.original`.

Smoke test status: module imports clean, default discovery returns the
correct FLUX1 stack, routes register. **Live end-to-end gen not yet
benched** — sidecar needs a restart to pick up the new token + module
(currently-running instance has `civitai.api_token=''` cached in
`config_store._cache`). User can flip the switch by setting
`KRAKEN_COVER_URL=http://127.0.0.1:7780` in the Song Studio launcher.

Out of scope:
- Z-Image pipeline (task #57) — its Qwen3-4B TE wiring is the next thing
  to land for Phase C to be "fully on the user's chosen model."
- Phase D (MP3 export + cover embed) — task #56.
- FLUX-speed experiments still uncommitted on disk.

## 2026-05-23 (evening) — Phase B complete: Suno-style Music tab UI

**Branch:** `feature/audio-integration`.
**Tags:** `checkpoint/2026-05-23-1925-phase-b1-library-grid`,
`checkpoint/2026-05-23-1950-phase-b2-sidebar`,
`checkpoint/2026-05-23-1910-phase-b3-player-bar` (post-work, this commit).

Three back-to-back commits (B1, B2, B3) rebuild the Music tab into the
Suno-style workstation the user demonstrated with the Audio Studio
screenshots — using only the Song Studio proxy endpoints landed in Phase A.
Each increment shipped with its own pre-work filesystem backup +
pre-work checkpoint tag + commit + post-work tag, per the user's
"document and back up as you go" directive.

- **B1** (`2678a5c`) — Suno-style library cover grid with search, multi-select,
  bulk delete, per-card delete, inline player.
- **B2** (`a983ec8`) — Left sidebar with workspaces (derived from library) +
  playlists, All-songs filter, inline playlist creation, workspace-create
  hint, refresh; "Add to playlist ▾" dropdown in the toolbar when songs
  are selected.
- **B3** (this commit) — Persistent bottom player bar (`position: fixed`)
  with cover thumbnail, title, prev / play-pause / next (walks the
  currently-filtered grid, wraps), draggable scrubber, volume slider,
  autoplay-next on `<audio>` `ended`. Inline player removed from center
  pane.

Files touched in B3: `src/Music.tsx` (player state + effects + JSX moved to
top-level sibling), `src/App.css` (+~210 lines of player-bar styling),
`Documentation/Cursor-audio-adaption.md` (sections 13.5 result + 13.6 Phase
B summary).

TypeScript: clean for new code (two pre-existing TS6133 unused-var warnings
in `App.tsx` left alone per branch policy).

Out of scope (still uncommitted on disk, will land separately):
FLUX-speed experiments and bench scripts — same list as the Phase A entry
below.

Next: Phase C (cover-art redirect from ComfyUI to Kraken Art FLUX/SDXL,
task #55), then Phase D (MP3 320 kbps export verify + mutagen fallback,
task #56).

## 2026-05-23 (afternoon) — Audio integration: Song Studio (port 8010) proxy

**Commit:** `17b9fc1` on `feature/audio-integration`, pushed to GitHub.
**Checkpoint tag:** `checkpoint/2026-05-23-phase-a-song-studio-proxy`.

Cursor's earlier 2026-05-23 work (sections 7-11 of `Cursor-audio-adaption.md`)
bridged the bare ACE-Step API on port 8001 — raw "submit a job, get a .wav."
This commit completes that by also wiring Kraken_Audio's **Codex Song Studio**
on port 8010 (the actual user-facing workstation the user demonstrated with the
Audio Studio screenshots).

The user-facing payoff: **all 364 of the existing library songs** now show up
through Kraken Art's port 7780 — the Music tab UI never has to know that
8001 (raw ACE) and 8010 (workstation) even exist.

The Phase A endpoints landing in this commit:

| Method | Route | Forwards to |
|---|---|---|
| GET | `/api/audio/song-studio/health` | 8010 `/api/config` (catalog + service health) |
| GET | `/api/audio/library` | 8010 `/api/library` (all songs incl. metadata) |
| GET | `/api/audio/playlists` | 8010 `/api/playlists` |
| POST | `/api/audio/playlists` | 8010 `/api/playlists` |
| POST | `/api/audio/playlists/{id}/songs` | 8010 same |
| POST/PATCH | `/api/audio/workspaces[/{id}]` | 8010 same |
| DELETE | `/api/audio/songs/{id}` | 8010 `/api/library/songs/{id}` |
| POST | `/api/audio/songs/bulk-delete` | 8010 same |
| GET | `/api/audio/stream?path=...` | 8010 `/api/audio?path=...` (audio playback) |
| GET | `/api/audio/songs/{id}/download` | 8010 same (MP3 + Content-Disposition) |

Plus the Cursor-built Music tab + bare ACE bridge (sections 7-11) — those
moved from disk into git here as part of one cohesive "audio integration"
commit. Phase B (Suno-style library UI) and Phase C (cover-art redirect to
internal FLUX/SDXL) are the next two commits — see tasks #54 and #55.

**Out of scope, intentionally left uncommitted on this branch:**
- FLUX-speed experiments (`streaming_linear.py`, `kraken_flux_attn.py`,
  `kraken_rope.py`, `kraken_fbcache.py`, all `*.bak` files, bench scripts,
  `Documentation/FLUX-PERFORMANCE-EXPERIMENTS.md`, `gemini.md`,
  `Documentation/FLUX-SPEED-{WIP,RESEARCH}.md` updates).
- These will land in a separate FLUX-speed commit after the audio work
  reaches Phase D. Listed in section 12 of `Cursor-audio-adaption.md`.

## 2026-05-21 (evening) — FLUX speed pass: StreamingLinear offload

The goal was ComfyUI-parity throughput for FLUX-dev on a 24 GB GPU. The diffusers
default (`enable_model_cpu_offload`) gave ~138 s per 1024² × 28-step gen and crashed
on follow-up gens (RAM blowup). After reading Forge's, Fooocus's, and A1111's
memory-management code, we ported the **Forge-style per-Linear streaming** approach.

### Results (3090, 1024² × 28 steps, FP32 flux_dev)

| | Before | After |
|---|---|---|
| Cold gen | ~138 s | ~75 s |
| Warm gen | crashed | ~73 s |
| Stable across gens | no | yes |
| Stable across arch switches | partial | yes |

(For reference: ComfyUI on the same machine = 107 s cold / 48 s warm. We're under
their cold but still ~25 s behind warm — gap is true layer-prefetch, planned next.)

### What landed

- **`python/pipelines/streaming_linear.py`** (new) — `StreamingLinear(nn.Linear)`
  subclass with a unified forward that handles four cases in one path: GPU-resident
  plain weight, GPU-resident FP8 (Ampere cast-per-call), GPU-resident scaled-FP8
  (cast + multiply by `_kraken_weight_scale`), and CPU-resident streaming via
  pinned host memory on a dedicated mover CUDA stream.
- `swap_linears(transformer)` walks the module tree and in-place replaces every
  `nn.Linear` with a `StreamingLinear`. Preserves parameter identities so a
  pre-loaded state_dict and any `register_buffer` scales carry over verbatim.
- `apply_streaming(transformer, budget_bytes, device, pin_memory=True)` runs a
  three-tier partition:
  - **Always GPU**: VAE + small transformer params (embedders, norms, proj_out, etc.).
  - **GPU as budget allows**: full `transformer_blocks.N` / `single_transformer_blocks.N`
    until the VRAM budget is exhausted.
  - **CPU-streamed**: remaining blocks. Linear weights cloned into private heap
    memory (detaches from the source mmap), then `.pin_memory()`'d. Norms / biases
    inside streamed blocks moved to GPU to avoid per-call device-mismatch errors
    in RMSNorm and to skip a redundant H2D per call.
- **`python/pipelines/flux.py`** — `_ensure_pipeline` now calls `swap_linears`
  after `_apply_flux_weight_scales`. The fast-vs-offload branch keeps the existing
  fully-resident path; the fallback (used to be `enable_model_cpu_offload`) is
  replaced by `apply_streaming` with `pin_memory=True`.
- **`python/pipelines/load_utils.py`** — `unload_pipeline` no longer calls
  `pipe.to("cpu")` before tearing down. That move was costing a 20+ GB transient
  RAM peak on Windows during arch switches; GC handles the GPU release just as
  well without the peak. `stream_safetensors_into_meta` also gained an explicit
  `dtype=dtype` arg on `set_module_tensor_to_device` so accelerate doesn't
  silently promote bf16 values to fp32 (this was the root cause of the second-gen
  OOM that the user hit earlier in the evening).

### Bugs caught and fixed along the way

- **Device mismatch in streamed RMSNorm**: the first integration only moved
  Linears for streaming, leaving RMSNorm weights on CPU while activations were on
  GPU. Fixed by moving every non-Linear param/buffer in a streamed block to GPU.
- **Windows OSError 1455 "paging file too small"**: the source safetensors mmap
  stayed alive because streamed weights referenced it, blocking later mmaps
  (T5 cache, model switch). Fixed by `.clone()`-ing each streamed weight into
  private heap memory so the source file can close.
- **Arch-switch OOM**: `pipe.to("cpu")` during unload was making a transient 20+ GB
  copy of GPU state on CPU, blowing past the Windows commit limit when combined
  with the in-flight new-pipeline allocations. Fixed by dropping that move.
- **404 log spam demoted to INFO**: a stale background polling task was hitting
  `/api/jobs/<dead-id>` and flooding the log drawer with WARNINGs. Demoted 404
  responses in the global FastAPI exception handler to INFO — they're normal
  client-side state (e.g. UI re-polling after a sidecar restart).

### Files touched

- `python/pipelines/streaming_linear.py` (+260 LOC)
- `python/pipelines/flux.py` (~60 LOC modified)
- `python/pipelines/load_utils.py` (~30 LOC modified)
- `python/main.py` (env var + 404 log level)
- `Documentation/PIPELINES.md` (rewritten FLUX section)
- `Launch Kraken Art.bat` (new — one-click launcher)

## 2026-05-22 (afternoon) — torch.compile installed + tested + opt-in only

After yesterday's WIP doc, ran a research pass via an agent (URLs in
[FLUX-SPEED-RESEARCH.md](FLUX-SPEED-RESEARCH.md)) to find what others have
done. The biggest claimed lever was **torch.compile**: PyTorch blog reports
~1.5× on H100, diffusers docs claim ~19% on Ampere with
`compile_repeated_blocks + channels_last`.

Implemented it: `transformer.compile_repeated_blocks(fullgraph=False, dynamic=False)`
on the fast-mode (no-streaming) path only, gated by
`performance.flux_compile` setting.

**Got Triton working on Windows:**
- Default PyTorch on Windows ships without Triton; `torch.compile` errors with
  "Cannot find a working triton installation"
- The community package `triton-windows` provides Windows builds
- Version pairing matters: torch 2.6.0+cu124 needs `triton-windows~=3.2`. The
  latest 3.7 fails with `AttrsDescriptor` import error.
- After `pip install "triton-windows>=3.2,<3.3"`, `torch.compile` works for
  trivial functions.

**Result: torch.compile is NET NEGATIVE on our arch.** Per-step bench:
- Baseline (no compile): median 2061 ms / min 2012 ms
- With compile + buffer=0.3 (forced fast mode): median **3445 ms / min 2988 ms**

That's 1.67× slower. Root cause: our `StreamingLinear` subclass's dynamic
dispatch (via `__dict__.get("_kraken_pending_w")`) prevents Dynamo from
proving its guards. Every step has guard misses → recompile churn → eager
fallback for affected blocks. Compiled blocks alternate with eager-falling
blocks in the per-step times (bimodal 3000/3450 ms pattern).

**Decision:** keep the implementation but ship `flux_compile=off` as default.
The real fix is dropping `StreamingLinear` on the fast path entirely
(separate refactor, only worthwhile for users with enough VRAM to skip
streaming). Documented in FLUX-SPEED-WIP.md + RESEARCH.md.

**Side benefit:** triton-windows is now installed, so anyone who *does*
refactor the fast path or wants to compile a smaller model (SDXL, fluxmania
with the right partition) gets compile for free.

## 2026-05-22 (midday) — Patches A/B/C benched and backed out

Per-step bench results showed all three patches are net-negative on this
hardware (3090 + torch 2.6 + diffusers 0.38). Detailed findings in
[FLUX-SPEED-WIP.md](FLUX-SPEED-WIP.md).

**The headline finding:** min step time is essentially constant at ~2010–2030 ms
across configs (no-patches, C-only, B+C, buffer=0.5). What varies is the median,
which the patches *worsen* by adding sync overhead. Bottom line: per-step
compute on this stack is ~2.06 s and that's the floor. Patches B and C have been
reverted; their code is preserved in `pipelines/kraken_flux_attn.py` and
`pipelines/flux.py::_install_pos_embed_cache` (both behind comments) for future
re-evaluation after a torch upgrade.

**The other finding:** the apparent morning slowdown was thermal — the 3090 was
soft-throttling at 82 °C with fans pegged at 100 %. The user resolved it via
MSI Afterburner (temp limit 91 °C + manual fan curve). Worth documenting in
onboarding for any future user reporting "FLUX is slow on a 3090."

**Shipping configuration unchanged from 2026-05-21 PM:** streaming offload
(Phases 1–3) at default 2.0 GB activation buffer = ~73 s warm / 28 steps on
FP32 flux_dev. The remaining ~25 s gap to ComfyUI's 48 s warm requires custom
CUDA kernels (their `comfy_kitchen` C++ extension), tracked as task #46
(DEFERRED).

**New tooling:** `python/bench_step_times.sh` — per-step wall-clock bench via
the progress endpoint. Records median/min/mean over steady-state steps (skips
the first 3 warmup steps). Use this for any future speed work.

## 2026-05-22 (morning) — Patches A/B/C started, blocked by env regression

Began the next round of FLUX speed work targeting ComfyUI's ~48 s warm number (we landed at ~73 s last night). Research agent identified the top 3 differences: unfused QKV projections, fp32 round-trip inside `apply_rotary_emb`, and per-step RoPE recompute. Wrote all three patches.

**Blocked by environmental regression** — see [FLUX-SPEED-WIP.md](FLUX-SPEED-WIP.md) for full state. Same code that ran 28-step gen in 75 s last night now takes 70+ s **per step** this morning. GPU clocks healthy (P0, 1935/2130 MHz, no throttle), 23 GB VRAM free at gen start. Likely culprit: Windows Defender Security Intelligence Update installed at 1:48 AM today. Resolution path: reboot + add Defender exclusions for the project folders. Code is reverted to last night's known-good state with the new patches commented out, ready to re-enable once baseline is restored.

**Files added/modified this morning:**
- `python/pipelines/kraken_flux_attn.py` (new, ~280 LOC) — `apply_rotary_emb_bf16`, `fuse_attention_qkv`, `install_kraken_attn_processor`, `KrakenFluxAttnProcessor`. All ready; fusion call commented out in flux.py pending a `chunk(3)` → contiguous-split fix documented in the WIP doc.
- `python/pipelines/flux.py` — added `_install_pos_embed_cache()` helper (commented out), wiring for Patch B (commented out).
- `python/main.py` — `DIFFUSERS_ATTN_BACKEND=_native_cudnn` env hint (kept commented; broke things on torch 2.6+cu124).
- `Documentation/FLUX-SPEED-WIP.md` (new) — full state-of-work doc with re-enable instructions, follow-up fix for Patch B, and bench plan.

## 2026-05-21 (late) — Phase 3 prefetch + Phase 4 attempt + watchdog + perf UI

### Phase 3 — layer prefetch
- Each streamed transformer block now has a pre-forward hook that kicks off the
  *next* streamed block's H2D copies on the mover stream while the current
  block is computing. By the time block N+1's forward runs, its weights are
  already on (or nearly on) GPU. `StreamingLinear.forward` checks for prefetched
  weights first and only falls back to sync streaming if no prefetch is in flight.
- Wrap-around chain: the last streamed block prefetches the first, so step N+1
  starts with the first block's weights already queued.
- Bypasses `nn.Module.__setattr__` (which tries to register tensor attributes as
  buffers) by writing prefetch slots directly into `__dict__`.
- Results: 79 s cold / 74 s warm on FP32 flux_dev (was 83/75 in Phase 2 — 1–4 s win).
  Modest because only 5/57 blocks stream — the bulk of compute is already on GPU.

### Phase 4 — persistent TE cache (DEFERRED)
- Attempted CPU-RAM caching of CLIP+T5 between gens. Code paths wired but
  caching itself is disabled. Hit Windows commit-limit OOMs (TE cache 10 GB
  + pinned streamed weights 1.3 GB + transformer mmap 22 GB > available commit).
  Infrastructure left in place (`_te_cache`, `_drop_te_cache`, `unload(drop_te=…)`
  flag) for a future opt-in setting once we have a commit-aware budget check.

### Watchdog (task #25)
- `src-tauri/src/lib.rs` — new background thread polls the spawned sidecar's
  `child.try_wait()` every 2 s. If the process has exited, respawns it,
  waits for `/health`, and emits a `sidecar-restarted` Tauri event.
- React side: `src/App.tsx` listens for the event, re-bootstraps the UI state
  (so model lists, GPU info etc. refresh against the new sidecar), and shows
  an amber banner ("Sidecar restarted after a crash — generate again to retry")
  that auto-dismisses after 15 s. Banner styles in `src/App.css`.
- Graceful shutdown via `RunEvent::Exit` sets a flag that stops the watchdog
  before it can respawn the sidecar we just killed.

### Performance settings UI
- `src/Settings.tsx` — new "FLUX performance" section with:
  - **Offload strategy** dropdown: Auto / Force fully resident / Force streaming
  - **Activation headroom** slider: 0.5–4 GB (default 2.0)
- Wired through `src/api/sidecar.ts` (Settings type gained `performance` field)
  to the existing backend `config_store.performance.*` keys. Backend reads
  these on every FLUX gen, so changes take effect on the next Generate without
  a restart.

## 2026-05-21 — Phase 1 foundation + Phase 2 (FLUX + ESRGAN)

### Phase 1 — foundation

- Scaffolded Tauri 2 + React 19 + Vite + TypeScript at `F:\Kraken Art`.
- Created Python 3.11 venv at `python\venv\`, installed torch 2.6.0+cu124 + torchvision, diffusers 0.38, transformers 4.57.6 (pinned <5 — see DEV.md), accelerate, safetensors, fastapi, uvicorn[standard], websockets, pydantic, pynvml, structlog. Real-ESRGAN/upscale support via spandrel 0.4.2.
- Built the Python sidecar core: FastAPI on `127.0.0.1:7780` with `/api/gpu`, `/api/deps`, `/api/models`, `/api/models/refresh`, `/api/generate`, `/api/jobs/{id}`, `/api/jobs/{id}/cancel`, `/ws/jobs/{id}`. Model scanner walks `models/` recursively, categorized by folder name. CORS allows the Tauri webview origin.
- Wired Tauri Rust host to spawn + supervise the sidecar with stdout/stderr piping to `[sidecar]` log lines, port-conflict detection, and kill-on-exit.
- Implemented SDXL pipeline (`pipelines/sdxl.py`) using `StableDiffusionXLPipeline.from_single_file` with sampler/scheduler routing (Euler/Euler-A/DDIM/UniPC/DPM++ 2M × normal/karras/exponential/sgm_uniform/beta/simple), LoRA multi-adapter loading via `set_adapters`, textual-inversion support, VAE override, VAE tiling for 24 GB headroom.
- WebSocket progress hub with per-job event bus, thread-safe via `loop.call_soon_threadsafe`. Single-worker FIFO queue keeps VRAM predictable.
- Robocopied 462 GB / 184 model files / 90 dirs from `D:\AI_Art\ComfyUI\models\` to `F:\Kraken Art\models\` (D: was at 7 GB free; F: had 1.3 TB).

### React UI

- Built the Kraken deep-sea theme: abyssal blue/teal background, bioluminescent cyan accent (`#00E5FF`), CSS variables for the full palette.
- Three-pane layout: left (system status + model counts), center (prompt + generated image grid), right (model selectors + params).
- Full selector surface: architecture mode (SDXL / Illustrious / FLUX1 supported; FLUX2 / Qwen-Image / HunYuan / WAN / LTX listed but backend-rejected with hints), checkpoint vs diffusion-model split, VAE override, text encoders (1 or 2 slots per arch), LoRA multi-select with -1..+2 weight sliders, embeddings, dimensions (presets + custom W/H), sampler + scheduler, batch count, random/fixed seed, upscale (toggle + mode + model + factor + denoise + tile size).
- Generation flow: submit → WebSocket subscribe → progress bar + per-step counter + cancel; thumbnails on each image-emit event; failure box for errors; auto-tune CFG + steps from model filename (`*schnell*` → CFG 1/4 steps, `*turbo* / *lightning* / *lcm* / *hyper*` → CFG 1/8 steps, `*flux* / *z_image*` → CFG 3.5/28 steps).

### Resilience + UX

- Ring-buffer logger (2000 entries) with disk rotation to `logs/sidecar-YYYY-MM-DD.log`. UI log drawer with timestamp + level + logger + message; ERROR/CRITICAL auto-opens the drawer and adds a red topbar badge; Copy-all (clipboard) / Pause / Clear / ✕. Drawer is drag-resizable from its top edge; height persists via localStorage.
- Clear VRAM button in topbar — POSTs `/api/clear_memory` which unloads every cached pipeline (SDXL + FLUX + upscalers), runs `gc.collect()` + `torch.cuda.empty_cache()` + `torch.cuda.ipc_collect()`. GPU card shows VRAM-free with 5 s auto-refresh, turning red below 4 GB.
- WebSocket-close auto-reset: if the WS dies on a non-terminal status (sidecar crashed mid-job), the UI snaps back to idle with a red message instead of being stuck on "Generating…".
- Global FastAPI exception handlers route all HTTPException 4xx + RequestValidationError 422 + unhandled 5xx into the log buffer with full traceback for 5xx.
- Friendly architecture-mismatch hints when SDXL loader is fed a non-SDXL checkpoint (Qwen / FLUX / HunYuan all detected from the exception message).

### Phase 2 (in progress)

- **FLUX1 pipeline** (`pipelines/flux.py`): assembles `FluxPipeline` from local transformer + VAE + CLIP-L + T5-XXL safetensors. HF configs (tiny) auto-downloaded for tokenizer + model architecture only; heavy weights stay local. `low_cpu_mem_usage=True` streams the load via mmap to avoid CPU-RAM OOM. `enable_model_cpu_offload()` fits FLUX FP16 + T5 on a 24 GB card. LoRA multi-adapter support. Negative prompts ignored (FLUX doesn't use them).
- **ESRGAN upscale** (`pipelines/upscale_esrgan.py`): spandrel-based universal loader. Works for any of the user's 10 upscale models (Real-ESRGAN, NMKD, UltraSharp, AnimeSharp, Remacri, ESRGAN_4x, etc.). Auto-resizes from the model's intrinsic scale to the user's target factor. Outputs both original + `-up.png`. Hooked into SDXL and FLUX pipelines as post-processing when "upscale enabled".
- Cross-architecture VRAM hygiene: `/api/generate` calls `unload()` on the *other* pipelines before submitting (SDXL→FLUX unloads SDXL, FLUX→SDXL unloads FLUX).
- Logo + app icons: integrated user's hand-drawn kraken into the topbar (36 px with cyan drop-shadow glow) and regenerated all Windows/macOS/iOS/Android icon variants via `cargo tauri icon`.

### Documentation

- Created `Documentation/` with [README](README.md), [ARCHITECTURE](ARCHITECTURE.md), [PIPELINES](PIPELINES.md), [MODELS](MODELS.md), [DEV](DEV.md), and this CHANGELOG.

### Pending (tracked tasks)

- #9 dependency-remedy wizard (auto-detect missing pieces, offer one-click fix)
- #11 FLUX2, Qwen-Image, HunYuan, Z-Image image archs
- #12 WAN / LTX video pipelines + video UI surface
- #13 / #18 / #19 / #20 port ACE-Step / LuxTTS / lyric-timing / audio video renderer from `F:\Kraken_Audio`
- #15 Ultimate SD Upscale (tile + img2img refine, four seam-fix modes)
- #16 face / region detailer (detect → crop → refine → paste)
- #17 iterative upscale (progressive scale steps)
- #21 MCP server exposing all capabilities to a local AI agent
- #22 UI tabs for Image / Video / Music + agent activity feed
- #25 Tauri Rust sidecar watchdog + auto-restart on crash
