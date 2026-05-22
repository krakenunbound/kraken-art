# TODO

Roadmap items beyond the live task tracker. The task tracker (visible in-session) holds discrete implementation work; this is the wider product / polish / contributor view, organised by area.

The pole-star: **a robust, public-release ComfyUI replacement that runs on a variety of user systems**. See [feedback-no-bandaids](../../C:/Users/The Kraken/.claude/projects/F--Kraken-Art/memory/feedback_no_bandaids.md).

---

## Image generation — architecture coverage

Each architecture needs its own pipeline module under `python/pipelines/`, registered in `python/api/generate.py`'s arch router, and validated end-to-end with the per-file picker.

- [x] **SDXL** / **Illustrious** — `pipelines/sdxl.py`
- [~] **FLUX1** — `pipelines/flux.py` (works but slow; FP8 T5 fix in flight — see [BUGS B-001](BUGS.md))
- [ ] **FLUX2** — needs Gemma text encoder; pipeline + config + UI exposure (task #11)
- [ ] **Qwen-Image** — uses Qwen2.5-VL text encoder + shared FLUX VAE
- [ ] **Z-Image (base + turbo)** — uses Qwen-derived TE
- [ ] **HunYuan-DiT** — bilingual text encoder
- [ ] **Stable Diffusion 3 / 3.5** — for completeness, if community asks

For each: ship a `model_configs/<arch>/` directory with transformer + VAE + scheduler configs so the load is fully offline (no HF gating).

## Video

- [ ] **WAN i2v / t2v** (`wan2.2_*` files) — separate UI surface (`Video` tab), per-frame progress, mp4 output
- [ ] **LTX video** (`ltx-2.3-22b-dev.safetensors`)
- [ ] **HunYuan Video** when models land

Video output: ffmpeg-encoded mp4, frame thumbnails for preview, configurable fps/aspect/duration.

## Audio (port from F:\Kraken_Audio)

See [reference-kraken-audio](../../C:/Users/The Kraken/.claude/projects/F--Kraken-Art/memory/reference_kraken_audio.md). User decision: copy code into Kraken Art, leave original `F:\Kraken_Audio` working.

- [ ] **ACE-Step music gen** (task #13) — port the orchestration; spec fields from the Kraken_Audio AI Agent guide
- [ ] **LuxTTS voice cloning** (task #18) — reference audio + target text → cloned WAV
- [ ] **Lyric-timing pipeline** (task #19) — Demucs vocal-sep + Faster-Whisper `large-v3-turbo` → `timed_lyrics.{json,lrc,ass}`
- [ ] **Audio video renderer** (task #20) — visualizer + lyric overlay + ffmpeg → mp4 (16:9 / 9:16)

## Upscaling

- [x] **Basic ESRGAN** (spandrel-based, single-pass) — works for SDXL + FLUX
- [ ] **Ultimate SD Upscale** (task #15) — tile + img2img refine. All four seam-fix modes (None / Band-pass / Half-tile / Half-tile+intersections)
- [ ] **Iterative upscale** (task #17) — progressive scale steps
- [ ] **Face / region detailer** (task #16) — detect (YOLO face / SAM) → crop → refine → paste

## AI Agent interface (MCP)

Task #21. The whole point of the original architecture decision — a local AI agent should drive Kraken Art end-to-end.

- [ ] MCP server over HTTP, exposing tools: `list_models`, `refresh_models`, `gpu_status`, `generate_image`, `generate_video`, `generate_music`, `clone_voice`, `sync_lyrics`, `render_music_video`, `get_job`, `cancel_job`
- [ ] Live tool calls (not spec-JSON-batch) — conversational UX
- [ ] Provide MCP config snippets for Claude Desktop, LM Studio, Cherry Studio, OpenWebUI
- [ ] In-UI "Agent activity" feed showing tool calls in real time (task #22)

## Reliability + hygiene

These are the items that separate "works on my machine" from "ships on GitHub."

- [ ] **Sidecar watchdog + auto-restart** (task #25) — Rust host monitors `/health`, respawns on crash; surfaces "sidecar crashed N times" to UI after repeated failures
- [ ] **True cancel during model load** ([B-003](BUGS.md)) — option: kill-and-respawn sidecar for hard cancel; option: chunked load with cancel checks
- [ ] **Dependency-remedy wizard** (task #9) — on first launch, detect missing pieces (CUDA mismatch, transformers version, etc.) and offer one-click fix
- [ ] **HF token configuration UI** — for users who DO have access to gated repos (FLUX.1-dev, etc.), let them paste a token in settings
- [ ] **Test matrix CI** — on PR, run a minimal SDXL gen + a FLUX1 gen on a CPU-only fixture (small models, low steps) to catch regressions before they ship

## UX / polish

- [ ] **Output gallery** — browse `outputs/YYYY-MM-DD/`, click an image to see params, re-use seed
- [ ] **Generation history** — last N jobs with their settings, "re-run with these settings" button
- [ ] **Workflow presets** — save current arch + model + lora + sampler combo as a named preset
- [ ] **Per-LoRA arch tagging** — peek safetensors metadata at scan time to identify each LoRA's target arch (FLUX vs SDXL vs etc.); filter the LoRA dropdown to compatible ones for the current architecture
- [ ] **Logs drawer**: rate-limit identical lines (currently the orphan-poll 404 spam fills the buffer); collapsible tracebacks
- [ ] **First-run experience**: detect that `models/` is empty and offer to symlink/copy from an existing ComfyUI install
- [ ] **Tabs for Image / Video / Music** (task #22) — currently everything is the image surface

## Documentation

- [x] [README.md](README.md), [ARCHITECTURE.md](ARCHITECTURE.md), [PIPELINES.md](PIPELINES.md), [MODELS.md](MODELS.md), [DEV.md](DEV.md), [CHANGELOG.md](CHANGELOG.md), [BUGS.md](BUGS.md), [TODO.md](TODO.md)
- [ ] **CONTRIBUTING.md** — coding standards, the no-band-aids principle, PR template
- [ ] **INSTALL.md** — fresh-machine setup from a release artifact (not from source)
- [ ] **MODELS_GUIDE.md** — for end users: which file to download for each arch, where to put it, what to expect

## Packaging + release

- [ ] **Bundled installer** — Tauri can produce a Windows `.msi` / macOS `.dmg` / Linux `.AppImage`. Decide whether Python venv ships embedded or the installer creates one on first run
- [ ] **Python embedded** — for a portable one-folder Windows install (similar to ComfyUI's `python_embeded` approach) — works without a separately-installed Python
- [ ] **GitHub Actions** — build matrix (Win / mac / Linux), release on tag
- [ ] **License** — pick one (MIT? Apache 2.0?) before public push
- [ ] **GitHub repo bootstrap** — README, issue templates, PR template, code of conduct

## Stretch / future

- [ ] **Multi-GPU support** — distribute transformer across GPUs (diffusers + accelerate support this)
- [ ] **Cloud burst mode** — fall back to a remote API for users without local GPU
- [ ] **Plugin / extension system** — let third parties add custom pipelines without forking
- [ ] **Custom workflow chain** — sequence of generations (image → upscale → animate → music) configured visually
- [ ] **TI / hypernetwork support** — currently embeddings folder is wired but pipeline integration is partial
