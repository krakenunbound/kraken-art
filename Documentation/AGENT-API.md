# Kraken Art — Headless / Agent API

Drive Kraken Art **without the GUI**: list models, pick settings + resolution, generate
images, and find the files — all over a small local HTTP API. An agent only needs this
document; it does not need to read the codebase.

> TL;DR: the app is a FastAPI sidecar on `http://127.0.0.1:7780`. You `POST /api/generate`,
> get a `job_id`, poll `GET /api/jobs/{job_id}` until `status == "succeeded"`, then read the
> saved file paths from `result.outputs`. Images land in `F:\Kraken Art\outputs\<date>\`.

---

## 1. Make sure the sidecar is running

The sidecar is up automatically whenever the **Kraken Art desktop app is open**. To run it
**headless** (no GUI), start it directly:

```bash
cd "F:/Kraken Art/python"
venv/Scripts/python.exe main.py     # serves on 127.0.0.1:7780, logs to ../logs/
```

Check it's alive:

```bash
curl -s http://127.0.0.1:7780/health
# {"ok":true,"service":"kraken-art-sidecar","models_root":"...","outputs_root":"F:\\Kraken Art\\outputs"}
```

Notes:
- **One job at a time.** A single FIFO worker runs jobs sequentially (keeps VRAM
  predictable). Submitting while another job runs just queues yours.
- All endpoints are under `/api` **except** the health check (`/health`) and the progress
  WebSocket (`/ws/...`).

---

## 2. Discover what models are installed

```bash
curl -s http://127.0.0.1:7780/api/models
```

Returns categories, each a list of entries. The field you care about is **`name`** — a
path **relative to its category folder**, with forward slashes. That exact string is what
you pass back in a generate request.

```jsonc
{
  "root": "F:\\Kraken Art\\models",
  "categories": {
    "checkpoints":      [{ "name": "SDXL/epicrealismXL_vxviiCrystalclear.safetensors", ... }],
    "diffusion_models": [{ "name": "Flux 1D FP16/fluxmania_kreamania.safetensors", ... }],
    "vae":              [{ "name": "FLUX1/fluxVaeSft_aeSft.sft", ... }],
    "text_encoders":    [{ "name": "clip_l.safetensors", ... }, { "name": "t5/t5xxl_fp8_e4m3fn.safetensors", ... }],
    "loras":            [ ... ],
    "upscale_models":   [{ "name": "4x-UltraSharp.pth", ... }]
  },
  "counts": { "checkpoints": 12, "diffusion_models": 4, ... }
}
```

**Model files live under** `F:\Kraken Art\models\<category>\`. Drop new models in the
matching folder (`checkpoints`, `diffusion_models`, `vae`, `loras`, `upscale_models`, …) and
they appear here (the GUI's "Refresh models" rescans; a fresh sidecar start also rescans).

---

## 3. Generate an image

`POST /api/generate` with a JSON body. It returns immediately with a `job_id`; the work runs
in the background. The **`arch`** field selects the pipeline and decides which model fields
are required.

### Architectures and their required fields

| `arch`       | Required model fields                                                                 | Notes |
|--------------|----------------------------------------------------------------------------------------|-------|
| `sdxl`       | `checkpoint` (all-in-one .safetensors)                                                  | also `illustrious`. Realistic/anime SDXL checkpoints. |
| `flux1`      | `diffusion_model`, `vae`, `text_encoders` = `[clip_l, t5xxl]` (2 slots)               | best prompt adherence. **Keep `cfg: 1`.** |
| `z_image`    | `diffusion_model`, `vae`, `text_encoders` = `[qwen3_te]` (slot 1)                      | Z-Image Turbo. |
| `ideogram4`  | `diffusion_model` = the virtual `"Ideogram 4 NF4 (local)"`                              | great text-in-image; uses structured magic-prompt. |

### Common parameters (all archs)

| Field | Default | Meaning |
|-------|---------|---------|
| `prompt` | `""` | positive prompt |
| `negative` | `""` | negative prompt (FLUX ignores it) |
| `width`, `height` | `1024` | output resolution in px (use multiples of 64) |
| `steps` | `30` | denoise steps |
| `cfg` | `7.0` | guidance. **SDXL ~5–7; FLUX/Z-Image = 1; Ideogram ~7** |
| `sampler` | `dpmpp_2m` | `euler`, `euler_a`, `dpmpp_2m`, `dpmpp_sde`, `heun`, `unipc`, `ddim`, … |
| `scheduler` | `normal` | `normal`, `karras`, `exponential`, `sgm_uniform`, `beta`, `simple` |
| `clip_skip` | `null` | SDXL/Illustrious only |
| `count` | `1` | how many images (sequential, seed increments per image) |
| `seed` | `null` | `null` = random; integer = reproducible |
| `loras` | `[]` | `[{ "name": "<lora rel path>", "weight": 0.8 }]` |

### Example — SDXL, realistic, 1536×1024

```bash
curl -s -X POST http://127.0.0.1:7780/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "arch": "sdxl",
    "checkpoint": "SDXL/epicrealismXL_vxviiCrystalclear.safetensors",
    "prompt": "a weathered dwarven blacksmith at his forge, embers, dramatic rim light, fantasy concept art",
    "negative": "blurry, lowres, extra fingers, watermark, text",
    "width": 1536, "height": 1024,
    "steps": 30, "cfg": 6, "sampler": "dpmpp_2m", "scheduler": "karras",
    "count": 1, "seed": null
  }'
# -> {"job_id":"ab12...","status":"queued"}
```

### Example — FLUX (best prompt adherence; cfg = 1)

```bash
curl -s -X POST http://127.0.0.1:7780/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "arch": "flux1",
    "diffusion_model": "Flux 1D FP16/fluxmania_kreamania.safetensors",
    "vae": "FLUX1/fluxVaeSft_aeSft.sft",
    "text_encoders": ["clip_l.safetensors", "t5/t5xxl_fp8_e4m3fn.safetensors"],
    "prompt": "a sprawling tavern interior, adventurers at a long oak table, warm candlelight",
    "width": 1536, "height": 1024,
    "steps": 28, "cfg": 1, "sampler": "euler_a", "scheduler": "beta",
    "count": 1
  }'
```

### Example — Ideogram 4 (text in image)

```bash
curl -s -X POST http://127.0.0.1:7780/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "arch": "ideogram4",
    "diffusion_model": "Ideogram 4 NF4 (local)",
    "prompt": "a tavern sign reading \"The Moist Mermaid\", carved wood, hanging from an iron bracket",
    "width": 1024, "height": 1024,
    "steps": 30, "cfg": 7,
    "ideogram_magic": true, "ideogram_magic_mode": "local", "ideogram_speed_mode": "high"
  }'
```

---

## 4. Wait for it + get the file paths

Poll the job until it finishes:

```bash
curl -s http://127.0.0.1:7780/api/jobs/<job_id>
```

Snapshot shape:

```jsonc
{
  "id": "ab12...",
  "status": "queued | running | succeeded | failed | cancelled",
  "progress": { "step": 18, "total_steps": 30, "image_index": 0, "total_images": 1, "message": "" },
  "error": null,
  "result": {                       // present when status == "succeeded"
    "kind": "image",
    "count_produced": 1,
    "output_dir": "F:\\Kraken Art\\outputs\\2026-06-17",
    "outputs": [
      {
        "path": "F:\\Kraken Art\\outputs\\2026-06-17\\181255-ab12cdef-00.png",
        "rel_path": "2026-06-17/181255-ab12cdef-00.png",
        "filename": "181255-ab12cdef-00.png",
        "seed": 12345, "width": 1536, "height": 1024,
        "upscaled_path": "...-00-up.png"   // only if upscale_enabled
      }
    ]
  }
}
```

Read `result.outputs[*].path` for the absolute file on disk, or `rel_path` to fetch it over
HTTP (next section). `progress.step / total_steps` lets you show a progress meter while
waiting.

### Minimal submit-and-wait helper (bash)

```bash
gen() {
  local body="$1"
  local id
  id=$(curl -s -X POST http://127.0.0.1:7780/api/generate \
        -H "Content-Type: application/json" -d "$body" | python -c "import sys,json;print(json.load(sys.stdin)['job_id'])")
  while :; do
    local snap status
    snap=$(curl -s "http://127.0.0.1:7780/api/jobs/$id")
    status=$(printf '%s' "$snap" | python -c "import sys,json;print(json.load(sys.stdin)['status'])")
    case "$status" in
      succeeded) printf '%s' "$snap" | python -c "import sys,json;[print(o['path']) for o in json.load(sys.stdin)['result']['outputs']]"; break ;;
      failed|cancelled) printf '%s\n' "$snap"; break ;;
    esac
    sleep 2
  done
}
# gen '{"arch":"sdxl","checkpoint":"SDXL/epicrealismXL_vxviiCrystalclear.safetensors","prompt":"a goblin warband","width":1024,"height":1024,"steps":30,"cfg":6}'
```

---

## 5. Where images go

```
F:\Kraken Art\outputs\<YYYY-MM-DD>\<HHMMSS>-<job-id-8>-<NN>.png
```

- `<NN>` is the image index within a batch (`00`, `01`, …).
- Upscaled versions are saved alongside as `...-<NN>-up.png` (or `...-usdu.png` from the
  standalone upscale endpoint).
- **Every PNG has the full generation settings embedded** in its metadata
  (`kraken_settings` JSON + an A1111/Civitai-readable `parameters` block), so the files are
  self-documenting — seed, model, prompt, sampler, size, etc. travel with the image.

### Browse / fetch outputs over HTTP

```bash
curl -s "http://127.0.0.1:7780/api/outputs/latest-images?limit=20"   # newest images + model labels
curl -s "http://127.0.0.1:7780/api/outputs/list?limit=200"           # all output files, newest first
curl -s "http://127.0.0.1:7780/api/outputs/settings/<rel_path>"      # embedded settings for one image
curl -s "http://127.0.0.1:7780/api/outputs/file/<rel_path>" -o out.png   # download the actual file
```

---

## 6. Upscale an existing image (optional)

`POST /api/upscale` runs ESRGAN (fast) or **USDU** (tile + img2img refine) on any gallery
image. Same job/poll flow as generate.

```bash
curl -s -X POST http://127.0.0.1:7780/api/upscale \
  -H "Content-Type: application/json" \
  -d '{
    "source_rel_path": "2026-06-17/181255-ab12cdef-00.png",
    "mode": "usdu",                       // "esrgan" | "usdu"
    "upscale_model": "4x-UltraSharp.pth",
    "size_mode": "resolution",            // "factor" | "resolution"
    "target_w": 3840, "target_h": 2160,   // (or "size_mode":"factor","factor":2)
    "refine_arch": "sdxl",
    "refine_checkpoint": "SDXL/epicrealismXL_vxviiCrystalclear.safetensors",
    "steps": 20, "denoise": 0.15, "tile_size": 1024
  }'
```

- `denoise` = how much detail is **replaced**. Low (0.10–0.15) = upscale while preserving the
  original; higher lets the model repaint.
- Output lands in `outputs\<date>\` like any other image, exact target resolution.

---

## 7. Optional: prompt helper

If you have a one-line scene idea and want it expanded into a model's native prompt dialect
(no LLM, deterministic):

```bash
curl -s "http://127.0.0.1:7780/api/prompt-builder/options"        # lists style/lighting/camera/mood choices
curl -s -X POST http://127.0.0.1:7780/api/prompt-builder/build \
  -H "Content-Type: application/json" \
  -d '{"arch":"flux1","subject":"a haunted lighthouse on a storm cliff","style":"auto","lighting":"auto","camera":"auto","mood":"auto"}'
# -> { "prompt": "...", "aspect_ratio": "..." }   feed .prompt straight into /api/generate
```

---

## 8. Endpoint quick reference

| Method | Path | Purpose |
|--------|------|---------|
| GET  | `/health` | sidecar alive + roots |
| GET  | `/api/models` | list installed models (use `name` in requests) |
| POST | `/api/generate` | start an image job → `{job_id}` |
| POST | `/api/generate/video` | WAN i2v video job |
| POST | `/api/upscale` | ESRGAN / USDU upscale job |
| GET  | `/api/jobs/{id}` | poll status + result |
| POST | `/api/jobs/{id}/cancel` | cancel a running/queued job |
| GET  | `/api/jobs` | list all jobs |
| GET  | `/api/outputs/latest-images?limit=N` | newest images |
| GET  | `/api/outputs/list?limit=N` | all output files |
| GET  | `/api/outputs/file/{rel_path}` | download a file |
| GET  | `/api/outputs/settings/{rel_path}` | embedded settings of an image |
| GET/POST | `/api/prompt-builder/options` · `/build` | scene → native prompt |
| WS   | `/ws/jobs/{id}` | live progress events (optional; polling works fine) |

## 9. Gotchas

- **FLUX & Z-Image want `cfg: 1`.** Higher CFG washes them out.
- **FLUX needs 2 text encoders** in order: `clip_l` (slot 1), `t5xxl` (slot 2).
- **Resolution**: multiples of 64 are safest. The GPU is a 24 GB RTX 3090 — 1536×1024 / 1024×1536
  are comfortable; very large direct gens may OOM (use upscale to go bigger).
- **One job at a time** — don't fire a batch of parallel `/api/generate` calls expecting
  concurrency; they queue.
- A model `name` must match `/api/models` exactly (relative path, forward slashes).
