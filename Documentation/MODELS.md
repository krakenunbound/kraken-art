# Models folder

Everything under `F:\Kraken Art\models\` is scanned at sidecar startup and on **Refresh models**. The scan walks each category folder recursively (subdirs are fine — paths are shown as `Subdir/file.safetensors` in the dropdowns).

## Recognized extensions

`.safetensors`, `.sft`, `.ckpt`, `.pt`, `.pth`, `.bin`, `.gguf`

If you drop a model with an extension not in that list, add it to `python/config.py → MODEL_EXTS` and refresh.

## Categories

| Folder under `models/` | Used by | Notes |
|---|---|---|
| `checkpoints/`         | SDXL, Illustrious, and other all-in-one archs | Includes UNet + bundled VAE + bundled text encoders in a single file. |
| `diffusion_models/`    | FLUX1, FLUX2, Qwen-Image, HunYuan, Z-Image, WAN, LTX | The transformer/DiT only. Requires separate VAE + text-encoder picks. |
| `vae/`                 | All archs (optional override for checkpoint-style; required for components-style) | FLUX and Qwen-Image share `ae.safetensors`. |
| `text_encoders/`       | Components-style archs (FLUX*, Qwen, HunYuan, Z-Image, WAN) | CLIP-L + T5-XXL for FLUX, Gemma for FLUX2, Qwen-derived for Z-Image / Qwen-Image, UMT5 for WAN. |
| `clip_vision/`         | IP-Adapter style image conditioning (future) | Empty for now. |
| `loras/`               | All archs | LoRA is architecture-specific — a FLUX LoRA won't work on SDXL and vice versa. |
| `controlnet/`          | ControlNet conditioning (future) | Empty for now. |
| `upscale_models/`      | ESRGAN / Real-ESRGAN / SwinIR / etc. | `.pth` files, loaded via spandrel. See [PIPELINES.md → Upscale](PIPELINES.md#upscale-esrgan). |
| `embeddings/`          | SDXL textual inversions | Empty for now. |
| `style_models/`        | FLUX redux, etc. (future) | Empty for now. |
| `ip_adapter/`          | IP-Adapter (future) | Empty for now. |

## Cross-folder file mapping

Same file occasionally appears under multiple categories because ComfyUI's user folder organization is loose. The display name shows the path relative to its category folder, so:

```
checkpoints/Flux1DFP16/flux1CompactCLIPAnd_Flux1DevFp16.safetensors
```

…is an *all-in-one* FLUX variant designed for ComfyUI's `CheckpointLoaderSimple` (everything baked into one file). Kraken Art's FLUX1 pipeline expects **separated** components from `diffusion_models/` + `vae/` + `text_encoders/`. The all-in-ones in `checkpoints/` will likely fail to load via our FLUX path. The 22 GB pure transformers under `diffusion_models/` are the right pick for our pipeline.

## "Mode" by architecture

- **Checkpoint mode** (SDXL, Illustrious): the UI shows only a `Checkpoint` dropdown and an optional VAE override.
- **Components mode** (FLUX, Qwen-Image, etc.): the UI shows `Diffusion model` + `VAE` + N `Text encoder` slots.

The mode flips automatically when you change the Architecture dropdown — see [PIPELINES.md](PIPELINES.md) for which arches need which components.

## Refreshing

Two ways:
1. **Refresh models** button in the topbar — re-scans without restarting the sidecar.
2. Restart the sidecar — picks up any change to `config.py` (e.g. new file extensions).

The scan is fast even on a 462 GB model collection — it walks file metadata only, not contents.

## What's in the user's collection today

Roughly:
- **7 checkpoints** (SDXL variants + FLUX all-in-ones + LTX + stable-audio)
- **13 diffusion_models** (FLUX, FLUX2, Qwen-Image, Z-Image variants, WAN, JuggernautZ)
- **5 VAEs** (FLUX/Qwen shared, Qwen-Image, WAN, Z-Image)
- **11 text encoders** (CLIP-L, T5-XXL FP16/FP8, Gemma 12B, Qwen 2.5 VL, Qwen 3 4B, UMT5, Z-Image, t5-base, ViT-L-14)
- **27 LoRAs** (mixed SDXL + FLUX)
- **10 upscalers** (Real-ESRGAN, NMKD family, UltraSharp, AnimeSharp, Remacri, ESRGAN_4x)

Total ~462 GB. Counts update on Refresh.
