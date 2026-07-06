# WAN 2.2 Video Generation

Kraken Art has a dedicated Video tab for local WAN 2.2 generation. The current
implementation supports both:

- **T2V**: text prompt directly to video.
- **I2V**: first frame image to video, including captured frames from a previous
  video so a clip can be extended.

This guide records the July 5, 2026 session work, the model layout now expected
by the app, and the verification baseline.

## Why this changed

WAN 2.2 is the best fit for the local RTX 3090 target because it has open local
weights, strong community tooling, and FP8 14B variants that can be loaded on a
24 GB card with CPU offload. The app already had an i2v path; this session
expanded it into a practical local video workflow:

- T2V and I2V are now separate modes.
- Dual WAN 2.2 high-noise / low-noise experts are exposed explicitly.
- WAN speed LoRAs are selectable for each expert.
- Generated videos can provide their own first frame for I2V continuation.

## Model files used

The T2V FP8 14B experts were downloaded and placed here:

```text
F:\Kraken Art\models\diffusion_models\WAN22\wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors
F:\Kraken Art\models\diffusion_models\WAN22\wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors
```

The user's WAN 2.2 InterFrame LoRAs were copied from Downloads into the app's
model tree:

```text
F:\Kraken Art\models\loras\WAN22\WAN2.2-HighNoise_IF.safetensors
F:\Kraken Art\models\loras\WAN22\WAN2.2-LowNoise_IF.safetensors
```

The previous I2V LightX2V LoRAs remain usable for I2V. The app now scans both
mode-specific WAN LoRAs and generic WAN LoRAs so the new `HighNoise_IF` and
`LowNoise_IF` files are not hidden just because older i2v LoRAs exist.

## Implementation notes

Key files:

- `src/Video.tsx`
  - Adds the T2V/I2V mode switch.
  - Allows T2V generation without an input image.
  - Adds explicit high-noise and low-noise LoRA dropdowns plus weights.
  - Adds a `Use last frame` button on generated video results.
- `src/api/sidecar.ts`
  - Adds `mode?: "i2v" | "t2v"` to `VideoGenerateParams`.
  - Adds `captureLastVideoFrame(source_rel_path, offset_seconds)`.
- `python/api/generate.py`
  - Validates `mode`.
  - Requires an input image only for I2V.
  - Keeps the requirement that WAN generation needs both experts, WAN VAE, and
    UMT5 text encoder.
- `python/pipelines/wan_video.py`
  - Uses `WanPipeline` for T2V.
  - Uses `WanImageToVideoPipeline` for I2V.
  - Uses the T2V transformer config override `in_channels=16`.
  - Uses boundary ratio `0.875` for T2V and the existing `0.9` I2V boundary.
  - Includes `image` in pipeline kwargs only for I2V.
- `python/api/outputs.py`
  - Adds `POST /api/outputs/capture-last-frame`.
  - Captures the final decodable frame with ffmpeg.
  - Falls back to exact final-frame selection by `ffprobe` frame count when
    `ffmpeg -sseof` succeeds but produces no file on very short clips.

## Last-frame extension workflow

1. Generate or import a source video.
2. On the Video tab, click `Use last frame` on the video result.
3. Kraken saves a PNG under:

```text
F:\Kraken Art\outputs\<date>\frames\
```

4. The app switches to I2V mode and loads that PNG as the input frame.
5. Generate the next segment.

This is intentionally frame-based instead of trying to continue latent state.
It is robust across restarts, editable by the user, and compatible with any
previous MP4 that ffmpeg can decode.

## Verification baseline

Smoke tests run on July 5, 2026:

- Frontend build:

```powershell
npm run build
```

- Python compile check:

```powershell
python\venv\Scripts\python.exe -m py_compile python\api\generate.py python\api\outputs.py python\pipelines\wan_video.py
```

- Real T2V smoke:

```text
F:\Kraken Art\outputs\2026-07-05\154131-t2vsmoke.mp4
```

Parameters: WAN 2.2 T2V high/low FP8 experts, IF LoRAs, 320x192, 5 frames,
1 step, 8 fps.

- Captured last frame:

```text
F:\Kraken Art\outputs\2026-07-05\frames\154558-lastframe-154131-t2vsmoke.png
```

Verified by PIL as `320x192 RGB`.

- Real I2V continuation smoke:

```text
F:\Kraken Art\outputs\2026-07-05\154631-i2vexten.mp4
```

Probe result: `320x192`, `5` frames, `8 fps`, `0.625s`.

## Known notes

- Current tests used tiny frame counts and one sampling step to prove plumbing,
  not quality.
- Longer, higher quality clips should use normal WAN step counts and practical
  frame counts for the 3090 VRAM/time budget.
- PEFT may warn about multiple adapters when loading both high/low LoRAs. The
  smoke tests completed and unload freed both LoRAs, so this is currently a
  warning to monitor rather than a blocking issue.
