# Video Upscale

Kraken Art includes a dedicated **Video Up** tab for upscaling existing video
clips. The current production default is the preservation path:
**ESRGAN / RealESRGAN_x2 + optional RIFE 60fps interpolation**.

## Workflow

1. Add source clips in the top **Recent / imported videos** lane.
   - Use **Add videos** for file picker import.
   - Drag video files onto the Kraken window.
   - Paste a local file path and click **Import path**.
2. Pick an engine, model, target size, framerate, and quality settings.
3. Click **Upscale video**.
4. Review the result in the middle **Compare preview**:
   - Left: source video.
   - Right: selected or latest upscale result.
5. Use the bottom **Upscale results** lane to select, delete, or open finished
   `*-vup.mp4` outputs.

The app chrome also includes **Open outputs**, which opens the real
`outputs/` folder in Explorer.

## Source And Result Bins

The video tab intentionally separates media:

- Top lane: source/imported/generated videos that can be used as input.
- Bottom lane: finished video-upscale results, identified by `-vup` filenames.

Both lanes support:

- Checkbox selection.
- Check all.
- Delete checked.
- Delete all.
- Per-thumbnail delete button.

Video deletes happen directly through the Kraken outputs API; there is no
browser confirmation dialog.

## Engines

### ESRGAN Preserve Source

Default for normal use.

- Best current speed/quality balance on the RTX 3090 test system.
- Uses Spandrel-loaded image upscalers frame by frame.
- `RealESRGAN_x2.pth` is the default because it was the most visually pleasing
  test result on the Grok Imagine clip.
- `Tile size 0` means full-frame processing. Use tiled mode only when a 2K/4K
  render hits VRAM limits.

Recommended baseline:

```text
Engine: ESRGAN - preserve source
Model: RealESRGAN_x2.pth
Output: 2K / 1440p
Interpolation: RIFE Vulkan
Framerate: 60fps
Tile size: 0
Keep source audio: on
```

### SeedVR2 Cinematic Detail

Best for reconstruction and added AI detail, but much slower and more likely to
change identity, text, logos, clothing detail, or background structure.

SeedVR2 does not expose a true Ultimate-SD-Upscale-style denoise strength.
Kraken implements **AI detail strength** by blending a preservation upscale with
the SeedVR2 output:

- `0.00`: preservation base only.
- `0.10-0.15`: mostly source-preserving with a small amount of SeedVR2 detail.
- `1.00`: full SeedVR2 output.

SeedVR2 advanced controls exposed in the UI include temporal batch/overlap,
input noise, latent noise, VAE encode/decode tiling, block swap, attention mode,
compile toggles, and seed.

## Audio

The final output defaults to **Keep source audio**.

Implementation detail: intermediate upscale and interpolation renders are silent
frame/video transforms. At the final save step, Kraken muxes the original
source audio stream back onto the finished upscaled video using ffmpeg.

Turn **Keep source audio** off for silent exports.

## Output Files

Imported source clips are copied to:

```text
outputs/YYYY-MM-DD/imports/
```

Upscale results are saved as:

```text
outputs/YYYY-MM-DD/<time>-<job>-vup.mp4
```

The result preview can also open the finished file in the system player or VLC
when a local path is available.

## Current Test Baseline

On the RTX 3090 workstation:

- 1080p60 SeedVR2 test output completed successfully but was slower and more
  generative.
- 1080p60 ESRGAN/RealESRGAN_x2 output was preferred visually by the user.
- 2K ESRGAN/RealESRGAN_x2 runs at about `1 fps` during the frame upscale pass
  on a 20 second clip, before RIFE/encode overhead.

## Known Follow-Ups

- Add an **Analyze** button that samples frames and recommends a preset based on
  motion, faces, text/logos, compression, blur, and source type.
- Add a short probe-render workflow for testing a 2-3 second segment before a
  full 2K/4K run.
