"""Ultimate-SD-Upscale (USDU) engine — Kraken Art.

Faithful port of the user's ComfyUI workflow, which is three stages:

  1. tile_calc()        — port of kraken_upscale_tile_calc.KrakenUpscaleTileCalc.
                          Target-resolution-driven: given a source size and a
                          wanted output size, compute the cover upscale factor
                          (+ a min-upscale policy so USDU always does real work)
                          and the tile/overlap/seam geometry. Deliberately
                          OVERSHOOTS the target.
  2. run_usdu()         — the USDU sampler: ESRGAN pre-upscale to the predicted
                          canvas, then tile + img2img each tile through a refine
                          callback at (steps, denoise), feather-blending tiles so
                          seams disappear. (The model-specific img2img lives in
                          the arch modules and is passed in as `refine_fn`.)
  3. resolution_snap()  — port of kraken_resolution_helper.KrakenResolutionHelper.
                          Snaps the overshot result to EXACTLY the target dims
                          (default "fill / crop": cover-scale + centre-crop).

`refine_fn` decouples USDU orchestration from the model: each arch (SDXL, FLUX)
supplies `refine_fn(tile: Image, *, steps, denoise, seed) -> Image`. When no
refine_fn is given the engine runs ESRGAN-only (pure upscaler, no diffusion).
"""
from __future__ import annotations

import logging
import math
from typing import Callable, Optional

from PIL import Image, ImageFilter

log = logging.getLogger("kraken.usdu")

RefineFn = Callable[..., Image.Image]


def job_progress(job, image_index: int = 0, total_images: int = 1):
    """Return a run_usdu `progress` callback that drives the UI bar + WS for a job.

    Emits the same {"type":"progress"} shape generation uses, so the Generate/
    Upscale progress bar moves per tile. Raises to abort on cancel.
    """
    def cb(done: int, total: int, message: str) -> None:
        if job.cancel.is_set():
            raise RuntimeError("cancelled")
        job.progress.step = done
        job.progress.total_steps = max(1, total)
        job.progress.message = message
        job.emit({
            "type": "progress",
            "phase": "upscale",
            "step": done,
            "total_steps": max(1, total),
            "image_index": image_index,
            "total_images": total_images,
            "message": message,
        })
    return cb


# --------------------------------------------------------------------------
# small math helpers (mirror the ComfyUI nodes exactly)
# --------------------------------------------------------------------------

def _round_to(x: float, base: int) -> int:
    return int(round(x / base) * base)


def _ceil_to(x: float, base: int) -> int:
    return int(math.ceil(x / base) * base)


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


# --------------------------------------------------------------------------
# Stage 1 — tile calc  (port of KrakenUpscaleTileCalc.calculate)
# --------------------------------------------------------------------------

def _choose_min_upscale(wi: int, hi: int, Wt: int, Ht: int, policy: str, custom: float) -> float:
    policy = (policy or "auto").lower()
    if policy == "off":
        return 1.0
    if policy == "1.5x":
        return 1.5
    if policy == "2.0x":
        return 2.0
    if policy == "custom":
        try:
            return max(1.0, min(4.0, float(custom)))
        except Exception:
            return 1.5
    # auto
    src_mp = (wi * hi) / 1_000_000.0
    tgt_mp = (Wt * Ht) / 1_000_000.0
    if max(Wt, Ht) <= 1024 or tgt_mp <= 1.0:
        return 2.0
    if src_mp <= 2.0:
        return 2.0
    return 1.5


def tile_calc(
    wi: int,
    hi: int,
    Wt: int,
    Ht: int,
    *,
    scale_step: float = 0.01,
    crop_margin_px: int = 0,
    min_overshoot_px: int = 2,
    mask_blur: int = 16,
    tile_padding: int = 16,
    seam_fix_width: int = 64,
    seam_fix_mask_blur: int = 16,
    seam_fix_padding: int = 32,
    min_upscale_policy: str = "auto",
    min_upscale_custom: float = 1.5,
) -> dict:
    """Compute upscale factor + tile geometry for a source (wi×hi) -> target (Wt×Ht).

    Returns a dict with: upscale_by, pred_w, pred_h, tile_w, tile_h, mask_blur,
    tile_padding, seam_fix_*  — matching the ComfyUI node's outputs.
    """
    margin = _clamp(int(crop_margin_px), 0, max(Wt, Ht) // 10)
    min_over = max(0, int(min_overshoot_px))

    want_w = Wt + 2 * margin + 2 * min_over
    want_h = Ht + 2 * margin + 2 * min_over
    required = max(want_w / wi, want_h / hi)

    s = max(1.0, round(math.ceil(required / scale_step) * scale_step, 3))
    min_s = _choose_min_upscale(wi, hi, Wt, Ht, min_upscale_policy, min_upscale_custom)
    if min_s > 1.0:
        s = max(s, round(min_s / scale_step) * scale_step)
    s = round(s, 3)

    pred_align = 8
    pred_w = _ceil_to(wi * s, pred_align)
    pred_h = _ceil_to(hi * s, pred_align)

    guard = 0
    while (pred_w < Wt + min_over or pred_h < Ht + min_over) and guard < 6:
        s = round(s + scale_step, 3)
        pred_w = _ceil_to(wi * s, pred_align)
        pred_h = _ceil_to(hi * s, pred_align)
        guard += 1

    overlap = _clamp(int(round(min(pred_w, pred_h) * 0.015)), 32, 96)
    tile_w = _round_to(_clamp(pred_w // 2 + overlap, 512, 2048), 64)
    tile_h = _round_to(_clamp(pred_h // 2 + overlap, 512, 2048), 64)

    return {
        "upscale_by": s,
        "pred_w": pred_w,
        "pred_h": pred_h,
        "tile_w": tile_w,
        "tile_h": tile_h,
        "overlap": overlap,
        "mask_blur": int(mask_blur),
        "tile_padding": int(tile_padding),
        "seam_fix_width": int(seam_fix_width),
        "seam_fix_mask_blur": int(seam_fix_mask_blur),
        "seam_fix_padding": int(seam_fix_padding),
        "target_w": Wt,
        "target_h": Ht,
    }


# --------------------------------------------------------------------------
# Stage 3 — exact resolution snap  (port of KrakenResolutionHelper._process_frame)
# --------------------------------------------------------------------------

_INTERP = {
    "lanczos": Image.LANCZOS,
    "bicubic": Image.BICUBIC,
    "bilinear": Image.BILINEAR,
    "nearest": Image.NEAREST,
}


def _ceil_to_multiple(x: int, m: int) -> int:
    if m <= 1:
        return x
    return int(math.ceil(x / m) * m)


def _anchor_offsets(big_w, big_h, small_w, small_h, ax, ay):
    off_x = 0 if ax == "left" else (big_w - small_w if ax == "right" else (big_w - small_w) // 2)
    off_y = 0 if ay == "top" else (big_h - small_h if ay == "bottom" else (big_h - small_h) // 2)
    return max(0, off_x), max(0, off_y)


def resolution_snap(
    pil: Image.Image,
    Wt: int,
    Ht: int,
    *,
    mode: str = "fill / crop",
    interpolation: str = "lanczos",
    allow_upscale: bool = True,
    anchor_x: str = "center",
    anchor_y: str = "center",
    crop_margin_px: int = 0,
    min_overshoot_px: int = 2,
    scale_step: float = 0.01,
    multiple_of: int = 0,
) -> Image.Image:
    """Snap `pil` to exactly Wt×Ht. Default 'fill / crop' = cover-scale + crop."""
    pil = pil.convert("RGB")
    Wi, Hi = pil.size
    interp = _INTERP.get(interpolation, Image.LANCZOS)

    if mode == "stretch":
        return pil.resize((Wt, Ht), interp)

    if mode in ("keep proportion", "pad"):
        s = min(Wt / Wi, Ht / Hi)  # contain
        if s > 1.0 and not allow_upscale:
            s = 1.0
        new_w = max(1, int(round(Wi * s)))
        new_h = max(1, int(round(Hi * s)))
        if multiple_of > 1:
            new_w = _ceil_to_multiple(new_w, multiple_of)
            new_h = _ceil_to_multiple(new_h, multiple_of)
        if not allow_upscale:
            new_w = min(new_w, Wt)
            new_h = min(new_h, Ht)
        scaled = pil.resize((new_w, new_h), interp)
        canvas = Image.new("RGB", (Wt, Ht), (0, 0, 0))
        ax, ay = (anchor_x, anchor_y) if mode == "pad" else ("center", "center")
        off_x, off_y = _anchor_offsets(Wt, Ht, new_w, new_h, ax, ay)
        canvas.paste(scaled, (off_x, off_y))
        return canvas

    # "fill / crop"
    margin = max(0, int(crop_margin_px))
    min_over = max(0, int(min_overshoot_px))
    want_w = Wt + 2 * margin + 2 * min_over
    want_h = Ht + 2 * margin + 2 * min_over
    s_req = max(want_w / Wi, want_h / Hi)  # cover
    s = math.ceil(s_req / max(0.001, scale_step)) * max(0.001, scale_step)

    if s > 1.0 and not allow_upscale:
        s = 1.0
        new_w = max(1, int(round(Wi * s)))
        new_h = max(1, int(round(Hi * s)))
        if multiple_of > 1:
            new_w = _ceil_to_multiple(new_w, multiple_of)
            new_h = _ceil_to_multiple(new_h, multiple_of)
        scaled = pil.resize((new_w, new_h), interp)
        canvas = Image.new("RGB", (Wt, Ht), (0, 0, 0))
        off_x, off_y = _anchor_offsets(Wt, Ht, new_w, new_h, "center", "center")
        canvas.paste(scaled, (off_x, off_y))
        return canvas

    pred_w = int(math.ceil(Wi * s))
    pred_h = int(math.ceil(Hi * s))
    if multiple_of > 1:
        pred_w = _ceil_to_multiple(pred_w, multiple_of)
        pred_h = _ceil_to_multiple(pred_h, multiple_of)

    guard = 0
    while (pred_w < Wt + min_over or pred_h < Ht + min_over) and guard < 6:
        s = round(s + scale_step, 6)
        pred_w = int(math.ceil(Wi * s))
        pred_h = int(math.ceil(Hi * s))
        if multiple_of > 1:
            pred_w = _ceil_to_multiple(pred_w, multiple_of)
            pred_h = _ceil_to_multiple(pred_h, multiple_of)
        guard += 1

    scaled = pil.resize((pred_w, pred_h), interp)
    left, top = _anchor_offsets(pred_w, pred_h, Wt, Ht, anchor_x, anchor_y)
    return scaled.crop((left, top, left + Wt, top + Ht))


# --------------------------------------------------------------------------
# Stage 2 — the USDU sampler
# --------------------------------------------------------------------------

def _pre_upscale(image: Image.Image, pred_w: int, pred_h: int, upscale_model: Optional[str]) -> Image.Image:
    """ESRGAN pre-upscale to the predicted canvas. Falls back to lanczos."""
    if upscale_model:
        try:
            from pipelines import upscale_esrgan
            factor = max(pred_w / image.width, pred_h / image.height)
            big = upscale_esrgan.upscale(image, upscale_model, factor)
            if big.size != (pred_w, pred_h):
                big = big.resize((pred_w, pred_h), Image.LANCZOS)
            return big
        except Exception as e:
            log.warning("ESRGAN pre-upscale failed (%s); using lanczos", e)
    return image.convert("RGB").resize((pred_w, pred_h), Image.LANCZOS)


def _snap16(v: int) -> int:
    # 16 satisfies both SDXL (/8) and FLUX latent packing (/16).
    return max(16, int(round(v / 16) * 16))


def _tile_grid(pred_w: int, pred_h: int, tile_w: int, tile_h: int):
    """Non-overlapping cell grid covering the canvas (USDU 'Linear')."""
    cols = max(1, math.ceil(pred_w / tile_w))
    rows = max(1, math.ceil(pred_h / tile_h))
    cells = []
    for ry in range(rows):
        for cx in range(cols):
            x0 = cx * tile_w
            y0 = ry * tile_h
            x1 = min(pred_w, x0 + tile_w)
            y1 = min(pred_h, y0 + tile_h)
            cells.append((x0, y0, x1, y1))
    return cells, cols, rows


def _feather_mask(region_w: int, region_h: int, cell_box, mask_blur: int) -> Image.Image:
    """White over the cell rectangle (in region-local coords), feathered by mask_blur."""
    m = Image.new("L", (region_w, region_h), 0)
    from PIL import ImageDraw
    d = ImageDraw.Draw(m)
    cx0, cy0, cx1, cy1 = cell_box
    d.rectangle([cx0, cy0, cx1 - 1, cy1 - 1], fill=255)
    if mask_blur > 0:
        m = m.filter(ImageFilter.GaussianBlur(mask_blur))
    return m


def run_usdu(
    image: Image.Image,
    target_w: int,
    target_h: int,
    *,
    upscale_model: Optional[str] = None,
    refine_fn: Optional[RefineFn] = None,
    steps: int = 20,
    denoise: float = 0.2,
    seed: int = 0,
    tile_size: Optional[int] = None,
    progress: Optional[Callable[[int, int, str], None]] = None,
    snap_mode: str = "fill / crop",
    anchor_x: str = "center",
    anchor_y: str = "center",
    **calc_opts,
) -> dict:
    """Full USDU pass. Returns {'image': PIL, 'pred_w','pred_h','tile_w','tile_h','tiles'}.

    `progress(done, total, message)` is called as tiles complete so the caller can
    forward live progress to the job WebSocket.
    """
    image = image.convert("RGB")
    wi, hi = image.size

    tc = tile_calc(wi, hi, target_w, target_h, **calc_opts)
    pred_w, pred_h = tc["pred_w"], tc["pred_h"]
    tile_w = int(tile_size) if tile_size else tc["tile_w"]
    tile_h = int(tile_size) if tile_size else tc["tile_h"]
    mask_blur = tc["mask_blur"]
    padding = tc["tile_padding"]

    log.info(
        "USDU %dx%d -> target %dx%d : upscale_by=%.3f pred=%dx%d tile=%dx%d",
        wi, hi, target_w, target_h, tc["upscale_by"], pred_w, pred_h, tile_w, tile_h,
    )

    # Stage 2a: ESRGAN pre-upscale to predicted canvas
    if progress:
        progress(0, 1, "pre-upscale")
    canvas = _pre_upscale(image, pred_w, pred_h, upscale_model)

    # ESRGAN-only mode (no diffusion refine): snap and return now.
    if refine_fn is None:
        final = resolution_snap(
            canvas, target_w, target_h,
            mode=snap_mode, anchor_x=anchor_x, anchor_y=anchor_y,
        )
        return {"image": final, "pred_w": pred_w, "pred_h": pred_h,
                "tile_w": tile_w, "tile_h": tile_h, "tiles": 0}

    # Stage 2b: tile + img2img refine, feather-blend back
    cells, cols, rows = _tile_grid(pred_w, pred_h, tile_w, tile_h)
    total = len(cells)
    log.info("USDU tiling: %d cells (%dx%d grid), padding=%d mask_blur=%d denoise=%.2f steps=%d",
             total, cols, rows, padding, mask_blur, denoise, steps)

    for i, (x0, y0, x1, y1) in enumerate(cells):
        # Expand cell by padding (clamped to canvas) for context.
        rx0 = max(0, x0 - padding)
        ry0 = max(0, y0 - padding)
        rx1 = min(pred_w, x1 + padding)
        ry1 = min(pred_h, y1 + padding)

        crop = canvas.crop((rx0, ry0, rx1, ry1))
        # img2img needs dims that are multiples of 16 (SDXL /8, FLUX /16).
        cw, ch = crop.size
        cw8, ch8 = _snap16(cw), _snap16(ch)
        work = crop.resize((cw8, ch8), Image.LANCZOS) if (cw8, ch8) != (cw, ch) else crop

        try:
            refined = refine_fn(work, steps=steps, denoise=denoise, seed=seed + i)
        except Exception as e:
            log.warning("tile %d/%d refine failed (%s); keeping pre-upscaled", i + 1, total, e)
            refined = work
        if refined.size != (cw, ch):
            refined = refined.resize((cw, ch), Image.LANCZOS)

        # Feathered paste: mask is the cell rect within the region, blurred.
        cell_local = (x0 - rx0, y0 - ry0, x1 - rx0, y1 - ry0)
        mask = _feather_mask(cw, ch, cell_local, mask_blur)
        canvas.paste(refined, (rx0, ry0), mask)

        log.info("USDU tile %d/%d refined (%dx%d)", i + 1, total, cw, ch)
        if progress:
            progress(i + 1, total, f"tile {i + 1}/{total}")

    # Stage 3: exact resolution snap
    final = resolution_snap(
        canvas, target_w, target_h,
        mode=snap_mode, anchor_x=anchor_x, anchor_y=anchor_y,
    )
    return {"image": final, "pred_w": pred_w, "pred_h": pred_h,
            "tile_w": tile_w, "tile_h": tile_h, "tiles": total}
