"""WAN 2.2 video pipeline. Component assembly from local files.

WAN 2.2 A14B is a **dual-expert** diffusion transformer:
  - `transformer`   — the high-noise expert, used while t >= boundary.
  - `transformer_2` — the low-noise expert, used while t <  boundary.
The pipeline switches between them once per gen at `boundary_ratio * 1000`
(=900 here). Each expert is a 14B WanTransformer3DModel; in ComfyUI scaled-FP8
form each is ~14 GB. Both cannot co-reside on a 24 GB card, so we let diffusers'
`enable_model_cpu_offload` swap whichever expert is active onto the GPU — this is
the correct mechanism for a one-time expert switch (unlike FLUX, where a single
mostly-resident transformer favours per-Linear streaming). ComfyUI swaps the two
experts the same way.

Weights are loaded the same way as FLUX: ComfyUI scaled-FP8 layers carry
`.scale_weight` tensors; the true weight is `fp8.to(bf16) * scale`. We extract
those scales, run diffusers' pure-rename `convert_wan_transformer_to_diffusers`
(which preserves FP8 dtype) on both the weights AND the scale dict, then attach
the scales as `_kraken_weight_scale` buffers and swap every Linear to a
StreamingLinear so the scaled-FP8 dequant happens correctly in the forward.

Text conditioning uses the UMT5-XXL encoder, loaded out-of-band and freed before
the transformers load (encode-then-free, like Z-Image). The A14B i2v transformer
has `image_dim=None`, so there is NO CLIP-vision image encoder — image
conditioning is purely the VAE-encoded first frame concatenated into the latent.

Architecture metadata (configs + tokenizer) lives under
`model_configs/wan22_i2v/`; only the heavy weights come from the user's local
model folders. Output frames are muxed to H.264 MP4 via the system ffmpeg
(already a documented dependency for the MP3 export path).
"""
from __future__ import annotations

import gc
import json
import logging
import shutil
import subprocess
import time
from base64 import b64decode
from pathlib import Path
from typing import Any

import safetensors.torch as sft
import torch
import torch.nn as nn
from PIL import Image

from config import MODELS_ROOT, OUTPUTS_ROOT
from pipelines.load_utils import (
    ensure_module_dtype,
    file_size_gb,
    materialize_meta_tensors,
    unload_pipeline,
)

log = logging.getLogger("kraken.wan_video")

_CFG_ROOT = Path(__file__).resolve().parent.parent / "model_configs" / "wan22_i2v"
_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)

_pipeline: Any | None = None
_pipeline_key: tuple | None = None
_loaded_loras: list[str] = []


# ---------- utilities ----------

def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _dtype() -> torch.dtype:
    return torch.bfloat16 if torch.cuda.is_available() else torch.float32


def _resolve(category: str, name: str | None) -> Path | None:
    if not name:
        return None
    base = MODELS_ROOT / category
    p = base / name
    if p.exists():
        return p
    for f in base.rglob(name):
        return f
    return None


def _free_vram_gb() -> float | None:
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        pynvml.nvmlShutdown()
        return mem.free / 1024**3
    except Exception:
        return None


def unload() -> dict:
    global _pipeline, _pipeline_key, _loaded_loras
    had = _pipeline is not None
    n_loras = len(_loaded_loras)
    pipe = _pipeline
    _pipeline = None
    _pipeline_key = None
    _loaded_loras = []
    if pipe is not None:
        unload_pipeline(pipe)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"had_pipeline": had, "loras_freed": n_loras}


# ---------- input image decoding ----------

def _load_input_image(spec: str | None) -> Image.Image:
    """Resolve the i2v conditioning image from either a base64 data URL, a raw
    base64 string, or a filesystem path (absolute or relative to OUTPUTS_ROOT)."""
    if not spec:
        raise ValueError("WAN i2v requires an input image (the first frame).")

    if spec.startswith("data:"):
        spec = spec.split(",", 1)[1]

    # Heuristic: a real path won't be valid base64 of meaningful length.
    looks_like_path = len(spec) < 1024 and ("/" in spec or "\\" in spec or spec.lower().endswith(
        (".png", ".jpg", ".jpeg", ".webp", ".bmp")
    ))
    if looks_like_path:
        p = Path(spec)
        if not p.is_absolute():
            cand = OUTPUTS_ROOT / spec
            if cand.exists():
                p = cand
        if p.exists():
            return Image.open(p).convert("RGB")

    try:
        raw = b64decode(spec)
        import io
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise ValueError(f"could not decode input image: {e}") from e


# ---------- text encoding (UMT5-XXL, encode-then-free) ----------

def _load_umt5_encoder(te_path: Path, dtype: torch.dtype):
    """Build UMT5EncoderModel on meta and stream weights, applying ComfyUI
    scaled-FP8 dequant inline (weight = fp8.to(bf16) * scale) so the encoder is
    correct regardless of whether the file is bf16, plain FP8, or scaled FP8."""
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from safetensors import safe_open
    from transformers import UMT5Config, UMT5EncoderModel

    cfg = json.loads((_CFG_ROOT / "text_encoder" / "config.json").read_text())
    cfg.pop("architectures", None)
    umt5_cfg = UMT5Config(**cfg)

    t0 = time.time()
    log.info("loading WAN UMT5-XXL text encoder from %s (streaming)", te_path)
    with init_empty_weights():
        model = UMT5EncoderModel(umt5_cfg)

    with safe_open(str(te_path), framework="pt", device="cpu") as f:
        keys = list(f.keys())
        scale_layers: dict[str, torch.Tensor] = {}
        for k in keys:
            if k.endswith(".scale_weight"):
                scale_layers[k[: -len(".scale_weight")]] = f.get_tensor(k).to(torch.bfloat16)

        n_fp8 = 0
        placed = 0
        unmatched: list[str] = []
        for i, key in enumerate(keys):
            if key == "scaled_fp8" or key.endswith((".scale_weight", ".scaled_fp8")):
                continue
            t = f.get_tensor(key)
            if t.dtype in _FP8_DTYPES:
                n_fp8 += 1
                t = t.to(torch.bfloat16)
                if key.endswith(".weight"):
                    layer = key[: -len(".weight")]
                    scale = scale_layers.get(layer)
                    if scale is not None:
                        t = t * scale
            try:
                value = t.to(dtype) if t.is_floating_point() else t
                set_module_tensor_to_device(
                    model, key, "cpu", value=value,
                    dtype=dtype if value.is_floating_point() else None,
                )
                placed += 1
            except (ValueError, KeyError, AttributeError):
                unmatched.append(key)
            del t
            if (i + 1) % 32 == 0:
                gc.collect()

    materialize_meta_tensors(model, dtype, "UMT5-XXL TE")
    ensure_module_dtype(model, dtype, "UMT5-XXL TE")
    if n_fp8:
        log.info("UMT5-XXL TE: dequantized %d FP8 tensors (%d scaled)", n_fp8, len(scale_layers))
    if unmatched:
        log.warning("UMT5-XXL TE: %d keys unmatched (first: %s)", len(unmatched), unmatched[:3])
    log.info("UMT5-XXL TE: placed %d tensors in %.1fs", placed, time.time() - t0)
    return model


def _encode_wan_prompt(
    te_path: Path,
    prompts: list[str],
    *,
    dtype: torch.dtype,
    max_sequence_length: int = 512,
) -> torch.Tensor:
    """Encode prompts with UMT5-XXL, mirroring WanPipeline._get_t5_prompt_embeds.
    The encoder is loaded, used on GPU, and freed entirely within this call.
    Returns a [batch, max_sequence_length, 4096] CPU tensor."""
    from transformers import AutoTokenizer

    from diffusers.pipelines.wan.pipeline_wan import prompt_clean

    text_encoder = _load_umt5_encoder(te_path, dtype)
    tokenizer = AutoTokenizer.from_pretrained(str(_CFG_ROOT / "tokenizer"))

    device = torch.device(_device())
    text_encoder = text_encoder.to(device=device, dtype=dtype)

    cleaned = [prompt_clean(p) for p in prompts]
    text_inputs = tokenizer(
        cleaned,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    input_ids = text_inputs.input_ids.to(device)
    mask = text_inputs.attention_mask.to(device)
    seq_lens = mask.gt(0).sum(dim=1).long()

    with torch.inference_mode():
        embeds = text_encoder(input_ids, mask).last_hidden_state
    embeds = embeds.to(dtype=dtype)
    embeds = [u[:v] for u, v in zip(embeds, seq_lens)]
    embeds = torch.stack(
        [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in embeds],
        dim=0,
    ).cpu()

    del text_encoder, tokenizer, input_ids, mask
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    return embeds


# ---------- transformer expert loaders ----------

def _apply_weight_scales(module, weight_scales: dict[str, torch.Tensor]) -> None:
    """Attach ComfyUI scaled-FP8 per-Linear scales as `_kraken_weight_scale`
    buffers. Must run BEFORE swap_linears so StreamingLinear.from_linear copies
    them; StreamingLinear.forward applies the scale after the FP8 cast."""
    if not weight_scales:
        return
    attached = 0
    for weight_key, scale in weight_scales.items():
        if not weight_key.endswith(".weight"):
            continue
        module_name = weight_key[: -len(".weight")]
        try:
            target = module.get_submodule(module_name)
        except AttributeError:
            log.warning("scaled-FP8 target module not found: %s", module_name)
            continue
        if not isinstance(target, torch.nn.Linear):
            continue
        target.register_buffer("_kraken_weight_scale", scale.to(torch.bfloat16), persistent=False)
        attached += 1
    log.info("  attached scaled-FP8 scales to %d Linear modules", attached)


def _load_wan_expert(tx_path: Path, cfg: dict, label: str, dtype: torch.dtype):
    """Load one WanTransformer3DModel expert from a ComfyUI single-file
    checkpoint: extract scaled-FP8 scales, convert native keys to diffusers
    layout (FP8 preserved), apply scales, swap to StreamingLinear."""
    from diffusers import WanTransformer3DModel
    from diffusers.loaders.single_file_utils import convert_wan_transformer_to_diffusers

    from pipelines.flux import (
        _ensure_module_dtype_preserving_fp8,
        _extract_comfy_scaled_fp8_scales,
    )
    from pipelines.streaming_linear import swap_linears

    size_gb = file_size_gb(tx_path)
    log.info("loading WAN %s expert from %s (%.1f GB on disk)", label, tx_path, size_gb)
    t0 = time.time()

    raw = sft.load_file(str(tx_path))
    # Extract scaled-FP8 scales (keyed by native weight key) and strip the
    # scale/marker keys; weights are cast to FP8 storage dtype in place.
    source_scales = _extract_comfy_scaled_fp8_scales(raw)

    # The WAN converter is a pure 1:1 rename for i2v (no QKV fusion, no value
    # reshape outside motion_encoder models we don't have), so we can run it on
    # the scale dict to remap scale keys to diffusers naming verbatim.
    diffusers_scales: dict[str, torch.Tensor] = {}
    if source_scales:
        diffusers_scales = convert_wan_transformer_to_diffusers(dict(source_scales))
        log.info("  remapped %d scaled-FP8 scales to diffusers keys", len(diffusers_scales))

    tx_sd = convert_wan_transformer_to_diffusers(raw)
    del raw
    gc.collect()

    with torch.device("meta"):
        transformer = WanTransformer3DModel.from_config(cfg)
    missing, unexpected = transformer.load_state_dict(tx_sd, strict=False, assign=True)
    if missing:
        log.warning("%s missing %d keys (first: %s)", label, len(missing), missing[:3])
    if unexpected:
        log.warning("%s unexpected %d keys (first: %s)", label, len(unexpected), list(unexpected)[:3])
    del tx_sd
    materialize_meta_tensors(transformer, dtype, label)
    _ensure_module_dtype_preserving_fp8(transformer, dtype, label, preserve_fp8_linears_only=True)
    _apply_weight_scales(transformer, diffusers_scales)
    n_swapped = swap_linears(transformer)
    if n_swapped:
        log.info("  swapped %d Linear modules to StreamingLinear", n_swapped)
    # Diffusers keeps `patch_embedding` / `condition_embedder` / `norm` out of
    # FP8 layerwise casting (see WanTransformer3DModel._skip_layerwise_casting_patterns).
    # The condition embedder matters most: WanTimeTextImageEmbedding.forward casts
    # the timestep to `next(time_embedder.parameters()).dtype`, so if that Linear is
    # FP8 the sinusoidal timestep is cast to float8 and the matmul dies with
    # `"mul_cuda" not implemented for 'Float8_e4m3fn'`. Fold the scale and keep
    # these small layers in bf16, exactly as diffusers intends.
    n_deq = 0
    for attr in ("condition_embedder", "patch_embedding"):
        sub = getattr(transformer, attr, None)
        if sub is not None:
            n_deq += _dequantize_fp8_submodule(sub, dtype)
    if n_deq:
        log.info("  dequantized %d FP8 Linears in embedders to %s", n_deq, dtype)
    gc.collect()
    log.info("  %s loaded in %.1fs", label, time.time() - t0)
    return transformer


def _dequantize_fp8_submodule(module: nn.Module, dtype: torch.dtype) -> int:
    """Permanently fold scaled-FP8 weights in `module` to `dtype` (bf16).

    For each Linear under `module` with an FP8 weight and/or a `_kraken_weight_scale`
    buffer: weight := fp8.to(dtype) * scale, then drop the scale buffer so the
    StreamingLinear forward becomes a plain bf16 matmul and the module reports a
    bf16 parameter dtype. Used for the embedders diffusers expects in high precision."""
    n = 0
    for sub in module.modules():
        if not isinstance(sub, nn.Linear):
            continue
        w = sub.weight.data
        scale = sub._buffers.get("_kraken_weight_scale", None)
        if w.dtype not in _FP8_DTYPES and scale is None:
            continue
        if w.dtype in _FP8_DTYPES:
            w = w.to(dtype)
        if scale is not None:
            w = w * scale.to(device=w.device, dtype=dtype)
            del sub._buffers["_kraken_weight_scale"]
        sub.weight.data = w.to(dtype=dtype)
        n += 1
    return n


def _load_wan_vae(vae_path: Path, dtype: torch.dtype):
    from diffusers import AutoencoderKLWan
    from diffusers.loaders.single_file_utils import convert_wan_vae_to_diffusers

    log.info("loading WAN VAE from %s (native->diffusers convert)", vae_path)
    cfg = json.loads((_CFG_ROOT / "vae" / "config.json").read_text())
    cfg.pop("_class_name", None)
    cfg.pop("_diffusers_version", None)

    raw = sft.load_file(str(vae_path))
    vae_sd = convert_wan_vae_to_diffusers(raw)
    del raw
    gc.collect()

    with torch.device("meta"):
        vae = AutoencoderKLWan.from_config(cfg)
    missing, unexpected = vae.load_state_dict(vae_sd, strict=False, assign=True)
    if missing:
        log.warning("VAE missing %d keys (first: %s)", len(missing), missing[:3])
    if unexpected:
        log.warning("VAE unexpected %d keys (first: %s)", len(unexpected), list(unexpected)[:3])
    del vae_sd
    materialize_meta_tensors(vae, dtype, "VAE")
    vae = vae.to("cpu", dtype=dtype)
    ensure_module_dtype(vae, dtype, "VAE")
    gc.collect()
    return vae


# ---------- pipeline lifecycle ----------

def _ensure_pipeline(tx_high: str, tx_low: str, vae_name: str, mode: str):
    global _pipeline, _pipeline_key

    high_path = _resolve("diffusion_models", tx_high)
    low_path = _resolve("diffusion_models", tx_low)
    vae_path = _resolve("vae", vae_name)
    if not high_path:
        raise FileNotFoundError(f"high-noise expert not found: {tx_high}")
    if not low_path:
        raise FileNotFoundError(f"low-noise expert not found: {tx_low}")
    if not vae_path:
        raise ValueError("WAN i2v requires the WAN VAE — pick it from the VAE dropdown.")

    key = (mode, str(high_path), str(low_path), str(vae_path))
    if _pipeline is not None and _pipeline_key == key:
        return _pipeline

    unload()
    log.info("WAN %s pipeline load starting (2 experts + VAE)", mode)

    from diffusers import UniPCMultistepScheduler, WanImageToVideoPipeline, WanPipeline

    dtype = _dtype()
    tx_cfg = json.loads((_CFG_ROOT / "transformer" / "config.json").read_text())
    tx_cfg.pop("_class_name", None)
    tx_cfg.pop("_diffusers_version", None)
    if mode == "t2v":
        # Wan2.2 T2V-A14B uses pure video latents. I2V-A14B uses 36 channels
        # because it concatenates first-frame conditioning latents.
        tx_cfg["in_channels"] = 16

    transformer = _load_wan_expert(high_path, tx_cfg, "high-noise transformer", dtype)
    transformer_2 = _load_wan_expert(low_path, tx_cfg, "low-noise transformer", dtype)
    vae = _load_wan_vae(vae_path, dtype)

    sched_cfg = json.loads((_CFG_ROOT / "scheduler" / "scheduler_config.json").read_text())
    scheduler = UniPCMultistepScheduler.from_config(sched_cfg)

    model_index = json.loads((_CFG_ROOT / "model_index.json").read_text())
    boundary_ratio = 0.875 if mode == "t2v" else float(model_index.get("boundary_ratio", 0.9))

    # Text encoder + tokenizer are handled out-of-band, so the pipeline holds
    # none of them. prompt_embeds are passed at call time.
    if mode == "t2v":
        pipe = WanPipeline(
            transformer=transformer,
            transformer_2=transformer_2,
            vae=vae,
            scheduler=scheduler,
            text_encoder=None,
            tokenizer=None,
            boundary_ratio=boundary_ratio,
        )
    else:
        pipe = WanImageToVideoPipeline(
            transformer=transformer,
            transformer_2=transformer_2,
            vae=vae,
            scheduler=scheduler,
            text_encoder=None,
            tokenizer=None,
            image_encoder=None,
            image_processor=None,
            boundary_ratio=boundary_ratio,
        )
    pipe.set_progress_bar_config(disable=True)

    # Dual 14B experts won't both fit on 24 GB, so swap the active expert via
    # accelerate's model_cpu_offload (one-time boundary switch — cheap). On a
    # hypothetical card with room for both, place everything on GPU.
    free_vram = _free_vram_gb()
    both_experts_gb = sum(
        p.numel() * p.element_size() for p in transformer.parameters()
    ) / 1024**3 * 2
    if free_vram is not None and free_vram >= both_experts_gb + 6.0:
        pipe.to(_device())
        pipe._kraken_offload_strategy = "no_offload_full"  # type: ignore[attr-defined]
        log.info("WAN: full GPU placement (free=%.1f GB)", free_vram)
    else:
        pipe.enable_model_cpu_offload(device=_device())
        pipe._kraken_offload_strategy = "model_cpu_offload"  # type: ignore[attr-defined]
        log.info(
            "WAN: model_cpu_offload — active expert swaps onto GPU (free=%s GB)",
            f"{free_vram:.1f}" if free_vram is not None else "?",
        )

    _pipeline = pipe
    _pipeline_key = key
    return pipe


# ---------- LoRA (lightx2v 4-step etc.) ----------

def _adapter_name(lora_name: str) -> str:
    return "lora_" + "".join(ch if ch.isalnum() else "_" for ch in lora_name)[:48]


def _lora_model_weight(entry: dict) -> float:
    value = entry.get("model_weight")
    if value is None:
        value = entry.get("weight")
    if value is None:
        value = 1.0
    return float(value)


def _apply_loras(pipe, loras: list[dict]) -> None:
    global _loaded_loras
    try:
        if _loaded_loras:
            pipe.unload_lora_weights()
    except Exception as e:
        log.warning("unload_lora_weights raised: %s", e)
    _loaded_loras = []
    if not loras:
        return
    names, weights = [], []
    for entry in loras:
        path = _resolve("loras", entry["name"])
        if not path:
            log.warning("lora not found, skipping: %s", entry["name"])
            continue
        adapter = _adapter_name(entry["name"])
        try:
            pipe.load_lora_weights(str(path.parent), weight_name=path.name, adapter_name=adapter)
            names.append(adapter)
            weights.append(_lora_model_weight(entry))
            log.info("loaded LoRA %s @ %.2f", entry["name"], weights[-1])
        except Exception as e:
            log.warning("failed to load LoRA %s: %s — continuing without it", entry["name"], e)
    if names:
        try:
            pipe.set_adapters(names, adapter_weights=weights)
            _loaded_loras = names
        except Exception as e:
            log.warning("set_adapters failed: %s", e)
        # PEFT creates the injected LoRA tensors in the base Linear's dtype. On
        # our scaled-FP8 experts that means the adapters land as float8_e4m3fn,
        # and the `lora_B(lora_A(x)) * scaling` multiply then dies with
        # `"mul_cuda" not implemented for 'Float8_e4m3fn'`. LoRA math must run in
        # the compute dtype — cast every injected adapter tensor to bf16.
        _coerce_lora_dtype(pipe, _dtype())


def _coerce_lora_dtype(pipe, dtype: torch.dtype) -> int:
    """Cast every injected LoRA adapter param/buffer to `dtype` (bf16).

    The base scaled-FP8 weights are left untouched — StreamingLinear dequants
    those in forward. Only the `lora_A`/`lora_B`/`lora_*` tensors PEFT created
    are coerced, so the adapter math no longer touches FP8 dtypes."""
    coerced = 0
    for comp in (getattr(pipe, "transformer", None), getattr(pipe, "transformer_2", None)):
        if comp is None:
            continue
        for name, param in comp.named_parameters():
            if "lora_" in name and param.dtype in _FP8_DTYPES:
                param.data = param.data.to(dtype=dtype)
                coerced += 1
        for name, buf in comp.named_buffers():
            if "lora_" in name and buf.dtype in _FP8_DTYPES:
                buf.data = buf.data.to(dtype=dtype)
                coerced += 1
    if coerced:
        log.info("coerced %d FP8 LoRA tensors -> %s", coerced, dtype)
    return coerced


# ---------- progress + mp4 export ----------

def _make_callback(job, total_steps: int):
    def cb(pipe, step: int, timestep, callback_kwargs):
        if job.cancel.is_set():
            raise RuntimeError("cancelled")
        job.progress.step = step + 1
        job.progress.total_steps = total_steps
        job.progress.image_index = 0
        job.progress.total_images = 1
        job.emit({
            "type": "progress",
            "step": step + 1, "total_steps": total_steps,
            "image_index": 0, "total_images": 1,
        })
        return callback_kwargs
    return cb


_FFMPEG = shutil.which("ffmpeg")


def _encode_frames_to_mp4(frames: list[Image.Image], dest: Path, fps: int) -> None:
    """Mux RGB frames to an H.264 MP4 by piping rawvideo to the system ffmpeg.
    No temp PNGs, no extra Python deps."""
    if not _FFMPEG:
        raise RuntimeError(
            "ffmpeg not found on PATH. Install it (Windows: `winget install Gyan.FFmpeg`) "
            "to export video — the WAN i2v pipeline muxes frames with ffmpeg."
        )
    if not frames:
        raise RuntimeError("no frames produced by the WAN pipeline.")
    w, h = frames[0].size
    cmd = [
        _FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        "-movflags", "+faststart",
        str(dest),
    ]
    log.info("ffmpeg mux: %d frames @ %dx%d %dfps -> %s", len(frames), w, h, fps, dest.name)
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for frame in frames:
            proc.stdin.write(frame.convert("RGB").tobytes())
        proc.stdin.close()
        err = proc.stderr.read().decode("utf-8", "replace")
        rc = proc.wait()
    finally:
        if proc.poll() is None:
            proc.kill()
    if rc != 0:
        raise RuntimeError(f"ffmpeg returned {rc}: {err.strip()[:600]}")
    if not dest.exists() or dest.stat().st_size == 0:
        raise RuntimeError(f"ffmpeg ran but {dest} is missing or empty.")


def _thumbnail_b64(img: Image.Image, max_side: int = 320) -> str:
    import io
    from base64 import b64encode
    thumb = img.copy()
    thumb.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    thumb.convert("RGB").save(buf, format="JPEG", quality=80)
    return b64encode(buf.getvalue()).decode("ascii")


# ---------- run ----------

def run(job) -> dict:
    # Free image-arch pipelines so we don't double-book VRAM.
    from pipelines import flux as flux_mod, sdxl as sdxl_mod, z_image as zi_mod
    sdxl_mod.unload()
    flux_mod.unload()
    zi_mod.unload()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    p = job.params
    dtype = _dtype()

    mode = str(p.get("mode") or "i2v").lower()
    if mode not in ("i2v", "t2v"):
        raise ValueError("WAN video mode must be 'i2v' or 't2v'.")

    tx_high = p.get("diffusion_model")
    tx_low = p.get("diffusion_model_2")
    if not tx_high or not tx_low:
        raise ValueError("WAN needs both high-noise (diffusion_model) and low-noise (diffusion_model_2) experts.")

    te = p.get("text_encoders") or []
    te_name = te[0] if te else None
    te_path = _resolve("text_encoders", te_name) if te_name else None
    if not te_path:
        raise ValueError("WAN i2v requires the UMT5-XXL text encoder in slot 1.")

    image = _load_input_image(p.get("input_image")) if mode == "i2v" else None

    width = int(p.get("width") or 832)
    height = int(p.get("height") or 480)
    num_frames = int(p.get("num_frames") or 81)
    fps = int(p.get("fps") or 16)
    steps = int(p.get("steps") or 6)
    guidance = float(p.get("cfg") or 1.0)
    seed = p.get("seed")

    # WAN: (num_frames - 1) must be divisible by the temporal scale factor (4).
    if (num_frames - 1) % 4 != 0:
        num_frames = ((num_frames - 1) // 4) * 4 + 1

    # 1) Encode prompt(s) with UMT5, then free the encoder.
    _t_enc = time.time()
    prompt_embeds_cpu = _encode_wan_prompt(te_path, [p.get("prompt", "")], dtype=dtype)
    negative_embeds_cpu: torch.Tensor | None = None
    if guidance > 1:
        negative_embeds_cpu = _encode_wan_prompt(te_path, [p.get("negative", "") or ""], dtype=dtype)
    _enc_s = time.time() - _t_enc

    # 2) Build the dual-expert pipeline + VAE.
    _t_setup = time.time()
    pipe = _ensure_pipeline(tx_high, tx_low, p.get("vae"), mode)
    _apply_loras(pipe, p.get("loras") or [])
    log.info(
        "WAN run stages: text-encode=%.1fs · pipeline-ensure+loras=%.1fs (sampling follows)",
        _enc_s, time.time() - _t_setup,
    )

    exec_device = pipe._execution_device
    prompt_embeds = prompt_embeds_cpu.to(device=exec_device)
    negative_prompt_embeds = (
        negative_embeds_cpu.to(device=exec_device) if negative_embeds_cpu is not None else None
    )
    del prompt_embeds_cpu, negative_embeds_cpu

    out_dir = OUTPUTS_ROOT / time.strftime("%Y-%m-%d")
    out_dir.mkdir(parents=True, exist_ok=True)
    base_stem = time.strftime("%H%M%S") + "-" + job.id[:8]

    per_seed = int(seed if seed is not None else torch.seed()) & 0x7FFFFFFF
    generator = torch.Generator(device=exec_device.type).manual_seed(per_seed)

    call_kwargs = {
        "prompt": None,
        "prompt_embeds": prompt_embeds,
        "negative_prompt_embeds": negative_prompt_embeds,
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "num_inference_steps": steps,
        "guidance_scale": guidance,
        "generator": generator,
        "output_type": "pil",
        "callback_on_step_end": _make_callback(job, steps),
    }
    if mode == "i2v":
        call_kwargs["image"] = image
    result = pipe(**call_kwargs)
    frames = result.frames[0]

    if job.cancel.is_set():
        return {"kind": "video", "count_requested": 1, "count_produced": 0, "outputs": [], "output_dir": str(out_dir)}

    fname = f"{base_stem}.mp4"
    fpath = out_dir / fname
    _encode_frames_to_mp4(frames, fpath, fps)
    rel_path = fpath.relative_to(OUTPUTS_ROOT).as_posix()

    entry = {
        "path": str(fpath), "filename": fname, "rel_path": rel_path,
        "seed": per_seed, "width": width, "height": height, "mode": mode,
        "num_frames": len(frames), "fps": fps,
    }
    job.emit({
        "type": "video",
        "image_index": 0, "path": str(fpath), "rel_path": rel_path, "filename": fname,
        "seed": per_seed, "preview_b64": _thumbnail_b64(frames[0]),
    })

    return {
        "kind": "video",
        "count_requested": 1,
        "count_produced": 1,
        "outputs": [entry],
        "output_dir": str(out_dir),
    }
