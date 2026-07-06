"""Krea 2 pipeline (Krea.ai's 12B flow-matching MMDiT text-to-image model).

Krea 2 ships as a self-contained *diffusers folder* (not a single-file
checkpoint), so unlike FLUX / Z-Image we load it folder-native rather than
converting single-file safetensors. The custom `Krea2Pipeline` /
`Krea2Transformer2DModel` classes are not in the released diffusers 0.38.0 that
Kraken pins, so they are vendored in `kraken_krea2_pipeline.py` /
`kraken_krea2_model.py` (see those files for why we vendor instead of upgrading
the whole library).

Components (all under one model folder, e.g. `krea2/Krea-2-Raw/`):
  - transformer: `Krea2Transformer2DModel`, 12.8B params (~24 GB bf16), 28 blocks
  - text encoder: `Qwen3VLModel` (Qwen3-VL). The transformer consumes a *stack*
    of hidden states tapped from 12 selected decoder layers
    (`text_encoder_select_layers` in model_index.json), not the last hidden state.
  - VAE: `AutoencoderKLQwenImage` (16-channel, f8), same family as Qwen-Image.
  - scheduler: `FlowMatchEulerDiscreteScheduler` (resolution-aware exp shift).

VRAM strategy (24 GB target): the transformer alone is ~24 GB, so it cannot be
GPU-resident. We:
  1. Encode the prompt(s) with the Qwen3-VL encoder, then FREE the encoder
     (peak ~9 GB) — exactly like z_image.
  2. Load the transformer to CPU, `swap_linears` + `apply_streaming` it (the same
     Forge/Fooocus-style per-Linear streaming FLUX uses) so most blocks stream
     from pinned CPU memory and only what fits stays on GPU.
  3. Load the small VAE to GPU and sample.

Checkpoints:
  - Krea 2 Raw  (`is_distilled=False`): base/midtrain. steps 28-52, cfg 3.5-4.5.
  - Krea 2 Turbo(`is_distilled=True`):  few-step distilled. steps 8, cfg 0.0,
    fixed timestep shift mu=1.15.
"""
from __future__ import annotations
import gc
import io
import json
import logging
import os
import time
from base64 import b64encode
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from config import ROOT, MODELS_ROOT, OUTPUTS_ROOT
from pipelines.load_utils import file_size_gb, unload_pipeline
from pipelines.output_metadata import build_output_stem, save_png_with_metadata

log = logging.getLogger("kraken.krea2")

# Folders scanned for Krea 2 diffusers model repos. The user cloned them under
# `<root>/krea2/` (where this work started); we also look under the normal
# diffusion_models category so a future move/symlink there is picked up too.
_SCAN_ROOTS = [ROOT / "krea2", MODELS_ROOT / "diffusion_models", MODELS_ROOT / "krea2"]

_pipeline: Any | None = None
_pipeline_key: tuple | None = None
_loaded_loras: list[tuple[str, float]] = []


# ---------- utilities ----------

def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _dtype() -> torch.dtype:
    return torch.bfloat16 if torch.cuda.is_available() else torch.float32


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


def _is_krea2_repo(folder: Path) -> dict | None:
    """Return the parsed model_index.json if `folder` is a Krea 2 diffusers repo."""
    mi = folder / "model_index.json"
    if not mi.is_file():
        return None
    try:
        data = json.loads(mi.read_text(encoding="utf-8"))
    except Exception:
        return None
    if data.get("_class_name") == "Krea2Pipeline":
        return data
    return None


def discover_models() -> list[dict]:
    """Find Krea 2 model folders and return model-listing entries (one per repo).

    Each entry mirrors the shape produced by api/models.py's file scanner so the
    UI and request flow treat Krea 2 like any other diffusion model. The model is
    selected by `filename`; the pipeline resolves that back to the folder.
    """
    out: list[dict] = []
    seen: set[str] = set()
    for base in _SCAN_ROOTS:
        if not base.exists():
            continue
        # (a) diffusers-folder models (model_index.json with _class_name Krea2Pipeline)
        for folder in sorted(base.iterdir()):
            if not folder.is_dir():
                continue
            mi = _is_krea2_repo(folder)
            if mi is None:
                continue
            resolved = str(folder.resolve())
            if resolved in seen:
                continue
            seen.add(resolved)
            distilled = bool(mi.get("is_distilled"))
            try:
                size = sum(p.stat().st_size for p in folder.rglob("*") if p.is_file())
            except OSError:
                size = 0
            label = "Turbo (distilled)" if distilled else "Raw (base)"
            out.append({
                "name": f"Krea 2 {label} — {folder.name}",
                "filename": folder.name,
                "subdir": "krea2",
                "abs_path": resolved,
                "size_bytes": size,
                "ext": "hf",
                "base_model": "Krea 2",
                "source_type": "krea2_diffusers_folder",
                "detected_arch": "krea2",
                "is_distilled": distilled,
                "recommended_settings": {
                    "cfg": 0.0 if distilled else 4.5,
                    "steps": 8 if distilled else 28,
                    "width": 1024, "height": 1024,
                    "source": "krea2_turbo" if distilled else "krea2_raw",
                },
            })
        # (b) single-file Krea 2 transformers (e.g. krea2_*_fp8_scaled.safetensors).
        # These are transformer-only; VAE/TE/scheduler come from a folder above.
        for f in sorted(base.rglob("*.safetensors")):
            n = f.name.lower()
            if "krea2" not in n and "krea-2" not in n and "krea_2" not in n:
                continue
            resolved = str(f.resolve())
            if resolved in seen:
                continue
            seen.add(resolved)
            distilled = "turbo" in n
            try:
                size = f.stat().st_size
            except OSError:
                size = 0
            scaled = "fp8" in n or "scaled" in n
            kind = ("Turbo" if distilled else "Raw") + (" fp8-scaled" if scaled else "")
            rel = f.relative_to(base).as_posix()
            out.append({
                "name": f"Krea 2 {kind} — {f.name}",
                "filename": rel,
                "subdir": f.parent.name,
                "abs_path": resolved,
                "size_bytes": size,
                "ext": "safetensors",
                "base_model": "Krea 2",
                "source_type": "krea2_single_file",
                "detected_arch": "krea2",
                "is_distilled": distilled,
                "recommended_settings": {
                    "cfg": 0.0 if distilled else 4.5,
                    "steps": 8 if distilled else 28,
                    "width": 1024, "height": 1024,
                    "source": "krea2_turbo" if distilled else "krea2_raw",
                },
            })
    return out


def _components_folder(prefer_distilled: bool) -> Path | None:
    """A Krea 2 diffusers folder to source VAE / text-encoder / tokenizer /
    scheduler from when loading a single-file transformer (components are
    identical across Raw/Turbo, so either works; prefer the matching one)."""
    folders: list[tuple[bool, Path]] = []
    for base in _SCAN_ROOTS:
        if not base.exists():
            continue
        for folder in sorted(base.iterdir()):
            if folder.is_dir():
                mi = _is_krea2_repo(folder)
                if mi is not None:
                    folders.append((bool(mi.get("is_distilled")), folder))
    if not folders:
        return None
    for distilled, folder in folders:
        if distilled == prefer_distilled:
            return folder
    return folders[0][1]


def _resolve_model(name: str | None) -> Path | None:
    """Resolve a selected model to a folder (diffusers repo) or file (single-file
    transformer). Accepts a discovered filename/name, an absolute path, or a
    single-file name found under MODELS_ROOT/diffusion_models."""
    if not name:
        return None
    p = Path(name)
    if p.is_absolute() and (_is_krea2_repo(p) is not None or p.is_file()):
        return p
    for entry in discover_models():
        if entry["filename"] == name or entry["abs_path"] == name or entry["name"] == name:
            return Path(entry["abs_path"])
    # Single-file picked via the main model scanner (relative to diffusion_models).
    cand = MODELS_ROOT / "diffusion_models" / name
    if cand.is_file():
        return cand
    for f in (MODELS_ROOT / "diffusion_models").rglob(Path(name).name):
        if f.is_file():
            return f
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


# ---------- text encoding (out-of-band, encoder freed before transformer load) ----------

def _encode_krea2_prompt(
    folder: Path,
    prompts: list[str],
    select_layers: tuple[int, ...],
    *,
    dtype: torch.dtype,
    max_sequence_length: int = 512,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Encode each prompt with the Qwen3-VL encoder using Krea 2's fixed-length
    template, tapping `select_layers` hidden states. The encoder is loaded, used
    and freed entirely within this call so it never coexists with the 24 GB
    transformer in VRAM.

    Returns a list of `(hidden_states, attention_mask)` CPU tensors, one per
    prompt, with shapes `(1, seq, num_text_layers, dim)` and `(1, seq)`.
    """
    from transformers import AutoConfig, AutoTokenizer, Qwen3VLModel

    t0 = time.time()
    device = torch.device(_device())
    # Krea's tokenizer_config.json stores `extra_special_tokens` as a list, which
    # transformers <5 (pinned here) tries to call `.keys()` on. The tokens are
    # already added tokens in tokenizer.json, so overriding the field with {} is
    # safe and yields identical tokenization.
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(folder / "tokenizer"))
    except (AttributeError, TypeError):
        tokenizer = AutoTokenizer.from_pretrained(str(folder / "tokenizer"), extra_special_tokens={})
    # Krea's text_encoder config uses the newer `rope_parameters` field name;
    # transformers <5 (pinned here) reads `rope_scaling` + `rope_theta`. Alias
    # them onto the text config so Qwen3-VL's mRoPE builds correctly.
    config = AutoConfig.from_pretrained(str(folder / "text_encoder"))
    tc = getattr(config, "text_config", None)
    if tc is not None:
        rp = getattr(tc, "rope_parameters", None) or getattr(tc, "rope_scaling", None)
        if rp:
            tc.rope_scaling = dict(rp)
            if getattr(tc, "rope_theta", None) in (None, 0):
                tc.rope_theta = rp.get("rope_theta", 10000)
    text_encoder = Qwen3VLModel.from_pretrained(str(folder / "text_encoder"), config=config, dtype=dtype)
    text_encoder = text_encoder.to(device=device, dtype=dtype).eval()

    # Krea 2 template: [system prefix | user prompt | PAD | assistant suffix].
    # The first `prefix_idx` (system) tokens are dropped from the encoder outputs.
    prefix = (
        "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, "
        "spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n"
    )
    suffix = "<|im_end|>\n<|im_start|>assistant\n"
    prefix_idx = 34
    num_suffix_tokens = 5

    results: list[tuple[torch.Tensor, torch.Tensor]] = []
    with torch.inference_mode():
        for prompt_item in prompts:
            text = [prefix + (prompt_item or "")]
            text_tokens = tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=max_sequence_length + prefix_idx - num_suffix_tokens,
                return_tensors="pt",
            ).to(device)
            suffix_tokens = tokenizer([suffix], return_tensors="pt").to(device)

            input_ids = torch.cat([text_tokens.input_ids, suffix_tokens.input_ids], dim=1)
            attention_mask = torch.cat(
                [text_tokens.attention_mask, suffix_tokens.attention_mask], dim=1
            ).bool()

            # Positions count only real tokens (padding does not consume a position),
            # broadcast across the 3 mRoPE axes (T/H/W equal for text).
            position_ids = (attention_mask.long().cumsum(dim=-1) - 1).clamp(min=0)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

            outputs = text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_hidden_states=True,
            )
            hidden = torch.stack([outputs.hidden_states[i] for i in select_layers], dim=2)
            hidden = hidden[:, prefix_idx:]
            mask = attention_mask[:, prefix_idx:]
            results.append(
                (hidden.to(dtype=dtype, device="cpu"), mask.to(device="cpu"))
            )

    del text_encoder, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    log.info(
        "encoded %d Krea 2 prompt(s) in %.1fs; Qwen3-VL encoder unloaded before transformer load",
        len(prompts), time.time() - t0,
    )
    return results


# ---------- component loaders ----------

# Resident size of the fp8-quantized 12.8B transformer: ~11.6 GB block weights in
# fp8 + ~1.2 GB non-block (embedders/projections/final/norms) kept in bf16/fp32.
_FP8_RESIDENT_GB = 13.0
_FP8_HEADROOM_GB = 3.0  # activations + VAE temporaries + Windows commit margin


def _fp8_mode() -> str:
    """KRAKEN_KREA2_FP8: 'auto' (default — fp8-resident when it fits), 'on'/'1'
    (force fp8), 'off'/'0' (force bf16 streaming)."""
    v = (os.environ.get("KRAKEN_KREA2_FP8") or "auto").strip().lower()
    if v in ("1", "on", "true", "yes", "fp8"):
        return "on"
    if v in ("0", "off", "false", "no", "bf16", "stream"):
        return "off"
    return "auto"


def _use_fp8_resident(free_after_vae_gb: float | None) -> bool:
    """Decide fp8-resident vs bf16-streaming BEFORE loading (so we never double
    load from disk). fp8-resident wins when the whole quantized model + headroom
    fits the free VRAM — the common case on a 24 GB card."""
    mode = _fp8_mode()
    if mode == "off":
        return False
    if mode == "on":
        return True
    free = free_after_vae_gb if free_after_vae_gb is not None else 20.0
    return free >= _FP8_RESIDENT_GB + _FP8_HEADROOM_GB


def _load_transformer_fp8(folder: Path, dtype: torch.dtype):
    """Load the transformer, quantize its blocks to fp8 (~24 -> ~13 GB), and keep
    the whole model GPU-resident — no streaming, ~5-10x faster per forward pass."""
    from pipelines.kraken_fp8_linear import quantize_blocks_to_fp8
    from pipelines.kraken_krea2_model import Krea2Transformer2DModel

    t0 = time.time()
    log.info("loading Krea 2 transformer from %s (fp8-resident path)", folder / "transformer")
    transformer = Krea2Transformer2DModel.from_pretrained(
        str(folder / "transformer"),
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    transformer.eval()
    n, before_gb, after_gb = quantize_blocks_to_fp8(transformer)
    gc.collect()
    transformer = transformer.to(_device())
    transformer._kraken_strategy = "fp8_resident"  # type: ignore[attr-defined]
    log.info(
        "  Krea 2 fp8-resident: quantized %d block Linears %.1f->%.1f GB; whole transformer "
        "on GPU (no streaming) in %.1fs",
        n, before_gb, after_gb, time.time() - t0,
    )

    # Optional torch.compile of the 28 repeated Krea2TransformerBlocks (regional
    # compilation). Safe only on this fully-resident path — the streaming path's
    # dynamic weight handoff would graph-break. Opt-in via KRAKEN_KREA2_COMPILE=on
    # because the first gen pays a one-time ~10-30 s compile cost.
    if (os.environ.get("KRAKEN_KREA2_COMPILE") or "off").strip().lower() in ("1", "on", "true"):
        try:
            import triton  # noqa: F401
            import torch._dynamo
            torch._dynamo.config.cache_size_limit = 256
            transformer.compile_repeated_blocks(fullgraph=False, dynamic=False)
            transformer._kraken_compiled = True  # type: ignore[attr-defined]
            log.info("  Krea 2 fp8: torch.compile applied to Krea2TransformerBlock (28 blocks)")
        except ImportError:
            log.info("  Krea 2: torch.compile skipped — Triton not installed (pip install triton-windows)")
        except Exception as e:
            log.warning("  Krea 2: torch.compile setup failed (%s) — continuing uncompiled", e)
    return transformer


def _set_submodule(root, path: str, mod) -> None:
    parts = path.split(".")
    obj = root
    for p in parts[:-1]:
        obj = obj[int(p)] if p.isdigit() else getattr(obj, p)
    setattr(obj, parts[-1], mod)


def _load_transformer_scaled_fp8(file_path: Path, cfg_folder: Path, dtype: torch.dtype):
    """Load a single-file ComfyUI-style *scaled* fp8 Krea 2 transformer.

    The 256 block Linears ship as `float8_e4m3fn` weights with a per-tensor F32
    `weight_scale`; everything else (norms, biases, img_in/time/txt/final, the
    text-fusion projector) is bf16. We convert the native keys to diffusers
    layout, install `Fp8Linear` for the quantized weights (re-using the stored
    scale instead of computing one), and load the rest plainly. The model is
    already ~13 GB, so it stays GPU-resident with no streaming.
    """
    from safetensors.torch import load_file

    from pipelines.kraken_fp8_linear import Fp8Linear
    from pipelines.kraken_krea2_convert import convert
    from pipelines.kraken_krea2_model import Krea2Transformer2DModel

    t0 = time.time()
    log.info("loading single-file scaled-fp8 Krea 2 transformer from %s", file_path)
    native = load_file(str(file_path))
    weights, scales = convert(native)
    unmapped = weights.pop("__unmapped__", [])
    if unmapped:
        log.warning("  scaled-fp8: %d unmapped keys (first: %s)", len(unmapped), unmapped[:3])

    cfg = json.loads((cfg_folder / "transformer" / "config.json").read_text(encoding="utf-8"))
    cfg.pop("_class_name", None)
    cfg.pop("_diffusers_version", None)
    with torch.device("meta"):
        transformer = Krea2Transformer2DModel.from_config(cfg)

    # Install Fp8Linear for the quantized weights (those that carry a scale).
    fp8_weight_keys = set(scales.keys())
    for wkey in fp8_weight_keys:
        module_path = wkey[: -len(".weight")]  # strip trailing '.weight'
        w_fp8 = weights[wkey]
        sc = scales[wkey].to(torch.float32).reshape(1)
        _set_submodule(transformer, module_path, Fp8Linear(w_fp8, sc, None))

    # Load everything else (norms, biases, bf16 Linears, scale_shift_tables).
    regular = {k: v for k, v in weights.items() if k not in fp8_weight_keys}
    missing, unexpected = transformer.load_state_dict(regular, strict=False, assign=True)
    # `missing` will list the fp8 buffers (already installed) — filter those out.
    real_missing = [m for m in missing if m.rsplit(".", 1)[0] not in
                    {k[: -len(".weight")] for k in fp8_weight_keys}]
    if real_missing:
        log.warning("  scaled-fp8: %d genuinely missing keys (first: %s)", len(real_missing), real_missing[:5])
    if unexpected:
        log.warning("  scaled-fp8: %d unexpected keys (first: %s)", len(unexpected), list(unexpected)[:5])

    del native, weights, scales
    gc.collect()
    transformer = transformer.to(_device()).eval()
    transformer._kraken_strategy = "scaled_fp8_resident"  # type: ignore[attr-defined]
    log.info("  Krea 2 scaled-fp8: transformer on GPU (no streaming) in %.1fs", time.time() - t0)
    return transformer


def _load_transformer_streamed(folder: Path, dtype: torch.dtype, free_after_vae_gb: float | None):
    """Load the 12.8B transformer to CPU, then swap to StreamingLinear + place it
    with the FLUX-style budget so most blocks stream from pinned CPU memory."""
    from pipelines.kraken_krea2_model import Krea2Transformer2DModel
    from pipelines.streaming_linear import apply_streaming, count_streaming, swap_linears

    t0 = time.time()
    log.info("loading Krea 2 transformer from %s (sharded diffusers, ~24 GB)", folder / "transformer")
    transformer = Krea2Transformer2DModel.from_pretrained(
        str(folder / "transformer"),
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    transformer.eval()
    log.info("  transformer materialized on CPU in %.1fs", time.time() - t0)

    n_swapped = swap_linears(transformer)
    log.info("  swapped %d Linear modules to StreamingLinear", n_swapped)

    # Budget = free VRAM after VAE, minus the headroom the resident weights must
    # leave behind. Three things compete for that headroom and ALL must fit or the
    # forward spills to Windows shared memory (≈30× slowdown — the 1969 s/1024²
    # bug): per-step activations, plus the flat prefetch buffers that
    # `apply_streaming` allocates AFTER this partition (so they are NOT in the
    # budget and would otherwise eat the activation reserve). Krea 2 is a full
    # bf16 ~24 GB model, so unlike FP8 FLUX it sits right at the 24 GB edge — be
    # generous: better to stream a few more blocks (cheap, overlapped) than spill.
    activation_gb = 5.0   # Qwen-Image-style activations at 1024²–1536²
    prefetch_gb = 2.0     # flat prefetch buffers allocated outside the budget
    windows_margin = 1.0
    if free_after_vae_gb is None:
        free_after_vae_gb = 20.0
    budget_gb = max(0.5, free_after_vae_gb - activation_gb - prefetch_gb - windows_margin)
    summary = apply_streaming(
        transformer,
        budget_bytes=int(budget_gb * 1024**3),
        device=torch.device(_device()),
        pin_memory=True,
        label="Krea 2 transformer",
    )
    n_total, n_streamed = count_streaming(transformer)
    log.info(
        "  Krea 2 streaming: budget=%.1f GB · %d/%d StreamingLinears stream from CPU · %s",
        budget_gb, n_streamed, n_total, summary,
    )
    return transformer


def _resolve_vae(name: str | None) -> Path | None:
    if not name:
        return None
    base = MODELS_ROOT / "vae"
    p = base / name
    if p.exists():
        return p
    for f in base.rglob(name):
        return f
    return None


def _load_vae(folder: Path, dtype: torch.dtype, vae_name: str | None = None):
    from diffusers import AutoencoderKLQwenImage, AutoencoderKLWan

    vae_path = _resolve_vae(vae_name)
    if vae_path is not None:
        log.info("loading Krea 2 VAE override from %s", vae_path)
        if vae_path.is_file():
            # Qwen/Wan single-file VAEs are supported through the Wan loader in
            # diffusers. Krea's bundled folder VAE still uses AutoencoderKLQwenImage.
            vae = AutoencoderKLWan.from_single_file(str(vae_path), torch_dtype=dtype)
        else:
            vae = AutoencoderKLQwenImage.from_pretrained(str(vae_path), torch_dtype=dtype)
    else:
        if vae_name:
            raise FileNotFoundError(f"Krea 2 VAE override not found: {vae_name!r}")
        log.info("loading Krea 2 VAE (AutoencoderKLQwenImage) from %s", folder / "vae")
        vae = AutoencoderKLQwenImage.from_pretrained(str(folder / "vae"), torch_dtype=dtype)
    vae = vae.to(device=_device(), dtype=dtype).eval()
    return vae


# ---------- pipeline lifecycle ----------

def _ensure_pipeline(model_path: Path, vae_name: str | None = None, loras: list[dict] | None = None):
    global _pipeline, _pipeline_key

    key = (str(model_path), vae_name or "", _lora_signature(loras or []))
    if _pipeline is not None and _pipeline_key == key:
        return _pipeline

    unload()

    from diffusers import FlowMatchEulerDiscreteScheduler
    from pipelines.kraken_krea2_pipeline import Krea2Pipeline

    is_single_file = model_path.is_file()
    if is_single_file:
        # Single-file transformer (e.g. scaled-fp8): source VAE/TE/scheduler/config
        # from a Krea 2 diffusers folder. is_distilled inferred from the filename.
        is_distilled = "turbo" in model_path.name.lower()
        comp = _components_folder(is_distilled)
        if comp is None:
            raise FileNotFoundError(
                "Single-file Krea 2 checkpoint needs a Krea 2 diffusers folder "
                "(Krea-2-Raw / Krea-2-Turbo) alongside it for the VAE + text encoder. "
                "None found under the Krea 2 scan roots."
            )
        log.info("Krea 2 pipeline load starting: %s (single-file, components from %s)",
                 model_path.name, comp.name)
    else:
        comp = model_path
        log.info("Krea 2 pipeline load starting: %s", model_path.name)

    mi = json.loads((comp / "model_index.json").read_text(encoding="utf-8"))
    select_layers = tuple(mi.get("text_encoder_select_layers") or
                          (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35))
    if not is_single_file:
        is_distilled = bool(mi.get("is_distilled"))
    patch_size = int(mi.get("patch_size") or 2)

    dtype = _dtype()

    # VAE first (small) so we can measure remaining VRAM for the load strategy.
    vae = _load_vae(comp, dtype, vae_name)
    free_after_vae = _free_vram_gb()
    if is_single_file:
        transformer = _load_transformer_scaled_fp8(model_path, comp, dtype)
    elif _use_fp8_resident(free_after_vae):
        transformer = _load_transformer_fp8(comp, dtype)
    else:
        transformer = _load_transformer_streamed(comp, dtype, free_after_vae)
    strategy = getattr(transformer, "_kraken_strategy", "streaming")

    sched_cfg = json.loads((comp / "scheduler" / "scheduler_config.json").read_text(encoding="utf-8"))
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(sched_cfg)

    # Text encoder + tokenizer are handled out-of-band in `_encode_krea2_prompt`
    # and freed before this runs, so the pipeline holds no encoder; we pass
    # prompt_embeds directly at call time.
    pipe = Krea2Pipeline(
        scheduler=scheduler,
        vae=vae,
        text_encoder=None,
        tokenizer=None,
        transformer=transformer,
        text_encoder_select_layers=select_layers,
        is_distilled=is_distilled,
        patch_size=patch_size,
    )
    pipe.set_progress_bar_config(disable=True)
    # Tile/slice the VAE decode so its peak VRAM stays bounded at large canvases
    # (decode of a 1536²/2048 image is otherwise the tallest single allocation and
    # can spill even though sampling fits). Negligible quality impact for this VAE.
    try:
        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()
    except Exception as e:
        log.warning("Krea 2: VAE tiling unavailable (%s)", e)
    pipe._kraken_offload_strategy = strategy  # type: ignore[attr-defined]
    _apply_loras(pipe, loras or [])

    _pipeline = pipe
    _pipeline_key = key
    return pipe


# ---------- LoRA ----------

def _adapter_name(lora_name: str) -> str:
    return "lora_" + "".join(ch if ch.isalnum() else "_" for ch in lora_name)[:48]


def _lora_model_weight(entry: dict) -> float:
    value = entry.get("model_weight")
    if value is None:
        value = entry.get("weight")
    if value is None:
        value = 1.0
    return float(value)


def _resolve_lora(name: str | None) -> Path | None:
    if not name:
        return None
    base = MODELS_ROOT / "loras"
    p = base / name
    if p.exists():
        return p
    for f in base.rglob(name):
        return f
    return None


def _lora_signature(loras: list[dict]) -> tuple[tuple[str, float], ...]:
    return tuple((str(l.get("name") or ""), _lora_model_weight(l)) for l in (loras or []) if l.get("name"))


def _validate_lora_requests(loras: list[dict]) -> None:
    """Fail fast before loading/encoding if a requested Krea LoRA cannot apply."""
    if not loras:
        return
    from safetensors import safe_open

    supported_suffixes = (
        ".diff",
        ".diff_b",
        ".lora_down.weight",
        ".lora_A.weight",
    )
    errors: list[str] = []
    for entry in loras:
        name = entry.get("name")
        if not name:
            continue
        path = _resolve_lora(str(name))
        if path is None:
            errors.append(f"{name}: file not found under {MODELS_ROOT / 'loras'}")
            continue
        try:
            with safe_open(str(path), framework="pt", device="cpu") as f:
                keys = list(f.keys())
        except Exception as e:
            errors.append(f"{name}: could not read safetensors ({e})")
            continue
        targetable = 0
        for key in keys:
            prefix = None
            if key.endswith(".diff") or key.endswith(".diff_b"):
                prefix = key.rsplit(".", 1)[0]
            elif key.endswith(".lora_down.weight"):
                prefix = key[: -len(".lora_down.weight")]
            elif key.endswith(".lora_A.weight"):
                prefix = key[: -len(".lora_A.weight")]
            if prefix and _native_krea_lora_target(prefix):
                targetable += 1
        if not any(key.endswith(supported_suffixes) for key in keys):
            errors.append(f"{name}: no supported Krea LoRA tensors found")
        elif targetable == 0:
            errors.append(f"{name}: LoRA tensors do not target Kraken's Krea 2 transformer")
    if errors:
        raise RuntimeError("Invalid Krea LoRA selection: " + "; ".join(errors))


def _get_submodule(root, path: str):
    obj = root
    for part in path.split("."):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj


def _native_krea_lora_target(prefix: str) -> str | None:
    p = prefix
    known_namespace = False
    for start in ("diffusion_model.", "base_model.model."):
        if p.startswith(start):
            p = p[len(start):]
            known_namespace = True
            break

    if not known_namespace:
        raw_prefixes = ("blocks.", "txtfusion.", "txtmlp.", "first", "tmlp.", "tproj.", "last.linear")
        native_prefixes = (
            "transformer_blocks.",
            "text_fusion.",
            "txt_in.",
            "img_in",
            "time_embed.",
            "time_mod_proj",
            "final_layer.",
        )
        if p.startswith(raw_prefixes) or p.startswith(native_prefixes):
            known_namespace = True
    if not known_namespace:
        return None

    for old, new in (("blocks.", "transformer_blocks."), ("txtfusion.", "text_fusion.")):
        if p.startswith(old):
            p = f"{new}{p[len(old):]}"
            break
    exact_replacements = [
        ("txtmlp.1", "txt_in.linear_1"),
        ("txtmlp.3", "txt_in.linear_2"),
        ("first", "img_in"),
        ("tmlp.0", "time_embed.linear_1"),
        ("tmlp.2", "time_embed.linear_2"),
        ("tproj.1", "time_mod_proj"),
        ("last.linear", "final_layer.linear"),
    ]
    for old, new in exact_replacements:
        if p == old or p.startswith(f"{old}."):
            p = f"{new}{p[len(old):]}"
            break

    p = p.replace(".attn.wq", ".attn.to_q")
    p = p.replace(".attn.wk", ".attn.to_k")
    p = p.replace(".attn.wv", ".attn.to_v")
    p = p.replace(".attn.wo", ".attn.to_out.0")
    p = p.replace(".attn.gate", ".attn.to_gate")
    p = p.replace(".mlp.", ".ff.")
    return p


def _requantize_fp8_linear(module, weight: torch.Tensor) -> None:
    from pipelines.kraken_fp8_linear import _E4M3_MAX

    wf = weight.float()
    scale = (wf.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / _E4M3_MAX).to(torch.float32)
    q = (wf / scale).clamp(-_E4M3_MAX, _E4M3_MAX).to(torch.float8_e4m3fn)
    module.weight = q.to(device=module.weight.device)
    module.scale = scale.to(device=module.scale.device)


def _add_linear_delta(module, delta: torch.Tensor) -> None:
    target_device = module.weight.device
    if module.__class__.__name__ == "Fp8Linear":
        base = module.weight.to(dtype=torch.float32) * module.scale.to(dtype=torch.float32)
        new_weight = base + delta.to(device=target_device, dtype=torch.float32)
        _requantize_fp8_linear(module, new_weight)
        return

    with torch.no_grad():
        module.weight.data.add_(delta.to(device=target_device, dtype=module.weight.dtype))


def _apply_native_krea_lora(transformer, path: Path, strength: float) -> tuple[int, int, int]:
    """Apply Comfy/fal Krea 2 LoRA formats directly to Kraken's vendored model.

    The public Krea LoRAs are not all diffusers-PEFT adapters:
    - Comfy format: `diffusion_model.blocks.*.lora_down/up`
    - fal format: `base_model.model.blocks.*.lora_A/B`
    - tiny bypass files: direct `txtfusion.projector.diff`

    We fuse them at load time, and the pipeline cache key includes the full LoRA
    stack, so changing LoRAs reloads the base transformer instead of stacking
    deltas twice.
    """
    from safetensors import safe_open

    applied = 0
    direct = 0
    skipped = 0
    with safe_open(str(path), framework="pt", device="cpu") as f:
        keys = set(f.keys())
        metadata = f.metadata() or {}
        alpha_raw = metadata.get("lora_alpha")

        # Direct projector-vector patches used by the Krea filter-bypass files.
        for key in sorted(k for k in keys if k.endswith(".diff") or k.endswith(".diff_b")):
            prefix = key.rsplit(".", 1)[0]
            target = _native_krea_lora_target(prefix)
            if not target:
                skipped += 1
                continue
            try:
                tensor = f.get_tensor(key).to(torch.float32) * float(strength)
                if key.endswith(".diff"):
                    module = _get_submodule(transformer, target)
                    with torch.no_grad():
                        module.weight.data.add_(tensor.to(device=module.weight.device, dtype=module.weight.dtype))
                else:
                    module = _get_submodule(transformer, target)
                    if getattr(module, "bias", None) is None:
                        skipped += 1
                        continue
                    with torch.no_grad():
                        module.bias.data.add_(tensor.reshape_as(module.bias).to(device=module.bias.device, dtype=module.bias.dtype))
                direct += 1
            except Exception as e:
                skipped += 1
                log.warning("failed to apply Krea direct LoRA key %s from %s: %s", key, path.name, e)

        down_suffixes = (".lora_down.weight", ".lora_A.weight")
        for down_key in sorted(k for k in keys if k.endswith(down_suffixes)):
            if down_key.endswith(".lora_down.weight"):
                prefix = down_key[: -len(".lora_down.weight")]
                up_key = f"{prefix}.lora_up.weight"
            else:
                prefix = down_key[: -len(".lora_A.weight")]
                up_key = f"{prefix}.lora_B.weight"
            if up_key not in keys:
                skipped += 1
                continue
            target = _native_krea_lora_target(prefix)
            if not target:
                skipped += 1
                continue
            try:
                module = _get_submodule(transformer, target)
                down = f.get_tensor(down_key)
                up = f.get_tensor(up_key)
                rank = max(int(down.shape[0]), 1)
                alpha = float(alpha_raw) if alpha_raw is not None else float(rank)
                scale = float(strength) * (alpha / float(rank))
                device = module.weight.device
                compute_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
                delta = torch.matmul(
                    up.to(device=device, dtype=compute_dtype),
                    down.to(device=device, dtype=compute_dtype),
                ).mul_(scale)
                _add_linear_delta(module, delta)
                applied += 1
                del delta, down, up
            except Exception as e:
                skipped += 1
                log.warning("failed to apply Krea LoRA target %s from %s: %s", target, path.name, e)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return applied, direct, skipped


def _apply_loras(pipe, loras: list[dict]) -> None:
    global _loaded_loras
    sig = _lora_signature(loras)
    if tuple((name, weight) for name, weight in _loaded_loras) == sig:
        return
    if _loaded_loras:
        raise RuntimeError("Krea LoRA stack changed without reloading the transformer.")
    _loaded_loras = []
    if not loras:
        return
    applied_names: list[tuple[str, float]] = []
    for entry in loras:
        name = str(entry.get("name") or "")
        path = _resolve_lora(name)
        if not path:
            raise RuntimeError(f"Krea LoRA not found: {name}")
        weight = _lora_model_weight(entry)
        try:
            n_pairs, n_direct, n_skipped = _apply_native_krea_lora(pipe.transformer, path, weight)
            if not (n_pairs or n_direct):
                raise RuntimeError(f"applied no compatible weights ({n_skipped} skipped)")
            if n_skipped:
                raise RuntimeError(f"only partially applied ({n_pairs} rank pairs, {n_direct} direct deltas, {n_skipped} skipped)")
            applied_names.append((name, weight))
            log.info(
                "applied Krea LoRA %s @ %.2f (%d rank pairs, %d direct deltas, %d skipped)",
                name, weight, n_pairs, n_direct, n_skipped,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to apply Krea LoRA {name}: {e}") from e
    _loaded_loras = applied_names


# ---------- run ----------

def _make_callback(job, total_steps: int, image_index: int, total_images: int):
    last = {"t": time.time()}

    def cb(pipe, step: int, timestep, callback_kwargs):
        if job.cancel.is_set():
            raise RuntimeError("cancelled")
        now = time.time()
        dt = now - last["t"]
        last["t"] = now
        # Per-step timing + VRAM. A step time that's ~30× the GPU's compute rate,
        # with reserved pinned near the card limit, means activations are spilling
        # to Windows shared memory — bump the streaming activation reserve.
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            log.info("Krea 2 step %d/%d: %.1fs · VRAM alloc=%.2f GB reserved=%.2f GB",
                     step + 1, total_steps, dt, alloc, reserved)
        else:
            log.info("Krea 2 step %d/%d: %.1fs", step + 1, total_steps, dt)
        job.progress.step = step + 1
        job.progress.total_steps = total_steps
        job.progress.image_index = image_index
        job.progress.total_images = total_images
        job.emit({
            "type": "progress",
            "step": step + 1, "total_steps": total_steps,
            "image_index": image_index, "total_images": total_images,
        })
        return callback_kwargs
    return cb


def _thumbnail_b64(img: Image.Image, max_side: int = 320) -> str:
    thumb = img.copy()
    thumb.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    thumb.save(buf, format="JPEG", quality=80)
    return b64encode(buf.getvalue()).decode("ascii")


def run(job) -> dict:
    # Free the other arch pipelines so we don't double-book VRAM.
    from pipelines import flux as flux_mod, ideogram4 as ideogram4_mod, sdxl as sdxl_mod, z_image as z_image_mod
    sdxl_mod.unload()
    flux_mod.unload()
    z_image_mod.unload()
    try:
        ideogram4_mod.unload()
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    p = job.params
    _validate_lora_requests(p.get("loras") or [])

    sampler = str(p.get("sampler") or "euler").lower()
    scheduler = str(p.get("scheduler") or "normal").lower()
    if sampler not in {"", "euler"} or scheduler not in {"", "normal"}:
        raise RuntimeError(
            "Krea 2 currently supports only Kraken's FlowMatch Euler sampler path "
            f"(sampler='euler', scheduler='normal'); requested sampler={p.get('sampler')!r}, "
            f"scheduler={p.get('scheduler')!r}. Add native Krea sampler support before using "
            "RES4LYF/Comfy sampler names here."
        )

    dtype = _dtype()

    model_path = _resolve_model(p.get("diffusion_model"))
    if model_path is None:
        raise FileNotFoundError(
            f"Krea 2 model not found: {p.get('diffusion_model')!r}. "
            "Place the Krea-2-Raw / Krea-2-Turbo diffusers folder under <root>/krea2/."
        )

    # Single-file transformers source their non-transformer components (VAE, text
    # encoder, tokenizer, scheduler, config) from a Krea 2 diffusers folder.
    if model_path.is_file():
        is_distilled = "turbo" in model_path.name.lower()
        comp_folder = _components_folder(is_distilled)
        if comp_folder is None:
            raise FileNotFoundError(
                "Single-file Krea 2 checkpoint needs a Krea 2 diffusers folder "
                "(Krea-2-Raw / Krea-2-Turbo) for its VAE + text encoder; none found."
            )
    else:
        comp_folder = model_path

    mi = json.loads((comp_folder / "model_index.json").read_text(encoding="utf-8"))
    select_layers = tuple(mi.get("text_encoder_select_layers") or
                          (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35))
    if not model_path.is_file():
        is_distilled = bool(mi.get("is_distilled"))

    width = int(p.get("width") or 1024)
    height = int(p.get("height") or 1024)
    # Turbo (distilled): few steps, no CFG. Raw: honour UI, default 28 steps / cfg 4.5.
    steps = int(p.get("steps") or (8 if is_distilled else 28))
    guidance = 0.0 if is_distilled else float(p.get("cfg") if p.get("cfg") is not None else 4.5)
    count = int(p.get("count") or 1)
    seed = p.get("seed")

    # Krea 2 requires height/width divisible by vae_scale_factor*patch_size (=16). Snap down.
    width -= width % 16
    height -= height % 16

    # 1) Encode prompt (and negative when guided) with Qwen3-VL, then free it.
    _t_enc = time.time()
    pos = _encode_krea2_prompt(comp_folder, [p.get("prompt", "")], select_layers, dtype=dtype)[0]
    neg = None
    if guidance > 0:
        neg = _encode_krea2_prompt(comp_folder, [p.get("negative", "") or ""], select_layers, dtype=dtype)[0]
    _enc_s = time.time() - _t_enc

    # 2) Build transformer + VAE pipeline.
    _t_setup = time.time()
    pipe = _ensure_pipeline(model_path, p.get("vae"), p.get("loras") or [])
    log.info(
        "Krea 2 run stages: text-encode=%.1fs · pipeline-ensure+loras=%.1fs (sampling follows)",
        _enc_s, time.time() - _t_setup,
    )

    exec_device = torch.device(_device())
    prompt_embeds = pos[0].to(device=exec_device)
    prompt_embeds_mask = pos[1].to(device=exec_device)
    negative_prompt_embeds = neg[0].to(device=exec_device) if neg else None
    negative_prompt_embeds_mask = neg[1].to(device=exec_device) if neg else None
    del pos, neg

    out_dir = OUTPUTS_ROOT / time.strftime("%Y-%m-%d")
    out_dir.mkdir(parents=True, exist_ok=True)
    base_stem = build_output_stem(p, job.id)

    saved: list[dict] = []
    for i in range(count):
        if job.cancel.is_set():
            break
        per_seed = (seed if seed is not None else torch.seed()) + i
        generator = torch.Generator(device=exec_device.type).manual_seed(int(per_seed) & 0x7FFFFFFF)
        result = pipe(
            prompt=None,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_embeds_mask=negative_prompt_embeds_mask,
            width=width, height=height,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
            output_type="pil",
            callback_on_step_end=_make_callback(job, steps, i, count),
        )
        img = result.images[0]

        fname = f"{base_stem}-{i:02d}.png"
        fpath = out_dir / fname
        save_png_with_metadata(img, fpath, p, int(per_seed))
        entry = {
            "path": str(fpath), "filename": fname, "seed": int(per_seed),
            "width": width, "height": height,
        }

        if p.get("upscale_enabled") and p.get("upscale_model"):
            try:
                from pipelines import upscale_esrgan
                up = upscale_esrgan.upscale(img, p["upscale_model"], float(p.get("upscale_factor", 2.0)))
                up_path = out_dir / f"{base_stem}-{i:02d}-up.png"
                save_png_with_metadata(up, up_path, p, int(per_seed))
                entry["upscaled_path"] = str(up_path)
                img = up
            except Exception as e:
                log.warning("upscale failed: %s", e)

        rel_path = fpath.relative_to(OUTPUTS_ROOT).as_posix()
        entry["rel_path"] = rel_path
        saved.append(entry)
        job.emit({
            "type": "image",
            "image_index": i, "path": str(fpath), "rel_path": rel_path, "filename": fname,
            "seed": int(per_seed), "preview_b64": _thumbnail_b64(img),
        })

    return {
        "kind": "image",
        "count_requested": count,
        "count_produced": len(saved),
        "outputs": saved,
        "output_dir": str(out_dir),
    }
