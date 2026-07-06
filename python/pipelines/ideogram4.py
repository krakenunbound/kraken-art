"""Ideogram 4 pipeline wrapper for Kraken Art.

Uses Ideogram's official open-weight inference package from external/ideogram4.
The RTX 3090 path is NF4 via bitsandbytes.
"""
from __future__ import annotations

import gc
import importlib.util
import io
import json
import logging
import os
import queue
import subprocess
import sys
import threading
import time
from base64 import b64encode
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from config import OUTPUTS_ROOT, ROOT
from pipelines.ideogram4_magic import preview_local_magic_prompt
from pipelines.output_metadata import build_output_stem, save_png_with_metadata

log = logging.getLogger("kraken.ideogram4")

IDEOGRAM_SRC = ROOT / "external" / "ideogram4" / "src"
NF4_REPO = "ideogram-ai/ideogram-4-nf4"
FP8_REPO = "ideogram-ai/ideogram-4-fp8"
LOCAL_NF4_REPO = ROOT / "models" / "ideogram4" / "nf4"
LOCAL_FP8_REPO = ROOT / "models" / "ideogram4" / "fp8"
LOCAL_Q4K_REPO = ROOT / "Ideogram" / "ideogram-4-gguf-q4_k"
LOCAL_Q4K_GGUF = LOCAL_Q4K_REPO / "ideogram4-q4_k.gguf"
LOCAL_INT8_FUSED_REPO = ROOT / "Ideogram" / "ideogram-4-int8-fused"
LOCAL_INT8_WEIGHTS = ROOT / "Ideogram" / "ideogram-4-int8-w8a8" / "ideogram4-int8-w8a8.safetensors"

_pipeline: Any | None = None
_pipeline_key: tuple[str, str] | None = None
WORKER_SCRIPT = ROOT / "python" / "pipelines" / "ideogram4_worker.py"
_worker_proc: subprocess.Popen | None = None
_worker_lines: queue.Queue[str | None] | None = None
_worker_lock = threading.Lock()


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _dtype() -> torch.dtype:
    return torch.bfloat16 if torch.cuda.is_available() else torch.float32


# Approx VRAM the LOAD needs resident, by quant (measured 2026-06-17 on a 3090,
# 1024px): NF4 = two DiT branches (~10 GB) + Qwen3 text encoder (~5.5 GB) + VAE
# -> ~16-18 GB. On Windows, a CUDA allocation that can't fit doesn't raise a
# clean OOM — bitsandbytes' Linear4bit build dies with a fatal access violation
# (the worker crash you can hit when the card is shared with other GPU apps).
# So we pre-check free VRAM and fail with a readable message instead.
_LOAD_VRAM_GB = {"nf4": 16.5, "fp8": 20.0, "q4_k": 18.0, "int8_fused": 18.0}


def _require_free_vram(quant: str) -> None:
    if not torch.cuda.is_available():
        return
    need = _LOAD_VRAM_GB.get(quant, 16.5)
    free_gb = torch.cuda.mem_get_info()[0] / (1024 ** 3)
    if free_gb + 0.5 < need:
        raise RuntimeError(
            f"Not enough free VRAM to load Ideogram 4 {quant.upper()} "
            f"(only {free_gb:.1f} GB free, needs ~{need:.0f} GB). Ideogram 4 is a "
            f"large model and wants most of a 24 GB card to itself. Close other GPU "
            f"apps (a browser, a game, another AI tool, a second Kraken window) or "
            f"click Clear VRAM, then try again."
        )


def _ensure_ideogram_import_path() -> None:
    if not IDEOGRAM_SRC.exists():
        raise FileNotFoundError(
            f"Ideogram source not found at {IDEOGRAM_SRC}. "
            "Run: git clone https://github.com/ideogram-oss/ideogram4 external\\ideogram4"
        )
    src = str(IDEOGRAM_SRC)
    if src not in sys.path:
        sys.path.insert(0, src)


def unload() -> dict:
    global _pipeline, _pipeline_key, _worker_proc, _worker_lines
    had = _pipeline is not None
    _pipeline = None
    _pipeline_key = None
    had_worker = _worker_proc is not None and _worker_proc.poll() is None
    if had_worker:
        try:
            _worker_proc.terminate()
            _worker_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _worker_proc.kill()
        except Exception:
            pass
    _worker_proc = None
    _worker_lines = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    return {"had_pipeline": had, "had_worker": had_worker}


def _quantization_repo(model_name: str | None) -> tuple[str, str]:
    name = (model_name or "").lower()
    if "int8" in name or "w8a8" in name or "fused" in name:
        if LOCAL_INT8_FUSED_REPO.exists() and LOCAL_INT8_WEIGHTS.exists():
            return "int8_fused", str(LOCAL_INT8_WEIGHTS)
        raise FileNotFoundError(
            "Ideogram 4 INT8 Fused requires "
            f"{LOCAL_INT8_FUSED_REPO} and {LOCAL_INT8_WEIGHTS}"
        )
    if "q4" in name or "gguf" in name:
        if LOCAL_Q4K_GGUF.exists():
            return "q4_k", str(LOCAL_Q4K_REPO)
        raise FileNotFoundError(f"Ideogram 4 Q4_K GGUF not found at {LOCAL_Q4K_GGUF}")
    if "fp8" in name:
        if (LOCAL_FP8_REPO / "model_index.json").exists():
            return "fp8", str(LOCAL_FP8_REPO)
        return "fp8", FP8_REPO
    if (LOCAL_NF4_REPO / "model_index.json").exists():
        return "nf4", str(LOCAL_NF4_REPO)
    return "nf4", NF4_REPO


def _pipeline_cache_state(model_name: str | None) -> tuple[bool, str, str, str, tuple[str, str]]:
    quant, repo = _quantization_repo(model_name)
    device = _device()
    key = (repo, device)
    return _pipeline is not None and _pipeline_key == key, quant, repo, device, key


def _fp8_base_repo() -> str:
    if (LOCAL_FP8_REPO / "model_index.json").exists():
        return str(LOCAL_FP8_REPO)
    return FP8_REPO


def _q4k_base_repo() -> str:
    if (LOCAL_FP8_REPO / "model_index.json").exists():
        return str(LOCAL_FP8_REPO)
    if (LOCAL_NF4_REPO / "model_index.json").exists():
        return str(LOCAL_NF4_REPO)
    return FP8_REPO


def _load_fused_int8_module():
    loader_path = LOCAL_INT8_FUSED_REPO / "fused_int8.py"
    if not loader_path.exists():
        raise FileNotFoundError(f"Ideogram 4 fused INT8 loader not found at {loader_path}")
    repo = str(LOCAL_INT8_FUSED_REPO)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    spec = importlib.util.spec_from_file_location("kraken_ideogram4_fused_int8", loader_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import Ideogram 4 fused INT8 loader from {loader_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _install_fused_int8(pipe: Any, weights_path: Path, fused_module: Any) -> tuple[int, int]:
    from safetensors import safe_open

    f = safe_open(str(weights_path), framework="pt")
    keys = set(f.keys())
    fused = protected = 0
    for branch, dit in [
        ("conditional", pipe.conditional_transformer),
        ("unconditional", pipe.unconditional_transformer),
    ]:
        for pname, parent in dit.named_modules():
            for cname, child in list(parent.named_children()):
                full = f"{pname}.{cname}" if pname else cname
                base = f"{branch}.{full}"
                has_protected = f"{base}.weight" in keys
                has_fused = f"{base}.wq" in keys
                if not has_protected and not has_fused:
                    continue

                dev = pipe.device
                # Drop the existing NF4/FP8 module before reading replacement weights.
                # On 24 GB cards, briefly holding both can kill the process.
                setattr(parent, cname, torch.nn.Identity())
                del child
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                bias = f.get_tensor(f"{base}.bias").to(dev) if f"{base}.bias" in keys else None
                if has_protected:
                    weight = f.get_tensor(f"{base}.weight").to(dev)
                    linear = torch.nn.Linear(
                        weight.shape[1],
                        weight.shape[0],
                        bias=bias is not None,
                        dtype=torch.bfloat16,
                        device=dev,
                    )
                    linear.weight = torch.nn.Parameter(weight, requires_grad=False)
                    if bias is not None:
                        linear.bias = torch.nn.Parameter(bias, requires_grad=False)
                    setattr(parent, cname, linear)
                    protected += 1
                elif has_fused:
                    wq = f.get_tensor(f"{base}.wq").to(dev)
                    ws = f.get_tensor(f"{base}.ws").to(dev)
                    smooth = f.get_tensor(f"{base}.smooth").to(dev) if f"{base}.smooth" in keys else None
                    setattr(parent, cname, fused_module.FusedW8A8(wq, ws, smooth, bias))
                    fused += 1
                installed = fused + protected
                if installed and installed % 16 == 0:
                    log.info("Ideogram 4 INT8 Fused installed %d layers", installed)
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
    return fused, protected


def _q4k_dequantized_bf16_gb() -> float:
    try:
        from gguf import GGUFReader
    except Exception:
        return 0.0
    reader = GGUFReader(str(LOCAL_Q4K_GGUF))
    total_elements = 0
    for tensor in reader.tensors:
        elements = 1
        for dim in tensor.shape:
            elements *= int(dim)
        total_elements += elements
    return (total_elements * 2) / (1024**3)


def _gguf_tensor_to_bf16(tensor: Any) -> torch.Tensor:
    import numpy as np
    from gguf import GGMLQuantizationType, dequantize

    plain = {GGMLQuantizationType.F16, GGMLQuantizationType.F32}
    arr = np.array(tensor.data) if tensor.tensor_type in plain else dequantize(tensor.data, tensor.tensor_type)
    shape = tuple(int(d) for d in reversed(tensor.shape))
    return torch.from_numpy(np.ascontiguousarray(arr).reshape(shape)).to(torch.bfloat16)


def _swap_branch_from_gguf_reader(dit: Any, tensor_index: dict[str, Any], branch: str, device: str) -> int:
    swapped = 0
    for pname, parent in dit.named_modules():
        for cname, child in list(parent.named_children()):
            full = f"{pname}.{cname}" if pname else cname
            wkey = f"{branch}.{full}.weight"
            tensor = tensor_index.get(wkey)
            if tensor is None:
                continue
            weight = _gguf_tensor_to_bf16(tensor)
            if weight.ndim != 2:
                continue
            bkey = f"{branch}.{full}.bias"
            bias_tensor = tensor_index.get(bkey)
            weight = weight.to(device)
            linear = torch.nn.Linear(
                weight.shape[1],
                weight.shape[0],
                bias=bias_tensor is not None,
                dtype=torch.bfloat16,
                device=device,
            )
            linear.weight = torch.nn.Parameter(weight, requires_grad=False)
            if bias_tensor is not None:
                linear.bias = torch.nn.Parameter(_gguf_tensor_to_bf16(bias_tensor).to(device), requires_grad=False)
            setattr(parent, cname, linear)
            del child, weight
            swapped += 1
            if swapped % 64 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    return swapped


def _ensure_pipeline(model_name: str | None):
    global _pipeline, _pipeline_key

    _ensure_ideogram_import_path()
    from ideogram4 import Ideogram4Pipeline, Ideogram4PipelineConfig

    cached, quant, repo, device, key = _pipeline_cache_state(model_name)
    if quant in {"nf4", "q4_k", "int8_fused"} and device != "cuda":
        raise RuntimeError(f"Ideogram 4 {quant.upper()} requires CUDA.")

    if cached:
        log.info("reusing cached Ideogram 4 %s pipeline from %s on %s", quant.upper(), repo, device)
        return _pipeline

    unload()
    # Fail fast with a readable message if the card is too contended to fit the
    # model, instead of letting bitsandbytes hard-crash the worker (access
    # violation) mid-load. Checked AFTER unload() so we count our own freed VRAM.
    _require_free_vram(quant)
    log.info("loading Ideogram 4 %s from %s on %s (%.1f GB free)",
             quant.upper(), repo, device, torch.cuda.mem_get_info()[0] / (1024 ** 3) if torch.cuda.is_available() else 0.0)
    try:
        import bitsandbytes  # noqa: F401
    except Exception as e:
        if quant == "nf4":
            raise RuntimeError(
                "Ideogram 4 NF4 requires bitsandbytes. Install it in python\\venv with "
                "`python -m pip install bitsandbytes>=0.49.2`."
            ) from e

    if quant == "q4_k":
        if os.environ.get("KRAKEN_IDEOGRAM4_ALLOW_Q4K") != "1":
            raise RuntimeError(
                "Ideogram 4 Q4_K is guarded because the current in-process loader crashed "
                "during cold load on this RTX 3090 stack. Q4_K is a quality/memory experiment, "
                "not the speed path. Use Ideogram 4 NF4 for normal local generation, or set "
                "KRAKEN_IDEOGRAM4_ALLOW_Q4K=1 only for deliberate isolated testing."
            )
        try:
            import gguf  # noqa: F401
        except Exception as e:
            raise RuntimeError(
                "Ideogram 4 Q4_K requires the gguf package. Install it in python\\venv with "
                "`python -m pip install gguf`."
            ) from e
        from gguf import GGUFReader

        bf16_gb = _q4k_dequantized_bf16_gb()
        log.warning(
            "Ideogram 4 Q4_K uses Transformer Lab's reference GGUF loader path. "
            "It streams dequantized tensors into the DiT, but the active PyTorch "
            "linears are bf16 after swap; decoded tensor volume is about %.1f GB. "
            "Transformer Lab reports ~203 s/img at 48 steps on RTX 3090, slower "
            "than NF4 but higher quality.",
            bf16_gb,
        )
        base_repo = _q4k_base_repo()
        log.info("loading Ideogram 4 base for Q4_K swap from %s", base_repo)
        pipe = Ideogram4Pipeline.from_pretrained(
            config=Ideogram4PipelineConfig(weights_repo=base_repo),
            device=device,
            dtype=_dtype(),
        )
        log.info("streaming Ideogram 4 Q4_K GGUF tensors from %s", LOCAL_Q4K_GGUF)
        reader = GGUFReader(str(LOCAL_Q4K_GGUF))
        tensor_index = {tensor.name: tensor for tensor in reader.tensors}
        cond = _swap_branch_from_gguf_reader(pipe.conditional_transformer, tensor_index, "cond", device)
        uncond = _swap_branch_from_gguf_reader(pipe.unconditional_transformer, tensor_index, "uncond", device)
        del tensor_index, reader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if cond == 0 or uncond == 0:
            raise RuntimeError(
                f"Ideogram 4 Q4_K swap did not replace expected linears (cond={cond}, uncond={uncond})."
            )
        log.info("Ideogram 4 Q4_K swapped %d conditional and %d unconditional linears", cond, uncond)
        _pipeline = pipe
        _pipeline_key = key
        return pipe

    if quant == "int8_fused":
        if os.environ.get("KRAKEN_IDEOGRAM4_ALLOW_INT8_FUSED") != "1":
            raise RuntimeError(
                "Ideogram 4 INT8 Fused is installed, but it is guarded because this Windows "
                "RTX 3090 stack hard-crashed during three load paths: FP8 base load, NF4 "
                "scaffold swap, and NF4 swap with text/VAE CPU staging. The files are present "
                "for future isolated-process work. Set KRAKEN_IDEOGRAM4_ALLOW_INT8_FUSED=1 "
                "only if you intentionally want to retry the experimental loader."
            )
        try:
            import triton  # noqa: F401
        except Exception as e:
            raise RuntimeError(
                "Ideogram 4 INT8 Fused requires Triton. Install a torch-compatible Triton build."
            ) from e
        fused_module = _load_fused_int8_module()
        base_repo = str(LOCAL_NF4_REPO) if (LOCAL_NF4_REPO / "model_index.json").exists() else _fp8_base_repo()
        log.info("loading Ideogram 4 base for fused INT8 swap from %s", base_repo)
        pipe = Ideogram4Pipeline.from_pretrained(
            config=Ideogram4PipelineConfig(weights_repo=base_repo),
            device=device,
            dtype=_dtype(),
        )
        log.info("installing Ideogram 4 fused INT8 weights from %s", LOCAL_INT8_WEIGHTS)
        using_nf4_scaffold = str(base_repo).lower().endswith("\\nf4") or str(base_repo).lower().endswith("/nf4")
        if using_nf4_scaffold:
            log.info("moving Ideogram text encoder and VAE to CPU during INT8 swap")
            pipe.text_encoder.to("cpu")
            pipe.autoencoder.to("cpu")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            fused, protected = _install_fused_int8(pipe, LOCAL_INT8_WEIGHTS, fused_module)
            log.info("moving Ideogram text encoder and VAE back to %s after INT8 swap", device)
            pipe.text_encoder.to(device)
            pipe.autoencoder.to(device)
        else:
            fused, protected = fused_module.load_fused_int8(pipe, str(LOCAL_INT8_WEIGHTS))
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if fused == 0:
            raise RuntimeError("Ideogram 4 INT8 Fused did not replace any linears.")
        log.info("Ideogram 4 INT8 Fused installed on %d linears (+%d bf16 protected)", fused, protected)
        _pipeline = pipe
        _pipeline_key = key
        return pipe

    pipe = Ideogram4Pipeline.from_pretrained(
        config=Ideogram4PipelineConfig(weights_repo=repo),
        device=device,
        dtype=_dtype(),
    )
    _pipeline = pipe
    _pipeline_key = key
    return pipe


class _IdeogramCancelled(Exception):
    pass


def _official_magic_prompt(prompt: str, width: int, height: int) -> str:
    _ensure_ideogram_import_path()
    from ideogram4 import DEFAULT_MAGIC_PROMPT, MAGIC_PROMPTS, aspect_ratio_from_size

    api_key = os.environ.get("IDEOGRAM_API_KEY") or os.environ.get("MAGIC_PROMPT_API_KEY")
    if not api_key:
        raise RuntimeError("No Ideogram/OpenRouter magic prompt API key is configured.")

    aspect_ratio = aspect_ratio_from_size(width, height)
    magic = MAGIC_PROMPTS[DEFAULT_MAGIC_PROMPT](api_key=api_key)  # type: ignore[call-arg]
    log.info("expanding Ideogram prompt with %s for %s", DEFAULT_MAGIC_PROMPT, aspect_ratio)
    return magic.expand(prompt, aspect_ratio=aspect_ratio)


def expand_prompt(
    prompt: str,
    negative: str | None,
    width: int,
    height: int,
    mode: str = "local",
    *,
    pretty: bool = False,
) -> dict[str, str]:
    """Expand an Ideogram prompt without loading the image model."""
    raw = (prompt or "").strip()
    mode = (mode or "local").strip().lower()
    if mode in {"raw", "off", "none", "plain", "plain_text"}:
        out = raw
        return {"mode": "raw", "prompt": out, "pretty_prompt": out}

    if raw.startswith("{"):
        try:
            out = json.dumps(json.loads(raw), separators=(",", ":"), ensure_ascii=False)
            pretty_out = json.dumps(json.loads(out), indent=2, ensure_ascii=False)
            return {"mode": "json", "prompt": out, "pretty_prompt": pretty_out}
        except Exception:
            return {"mode": "json", "prompt": raw, "pretty_prompt": raw}

    if mode in {"api", "official", "ideogram"}:
        try:
            out = _official_magic_prompt(raw, width, height)
            pretty_out = json.dumps(json.loads(out), indent=2, ensure_ascii=False) if pretty else out
            return {"mode": "official", "prompt": out, "pretty_prompt": pretty_out}
        except Exception as e:
            log.warning("Ideogram hosted magic prompt failed; falling back to local JSON: %s", e)

    return preview_local_magic_prompt(raw, negative, width, height)


def _prepare_prompt(prompt: str, negative: str | None, width: int, height: int, p: dict[str, Any]) -> str:
    raw = (prompt or "").strip()
    if not raw:
        return raw
    if raw.startswith("{"):
        return raw

    _ensure_ideogram_import_path()
    prompt_mode = str(
        p.get("ideogram_magic_mode")
        or os.environ.get("KRAKEN_IDEOGRAM_PROMPT_MODE", "local")
    ).strip().lower()
    magic_enabled = bool(p.get("ideogram_magic", True))
    if not magic_enabled:
        prompt_mode = "raw"
    if prompt_mode in {"raw", "plain", "plain_text"}:
        log.warning(
            "Ideogram raw non-JSON prompt requested; converting to Kraken local JSON to avoid placeholder output."
        )
        return expand_prompt(raw, negative, width, height, "local")["prompt"]

    return expand_prompt(raw, negative, width, height, prompt_mode)["prompt"]


def _preset_for_steps(steps: int):
    _ensure_ideogram_import_path()
    from ideogram4 import PRESETS

    if steps <= 12:
        return PRESETS["V4_TURBO_12"]
    if steps <= 20:
        return PRESETS["V4_DEFAULT_20"]
    return PRESETS["V4_QUALITY_48"]


def _thumbnail_b64(img: Image.Image, max_side: int = 320) -> str:
    thumb = img.copy()
    thumb.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    thumb.save(buf, format="JPEG", quality=80)
    return b64encode(buf.getvalue()).decode("ascii")


# ---- Adaptive velocity-cache (the 3090 speed win, measured 2026-06-17) --------
# Ideogram 4 runs TWO transformer passes per sampling step (asymmetric CFG). The
# velocity `v` it produces changes a lot in the first few steps (composition) and
# the last few (detail polish), but barely changes through the middle. The cache
# below recomputes the two passes only when the latent has moved more than
# `threshold` (relative L1) since the last compute — always recomputing the first
# `warmup` and last `cooldown` steps. Measured on an RTX 3090 (NF4, 1024², 48
# steps): 4.3 min -> ~2.0 min at near-identical quality (21/48 forwards at
# threshold 0.08). threshold=0 reproduces the exact full-compute trajectory.
#
# Speed modes (UI):
#   max  -> threshold 0     : every step computed, reference quality   (~4.3 min)
#   high -> threshold 0.08  : near-identical quality, ~2x faster        (~2.0 min)  [default]
#   fast -> threshold 0.12  : great drafts, ~2.7x faster                (~1.6 min)

def _ideogram_speed_params(mode: str | None) -> dict[str, float]:
    m = (mode or "high").strip().lower()
    if m in ("max", "full", "quality", "off", "none"):
        return {"threshold": 0.0, "warmup": 0, "cooldown": 0}
    if m in ("fast", "draft", "turbo"):
        return {"threshold": 0.12, "warmup": 4, "cooldown": 4}
    return {"threshold": 0.08, "warmup": 4, "cooldown": 4}  # high (default)


@torch.no_grad()
def _generate_cached(pipe, prompt, height, width, preset, seed, *,
                     threshold, warmup, cooldown, progress_callback=None):
    """Ideogram denoise loop with the adaptive velocity-cache described above.

    Mirrors ideogram4.pipeline_ideogram4.Ideogram4Pipeline.__call__ exactly when
    threshold == 0; with threshold > 0 it skips the two transformer passes on
    low-change steps and reuses the cached velocity. Returns a list[PIL.Image].
    """
    from ideogram4.scheduler import get_schedule_for_resolution, make_step_intervals

    dev = pipe.device
    ns = preset.num_steps
    schedule = get_schedule_for_resolution((height, width), known_mean=preset.mu, std=preset.std)
    si = make_step_intervals(ns).to(dev)
    gw = torch.tensor(preset.guidance_schedule, dtype=torch.float32, device=dev)

    inp = pipe._build_inputs([prompt], height=height, width=width)
    nit = inp["num_image_tokens"]
    gh, gw_grid = inp["grid_h"], inp["grid_w"]
    mt = inp["max_text_tokens"]
    ld = pipe.conditional_transformer.config.in_channels

    llm = pipe._encode_text(inp["token_ids"], inp["text_position_ids"], inp["indicator"])
    npos = inp["position_ids"][:, mt:]
    nseg = inp["segment_ids"][:, mt:]
    nind = inp["indicator"][:, mt:]
    nllm = torch.zeros(1, nit, llm.shape[-1], dtype=llm.dtype, device=dev)

    g = torch.Generator(device=dev)
    if seed is not None:
        g.manual_seed(int(seed) & 0x7FFFFFFF)
    z = torch.randn(1, nit, ld, dtype=torch.float32, device=dev, generator=g)
    tp = torch.zeros(1, mt, ld, dtype=torch.float32, device=dev)

    cached_v = None
    z_at_last = None
    for i in range(ns - 1, -1, -1):
        fidx = ns - 1 - i
        tv = float(schedule(si[i + 1].unsqueeze(0)).item())
        sv = float(schedule(si[i].unsqueeze(0)).item())
        t = torch.full((1,), tv, dtype=torch.float32, device=dev)

        if cached_v is None or threshold <= 0 or fidx < warmup or fidx >= ns - cooldown:
            recompute = True
        else:
            rel = (z - z_at_last).abs().mean() / (z_at_last.abs().mean() + 1e-6)
            recompute = bool(rel.item() > threshold)

        if recompute:
            pz = torch.cat([tp, z], dim=1)
            pv = pipe.conditional_transformer(
                llm_features=llm, x=pz, t=t,
                position_ids=inp["position_ids"], segment_ids=inp["segment_ids"], indicator=inp["indicator"],
            )[:, mt:]
            nv = pipe.unconditional_transformer(
                llm_features=nllm, x=z, t=t,
                position_ids=npos, segment_ids=nseg, indicator=nind,
            )
            v = gw[i] * pv + (1.0 - gw[i]) * nv
            cached_v = v
            z_at_last = z.clone()
        else:
            v = cached_v

        z = z + v * (sv - tv)
        if progress_callback is not None:
            progress_callback(ns - i, ns)

    return pipe._decode(z, grid_h=gh, grid_w=gw_grid)


def _run_inprocess(job) -> dict:
    from pipelines import flux as flux_mod, krea2 as krea2_mod, sdxl as sdxl_mod, wan_video as wan_mod, z_image as zi_mod

    p = job.params
    cached_before, quant, _, _, _ = _pipeline_cache_state(p.get("diffusion_model"))

    sdxl_mod.unload()
    flux_mod.unload()
    krea2_mod.unload()
    zi_mod.unload()
    wan_mod.unload()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    width = int(p.get("width") or 1024)
    height = int(p.get("height") or 1024)
    width -= width % 16
    height -= height % 16
    steps = int(p.get("steps") or 20)
    count = int(p.get("count") or 1)
    seed = p.get("seed")

    preset = _preset_for_steps(steps)
    prompt = _prepare_prompt(p.get("prompt", ""), p.get("negative") or "", width, height, p)

    job.progress.total_steps = preset.num_steps
    job.progress.total_images = count
    job.progress.message = (
        f"Reusing cached Ideogram 4 {quant.upper()}"
        if cached_before
        else f"Loading Ideogram 4 {quant.upper()}"
    )
    job.emit({
        "type": "progress",
        "step": 0,
        "total_steps": preset.num_steps,
        "image_index": 0,
        "total_images": count,
        "message": job.progress.message,
    })

    t0 = time.time()
    pipe = _ensure_pipeline(p.get("diffusion_model"))
    load_elapsed = time.time() - t0
    log.info(
        "Ideogram 4 pipeline ready in %.1fs (%s)",
        load_elapsed,
        "cache hit" if cached_before else "cold load",
    )
    job.progress.message = (
        f"Ideogram 4 cached pipeline ready in {load_elapsed:.1f}s"
        if cached_before
        else f"Ideogram 4 loaded in {load_elapsed:.1f}s"
    )
    job.emit({
        "type": "progress",
        "step": 0,
        "total_steps": preset.num_steps,
        "image_index": 0,
        "total_images": count,
        "message": job.progress.message,
    })

    out_dir = OUTPUTS_ROOT / time.strftime("%Y-%m-%d")
    out_dir.mkdir(parents=True, exist_ok=True)
    base_stem = build_output_stem(p, job.id)

    saved: list[dict] = []
    for i in range(count):
        if job.cancel.is_set():
            break
        per_seed = (int(seed if seed is not None else torch.seed()) + i) & 0x7FFFFFFF
        job.progress.image_index = i
        job.progress.message = "Generating with Ideogram 4"
        gen_start = time.time()
        first_step_seen = False
        last_step_log = gen_start

        def _progress(done: int, total: int) -> None:
            nonlocal first_step_seen, last_step_log
            if job.cancel.is_set():
                raise _IdeogramCancelled()
            now = time.time()
            if not first_step_seen:
                first_step_seen = True
                last_step_log = now
                log.info("Ideogram first visible step after %.1fs", now - gen_start)
            elif done == total or done % max(1, total // 6) == 0:
                log.info(
                    "Ideogram step %d/%d after %.1fs total (%.1fs since previous report)",
                    done,
                    total,
                    now - gen_start,
                    now - last_step_log,
                )
                last_step_log = now
            job.progress.step = done
            job.progress.total_steps = total
            job.progress.image_index = i
            job.progress.total_images = count
            job.progress.message = f"Generating with Ideogram 4 - step {done}/{total}"
            job.emit({
                "type": "progress",
                "step": done,
                "total_steps": total,
                "image_index": i,
                "total_images": count,
                "message": job.progress.message,
            })

        try:
            # Adaptive velocity-cache for the big 3090 speed win. "max" mode
            # (threshold 0) reproduces the exact full-compute trajectory; "high"
            # (default) and "fast" skip low-change steps. See _generate_cached.
            sp = _ideogram_speed_params(p.get("ideogram_speed_mode"))
            images = _generate_cached(
                pipe, prompt, height, width, preset, per_seed,
                threshold=sp["threshold"], warmup=int(sp["warmup"]), cooldown=int(sp["cooldown"]),
                progress_callback=_progress,
            )
        except _IdeogramCancelled:
            break
        log.info(
            "Ideogram image %d/%d seed %d generated in %.1fs",
            i + 1,
            count,
            per_seed,
            time.time() - gen_start,
        )
        img = images[0]

        fname = f"{base_stem}-{i:02d}.png"
        fpath = out_dir / fname
        save_png_with_metadata(img, fpath, p, per_seed)
        rel_path = fpath.relative_to(OUTPUTS_ROOT).as_posix()
        entry = {
            "path": str(fpath),
            "filename": fname,
            "seed": per_seed,
            "width": width,
            "height": height,
            "rel_path": rel_path,
        }

        # Optional upscale — runs on the finished image, so it's arch-agnostic.
        # This is the recommended path for big Ideogram output: render native
        # ~1 MP (fast, fits 24 GB), then upscale 2x/4x instead of native-2K
        # (which spills VRAM and crawls).
        if p.get("upscale_enabled") and p.get("upscale_model"):
            try:
                from pipelines import upscale_esrgan
                job.progress.message = "Upscaling Ideogram 4 image"
                job.emit({"type": "progress", "step": preset.num_steps, "total_steps": preset.num_steps,
                          "image_index": i, "total_images": count, "message": "Upscaling..."})
                up = upscale_esrgan.upscale(img, p["upscale_model"], float(p.get("upscale_factor", 2.0)))
                up_path = out_dir / f"{base_stem}-{i:02d}-up.png"
                save_png_with_metadata(up, up_path, p, int(per_seed))
                entry["upscaled_path"] = str(up_path)
                img = up  # preview the upscaled version
            except Exception as e:
                log.warning("Ideogram upscale failed: %s", e)

        saved.append(entry)
        job.progress.step = preset.num_steps
        job.progress.message = "Saving Ideogram 4 image"
        job.emit({
            "type": "progress",
            "step": preset.num_steps,
            "total_steps": preset.num_steps,
            "image_index": i,
            "total_images": count,
            "message": "Saving Ideogram 4 image",
        })
        job.emit({
            "type": "image",
            "image_index": i,
            "path": str(fpath),
            "rel_path": rel_path,
            "filename": fname,
            "seed": per_seed,
            "preview_b64": _thumbnail_b64(img),
        })

    return {
        "kind": "image",
        "count_requested": count,
        "count_produced": len(saved),
        "outputs": saved,
        "output_dir": str(out_dir),
        "pipeline_cached": cached_before,
        "pipeline_ready_seconds": load_elapsed,
    }


def _relay_worker_event(job, event: dict) -> dict | None:
    kind = event.get("type")
    if kind == "progress":
        job.progress.step = int(event.get("step") or 0)
        job.progress.total_steps = int(event.get("total_steps") or event.get("total") or 0)
        job.progress.image_index = int(event.get("image_index") or 0)
        job.progress.total_images = int(event.get("total_images") or 0)
        job.progress.message = str(event.get("message") or "")
        job.emit(event)
        return None
    if kind == "image":
        job.emit(event)
        return None
    if kind == "done":
        result = event.get("result")
        return result if isinstance(result, dict) else {}
    if kind == "error":
        err = str(event.get("error") or "Ideogram worker failed")
        if err.startswith("RuntimeError: "):
            err = err.removeprefix("RuntimeError: ")
        raise RuntimeError(err)
    if kind == "log":
        log.info("Ideogram worker: %s", event.get("message") or "")
        return None
    job.emit(event)
    return None


def _start_worker_process() -> tuple[subprocess.Popen, queue.Queue[str | None]]:
    global _worker_proc, _worker_lines
    if not WORKER_SCRIPT.exists():
        raise FileNotFoundError(f"Ideogram worker script not found at {WORKER_SCRIPT}")

    if _worker_proc is not None and _worker_proc.poll() is None and _worker_lines is not None:
        return _worker_proc, _worker_lines

    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    cmd = [sys.executable, "-u", str(WORKER_SCRIPT), "--server"]
    log.info("starting persistent Ideogram worker")
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.PIPE,
        text=True,
        bufsize=1,
        creationflags=creationflags,
    )
    lines: queue.Queue[str | None] = queue.Queue()

    def _reader() -> None:
        assert proc.stdout is not None
        try:
            for line in proc.stdout:
                lines.put(line)
        finally:
            lines.put(None)

    threading.Thread(target=_reader, daemon=True, name="ideogram-worker-reader").start()
    _worker_proc = proc
    _worker_lines = lines
    return proc, lines


def _run_with_persistent_worker(job) -> dict:
    global _worker_proc, _worker_lines
    with _worker_lock:
        proc, lines = _start_worker_process()
        if proc.stdin is None:
            raise RuntimeError("Ideogram worker stdin is not available.")

        proc.stdin.write(json.dumps({"id": job.id, "params": job.params}, ensure_ascii=False) + "\n")
        proc.stdin.flush()

        result: dict | None = None
        last_output = ""
        while True:
            if job.cancel.is_set() and proc.poll() is None:
                log.info("terminating persistent Ideogram worker for cancelled job %s", job.id)
                unload()
                return {
                    "kind": "image",
                    "count_requested": int(job.params.get("count") or 1),
                    "count_produced": 0,
                    "outputs": [],
                    "cancelled": True,
                }

            try:
                line = lines.get(timeout=0.5)
            except queue.Empty:
                if proc.poll() is not None:
                    break
                continue

            if line is None:
                if proc.poll() is not None:
                    break
                continue

            text = line.strip()
            if not text:
                continue
            last_output = text[-1000:]
            try:
                event = json.loads(text)
            except json.JSONDecodeError:
                log.info("Ideogram worker: %s", text)
                continue
            maybe_result = _relay_worker_event(job, event)
            if maybe_result is not None:
                result = maybe_result
                return result

        code = proc.poll()
        _worker_proc = None
        _worker_lines = None
        raise RuntimeError(f"Ideogram worker exited with code {code}. Last output: {last_output}")


def _is_worker_hard_crash(error: Exception) -> bool:
    text = str(error).lower()
    return (
        "ideogram worker exited with code" in text
        and (
            "3221225477" in text
            or "-1073741819" in text
            or "0xc0000005" in text
            or "access violation" in text
        )
    )


def _emit_worker_retry(job, message: str) -> None:
    job.progress.step = 0
    job.progress.total_steps = 0
    job.progress.message = message
    job.emit({
        "type": "progress",
        "step": 0,
        "total_steps": 0,
        "image_index": 0,
        "total_images": int(job.params.get("count") or 1),
        "message": message,
    })


def _run_one_shot_worker(job, params: dict | None = None, *, label: str = "") -> dict:
    worker_params = dict(params or job.params)
    tmp_dir = ROOT / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    safe_label = "".join(ch if ch.isalnum() else "-" for ch in label).strip("-")
    suffix = f"-{safe_label}" if safe_label else ""
    spec_path = tmp_dir / f"ideogram-job-{job.id}{suffix}.json"
    spec_path.write_text(json.dumps({"id": job.id, "params": worker_params}, ensure_ascii=False), encoding="utf-8")

    cmd = [sys.executable, "-u", str(WORKER_SCRIPT), str(spec_path)]
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    log.info("starting Ideogram worker for job %s%s", job.id, f" ({label})" if label else "")
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        text=True,
        bufsize=1,
        creationflags=creationflags,
    )

    lines: queue.Queue[str | None] = queue.Queue()

    def _reader() -> None:
        assert proc.stdout is not None
        try:
            for line in proc.stdout:
                lines.put(line)
        finally:
            lines.put(None)

    reader = threading.Thread(target=_reader, daemon=True, name=f"ideogram-worker-{job.id[:8]}")
    reader.start()

    result: dict | None = None
    last_output = ""
    try:
        while True:
            if job.cancel.is_set() and proc.poll() is None:
                log.info("terminating Ideogram worker for cancelled job %s", job.id)
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                return {
                    "kind": "image",
                    "count_requested": int(worker_params.get("count") or 1),
                    "count_produced": 0,
                    "outputs": [],
                    "cancelled": True,
                }

            try:
                line = lines.get(timeout=0.5)
            except queue.Empty:
                if proc.poll() is not None and lines.empty():
                    break
                continue

            if line is None:
                if proc.poll() is not None:
                    break
                continue

            text = line.strip()
            if not text:
                continue
            last_output = text[-1000:]
            try:
                event = json.loads(text)
            except json.JSONDecodeError:
                log.info("Ideogram worker: %s", text)
                continue
            maybe_result = _relay_worker_event(job, event)
            if maybe_result is not None:
                result = maybe_result

        code = proc.wait(timeout=5)
        if code != 0:
            raise RuntimeError(f"Ideogram worker exited with code {code}. Last output: {last_output}")
        if result is None:
            raise RuntimeError("Ideogram worker exited without a result.")
        return result
    finally:
        try:
            spec_path.unlink(missing_ok=True)
        except Exception:
            pass


def run(job) -> dict:
    """Run Ideogram in an isolated one-shot worker by default.

    Ideogram 4 NF4 can hard-crash the Python interpreter during bitsandbytes'
    4-bit module construction. A crash in-process takes the whole sidecar down,
    which loses the job websocket and makes the app look broken. The one-shot
    worker keeps that failure mode contained: a bad Ideogram load becomes a
    normal failed job while the sidecar survives. Set
    KRAKEN_IDEOGRAM4_INPROCESS=1 only when debugging the Ideogram package itself.
    Set KRAKEN_IDEOGRAM4_PERSISTENT_WORKER=1 to reuse a worker between jobs.
    """
    if os.environ.get("KRAKEN_IDEOGRAM4_INPROCESS") == "1" or os.environ.get("KRAKEN_IDEOGRAM4_USE_WORKER") == "0":
        return _run_inprocess(job)

    from pipelines import flux as flux_mod, krea2 as krea2_mod, sdxl as sdxl_mod, wan_video as wan_mod, z_image as zi_mod

    # Release other resident visual pipelines in the sidecar before the child grabs VRAM.
    sdxl_mod.unload()
    flux_mod.unload()
    krea2_mod.unload()
    zi_mod.unload()
    wan_mod.unload()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    _, quant, _, _, _ = _pipeline_cache_state(job.params.get("diffusion_model"))

    if os.environ.get("KRAKEN_IDEOGRAM4_PERSISTENT_WORKER") == "1":
        return _run_with_persistent_worker(job)

    attempts = max(1, int(os.environ.get("KRAKEN_IDEOGRAM4_WORKER_ATTEMPTS", "2") or "2"))
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            label = f"{quant}-attempt-{attempt}" if attempts > 1 else ""
            return _run_one_shot_worker(job, label=label)
        except RuntimeError as e:
            last_error = e
            if attempt >= attempts or not _is_worker_hard_crash(e):
                break
            log.warning("Ideogram worker hard-crashed on attempt %d/%d; retrying", attempt, attempts)
            _emit_worker_retry(job, f"Ideogram worker crashed during load; retrying ({attempt + 1}/{attempts})")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            time.sleep(3)

    fallback_enabled = os.environ.get("KRAKEN_IDEOGRAM4_FALLBACK_FP8", "0") == "1"
    if (
        fallback_enabled
        and quant == "nf4"
        and (LOCAL_FP8_REPO / "model_index.json").exists()
        and last_error is not None
        and _is_worker_hard_crash(last_error)
    ):
        fallback_params = dict(job.params)
        fallback_params["diffusion_model"] = "Ideogram 4 FP8 (local fallback)"
        log.warning("Ideogram NF4 worker hard-crashed; falling back to local FP8")
        _emit_worker_retry(job, "Ideogram NF4 crashed during load; falling back to FP8")
        return _run_one_shot_worker(job, fallback_params, label="fp8-fallback")

    assert last_error is not None
    raise last_error
