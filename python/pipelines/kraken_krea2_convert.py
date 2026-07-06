"""Convert ComfyUI-native Krea 2 checkpoints to the diffusers key layout.

Community single-file Krea 2 checkpoints (e.g. `krea2_*_fp8_scaled.safetensors`)
use ComfyUI/native module names — `blocks.N`, `txtfusion`, `first`, `last`,
`tmlp`, `tproj`, `txtmlp` — rather than the diffusers
`Krea2Transformer2DModel` names (`transformer_blocks.N`, `text_fusion`, `img_in`,
`final_layer`, `time_embed`, `time_mod_proj`, `txt_in`). This maps between them.

The "fp8_scaled" variants store each quantized Linear as a `float8_e4m3fn`
`<name>.weight` plus a per-tensor F32 `<name>.weight_scale`; the real weight is
`weight.to(compute) * weight_scale`. RMSNorm `.scale`, `mod.lin`, biases and the
full-precision (bf16) Linears (img_in/time/txt/final/projector) are stored
plainly. Verified against the official diffusers folder: norm scales and the
modulation tables match value-for-value (no zero-centering offset to apply).
"""
from __future__ import annotations

import re

# Block-internal native->diffusers leaf renames (shared by transformer_blocks
# and the two text-fusion block stacks).
_LEAF_RENAMES = (
    ("attn.wq.", "attn.to_q."),
    ("attn.wk.", "attn.to_k."),
    ("attn.wv.", "attn.to_v."),
    ("attn.wo.", "attn.to_out.0."),
    ("attn.gate.", "attn.to_gate."),
    ("attn.qknorm.qnorm.scale", "attn.norm_q.weight"),
    ("attn.qknorm.knorm.scale", "attn.norm_k.weight"),
    ("mlp.gate.", "ff.gate."),
    ("mlp.up.", "ff.up."),
    ("mlp.down.", "ff.down."),
    ("prenorm.scale", "norm1.weight"),
    ("postnorm.scale", "norm2.weight"),
)

# Exact top-level renames (no numeric index inside).
_TOP_RENAMES = {
    "first.weight": "img_in.weight",
    "first.bias": "img_in.bias",
    "last.linear.weight": "final_layer.linear.weight",
    "last.linear.bias": "final_layer.linear.bias",
    "last.modulation.lin": "final_layer.scale_shift_table",  # reshaped below
    "last.norm.scale": "final_layer.norm.weight",
    "tmlp.0.weight": "time_embed.linear_1.weight",
    "tmlp.0.bias": "time_embed.linear_1.bias",
    "tmlp.2.weight": "time_embed.linear_2.weight",
    "tmlp.2.bias": "time_embed.linear_2.bias",
    "tproj.1.weight": "time_mod_proj.weight",
    "tproj.1.bias": "time_mod_proj.bias",
    # txt_in is Sequential[RMSNorm(0), Linear(1), GELU(2), Linear(3)] in the
    # native layout, so the projections live at indices 1 and 3 (not 0/2).
    "txtmlp.0.scale": "txt_in.norm.weight",
    "txtmlp.1.weight": "txt_in.linear_1.weight",
    "txtmlp.1.bias": "txt_in.linear_1.bias",
    "txtmlp.3.weight": "txt_in.linear_2.weight",
    "txtmlp.3.bias": "txt_in.linear_2.bias",
    "txtfusion.projector.weight": "text_fusion.projector.weight",
}


def _rename_leaf(s: str) -> str:
    for a, b in _LEAF_RENAMES:
        if a in s:
            return s.replace(a, b)
    return s


def map_key(key: str) -> str | None:
    """Map one native key to its diffusers name. `.weight_scale` is preserved as a
    sidecar suffix on the mapped weight path. Returns None for unknown keys."""
    # A quantized weight's scale is named `<weight>_scale` (i.e. `....weight_scale`),
    # distinct from RMSNorm `.scale`. Strip only the `_scale` suffix so the base is
    # `....weight` and the leaf renames (which expect e.g. `attn.wq.`) still match.
    scale_suffix = ""
    base = key
    if key.endswith("weight_scale"):
        scale_suffix = "_scale"
        base = key[: -len("_scale")]

    mapped: str | None = None
    if base in _TOP_RENAMES:
        mapped = _TOP_RENAMES[base]
    else:
        m = re.match(r"^blocks\.(\d+)\.(.+)$", base)
        if m:
            mapped = f"transformer_blocks.{m.group(1)}." + (
                "scale_shift_table" if m.group(2) == "mod.lin" else _rename_leaf(m.group(2))
            )
        else:
            m = re.match(r"^txtfusion\.(layerwise_blocks|refiner_blocks)\.(\d+)\.(.+)$", base)
            if m:
                mapped = f"text_fusion.{m.group(1)}.{m.group(2)}." + _rename_leaf(m.group(3))

    if mapped is None:
        return None
    return mapped + scale_suffix


def convert(native_sd: dict) -> tuple[dict, dict]:
    """Convert a native Krea 2 state dict.

    Returns `(weights, scales)`:
      - `weights`: diffusers-keyed tensors (fp8 weights kept fp8; `scale_shift_table`
        / `final_layer.scale_shift_table` reshaped from flat to (6/2, hidden)).
      - `scales`:  diffusers weight-path -> per-tensor scale (from `*.weight_scale`).
    Unknown keys are collected on the returned `weights` under `__unmapped__`.
    """
    weights: dict = {}
    scales: dict = {}
    unmapped: list[str] = []
    for k, v in native_sd.items():
        mk = map_key(k)
        if mk is None:
            unmapped.append(k)
            continue
        if mk.endswith("weight_scale"):
            scales[mk[: -len("_scale")]] = v  # diffusers weight path (….weight)
            continue
        if mk == "final_layer.scale_shift_table" and v.ndim == 1:
            v = v.reshape(2, -1)
        elif mk.endswith("scale_shift_table") and v.ndim == 1:
            v = v.reshape(6, -1)
        weights[mk] = v
    if unmapped:
        weights["__unmapped__"] = unmapped  # surfaced to the caller for diagnostics
    return weights, scales
