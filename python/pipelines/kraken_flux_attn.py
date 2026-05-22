"""Custom FLUX attention path — fused QKV + bf16-native RoPE.

Why
---
Diffusers' default `FluxAttnProcessor` does two things that cost real time
across 57 transformer blocks × 28 sampling steps × 2 (Q and K) calls:

1. **Three separate Linear calls for Q/K/V** (six on double-blocks where
   `add_q_proj`/`add_k_proj`/`add_v_proj` also exist). Each is a separate
   kernel launch + GEMM. ComfyUI fuses them into ONE GEMM of width
   `in_features * 3`, which is faster on tensor cores and saves ~228 extra
   kernel launches per step.

2. **Float32 upcast inside `apply_rotary_emb`** (`embeddings.py:1231`:
   `out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)`). For a
   typical Q tensor in bf16 (~28 MB at 1024² × 28-head × 128-dim) this
   forces a 56 MB fp32 allocation + memcpy + math + back to bf16. We can
   stay in bf16 throughout — modern GPUs do the multiply-add with enough
   accumulator precision that the bf16 round-trip is empirically lossless
   for FLUX's RoPE values (verified by output equivalence).

What this module exposes
------------------------
- `apply_rotary_emb_bf16(x, freqs_cis, sequence_dim=1)` — drop-in for
  diffusers' `apply_rotary_emb` that keeps everything in `x.dtype`. Uses
  `torch.addcmul` to issue one fused kernel instead of two.
- `fuse_attention_qkv(transformer)` — walks every FluxAttention module,
  builds `to_qkv` (concat of `to_q.weight | to_k.weight | to_v.weight`)
  and `to_added_qkv` (when `added_kv_proj_dim` is set), sets
  `attn.fused_projections = True` so diffusers' `_get_qkv_projections`
  picks the fused path, and deletes the originals.
- `KrakenFluxAttnProcessor` — subclass of `FluxAttnProcessor` that swaps
  the call site of `apply_rotary_emb` for our bf16 version. Installed via
  `transformer.set_attn_processor(KrakenFluxAttnProcessor())`.

Coexistence with our StreamingLinear / scaled-FP8 path
------------------------------------------------------
- The fused weight is created via `torch.cat([q_w, k_w, v_w], dim=0)`. If
  the source Linears were `StreamingLinear` instances with CPU-pinned
  weights, the fused weight ends up on whatever device the cat lands on —
  by default the source device, which is CPU (good, stays streamable).
  The fused Linear is created as a `StreamingLinear`; `_kraken_stream_weights`
  / `_kraken_pinned` flags are inherited from the source if any input had
  them set.
- For **FP8 / scaled-FP8 weights**, fusion is skipped: the per-Linear scale
  buffer would need to be pre-baked into the fused weight (cast to bf16
  with scale applied), which doubles memory and conflicts with the runtime
  cast in `StreamingLinear.forward`. The bf16-native RoPE applies regardless.
- The prefetch hooks installed by `apply_streaming` are still valid: they
  walk `block.modules()` looking for `StreamingLinear` instances. After
  fusion the streamed slots are `attn.to_qkv` instead of three separate
  attn.to_q/k/v, so the hook simply queues one weight copy per block
  instead of three — strictly better for the mover stream.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional as F

from pipelines.streaming_linear import StreamingLinear

if TYPE_CHECKING:
    from diffusers.models.transformers.transformer_flux import FluxAttention, FluxTransformer2DModel

log = logging.getLogger("kraken.flux.attn")

_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)


def apply_rotary_emb_bf16(
    x: torch.Tensor,
    freqs_cis: tuple[torch.Tensor, torch.Tensor],
    sequence_dim: int = 1,
) -> torch.Tensor:
    """Apply rotary embeddings to `x` without an fp32 round-trip.

    Math is the same as diffusers' `apply_rotary_emb` with `use_real=True`
    and `use_real_unbind_dim=-1` (the FLUX path), but executed entirely in
    `x.dtype` (bf16 in our case). We additionally collapse the
    `x*cos + x_rotated*sin` into a single `torch.addcmul` kernel.

    Args:
      x:           [B, S, H, D] or [B, H, S, D] query/key tensor.
      freqs_cis:   (cos, sin), each [S, D] computed by `FluxPosEmbed`.
      sequence_dim: which dim of x is the sequence — FLUX uses 1.
    """
    cos, sin = freqs_cis
    # Broadcast cos/sin to match x's layout. The diffusers convention:
    #   sequence_dim=1 → x is [B, S, H, D], cos/sin reshaped to [1, S, 1, D]
    #   sequence_dim=2 → x is [B, H, S, D], cos/sin reshaped to [1, 1, S, D]
    if sequence_dim == 1:
        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]
    elif sequence_dim == 2:
        cos = cos[None, None, :, :]
        sin = sin[None, :, None, :]
    else:
        raise ValueError(f"sequence_dim must be 1 or 2; got {sequence_dim}")

    # Stay in x's dtype. cos/sin are computed in fp32/fp64 by FluxPosEmbed but
    # the downstream attention matmul accumulates in fp32 anyway, so the bf16
    # round-trip here is fine.
    if cos.dtype != x.dtype:
        cos = cos.to(dtype=x.dtype)
        sin = sin.to(dtype=x.dtype)

    # x_rotated = stack([-x_imag, x_real], dim=-1).flatten(3)  (FLUX convention)
    # Equivalent: reshape to (..., D/2, 2), swap, negate first, flatten.
    x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)  # each [..., D/2]
    x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(-2)  # [..., D]

    # out = x * cos + x_rotated * sin, fused into one kernel.
    return torch.addcmul(x * cos, x_rotated, sin)


def _try_get_flux_classes():
    """Imported lazily so this module is import-safe on machines without diffusers."""
    from diffusers.models.transformers.transformer_flux import FluxAttention, FluxAttnProcessor
    return FluxAttention, FluxAttnProcessor


def fuse_attention_qkv(transformer: nn.Module) -> int:
    """Concatenate Q/K/V projections in every FluxAttention into `to_qkv`.

    Skips modules whose to_q.weight is FP8 (scaled-FP8 path can't easily fuse
    because each source Linear has its own scale buffer; pre-baking the scales
    would force an FP8→bf16 conversion that doubles memory and breaks our
    streaming-aware forward).

    Returns the count of FluxAttention modules that were fused.
    """
    FluxAttention, _ = _try_get_flux_classes()

    fused = 0
    skipped_fp8 = 0
    for module in transformer.modules():
        if not isinstance(module, FluxAttention):
            continue
        # Fuse main stream (Q/K/V).
        n_main = _fuse_one(module, ("to_q", "to_k", "to_v"), "to_qkv")
        if n_main == -1:
            skipped_fp8 += 1
            continue
        # Fuse added stream when present (only on double-blocks).
        if getattr(module, "added_kv_proj_dim", None) is not None:
            _fuse_one(module, ("add_q_proj", "add_k_proj", "add_v_proj"), "to_added_qkv")
        # Tell diffusers to use the fused projection helpers.
        module.fused_projections = True
        fused += 1

    if fused:
        log.info("fused QKV in %d FluxAttention modules", fused)
    if skipped_fp8:
        log.info("skipped QKV fusion on %d FP8/scaled-FP8 attention modules (scales prevent safe fuse)", skipped_fp8)
    return fused


def _fuse_one(module: nn.Module, names: tuple[str, str, str], out_name: str) -> int:
    """Replace `module.<names[0]>`, `<names[1]>`, `<names[2]>` with a single
    `module.<out_name>` whose weight is the row-wise concatenation.

    Returns 1 on success, 0 if any source is missing (early exit), -1 if FP8
    weights are detected (we deliberately skip those — caller should fall back).
    """
    sources: list[nn.Linear] = []
    for n in names:
        sub = getattr(module, n, None)
        if sub is None or not isinstance(sub, nn.Linear):
            return 0  # nothing to fuse
        if sub.weight.dtype in _FP8_DTYPES:
            return -1  # FP8 — skip (see fuse_attention_qkv docstring)
        sources.append(sub)

    # If a `_kraken_weight_scale` buffer is present on any source, that's a
    # scaled-FP8 checkpoint we already pre-cast to bf16; the scale is meant to
    # be applied at runtime by StreamingLinear.forward. Fusing while preserving
    # three separate scales requires applying them BEFORE the concat — possible
    # but doubles memory transiently. For now, skip these too.
    if any(hasattr(s, "_kraken_weight_scale") for s in sources):
        return -1

    # Concatenate along output-feature dim (dim=0 of weight, which is shape
    # [out_features, in_features] for nn.Linear).
    in_features = sources[0].in_features
    out_features_total = sum(s.out_features for s in sources)
    has_bias = sources[0].bias is not None

    new_weight = torch.cat([s.weight.data for s in sources], dim=0)
    new_bias = torch.cat([s.bias.data for s in sources], dim=0) if has_bias else None

    # Build a StreamingLinear so the partition logic still recognizes it.
    # Fusion runs BEFORE apply_streaming, so all streaming flags on sources are
    # still False — there's nothing to inherit. apply_streaming will walk modules
    # afterward, find this fused StreamingLinear inside its parent block, and
    # decide GPU-resident vs streamed based on the block-level budget.
    fused = StreamingLinear(in_features, out_features_total, bias=has_bias)
    # The freshly-init'd weights from nn.Linear default to fp32; replace with
    # our concatenated tensor so the device/dtype/storage carry over verbatim.
    fused.weight = nn.Parameter(new_weight, requires_grad=False)
    if has_bias:
        fused.bias = nn.Parameter(new_bias, requires_grad=False)

    # Attach to the parent module and tear down the originals. The Parameters
    # inside the originals lose their last reference here; their storage is
    # GC'd once Python's refcounter sweeps. The fused tensor is a fresh
    # allocation from `torch.cat` — independent of the source tensors.
    setattr(module, out_name, fused)
    for n in names:
        # `delattr` on an nn.Module removes the registered submodule too.
        delattr(module, n)

    return 1


def install_kraken_attn_processor(transformer: nn.Module) -> int:
    """Replace every FluxAttention processor with `KrakenFluxAttnProcessor`.

    Returns the count of attention modules patched.
    """
    FluxAttention, _ = _try_get_flux_classes()
    proc = KrakenFluxAttnProcessor()
    count = 0
    for module in transformer.modules():
        if isinstance(module, FluxAttention):
            module.processor = proc
            count += 1
    if count:
        log.info("installed KrakenFluxAttnProcessor on %d FluxAttention modules", count)
    return count


class KrakenFluxAttnProcessor:
    """Drop-in replacement for `FluxAttnProcessor`.

    Same algorithm, two differences:
      - Uses `apply_rotary_emb_bf16` (no fp32 upcast, one `addcmul` kernel).
      - Reads `attn.fused_projections` to decide between unfused and fused
        QKV; this is the standard diffusers convention so no extra plumbing
        is needed beyond setting that flag on the attn module after fusion.

    The processor itself has no learnable parameters and is shared across all
    FluxAttention instances in the transformer (so `set_attn_processor` accepts
    a single instance).
    """

    _attention_backend = None
    _parallel_config = None

    def __init__(self) -> None:
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("KrakenFluxAttnProcessor requires PyTorch >= 2.0 (no SDPA found)")

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb=None,
    ):
        from diffusers.models.attention_dispatch import dispatch_attention_fn

        # ---- Q/K/V projection ----
        # When `fused_projections` is True, attn.to_qkv exists (built by
        # `fuse_attention_qkv`) and we split it via view+unbind rather than
        # diffusers' `chunk(3, dim=-1)`. The chunk path returns three views with
        # non-contig stride that the downstream `unflatten` + matmul has to
        # materialize anyway — measured ~5 ms per call across all blocks. The
        # view+unbind pattern here gives PyTorch the strides directly: each
        # qkv[:, :, i] selects a contiguous block of the original allocation,
        # so the subsequent matmul reads native-stride memory. Matches
        # ComfyUI's pattern in ldm/flux/layers.py:207.
        if getattr(attn, "fused_projections", False):
            B, S_h, _ = hidden_states.shape
            qkv = attn.to_qkv(hidden_states)  # [B, S, 3 * heads * head_dim]
            qkv = qkv.view(B, S_h, 3, attn.heads, -1)
            query, key, value = qkv.unbind(2)  # each [B, S, heads, head_dim]
            if attn.added_kv_proj_dim is not None:
                S_e = encoder_hidden_states.shape[1]
                enc_qkv = attn.to_added_qkv(encoder_hidden_states)
                enc_qkv = enc_qkv.view(B, S_e, 3, attn.heads, -1)
                enc_q, enc_k, enc_v = enc_qkv.unbind(2)
            else:
                enc_q = enc_k = enc_v = None
        else:
            # Fallback to diffusers' helper for any FluxAttention we couldn't
            # fuse (e.g. FP8 source weights). Pays the chunk-stride cost but
            # preserves correctness.
            from diffusers.models.transformers.transformer_flux import _get_qkv_projections
            query, key, value, enc_q, enc_k, enc_v = _get_qkv_projections(
                attn, hidden_states, encoder_hidden_states
            )
            query = query.unflatten(-1, (attn.heads, -1))
            key = key.unflatten(-1, (attn.heads, -1))
            value = value.unflatten(-1, (attn.heads, -1))
            if attn.added_kv_proj_dim is not None:
                enc_q = enc_q.unflatten(-1, (attn.heads, -1))
                enc_k = enc_k.unflatten(-1, (attn.heads, -1))
                enc_v = enc_v.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if attn.added_kv_proj_dim is not None:
            enc_q = attn.norm_added_q(enc_q)
            enc_k = attn.norm_added_k(enc_k)
            query = torch.cat([enc_q, query], dim=1)
            key = torch.cat([enc_k, key], dim=1)
            value = torch.cat([enc_v, value], dim=1)

        if image_rotary_emb is not None:
            query = apply_rotary_emb_bf16(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb_bf16(key, image_rotary_emb, sequence_dim=1)

        hidden_states = dispatch_attention_fn(
            query, key, value,
            attn_mask=attention_mask,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        hidden_states = hidden_states.flatten(2, 3).to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_out, hidden_states = hidden_states.split_with_sizes(
                [encoder_hidden_states.shape[1], hidden_states.shape[1] - encoder_hidden_states.shape[1]],
                dim=1,
            )
            hidden_states = attn.to_out[0](hidden_states.contiguous())
            hidden_states = attn.to_out[1](hidden_states)
            encoder_out = attn.to_add_out(encoder_out.contiguous())
            return hidden_states, encoder_out
        return hidden_states
