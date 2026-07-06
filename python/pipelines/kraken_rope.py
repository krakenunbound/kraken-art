"""Use comfy-kitchen's fused `apply_rope` kernel for FLUX rotary embeddings.

Why
---
diffusers' `apply_rotary_emb` is pure Python with an fp32 round-trip (`x.float()
* cos + x_rotated.float() * sin`).to(x.dtype)`). Profiling against ComfyUI on
the same hardware (see `Documentation/FLUX-SPEED-RESEARCH.md`) showed their
~330 ms/step advantage is largely due to a fused C++ RoPE kernel from the
`comfy-kitchen` package, which they fall back to a pure-Python version when
missing — and explicitly warn the user. We measured `comfy_kitchen.apply_rope`
at ~13 ms per call on FLUX-sized tensors.

What this module does
---------------------
- `install_ck_rope_patch()` monkey-patches `diffusers.models.embeddings.apply_rotary_emb`
  to call `comfy_kitchen.apply_rope1` when available. Falls back silently to
  the original diffusers implementation if comfy-kitchen is not installed.
- The trick is shape conversion: diffusers passes `freqs_cis` as a `(cos, sin)`
  tuple where each is `(S, D)`. comfy-kitchen wants a single tensor of shape
  `(1, 1, S, D/2, 2, 2)` encoding a 2×2 rotation matrix per position-feature
  pair.

Conventions
-----------
diffusers' rotation (with `use_real_unbind_dim=-1`, the FLUX path):
    out_real = real*cos - imag*sin
    out_imag = real*sin + imag*cos
which is the 2×2 rotation matrix [[cos, -sin], [sin, cos]].

comfy-kitchen's `_apply_rope1` (and the CUDA kernel matching it) computes:
    out[..., 0] = freqs[..., 0, 0]*real + freqs[..., 1, 0]*imag
    out[..., 1] = freqs[..., 0, 1]*real + freqs[..., 1, 1]*imag

So to express diffusers' convention in comfy-kitchen's freqs format, the 2×2
matrix is laid out so that:
    freqs[0, 0] = cos    freqs[1, 0] = -sin
    freqs[0, 1] = sin    freqs[1, 1] = cos

i.e. row-stacked `[[cos, sin], [-sin, cos]]` — note this is the TRANSPOSE of
the standard rotation matrix, because comfy-kitchen indexes the i/j of the
matrix differently than naive math notation.
"""
from __future__ import annotations

import logging
from typing import Any

import torch

log = logging.getLogger("kraken.rope")

_PATCH_APPLIED = False
_ORIGINAL_APPLY_ROTARY_EMB: Any = None
_CK_APPLY_ROPE1: Any = None

# Cache: id(cos) → (cos_ref, ck_freqs_cis). Keyed by Python id() because
# diffusers' FluxPosEmbed returns the SAME cos/sin tensors for every block
# call within a gen (computed once at the start of the transformer forward,
# then passed through to every attn). 57 blocks × 28 steps × 2 (Q + K) calls
# all share one (cos, sin) → 3192 cache hits per gen.
#
# `cos_ref` is held to keep the tensor alive (id() reuse is otherwise possible).
# Cache size is implicitly bounded — diffusers calls FluxPosEmbed once per
# transformer forward, so at most ~28 unique freqs_cis per session. The cache
# is small even without explicit eviction.
_freqs_cache: dict[int, tuple[Any, torch.Tensor]] = {}


def _build_ck_freqs_cis(cos: torch.Tensor, sin: torch.Tensor, sequence_dim: int) -> torch.Tensor:
    """Convert diffusers (cos, sin) tuple → comfy-kitchen's freqs_cis layout.

    Inputs:
      cos, sin: shape (S, D). diffusers stores each frequency angle twice
        in the last dim — cos[2k] == cos[2k+1] — so we take stride-2 slices
        to recover the D/2 unique angles.
      sequence_dim: which dim of the x tensor holds the sequence axis.
        - 2 → x is [B, H, S, D]; freqs broadcasts as (1, 1, S, D/2, 2, 2)
        - 1 → x is [B, S, H, D]; freqs broadcasts as (1, S, 1, D/2, 2, 2)

    Output dtype matches cos/sin (typically fp32; the kernel handles the
    bf16 query/key by broadcasting in compute).
    """
    cos_h = cos[..., 0::2]  # (S, D/2)
    sin_h = sin[..., 0::2]
    # Match diffusers' use_real_unbind_dim=-1 convention. See module docstring.
    # row layout per position-feature: [[cos, sin], [-sin, cos]]
    row0 = torch.stack([cos_h, sin_h], dim=-1)        # (S, D/2, 2)
    row1 = torch.stack([-sin_h, cos_h], dim=-1)       # (S, D/2, 2)
    matrix = torch.stack([row0, row1], dim=-2)        # (S, D/2, 2, 2)
    if sequence_dim == 2:
        return matrix[None, None, ...]                # (1, 1, S, D/2, 2, 2)
    if sequence_dim == 1:
        return matrix[None, :, None, ...]             # (1, S, 1, D/2, 2, 2)
    raise ValueError(f"unsupported sequence_dim={sequence_dim}")


def _ck_apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: Any,
    use_real: bool = True,
    use_real_unbind_dim: int = -1,
    sequence_dim: int = 2,
    **kwargs: Any,
):
    """Drop-in replacement for diffusers' `apply_rotary_emb`.

    Matches diffusers' full signature (use_real, use_real_unbind_dim,
    sequence_dim). When freqs_cis is the (cos, sin) tuple that FluxPosEmbed
    produces and we're on the FLUX-style path (use_real=True, unbind=-1),
    we convert and call comfy-kitchen's fused `apply_rope1`. Any other call
    pattern falls back to diffusers' original — keeps non-FLUX models working
    if they happen through this path.
    """
    if (
        isinstance(freqs_cis, tuple) and len(freqs_cis) == 2
        and use_real and use_real_unbind_dim == -1
        and sequence_dim in (1, 2)
        and not kwargs  # unknown kwargs → bail to original to be safe
    ):
        cos, sin = freqs_cis
        # Cache the converted ck_freqs by id(cos). Within a single gen FluxPosEmbed
        # is called once and the resulting (cos, sin) flows through every block.
        # Hit rate is ~3191 out of 3192 calls per gen — the build cost gets paid
        # only on the FIRST forward of each gen.
        cache_key = (id(cos), sequence_dim, x.device.index if x.device.type == "cuda" else -1)
        cached = _freqs_cache.get(cache_key)
        if cached is not None and cached[0] is cos:
            ck_freqs = cached[1]
        else:
            if cos.device != x.device:
                cos = cos.to(x.device)
                sin = sin.to(x.device)
            ck_freqs = _build_ck_freqs_cis(cos, sin, sequence_dim)
            _freqs_cache[cache_key] = (cos, ck_freqs)
        return _CK_APPLY_ROPE1(x, ck_freqs)

    # Fallback to diffusers' original for any path we don't handle.
    return _ORIGINAL_APPLY_ROTARY_EMB(
        x, freqs_cis,
        use_real=use_real, use_real_unbind_dim=use_real_unbind_dim,
        sequence_dim=sequence_dim, **kwargs,
    )


def install_ck_rope_patch() -> bool:
    """Monkey-patch diffusers' apply_rotary_emb to use comfy-kitchen.

    Returns True if the patch was applied, False if comfy-kitchen isn't
    installed (silent fallback to diffusers' original). Idempotent.
    """
    global _PATCH_APPLIED, _ORIGINAL_APPLY_ROTARY_EMB, _CK_APPLY_ROPE1
    if _PATCH_APPLIED:
        return True

    try:
        import comfy_kitchen as ck
    except ImportError:
        log.info("comfy-kitchen not installed — diffusers' built-in RoPE will be used")
        return False

    # Eager backend is always available; CUDA backend gives the speedup on Ampere+.
    backends = ck.list_backends()
    has_cuda = backends.get("cuda", {}).get("available")
    log.info(
        "comfy-kitchen present (backends: %s)",
        ", ".join(f"{k}={'on' if v.get('available') and not v.get('disabled') else 'off'}" for k, v in backends.items()),
    )
    _CK_APPLY_ROPE1 = ck.apply_rope1

    try:
        from diffusers.models import embeddings as diffusers_embeddings
    except ImportError:
        log.warning("diffusers.models.embeddings unavailable; RoPE patch skipped")
        return False

    _ORIGINAL_APPLY_ROTARY_EMB = diffusers_embeddings.apply_rotary_emb

    # Patch the symbol in diffusers.models.embeddings AND the transformer_flux
    # module that already imported it at startup (Python's `from ... import`
    # makes a local rebinding). Both must be patched.
    diffusers_embeddings.apply_rotary_emb = _ck_apply_rotary_emb
    try:
        from diffusers.models.transformers import transformer_flux
        transformer_flux.apply_rotary_emb = _ck_apply_rotary_emb
    except (ImportError, AttributeError) as e:
        log.warning("could not re-bind apply_rotary_emb in transformer_flux (%s)", e)

    _PATCH_APPLIED = True
    log.info("RoPE patch installed: diffusers' apply_rotary_emb → comfy_kitchen.apply_rope1 (%s backend)",
             "CUDA" if has_cuda else "eager")
    return True
