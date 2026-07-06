"""First Block Cache for FLUX, adapted to diffusers 0.38's transformer API.

Background
----------
`para-attn` 0.3.38's `CachedTransformerBlocks.call_remaining_transformer_blocks`
was written against an older diffusers where `FluxSingleTransformerBlock.forward`
took a single pre-concatenated `hidden_states` argument. Diffusers 0.38
refactored these blocks to accept `(hidden_states, encoder_hidden_states, temb,
image_rotary_emb=...)` and return `(encoder_hidden_states, hidden_states)`
(see `diffusers/models/transformers/transformer_flux.py:377-406`).

Calling para-attn unpatched against diffusers 0.38 raises:
    TypeError: FluxSingleTransformerBlock.forward() missing 1 required
               positional argument: 'encoder_hidden_states'

This module monkey-patches `para_attn.first_block_cache.utils.CachedTransformerBlocks`
to use the new diffusers signature for single_transformer_blocks. The
double-stream transformer_blocks path was already correct in para-attn.

What FBCache does (for the curious)
-----------------------------------
After the FIRST transformer block runs, compare its residual to the residual
from the previous sampling step. If the diff is below `residual_diff_threshold`,
skip blocks 2..N entirely and reuse the previous step's full residual — the
diffusion process is smooth enough that early-step similarity strongly predicts
late-step similarity. Grok's analysis (`Documentation/grok_findings.md`):
"the single biggest reason ComfyUI feels dramatically faster" — published
1.5–2× effective speedup with minimal quality cost.

Usage
-----
    from pipelines.kraken_fbcache import install_fbcache_patch
    install_fbcache_patch()       # one-time; safe to call repeatedly
    # Then call apply_cache_on_pipe from para-attn as normal.
"""
from __future__ import annotations

import logging

import torch

log = logging.getLogger("kraken.fbcache")

_PATCH_APPLIED = False


def install_fbcache_patch() -> bool:
    """Replace para-attn's call_remaining_transformer_blocks with a version
    that uses diffusers 0.38's single_transformer_block signature.

    Returns True if installed (or was already installed). False if para-attn
    is not present (caller can choose to fall back / disable FBCache).
    """
    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        return True
    try:
        from para_attn.first_block_cache import utils as fbc_utils
    except ImportError:
        log.info("para-attn not installed — FBCache patch skipped")
        return False

    from para_attn.first_block_cache.utils import (  # type: ignore
        is_slg_enabled, slg_should_skip_block,
    )

    Cls = fbc_utils.CachedTransformerBlocks

    def patched_call_remaining(self, hidden_states, encoder_hidden_states, *args, **kwargs):
        """Run the non-first transformer blocks, then all single-stream blocks,
        using diffusers 0.38's two-arg-two-return signature for the single blocks.

        Returns 4-tuple: (hidden_states, encoder_hidden_states,
                          hidden_states_residual, encoder_hidden_states_residual)
        — the residuals are computed relative to the inputs BEFORE
        call_remaining ran (which is after the first block ran in the caller).
        """
        original_hidden_states = hidden_states
        original_encoder_hidden_states = encoder_hidden_states

        slg = is_slg_enabled()

        # ---- Double-stream blocks (transformer_blocks[1:]) ------------------
        # These have always taken `(hidden_states, encoder_hidden_states, ...)`
        # and returned a tuple in diffusers, so we keep the original logic.
        for i, block in enumerate(self.transformer_blocks[1:]):
            if slg and slg_should_skip_block(i + 1):
                continue
            result = block(hidden_states, encoder_hidden_states, *args, **kwargs)
            if isinstance(result, torch.Tensor):
                hidden_states = result
            else:
                hidden_states, encoder_hidden_states = result
                if not self.return_hidden_states_first:
                    hidden_states, encoder_hidden_states = encoder_hidden_states, hidden_states

        # ---- Single-stream blocks (single_transformer_blocks) ---------------
        # Diffusers 0.38: each block accepts and returns (enc, hidden) as
        # separate tensors. Concat-then-call-then-split (the old para-attn
        # pattern) does NOT work — the block now does its own internal concat.
        if self.single_transformer_blocks is not None:
            for i, block in enumerate(self.single_transformer_blocks):
                if slg and slg_should_skip_block(len(self.transformer_blocks) + i):
                    continue
                result = block(hidden_states, encoder_hidden_states, *args, **kwargs)
                if isinstance(result, torch.Tensor):
                    # Defensive: some custom blocks may still return a single
                    # concatenated tensor. Treat the same as the old path.
                    encoder_hidden_states, hidden_states = result.split(
                        [encoder_hidden_states.shape[1],
                         result.shape[1] - encoder_hidden_states.shape[1]],
                        dim=1,
                    )
                else:
                    encoder_hidden_states, hidden_states = result

        # Residuals relative to the inputs to this function (post-first-block).
        hidden_states_residual = hidden_states - original_hidden_states
        encoder_hidden_states_residual = (
            encoder_hidden_states - original_encoder_hidden_states
        )
        return (hidden_states, encoder_hidden_states,
                hidden_states_residual, encoder_hidden_states_residual)

    Cls.call_remaining_transformer_blocks = patched_call_remaining
    _PATCH_APPLIED = True
    log.info("para-attn FBCache patched for diffusers 0.38 single-block API")
    return True
