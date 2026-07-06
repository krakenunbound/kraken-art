"""Kraken Art FLUX pipeline — correct behavior when assembling from local components.

diffusers' `FluxPipeline.from_pretrained(..., torch_dtype=...)` wires dtype
consistently. Our manual assembly path must match that contract: latent noise
uses the transformer's compute dtype, not `prompt_embeds.dtype` (which can be
float32 when LayerNorm runs in fp32).
"""
from __future__ import annotations

import logging

from diffusers import FluxPipeline, FluxImg2ImgPipeline

log = logging.getLogger("kraken.flux")


class KrakenFluxPipeline(FluxPipeline):
    # Standard offload seq — VAE decode through diffusers hooks (sequential path is stable).
    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        # Match from_pretrained: latents share the transformer's compute dtype.
        latent_dtype = (
            getattr(self, "_kraken_compute_dtype", None)
            or getattr(self.transformer, "_kraken_compute_dtype", None)
            or self.transformer.dtype
        )
        return super().prepare_latents(
            batch_size,
            num_channels_latents,
            height,
            width,
            latent_dtype,
            device,
            generator,
            latents,
        )

    def encode_prompt(self, *args, **kwargs):
        prompt_embeds, pooled_prompt_embeds, text_ids = super().encode_prompt(*args, **kwargs)
        compute_dtype = (
            getattr(self, "_kraken_compute_dtype", None)
            or getattr(self.transformer, "_kraken_compute_dtype", None)
        )
        if compute_dtype is not None:
            text_ids = text_ids.to(dtype=compute_dtype)
        return prompt_embeds, pooled_prompt_embeds, text_ids

    def maybe_free_model_hooks(self) -> None:
        # diffusers' default cleanup calls module.to() through accelerate hooks and
        # native-crashes on Windows after model_cpu_offload FLUX runs. Pipeline stays
        # cached between jobs; hooks are stripped in unload() when we switch arch.
        log.debug("skipping maybe_free_model_hooks (stability on Windows)")


class KrakenFluxImg2ImgPipeline(FluxImg2ImgPipeline):
    """FLUX img2img for USDU tile-refine. Same compute-dtype + Windows-hook fixes
    as KrakenFluxPipeline, applied to the img2img prepare_latents signature."""

    def prepare_latents(
        self,
        image,
        timestep,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        latent_dtype = (
            getattr(self, "_kraken_compute_dtype", None)
            or getattr(self.transformer, "_kraken_compute_dtype", None)
            or self.transformer.dtype
        )
        return super().prepare_latents(
            image,
            timestep,
            batch_size,
            num_channels_latents,
            height,
            width,
            latent_dtype,
            device,
            generator,
            latents,
        )

    def encode_prompt(self, *args, **kwargs):
        prompt_embeds, pooled_prompt_embeds, text_ids = super().encode_prompt(*args, **kwargs)
        compute_dtype = (
            getattr(self, "_kraken_compute_dtype", None)
            or getattr(self.transformer, "_kraken_compute_dtype", None)
        )
        if compute_dtype is not None:
            text_ids = text_ids.to(dtype=compute_dtype)
        return prompt_embeds, pooled_prompt_embeds, text_ids

    def maybe_free_model_hooks(self) -> None:
        log.debug("skipping maybe_free_model_hooks (stability on Windows)")
