# Copyright 2026 Krea AI and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ---------------------------------------------------------------------------
# Kraken Art vendoring note
# ---------------------------------------------------------------------------
# Verbatim vendoring of diffusers `main`:
#   diffusers/src/diffusers/pipelines/krea2/pipeline_krea2.py   (Krea2Pipeline)
#   diffusers/src/diffusers/pipelines/krea2/pipeline_output.py  (Krea2PipelineOutput)
#   diffusers/src/diffusers/loaders/lora_pipeline.py            (Krea2LoraLoaderMixin)
# See kraken_krea2_model.py for why we vendor rather than upgrade diffusers.
# Only the relative imports were rewritten to absolute, the output class and
# the LoRA mixin were inlined here, and two newer symbols (TRANSFORMER_NAME,
# _LOW_CPU_MEM_USAGE_DEFAULT_LORA) are imported with 0.38.0 fallbacks.
# ---------------------------------------------------------------------------
from __future__ import annotations

import inspect
import os
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import PIL.Image
import torch
from transformers import AutoTokenizer, Qwen3VLModel

from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders.lora_base import LoraBaseMixin, _fetch_state_dict
from diffusers.models import AutoencoderKLQwenImage
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import (
    BaseOutput,
    USE_PEFT_BACKEND,
    is_peft_version,
    is_torch_xla_available,
    logging,
    replace_example_docstring,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from huggingface_hub.utils import validate_hf_hub_args

from pipelines.kraken_krea2_model import Krea2Transformer2DModel


# TRANSFORMER_NAME lives in lora_pipeline in 0.38; it is simply "transformer".
try:
    from diffusers.loaders.lora_pipeline import TRANSFORMER_NAME
except ImportError:  # pragma: no cover
    TRANSFORMER_NAME = "transformer"

# Added to diffusers.utils after 0.38; default to False (disables the low-cpu-mem
# LoRA load path) when unavailable so loading still works on the pinned version.
try:
    from diffusers.utils import _LOW_CPU_MEM_USAGE_DEFAULT_LORA
except ImportError:  # pragma: no cover
    _LOW_CPU_MEM_USAGE_DEFAULT_LORA = False


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from pipelines.kraken_krea2_pipeline import Krea2Pipeline

        >>> pipe = Krea2Pipeline.from_pretrained("path/to/krea2-diffusers", torch_dtype=torch.bfloat16)
        >>> pipe.to("cuda")
        >>> image = pipe("a fox in the snow", num_inference_steps=28, guidance_scale=4.5).images[0]
        >>> image.save("krea2.png")
        ```
"""


@dataclass
class Krea2PipelineOutput(BaseOutput):
    """Output class for Krea 2 pipelines.

    Args:
        images (`list[PIL.Image.Image]` or `np.ndarray`):
            List of denoised PIL images of length `batch_size` or numpy array of shape `(batch_size, height, width,
            num_channels)`.
    """

    images: list[PIL.Image.Image] | np.ndarray


class Krea2LoraLoaderMixin(LoraBaseMixin):
    r"""
    Load LoRA layers into [`Krea2Transformer2DModel`]. Specific to [`Krea2Pipeline`].
    """

    _lora_loadable_modules = ["transformer"]
    transformer_name = TRANSFORMER_NAME

    @classmethod
    @validate_hf_hub_args
    # Copied from diffusers.loaders.lora_pipeline.CogVideoXLoraLoaderMixin.lora_state_dict
    def lora_state_dict(
        cls,
        pretrained_model_name_or_path_or_dict: str | dict[str, torch.Tensor],
        **kwargs,
    ):
        r"""
        See [`~loaders.StableDiffusionLoraLoaderMixin.lora_state_dict`] for more details.
        """
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", None)
        token = kwargs.pop("token", None)
        revision = kwargs.pop("revision", None)
        subfolder = kwargs.pop("subfolder", None)
        weight_name = kwargs.pop("weight_name", None)
        use_safetensors = kwargs.pop("use_safetensors", None)
        return_lora_metadata = kwargs.pop("return_lora_metadata", False)

        allow_pickle = False
        if use_safetensors is None:
            use_safetensors = True
            allow_pickle = True

        user_agent = {"file_type": "attn_procs_weights", "framework": "pytorch"}

        state_dict, metadata = _fetch_state_dict(
            pretrained_model_name_or_path_or_dict=pretrained_model_name_or_path_or_dict,
            weight_name=weight_name,
            use_safetensors=use_safetensors,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            token=token,
            revision=revision,
            subfolder=subfolder,
            user_agent=user_agent,
            allow_pickle=allow_pickle,
        )

        is_dora_scale_present = any("dora_scale" in k for k in state_dict)
        if is_dora_scale_present:
            warn_msg = "It seems like you are using a DoRA checkpoint that is not compatible in Diffusers at the moment. So, we are going to filter out the keys associated to 'dora_scale` from the state dict. If you think this is a mistake please open an issue https://github.com/huggingface/diffusers/issues/new."
            logger.warning(warn_msg)
            state_dict = {k: v for k, v in state_dict.items() if "dora_scale" not in k}

        out = (state_dict, metadata) if return_lora_metadata else state_dict
        return out

    # Copied from diffusers.loaders.lora_pipeline.CogVideoXLoraLoaderMixin.load_lora_weights
    def load_lora_weights(
        self,
        pretrained_model_name_or_path_or_dict: str | dict[str, torch.Tensor],
        adapter_name: str | None = None,
        hotswap: bool = False,
        **kwargs,
    ):
        """
        See [`~loaders.StableDiffusionLoraLoaderMixin.load_lora_weights`] for more details.
        """
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for this method.")

        low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", _LOW_CPU_MEM_USAGE_DEFAULT_LORA)
        if low_cpu_mem_usage and is_peft_version("<", "0.13.0"):
            raise ValueError(
                "`low_cpu_mem_usage=True` is not compatible with this `peft` version. Please update it with `pip install -U peft`."
            )

        if isinstance(pretrained_model_name_or_path_or_dict, dict):
            pretrained_model_name_or_path_or_dict = pretrained_model_name_or_path_or_dict.copy()

        kwargs["return_lora_metadata"] = True
        state_dict, metadata = self.lora_state_dict(pretrained_model_name_or_path_or_dict, **kwargs)

        is_correct_format = all("lora" in key for key in state_dict.keys())
        if not is_correct_format:
            raise ValueError("Invalid LoRA checkpoint. Make sure all LoRA param names contain `'lora'` substring.")

        self.load_lora_into_transformer(
            state_dict,
            transformer=getattr(self, self.transformer_name) if not hasattr(self, "transformer") else self.transformer,
            adapter_name=adapter_name,
            metadata=metadata,
            _pipeline=self,
            low_cpu_mem_usage=low_cpu_mem_usage,
            hotswap=hotswap,
        )

    @classmethod
    # Copied from diffusers.loaders.lora_pipeline.SD3LoraLoaderMixin.load_lora_into_transformer with SD3Transformer2DModel->Krea2Transformer2DModel
    def load_lora_into_transformer(
        cls,
        state_dict,
        transformer,
        adapter_name=None,
        _pipeline=None,
        low_cpu_mem_usage=False,
        hotswap: bool = False,
        metadata=None,
    ):
        """
        See [`~loaders.StableDiffusionLoraLoaderMixin.load_lora_into_unet`] for more details.
        """
        if low_cpu_mem_usage and is_peft_version("<", "0.13.0"):
            raise ValueError(
                "`low_cpu_mem_usage=True` is not compatible with this `peft` version. Please update it with `pip install -U peft`."
            )

        logger.info(f"Loading {cls.transformer_name}.")
        transformer.load_lora_adapter(
            state_dict,
            network_alphas=None,
            adapter_name=adapter_name,
            metadata=metadata,
            _pipeline=_pipeline,
            low_cpu_mem_usage=low_cpu_mem_usage,
            hotswap=hotswap,
        )

    @classmethod
    # Copied from diffusers.loaders.lora_pipeline.CogVideoXLoraLoaderMixin.save_lora_weights
    def save_lora_weights(
        cls,
        save_directory: str | os.PathLike,
        transformer_lora_layers: dict[str, torch.nn.Module | torch.Tensor] = None,
        is_main_process: bool = True,
        weight_name: str = None,
        save_function: Callable = None,
        safe_serialization: bool = True,
        transformer_lora_adapter_metadata: dict | None = None,
    ):
        r"""
        See [`~loaders.StableDiffusionLoraLoaderMixin.save_lora_weights`] for more information.
        """
        lora_layers = {}
        lora_metadata = {}

        if transformer_lora_layers:
            lora_layers[cls.transformer_name] = transformer_lora_layers
            lora_metadata[cls.transformer_name] = transformer_lora_adapter_metadata

        if not lora_layers:
            raise ValueError("You must pass at least one of `transformer_lora_layers` or `text_encoder_lora_layers`.")

        cls._save_lora_weights(
            save_directory=save_directory,
            lora_layers=lora_layers,
            lora_metadata=lora_metadata,
            is_main_process=is_main_process,
            weight_name=weight_name,
            save_function=save_function,
            safe_serialization=safe_serialization,
        )

    # Copied from diffusers.loaders.lora_pipeline.CogVideoXLoraLoaderMixin.fuse_lora
    def fuse_lora(
        self,
        components: list[str] = ["transformer"],
        lora_scale: float = 1.0,
        safe_fusing: bool = False,
        adapter_names: list[str] | None = None,
        **kwargs,
    ):
        r"""
        See [`~loaders.StableDiffusionLoraLoaderMixin.fuse_lora`] for more details.
        """
        super().fuse_lora(
            components=components,
            lora_scale=lora_scale,
            safe_fusing=safe_fusing,
            adapter_names=adapter_names,
            **kwargs,
        )

    # Copied from diffusers.loaders.lora_pipeline.CogVideoXLoraLoaderMixin.unfuse_lora
    def unfuse_lora(self, components: list[str] = ["transformer"], **kwargs):
        r"""
        See [`~loaders.StableDiffusionLoraLoaderMixin.unfuse_lora`] for more details.
        """
        super().unfuse_lora(components=components, **kwargs)


# Copied from diffusers.pipelines.flux.pipeline_flux.calculate_shift
def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: int | None = None,
    device: str | torch.device | None = None,
    timesteps: list[int] | None = None,
    sigmas: list[float] | None = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class Krea2Pipeline(DiffusionPipeline, Krea2LoraLoaderMixin):
    r"""
    The Krea 2 pipeline for text-to-image generation.

    Args:
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            Euler flow-matching scheduler. The Krea 2 sigma schedule is the resolution-aware exponential time shift, so
            the scheduler config is expected to set `use_dynamic_shifting=True` together with the Krea 2 shift
            parameters (`base_shift=0.5`, `max_shift=1.15`, `base_image_seq_len=256`, `max_image_seq_len=6400`).
        vae ([`AutoencoderKLQwenImage`]):
            The Qwen-Image variational auto-encoder (f8, 16 latent channels) used to decode latents to images.
        text_encoder ([`~transformers.PreTrainedModel`]):
            A Qwen3-VL model. The pipeline consumes a stack of hidden states tapped from several decoder layers rather
            than the last hidden state.
        tokenizer ([`~transformers.AutoTokenizer`]):
            The tokenizer paired with the text encoder.
        transformer ([`Krea2Transformer2DModel`]):
            The Krea 2 single-stream MMDiT that predicts the flow-matching velocity.
        text_encoder_select_layers (`tuple[int, ...]`, *optional*):
            Indices into the text encoder's `hidden_states` tuple (0 is the embedding output) whose states are stacked
            per token as the transformer's text conditioning. Must have `transformer.config.num_text_layers` entries.
        is_distilled (`bool`, *optional*, defaults to `False`):
            Whether the transformer is the few-step distilled (TDM/turbo) checkpoint. When `True` a fixed timestep
            shift `mu=1.15` is used; otherwise `mu` is computed from the image resolution.
        patch_size (`int`, *optional*, defaults to 2):
            Side length of the square patches the latents are packed into before entering the transformer.
    """

    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKLQwenImage,
        text_encoder: Qwen3VLModel,
        tokenizer: AutoTokenizer,
        transformer: Krea2Transformer2DModel,
        text_encoder_select_layers: tuple[int, ...] | list[int] | None = None,
        is_distilled: bool = False,
        patch_size: int = 2,
    ):
        super().__init__()

        self.register_modules(
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
        )
        if text_encoder_select_layers is None:
            text_encoder_select_layers = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)
        self.register_to_config(text_encoder_select_layers=tuple(text_encoder_select_layers))
        self.text_encoder_select_layers = tuple(text_encoder_select_layers)
        self.register_to_config(is_distilled=is_distilled)
        self.vae_scale_factor = 2 ** len(self.vae.temperal_downsample) if getattr(self, "vae", None) else 8
        self.register_to_config(patch_size=patch_size)
        self.patch_size = patch_size
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * self.patch_size)

        self.prompt_template_encode_prefix = (
            "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, "
            "spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n"
        )
        self.prompt_template_encode_suffix = "<|im_end|>\n<|im_start|>assistant\n"
        self.prompt_template_encode_start_idx = 34
        self.prompt_template_encode_num_suffix_tokens = 5

    def get_text_hidden_states(
        self,
        prompt: str | list[str],
        max_sequence_length: int = 512,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize `prompt` into the fixed-length Krea 2 layout and tap the selected encoder hidden states."""
        device = device or self._execution_device
        prompt = [prompt] if isinstance(prompt, str) else prompt
        prefix_idx = self.prompt_template_encode_start_idx
        text = [self.prompt_template_encode_prefix + e for e in prompt]
        text_tokens = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=max_sequence_length + prefix_idx - self.prompt_template_encode_num_suffix_tokens,
            return_tensors="pt",
        ).to(device)
        suffix_tokens = self.tokenizer([self.prompt_template_encode_suffix] * len(text), return_tensors="pt").to(
            device
        )

        input_ids = torch.cat([text_tokens.input_ids, suffix_tokens.input_ids], dim=1)
        attention_mask = torch.cat([text_tokens.attention_mask, suffix_tokens.attention_mask], dim=1).bool()

        position_ids = (attention_mask.long().cumsum(dim=-1) - 1).clamp(min=0)
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
        )
        hidden_states = torch.stack([outputs.hidden_states[i] for i in self.text_encoder_select_layers], dim=2)

        hidden_states = hidden_states[:, prefix_idx:]
        attention_mask = attention_mask[:, prefix_idx:]
        return hidden_states, attention_mask

    def encode_prompt(
        self,
        prompt: str | list[str],
        device: torch.device | None = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: torch.Tensor | None = None,
        prompt_embeds_mask: torch.Tensor | None = None,
        max_sequence_length: int = 512,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = device or self._execution_device

        if prompt_embeds is None:
            prompt_embeds, prompt_embeds_mask = self.get_text_hidden_states(prompt, max_sequence_length, device)

        batch_size, seq_len, num_text_layers, dim = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, num_text_layers, dim)
        prompt_embeds_mask = prompt_embeds_mask.repeat(1, num_images_per_prompt)
        prompt_embeds_mask = prompt_embeds_mask.view(batch_size * num_images_per_prompt, seq_len)

        return prompt_embeds, prompt_embeds_mask

    def check_inputs(
        self,
        prompt,
        height,
        width,
        negative_prompt=None,
        prompt_embeds=None,
        prompt_embeds_mask=None,
        negative_prompt_embeds=None,
        negative_prompt_embeds_mask=None,
        callback_on_step_end_tensor_inputs=None,
        max_sequence_length=None,
    ):
        multiple = self.vae_scale_factor * self.patch_size
        if height % multiple != 0 or width % multiple != 0:
            raise ValueError(f"`height` and `width` must be divisible by {multiple} but are {height} and {width}.")

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and prompt_embeds_mask is None:
            raise ValueError(
                "If `prompt_embeds` are provided, `prompt_embeds_mask` also have to be passed. Make sure to generate `prompt_embeds_mask` from the same text encoder that was used to generate `prompt_embeds`."
            )
        if negative_prompt_embeds is not None and negative_prompt_embeds_mask is None:
            raise ValueError(
                "If `negative_prompt_embeds` are provided, `negative_prompt_embeds_mask` also have to be passed. Make sure to generate `negative_prompt_embeds_mask` from the same text encoder that was used to generate `negative_prompt_embeds`."
            )

        if max_sequence_length is not None and max_sequence_length <= 0:
            raise ValueError(f"`max_sequence_length` must be a positive integer but is {max_sequence_length}")

    def _pack_latents(self, latents, batch_size, num_channels_latents, height, width):
        p = self.patch_size
        latents = latents.view(batch_size, num_channels_latents, height // p, p, width // p, p)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // p) * (width // p), num_channels_latents * p * p)

        return latents

    def _unpack_latents(self, latents, height, width):
        batch_size, _, channels = latents.shape
        p = self.patch_size

        height = p * (int(height) // (self.vae_scale_factor * p))
        width = p * (int(width) // (self.vae_scale_factor * p))

        latents = latents.view(batch_size, height // p, width // p, channels // (p * p), p, p)
        latents = latents.permute(0, 3, 1, 4, 2, 5)

        latents = latents.reshape(batch_size, channels // (p * p), 1, height, width)

        return latents

    @staticmethod
    def prepare_position_ids(text_seq_len: int, grid_height: int, grid_width: int, device: torch.device):
        """Build the `(text_seq_len + grid_height * grid_width, 3)` rotary coordinates for the combined sequence."""
        text_ids = torch.zeros(text_seq_len, 3, device=device)
        image_ids = torch.zeros(grid_height, grid_width, 3, device=device)
        image_ids[..., 1] = torch.arange(grid_height, device=device)[:, None]
        image_ids[..., 2] = torch.arange(grid_width, device=device)[None, :]
        image_ids = image_ids.reshape(grid_height * grid_width, 3)
        return torch.cat([text_ids, image_ids], dim=0)

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
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        latent_height = height // self.vae_scale_factor
        latent_width = width // self.vae_scale_factor
        shape = (batch_size, num_channels_latents, latent_height, latent_width)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        latents = self._pack_latents(latents, batch_size, num_channels_latents, latent_height, latent_width)

        return latents

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 0

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: str | list[str] | None = None,
        negative_prompt: str | list[str] | None = None,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 28,
        sigmas: list[float] | None = None,
        guidance_scale: float = 4.5,
        num_images_per_prompt: int = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        prompt_embeds_mask: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds_mask: torch.Tensor | None = None,
        output_type: str | None = "pil",
        return_dict: bool = True,
        callback_on_step_end: Callable[[int, int, dict], None] | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],
        attention_kwargs: dict[str, Any] | None = None,
        max_sequence_length: int = 512,
    ):
        r"""
        Function invoked when calling the pipeline for generation. See the upstream Krea 2 docs for argument details.

        Examples:

        Returns:
            [`Krea2PipelineOutput`] or `tuple`: [`Krea2PipelineOutput`] if `return_dict` is True, otherwise a `tuple`
            whose first element is a list with the generated images.
        """
        multiple = self.vae_scale_factor * self.patch_size
        if height % multiple != 0 or width % multiple != 0:
            rounded_height = ((height + multiple - 1) // multiple) * multiple
            rounded_width = ((width + multiple - 1) // multiple) * multiple
            logger.warning(
                f"`height` and `width` must be multiples of {multiple}; rounding up from {height}x{width} to"
                f" {rounded_height}x{rounded_width}."
            )
            height, width = rounded_height, rounded_width

        # 1. Check inputs
        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_embeds_mask=negative_prompt_embeds_mask,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Encode the prompts
        prompt_embeds, prompt_embeds_mask = self.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            max_sequence_length=max_sequence_length,
        )
        if self.do_classifier_free_guidance:
            if negative_prompt is None and negative_prompt_embeds is None:
                negative_prompt = ""
            if isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt] * batch_size
            negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt(
                prompt=negative_prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                prompt_embeds=negative_prompt_embeds,
                prompt_embeds_mask=negative_prompt_embeds_mask,
                max_sequence_length=max_sequence_length,
            )

        # 4. Prepare latents and position ids
        num_channels_latents = self.transformer.config.in_channels // (self.patch_size**2)
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        grid_height = height // (self.vae_scale_factor * self.patch_size)
        grid_width = width // (self.vae_scale_factor * self.patch_size)
        position_ids = self.prepare_position_ids(prompt_embeds.shape[1], grid_height, grid_width, device)

        # 5. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        image_seq_len = latents.shape[1]
        if self.config.is_distilled:
            mu = 1.15
        else:
            mu = calculate_shift(
                image_seq_len,
                self.scheduler.config.get("base_image_seq_len", 256),
                self.scheduler.config.get("max_image_seq_len", 6400),
                self.scheduler.config.get("base_shift", 0.5),
                self.scheduler.config.get("max_shift", 1.15),
            )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # 6. Denoising loop
        self.scheduler.set_begin_index(0)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = (t / self.scheduler.config.num_train_timesteps).expand(latents.shape[0]).to(latents.dtype)

                noise_pred = self.transformer(
                    hidden_states=latents,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timestep,
                    position_ids=position_ids,
                    encoder_attention_mask=prompt_embeds_mask,
                    attention_kwargs=self.attention_kwargs,
                    return_dict=False,
                )[0]

                if self.do_classifier_free_guidance:
                    neg_noise_pred = self.transformer(
                        hidden_states=latents,
                        encoder_hidden_states=negative_prompt_embeds,
                        timestep=timestep,
                        position_ids=position_ids,
                        encoder_attention_mask=negative_prompt_embeds_mask,
                        attention_kwargs=self.attention_kwargs,
                        return_dict=False,
                    )[0]
                    noise_pred = noise_pred + guidance_scale * (noise_pred - neg_noise_pred)

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        self._current_timestep = None

        # 7. Decode latents
        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents(latents, height, width)
            latents = latents.to(self.vae.dtype)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents / latents_std + latents_mean
            image = self.vae.decode(latents, return_dict=False)[0][:, :, 0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return Krea2PipelineOutput(images=image)
