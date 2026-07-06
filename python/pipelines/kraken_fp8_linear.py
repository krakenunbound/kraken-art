"""FP8-resident linear layers for fitting large transformers on a 24 GB card.

Krea 2's transformer is ~24 GB in bf16 — it does not fit on a 24 GB GPU, so the
default path streams blocks from CPU (slow: ~30 s/forward-pass). Quantizing the
block weights to `float8_e4m3fn` halves them to ~12 GB so the whole transformer
stays GPU-resident with no streaming at all — the same trick the FLUX/Ideogram
paths use, and what ComfyUI does for fp8 weights on Ampere.

The RTX 3090 (Ampere) has no native FP8 matmul, so we store weights in fp8 and
upcast to the compute dtype (bf16) per forward — a memory-bandwidth-bound cast,
not a PCIe transfer. That keeps the big win (no streaming) while staying numeric.

Quantization is **per-output-channel** (a scale per weight row), which preserves
quality far better than a single per-tensor scale and costs almost nothing
(scale is `[out_features, 1]`). Only the bulky transformer blocks are quantized;
small/sensitive layers (embedders, projections, final layer) stay bf16 — the
"mixed" precision the community fp8 Krea 2 checkpoints also use.
"""
from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger("kraken.fp8")

# Largest finite magnitude representable by float8_e4m3fn.
_E4M3_MAX = 448.0


class Fp8Linear(nn.Module):
    """Drop-in replacement for `nn.Linear` storing a per-channel-scaled
    `float8_e4m3fn` weight and upcasting to the activation dtype in forward."""

    def __init__(
        self,
        weight_fp8: torch.Tensor,   # [out, in] float8_e4m3fn
        scale: torch.Tensor,        # [out, 1] float32 (per-output-channel)
        bias: torch.Tensor | None,  # [out] (compute dtype) or None
    ) -> None:
        super().__init__()
        self.in_features = weight_fp8.shape[1]
        self.out_features = weight_fp8.shape[0]
        # Buffers (not Parameters): inference only, and fp8 isn't a valid autograd dtype.
        self.register_buffer("weight", weight_fp8, persistent=True)
        self.register_buffer("scale", scale, persistent=True)
        if bias is not None:
            self.register_buffer("bias", bias, persistent=True)
        else:
            self.bias = None

    @classmethod
    def from_linear(cls, src: nn.Linear) -> "Fp8Linear":
        w = src.weight.data
        compute_dtype = w.dtype if w.dtype in (torch.bfloat16, torch.float16) else torch.bfloat16
        wf = w.float()
        # Per-output-channel scale: each row mapped into the fp8 range independently.
        amax = wf.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
        scale = (amax / _E4M3_MAX).to(torch.float32)
        q = (wf / scale).clamp(-_E4M3_MAX, _E4M3_MAX).to(torch.float8_e4m3fn)
        bias = src.bias.data.to(compute_dtype) if src.bias is not None else None
        return cls(q, scale, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Upcast fp8 -> activation dtype and rescale, then a normal matmul.
        w = self.weight.to(x.dtype) * self.scale.to(x.dtype)
        return F.linear(x, w, self.bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, weight_dtype=float8_e4m3fn"


def _is_in_block(name: str, block_roots: tuple[str, ...]) -> bool:
    parts = name.split(".")
    return len(parts) >= 2 and parts[0] in block_roots


def quantize_blocks_to_fp8(
    root: nn.Module,
    block_roots: tuple[str, ...] = ("transformer_blocks",),
) -> tuple[int, float, float]:
    """Replace every `nn.Linear` under any `block_roots.*` subtree with an
    `Fp8Linear`. Small/sensitive modules outside those subtrees are left in their
    original dtype.

    Returns `(num_quantized, bytes_before_gb, bytes_after_gb)` for the quantized
    layers (weight storage only), for logging the VRAM saving.
    """
    targets: list[tuple[nn.Module, str, nn.Linear]] = []
    name_by_id = {id(m): n for n, m in root.named_modules()}
    for name, module in root.named_modules():
        if not _is_in_block(name, block_roots):
            continue
        for child_name, child in module.named_children():
            if isinstance(child, nn.Linear):
                targets.append((module, child_name, child))

    before = 0
    after = 0
    n = 0
    for parent, child_name, lin in targets:
        before += lin.weight.numel() * lin.weight.element_size()
        fp8 = Fp8Linear.from_linear(lin)
        after += fp8.weight.numel() * fp8.weight.element_size() + fp8.scale.numel() * fp8.scale.element_size()
        setattr(parent, child_name, fp8)
        n += 1
    # Drop references to the originals so CPU RAM is reclaimed.
    del targets, name_by_id
    return n, before / 1024**3, after / 1024**3


def estimate_fp8_resident_gb(root: nn.Module, block_roots: tuple[str, ...] = ("transformer_blocks",)) -> float:
    """Estimate resident GB after fp8 quantization without mutating the model:
    block Linear weights counted at 1 byte/elem, everything else at its real size.
    """
    total = 0.0
    block_linear_ids: set[int] = set()
    for name, module in root.named_modules():
        if not _is_in_block(name, block_roots):
            continue
        for child in module.children():
            if isinstance(child, nn.Linear):
                block_linear_ids.add(id(child.weight))
                if child.bias is not None:
                    pass  # bias stays compute dtype, counted below
    for p in root.parameters():
        if id(p) in block_linear_ids:
            total += p.numel() * 1  # fp8 = 1 byte
        else:
            total += p.numel() * p.element_size()
    for b in root.buffers():
        total += b.numel() * b.element_size()
    return total / 1024**3
