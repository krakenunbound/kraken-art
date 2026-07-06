"""Per-Linear streaming offload for FLUX-class transformers.

Background
----------
On a 24 GB GPU a BF16 FLUX-dev transformer (~22 GB in memory) does not leave
enough headroom for activations + the VAE decode. Diffusers' default fallback
is `enable_model_cpu_offload`, which moves entire submodules CPU↔GPU on every
forward — fine for small models, devastating for FLUX (138 s vs ComfyUI's 48 s
on the same machine).

Forge and Fooocus solve this by streaming weights at *Linear* granularity: each
`nn.Linear` learns whether its weight lives on CPU or GPU; if on CPU, it copies
the weight to GPU per forward with `non_blocking=True`. Phase 2 of this module
adds Forge's full overlap (pinned host memory + dedicated mover CUDA stream)
so the H2D copy of layer N+1 runs concurrently with the matmul of layer N.

What this module does
---------------------
1. `StreamingLinear(nn.Linear)` — a drop-in subclass whose `forward()` handles
   four cases in one unified path:
     - GPU-resident plain weight: vanilla F.linear (zero overhead vs nn.Linear)
     - GPU-resident FP8 weight: cast to compute dtype per call
     - GPU-resident scaled-FP8 weight: cast + multiply by `_kraken_weight_scale`
     - CPU-resident weight (any of the above): copy to GPU first, then same logic
2. `swap_linears(module)` — walks the tree and replaces every `nn.Linear` with
   a `StreamingLinear`, preserving weight/bias parameter objects (no data copy).
3. `apply_streaming(module, budget_bytes, device)` — three-tier partition:
     - Small leaf modules and norms stay fully on GPU (always)
     - Transformer blocks consumed until budget is hit → on GPU
     - Remaining blocks → CPU, all StreamingLinears in them get `stream_weights=True`

The partition runs at load time, once per pipeline. The hot path (StreamingLinear.forward)
has zero allocations beyond the H2D weight copy itself.

Integration with the existing scaled-FP8 path in flux.py
--------------------------------------------------------
The legacy `_scaled_linear_forward` / `_fp8_linear_forward` monkey-patches on
plain `nn.Linear` are superseded by `StreamingLinear.forward`. The `_kraken_weight_scale`
buffer convention is preserved — StreamingLinear reads it the same way.
`_apply_flux_weight_scales` and `_patch_fp8_linears` should be no-ops once the
swap to StreamingLinear has happened; both check for an instance flag and skip
StreamingLinears.
"""
from __future__ import annotations

import logging
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

log = logging.getLogger("kraken.stream")

_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)


# ---- ComfyUI-style 2-stream weight mover --------------------------------
#
# Updated 2026-05-22 after reading `D:\AI_Art\ComfyUI\comfy\model_management.py`
# (lines 1155-1318) and `comfy\ops.py` (lines 210-280). Their mechanism on
# NVIDIA defaults to 2 CUDA streams in round-robin: while stream A is doing
# H2D for layer N, the next call can use stream B for layer N+1 IN PARALLEL.
# That's pipeline depth 2 vs our previous single-stream depth 1 — closes a
# measured ~330 ms/step gap on FLUX-dev (1024² × 28 steps).
#
# We also follow their preallocated-buffer pattern: instead of allocating a
# fresh GPU tensor with `weight.to(device)` per call (1008 allocations/gen for
# our 6 streamed blocks × 6 Linears × 28 steps), we copy_ into a buffer that
# was allocated ONCE during apply_streaming. Zero allocator pressure on the
# hot path.

_NUM_MOVER_STREAMS = 2
_mover_streams: list["torch.cuda.Stream"] = []
_mover_counter = 0
_mover_buffers: list["torch.Tensor"] = []


def _init_mover_streams() -> bool:
    """Create the 2-stream pool on first need. Returns False on CPU-only hosts."""
    global _mover_streams
    if _mover_streams:
        return True
    if not torch.cuda.is_available():
        return False
    try:
        _mover_streams = [torch.cuda.Stream(priority=0) for _ in range(_NUM_MOVER_STREAMS)]
        return True
    except RuntimeError as e:
        log.warning("could not create mover streams (%s); falling back to sync H2D", e)
        return False


def _get_offload_stream(device: torch.device) -> "torch.cuda.Stream | None":
    """Round-robin one of the mover streams.

    Per ComfyUI's pattern: the chosen stream waits for the current compute
    stream BEFORE the caller queues its copy. That sets up the
    compute-then-copy ordering needed for correctness. After the caller's copy
    is queued, the compute stream waits for THIS stream (via sync_stream)
    before consuming the result.
    """
    global _mover_counter
    if not _mover_streams and not _init_mover_streams():
        return None
    _mover_counter = (_mover_counter + 1) % len(_mover_streams)
    s = _mover_streams[_mover_counter]
    # Have the new copy stream wait for compute so we don't race ahead of
    # something the compute stream is still using.
    s.wait_stream(torch.cuda.current_stream(device))
    return s


def _sync_compute_to(stream: "torch.cuda.Stream", device: torch.device) -> None:
    """Make the compute stream wait for `stream` to finish. Matches ComfyUI's
    `sync_stream(device, stream)` helper."""
    if stream is not None:
        torch.cuda.current_stream(device).wait_stream(stream)


# Backwards-compat shim for code that still imports the old name.
def _get_mover_stream() -> "torch.cuda.Stream | None":  # noqa: D401
    """Legacy alias — returns the next round-robin offload stream."""
    if torch.cuda.is_available():
        return _get_offload_stream(torch.device("cuda"))
    return None


class StreamingLinear(nn.Linear):
    """nn.Linear that can have its weight live on CPU and stream to GPU on demand.

    Three runtime modes selected per-instance by simple flags:
      - `_kraken_stream_weights=False` (default): weight already on the target
        device. Forward is vanilla F.linear unless an FP8 cast or scale is needed.
      - `_kraken_stream_weights=True`: weight is on CPU. Forward copies it to
        the input's device (non_blocking=True) before the matmul. The H2D copy
        is staged through pinned host memory if `_kraken_pinned=True` (set by
        `apply_streaming`), which lets the copy overlap with prior compute when
        Phase 2's mover stream is wired up.

    The `_kraken_weight_scale` buffer (if registered) is multiplied into the
    weight after any dtype cast — matches the convention used by
    `flux._apply_flux_weight_scales` for ComfyUI-style scaled-FP8 checkpoints.
    """

    _kraken_stream_weights: bool = False
    _kraken_pinned: bool = False
    # NOTE: prefetch handoff slots (`_kraken_pending_w`, `_kraken_pending_b`,
    # `_kraken_pending_event`) are NOT declared as class attributes here on
    # purpose. nn.Module.__setattr__ inspects type-annotated class attrs that
    # could hold tensors and tries to register them as buffers — which then
    # raises `attribute already exists`. We set them via `object.__setattr__`
    # in the prefetch hook and read them via plain attribute access (which
    # falls through to the instance __dict__ once set).

    @classmethod
    def from_linear(cls, src: nn.Linear) -> "StreamingLinear":
        """Build a StreamingLinear that *shares* parameter storage with `src`.

        We don't allocate new tensors — we transplant `src`'s `weight` and
        `bias` Parameters onto the new instance so any pre-loaded state_dict
        values, FP8 dtypes, and registered scale buffers carry over verbatim.
        """
        new = cls.__new__(cls)
        nn.Module.__init__(new)
        new.in_features = src.in_features
        new.out_features = src.out_features
        new.weight = src.weight
        new.bias = src.bias
        # Preserve any kraken-specific buffers/flags already attached.
        for name, buf in list(src._buffers.items()):
            new._buffers[name] = buf
        for attr in (
            "_kraken_weight_scale",
            "_kraken_scaled_fp8_forward",
            "_kraken_fp8_forward",
        ):
            if hasattr(src, attr):
                try:
                    setattr(new, attr, getattr(src, attr))
                except Exception:
                    pass
        return new

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        weight = self.weight
        bias = self.bias

        compute_device = x.device
        compute_dtype = x.dtype

        # ---- Stage 1: get the weight onto the compute device ------------
        # Four sub-cases here, in order of preference:
        #   (a) prefetched by a previous block's hook (Phase 3): the GPU weight
        #       is already on this instance, queued on the mover stream by an
        #       earlier block. Wait on the recorded event (usually no-op — the
        #       copy finished while compute was running prior matmuls), then use.
        #   (b) already on compute device: zero-cost. Linears in fully-resident
        #       blocks land here.
        #   (c) on CPU, host memory pinned, mover stream available (Phase 2):
        #       sync-style streaming. H2D on mover stream, compute waits.
        #       Used as a fallback when prefetch hasn't happened (e.g. very
        #       first forward of the session before the prefetch chain warms up).
        #   (d) on CPU but unpinned, or no mover stream (Phase 1 baseline):
        #       synchronous H2D on the compute stream.
        pending_w = self.__dict__.get("_kraken_pending_w")
        if pending_w is not None:
            event = self.__dict__.get("_kraken_pending_event")
            if event is not None:
                torch.cuda.current_stream(compute_device).wait_event(event)
            weight = pending_w
            pending_b = self.__dict__.get("_kraken_pending_b")
            if pending_b is not None:
                bias = pending_b
            # Consume: drop the prefetch refs so this Linear's GPU buffer can
            # be reclaimed after F.linear returns. The next forward (next step)
            # will get a fresh prefetch from the previous block's hook. We
            # write to __dict__ directly to bypass nn.Module.__setattr__'s
            # buffer-registration logic (which choked on tensor-typed attrs).
            self.__dict__["_kraken_pending_w"] = None
            self.__dict__["_kraken_pending_b"] = None
            self.__dict__["_kraken_pending_event"] = None
        elif self._kraken_stream_weights or weight.device != compute_device:
            mover = _get_mover_stream() if (self._kraken_pinned and compute_device.type == "cuda") else None
            if mover is not None:
                with torch.cuda.stream(mover):
                    weight_gpu = weight.to(device=compute_device, non_blocking=True)
                    bias_gpu = (
                        bias.to(device=compute_device, non_blocking=True)
                        if bias is not None and bias.device != compute_device else bias
                    )
                torch.cuda.current_stream(compute_device).wait_stream(mover)
                weight = weight_gpu
                bias = bias_gpu
            else:
                weight = weight.to(device=compute_device, non_blocking=True)
                if bias is not None and bias.device != compute_device:
                    bias = bias.to(device=compute_device, non_blocking=True)

        # ---- Stage 2: cast FP8 → compute dtype (if needed) --------------
        if weight.dtype in _FP8_DTYPES:
            weight = weight.to(dtype=compute_dtype)
        if bias is not None and bias.dtype in _FP8_DTYPES:
            bias = bias.to(dtype=compute_dtype)

        # ---- Stage 3: apply per-layer scale (ComfyUI scaled-FP8) --------
        scale = getattr(self, "_kraken_weight_scale", None)
        if scale is not None:
            if scale.device != weight.device or scale.dtype != weight.dtype:
                scale = scale.to(device=weight.device, dtype=weight.dtype, non_blocking=True)
            weight = weight * scale

        return F.linear(x, weight, bias)


def swap_linears(root: nn.Module) -> int:
    """Replace every `nn.Linear` descendant of `root` with a `StreamingLinear`.

    Walks parent modules and reassigns their named children — preserves the
    module tree structure and parameter identities. Idempotent: skips modules
    that are already StreamingLinear.

    Returns the number of swaps performed.
    """
    swapped = 0
    # We need parent references to do `setattr`, so iterate over named_modules
    # and look at each module's _modules dict directly.
    for parent in root.modules():
        for child_name, child in list(parent._modules.items()):
            if child is None:
                continue
            if isinstance(child, StreamingLinear):
                continue
            if type(child) is nn.Linear:
                new = StreamingLinear.from_linear(child)
                parent._modules[child_name] = new
                swapped += 1
    return swapped


def _module_param_bytes(module: nn.Module) -> int:
    """Sum of bytes occupied by *all* parameters and buffers reachable from `module`.

    Includes children recursively. Used both for budget accounting and to decide
    whether a given block fits in the remaining VRAM budget.
    """
    total = 0
    for p in module.parameters(recurse=True):
        total += p.numel() * p.element_size()
    for b in module.buffers(recurse=True):
        total += b.numel() * b.element_size()
    return total


def _is_streamable_block(name: str) -> bool:
    """Identify FLUX transformer blocks that are safe to stream from CPU.

    These are the heavy repeating units; small one-off modules (embedders,
    norm_out, proj_out, time_text_embed, x_embedder, context_embedder) are kept
    fully on GPU because per-call streaming overhead matters more when the
    module is tiny.
    """
    # FLUX (diffusers naming): transformer_blocks.N / single_transformer_blocks.N
    parts = name.split(".")
    if len(parts) >= 2 and parts[0] in ("transformer_blocks", "single_transformer_blocks"):
        # Match the block root, not a sub-child — e.g. "transformer_blocks.5"
        # but not "transformer_blocks.5.attn".
        return len(parts) == 2 and parts[1].isdigit()
    return False


def apply_streaming(
    root: nn.Module,
    *,
    budget_bytes: int,
    device: torch.device | str,
    pin_memory: bool = False,
    label: str = "transformer",
) -> dict[str, Any]:
    """Decide which modules live on GPU vs. stream from CPU, then place them.

    Algorithm:
      1. Identify streamable block roots (FLUX transformer_blocks.* /
         single_transformer_blocks.*) and measure each.
      2. Move all *non-block* params (embedders, norms, proj_out, etc.) to GPU
         first — these are small and always-resident. Subtract from budget.
      3. Walk blocks in order; while there's budget, .to(device) the block.
      4. Remaining blocks: leave on CPU. For every StreamingLinear inside,
         set `_kraken_stream_weights=True` (and `_kraken_pinned=True` if
         `pin_memory=True`, after pinning the storage).

    Returns a summary dict for logging.
    """
    device = torch.device(device)
    total_bytes = _module_param_bytes(root)

    # ---- Step 1: enumerate blocks --------------------------------------
    blocks: list[tuple[str, nn.Module, int]] = []  # (name, module, bytes)
    block_param_ids: set[int] = set()
    for name, module in root.named_modules():
        if _is_streamable_block(name):
            sz = _module_param_bytes(module)
            blocks.append((name, module, sz))
            for p in module.parameters(recurse=True):
                block_param_ids.add(id(p))
            for b in module.buffers(recurse=True):
                block_param_ids.add(id(b))

    non_block_bytes = total_bytes - sum(sz for _, _, sz in blocks)

    # ---- Step 2: place small always-resident params on GPU first -------
    if non_block_bytes > budget_bytes:
        log.warning(
            "%s: non-block params (%.2f GB) exceed budget (%.2f GB); "
            "GPU will be oversubscribed",
            label, non_block_bytes / 1024**3, budget_bytes / 1024**3,
        )

    # Move every param/buffer NOT inside a block to GPU. We do this by walking
    # parameters/buffers; `tensor.data = tensor.data.to(...)` keeps the Parameter
    # object alive (we don't want to replace it, we just want to retarget storage).
    for p in root.parameters(recurse=True):
        if id(p) in block_param_ids:
            continue
        if p.device != device:
            p.data = p.data.to(device=device, non_blocking=True)
    for b in root.buffers(recurse=True):
        if id(b) in block_param_ids:
            continue
        if b.device != device:
            b.data = b.data.to(device=device, non_blocking=True)

    remaining_budget = budget_bytes - non_block_bytes

    # ---- Step 3: blocks on GPU until budget exhausted ------------------
    on_gpu_blocks: list[str] = []
    on_cpu_blocks: list[str] = []
    streamed_bytes = 0  # bytes left on CPU (Linear weights + biases only)

    for name, module, sz in blocks:
        if sz <= remaining_budget:
            module.to(device=device)
            remaining_budget -= sz
            on_gpu_blocks.append(name)
            continue

        # ---- Streaming tier for this block ----------------------------
        # Inside a streamed block, only StreamingLinear weights (and biases)
        # stay on CPU — they dominate the byte count. Everything else (norms,
        # layer-scale, RoPE buffers, register_buffer'd flags, etc.) is small
        # and MUST be on GPU because the activation tensor flowing through
        # the block is on GPU and ops like RMSNorm require matching devices.
        on_cpu_blocks.append(name)
        streamed_param_ids: set[int] = set()
        for child in module.modules():
            if not isinstance(child, StreamingLinear):
                continue
            child._kraken_stream_weights = True
            # Detach the weight tensor from any safetensors mmap backing.
            # Without this, the original 22 GB transformer file stays mmap'd
            # for the lifetime of these residual references — on Windows this
            # consumes commit-limit virtual address space (RAM + pagefile),
            # and a subsequent mmap (the T5 bf16 cache, etc.) hits OSError 1455
            # "paging file too small". `.clone()` copies into private heap memory,
            # so the safetensors file can close once we drop the last refs.
            child.weight.data = child.weight.data.clone()
            streamed_param_ids.add(id(child.weight))
            if child.bias is not None and child.bias.device != device:
                # Bias is small — keep on GPU to avoid a per-call H2D. Also
                # detaches from the mmap in the move.
                child.bias.data = child.bias.data.to(device=device, non_blocking=True)
            streamed_bytes += child.weight.numel() * child.weight.element_size()
            if pin_memory:
                try:
                    child.weight.data = child.weight.data.pin_memory()
                    child._kraken_pinned = True
                except RuntimeError as e:
                    # Pinning can fail if RAM is too fragmented. Non-fatal —
                    # pageable H2D still correct, just no overlap with compute.
                    log.warning("%s: pin_memory failed on %s (%s)", label, name, e)

            # NOTE: an earlier draft preallocated a per-Linear `_kraken_gpu_dest`
            # buffer here (ComfyUI ops.py:241 pattern). On a 24 GB card that
            # doubled the streamed footprint (CPU pinned 1.58 GB + GPU dest
            # 1.58 GB) and pushed VRAM into thrash (239 MB free during sampling
            # → 80 s/step). ComfyUI shares a single int8 buffer per stream
            # sized to the LARGEST weight using that stream, ~76 MB total —
            # not per-Linear. We're keeping the per-call alloc pattern for now
            # because torch's allocator caches recently-freed buffers anyway,
            # so the per-call cost is small. A future port could implement
            # comfy's interpret_gathered_like shared-buffer pattern.

        # Move everything else in this block to GPU.
        for p in module.parameters(recurse=True):
            if id(p) in streamed_param_ids:
                continue
            if p.device != device:
                p.data = p.data.to(device=device, non_blocking=True)
        for b in module.buffers(recurse=True):
            if id(b) in streamed_param_ids:
                continue
            if b.device != device:
                b.data = b.data.to(device=device, non_blocking=True)

        # Pin per-layer scale buffers (tiny, used every forward).
        if pin_memory:
            for sub_name, buf in module.named_buffers():
                if "_kraken_weight_scale" in sub_name:
                    try:
                        buf.data = buf.data.pin_memory()
                    except RuntimeError:
                        pass

    global _mover_buffers
    _mover_buffers = []
    if on_cpu_blocks:
        max_block_bytes = 0
        for name in on_cpu_blocks:
            block = root.get_submodule(name)
            sls = _ordered_streamed_linears(block)
            block_bytes = sum(sl.weight.numel() * sl.weight.element_size() for sl in sls)
            if block_bytes > max_block_bytes:
                max_block_bytes = block_bytes
        if max_block_bytes > 0:
            try:
                _mover_buffers = [
                    torch.zeros(max_block_bytes, dtype=torch.uint8, device=device)
                    for _ in range(_NUM_MOVER_STREAMS)
                ]
                log.info(
                    "%s: allocated %d flat prefetch buffers of size %.2f MB on %s",
                    label, _NUM_MOVER_STREAMS, max_block_bytes / 1024**2, device
                )
            except RuntimeError as e:
                log.warning("%s: failed to allocate flat prefetch buffers: %s; falling back to per-call alloc", label, e)
                _mover_buffers = []

    summary = {
        "total_gb": total_bytes / 1024**3,
        "non_block_gb": non_block_bytes / 1024**3,
        "blocks_on_gpu": len(on_gpu_blocks),
        "blocks_on_cpu": len(on_cpu_blocks),
        "streamed_gb": streamed_bytes / 1024**3,
        "budget_remaining_gb": max(0, remaining_budget) / 1024**3,
        "fully_resident": not on_cpu_blocks,
    }
    # Aggressive GC: the partition just replaced parameter data tensors with
    # cloned/moved copies. Without an immediate sweep, the old mmap-backed
    # tensors stay alive until the next allocator pass — long enough on Windows
    # to block a subsequent large mmap (T5 cache) with OSError 1455.
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Phase 3 prefetch hooks. Only relevant if we actually have blocks on CPU
    # (otherwise streamed_linears is empty and the hooks would be no-ops anyway).
    if on_cpu_blocks:
        hooks_attached = _install_block_prefetch_hooks(root, on_cpu_blocks, device)
        summary["prefetch_hooks"] = hooks_attached
    else:
        summary["prefetch_hooks"] = 0

    log.info(
        "%s: streaming partition — total=%.2f GB · always-GPU=%.2f GB · "
        "blocks on GPU=%d · streamed from CPU=%d (%.2f GB) · budget left=%.2f GB "
        "· prefetch hooks=%d",
        label, summary["total_gb"], summary["non_block_gb"],
        summary["blocks_on_gpu"], summary["blocks_on_cpu"],
        summary["streamed_gb"], summary["budget_remaining_gb"],
        summary["prefetch_hooks"],
    )
    return summary


def count_streaming(root: nn.Module) -> tuple[int, int]:
    """(linears_total, linears_streaming) — for logging / smoke tests."""
    total = 0
    streaming = 0
    for m in root.modules():
        if isinstance(m, StreamingLinear):
            total += 1
            if m._kraken_stream_weights:
                streaming += 1
    return total, streaming


# ---- Phase 3: layer prefetch ---------------------------------------------
#
# Phase 2's mover stream did per-Linear H2D, but the compute stream still waited
# on the *same* layer's copy — so there was no real overlap, just a slightly
# nicer transfer mechanism. Phase 3 closes the gap: each streamed block's
# pre-forward hook kicks off the H2D for the *next* streamed block's weights,
# so by the time that next block's forward runs, the weights are already on
# GPU (or nearly so) and StreamingLinear.forward only waits on the prefetch
# event before reading.
#
# Chain layout:
#   streamed_block[0].pre_forward  → prefetch streamed_block[1]
#   streamed_block[1].pre_forward  → prefetch streamed_block[2]
#   ...
#   streamed_block[-1].pre_forward → prefetch streamed_block[0]   (wrap-around for next step)
#
# Wrap-around matters because each sampling step is one full transformer
# forward, and we want streamed_block[0]'s weights to be ready when step N+1
# begins. The very first forward of step 1 finds no prefetch in flight and
# falls back to the Phase 2 sync-streaming path — a one-time ~50 ms cost.

class _BlockInfo:
    """Bookkeeping for one streamed block: ordered list of its StreamingLinears
    and a pointer to the next streamed block in execution order."""
    __slots__ = ("name", "streamed_linears", "next_info", "event")

    def __init__(self, name: str, streamed_linears: list[StreamingLinear]) -> None:
        self.name = name
        self.streamed_linears = streamed_linears
        self.next_info: "_BlockInfo | None" = None
        self.event = torch.cuda.Event(enable_timing=False) if torch.cuda.is_available() else None


def _ordered_streamed_linears(block: nn.Module) -> list[StreamingLinear]:
    """All StreamingLinears in `block` that need streaming, in named_modules order.

    named_modules() walks the tree depth-first in registration order. For FLUX
    transformer blocks this gives the natural execution order (attn.to_q,
    attn.to_k, attn.to_v, attn.to_out.0, ff.net.0.proj, ff.net.2, etc.).
    """
    seen: set[int] = set()
    ordered: list[StreamingLinear] = []
    for _name, sub in block.named_modules():
        if isinstance(sub, StreamingLinear) and sub._kraken_stream_weights and id(sub) not in seen:
            ordered.append(sub)
            seen.add(id(sub))
    return ordered

def _interpret_buffer_views(flat_buffer: torch.Tensor, weights: list[torch.Tensor]) -> list[torch.Tensor]:
    views = []
    offset = 0
    for w in weights:
        size_bytes = w.numel() * w.element_size()
        slice_buf = flat_buffer[offset : offset + size_bytes]
        view_buf = slice_buf.view(w.dtype)
        reshaped = view_buf.view(w.shape)
        views.append(reshaped)
        offset += size_bytes
    return views


def _make_prefetch_hook(next_info: _BlockInfo, device: torch.device):
    """Return a pre-forward hook that kicks off H2D for `next_info`'s weights.

    ComfyUI-style two-stream pattern (`comfy/model_management.py:1226-1261`):
      - `_get_offload_stream()` round-robins across 2 streams, AND has the
        chosen stream wait for the compute stream first (so we don't race).
      - Each streamed Linear's GPU destination buffer was preallocated in
        `apply_streaming` (no per-call alloc). We `copy_` the CPU weight into
        the buffer on the offload stream — non_blocking=True works here
        because the source is pinned.
      - Pipeline depth = 2 across blocks: block N's hook may use stream A
        while block N+2's hook will pick stream B, so two copies are in
        flight at once.
      - StreamingLinear.forward will wait on the recorded event before
        reading the weight (see forward()).

    If flat buffer preallocation is active, we slice views from the global
    `_mover_buffers` corresponding to the chosen stream. Otherwise, we fall
    back to the old per-call allocation pattern so we stay correct.
    """
    def hook(_module, _args):
        stream = _get_offload_stream(device)
        if stream is None:
            return

        # Determine stream index to match with preallocated flat buffers
        stream_idx = None
        if _mover_buffers:
            try:
                stream_idx = _mover_streams.index(stream)
            except ValueError:
                pass

        any_queued = False
        with torch.cuda.stream(stream):
            # Attempt to use flat buffers if available
            if stream_idx is not None and stream_idx < len(_mover_buffers) and _mover_buffers[stream_idx] is not None:
                try:
                    flat_buf = _mover_buffers[stream_idx]
                    weights_to_stream = [sl.weight.data for sl in next_info.streamed_linears]
                    gpu_dests = _interpret_buffer_views(flat_buf, weights_to_stream)

                    for sl, gpu_dest in zip(next_info.streamed_linears, gpu_dests):
                        if sl.__dict__.get("_kraken_pending_w") is not None:
                            continue  # already prefetched

                        gpu_dest.copy_(sl.weight.data, non_blocking=True)
                        sl.__dict__["_kraken_pending_w"] = gpu_dest

                        if sl.bias is not None and sl.bias.device != device:
                            sl.__dict__["_kraken_pending_b"] = sl.bias.to(
                                device=device, non_blocking=True
                            )
                        else:
                            sl.__dict__["_kraken_pending_b"] = None
                        any_queued = True
                except Exception as e:
                    log.warning("Flat buffer prefetch copy failed (%s); falling back to per-call alloc", e)
                    any_queued = False

            # Fallback path (or when flat buffers are not used/available)
            if not any_queued:
                for sl in next_info.streamed_linears:
                    if sl.__dict__.get("_kraken_pending_w") is not None:
                        continue  # already prefetched
                    sl.__dict__["_kraken_pending_w"] = sl.weight.to(
                        device=device, non_blocking=True
                    )
                    if sl.bias is not None and sl.bias.device != device:
                        sl.__dict__["_kraken_pending_b"] = sl.bias.to(
                            device=device, non_blocking=True
                        )
                    else:
                        sl.__dict__["_kraken_pending_b"] = None
                    any_queued = True

            if any_queued and next_info.event is not None:
                event = next_info.event
                event.record(stream)
                for sl in next_info.streamed_linears:
                    sl.__dict__["_kraken_pending_event"] = event
    return hook


def _install_block_prefetch_hooks(
    root: nn.Module,
    streamed_block_names: list[str],
    device: torch.device,
) -> int:
    """Attach prefetch hooks to each streamed block. Returns count attached.

    Walks `root` to find each named streamed block, builds a _BlockInfo,
    chains them in order (wrap-around to first), then registers a pre-forward
    hook on each that kicks off the next block's prefetch.
    """
    infos: list[_BlockInfo] = []
    for name in streamed_block_names:
        block = root.get_submodule(name)
        sls = _ordered_streamed_linears(block)
        if sls:
            infos.append(_BlockInfo(name, sls))
    if not infos:
        return 0
    # Chain with wrap-around.
    for i, info in enumerate(infos):
        info.next_info = infos[(i + 1) % len(infos)]

    attached = 0
    for info in infos:
        block = root.get_submodule(info.name)
        next_info = info.next_info  # captured in closure; never None given infos non-empty
        assert next_info is not None
        block.register_forward_pre_hook(_make_prefetch_hook(next_info, device))
        attached += 1
    return attached
