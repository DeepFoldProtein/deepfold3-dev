"""Tiled linear computation for memory-efficient pair transitions.

Chunks input along a spatial dimension and processes each chunk independently.
During training, intermediates are not stored — each chunk's forward is
recomputed during backward to compute gradients on-demand (Liger-Kernel style).
During inference, chunking reduces peak memory for very long sequences.

All operations in TransitionBlock (LayerNorm, Linear, SwiGLU) act on the last
dimension independently per spatial position, so chunking along dim=-2 produces
bitwise identical results to the full forward pass.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def _auto_shards(x: torch.Tensor) -> int:
    """Compute number of shards from input shape.

    Heuristic: ceil(spatial_dim / channel_dim). For pair [N, N, 128] with
    N=384 this gives ceil(384/128) = 3 shards.
    """
    return max(1, math.ceil(x.shape[-2] / x.shape[-1]))


class _TiledForwardFn(torch.autograd.Function):
    """Tiled forward with recomputation-based backward.

    Forward: chunks input along dim=-2, runs each through module with no_grad.
    Backward: recomputes each chunk's forward to get gradients on-demand.
    Parameter gradients accumulate across chunks automatically.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, module: nn.Module, num_shards: int) -> torch.Tensor:
        ctx.module = module
        ctx.num_shards = num_shards
        ctx.save_for_backward(x)

        chunks = x.chunk(num_shards, dim=-2)
        with torch.no_grad():
            outputs = [module(chunk) for chunk in chunks]
        return torch.cat(outputs, dim=-2)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (x,) = ctx.saved_tensors
        chunks_x = x.chunk(ctx.num_shards, dim=-2)
        chunks_grad = grad_output.chunk(ctx.num_shards, dim=-2)

        grad_inputs = []
        for x_chunk, g_chunk in zip(chunks_x, chunks_grad, strict=True):
            x_chunk = x_chunk.detach().requires_grad_(True)
            with torch.enable_grad():
                out = ctx.module(x_chunk)
            torch.autograd.backward(out, g_chunk)
            grad_inputs.append(x_chunk.grad)

        return torch.cat(grad_inputs, dim=-2), None, None


class _TiledAdaForwardFn(torch.autograd.Function):
    """Tiled forward for modules that take (x, cond) — e.g. AdaTransitionBlock.

    Both x and cond are chunked along dim=-2 (they share spatial layout).
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, cond: torch.Tensor, module: nn.Module, num_shards: int) -> torch.Tensor:
        ctx.module = module
        ctx.num_shards = num_shards
        ctx.save_for_backward(x, cond)

        chunks_x = x.chunk(num_shards, dim=-2)
        chunks_c = cond.chunk(num_shards, dim=-2)
        with torch.no_grad():
            outputs = [module(xc, cc) for xc, cc in zip(chunks_x, chunks_c, strict=True)]
        return torch.cat(outputs, dim=-2)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, cond = ctx.saved_tensors
        chunks_x = x.chunk(ctx.num_shards, dim=-2)
        chunks_c = cond.chunk(ctx.num_shards, dim=-2)
        chunks_grad = grad_output.chunk(ctx.num_shards, dim=-2)

        grad_inputs = []
        grad_conds = []
        for x_chunk, c_chunk, g_chunk in zip(chunks_x, chunks_c, chunks_grad, strict=True):
            x_chunk = x_chunk.detach().requires_grad_(True)
            c_chunk = c_chunk.detach().requires_grad_(True)
            with torch.enable_grad():
                out = ctx.module(x_chunk, c_chunk)
            torch.autograd.backward(out, g_chunk)
            grad_inputs.append(x_chunk.grad)
            grad_conds.append(c_chunk.grad)

        return torch.cat(grad_inputs, dim=-2), torch.cat(grad_conds, dim=-2), None, None


class _TiledOPMContractFn(torch.autograd.Function):
    """2D-tiled outer product mean contraction with recomputation-based backward.

    Chunks both `a` (dim=2, rows) and `b` (dim=2, cols) so the intermediate
    is [B, chunk_i, chunk_j, c_h, c_h] — constant in N.
    """

    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor, contract_fn, num_shards: int) -> torch.Tensor:
        ctx.contract_fn = contract_fn
        ctx.num_shards = num_shards
        ctx.save_for_backward(a, b)

        chunks_a = a.chunk(num_shards, dim=2)
        chunks_b = b.chunk(num_shards, dim=2)
        with torch.no_grad():
            row_outputs = []
            for a_chunk in chunks_a:
                col_outputs = [contract_fn(a_chunk, b_chunk) for b_chunk in chunks_b]
                row_outputs.append(torch.cat(col_outputs, dim=2))  # cat along j
            return torch.cat(row_outputs, dim=1)  # cat along i

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        a, b = ctx.saved_tensors
        chunks_a = a.chunk(ctx.num_shards, dim=2)
        chunks_b = b.chunk(ctx.num_shards, dim=2)
        # Output is [B, N, N, c_z], chunked along dim=1 (i) and dim=2 (j)
        chunks_grad_i = grad_output.chunk(ctx.num_shards, dim=1)

        grad_a_chunks = []
        grad_b_chunks = [torch.zeros_like(bc) for bc in chunks_b]
        for a_chunk, g_row in zip(chunks_a, chunks_grad_i, strict=True):
            chunks_grad_ij = g_row.chunk(ctx.num_shards, dim=2)
            grad_a_accum = torch.zeros_like(a_chunk)
            for j, (b_chunk, g_chunk) in enumerate(zip(chunks_b, chunks_grad_ij, strict=True)):
                a_det = a_chunk.detach().requires_grad_(True)
                b_det = b_chunk.detach().requires_grad_(True)
                with torch.enable_grad():
                    out = ctx.contract_fn(a_det, b_det)
                torch.autograd.backward(out, g_chunk)
                grad_a_accum += a_det.grad
                grad_b_chunks[j] += b_det.grad
            grad_a_chunks.append(grad_a_accum)

        return torch.cat(grad_a_chunks, dim=2), torch.cat(grad_b_chunks, dim=2), None, None


class _TransparentWrapper(nn.Module):
    """Base class for wrappers with transparent state dict keys.

    Re-registers inner module's parameters, buffers, and submodules directly
    on the wrapper so that state_dict keys match the unwrapped module exactly.
    The inner module and wrapper share the same parameter objects.
    """

    def __init__(self, inner: nn.Module):
        super().__init__()
        # Re-register inner's direct attributes on self for transparent state dict
        for name, param in inner._parameters.items():
            self.register_parameter(name, param)
        for name, buf in inner._buffers.items():
            self.register_buffer(name, buf)
        for name, mod in inner._modules.items():
            self.add_module(name, mod)
        # Keep a reference to inner for calling its forward method
        object.__setattr__(self, "_inner_ref", inner)

    @property
    def inner(self) -> nn.Module:
        return self._inner_ref


class TiledLinear(_TransparentWrapper):
    """Transparent wrapper that tiles computation along dim=-2.

    During training (requires_grad): uses custom autograd function that
    recomputes each chunk's forward during backward, avoiding storage of
    intermediate activations.

    During inference: optionally chunks forward pass to reduce peak memory
    for very long sequences (controlled by tile_inference flag).

    State dict is transparent — keys match the inner module exactly (no
    '_inner.' prefix) for full checkpoint compatibility.

    Args:
        inner: The module to wrap (TransitionBlock, FusedTransitionBlock, etc.)
        num_shards: Number of chunks along dim=-2. 0 = auto heuristic.
        tile_inference: Also tile during inference (for memory-constrained cases).
    """

    def __init__(self, inner: nn.Module, num_shards: int = 0, tile_inference: bool = False):
        super().__init__(inner)
        self._num_shards = num_shards
        self._tile_inference = tile_inference

    @torch.compiler.disable
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shards = self._num_shards or _auto_shards(x)
        if shards <= 1:
            return self.inner(x)

        if self.training and x.requires_grad:
            return _TiledForwardFn.apply(x, self.inner, shards)

        if self._tile_inference or self.training:
            # Chunked forward (no recomputation needed without grad)
            chunks = x.chunk(shards, dim=-2)
            return torch.cat([self.inner(chunk) for chunk in chunks], dim=-2)

        return self.inner(x)

    def __repr__(self) -> str:
        return f"TiledLinear(shards={self._num_shards or 'auto'}, tile_inference={self._tile_inference}, inner={self.inner})"


class TiledAdaLinear(_TransparentWrapper):
    """Transparent wrapper for modules taking (x, cond) — e.g. AdaTransitionBlock.

    Same as TiledLinear but handles the conditioning input. Both x and cond
    are chunked along dim=-2.
    """

    def __init__(self, inner: nn.Module, num_shards: int = 0, tile_inference: bool = False):
        super().__init__(inner)
        self._num_shards = num_shards
        self._tile_inference = tile_inference

    @torch.compiler.disable
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shards = self._num_shards or _auto_shards(x)
        if shards <= 1:
            return self.inner(x, cond)

        if self.training and x.requires_grad:
            return _TiledAdaForwardFn.apply(x, cond, self.inner, shards)

        if self._tile_inference or self.training:
            chunks_x = x.chunk(shards, dim=-2)
            chunks_c = cond.chunk(shards, dim=-2)
            return torch.cat([self.inner(xc, cc) for xc, cc in zip(chunks_x, chunks_c, strict=True)], dim=-2)

        return self.inner(x, cond)

    def __repr__(self) -> str:
        return f"TiledAdaLinear(shards={self._num_shards or 'auto'}, tile_inference={self._tile_inference}, inner={self.inner})"
