"""Fused SwiGLU Triton kernel for DeepFold3.

Computes silu(a) * b where [a, b] = split(x, 2, dim=-1).
This matches DeepFold3's convention (silu on the FIRST half).

Adapted from MegaFold's FusedSwiGLU kernel with convention swap:
MegaFold computes silu(b) * a; DeepFold3 computes silu(a) * b.
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


from deepfold3.model.triton import calculate_settings as _calculate_settings

if HAS_TRITON:

    @triton.jit
    def _swiglu_forward_kernel(x_ptr, y_ptr, stride_y, D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
        """Forward: y = silu(a) * b where x = [a, b]."""
        row_idx = tl.program_id(axis=0).to(tl.int64)

        # DeepFold3 convention: a = left half (apply silu), b = right half
        a_ptr = x_ptr + row_idx * stride_y * 2
        b_ptr = x_ptr + row_idx * stride_y * 2 + stride_y
        y_ptr += row_idx * stride_y

        col_offsets = tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < stride_y

        a_row = tl.load(a_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
        b_row = tl.load(b_ptr + col_offsets, mask=mask, other=0)
        c_row = a_row * tl.sigmoid(a_row) * b_row  # silu(a) * b

        tl.store(y_ptr + col_offsets, c_row, mask=mask)

    @triton.jit
    def _swiglu_backward_kernel(dy_ptr, x_ptr, stride, D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
        """Backward: compute da, db in-place into x buffer."""
        row_idx = tl.program_id(axis=0).to(tl.int64)

        dy_ptr += row_idx * stride
        a_ptr = x_ptr + row_idx * stride * 2
        b_ptr = x_ptr + row_idx * stride * 2 + stride

        col_offsets = tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < D

        dy_row = tl.load(dy_ptr + col_offsets, mask=mask, other=0.0)
        a_row = tl.load(a_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
        b_row = tl.load(b_ptr + col_offsets, mask=mask, other=0.0)

        sig_a = tl.sigmoid(a_row)
        silu_a = a_row * sig_a
        # da = dy * b * (silu(a) * (1 - sig(a)) + sig(a))  = dy * b * d(silu(a))/da
        da_row = dy_row * (silu_a * (1 - sig_a) + sig_a) * b_row
        # db = dy * silu(a)
        db_row = dy_row * silu_a

        tl.store(a_ptr + col_offsets, da_row, mask=mask)
        tl.store(b_ptr + col_offsets, db_row, mask=mask)


def swiglu_forward(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward pass: x=[a,b] -> y=silu(a)*b. Returns (x_saved, y)."""
    ori_shape = x.shape
    D = ori_shape[-1]
    dim_out = D // 2

    x = x.view(-1, D)
    M = x.shape[0]

    y = torch.empty((M, dim_out), device=x.device, dtype=x.dtype)

    BLOCK_SIZE, num_warps = _calculate_settings(dim_out)
    _swiglu_forward_kernel[(M,)](
        x,
        y,
        y.stride(0),
        D=D,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return x, y.view(ori_shape[:-1] + (dim_out,))


def swiglu_backward(x: torch.Tensor, dy: torch.Tensor) -> torch.Tensor:
    """Backward pass: writes gradients in-place into x buffer."""
    ori_shape = dy.shape
    D = ori_shape[-1]
    dy = dy.view(-1, D)
    M = dy.shape[0]

    BLOCK_SIZE, num_warps = _calculate_settings(D)
    _swiglu_backward_kernel[(M,)](
        dy,
        x,
        dy.stride(0),
        D=D,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return x.view(ori_shape[:-1] + (D * 2,))


class FusedSwiGLUFunction(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.bfloat16)
    def forward(ctx, input):
        input = input.contiguous()
        input, output = swiglu_forward(input)
        ctx.save_for_backward(input)
        return output

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_input = swiglu_backward(input, grad_output)
        return grad_input


class FusedSwiGLU(nn.Module):
    """Drop-in replacement for SwiGLU using Triton kernels.

    Computes silu(a) * b where [a, b] = split(x, 2, dim=-1).
    Same forward signature as deepfold3.model.primitives.SwiGLU.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return FusedSwiGLUFunction.apply(x)
