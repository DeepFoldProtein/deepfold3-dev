"""Fused SwiGLU Triton kernel for DeepFold3.

Computes silu(a) * b where [a, b] = split(x, 2, dim=-1).
This matches DeepFold3's convention (silu on the FIRST half).

Performance note: At DeepFold3's channel dimensions (C=128→D=64, C=384→D=192),
the naive 1-row-per-block Triton kernel loses to PyTorch's vectorized CUDA
elementwise kernels because grid dispatch overhead dominates the trivial per-row
compute.  Fix: process ROWS_PER_BLOCK rows per thread block via tl.static_range,
reducing grid size 32-128x.  Each block does more work, amortizing launch cost
and improving SM occupancy on high-SM-count GPUs (e.g. 170 SMs on RTX 5090).
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


if HAS_TRITON:

    @triton.jit
    def _swiglu_forward_kernel(
        x_ptr,
        y_ptr,
        stride_x,  # stride between rows of x (= 2*D for contiguous)
        stride_y,  # stride between rows of y (= D for contiguous)
        M,  # total number of rows
        D: tl.constexpr,  # output dimension (half of input)
        BLOCK_D: tl.constexpr,  # >= D, power of 2
        ROWS_PER_BLOCK: tl.constexpr,  # rows processed per thread block
    ):
        """Forward: y = silu(a) * b where x = [a | b] concatenated."""
        pid = tl.program_id(0).to(tl.int64)
        row_start = pid * ROWS_PER_BLOCK

        cols = tl.arange(0, BLOCK_D)
        col_mask = cols < D

        for i in tl.static_range(ROWS_PER_BLOCK):
            row = row_start + i
            if row < M:
                a_off = row * stride_x + cols
                b_off = row * stride_x + D + cols
                y_off = row * stride_y + cols

                a = tl.load(x_ptr + a_off, mask=col_mask, other=0).to(tl.float32)
                b = tl.load(x_ptr + b_off, mask=col_mask, other=0)
                y = (a * tl.sigmoid(a)).to(b.dtype) * b
                tl.store(y_ptr + y_off, y, mask=col_mask)

    @triton.jit
    def _swiglu_backward_kernel(
        dy_ptr,
        x_ptr,
        stride_dy,
        stride_x,
        M,
        D: tl.constexpr,
        BLOCK_D: tl.constexpr,
        ROWS_PER_BLOCK: tl.constexpr,
    ):
        """Backward: compute da, db in-place into x buffer."""
        pid = tl.program_id(0).to(tl.int64)
        row_start = pid * ROWS_PER_BLOCK

        cols = tl.arange(0, BLOCK_D)
        col_mask = cols < D

        for i in tl.static_range(ROWS_PER_BLOCK):
            row = row_start + i
            if row < M:
                dy_off = row * stride_dy + cols
                a_off = row * stride_x + cols
                b_off = row * stride_x + D + cols

                dy = tl.load(dy_ptr + dy_off, mask=col_mask, other=0)
                a = tl.load(x_ptr + a_off, mask=col_mask, other=0).to(tl.float32)
                b = tl.load(x_ptr + b_off, mask=col_mask, other=0)

                sig_a = tl.sigmoid(a)
                silu_a = a * sig_a
                da = dy * (silu_a * (1 - sig_a) + sig_a) * b
                db = dy * silu_a

                tl.store(x_ptr + a_off, da, mask=col_mask)
                tl.store(x_ptr + b_off, db, mask=col_mask)


def _kernel_config(D: int, M: int) -> tuple[int, int, int]:
    """Choose BLOCK_D, ROWS_PER_BLOCK, num_warps for given dimensions."""
    BLOCK_D = max(triton.next_power_of_2(D), 16)

    # Target: enough blocks to fill SMs (aim for ~4 waves on 170 SMs),
    # but each block should do meaningful work (32-128 rows).
    if BLOCK_D <= 64:
        # Small D: each row is tiny, process many rows per block.
        ROWS_PER_BLOCK = 128
        num_warps = 2
    elif BLOCK_D <= 256:
        ROWS_PER_BLOCK = 64
        num_warps = 2
    else:
        ROWS_PER_BLOCK = 32
        num_warps = 4

    # Don't over-chunk for small M (avoid blocks with no work)
    if M < ROWS_PER_BLOCK * 128:
        ROWS_PER_BLOCK = max(ROWS_PER_BLOCK // 4, 1)

    return BLOCK_D, ROWS_PER_BLOCK, num_warps


def swiglu_forward(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward pass: x=[a,b] -> y=silu(a)*b. Returns (x_saved, y)."""
    ori_shape = x.shape
    D_full = ori_shape[-1]
    D = D_full // 2

    x = x.view(-1, D_full)
    M = x.shape[0]

    y = torch.empty((M, D), device=x.device, dtype=x.dtype)

    BLOCK_D, ROWS_PER_BLOCK, num_warps = _kernel_config(D, M)
    grid = (triton.cdiv(M, ROWS_PER_BLOCK),)

    _swiglu_forward_kernel[grid](
        x,
        y,
        x.stride(0),
        y.stride(0),
        M,
        D=D,
        BLOCK_D=BLOCK_D,
        ROWS_PER_BLOCK=ROWS_PER_BLOCK,
        num_warps=num_warps,
    )
    return x, y.view(ori_shape[:-1] + (D,))


def swiglu_backward(x: torch.Tensor, dy: torch.Tensor) -> torch.Tensor:
    """Backward pass: writes gradients in-place into x buffer."""
    ori_shape = dy.shape
    D = ori_shape[-1]
    dy = dy.view(-1, D)
    M = dy.shape[0]

    BLOCK_D, ROWS_PER_BLOCK, num_warps = _kernel_config(D, M)
    grid = (triton.cdiv(M, ROWS_PER_BLOCK),)

    _swiglu_backward_kernel[grid](
        dy,
        x,
        dy.stride(0),
        x.stride(0),
        M,
        D=D,
        BLOCK_D=BLOCK_D,
        ROWS_PER_BLOCK=ROWS_PER_BLOCK,
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
