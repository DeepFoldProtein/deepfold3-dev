"""Fused dropout (row- or column-shared mask) + residual add.

Replaces the pattern ``z = z + dropout_rowwise(x)`` / ``dropout_columnwise(x)``
with a single fused Triton kernel that:

  - Samples the mask on the fly via ``tl.rand(seed, mask_idx)`` — the mask is
    never materialized. For a 384-token pair rep ``[1, 384, 384, 128]`` this
    saves ``75 MB`` per call (× 5 drops × 48 layers × fwd+bwd ≈ 36 GB per step).
  - Folds the residual add into the same read/write pass, removing one HBM
    round-trip per drop site.
  - Regenerates the mask in backward from the saved seed — no stored mask.

API:
    dropout_add_rowwise(residual, x, p_drop, training) -> residual + dropout_rowwise(x)
    dropout_add_columnwise(residual, x, p_drop, training) -> residual + dropout_columnwise(x)

Both functions fall back to unfused PyTorch when Triton is unavailable, when
``training=False``, or when ``p_drop == 0``.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover - triton required on training GPUs
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _dropout_add_fwd_kernel(
        RES_ptr,
        X_ptr,
        OUT_ptr,
        N_total,
        RCD,  # R * C * D (batch stride)
        CD,  # C * D (stride along R axis)
        D,  # inner channel
        SHARED_STRIDE,  # mask block size per batch: CD (row) or R*D (col)
        ROWWISE: tl.constexpr,
        p_drop,
        seed,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        valid = offsets < N_total

        batch_idx = offsets // RCD
        local = offsets % RCD

        if ROWWISE:
            # Mask shared along R → mask_idx = batch * CD + (c*D + d)
            mask_idx = batch_idx * SHARED_STRIDE + (local % CD)
        else:
            # Mask shared along C → mask_idx = batch * R*D + r*D + d
            d_idx = local % D
            r_idx = local // CD
            mask_idx = batch_idx * SHARED_STRIDE + r_idx * D + d_idx

        r = tl.load(RES_ptr + offsets, mask=valid, other=0.0)
        x = tl.load(X_ptr + offsets, mask=valid, other=0.0)

        rand_vals = tl.rand(seed, mask_idx)
        keep = rand_vals >= p_drop
        scale = 1.0 / (1.0 - p_drop)
        out = r + tl.where(keep, x * scale, tl.zeros_like(x))
        tl.store(OUT_ptr + offsets, out, mask=valid)

    @triton.jit
    def _dropout_bwd_kernel(
        GRAD_OUT_ptr,
        GRAD_X_ptr,
        N_total,
        RCD,
        CD,
        D,
        SHARED_STRIDE,
        ROWWISE: tl.constexpr,
        p_drop,
        seed,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        valid = offsets < N_total

        batch_idx = offsets // RCD
        local = offsets % RCD

        if ROWWISE:
            mask_idx = batch_idx * SHARED_STRIDE + (local % CD)
        else:
            d_idx = local % D
            r_idx = local // CD
            mask_idx = batch_idx * SHARED_STRIDE + r_idx * D + d_idx

        grad_out = tl.load(GRAD_OUT_ptr + offsets, mask=valid, other=0.0)
        rand_vals = tl.rand(seed, mask_idx)
        keep = rand_vals >= p_drop
        scale = 1.0 / (1.0 - p_drop)
        grad_x = tl.where(keep, grad_out * scale, tl.zeros_like(grad_out))
        tl.store(GRAD_X_ptr + offsets, grad_x, mask=valid)

    class _FusedDropoutAdd(torch.autograd.Function):
        @staticmethod
        def forward(ctx, residual, x, p_drop, rowwise):
            assert residual.shape == x.shape
            assert x.ndim >= 3
            residual = residual.contiguous()
            x = x.contiguous()

            *batch_dims, R, C, D = x.shape
            B = 1
            for s in batch_dims:
                B *= s
            N_total = B * R * C * D
            RCD = R * C * D
            CD = C * D
            shared_stride = CD if rowwise else (R * D)

            seed = torch.randint(0, 2**31 - 1, (1,), device="cpu").item()
            out = torch.empty_like(x)
            BLOCK = 1024
            grid = ((N_total + BLOCK - 1) // BLOCK,)
            _dropout_add_fwd_kernel[grid](
                residual,
                x,
                out,
                N_total,
                RCD,
                CD,
                D,
                shared_stride,
                ROWWISE=rowwise,
                p_drop=p_drop,
                seed=seed,
                BLOCK=BLOCK,
            )
            ctx.save_for_backward(torch.tensor([seed], dtype=torch.int64))
            ctx.p_drop = p_drop
            ctx.shape_info = (N_total, RCD, CD, D, shared_stride)
            ctx.rowwise = rowwise
            return out

        @staticmethod
        def backward(ctx, grad_output):
            (seed_tensor,) = ctx.saved_tensors
            seed = int(seed_tensor.item())
            p_drop = ctx.p_drop
            N_total, RCD, CD, D, shared_stride = ctx.shape_info
            rowwise = ctx.rowwise

            grad_output = grad_output.contiguous()
            grad_x = torch.empty_like(grad_output)

            BLOCK = 1024
            grid = ((N_total + BLOCK - 1) // BLOCK,)
            _dropout_bwd_kernel[grid](
                grad_output,
                grad_x,
                N_total,
                RCD,
                CD,
                D,
                shared_stride,
                ROWWISE=rowwise,
                p_drop=p_drop,
                seed=seed,
                BLOCK=BLOCK,
            )
            return grad_output, grad_x, None, None


def _unfused_dropout_add(
    residual: torch.Tensor,
    x: torch.Tensor,
    p_drop: float,
    share_dim: int,
) -> torch.Tensor:
    shape = list(x.shape)
    shape[share_dim] = 1
    mask = x.new_ones(shape)
    mask = torch.nn.functional.dropout(mask, p=p_drop, training=True)
    return residual + x * mask


def dropout_add_rowwise(
    residual: torch.Tensor,
    x: torch.Tensor,
    p_drop: float,
    training: bool,
    use_fused: bool = True,
) -> torch.Tensor:
    """Fused row-wise dropout + residual add.

    Mask is shared along dim ``-3`` (row axis). For pair activations
    ``[..., N_row, N_col, C]`` or MSA activations ``[..., S, N, C]``, entire
    rows share the same drop decision — applied AFTER the update but BEFORE
    the residual add, in one fused pass.

    Set ``use_fused=False`` to force the unfused PyTorch fallback (A/B
    diagnostics). Wired from ``OptimConfig.use_fused_dropout_add``.
    """
    if not training or p_drop == 0.0:
        return residual + x
    if _HAS_TRITON and x.is_cuda and use_fused:
        return _FusedDropoutAdd.apply(residual, x, p_drop, True)
    return _unfused_dropout_add(residual, x, p_drop, share_dim=-3)


def dropout_add_columnwise(
    residual: torch.Tensor,
    x: torch.Tensor,
    p_drop: float,
    training: bool,
    use_fused: bool = True,
) -> torch.Tensor:
    """Fused column-wise dropout + residual add.

    Mask is shared along dim ``-2`` (column axis). Used for residual updates
    whose pattern is the transpose of the row-wise case (e.g. ending
    triangle attention).

    Set ``use_fused=False`` to force the unfused PyTorch fallback (A/B
    diagnostics). Wired from ``OptimConfig.use_fused_dropout_add``.
    """
    if not training or p_drop == 0.0:
        return residual + x
    if _HAS_TRITON and x.is_cuda and use_fused:
        return _FusedDropoutAdd.apply(residual, x, p_drop, False)
    return _unfused_dropout_add(residual, x, p_drop, share_dim=-2)
