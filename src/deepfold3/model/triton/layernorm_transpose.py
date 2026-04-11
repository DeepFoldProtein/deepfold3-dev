"""Fused LayerNorm + dim-transpose Triton kernel for DeepFold3.

Fuses nn.LayerNorm on the last dimension with a transpose of two middle
dimensions in a single memory pass:

    out = layer_norm(x, [C]).permute(0, 2, 1, 3)    # [B, N1, N2, C] → [B, N2, N1, C]

Without fusion this requires two full memory passes (LN writes intermediate,
permute reads and writes). The fused kernel reads x once, normalizes, and
writes directly to the transposed output location.

When transpose is disabled (do_transpose=False), this is a standard LayerNorm.

Falls back to F.layer_norm(x, [C], w, b).permute(0, 2, 1, 3) when Triton is
unavailable or input is not on CUDA.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

from deepfold3.model.triton import calculate_settings as _calculate_settings

if HAS_TRITON:

    @triton.jit
    def _ln_transpose_fwd_kernel(
        X_ptr,
        OUT_ptr,
        W_ptr,
        B_ptr,
        Mean_ptr,
        RSTD_ptr,
        N1,
        N2,
        C,
        eps,
        DO_TRANSPOSE: tl.constexpr,
        HAS_AFFINE: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        """Forward: LayerNorm over C, optionally transposing dims 1 and 2.

        Grid: (B * N1 * N2,) — one program per (b, n1, n2) row.
        """
        row = tl.program_id(0).to(tl.int64)
        cols = tl.arange(0, BLOCK_C)
        mask = cols < C

        # Read input row: x[b, n1, n2, :] at contiguous offset row * C
        x = tl.load(X_ptr + row * C + cols, mask=mask, other=0.0).to(tl.float32)

        # LayerNorm statistics
        mean = tl.sum(x, axis=0) / C
        xc = x - mean
        var = tl.sum(xc * xc, axis=0) / C
        rstd = 1.0 / tl.sqrt(var + eps)
        xn = xc * rstd

        # Affine transform
        if HAS_AFFINE:
            w = tl.load(W_ptr + cols, mask=mask, other=1.0).to(tl.float32)
            b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            xn = xn * w + b

        # Save statistics for backward
        tl.store(Mean_ptr + row, mean)
        tl.store(RSTD_ptr + row, rstd)

        # Compute output offset: optionally transpose dims 1 and 2
        if DO_TRANSPOSE:
            # row = b * N1 * N2 + n1 * N2 + n2
            # output row = b * N1 * N2 + n2 * N1 + n1
            n2_idx = row % N2
            rem = row // N2
            n1_idx = rem % N1
            b_idx = rem // N1
            out_row = b_idx * (N1 * N2) + n2_idx * N1 + n1_idx
        else:
            out_row = row

        tl.store(OUT_ptr + out_row * C + cols, xn, mask=mask)

    @triton.jit
    def _ln_transpose_bwd_kernel(
        GRAD_OUT_ptr,
        X_ptr,
        W_ptr,
        Mean_ptr,
        RSTD_ptr,
        GRAD_X_ptr,
        PARTIAL_DW_ptr,
        PARTIAL_DB_ptr,
        N1,
        N2,
        C,
        NUM_PARTITIONS,
        DO_TRANSPOSE: tl.constexpr,
        HAS_AFFINE: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        """Backward: compute grad_x and accumulate partial grad_w/grad_b.

        Grid: (B * N1 * N2,) — one program per input row.
        Each program writes grad_x for its row and atomically accumulates
        xhat*dy and dy into a partitioned [NUM_PARTITIONS, C] buffer.
        Partitioning spreads atomic contention: with P partitions and ~680
        concurrent programs, only ~680/P ≈ 2-3 programs contend per slot.
        """
        row = tl.program_id(0).to(tl.int64)
        cols = tl.arange(0, BLOCK_C)
        mask = cols < C

        # Where is this row's gradient in the output tensor?
        if DO_TRANSPOSE:
            n2_idx = row % N2
            rem = row // N2
            n1_idx = rem % N1
            b_idx = rem // N1
            grad_row = b_idx * (N1 * N2) + n2_idx * N1 + n1_idx
        else:
            grad_row = row

        # Load grad_out, x, stats
        dy = tl.load(GRAD_OUT_ptr + grad_row * C + cols, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(X_ptr + row * C + cols, mask=mask, other=0.0).to(tl.float32)
        mean = tl.load(Mean_ptr + row)
        rstd = tl.load(RSTD_ptr + row)

        xhat = (x - mean) * rstd

        if HAS_AFFINE:
            w = tl.load(W_ptr + cols, mask=mask, other=1.0).to(tl.float32)
            wdy = w * dy
        else:
            wdy = dy

        # Standard LN backward: dx = (wdy - mean(xhat*wdy)*xhat - mean(wdy)) * rstd
        c1 = tl.sum(xhat * wdy, axis=0) / C
        c2 = tl.sum(wdy, axis=0) / C
        dx = (wdy - c1 * xhat - c2) * rstd

        tl.store(GRAD_X_ptr + row * C + cols, dx, mask=mask)

        # Accumulate grad_w/grad_b via partitioned atomics
        if HAS_AFFINE:
            part_id = tl.program_id(0) % NUM_PARTITIONS
            part_offset = part_id.to(tl.int64) * C + cols
            tl.atomic_add(PARTIAL_DW_ptr + part_offset, xhat * dy, mask=mask)
            tl.atomic_add(PARTIAL_DB_ptr + part_offset, dy, mask=mask)


class LayerNormTransposeFunction(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.bfloat16)
    def forward(ctx, x, weight, bias, N1, N2, eps, do_transpose):
        B = x.shape[0]
        C = x.shape[-1]
        M = B * N1 * N2
        x_flat = x.reshape(M, C).contiguous()

        out_flat = torch.empty_like(x_flat)
        mean = torch.empty(M, dtype=torch.float32, device=x.device)
        rstd = torch.empty(M, dtype=torch.float32, device=x.device)

        BLOCK_C, num_warps = _calculate_settings(C)
        has_affine = weight is not None

        _ln_transpose_fwd_kernel[(M,)](
            x_flat,
            out_flat,
            weight if has_affine else x_flat,  # dummy ptr when no affine
            bias if has_affine else x_flat,
            mean,
            rstd,
            N1,
            N2,
            C,
            eps,
            DO_TRANSPOSE=do_transpose,
            HAS_AFFINE=has_affine,
            BLOCK_C=BLOCK_C,
            num_warps=num_warps,
        )

        if do_transpose:
            out = out_flat.view(B, N2, N1, C)
        else:
            out = out_flat.view(B, N1, N2, C)

        # Save x in its original 4D shape so non-reentrant gradient
        # checkpointing sees matching metadata between original and recomputed
        # forward. The flat buffer is contiguous; .view() shares storage and
        # we re-flatten in backward for free.
        ctx.save_for_backward(x_flat.view(B, N1, N2, C), weight, mean, rstd)
        ctx.N1 = N1
        ctx.N2 = N2
        ctx.C = C
        ctx.do_transpose = do_transpose
        ctx.has_affine = has_affine
        return out

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_out):
        x_orig, weight, mean, rstd = ctx.saved_tensors
        N1, N2, C = ctx.N1, ctx.N2, ctx.C
        B = x_orig.shape[0]
        M = B * N1 * N2
        x_flat = x_orig.reshape(M, C).contiguous()

        grad_out_flat = grad_out.reshape(M, C).contiguous()
        grad_x_flat = torch.empty_like(x_flat)

        BLOCK_C, num_warps = _calculate_settings(C)

        # Partitioned atomic accumulation for grad_w/grad_b.
        # Spreads contention: ~680 concurrent programs / 256 partitions ≈ 2-3
        # programs per slot. Buffer is [256, C] ≈ 128KB (vs ~450MB old way).
        NUM_PARTITIONS = min(M, 256)
        if ctx.has_affine:
            partial_dw = torch.zeros(NUM_PARTITIONS, C, dtype=torch.float32, device=x_flat.device)
            partial_db = torch.zeros(NUM_PARTITIONS, C, dtype=torch.float32, device=x_flat.device)
        else:
            partial_dw = x_flat  # dummy ptr
            partial_db = x_flat

        _ln_transpose_bwd_kernel[(M,)](
            grad_out_flat,
            x_flat,
            weight if ctx.has_affine else x_flat,
            mean,
            rstd,
            grad_x_flat,
            partial_dw,
            partial_db,
            N1,
            N2,
            C,
            NUM_PARTITIONS,
            DO_TRANSPOSE=ctx.do_transpose,
            HAS_AFFINE=ctx.has_affine,
            BLOCK_C=BLOCK_C,
            num_warps=num_warps,
        )

        grad_x = grad_x_flat.view(B, N1, N2, C)

        if ctx.has_affine:
            grad_weight = partial_dw.sum(dim=0).to(x_flat.dtype)
            grad_bias = partial_db.sum(dim=0).to(x_flat.dtype)
        else:
            grad_weight = None
            grad_bias = None

        return grad_x, grad_weight, grad_bias, None, None, None, None


class FusedLayerNormTranspose(nn.Module):
    """Fused LayerNorm + optional dim transpose.

    Replaces:
        out = F.layer_norm(x, [C], weight, bias)
        if transpose:
            out = out.permute(0, 2, 1, 3)

    Input must be 4D: [B, N1, N2, C]. When do_transpose=True, output is
    [B, N2, N1, C]. LN is always over the last dimension C.

    Args:
        normalized_shape: Size of the last dimension (C).
        do_transpose: Whether to transpose dims 1 and 2 in the output.
        eps: LN epsilon.
    """

    def __init__(self, normalized_shape: int, do_transpose: bool = True, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.do_transpose = do_transpose
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N1, N2, C] — must be 4D with batch dim.
        Returns:
            [B, N2, N1, C] if do_transpose else [B, N1, N2, C].
        """
        if not HAS_TRITON or not x.is_cuda:
            out = F.layer_norm(x, (self.normalized_shape,), self.weight, self.bias, self.eps)
            if self.do_transpose:
                out = out.permute(0, 2, 1, 3).contiguous()
            return out

        B, N1, N2, C = x.shape
        return LayerNormTransposeFunction.apply(x, self.weight, self.bias, N1, N2, self.eps, self.do_transpose)

    def extra_repr(self):
        return f"normalized_shape={self.normalized_shape}, do_transpose={self.do_transpose}, eps={self.eps}"
