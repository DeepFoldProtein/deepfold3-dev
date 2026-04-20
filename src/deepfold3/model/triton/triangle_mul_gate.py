"""Fused TriangleMul post-processing Triton kernel for DeepFold3.

Fuses the pre-einsum portion of _TriangleMulBase._compute():

    z0 = LN(z)                       # LayerNorm
    proj = proj_linear(z0)            # [M, 2*c_z]
    gate = sigmoid(gate_linear(z0))   # [M, 2*c_z]
    out_gate = sigmoid(out_gate_linear(z0))  # [M, c_z]
    proj = proj * gate
    a = proj[..., 0::2]
    b = proj[..., 1::2]

This module packs the 3 projections into a single FusedLayerNormLinear
(LN + concat projection), then applies an elementwise Triton kernel for
the sigmoid/gate/deinterleave post-processing.
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

try:
    from deepfold3.model.triton.layernorm_linear import HAS_TRITON as HAS_LNL
    from deepfold3.model.triton.layernorm_linear import FusedLayerNormLinear
except ImportError:
    HAS_LNL = False

HAS_ALL = HAS_TRITON and HAS_LNL

if HAS_TRITON:

    @triton.jit
    def _trimul_postprocess_fwd(
        combined_ptr,
        a_ptr,
        b_ptr,
        out_gate_ptr,
        M,
        K,
        stride_cm,
        stride_am,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Post-process combined [M, 5K] -> a[M,K], b[M,K], out_gate[M,K].

        combined layout: [proj(2K) | gate(2K) | out_gate(K)]
        proj and gate are interleaved: proj[0::2]=a_proj, proj[1::2]=b_proj, same for gate.
        """
        row = tl.program_id(0).to(tl.int64)
        col_offsets = tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < K

        combined_row = combined_ptr + row * stride_cm
        # proj: indices 0..2K-1, gate: 2K..4K-1, out_gate_raw: 4K..5K-1
        # proj even indices (a_proj): 0, 2, 4, ... -> combined[2*col]
        # proj odd indices (b_proj): 1, 3, 5, ... -> combined[2*col+1]
        a_proj = tl.load(combined_row + col_offsets * 2, mask=mask, other=0.0)
        b_proj = tl.load(combined_row + col_offsets * 2 + 1, mask=mask, other=0.0)
        # gate even (a_gate): 2K + 2*col
        a_gate_raw = tl.load(combined_row + 2 * K + col_offsets * 2, mask=mask, other=0.0)
        b_gate_raw = tl.load(combined_row + 2 * K + col_offsets * 2 + 1, mask=mask, other=0.0)
        out_gate_raw = tl.load(combined_row + 4 * K + col_offsets, mask=mask, other=0.0)

        # Sigmoid gates
        a_gate = tl.sigmoid(a_gate_raw.to(tl.float32))
        b_gate = tl.sigmoid(b_gate_raw.to(tl.float32))
        out_g = tl.sigmoid(out_gate_raw.to(tl.float32))

        # Apply gates
        a_val = a_proj * a_gate
        b_val = b_proj * b_gate

        tl.store(a_ptr + row * stride_am + col_offsets, a_val, mask=mask)
        tl.store(b_ptr + row * stride_am + col_offsets, b_val, mask=mask)
        tl.store(out_gate_ptr + row * stride_am + col_offsets, out_g, mask=mask)

    @triton.jit
    def _trimul_postprocess_bwd(
        da_ptr,
        db_ptr,
        d_out_gate_ptr,
        combined_ptr,
        d_combined_ptr,
        M,
        K,
        stride_am,
        stride_cm,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Backward for post-processing.

        Given da, db, d_out_gate, reconstruct d_combined [M, 5K].
        """
        row = tl.program_id(0).to(tl.int64)
        col_offsets = tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < K

        # Load upstream grads
        da = tl.load(da_ptr + row * stride_am + col_offsets, mask=mask, other=0.0)
        db = tl.load(db_ptr + row * stride_am + col_offsets, mask=mask, other=0.0)
        d_outg = tl.load(d_out_gate_ptr + row * stride_am + col_offsets, mask=mask, other=0.0)

        # Reload forward values from combined
        combined_row = combined_ptr + row * stride_cm
        a_proj = tl.load(combined_row + col_offsets * 2, mask=mask, other=0.0)
        b_proj = tl.load(combined_row + col_offsets * 2 + 1, mask=mask, other=0.0)
        a_gate_raw = tl.load(combined_row + 2 * K + col_offsets * 2, mask=mask, other=0.0)
        b_gate_raw = tl.load(combined_row + 2 * K + col_offsets * 2 + 1, mask=mask, other=0.0)
        out_gate_raw = tl.load(combined_row + 4 * K + col_offsets, mask=mask, other=0.0)

        a_gate = tl.sigmoid(a_gate_raw.to(tl.float32))
        b_gate = tl.sigmoid(b_gate_raw.to(tl.float32))
        out_gate = tl.sigmoid(out_gate_raw.to(tl.float32))

        # d_a_proj = da * sigmoid(a_gate_raw)
        d_a_proj = da * a_gate
        # d_a_gate_raw = da * a_proj * sigmoid'(a_gate_raw) = da * a_proj * sig * (1-sig)
        d_a_gate_raw = da * a_proj * a_gate * (1.0 - a_gate)

        d_b_proj = db * b_gate
        d_b_gate_raw = db * b_proj * b_gate * (1.0 - b_gate)

        # d_out_gate_raw = d_outg * sig * (1-sig)
        d_out_gate_raw = d_outg * out_gate * (1.0 - out_gate)

        # Write back to d_combined [M, 5K] in interleaved order
        d_combined_row = d_combined_ptr + row * stride_cm
        tl.store(d_combined_row + col_offsets * 2, d_a_proj, mask=mask)
        tl.store(d_combined_row + col_offsets * 2 + 1, d_b_proj, mask=mask)
        tl.store(d_combined_row + 2 * K + col_offsets * 2, d_a_gate_raw, mask=mask)
        tl.store(d_combined_row + 2 * K + col_offsets * 2 + 1, d_b_gate_raw, mask=mask)
        tl.store(d_combined_row + 4 * K + col_offsets, d_out_gate_raw, mask=mask)


class TriMulPostProcessFunction(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.bfloat16)
    def forward(ctx, combined, K):
        orig_shape = combined.shape
        combined_flat = combined.reshape(-1, orig_shape[-1]).contiguous()
        M = combined_flat.shape[0]

        a = torch.empty((M, K), dtype=combined_flat.dtype, device=combined_flat.device)
        b = torch.empty((M, K), dtype=combined_flat.dtype, device=combined_flat.device)
        out_gate = torch.empty((M, K), dtype=combined_flat.dtype, device=combined_flat.device)

        BLOCK_SIZE, num_warps = _calculate_settings(K)
        _trimul_postprocess_fwd[(M,)](
            combined_flat,
            a,
            b,
            out_gate,
            M,
            K,
            combined_flat.stride(0),
            a.stride(0),
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )

        # Save the tensor in its original shape so non-reentrant gradient
        # checkpointing sees matching metadata between original forward and
        # recomputed forward. The flat buffer is contiguous, so .view() shares
        # storage and we can re-flatten in backward for free.
        ctx.save_for_backward(combined_flat.view(orig_shape))
        ctx.K = K

        a = a.view(orig_shape[:-1] + (K,))
        b = b.view(orig_shape[:-1] + (K,))
        out_gate = out_gate.view(orig_shape[:-1] + (K,))
        return a, b, out_gate

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, da, db, d_out_gate):
        (combined_orig,) = ctx.saved_tensors
        K = ctx.K
        combined_flat = combined_orig.reshape(-1, 5 * K).contiguous()
        M = combined_flat.shape[0]

        da = da.reshape(M, K).contiguous()
        db = db.reshape(M, K).contiguous()
        d_out_gate = d_out_gate.reshape(M, K).contiguous()

        d_combined = torch.empty_like(combined_flat)

        BLOCK_SIZE, num_warps = _calculate_settings(K)
        _trimul_postprocess_bwd[(M,)](
            da,
            db,
            d_out_gate,
            combined_flat,
            d_combined,
            M,
            K,
            da.stride(0),
            combined_flat.stride(0),
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )

        return d_combined.view(combined_orig.shape), None


class FusedTriangleMulGate(nn.Module):
    """Fused pre-einsum processing for TriangleMul — STATELESS.

    Replaces:
        z0 = LN(z)
        proj = proj_linear(z0)            # [*, 2K]
        gate = sigmoid(gate_linear(z0))   # [*, 2K]
        out_gate = sigmoid(out_gate_linear(z0))  # [*, K]
        proj = proj * gate
        a = proj[..., 0::2]
        b = proj[..., 1::2]

    With:
        combined = FusedLayerNormLinear(z)   # [*, 5K]
        a, b, out_gate = TriMulPostProcess(combined, K)

    Holds no parameters; the caller (`_TriangleMulBase`) owns the weights and
    passes them at call time. The fused path packs the three linear weights
    into one [K, 5K] tensor via `torch.cat` — differentiable end-to-end, so
    gradients flow back into proj/gate/out_gate correctly.

    Args:
        c_z: Pair channel dimension.
    """

    def __init__(self, c_z: int):
        super().__init__()
        self.c_z = c_z

        # Stateless leaf.
        if HAS_ALL:
            self._ln_linear = FusedLayerNormLinear(c_z, 5 * c_z, has_linear_bias=False)
        else:
            self._ln_linear = None

    def forward(
        self,
        z: torch.Tensor,
        norm_weight: torch.Tensor,
        norm_bias: torch.Tensor,
        proj_weight: torch.Tensor,
        gate_weight: torch.Tensor,
        out_gate_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z: Input tensor [..., c_z].
            norm_weight: [c_z] LN scale.
            norm_bias: [c_z] LN offset.
            proj_weight: [2*c_z, c_z] projection weight (standard nn.Linear shape).
            gate_weight: [2*c_z, c_z] gate weight.
            out_gate_weight: [c_z, c_z] out-gate weight.

        Returns:
            (a, b, out_gate) each [..., c_z].
        """
        if HAS_ALL and z.is_cuda and self._ln_linear is not None:
            # Pack [proj.T | gate.T | out_gate.T] as [K, 5K] in one fresh
            # tensor. torch.cat is differentiable — autograd routes grads back
            # into the three separate weights.
            packed = torch.cat([proj_weight.T, gate_weight.T, out_gate_weight.T], dim=1)
            combined = self._ln_linear(z, norm_weight, norm_bias, packed)  # [..., 5K]
            a, b, out_gate = TriMulPostProcessFunction.apply(combined, self.c_z)
            return a, b, out_gate

        # Fallback (CPU or Triton unavailable). Construct functional LN + linear ops.
        z0 = F.layer_norm(z, (self.c_z,), norm_weight, norm_bias)
        proj = F.linear(z0, proj_weight)
        gate = torch.sigmoid(F.linear(z0, gate_weight))
        out_gate = torch.sigmoid(F.linear(z0, out_gate_weight))
        proj = proj * gate
        a = proj[..., 0::2]
        b = proj[..., 1::2]
        return a, b, out_gate
