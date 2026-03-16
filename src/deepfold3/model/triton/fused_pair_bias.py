"""Fused pair bias kernel: LN + Linear projection + layout permute + optional mask.

Replaces the GridSelfAttention pair bias preparation:

    z = layer_norm(z, [C], weight, bias)        # [B, N, N, C]
    pb = pair_bias_linear(z)                     # [B, N, N, H]  (H=4, C=128)
    pb = pb.permute(0, 3, 1, 2)                 # [B, H, N, N]

with a single kernel that reads z once and writes directly in [B, H, N, N]
layout, eliminating the intermediate LN'd tensor and the permute copy.

Design (adapted from NVIDIA cuEquivariance pair_bias.py pattern):
─────────────────────────────────────────────────────────────────
cuEQ uses a 3D grid (V_tiles × U × H_blocks×B) where each program
processes a tile of V positions for a block of heads, fusing LN + Linear
projection + mask into one pass. The C→H projection uses tl.dot with
TILE_K tiling, which is orders of magnitude faster than our previous
scalar dot-product loop.

Grid: (cdiv(V, TILE_V), U, cdiv(NUM_HEADS, NUM_HEADS_PER_BLK) * B)

Each program:
  1. Computes LN statistics (mean, rstd) for a tile of TILE_V positions
  2. Normalizes z in tiles of TILE_K along C dimension
  3. Projects to heads via tl.dot (tensor core GEMM)
  4. Optionally applies mask: output + (1.0 - mask) * (-1e9)
  5. Writes output in [B, H, U, V] layout

For training, backward uses PyTorch ops on saved intermediates.
Falls back to separate LN + Linear + permute when Triton is unavailable.
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

from deepfold3.model.triton import Precision
from deepfold3.model.triton import calculate_settings as _calculate_settings

if HAS_TRITON:
    from deepfold3.model.triton import precision_dot

    @triton.jit
    def _fused_pair_bias_fwd_kernel(
        Z_ptr,  # [B, U, V, C] input pair activations
        W_ln_ptr,  # [C] LN weight
        B_ln_ptr,  # [C] LN bias
        W_proj_ptr,  # [H, C] projection weight
        B_proj_ptr,  # [H] projection bias (optional)
        mask_ptr,  # [B, V] mask (optional)
        OUT_ptr,  # [B, H, U, V] output pair bias
        z_proj_ptr,  # [B, H, U, V] cached z_proj (optional)
        Mean_ptr,  # [B, U, V] saved mean (training)
        RSTD_ptr,  # [B, U, V] saved rstd (training)
        Z_norm_ptr,  # [B, U, V, C] saved z_norm (training)
        U,
        V,
        C: tl.constexpr,
        eps,
        stride_zb,
        stride_zu,
        stride_zv,
        stride_zc,
        stride_ob,
        stride_oh,
        stride_ou,
        stride_ov,
        stride_znb,
        stride_znu,
        stride_znv,
        stride_znc,
        TILE_V: tl.constexpr,
        TILE_K: tl.constexpr,
        NUM_HEADS: tl.constexpr,
        NUM_HEADS_PER_BLK: tl.constexpr,
        ELEMENTWISE_AFFINE: tl.constexpr,
        IS_TRAINING: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        APPLY_MASK: tl.constexpr,
        CACHE_Z_PROJ: tl.constexpr,
        PRECISION: tl.constexpr,
    ):
        """Fused LN + Linear([C]->[H]) + permute([B,U,V,H]->[B,H,U,V])."""
        pid_v = tl.program_id(0).to(tl.int64)
        pid_u = tl.program_id(1).to(tl.int64)
        pid_hb = tl.program_id(2).to(tl.int64)

        num_h_blocks: tl.constexpr = tl.cdiv(NUM_HEADS, NUM_HEADS_PER_BLK)
        b_idx = pid_hb // num_h_blocks
        h_blk = pid_hb % num_h_blocks
        h_start = h_blk * NUM_HEADS_PER_BLK

        offs_v = pid_v * TILE_V + tl.arange(0, TILE_V)
        v_mask = offs_v < V

        # Phase 1: LN statistics over C dimension
        mean = tl.zeros((TILE_V,), dtype=tl.float32)
        m2 = tl.zeros((TILE_V,), dtype=tl.float32)

        z_base = Z_ptr + b_idx * stride_zb + pid_u * stride_zu
        for k_start in tl.range(0, C, TILE_K):
            offs_k = k_start + tl.arange(0, TILE_K)
            k_mask = offs_k < C
            z_ptrs = z_base + offs_v[:, None] * stride_zv + offs_k[None, :] * stride_zc
            z_tile = tl.load(z_ptrs, mask=v_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float32)
            mean += tl.sum(z_tile, axis=1)
            m2 += tl.sum(z_tile * z_tile, axis=1)

        mean = mean / C
        var = m2 / C - mean * mean
        rstd = 1.0 / tl.sqrt(var + eps)

        if IS_TRAINING:
            stat_base = b_idx * (U * V) + pid_u * V
            stat_offs = stat_base + offs_v
            tl.store(Mean_ptr + stat_offs, mean, mask=v_mask)
            tl.store(RSTD_ptr + stat_offs, rstd, mask=v_mask)

        # Phase 2: Normalize + project via tl.dot
        acc = tl.zeros((TILE_V, NUM_HEADS_PER_BLK), dtype=tl.float32)
        offs_h = h_start + tl.arange(0, NUM_HEADS_PER_BLK)
        h_mask = offs_h < NUM_HEADS

        for k_start in tl.range(0, C, TILE_K):
            offs_k = k_start + tl.arange(0, TILE_K)
            k_mask = offs_k < C

            z_ptrs = z_base + offs_v[:, None] * stride_zv + offs_k[None, :] * stride_zc
            z_tile = tl.load(z_ptrs, mask=v_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float32)

            z_norm = (z_tile - mean[:, None]) * rstd[:, None]

            if ELEMENTWISE_AFFINE:
                w_ln = tl.load(W_ln_ptr + offs_k, mask=k_mask, other=1.0).to(tl.float32)
                b_ln = tl.load(B_ln_ptr + offs_k, mask=k_mask, other=0.0).to(tl.float32)
                z_norm = z_norm * w_ln[None, :] + b_ln[None, :]

            if IS_TRAINING:
                zn_ptrs = (
                    Z_norm_ptr
                    + b_idx * stride_znb
                    + pid_u * stride_znu
                    + offs_v[:, None] * stride_znv
                    + offs_k[None, :] * stride_znc
                )
                tl.store(zn_ptrs, z_norm, mask=v_mask[:, None] & k_mask[None, :])

            # W_proj is [H, C]: load [TILE_K, NUM_HEADS_PER_BLK] tile
            w_ptrs = W_proj_ptr + offs_h[None, :] * C + offs_k[:, None]
            w_tile = tl.load(w_ptrs, mask=h_mask[None, :] & k_mask[:, None], other=0.0).to(tl.float32)

            acc = precision_dot(z_norm, w_tile, acc, PRECISION)

        if HAS_BIAS:
            b_proj = tl.load(B_proj_ptr + offs_h, mask=h_mask, other=0.0).to(tl.float32)
            acc = acc + b_proj[None, :]

        # Store helpers: 2D [TILE_V, NUM_HEADS_PER_BLK] block with strided scatter.
        out_base = OUT_ptr + b_idx * stride_ob + pid_u * stride_ou
        out_ptrs = out_base + offs_h[None, :] * stride_oh + offs_v[:, None] * stride_ov
        out_mask = v_mask[:, None] & h_mask[None, :]

        # Cache z_proj BEFORE mask (for recycle reuse)
        if CACHE_Z_PROJ:
            cache_base = z_proj_ptr + b_idx * stride_ob + pid_u * stride_ou
            cache_ptrs = cache_base + offs_h[None, :] * stride_oh + offs_v[:, None] * stride_ov
            tl.store(cache_ptrs, acc.to(z_proj_ptr.type.element_ty), mask=out_mask)

        # Apply mask AFTER caching
        if APPLY_MASK:
            m = tl.load(mask_ptr + b_idx * V + offs_v, mask=v_mask, other=0.0).to(tl.float32)
            mask_bias = (1.0 - m) * (-1e9)
            acc = acc + mask_bias[:, None]

        tl.store(out_ptrs, acc.to(OUT_ptr.type.element_ty), mask=out_mask)


def fused_pair_bias_forward(
    z: torch.Tensor,
    ln_weight: torch.Tensor,
    ln_bias: torch.Tensor,
    proj_weight: torch.Tensor,
    proj_bias: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    eps: float = 1e-5,
    save_for_backward: bool = False,
    cache_z_proj: bool = False,
    precision: int = Precision.TF32x3,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Fused LN + Linear + permute for pair bias.

    Args:
        z: [B, U, V, C] pair activations.
        ln_weight: [C] LayerNorm weight.
        ln_bias: [C] LayerNorm bias.
        proj_weight: [H, C] pair bias projection weight.
        proj_bias: [H] optional projection bias.
        mask: [B, V] optional mask. Applied as output + (1-mask)*(-1e9).
        eps: LN epsilon.
        save_for_backward: Save mean, rstd, z_norm for backward.
        cache_z_proj: Cache projection output before mask for recycle reuse.
        precision: Dot product precision (Precision enum).

    Returns:
        (output, z_proj_cache, mean, rstd, z_norm)
    """
    B, U, V, C = z.shape
    H = proj_weight.shape[0]

    out = torch.empty((B, H, U, V), dtype=z.dtype, device=z.device)
    z_proj_cache = torch.empty((B, H, U, V), dtype=z.dtype, device=z.device) if cache_z_proj else None
    mean = torch.empty((B, U, V), dtype=torch.float32, device=z.device) if save_for_backward else None
    rstd = torch.empty((B, U, V), dtype=torch.float32, device=z.device) if save_for_backward else None
    z_norm = torch.empty((B, U, V, C), dtype=z.dtype, device=z.device) if save_for_backward else None

    TILE_V = max(min(V, 16), 16)
    TILE_K = max(min(C, 64), 16)
    NUM_HEADS_PER_BLK = min(H, 4)

    num_h_blocks = triton.cdiv(H, NUM_HEADS_PER_BLK)
    _, num_warps = _calculate_settings(TILE_V * NUM_HEADS_PER_BLK)
    num_warps = max(num_warps, 4)

    grid = (triton.cdiv(V, TILE_V), U, num_h_blocks * B)
    dummy = z
    z_cont = z.contiguous()

    _fused_pair_bias_fwd_kernel[grid](
        z_cont,
        ln_weight,
        ln_bias,
        proj_weight,
        proj_bias if proj_bias is not None else dummy,
        mask if mask is not None else dummy,
        out,
        z_proj_cache if z_proj_cache is not None else out,
        mean if mean is not None else dummy,
        rstd if rstd is not None else dummy,
        z_norm if z_norm is not None else dummy,
        U,
        V,
        C,
        eps,
        z_cont.stride(0),
        z_cont.stride(1),
        z_cont.stride(2),
        z_cont.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        z_norm.stride(0) if z_norm is not None else 0,
        z_norm.stride(1) if z_norm is not None else 0,
        z_norm.stride(2) if z_norm is not None else 0,
        z_norm.stride(3) if z_norm is not None else 0,
        TILE_V=TILE_V,
        TILE_K=TILE_K,
        NUM_HEADS=H,
        NUM_HEADS_PER_BLK=NUM_HEADS_PER_BLK,
        ELEMENTWISE_AFFINE=True,
        IS_TRAINING=save_for_backward,
        HAS_BIAS=proj_bias is not None,
        APPLY_MASK=mask is not None,
        CACHE_Z_PROJ=cache_z_proj,
        PRECISION=precision,
        num_warps=num_warps,
    )
    return out, z_proj_cache, mean, rstd, z_norm


class FusedPairBiasFunction(torch.autograd.Function):
    """Autograd wrapper: fused Triton forward + PyTorch backward.

    The forward fusion is where the performance win is (eliminates LN
    intermediate + permute copy). The backward uses standard PyTorch
    matmuls which lets cuBLAS handle the gradient GEMMs optimally.
    """

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.bfloat16)
    def forward(ctx, z, ln_weight, ln_bias, proj_weight, eps):
        out, _, mean, rstd, z_norm = fused_pair_bias_forward(
            z,
            ln_weight,
            ln_bias,
            proj_weight,
            proj_bias=None,
            mask=None,
            eps=eps,
            save_for_backward=True,
            cache_z_proj=False,
        )
        # Save z_norm instead of z — derive xhat from z_norm in backward
        # (avoids recomputing (z-mean)*rstd; same memory since z_norm is same shape as z)
        ctx.save_for_backward(ln_weight, ln_bias, proj_weight, rstd, z_norm)
        ctx.eps = eps
        return out

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_out):
        ln_weight, ln_bias, proj_weight, rstd, z_norm = ctx.saved_tensors
        B, H, U, V = grad_out.shape
        C = proj_weight.shape[1]

        # grad_out [B, H, U, V] -> [B, U, V, H]
        grad_out_perm = grad_out.permute(0, 2, 3, 1).contiguous()
        grad_out_flat = grad_out_perm.reshape(-1, H)  # [B*U*V, H]
        z_norm_flat = z_norm.reshape(-1, C)  # [B*U*V, C]

        # grad_proj_weight [H, C] = grad_out_flat.T @ z_norm_flat
        grad_proj_weight = (grad_out_flat.float().T @ z_norm_flat.float()).to(proj_weight.dtype)

        # grad_z_norm [B*U*V, C] = grad_out_flat @ proj_weight
        grad_z_norm = (grad_out_flat.float() @ proj_weight.float()).reshape(B, U, V, C)

        # Recover xhat from z_norm: z_norm = xhat * w + b → xhat = (z_norm - b) / w
        ln_weight_f = ln_weight.float()
        ln_bias_f = ln_bias.float()
        xhat = (z_norm.float() - ln_bias_f[None, None, None, :]) / ln_weight_f[None, None, None, :]

        grad_z_norm_f = grad_z_norm.float()
        rstd_exp = rstd.unsqueeze(-1)
        wdy = grad_z_norm_f * ln_weight_f[None, None, None, :]

        grad_ln_weight = (grad_z_norm_f * xhat).sum(dim=(0, 1, 2)).to(ln_weight.dtype)
        grad_ln_bias = grad_z_norm_f.sum(dim=(0, 1, 2)).to(ln_bias.dtype)

        c1 = (xhat * wdy).mean(dim=-1, keepdim=True)
        c2 = wdy.mean(dim=-1, keepdim=True)
        grad_z = ((wdy - c1 * xhat - c2) * rstd_exp).to(z_norm.dtype)

        # grad_z for the input z: since z_norm = (z-mean)*rstd*w+b, grad flows through
        # the LN backward. Return grad_z as the gradient for the first argument.
        return grad_z, grad_ln_weight, grad_ln_bias, grad_proj_weight, None


class FusedPairBias(nn.Module):
    """Fused LN + Linear + permute for GridSelfAttention pair bias.

    Drop-in replacement for:
        z_normed = self.norm(z)                  # LayerNorm [B, N, N, C]
        pb = self.pair_bias(z_normed)            # LinearNoBias [B, N, N, H]
        pb = pb.permute(0, 3, 1, 2)             # → [B, H, N, N]

    Supports both training (autograd with PyTorch backward) and inference
    (fully fused kernel with optional mask).

    Args:
        c: Pair channel dimension (128).
        num_heads: Number of attention heads (4).
        eps: LayerNorm epsilon.
    """

    def __init__(self, c: int, num_heads: int, eps: float = 1e-5):
        super().__init__()
        self.c = c
        self.num_heads = num_heads
        self.eps = eps

        self.ln_weight = nn.Parameter(torch.ones(c))
        self.ln_bias = nn.Parameter(torch.zeros(c))
        self.proj_weight = nn.Parameter(torch.empty(num_heads, c))
        nn.init.kaiming_uniform_(self.proj_weight)

    def forward(self, z: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            z: [B, N, N, C] pair activations (pre-LN).
            mask: [B, N] optional mask. Applied as output + (1-mask)*(-1e9).

        Returns:
            [B, H, N, N] pair bias ready for attention.
        """
        if not HAS_TRITON or not z.is_cuda:
            z_normed = F.layer_norm(z, (self.c,), self.ln_weight, self.ln_bias, self.eps)
            pb = F.linear(z_normed, self.proj_weight).permute(0, 3, 1, 2)
            if mask is not None:
                pb = pb + (1.0 - mask[:, None, None, :].float()) * (-1e9)
            return pb

        if z.requires_grad:
            pb = FusedPairBiasFunction.apply(z, self.ln_weight, self.ln_bias, self.proj_weight, self.eps)
            if mask is not None:
                pb = pb + (1.0 - mask[:, None, None, :].float()) * (-1e9)
            return pb

        out, _, _, _, _ = fused_pair_bias_forward(
            z,
            self.ln_weight,
            self.ln_bias,
            self.proj_weight,
            proj_bias=None,
            mask=mask,
            eps=self.eps,
            save_for_backward=False,
            cache_z_proj=False,
        )
        return out

    def extra_repr(self):
        return f"c={self.c}, num_heads={self.num_heads}"
