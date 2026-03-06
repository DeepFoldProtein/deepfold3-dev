"""Attention modules for DeepFold3.

Attribute names match the Haiku → PyTorch converter mapping.

GridSelfAttention Haiku → PyTorch:
    act_norm             → norm
    pair_bias_projection → pair_bias  (LinearNoBias, pair_channel → num_heads)
    q_projection         → to_q       (LinearNoBias, transpose_weights=True)
    k_projection         → to_k       (LinearNoBias, transpose_weights=True)
    v_projection         → to_v       (LinearNoBias)
    gating_query         → to_g       (LinearNoBias, transpose_weights=True)
    output_projection    → out_proj   (LinearNoBias)

MSAAttention Haiku → PyTorch:
    act_norm             → norm
    pair_norm            → pair_norm
    pair_logits          → pair_logits (LinearNoBias, pair_channel → num_heads)
    v_projection         → to_v
    gating_query         → to_g
    output_projection    → out_proj

SingleAttention (in PairFormer, single_cond=None):
    single_attention_layer_norm   → norm      (regular LayerNorm, scale+offset)
    single_attention_q_projection → to_q     (Linear w/ bias)
    single_attention_k_projection → to_k     (LinearNoBias)
    single_attention_v_projection → to_v     (LinearNoBias)
    single_attention_gating_query → to_g     (LinearNoBias)
    single_attention_transition2  → out_proj (LinearNoBias)

All modules support both unbatched and batched (leading B dim) inputs.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepfold3.model.config import OptimConfig
from deepfold3.model.primitives import LayerNorm, LinearNoBias


class GridSelfAttention(nn.Module):
    """Row-wise or column-wise gated self-attention on pair activations.

    Haiku: GridSelfAttention with transpose=False (row) or transpose=True (col).
    Supports both unbatched [N, N, C] and batched [B, N, N, C] inputs.
    """

    def __init__(self, c: int, num_heads: int = 4, transpose: bool = False, optim: OptimConfig | None = None):
        super().__init__()
        assert c % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = max(c // num_heads, 16)
        self.transpose = transpose
        self.scale = self.head_dim**-0.5

        self.norm = LayerNorm(c)
        self.pair_bias = LinearNoBias(c, num_heads)
        self.to_q = LinearNoBias(c, num_heads * self.head_dim)
        self.to_k = LinearNoBias(c, num_heads * self.head_dim)
        self.to_v = LinearNoBias(c, num_heads * self.head_dim)
        self.to_g = LinearNoBias(c, num_heads * self.head_dim)
        self.out_proj = LinearNoBias(num_heads * self.head_dim, c)

        self._evo_attn = None
        if optim is not None and optim.use_fused_evo_attention:
            try:
                from deepfold3.model.triton.evo_attention import HAS_TRITON, _EvoformerAttention

                if HAS_TRITON:
                    self._evo_attn = _EvoformerAttention
            except ImportError:
                pass

    def forward(self, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z:    [B?, N, N, C]  pair activations
            mask: [B?, N, N]     pair mask
        Returns:
            [B?, N, N, C]
        """
        has_batch = z.dim() == 4
        if not has_batch:
            z = z.unsqueeze(0)
            mask = mask.unsqueeze(0)

        B = z.shape[0]
        z = self.norm(z)

        # pair_bias: [B, N, N, H] → [B, H, N, N]
        pb = self.pair_bias(z).permute(0, 3, 1, 2)

        if self.transpose:
            # Column-wise: transpose act so we iterate over columns
            # AF3: pair_bias is NOT transposed (same [B, H, N, N] for row and col)
            z = z.permute(0, 2, 1, 3)  # [B, N_col, N_row, C]
            mask = mask.permute(0, 2, 1)

        _, N_rows, N_cols, C = z.shape
        H = self.num_heads
        D = self.head_dim

        # Fold batch and row dims for per-row attention
        z_flat = z.reshape(B * N_rows, N_cols, C)

        if self._evo_attn is not None and z.is_cuda:
            # Fused Triton path: kernel expects [B, N_seq, N_res, H, D]
            q = self.to_q(z_flat).view(B, N_rows, N_cols, H, D)
            k = self.to_k(z_flat).view(B, N_rows, N_cols, H, D)
            v = self.to_v(z_flat).view(B, N_rows, N_cols, H, D)
            res_mask = torch.where(mask > 0.5, 0.0, float("-inf")).view(B, N_rows, 1, 1, N_cols)
            pair_bias = pb.unsqueeze(1).float()  # [B, 1, H, N, N]
            out = self._evo_attn.apply(q, k, v, res_mask, pair_bias)  # [B, N_rows, N_cols, H, D]
            out = out.reshape(B * N_rows, N_cols, H * D)
        else:
            q = self.to_q(z_flat).view(B * N_rows, N_cols, H, D).permute(0, 2, 1, 3)
            k = self.to_k(z_flat).view(B * N_rows, N_cols, H, D).permute(0, 2, 1, 3)
            v = self.to_v(z_flat).view(B * N_rows, N_cols, H, D).permute(0, 2, 1, 3)

            logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B*N_rows, H, N_cols, N_cols]

            # pair_bias: [B, H, N, N] → same for every row within each batch element
            logits = logits.view(B, N_rows, H, N_cols, N_cols)
            logits = logits + pb.unsqueeze(1)  # [B, 1, H, N, N] broadcasts over N_rows
            logits = logits.reshape(B * N_rows, H, N_cols, N_cols)

            # mask: [B, N_rows, N_cols] → [B*N_rows, 1, 1, N_cols]
            mask_flat = mask.reshape(B * N_rows, N_cols)
            logits = logits + 1e9 * (mask_flat.float() - 1.0).view(B * N_rows, 1, 1, N_cols)

            w = F.softmax(logits.float(), dim=-1).to(z.dtype)
            out = torch.matmul(w, v)  # [B*N_rows, H, N_cols, D]
            out = out.permute(0, 2, 1, 3).reshape(B * N_rows, N_cols, H * D)

        g = torch.sigmoid(self.to_g(z_flat))
        out = out * g
        out = self.out_proj(out)

        # Unfold back to [B, N_rows, N_cols, C_out]
        out = out.view(B, N_rows, N_cols, -1)

        if self.transpose:
            out = out.permute(0, 2, 1, 3)

        if not has_batch:
            out = out.squeeze(0)
        return out


class MSAAttention(nn.Module):
    """Row-wise MSA attention with pair bias.

    Haiku module: msa_attention1
    Supports both unbatched [S, N, c] and batched [B, S, N, c] inputs.
    """

    def __init__(self, c_msa: int, c_pair: int, num_heads: int = 8, optim: OptimConfig | None = None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = c_msa // num_heads
        assert c_msa % num_heads == 0

        self.norm = LayerNorm(c_msa)
        self.pair_norm = LayerNorm(c_pair)
        self.pair_logits = LinearNoBias(c_pair, num_heads)
        self.to_v = LinearNoBias(c_msa, num_heads * self.head_dim)
        self.to_g = LinearNoBias(c_msa, num_heads * self.head_dim)
        self.out_proj = LinearNoBias(num_heads * self.head_dim, c_msa)
        if optim is not None and optim.use_torch_compile:
            self._compute = torch.compile(self._compute)

    def _compute(
        self, msa: torch.Tensor, mask: torch.Tensor, pair: torch.Tensor
    ) -> torch.Tensor:
        B, S, N, C = msa.shape
        H, D = self.num_heads, self.head_dim

        msa = self.norm(msa)
        pair = self.pair_norm(pair)

        logits = self.pair_logits(pair).permute(0, 3, 1, 2)
        col_mask, _ = mask.max(dim=-2)
        logits = logits + 1e9 * (col_mask.float() - 1.0).view(B, 1, 1, N)

        w = F.softmax(logits, dim=-1)

        v = self.to_v(msa).view(B, S, N, H, D)
        v_avg = torch.einsum("bhqk,bskhc->bsqhc", w, v)
        v_avg = v_avg.reshape(B, S, N, H * D)

        g = torch.sigmoid(self.to_g(msa))
        v_avg = v_avg * g
        return self.out_proj(v_avg)

    def forward(
        self,
        msa: torch.Tensor,  # [B?, S, N, c_msa]
        mask: torch.Tensor,  # [B?, S, N]
        pair: torch.Tensor,  # [B?, N, N, c_pair]
    ) -> torch.Tensor:
        has_batch = msa.dim() == 4
        if not has_batch:
            msa = msa.unsqueeze(0)
            mask = mask.unsqueeze(0)
            pair = pair.unsqueeze(0)

        result = self._compute(msa, mask, pair)

        if not has_batch:
            result = result.squeeze(0)
        return result


class SingleAttention(nn.Module):
    """Self-attention on single representation with pair-biased logits.

    Used in PairFormerBlock and ConfidencePairFormerBlock for the single track.

    AF3: self_attention(..., single_cond=None) → regular LayerNorm (scale+offset).
    adaptive_zero_init with single_cond=None → plain Linear output (no gate).

    Haiku params stored flat in PairFormer scope (prefix 'single_attention_'):
        layer_norm    → norm      (LayerNorm with scale=weight, offset=bias)
        q_projection  → to_q     (Linear w/ bias)
        k_projection  → to_k     (LinearNoBias)
        v_projection  → to_v     (LinearNoBias)
        gating_query  → to_g     (LinearNoBias)
        transition2   → out_proj (LinearNoBias)

    Supports both unbatched [N, c] and batched [B, N, c] inputs.
    """

    def __init__(
        self, c: int, num_heads: int = 16, key_dim: int = 384, value_dim: int = 384, optim: OptimConfig | None = None
    ):
        super().__init__()
        assert key_dim % num_heads == 0
        assert value_dim % num_heads == 0
        self.num_heads = num_heads
        self.key_dim_per_head = key_dim // num_heads
        self.val_dim_per_head = value_dim // num_heads
        self.scale = self.key_dim_per_head**-0.5

        # Regular LayerNorm (single_cond=None → no AdaLN)
        self.norm = LayerNorm(c)

        self.to_q = nn.Linear(c, key_dim)
        self.to_k = LinearNoBias(c, key_dim)
        self.to_v = LinearNoBias(c, value_dim)
        self.to_g = LinearNoBias(c, value_dim)
        self.out_proj = LinearNoBias(value_dim, c)

        # Fused EvoAttention: SingleAttention is the N_seq=1 case
        self._evo_attn = None
        if optim is not None and optim.use_fused_evo_attention and self.key_dim_per_head == self.val_dim_per_head:
            try:
                from deepfold3.model.triton.evo_attention import HAS_TRITON, _EvoformerAttention

                if HAS_TRITON:
                    self._evo_attn = _EvoformerAttention
            except ImportError:
                pass
        if self._evo_attn is None and optim is not None and optim.use_torch_compile:
            self._compute_baseline = torch.compile(self._compute_baseline)

    def _compute_baseline(
        self, xn: torch.Tensor, mask: torch.Tensor, pair_logits: torch.Tensor
    ) -> torch.Tensor:
        B, N, _ = xn.shape
        H, Dk, Dv = self.num_heads, self.key_dim_per_head, self.val_dim_per_head
        q = self.to_q(xn).view(B, N, H, Dk).permute(0, 2, 1, 3)
        k = self.to_k(xn).view(B, N, H, Dk).permute(0, 2, 1, 3)
        v = self.to_v(xn).view(B, N, H, Dv).permute(0, 2, 1, 3)
        logits = torch.matmul(q * self.scale, k.transpose(-2, -1))
        logits = logits + pair_logits
        bias_mask = 1e9 * (mask.float() - 1.0)
        logits = logits + bias_mask.view(B, 1, 1, N)
        w = F.softmax(logits.float(), dim=-1).to(xn.dtype)
        out = torch.matmul(w, v)
        out = out.permute(0, 2, 1, 3).reshape(B, N, H * Dv)
        g = torch.sigmoid(self.to_g(xn))
        return self.out_proj(out * g)

    def forward(
        self,
        x: torch.Tensor,  # [B?, N, c]
        mask: torch.Tensor,  # [B?, N]
        pair_logits: torch.Tensor,  # [B?, H, N, N]
    ) -> torch.Tensor:
        has_batch = x.dim() == 3
        if not has_batch:
            x = x.unsqueeze(0)
            mask = mask.unsqueeze(0)
            pair_logits = pair_logits.unsqueeze(0)

        B, N, C = x.shape
        H = self.num_heads
        Dk = self.key_dim_per_head
        Dv = self.val_dim_per_head

        xn = self.norm(x)

        if self._evo_attn is not None and x.is_cuda:
            q = self.to_q(xn).view(B, N, H, Dk).unsqueeze(1)  # [B, 1, N, H, D]
            k = self.to_k(xn).view(B, N, H, Dk).unsqueeze(1)
            v = self.to_v(xn).view(B, N, H, Dv).unsqueeze(1)
            res_mask = torch.where(mask > 0.5, 0.0, float("-inf")).view(B, 1, 1, 1, N)
            pair_bias = pair_logits.unsqueeze(1).float()  # [B, 1, H, N, N]
            out = self._evo_attn.apply(q, k, v, res_mask, pair_bias)  # [B, 1, N, H, D]
            out = out.squeeze(1).reshape(B, N, H * Dv)
            g = torch.sigmoid(self.to_g(xn))
            result = self.out_proj(out * g)
        else:
            result = self._compute_baseline(xn, mask, pair_logits)

        if not has_batch:
            result = result.squeeze(0)
        return result
