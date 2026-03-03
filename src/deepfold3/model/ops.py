"""Triangle multiplication and outer-product mean ops.

Attribute names map to Haiku parameter scopes in the AF3 converter.

TriangleMultiplication Haiku → PyTorch mapping:
    left_norm_input       → norm_left    (LayerNorm)
    right_norm_input      → norm_right   (LayerNorm)   [unused in AF3 — same input]
    projection            → proj         (LinearNoBias, c_z → c_z*2)
    gate                  → gate         (LinearNoBias, c_z → c_z*2)
    center_norm           → norm_center  (LayerNorm, axis=0 in Haiku ≡ on channel axis)
    output_projection     → out_proj     (LinearNoBias, c_z → c_z)
    gating_linear         → out_gate     (LinearNoBias, c_z → c_z)

OuterProductMean Haiku → PyTorch mapping:
    layer_norm_input  → norm      (LayerNorm, c_msa)
    left_projection   → proj_a    (LinearNoBias, c_msa → c_hidden)
    right_projection  → proj_b    (LinearNoBias, c_msa → c_hidden)
    output_w          → out_w     (parameter [c_hidden, c_hidden, c_z])
    output_b          → out_b     (parameter [c_z])

All modules support both unbatched and batched (leading B dim) inputs.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from deepfold3.model.primitives import LayerNorm, LinearNoBias


class TriangleMulOutgoing(nn.Module):
    """Triangle Multiplication — outgoing (equation: ikc,jkc->ijc).

    Haiku module name: triangle_multiplication_outgoing
    """

    def __init__(self, c_z: int):
        super().__init__()
        self.norm_left = LayerNorm(c_z)
        self.proj = LinearNoBias(c_z, c_z * 2)
        self.gate = LinearNoBias(c_z, c_z * 2)
        self.norm_center = LayerNorm(c_z)
        self.out_proj = LinearNoBias(c_z, c_z)
        self.out_gate = LinearNoBias(c_z, c_z)

    def forward(self, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [B?, N, N, c_z]
            mask: [B?, N, N]
        Returns:
            [B?, N, N, c_z]
        """
        has_batch = z.dim() == 4
        if not has_batch:
            z = z.unsqueeze(0)
            mask = mask.unsqueeze(0)

        z0 = self.norm_left(z)
        proj = self.proj(z0)  # [B, N, N, 2*c_z]
        gate = torch.sigmoid(self.gate(z0))  # [B, N, N, 2*c_z]
        out_gate = torch.sigmoid(self.out_gate(z0))  # compute early to free z0
        del z0
        proj = proj * gate
        # INTERLEAVED split (AF3: reshape [2c]->[c,2] then split axis=1)
        # a gets even channels (0,2,4,...), b gets odd channels (1,3,5,...)
        a = proj[..., 0::2]  # [B, N, N, c_z]
        b = proj[..., 1::2]  # [B, N, N, c_z]
        # apply mask: [B, N, N, 1]
        m = mask.unsqueeze(-1)
        a = a * m
        b = b * m
        # outgoing: sum over k for each (i,j) → result[b,i,j,c] = Σ_k a[b,i,k,c]*b[b,j,k,c]
        act = torch.einsum("bikc,bjkc->bijc", a, b)
        act = self.norm_center(act)
        act = self.out_proj(act)
        result = act * out_gate

        if not has_batch:
            result = result.squeeze(0)
        return result


class TriangleMulIncoming(nn.Module):
    """Triangle Multiplication — incoming (equation: kjc,kic->ijc).

    Haiku module name: triangle_multiplication_incoming
    """

    def __init__(self, c_z: int):
        super().__init__()
        self.norm_left = LayerNorm(c_z)
        self.proj = LinearNoBias(c_z, c_z * 2)
        self.gate = LinearNoBias(c_z, c_z * 2)
        self.norm_center = LayerNorm(c_z)
        self.out_proj = LinearNoBias(c_z, c_z)
        self.out_gate = LinearNoBias(c_z, c_z)

    def forward(self, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [B?, N, N, c_z]
            mask: [B?, N, N]
        Returns:
            [B?, N, N, c_z]
        """
        has_batch = z.dim() == 4
        if not has_batch:
            z = z.unsqueeze(0)
            mask = mask.unsqueeze(0)

        z0 = self.norm_left(z)
        proj = self.proj(z0)
        gate = torch.sigmoid(self.gate(z0))
        out_gate = torch.sigmoid(self.out_gate(z0))  # compute early to free z0
        del z0
        proj = proj * gate
        # INTERLEAVED split (AF3: reshape [2c]->[c,2] then split axis=1)
        a = proj[..., 0::2]  # [B, N, N, c_z]
        b = proj[..., 1::2]  # [B, N, N, c_z]
        m = mask.unsqueeze(-1)
        a = a * m
        b = b * m
        # incoming: result[b,i,j,c] = Σ_k a[b,k,j,c]*b[b,k,i,c]
        act = torch.einsum("bkjc,bkic->bijc", a, b)
        act = self.norm_center(act)
        act = self.out_proj(act)
        result = act * out_gate

        if not has_batch:
            result = result.squeeze(0)
        return result


class OuterProductMean(nn.Module):
    """Outer product mean for MSA → pair update.

    Haiku → PyTorch:
        layer_norm_input → norm
        left_projection  → proj_a
        right_projection → proj_b
        output_w         → out_w  (register_parameter)
        output_b         → out_b  (register_parameter)
    """

    def __init__(self, c_msa: int, c_z: int, c_hidden: int = 32):
        super().__init__()
        self.norm = LayerNorm(c_msa)
        self.proj_a = LinearNoBias(c_msa, c_hidden)
        self.proj_b = LinearNoBias(c_msa, c_hidden)
        self.out_w = nn.Parameter(torch.zeros(c_hidden, c_hidden, c_z))
        self.out_b = nn.Parameter(torch.zeros(c_z))
        nn.init.kaiming_uniform_(self.out_w, nonlinearity="linear")

    def forward(self, msa: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            msa:  [B?, num_seq, N, c_msa]
            mask: [B?, num_seq, N]
        Returns:
            [B?, N, N, c_z]
        """
        has_batch = msa.dim() == 4
        if not has_batch:
            msa = msa.unsqueeze(0)
            mask = mask.unsqueeze(0)

        msa = self.norm(msa)
        m = mask.unsqueeze(-1)  # [B, S, N, 1]
        a = self.proj_a(msa) * m  # [B, S, N, c_hidden]
        b = self.proj_b(msa) * m  # [B, S, N, c_hidden]
        # outer product then mean: [B, S, N, N, c_hidden, c_hidden] → contract
        ab = torch.einsum("bsie,bsjf->bijef", a, b)  # [B, N, N, c_hidden, c_hidden]
        act = torch.einsum("bijef,efc->bijc", ab, self.out_w) + self.out_b
        # normalize by number of valid sequence positions
        eps = 1e-3
        norm = torch.einsum("bsi,bsj->bij", mask.float(), mask.float())
        result = act / (eps + norm.unsqueeze(-1))

        if not has_batch:
            result = result.squeeze(0)
        return result
