"""Template embedding: structural templates → pair-space embedding.

Processes T structural templates independently, averages them, and projects
to pair_channel dimensions.  Integrated into the Evoformer trunk between
bond embedding and MSA stack.

Haiku scope hierarchy:
    diffuser/evoformer/template_embedding/
        output_linear                         → output_proj
        single_template_embedding/
            query_embedding_norm              → single_template_emb.query_norm
            template_pair_embedding_0         → single_template_emb.embed_dgram
            template_pair_embedding_1         → single_template_emb.embed_pseudo_beta
            template_pair_embedding_2         → single_template_emb.embed_aatype_row
            template_pair_embedding_3         → single_template_emb.embed_aatype_col
            template_pair_embedding_4         → single_template_emb.embed_unit_vec_x
            template_pair_embedding_5         → single_template_emb.embed_unit_vec_y
            template_pair_embedding_6         → single_template_emb.embed_unit_vec_z
            template_pair_embedding_7         → single_template_emb.embed_backbone_mask
            template_pair_embedding_8         → single_template_emb.embed_query
            output_layer_norm                 → single_template_emb.output_norm
            __layer_stack_no_per_layer/template_embedding_iteration/
                layer {i}:
                    triangle_multiplication_outgoing~/  → pairformer.{i}.tri_mul_out
                    triangle_multiplication_incoming~/  → pairformer.{i}.tri_mul_in
                    pair_attention1~/                   → pairformer.{i}.pair_attn_row
                    pair_attention2~/                   → pairformer.{i}.pair_attn_col
                    pair_transition~/                   → pairformer.{i}.pair_transition

Dense atom indices (from deepfold3.data.const):
    N=0, CA=1, C=2, CB=3
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepfold3.model.attention import GridSelfAttention
from deepfold3.model.config import TemplateEmbeddingConfig
from deepfold3.model.ops import TriangleMulIncoming, TriangleMulOutgoing
from deepfold3.model.primitives import LayerNorm, LinearNoBias, TransitionBlock

# Dense atom indices for backbone atoms
_N_IDX = 0
_CA_IDX = 1
_C_IDX = 2
_CB_IDX = 3

# GLY residue type index (uses CA instead of CB for pseudo-beta)
_GLY_IDX = 7


def pseudo_beta_from_positions(
    aatype: torch.Tensor,
    positions: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute pseudo-beta positions and mask.

    Pseudo-beta is CB for all residues except GLY (which uses CA).

    Args:
        aatype: [N] residue type indices.
        positions: [N, num_atoms, 3] dense atom positions.
        mask: [N, num_atoms] dense atom mask.

    Returns:
        Tuple of (pseudo_beta_pos [N, 3], pseudo_beta_mask [N]).
    """
    is_gly = aatype == _GLY_IDX  # [N]
    # Select CB for non-GLY, CA for GLY
    cb_pos = positions[:, _CB_IDX]  # [N, 3]
    ca_pos = positions[:, _CA_IDX]  # [N, 3]
    pb_pos = torch.where(is_gly.unsqueeze(-1), ca_pos, cb_pos)  # [N, 3]

    cb_mask = mask[:, _CB_IDX]  # [N]
    ca_mask = mask[:, _CA_IDX]  # [N]
    pb_mask = torch.where(is_gly, ca_mask, cb_mask)  # [N]

    return pb_pos, pb_mask


def backbone_frames_from_positions(
    positions: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build backbone local frames via Gram-Schmidt orthogonalization.

    Frame is built from N, CA, C atoms following AF3 convention:
        e1 = normalize(C - CA)
        e2 = normalize(orthogonalize(N - CA, e1))
        e3 = cross(e1, e2)

    Args:
        positions: [N, num_atoms, 3] dense atom positions.
        mask: [N, num_atoms] dense atom mask.

    Returns:
        Tuple of (frames [N, 3, 3], frame_mask [N]) where frames[i] is a
        rotation matrix [e1, e2, e3] (rows) for residue i.
    """
    eps = 1e-8
    n_pos = positions[:, _N_IDX]  # [N, 3]
    ca_pos = positions[:, _CA_IDX]  # [N, 3]
    c_pos = positions[:, _C_IDX]  # [N, 3]

    # Frame mask requires all three backbone atoms present
    frame_mask = mask[:, _N_IDX] * mask[:, _CA_IDX] * mask[:, _C_IDX]  # [N]

    # Gram-Schmidt: e1 from C-CA, e2 from N-CA orthogonalized to e1
    v1 = c_pos - ca_pos  # [N, 3]
    v2 = n_pos - ca_pos  # [N, 3]

    e1 = v1 / (v1.norm(dim=-1, keepdim=True) + eps)
    # Remove component along e1 from v2
    u2 = v2 - (v2 * e1).sum(dim=-1, keepdim=True) * e1
    e2 = u2 / (u2.norm(dim=-1, keepdim=True) + eps)
    e3 = torch.cross(e1, e2, dim=-1)

    # Stack as rotation matrix: [N, 3, 3] where row i is e_i
    frames = torch.stack([e1, e2, e3], dim=1)  # [N, 3, 3]
    return frames, frame_mask


def dgram_from_positions(
    positions: torch.Tensor,
    mask: torch.Tensor,
    min_bin: float = 3.25,
    max_bin: float = 50.75,
    num_bins: int = 39,
) -> torch.Tensor:
    """Compute pairwise distance histogram from positions.

    Bins are defined by squared-distance boundaries, following AF3 convention.

    Args:
        positions: [N, 3] atom positions.
        mask: [N] atom mask.
        min_bin: Left edge of first distance bin (Angstroms).
        max_bin: Left edge of last distance bin (Angstroms).
        num_bins: Number of histogram bins.

    Returns:
        Distance histogram [N, N, num_bins] float32.
    """
    # Squared distance boundaries
    lower_breaks = torch.linspace(min_bin, max_bin, num_bins, device=positions.device)
    lower_breaks = lower_breaks * lower_breaks  # squared
    upper_breaks = torch.cat([lower_breaks[1:], torch.tensor([1e8], device=positions.device)])

    # Pairwise squared distances
    diff = positions.unsqueeze(-2) - positions.unsqueeze(-3)  # [N, N, 3]
    dist2 = (diff * diff).sum(dim=-1, keepdim=True)  # [N, N, 1]

    # One-hot binning
    dgram = ((dist2 > lower_breaks) & (dist2 < upper_breaks)).float()  # [N, N, num_bins]

    # Mask: zero out where either position is missing
    pair_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)  # [N, N]
    dgram = dgram * pair_mask.unsqueeze(-1)

    return dgram


def _unit_vectors_from_frames(
    frames: torch.Tensor,
    translations: torch.Tensor,
    frame_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute inter-residue unit vectors in each residue's local frame.

    For each pair (i, j), computes the displacement vector from i's CA to j's CA
    in i's local coordinate frame, then normalizes.

    Args:
        frames: [N, 3, 3] rotation matrices (rows = local frame axes).
        translations: [N, 3] CA positions (frame origins).
        frame_mask: [N] mask for valid frames.

    Returns:
        Tuple of (unit_vec_x, unit_vec_y, unit_vec_z, backbone_mask_2d)
        each [N, N] or [N, N, 1].
    """
    eps = 1e-8
    # Displacement vectors: j_pos - i_pos → [N, N, 3]
    disp = translations.unsqueeze(-2) - translations.unsqueeze(-3)  # [N, N, 3]

    # Transform to local frame of residue i: frames[i] @ disp[i, j]
    # frames: [N, 3, 3], disp: [N, N, 3]
    # For each i: local_vec[i,j] = frames[i] @ disp[i,j]
    local_vec = torch.einsum("iae,ije->ija", frames, disp)  # [N, N, 3]

    # Normalize to unit vectors
    norm = local_vec.norm(dim=-1, keepdim=True).clamp(min=eps)
    unit_vec = local_vec / norm  # [N, N, 3]

    # Mask: both residues need valid backbone frames
    backbone_mask_2d = frame_mask.unsqueeze(-1) * frame_mask.unsqueeze(-2)  # [N, N]

    # Apply mask
    unit_vec = unit_vec * backbone_mask_2d.unsqueeze(-1)

    return (
        unit_vec[..., 0:1],  # x component [N, N, 1]
        unit_vec[..., 1:2],  # y component [N, N, 1]
        unit_vec[..., 2:3],  # z component [N, N, 1]
        backbone_mask_2d,
    )


class TemplatePairFormerBlock(nn.Module):
    """Simplified PairFormer block for template embedding (no single track).

    Contains triangle multiplications, grid self-attention (row + column),
    and a transition block with a smaller intermediate factor.

    Haiku scope: __layer_stack_no_per_layer/template_embedding_iteration/

    Attribute map:
        triangle_multiplication_outgoing~/  → tri_mul_out
        triangle_multiplication_incoming~/  → tri_mul_in
        pair_attention1~/                   → pair_attn_row
        pair_attention2~/                   → pair_attn_col
        pair_transition~/                   → pair_transition
    """

    def __init__(self, num_channels: int, num_heads: int = 4, num_intermediate_factor: int = 2):
        super().__init__()
        self.tri_mul_out = TriangleMulOutgoing(num_channels)
        self.tri_mul_in = TriangleMulIncoming(num_channels)
        self.pair_attn_row = GridSelfAttention(num_channels, num_heads=num_heads, transpose=False)
        self.pair_attn_col = GridSelfAttention(num_channels, num_heads=num_heads, transpose=True)
        self.pair_transition = TransitionBlock(num_channels, num_intermediate_factor=num_intermediate_factor)

    def forward(self, pair: torch.Tensor, pair_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass through one template PairFormer block.

        Args:
            pair: [N, N, num_channels] pair activations.
            pair_mask: [N, N] pair mask.

        Returns:
            Updated pair activations [N, N, num_channels].
        """
        pair = pair + self.tri_mul_out(pair, pair_mask)
        pair = pair + self.tri_mul_in(pair, pair_mask)
        pair = pair + self.pair_attn_row(pair, pair_mask)
        pair = pair + self.pair_attn_col(pair, pair_mask)
        pair = pair + self.pair_transition(pair)
        return pair


class SingleTemplateEmbedding(nn.Module):
    """Embed a single structural template into pair-space features.

    Constructs 9 input features from template structure data, projects each
    to num_channels via LinearNoBias, sums them, then refines through a
    short PairFormer stack.

    Haiku scope: single_template_embedding/

    Attribute map:
        query_embedding_norm              → query_norm
        template_pair_embedding_0         → embed_dgram
        template_pair_embedding_1         → embed_pseudo_beta
        template_pair_embedding_2         → embed_aatype_row
        template_pair_embedding_3         → embed_aatype_col
        template_pair_embedding_4         → embed_unit_vec_x
        template_pair_embedding_5         → embed_unit_vec_y
        template_pair_embedding_6         → embed_unit_vec_z
        template_pair_embedding_7         → embed_backbone_mask
        template_pair_embedding_8         → embed_query
        output_layer_norm                 → output_norm
        __layer_stack_no_per_layer/...    → pairformer.{i}
    """

    def __init__(self, cfg: TemplateEmbeddingConfig):
        super().__init__()
        c = cfg.num_channels
        pc = cfg.pair_channel

        # 9 input feature projections
        self.embed_dgram = LinearNoBias(cfg.dgram_num_bins, c)  # 0: distance histogram
        self.embed_pseudo_beta = LinearNoBias(1, c)  # 1: pseudo-beta mask
        self.embed_aatype_row = LinearNoBias(cfg.aatype_num_classes, c)  # 2: aatype (row)
        self.embed_aatype_col = LinearNoBias(cfg.aatype_num_classes, c)  # 3: aatype (col)
        self.embed_unit_vec_x = LinearNoBias(1, c)  # 4: backbone unit vec x
        self.embed_unit_vec_y = LinearNoBias(1, c)  # 5: backbone unit vec y
        self.embed_unit_vec_z = LinearNoBias(1, c)  # 6: backbone unit vec z
        self.embed_backbone_mask = LinearNoBias(1, c)  # 7: backbone mask
        self.embed_query = LinearNoBias(pc, c)  # 8: query embedding

        # LayerNorm on query embedding before projection
        self.query_norm = LayerNorm(pc)

        # PairFormer stack
        pf = cfg.pairformer
        self.pairformer = nn.ModuleList(
            [
                TemplatePairFormerBlock(
                    num_channels=pf.num_channels,
                    num_heads=pf.num_heads,
                    num_intermediate_factor=pf.num_intermediate_factor,
                )
                for _ in range(pf.num_layer)
            ]
        )

        # Output LayerNorm
        self.output_norm = LayerNorm(c)

    def forward(
        self,
        query_pair: torch.Tensor,
        template_aatype: torch.Tensor,
        template_pseudo_beta: torch.Tensor,
        template_pseudo_beta_mask: torch.Tensor,
        template_backbone_frame: torch.Tensor,
        template_backbone_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        multichain_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Embed a single template.

        Args:
            query_pair: [N, N, pair_channel] query pair embedding.
            template_aatype: [N] template residue types (int64).
            template_pseudo_beta: [N, 3] pseudo-beta positions.
            template_pseudo_beta_mask: [N] pseudo-beta mask.
            template_backbone_frame: [N, 3, 3] backbone rotation matrices.
            template_backbone_mask: [N] backbone frame mask.
            pair_mask: [N, N] padding mask for attention.
            multichain_mask: [N, N] intra-chain mask (1 for same-chain pairs).

        Returns:
            Template pair embedding [N, N, num_channels].
        """
        dtype = query_pair.dtype
        N = query_pair.shape[0]

        # Pseudo-beta pairwise distance histogram
        dgram = dgram_from_positions(
            template_pseudo_beta,
            template_pseudo_beta_mask,
            min_bin=3.25,
            max_bin=50.75,
            num_bins=self.embed_dgram.in_features,
        )  # [N, N, dgram_bins]
        # Apply multichain mask to dgram
        pb_mask_2d = template_pseudo_beta_mask.unsqueeze(-1) * template_pseudo_beta_mask.unsqueeze(-2)
        pb_mask_2d = pb_mask_2d * multichain_mask
        dgram = dgram * pb_mask_2d.unsqueeze(-1)

        # Unit vectors from backbone frames
        unit_x, unit_y, unit_z, bb_mask_2d = _unit_vectors_from_frames(
            template_backbone_frame,
            template_pseudo_beta,  # Use CA positions as translations; backbone frames use CA as origin
            template_backbone_mask,
        )
        bb_mask_2d = bb_mask_2d * multichain_mask

        # Apply backbone mask to unit vectors
        unit_x = unit_x * bb_mask_2d.unsqueeze(-1)
        unit_y = unit_y * bb_mask_2d.unsqueeze(-1)
        unit_z = unit_z * bb_mask_2d.unsqueeze(-1)

        # Amino acid type one-hot
        aatype_1hot = F.one_hot(template_aatype, self.embed_aatype_row.in_features).to(dtype)  # [N, C]

        # Sum all 9 projected features
        act = self.embed_dgram(dgram.to(dtype))
        act = act + self.embed_pseudo_beta(pb_mask_2d.unsqueeze(-1).to(dtype))
        act = act + self.embed_aatype_row(aatype_1hot[None, :, :].expand(N, -1, -1))  # broadcast row
        act = act + self.embed_aatype_col(aatype_1hot[:, None, :].expand(-1, N, -1))  # broadcast col
        act = act + self.embed_unit_vec_x(unit_x.to(dtype))
        act = act + self.embed_unit_vec_y(unit_y.to(dtype))
        act = act + self.embed_unit_vec_z(unit_z.to(dtype))
        act = act + self.embed_backbone_mask(bb_mask_2d.unsqueeze(-1).to(dtype))
        act = act + self.embed_query(self.query_norm(query_pair))

        # Template PairFormer stack
        for layer in self.pairformer:
            act = layer(act, pair_mask)

        # Output normalization
        act = self.output_norm(act)
        return act


class TemplateEmbedding(nn.Module):
    """Embed a set of structural templates into pair-space.

    Processes each template independently through SingleTemplateEmbedding,
    averages the results, applies ReLU, and projects to pair_channel dims.

    Haiku scope: diffuser/evoformer/template_embedding/

    Attribute map:
        output_linear → output_proj
        single_template_embedding/ → single_template_emb
    """

    def __init__(self, cfg: TemplateEmbeddingConfig):
        super().__init__()
        self.cfg = cfg
        self.single_template_emb = SingleTemplateEmbedding(cfg)
        self.output_proj = LinearNoBias(cfg.num_channels, cfg.pair_channel)

    def forward(
        self,
        query_pair: torch.Tensor,
        template_aatype: torch.Tensor,
        template_pseudo_beta: torch.Tensor,
        template_pseudo_beta_mask: torch.Tensor,
        template_backbone_frame: torch.Tensor,
        template_backbone_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        multichain_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Embed T templates and produce pair-space embedding.

        Args:
            query_pair: [N, N, pair_channel] query pair embedding from Evoformer.
            template_aatype: [T, N] template residue types.
            template_pseudo_beta: [T, N, 3] pseudo-beta positions.
            template_pseudo_beta_mask: [T, N] pseudo-beta mask.
            template_backbone_frame: [T, N, 3, 3] backbone rotation matrices.
            template_backbone_mask: [T, N] backbone frame mask.
            pair_mask: [N, N] padding mask for attention.
            multichain_mask: [N, N] intra-chain mask.

        Returns:
            Template pair embedding [N, N, pair_channel].
        """
        T = template_aatype.shape[0]
        N = query_pair.shape[0]
        device = query_pair.device
        dtype = query_pair.dtype

        # Sum embeddings across templates
        summed = torch.zeros(N, N, self.cfg.num_channels, device=device, dtype=dtype)
        for t in range(T):
            emb = self.single_template_emb(
                query_pair=query_pair,
                template_aatype=template_aatype[t],
                template_pseudo_beta=template_pseudo_beta[t],
                template_pseudo_beta_mask=template_pseudo_beta_mask[t],
                template_backbone_frame=template_backbone_frame[t],
                template_backbone_mask=template_backbone_mask[t],
                pair_mask=pair_mask,
                multichain_mask=multichain_mask,
            )
            summed = summed + emb

        # Average over templates (with epsilon for safety)
        embedding = summed / (T + 1e-7)

        # ReLU + project to pair_channel
        embedding = F.relu(embedding)
        embedding = self.output_proj(embedding)

        return embedding
