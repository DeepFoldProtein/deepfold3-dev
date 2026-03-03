"""Input feature embedders.

Produces ``target_feat`` [N_tok, seq_channel] from token-level features,
and relative position encodings for the pair channel.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def create_relative_encoding(
    token_index: torch.Tensor,  # [N]  int64
    residue_index: torch.Tensor,  # [N]  int64
    asym_id: torch.Tensor,  # [N]  int64
    entity_id: torch.Tensor,  # [N]  int64
    sym_id: torch.Tensor,  # [N]  int64
    max_relative_idx: int = 32,
    max_relative_chain: int = 2,
) -> torch.Tensor:
    """Create relative position encoding features [N, N, rel_feat_dim].

    Matches AF3 featurization.create_relative_encoding() exactly.

    Components:
        rel_pos:   residue_index diff, clamped, extra bin for cross-chain = 2*max_rel_idx+2 = 66
        rel_token: token_index diff within residue, extra bin for cross-residue = 2*max_rel_idx+2 = 66
        entity_id_same: bool(entity_i == entity_j)                              = 1
        rel_chain: sym_id diff, extra bin for cross-entity                       = 2*max_rel_chain+2 = 6
    Total = 66 + 66 + 1 + 6 = 139
    """
    n_res_bins = 2 * max_relative_idx + 2  # 66
    n_chain_bins = 2 * max_relative_chain + 2  # 6

    # Use negative indices: [..., N] → pairwise [..., N, N]
    asym_id_same = asym_id.unsqueeze(-1) == asym_id.unsqueeze(-2)  # [..., N, N] bool
    entity_id_same = entity_id.unsqueeze(-1) == entity_id.unsqueeze(-2)  # [..., N, N] bool

    # rel_pos: residue_index offset, clipped, cross-chain → extra bin
    offset = residue_index.unsqueeze(-1) - residue_index.unsqueeze(-2)
    clipped = (offset + max_relative_idx).clamp(0, 2 * max_relative_idx)
    extra_bin = torch.full_like(clipped, 2 * max_relative_idx + 1)
    final_offset = torch.where(asym_id_same, clipped, extra_bin)
    rel_pos = F.one_hot(final_offset, n_res_bins).float()  # [..., N, N, 66]

    # rel_token: token_index offset within same residue, cross-residue → extra bin
    token_offset = token_index.unsqueeze(-1) - token_index.unsqueeze(-2)
    clipped_tok = (token_offset + max_relative_idx).clamp(0, 2 * max_relative_idx)
    residue_same = asym_id_same & (residue_index.unsqueeze(-1) == residue_index.unsqueeze(-2))
    extra_bin_tok = torch.full_like(clipped_tok, 2 * max_relative_idx + 1)
    final_token_offset = torch.where(residue_same, clipped_tok, extra_bin_tok)
    rel_token = F.one_hot(final_token_offset, n_res_bins).float()  # [..., N, N, 66]

    # entity_id_same: [..., N, N, 1]
    entity_same_feat = entity_id_same.float().unsqueeze(-1)

    # rel_chain: sym_id offset, clipped, cross-entity → extra bin
    rel_sym_id = sym_id.unsqueeze(-1) - sym_id.unsqueeze(-2)
    clipped_chain = (rel_sym_id + max_relative_chain).clamp(0, 2 * max_relative_chain)
    extra_bin_chain = torch.full_like(clipped_chain, 2 * max_relative_chain + 1)
    final_rel_chain = torch.where(entity_id_same, clipped_chain, extra_bin_chain)
    rel_chain = F.one_hot(final_rel_chain, n_chain_bins).float()  # [..., N, N, 6]

    return torch.cat([rel_pos, rel_token, entity_same_feat, rel_chain], dim=-1)  # [..., N, N, 139]


class InputEmbedder(nn.Module):
    """Produce target_feat [N, seq_channel] from token features.

    AF3 uses a large one-hot feature for amino acid types, etc.
    Here we produce a simple embedding.

    The 'target_feat' fed into Evoformer has dimension seq_channel=384, but
    the input is a concatenation of:
        - res_type one-hot: 32 classes (20 AA + 1 UNK + RNA + DNA + ligand)
        - Other features (profile, etc.)
    We use a single Linear to map from the raw feature to seq_channel.
    """

    def __init__(self, feat_dim: int, seq_channel: int):
        super().__init__()
        self.proj = nn.Linear(feat_dim, seq_channel)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.proj(feats)
