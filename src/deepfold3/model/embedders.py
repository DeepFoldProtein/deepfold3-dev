"""Input feature embedders.

Produces ``target_feat`` [N_tok, seq_channel] from token-level features,
and relative position encodings for the pair channel.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_rel_feat_dim(max_relative_idx: int = 32, max_relative_chain: int = 2) -> int:
    """Compute the output dimension of :func:`create_relative_encoding`."""
    return 2 * (2 * max_relative_idx + 2) + 1 + (2 * max_relative_chain + 2)


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
    final_offset = torch.where(asym_id_same, clipped, 2 * max_relative_idx + 1)
    rel_pos = F.one_hot(final_offset, n_res_bins).float()  # [..., N, N, 66]

    # rel_token: token_index offset within same residue, cross-residue → extra bin
    token_offset = token_index.unsqueeze(-1) - token_index.unsqueeze(-2)
    clipped_tok = (token_offset + max_relative_idx).clamp(0, 2 * max_relative_idx)
    residue_same = asym_id_same & (residue_index.unsqueeze(-1) == residue_index.unsqueeze(-2))
    final_token_offset = torch.where(residue_same, clipped_tok, 2 * max_relative_idx + 1)
    rel_token = F.one_hot(final_token_offset, n_res_bins).float()  # [..., N, N, 66]

    # entity_id_same: [..., N, N, 1]
    entity_same_feat = entity_id_same.float().unsqueeze(-1)

    # rel_chain: sym_id offset, clipped, cross-entity → extra bin
    rel_sym_id = sym_id.unsqueeze(-1) - sym_id.unsqueeze(-2)
    clipped_chain = (rel_sym_id + max_relative_chain).clamp(0, 2 * max_relative_chain)
    final_rel_chain = torch.where(entity_id_same, clipped_chain, 2 * max_relative_chain + 1)
    rel_chain = F.one_hot(final_rel_chain, n_chain_bins).float()  # [..., N, N, 6]

    return torch.cat([rel_pos, rel_token, entity_same_feat, rel_chain], dim=-1)  # [..., N, N, 139]


# Distogram config (matches AF3 DistogramFeaturesConfig defaults)
DGRAM_MIN_BIN: float = 3.25
DGRAM_MAX_BIN: float = 50.75
DGRAM_NUM_BINS: int = 39


def dgram_from_positions(
    positions: torch.Tensor,
    min_bin: float = DGRAM_MIN_BIN,
    max_bin: float = DGRAM_MAX_BIN,
    num_bins: int = DGRAM_NUM_BINS,
) -> torch.Tensor:
    """Compute pairwise distance histogram from positions.

    Bins are defined by squared-distance boundaries, following AF3 convention.
    Supports both unbatched ``[N, 3]`` and batched ``[..., N, 3]`` inputs.

    Args:
        positions: ``[..., N, 3]`` atom positions.
        min_bin: Left edge of first distance bin (Angstroms).
        max_bin: Left edge of last distance bin (Angstroms).
        num_bins: Number of histogram bins.

    Returns:
        Distance histogram ``[..., N, N, num_bins]`` float32.
    """
    lower_breaks = torch.linspace(min_bin, max_bin, num_bins, device=positions.device)
    lower_breaks = lower_breaks**2
    upper_breaks = torch.cat([lower_breaks[1:], lower_breaks.new_tensor([1e8])])

    dist2 = ((positions.unsqueeze(-2) - positions.unsqueeze(-3)) ** 2).sum(-1, keepdim=True)
    return ((dist2 > lower_breaks) & (dist2 < upper_breaks)).float()
