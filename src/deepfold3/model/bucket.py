"""Bucket-based padding for torch.compile efficiency.

torch.compile traces/compiles a new kernel for each unique input shape.
By padding sequence length N to fixed bucket sizes, we ensure a small number
of compiled variants regardless of actual input length.

Pair operations are O(N²), so bucket spacing is denser at small N to keep
the pair waste ratio (bucket_N² / actual_N²) under ~1.56x.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from deepfold3.model.model import DeepFold3Output

BUCKET_SIZES = [
    128,
    160,
    192,
    224,
    256,
    320,
    384,
    448,
    512,
    640,
    768,
    896,
    1024,
    1280,
    1536,
    1792,
    2048,
    2560,
    3072,
    3584,
    4096,
]

# Features where dim 0 = N
DIM0_KEYS = {
    "res_type",
    "residue_index",
    "token_index",
    "asym_id",
    "entity_id",
    "sym_id",
    "mol_type",
    "token_res_name",
    "seq_mask",
    "frames_mask",
    "frame_atom_index",
    "frame_token_index",
    "token_to_rep_atom",
    "chirality_atom_idx",
    "chirality_target",
    "ref_pos",
    "ref_mask",
    "ref_element",
    "ref_charge",
    "ref_atom_name_chars",
    "ref_space_uid",
    "profile",
    "deletion_mean",
    # Training-only ground truth + masks
    "atom_gt_positions",
    "atom_gt_mask",
    "mask_fixed",
    "mask_region",
    "mask_loss_weight",
}

# Features where dims 0 and 1 = N, N
DIM01_KEYS = {"pair_mask", "bond_matrix"}

# Features where dim 1 = N
DIM1_KEYS = {
    "msa",
    "msa_mask",
    "deletion_matrix",
    "template_aatype",
    "template_pseudo_beta",
    "template_pseudo_beta_mask",
    "template_backbone_frame",
    "template_backbone_mask",
}

# Features that should be passed through ``pad_features`` unchanged because
# they are NOT N-dependent (per-bond, per-symmetry-pair, or scalar metadata).
# Listed explicitly so the unknown-key warning stays useful for genuinely
# missing entries.
PAD_PASSTHROUGH_KEYS = {
    # Bond loss (size = number of bonds in the sample)
    "bond_atom_indices",
    "bond_gt_distances",
    "bond_loss_mask",
    # Symmetry permutation tables (size = number of swap pairs)
    "atom_swap_pairs",
    "ligand_swap_pairs",
    # Per-sample scalars / metadata
    "resolution",
    "is_distillation",
}


def next_bucket(n: int, buckets: list[int] | None = None) -> int:
    """Return the smallest bucket size >= n."""
    if buckets is None:
        buckets = BUCKET_SIZES
    for b in buckets:
        if b >= n:
            return b
    return n  # larger than all buckets — no padding


def pad_features(
    features: dict[str, torch.Tensor],
    bucket_n: int | None = None,
) -> tuple[dict[str, torch.Tensor], int]:
    """Pad all N-dependent features to the next bucket size.

    Masks (seq_mask, pair_mask, msa_mask, ref_mask, frames_mask,
    template_*_mask) are padded with False so the model ignores
    padded positions.

    Args:
        features: Feature dict from featurizer.
        bucket_n: Target padded size. If None, computed from next_bucket().

    Returns:
        (padded_features, original_n)
    """
    n = features["res_type"].shape[0]
    if bucket_n is None:
        bucket_n = next_bucket(n)

    if bucket_n <= n:
        return features, n

    pad_n = bucket_n - n
    padded = {}

    for key, val in features.items():
        if key in DIM0_KEYS:
            padded[key] = _pad_dim(val, 0, pad_n)
        elif key in DIM01_KEYS:
            padded[key] = _pad_dim(_pad_dim(val, 0, pad_n), 1, pad_n)
        elif key in DIM1_KEYS:
            padded[key] = _pad_dim(val, 1, pad_n)
        elif key in PAD_PASSTHROUGH_KEYS:
            padded[key] = val
        else:
            warnings.warn(f"bucket.pad_features: unknown key {key!r} — passed through without padding", stacklevel=2)
            padded[key] = val

    return padded, n


def _pad_dim(t: torch.Tensor, dim: int, pad_size: int) -> torch.Tensor:
    """Pad a tensor with zeros along the given dimension."""
    if pad_size <= 0:
        return t
    # F.pad takes pairs in reverse dimension order: (last_dim_left, last_dim_right, ..., first_dim_left, first_dim_right)
    ndim = t.dim()
    # Build padding list: 2 entries per dimension, in reverse order
    pad_list = [0] * (2 * ndim)
    # dim from the end: reversed_idx = ndim - 1 - dim
    # In F.pad ordering: position 2 * reversed_idx + 1 = right pad for that dim
    reversed_idx = ndim - 1 - dim
    pad_list[2 * reversed_idx + 1] = pad_size
    return F.pad(t, pad_list, value=0)


def unpad_features(
    features: dict[str, torch.Tensor],
    original_n: int,
) -> dict[str, torch.Tensor]:
    """Slice padded features back to original sequence length."""
    result = {}
    for key, val in features.items():
        if key in DIM0_KEYS:
            result[key] = val.narrow(0, 0, original_n)
        elif key in DIM01_KEYS:
            result[key] = val.narrow(0, 0, original_n).narrow(1, 0, original_n)
        elif key in DIM1_KEYS:
            result[key] = val.narrow(1, 0, original_n)
        else:
            result[key] = val
    return result


def unpad_output(output: DeepFold3Output, original_n: int) -> DeepFold3Output:
    """Slice padded DeepFold3Output back to original sequence length.

    Args:
        output: DeepFold3Output with padded N dimensions.
        original_n: Original (unpadded) sequence length.

    Returns:
        DeepFold3Output with N dimensions sliced to original_n.
    """
    from deepfold3.model.model import DeepFold3Output  # noqa: F811

    n = original_n
    return DeepFold3Output(
        atom_positions=output.atom_positions[:, :n],
        atom_mask=output.atom_mask[:, :n],
        plddt=output.plddt[:, :n] if output.plddt is not None else None,
        pae=output.pae[:, :n, :n] if output.pae is not None else None,
        pde=output.pde[:, :n, :n] if output.pde is not None else None,
        exp_resolved=output.exp_resolved[:, :n] if output.exp_resolved is not None else None,
        distogram=output.distogram[:n, :n] if output.distogram is not None else None,
    )
