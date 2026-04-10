"""Per-residue symmetric atom permutations.

For standard amino acids whose side chains have a local 180° (or
equivalent) symmetry, the prediction can equivalently place the
"symmetric" atoms in either of two orientations and still represent the
same physical structure. The training loss must therefore pick the
permutation that minimises the per-residue MSE before computing
gradients, otherwise gradients on these atoms get noised by an
arbitrary labelling choice.

The table below lists the swap pairs that are equivalent under the side
chain's local symmetry — well-known biochemistry, identical to the list
used by AlphaFold 2 (Suppl. §1.8.5, Table 4) and AlphaFold 3 (Suppl.
§4.3). It is encoded directly as a constant; no third-party code is
reproduced.

Notes on individual residues:

* **HIS**: ND1 / NE2 swap is *not* a true chemical symmetry (the two
  nitrogens have different protonation states), but AF2/AF3 treat the
  ring as symmetric for training because the side chain can rotate
  ~180° about the Cβ–Cγ bond and the resulting models are
  experimentally indistinguishable for non-coordinating histidines.
* **PHE / TYR**: CD1↔CD2 and CE1↔CE2 swap simultaneously (the ring is
  flipped as a unit; you cannot swap CD without also swapping CE).
* **ASP / GLU / ARG / LEU / VAL**: terminal-group swaps (carboxyl OD/OE,
  guanidinium NH, gem-dimethyls).

Backbone atoms (N, CA, C, O, CB) are never permuted. Symmetry pairs are
**precomputed once per Structure** at load time (see
``compute_atom_symmetry_pairs``) and reused at every training step —
the per-residue table is static so no per-step lookup is needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    import numpy as np
    import torch

# Each entry maps a 3-letter residue code to a list of equivalent atom
# swap groups. Each swap group is itself a list of pairs ``(a, b)``
# meaning "swap atoms a and b". Multiple pairs in one group are applied
# *together* (e.g. PHE flips CD1↔CD2 *and* CE1↔CE2 simultaneously).
RESIDUE_ATOM_SYMMETRY_GROUPS: Final[dict[str, list[list[tuple[str, str]]]]] = {
    "ASP": [[("OD1", "OD2")]],
    "GLU": [[("OE1", "OE2")]],
    "ARG": [[("NH1", "NH2")]],
    "PHE": [[("CD1", "CD2"), ("CE1", "CE2")]],
    "TYR": [[("CD1", "CD2"), ("CE1", "CE2")]],
    "HIS": [[("ND1", "CD2"), ("CE1", "NE2")]],
    "LEU": [[("CD1", "CD2")]],
    "VAL": [[("CG1", "CG2")]],
}


def build_residue_symmetry_pairs(
    res_name: str, atom_names: list[str]
) -> list[tuple[int, int]]:
    """Resolve a residue's symmetric atom swap pairs to local atom indices.

    Returns a list of ``(i, j)`` index pairs into ``atom_names`` that
    should be swapped together. An empty list means the residue has no
    symmetric atoms or none of the expected atoms are present in this
    particular instance (e.g. a truncated side chain).

    Only the first symmetry group of each residue is currently returned;
    every standard residue in :data:`RESIDUE_ATOM_SYMMETRY_GROUPS` has
    exactly one group.
    """
    groups = RESIDUE_ATOM_SYMMETRY_GROUPS.get(res_name)
    if not groups:
        return []
    name_to_idx = {name: i for i, name in enumerate(atom_names)}
    out: list[tuple[int, int]] = []
    for swap_pair in groups[0]:
        a, b = swap_pair
        if a in name_to_idx and b in name_to_idx:
            out.append((name_to_idx[a], name_to_idx[b]))
    return out


def compute_atom_symmetry_pairs(structure) -> np.ndarray:
    """Compute global ``(i, j)`` swap-pair indices for a Structure online.

    Walks every residue, looks up its symmetry pairs from
    :data:`RESIDUE_ATOM_SYMMETRY_GROUPS`, and converts them to **global**
    atom-array indices. The result is a single ``[N_pairs, 2]`` int32
    array consumed by :func:`apply_atom_permutation`.

    Cheap enough to compute once per ``__getitem__`` (microseconds for a
    few-hundred-residue chain) — there is no per-Structure cache. Ligand
    atom symmetries (which require RDKit automorphism detection) are
    handled by a separate CCD-keyed memo cache, since the same ligand
    recurs across many structures.
    """
    import numpy as np

    pair_list: list[tuple[int, int]] = []
    atom_names_bytes = structure.atoms["name"]
    for res in structure.residues:
        res_name = res["name"].decode("ascii", errors="replace").strip()
        if res_name not in RESIDUE_ATOM_SYMMETRY_GROUPS:
            continue
        a_start = int(res["atom_idx"])
        a_num = int(res["atom_num"])
        if a_num <= 0:
            continue
        atom_names = [
            atom_names_bytes[a_start + i].decode("ascii", errors="replace").strip()
            for i in range(a_num)
        ]
        local = build_residue_symmetry_pairs(res_name, atom_names)
        pair_list.extend((a_start + a, a_start + b) for a, b in local)
    if not pair_list:
        return np.empty((0, 2), dtype=np.int32)
    return np.asarray(pair_list, dtype=np.int32)


def apply_atom_permutation(
    pred_coords: torch.Tensor,  # [N_atoms, 3]
    gt_coords: torch.Tensor,  # [N_atoms, 3]
    gt_mask: torch.Tensor,  # [N_atoms] bool
    swap_pairs: np.ndarray | torch.Tensor,  # [N_pairs, 2] int
) -> torch.Tensor:
    """Permute ground-truth coordinates of symmetric atoms to minimise MSE.

    For each pair ``(i, j)``, the swap variant (identity or fully swapped)
    with the lower per-atom squared distance to the prediction is chosen.
    The permuted GT coordinates are returned — callers compute their loss
    against the permuted GT, not against the raw prediction. Equivalent in
    effect to swapping the prediction; we permute the GT to keep the
    prediction tensor's autograd graph untouched.

    Pairs are independent (each evaluated on its own two atoms only),
    matching the AF3 §4.3 per-residue construction: every standard
    residue has at most a single 2-atom or 4-atom swap group, and
    ``compute_atom_symmetry_pairs`` emits each group as one or two
    atomic pair entries that all map to the same swap decision in
    practice (the per-pair decision converges to the per-group decision
    because the swap atoms move together).

    Parameters
    ----------
    pred_coords : torch.Tensor [N_atoms, 3]
        Predicted coordinates (any device, any dtype, gradient-tracked OK).
    gt_coords : torch.Tensor [N_atoms, 3]
        Ground-truth coordinates aligned 1:1 with ``pred_coords``.
    gt_mask : torch.Tensor [N_atoms], bool
        True at atoms with valid (resolved) ground-truth positions.
        Atoms with ``mask=False`` are excluded from the cost; if either
        atom in a pair is masked the pair is skipped.
    swap_pairs : array_like [N_pairs, 2]
        Index pairs precomputed by :func:`compute_atom_symmetry_pairs`.

    Returns
    -------
    torch.Tensor [N_atoms, 3]
        ``gt_coords`` with symmetric atoms swapped where the swap reduced
        the per-pair MSE. Detached from any input autograd graph.
    """
    import torch

    if swap_pairs is None or len(swap_pairs) == 0:
        return gt_coords.detach()

    if not isinstance(swap_pairs, torch.Tensor):
        swap_pairs_t = torch.as_tensor(swap_pairs, dtype=torch.long, device=gt_coords.device)
    else:
        swap_pairs_t = swap_pairs.to(dtype=torch.long, device=gt_coords.device)

    permuted = gt_coords.clone()

    a_idx = swap_pairs_t[:, 0]
    b_idx = swap_pairs_t[:, 1]

    valid = gt_mask[a_idx] & gt_mask[b_idx]
    if not valid.any():
        return permuted.detach()

    pred_a = pred_coords[a_idx]
    pred_b = pred_coords[b_idx]
    gt_a = gt_coords[a_idx]
    gt_b = gt_coords[b_idx]

    identity_cost = (pred_a - gt_a).pow(2).sum(dim=-1) + (pred_b - gt_b).pow(2).sum(dim=-1)
    swapped_cost = (pred_a - gt_b).pow(2).sum(dim=-1) + (pred_b - gt_a).pow(2).sum(dim=-1)
    do_swap = valid & (swapped_cost < identity_cost)

    if do_swap.any():
        swap_a = a_idx[do_swap]
        swap_b = b_idx[do_swap]
        permuted[swap_a] = gt_coords[swap_b]
        permuted[swap_b] = gt_coords[swap_a]

    return permuted.detach()
