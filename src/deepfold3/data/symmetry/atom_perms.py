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


def build_residue_symmetry_pairs(res_name: str, atom_names: list[str]) -> list[tuple[int, int]]:
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
        atom_names = [atom_names_bytes[a_start + i].decode("ascii", errors="replace").strip() for i in range(a_num)]
        local = build_residue_symmetry_pairs(res_name, atom_names)
        pair_list.extend((a_start + a, a_start + b) for a, b in local)
    if not pair_list:
        return np.empty((0, 2), dtype=np.int32)
    return np.asarray(pair_list, dtype=np.int32)


def compute_token_atom_symmetry_pairs(structure, tokens: np.ndarray) -> np.ndarray:
    """Token-aware variant: emit ``(token_idx, M_a, M_b)`` swap triples.

    The training loss receives atoms in the per-token ``[N, M, 3]`` layout
    (one row per token, ``M=MAX_ATOMS_PER_TOKEN`` slots), not the flat
    ``[N_atoms, 3]`` layout used by :func:`compute_atom_symmetry_pairs`.
    This helper walks each token, looks up its residue type from the
    token array (so it works post-crop where the residues table is
    intentionally not re-indexed), and converts the residue's symmetry
    pairs to local ``[0..M)`` atom slot indices that
    :func:`apply_token_atom_permutation` can apply directly to the loss
    tensors.

    Polymer tokens span multiple atoms via ``token['atom_idx']`` and
    ``token['atom_num']``, which point into ``structure.atoms`` (the
    cropper keeps these consistent — atoms are sliced and atom_idx is
    remapped). Ligand tokens carry one atom per token and have no
    per-token symmetry; they are skipped.
    """
    import numpy as np

    from deepfold3.data import const

    pair_list: list[tuple[int, int, int]] = []
    atom_names_bytes = structure.atoms["name"]

    for token_idx, tok in enumerate(tokens):
        atom_num = int(tok["atom_num"])
        if atom_num <= 1:
            continue
        res_type = int(tok["res_type"])
        res_name = const.RES_TYPE_INDEX_TO_NAME.get(res_type)
        if res_name is None or res_name not in RESIDUE_ATOM_SYMMETRY_GROUPS:
            continue
        atom_idx = int(tok["atom_idx"])
        atom_names = [atom_names_bytes[atom_idx + i].decode("ascii", errors="replace").strip() for i in range(atom_num)]
        for a, b in build_residue_symmetry_pairs(res_name, atom_names):
            pair_list.append((token_idx, a, b))
    if not pair_list:
        return np.empty((0, 3), dtype=np.int32)
    return np.asarray(pair_list, dtype=np.int32)


def compute_ligand_token_swap_pairs(
    structure,
    tokens: np.ndarray,
    ligand_perm_cache: dict[str, list[list[int]]] | None,
) -> np.ndarray:
    """Token-pair swaps for ligand atoms via the precomputed CCD automorphism cache.

    Each ligand atom in our layout occupies its own token (``atom_num=1``,
    only slot ``M=0``), so ligand symmetries become **inter-token** swaps.
    For every ligand chain we look up its CCD residue name, fetch the
    automorphism group from ``ligand_perm_cache`` (the JSON dict produced
    by ``deepfold3 build-ccd-cache --automorphisms``), extract the involution pairs
    from each automorphism, and map them back to current token indices
    via ``token['residue_index']`` — which the tokenizer sets to the
    local-within-ligand atom index, so the mapping survives cropping.

    Returns a ``[N_pairs, 3]`` int32 array of ``(token_a, token_b, group_id)``
    triples. ``group_id`` is unique per emitted pair, so ligand swaps are
    decided independently — but the pairs of a ligand are first reduced to a
    **valid matching** (no atom in two pairs), which removes the previous
    hazard of applying overlapping transpositions (e.g. ``(1,2)`` and
    ``(1,3)`` pulled from two different automorphisms) that swap one atom
    twice into an invalid combination. Only involution (2-cycle) components
    are used; odd cycles (e.g. a 3-fold rotation) cannot be expressed as
    simultaneous swaps and are skipped — the identity is a safe fallback.
    """
    import numpy as np

    from deepfold3.data import const

    empty = np.empty((0, 3), dtype=np.int32)
    if not ligand_perm_cache:
        return empty

    # Group cropped ligand tokens by asym_id; each group is one ligand
    # chain (= one residue under our prepare pipeline).
    ligand_groups: dict[int, dict[int, int]] = {}
    for i, tok in enumerate(tokens):
        if int(tok["mol_type"]) != const.MOL_NONPOLYMER:
            continue
        asym = int(tok["asym_id"])
        local_atom_idx = int(tok["residue_index"])
        ligand_groups.setdefault(asym, {})[local_atom_idx] = i

    if not ligand_groups:
        return empty

    # asym_id -> CCD code via the structure.chains/residues lookup
    chains = structure.chains
    residues = structure.residues

    pair_list: list[tuple[int, int, int]] = []
    group_id = 0
    for asym, local_to_token in ligand_groups.items():
        chain_rows = chains[chains["asym_id"] == asym]
        if len(chain_rows) == 0:
            continue
        res_idx = int(chain_rows[0]["res_idx"])
        if res_idx >= len(residues):
            continue
        ccd_code = residues[res_idx]["name"].decode("ascii", errors="replace").strip().upper()
        perms = ligand_perm_cache.get(ccd_code)
        if not perms or len(perms) <= 1:
            continue
        pair_set: set[tuple[int, int]] = set()
        for perm in perms:
            for i, j in enumerate(perm):
                if j > i and 0 <= j < len(perm) and perm[j] == i:
                    pair_set.add((i, j))
        # Reduce to a valid matching: greedily keep pairs whose atoms haven't
        # already been claimed, so no atom is swapped by two pairs at once.
        used_local: set[int] = set()
        for local_a, local_b in sorted(pair_set):
            if local_a in used_local or local_b in used_local:
                continue
            tok_a = local_to_token.get(local_a)
            tok_b = local_to_token.get(local_b)
            if tok_a is not None and tok_b is not None:
                used_local.add(local_a)
                used_local.add(local_b)
                pair_list.append((tok_a, tok_b, group_id))
                group_id += 1

    if not pair_list:
        return empty
    return np.asarray(pair_list, dtype=np.int32)


def _residue_name_for_atom(structure, atom_idx: int) -> str | None:
    """Name of the residue whose atom range contains ``atom_idx`` (else None).

    Linear scan over ``structure.residues`` — used only for the few atomized
    residues per sample, so O(residues) per group is fine. Robust to a
    partially-cropped residue (the surviving atoms still fall in the range).
    """
    for res in structure.residues:
        start = int(res["atom_idx"])
        if start <= atom_idx < start + int(res["atom_num"]):
            return res["name"].decode("ascii", errors="replace").strip()
    return None


def compute_atomized_residue_token_swap_pairs(structure, tokens: np.ndarray) -> np.ndarray:
    """Inter-token symmetry swaps for *atomized* standard residues.

    When a standard residue is stochastically atomized (``atom_num == 1`` per
    token, polymer ``mol_type``), its side-chain symmetric atoms — e.g. ASP
    OD1/OD2 — become separate tokens, so the swap is **inter-token** (like
    ligands) rather than the ``(token, M_a, M_b)`` intra-token form that
    :func:`compute_token_atom_symmetry_pairs` emits.

    The swap source is the curated :data:`RESIDUE_ATOM_SYMMETRY_GROUPS` table
    (AF2/AF3 §4.3), **not** the CCD automorphism cache — the latter misses HIS
    (ND1/CD2 are different elements) and would add spurious O↔OXT swaps that are
    invalid in a polymer context. Residue identity is read from the **structure**
    (atomized tokens carry ``res_type == UNK``, matching modified residues, so
    the token no longer names the residue), and atom tokens are grouped back
    into residues by ``(asym_id, residue_index)``.

    Returns a ``[N_pairs, 3]`` int32 array of ``(token_a, token_b, group_id)``
    triples, applied at slot 0 by :func:`apply_ligand_token_swap`. All pairs of
    one residue share a ``group_id`` so a symmetric ring (PHE/TYR/HIS: CD and CE
    pairs are separate tokens once atomized) flips *as a unit* rather than
    permitting a half-flip.
    """
    import numpy as np

    from deepfold3.data import const

    atom_names_bytes = structure.atoms["name"]

    # Group atomized polymer tokens by (asym_id, residue_index).
    groups: dict[tuple[int, int], list[int]] = {}
    for i, tok in enumerate(tokens):
        if int(tok["mol_type"]) == const.MOL_NONPOLYMER:
            continue
        if int(tok["atom_num"]) != 1:
            continue  # not atomized (full residue token) — handled elsewhere
        key = (int(tok["asym_id"]), int(tok["residue_index"]))
        groups.setdefault(key, []).append(i)

    pair_list: list[tuple[int, int, int]] = []
    group_id = 0
    for token_indices in groups.values():
        if len(token_indices) < 2:
            continue
        # Real residue identity from the structure — the token res_type is UNK
        # for atomized tokens (matching modified residues), so read the name via
        # the group's first atom index.
        res_name = _residue_name_for_atom(structure, int(tokens[token_indices[0]]["atom_idx"]))
        if res_name is None or res_name not in RESIDUE_ATOM_SYMMETRY_GROUPS:
            continue
        # Local atom-name → token index within this residue.
        name_to_tok: dict[str, int] = {}
        for ti in token_indices:
            a_idx = int(tokens[ti]["atom_idx"])
            name = atom_names_bytes[a_idx].decode("ascii", errors="replace").strip()
            name_to_tok[name] = ti
        atom_names = list(name_to_tok.keys())
        emitted = False
        for a, b in build_residue_symmetry_pairs(res_name, atom_names):
            # build_residue_symmetry_pairs returns indices into ``atom_names``;
            # map back through the name to the token index.
            tok_a = name_to_tok[atom_names[a]]
            tok_b = name_to_tok[atom_names[b]]
            pair_list.append((tok_a, tok_b, group_id))
            emitted = True
        if emitted:
            group_id += 1

    if not pair_list:
        return np.empty((0, 3), dtype=np.int32)
    return np.asarray(pair_list, dtype=np.int32)


def apply_ligand_token_swap(
    pred_pos: torch.Tensor,  # [..., N, M, 3]
    gt_pos: torch.Tensor,  # [..., N, M, 3]
    atom_mask: torch.Tensor,  # [N, M] bool
    swap_pairs: np.ndarray | torch.Tensor,  # [N_pairs, 2] or [N_pairs, 3] (a, b, group)
) -> torch.Tensor:
    """Apply best-of-two inter-token atom swaps in the token-atom layout.

    Ligand / atomized-residue atoms live at slot ``M=0`` of their own token,
    so a swap is a (``token_a``, ``token_b``) row-pair at position 0. Pairs
    that share a **group id** (optional third column) are decided *jointly* —
    the whole group swaps or not. This lets an atomized PHE/TYR/HIS ring flip
    as a unit (its CD/CE pairs are separate tokens, so per-pair decisions
    would allow a chemically-impossible half-flip). A 2-column input is
    treated as one pair per group (independent), matching the legacy path.
    Decisions are per leading dimension and made in the caller's aligned frame.
    """
    import torch

    if swap_pairs is None or len(swap_pairs) == 0:
        return gt_pos.detach()

    if not isinstance(swap_pairs, torch.Tensor):
        pairs = torch.as_tensor(swap_pairs, dtype=torch.long, device=gt_pos.device)
    else:
        pairs = swap_pairs.to(dtype=torch.long, device=gt_pos.device)

    # Drop collate zero-padding self-pairs (a == b) up front — padded rows are
    # (0, 0, 0) and must not join real group 0 or scatter over token 0's swap.
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    if pairs.numel() == 0:
        return gt_pos.detach()

    permuted = gt_pos.clone()
    a_idx = pairs[:, 0]
    b_idx = pairs[:, 1]
    # Group id: explicit 3rd column, else each pair is its own group.
    group = pairs[:, 2] if pairs.shape[1] >= 3 else torch.arange(pairs.shape[0], device=pairs.device)

    pair_mask = atom_mask[a_idx, 0] & atom_mask[b_idx, 0]  # [P]
    if not pair_mask.any():
        return permuted.detach()

    pred_a = pred_pos[..., a_idx, 0, :]
    pred_b = pred_pos[..., b_idx, 0, :]
    gt_a = gt_pos[..., a_idx, 0, :]
    gt_b = gt_pos[..., b_idx, 0, :]

    identity_cost = (pred_a - gt_a).pow(2).sum(dim=-1) + (pred_b - gt_b).pow(2).sum(dim=-1)
    swapped_cost = (pred_a - gt_b).pow(2).sum(dim=-1) + (pred_b - gt_a).pow(2).sum(dim=-1)

    valid = pair_mask.to(identity_cost.dtype)
    identity_cost = identity_cost * valid
    swapped_cost = swapped_cost * valid

    # Sum per group and decide the group jointly.
    _uniq, inv = torch.unique(group, return_inverse=True)
    ng = int(_uniq.numel())
    lead = identity_cost.shape[:-1]
    id_g = torch.zeros(*lead, ng, dtype=identity_cost.dtype, device=identity_cost.device).index_add_(-1, inv, identity_cost)
    sw_g = torch.zeros(*lead, ng, dtype=swapped_cost.dtype, device=swapped_cost.device).index_add_(-1, inv, swapped_cost)
    do_swap = (sw_g < id_g).index_select(-1, inv) & pair_mask

    if not do_swap.any():
        return permuted.detach()

    new_a = torch.where(do_swap.unsqueeze(-1), gt_b, gt_a)
    new_b = torch.where(do_swap.unsqueeze(-1), gt_a, gt_b)
    permuted[..., a_idx, 0, :] = new_a
    permuted[..., b_idx, 0, :] = new_b

    return permuted.detach()


def apply_token_atom_permutation(
    pred_pos: torch.Tensor,  # [..., N, M, 3]
    gt_pos: torch.Tensor,  # [..., N, M, 3]
    atom_mask: torch.Tensor,  # [N, M] bool
    swap_pairs: np.ndarray | torch.Tensor,  # [N_pairs, 3] (token, a, b)
) -> torch.Tensor:
    """Per-token best-swap atom permutation in the ``[..., N, M, 3]`` layout.

    The swap decision is made **jointly over all pairs of a token** (each
    token = one residue = one symmetry group): the cost of every pair is
    summed and the whole group is swapped or not. This is what keeps a
    PHE/TYR/HIS ring flipping *as a unit* (CD1↔CD2 and CE1↔CE2 together) —
    deciding the pairs independently would permit a chemically-impossible
    half-flip (swap CD but not CE). Decisions are made independently per
    leading dimension (each augmented diffusion sample picks its own swap)
    and, per finding, in the rigid-aligned frame set up by the caller.

    Returns a tensor with the same shape as ``gt_pos``, detached.
    """
    import torch

    if swap_pairs is None or len(swap_pairs) == 0:
        return gt_pos.detach()

    if not isinstance(swap_pairs, torch.Tensor):
        pairs = torch.as_tensor(swap_pairs, dtype=torch.long, device=gt_pos.device)
    else:
        pairs = swap_pairs.to(dtype=torch.long, device=gt_pos.device)

    # Drop collate zero-padding self-pairs (a == b) up front so they neither
    # join a real group's cost nor scatter over a genuine swap.
    pairs = pairs[pairs[:, 1] != pairs[:, 2]]
    if pairs.numel() == 0:
        return gt_pos.detach()

    permuted = gt_pos.clone()

    tok_idx = pairs[:, 0]
    a_idx = pairs[:, 1]
    b_idx = pairs[:, 2]

    pair_mask = atom_mask[tok_idx, a_idx] & atom_mask[tok_idx, b_idx]  # [P]
    if not pair_mask.any():
        return permuted.detach()

    # Gather per-pair atoms across all leading dims
    pred_a = pred_pos[..., tok_idx, a_idx, :]  # [..., P, 3]
    pred_b = pred_pos[..., tok_idx, b_idx, :]
    gt_a = gt_pos[..., tok_idx, a_idx, :]
    gt_b = gt_pos[..., tok_idx, b_idx, :]

    identity_cost = (pred_a - gt_a).pow(2).sum(dim=-1) + (pred_b - gt_b).pow(2).sum(dim=-1)
    swapped_cost = (pred_a - gt_b).pow(2).sum(dim=-1) + (pred_b - gt_a).pow(2).sum(dim=-1)

    # Zero masked pairs so a partially-resolved residue decides on its
    # resolved pairs only (0 == 0 contributes nothing to the group cost).
    valid = pair_mask.to(identity_cost.dtype)  # [P]
    identity_cost = identity_cost * valid
    swapped_cost = swapped_cost * valid

    # Sum costs per group (token) and decide the whole group together.
    uniq_tok, inv = torch.unique(tok_idx, return_inverse=True)  # inv: [P] → group id
    g = uniq_tok.numel()
    lead = identity_cost.shape[:-1]
    id_g = torch.zeros(*lead, g, dtype=identity_cost.dtype, device=identity_cost.device).index_add_(-1, inv, identity_cost)
    sw_g = torch.zeros(*lead, g, dtype=swapped_cost.dtype, device=swapped_cost.device).index_add_(-1, inv, swapped_cost)
    do_swap = (sw_g < id_g).index_select(-1, inv) & pair_mask  # [..., P]
    if not do_swap.any():
        return permuted.detach()

    new_a = torch.where(do_swap.unsqueeze(-1), gt_b, gt_a)
    new_b = torch.where(do_swap.unsqueeze(-1), gt_a, gt_b)

    permuted[..., tok_idx, a_idx, :] = new_a
    permuted[..., tok_idx, b_idx, :] = new_b

    return permuted.detach()


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
