"""Chain permutation alignment for homo- and hetero-multimer training.

In a multimeric assembly, chains that share an ``entity_id`` (i.e. the
same sequence) are physically indistinguishable — the prediction is
free to place "copy 0" at what the ground-truth labels "copy 1", and
vice versa. Without an explicit realignment, the loss would penalise
such perfectly-correct predictions and inject spurious gradients into
the training loop.

This module computes the optimal ground-truth-to-prediction chain
assignment following AlphaFold 3 §4.2: chains are grouped by
``entity_id``, and each group is permuted independently by solving a
linear assignment problem on a per-pair distance cost matrix. Entity
groups of size 1 are passed through unchanged.

Heteromers (e.g. A2B2) are handled naturally because the assignment is
solved one entity at a time — the permutation over the A copies is
independent of the permutation over the B copies.

Clean-room implementation; no third-party permutation code is
reproduced. Only :func:`scipy.optimize.linear_sum_assignment` and
standard numpy/torch primitives are used.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import torch

logger = logging.getLogger(__name__)


def compute_chain_permutation(
    pred_coords: torch.Tensor,  # [N_atoms, 3]
    gt_coords: torch.Tensor,  # [N_atoms, 3]
    gt_mask: torch.Tensor,  # [N_atoms] bool
    chains: np.ndarray,  # CHAIN_DTYPE structured array
) -> dict[int, int]:
    """Find the optimal pred-to-GT chain mapping for a multimer.

    Chains are grouped by ``entity_id``. For each entity group with more
    than one copy, an ``N×N`` cost matrix is built where
    ``cost[i, j]`` is the squared distance between the resolved-atom
    centroids of pred chain ``i`` and GT chain ``j``. Pairs whose chain
    lengths differ (should not happen when ``entity_id`` is derived from
    the sequence, but handled defensively) are masked to ``+inf``. The
    optimal assignment is then obtained from
    :func:`scipy.optimize.linear_sum_assignment`.

    The centroid-distance cost is a coarse but effective surrogate for
    the per-atom RMSD suggested in AlphaFold 3 §4.2: it is invariant to
    the arbitrary intra-chain atom labelling and avoids fitting a
    per-pair Kabsch transform, which would be wasteful for what is
    ultimately just a discrete assignment decision.

    Parameters
    ----------
    pred_coords : torch.Tensor [N_atoms, 3]
        Predicted atom coordinates in the flat atom layout described by
        ``chains``. Any device / dtype is accepted.
    gt_coords : torch.Tensor [N_atoms, 3]
        Ground-truth atom coordinates, same layout as ``pred_coords``.
    gt_mask : torch.Tensor [N_atoms], bool
        True at atoms with valid (resolved) GT positions. Atoms where
        this mask is False are excluded from the centroid computation.
    chains : np.ndarray
        Structured array with ``CHAIN_DTYPE`` fields — at minimum
        ``entity_id``, ``asym_id``, ``atom_idx`` and ``atom_num``.

    Returns
    -------
    dict[int, int]
        Mapping ``{pred_asym_id: gt_asym_id}`` giving the GT chain that
        should be matched to each pred chain. Chains whose entity has
        only one copy are mapped to themselves. Chains whose atoms are
        all masked fall back to the identity mapping.
    """
    import numpy as np
    import torch
    from scipy.optimize import linear_sum_assignment

    assignment: dict[int, int] = {}

    if len(chains) == 0:
        return assignment

    # Default: identity (caller can rely on every pred asym_id being present).
    for row in chains:
        assignment[int(row["asym_id"])] = int(row["asym_id"])

    # Group by entity_id.
    entity_ids = np.asarray(chains["entity_id"])
    unique_entities = np.unique(entity_ids)

    with torch.no_grad():
        for entity_id in unique_entities:
            group_mask = entity_ids == entity_id
            group_rows = chains[group_mask]
            n = len(group_rows)
            if n <= 1:
                continue

            asym_ids = [int(r["asym_id"]) for r in group_rows]
            atom_starts = [int(r["atom_idx"]) for r in group_rows]
            atom_nums = [int(r["atom_num"]) for r in group_rows]

            # Pre-compute per-chain pred and GT centroids (resolved atoms only).
            pred_centroids: list[torch.Tensor | None] = []
            gt_centroids: list[torch.Tensor | None] = []
            for start, num in zip(atom_starts, atom_nums, strict=True):
                end = start + num
                m = gt_mask[start:end]
                if num == 0 or not bool(m.any()):
                    pred_centroids.append(None)
                    gt_centroids.append(None)
                    continue
                m_f = m.to(pred_coords.dtype)
                denom = m_f.sum().clamp_min(1.0)
                pred_c = (pred_coords[start:end] * m_f.unsqueeze(-1)).sum(dim=0) / denom
                gt_c = (gt_coords[start:end] * m_f.unsqueeze(-1)).sum(dim=0) / denom
                pred_centroids.append(pred_c)
                gt_centroids.append(gt_c)

            inf = float("inf")
            cost = np.full((n, n), inf, dtype=np.float64)
            for i in range(n):
                if pred_centroids[i] is None:
                    continue
                for j in range(n):
                    if gt_centroids[j] is None:
                        continue
                    # Defensive length check — same entity should imply same
                    # length, but we drop mismatches rather than crash.
                    if atom_nums[i] != atom_nums[j]:
                        continue
                    diff = pred_centroids[i] - gt_centroids[j]
                    cost[i, j] = float((diff * diff).sum().item())

            if not np.isfinite(cost).any():
                # Entire group is unresolved — fall back to identity.
                logger.debug(
                    "chain_perm: entity_id=%s has no resolvable pairs; "
                    "using identity mapping",
                    entity_id,
                )
                continue

            # linear_sum_assignment does not tolerate rows/cols that are
            # entirely +inf. Fill any such entries with a large finite
            # penalty so the solver still runs; the assignment on the
            # degenerate rows will be arbitrary but is overridden below.
            finite_mask = np.isfinite(cost)
            row_ok = finite_mask.any(axis=1)
            col_ok = finite_mask.any(axis=0)
            if not (row_ok.all() and col_ok.all()):
                big = cost[finite_mask].max() * 1e3 + 1.0 if finite_mask.any() else 1.0
                cost = np.where(finite_mask, cost, big)

            row_ind, col_ind = linear_sum_assignment(cost)
            for i, j in zip(row_ind, col_ind, strict=True):
                # Skip unresolved pred chains — keep their identity entry.
                if not row_ok[i]:
                    continue
                assignment[asym_ids[i]] = asym_ids[j]

    return assignment


def apply_chain_permutation(
    coords: torch.Tensor,  # [N_atoms, 3]
    mask: torch.Tensor,  # [N_atoms] bool
    chains: np.ndarray,
    chain_assignment: dict[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reorder GT atom blocks according to a chain assignment.

    For each chain in ``chains`` (iterated in pred order), the returned
    tensors contain the atoms from the GT chain given by
    ``chain_assignment[asym_id]``. Chain lengths must match for the
    slices to line up — mismatched entries are skipped by the
    assignment builder, so the fallback here is the identity copy.

    Parameters
    ----------
    coords : torch.Tensor [N_atoms, 3]
        Typically the GT coordinates to be reordered. Any device / dtype.
    mask : torch.Tensor [N_atoms], bool
        Per-atom validity mask aligned with ``coords``.
    chains : np.ndarray
        Structured array with ``CHAIN_DTYPE`` fields.
    chain_assignment : dict[int, int]
        ``{pred_asym_id: gt_asym_id}`` mapping as returned by
        :func:`compute_chain_permutation`.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(permuted_coords, permuted_mask)`` with the same total length
        as the inputs. Both are detached from any input autograd graph.
    """
    import torch

    if len(chains) == 0:
        return coords.detach(), mask.detach()

    # Fast path: no-op if the assignment is identity everywhere.
    if all(int(k) == int(v) for k, v in chain_assignment.items()):
        return coords.detach(), mask.detach()

    # Lookup: asym_id -> (atom_idx, atom_num)
    chain_by_asym: dict[int, tuple[int, int]] = {
        int(r["asym_id"]): (int(r["atom_idx"]), int(r["atom_num"])) for r in chains
    }

    n_atoms = coords.shape[0]
    index = torch.arange(n_atoms, dtype=torch.long, device=coords.device)

    for row in chains:
        pred_asym = int(row["asym_id"])
        pred_start = int(row["atom_idx"])
        pred_num = int(row["atom_num"])
        gt_asym = int(chain_assignment.get(pred_asym, pred_asym))
        if gt_asym == pred_asym:
            continue
        gt_start, gt_num = chain_by_asym.get(gt_asym, (pred_start, pred_num))
        if gt_num != pred_num:
            # Length mismatch — cannot safely reorder, leave the slice in place.
            logger.debug(
                "chain_perm: length mismatch for pred asym_id=%d -> gt asym_id=%d "
                "(pred=%d, gt=%d); leaving slice unchanged",
                pred_asym,
                gt_asym,
                pred_num,
                gt_num,
            )
            continue
        index[pred_start : pred_start + pred_num] = torch.arange(
            gt_start, gt_start + gt_num, dtype=torch.long, device=coords.device
        )

    permuted_coords = coords.index_select(0, index).detach()
    permuted_mask = mask.index_select(0, index).detach()
    return permuted_coords, permuted_mask


# ---------------------------------------------------------------------------
# Token-aware variants for the diffusion loss path
# ---------------------------------------------------------------------------


def compute_chain_permutation_tokens(
    pred_pos: torch.Tensor,  # [N, M, 3]
    gt_pos: torch.Tensor,  # [N, M, 3]
    atom_mask: torch.Tensor,  # [N, M] bool
    asym_id: torch.Tensor,  # [N] int
    entity_id: torch.Tensor,  # [N] int
) -> dict[int, int]:
    """Token-layout version of :func:`compute_chain_permutation`.

    Operates on the per-token ``[N, M, 3]`` layout used by the diffusion
    loss. Tokens are grouped by ``asym_id`` (each chain is one group of
    contiguous tokens) and chains are grouped into permutation classes
    by ``entity_id``. For each entity class with multiple chains, the
    optimal pred → GT chain assignment is computed via Hungarian on a
    centroid-distance cost matrix; chains whose token counts disagree
    (e.g. cropped on different boundaries) are left in place since the
    token-by-token correspondence isn't well defined.

    Returns ``{pred_asym_id: gt_asym_id}``. Identity for chains in
    singleton entity classes or for length-mismatched groups.
    """
    import numpy as np
    import torch
    from scipy.optimize import linear_sum_assignment

    asym_ids_np = asym_id.detach().cpu().numpy()
    entity_ids_np = entity_id.detach().cpu().numpy()

    # Iterate chains in first-appearance order to keep results stable
    seen: set[int] = set()
    chain_order: list[int] = []
    for a in asym_ids_np.tolist():
        if a in seen:
            continue
        seen.add(a)
        chain_order.append(int(a))

    chain_token_indices: dict[int, np.ndarray] = {}
    chain_entity: dict[int, int] = {}
    for asym in chain_order:
        idx = np.where(asym_ids_np == asym)[0]
        chain_token_indices[asym] = idx
        chain_entity[asym] = int(entity_ids_np[idx[0]])

    assignment: dict[int, int] = {a: a for a in chain_order}

    # Group by entity_id
    entity_to_chains: dict[int, list[int]] = {}
    for asym in chain_order:
        entity_to_chains.setdefault(chain_entity[asym], []).append(asym)

    with torch.no_grad():
        for asym_list in entity_to_chains.values():
            n = len(asym_list)
            if n <= 1:
                continue

            # Per-chain centroids over (token × atom) pairs that are valid.
            pred_centroids: list[torch.Tensor | None] = []
            gt_centroids: list[torch.Tensor | None] = []
            token_counts: list[int] = []
            for asym in asym_list:
                tok_idx = torch.as_tensor(
                    chain_token_indices[asym], dtype=torch.long, device=pred_pos.device
                )
                token_counts.append(int(tok_idx.numel()))
                m = atom_mask[tok_idx]  # [n_tok, M]
                if not m.any():
                    pred_centroids.append(None)
                    gt_centroids.append(None)
                    continue
                m_f = m.to(pred_pos.dtype)
                denom = m_f.sum().clamp_min(1.0)
                pred_c = (pred_pos[tok_idx] * m_f.unsqueeze(-1)).sum(dim=(0, 1)) / denom
                gt_c = (gt_pos[tok_idx] * m_f.unsqueeze(-1)).sum(dim=(0, 1)) / denom
                pred_centroids.append(pred_c)
                gt_centroids.append(gt_c)

            inf = float("inf")
            cost = np.full((n, n), inf, dtype=np.float64)
            for i in range(n):
                if pred_centroids[i] is None:
                    continue
                for j in range(n):
                    if gt_centroids[j] is None:
                        continue
                    if token_counts[i] != token_counts[j]:
                        continue
                    diff = pred_centroids[i] - gt_centroids[j]
                    cost[i, j] = float((diff * diff).sum().item())

            if not np.isfinite(cost).any():
                continue

            finite_mask = np.isfinite(cost)
            row_ok = finite_mask.any(axis=1)
            if not finite_mask.all():
                big = cost[finite_mask].max() * 1e3 + 1.0
                cost = np.where(finite_mask, cost, big)

            row_ind, col_ind = linear_sum_assignment(cost)
            for i, j in zip(row_ind, col_ind, strict=True):
                if not row_ok[i]:
                    continue
                assignment[asym_list[i]] = asym_list[j]

    return assignment


def apply_chain_permutation_tokens(
    gt_pos: torch.Tensor,  # [..., N, M, 3]
    atom_mask: torch.Tensor,  # [N, M] bool
    asym_id: torch.Tensor,  # [N] int
    chain_assignment: dict[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reorder GT token rows according to a chain assignment.

    For each pred chain, the rows whose ``asym_id`` matches the pred
    chain are replaced with the rows of the GT chain given by the
    assignment. Length-mismatched groups are skipped (the assignment
    builder already does this so this branch is defensive). Returns the
    permuted ``gt_pos`` and matching mask, both detached.
    """
    import torch

    if not chain_assignment or all(int(k) == int(v) for k, v in chain_assignment.items()):
        return gt_pos.detach(), atom_mask.detach()

    n_tokens = gt_pos.shape[-3]
    asym_ids_t = asym_id.to(dtype=torch.long, device=gt_pos.device)
    index = torch.arange(n_tokens, dtype=torch.long, device=gt_pos.device)

    # Pre-build asym -> token index list for the source side
    chain_by_asym: dict[int, torch.Tensor] = {}
    for a in chain_assignment.values():
        a_int = int(a)
        if a_int not in chain_by_asym:
            chain_by_asym[a_int] = torch.where(asym_ids_t == a_int)[0]

    for pred_asym, gt_asym in chain_assignment.items():
        if int(pred_asym) == int(gt_asym):
            continue
        pred_idx = torch.where(asym_ids_t == int(pred_asym))[0]
        gt_idx = chain_by_asym.get(int(gt_asym), pred_idx)
        if pred_idx.numel() != gt_idx.numel():
            continue
        index[pred_idx] = gt_idx

    permuted = gt_pos.index_select(-3, index).detach()
    permuted_mask = atom_mask.index_select(0, index).detach()
    return permuted, permuted_mask
