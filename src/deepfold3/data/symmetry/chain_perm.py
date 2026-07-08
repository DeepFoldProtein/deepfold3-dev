"""Chain permutation alignment for homo- and hetero-multimer training.

In a multimeric assembly, chains that share an ``entity_id`` (i.e. the
same sequence) are physically indistinguishable — the prediction is
free to place "copy 0" at what the ground-truth labels "copy 1", and
vice versa. Without an explicit realignment, the loss would penalise
such perfectly-correct predictions and inject spurious gradients into
the training loop.

This module computes the optimal ground-truth-to-prediction chain
assignment following AlphaFold-Multimer / AlphaFold 3 §4.2: chains are
grouped by ``entity_id``, a single global rigid frame is fitted from an
anchor chain, and each predicted chain is greedily matched to the
spatially nearest ground-truth chain of the same entity. A per-pair
Kabsch cost is *not* used — it removes the global rotation and so scores
only internal shape (near-identical, hence ambiguous, for symmetric
copies), which cannot resolve the spatial correspondence the permutation
exists for. Entity groups of size 1 are passed through unchanged.

Heteromers (e.g. A2B2) are handled naturally: the anchor's global frame
plus the per-entity greedy match assign each entity's copies independently.

Clean-room implementation; no third-party permutation code is
reproduced. Only standard numpy/torch primitives are used.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import torch

logger = logging.getLogger(__name__)


def _kabsch_transform_np(p: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rigid transform ``(R, t)`` mapping ``p`` onto ``q`` (Kabsch 1976).

    Both inputs are ``[K, 3]`` float64 arrays with matching row order.
    Returns ``R`` ``[3, 3]`` and ``t`` ``[3]`` such that ``R @ p_k + t ≈ q_k``,
    with a determinant-sign flip guarding against reflections. numpy (not
    torch) because LAPACK's 3×3 SVD has far lower per-call overhead.
    """
    import numpy as np

    p_mean = p.mean(axis=0)
    q_mean = q.mean(axis=0)
    p_c = p - p_mean
    q_c = q - q_mean

    # 3×3 cross-covariance; order chosen so R maps p_c → q_c.
    h = p_c.T @ q_c
    u, _s, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    diag = np.eye(3, dtype=np.float64)
    diag[2, 2] = d
    r = vt.T @ diag @ u.T
    t = q_mean - r @ p_mean
    return r, t


def _anchor_greedy_assignment(
    order: list[int],
    ent_of: dict[int, int],
    cen_pred: dict[int, np.ndarray | None],
    cen_gt: dict[int, np.ndarray | None],
    at_pred: dict[int, np.ndarray],
    at_gt: dict[int, np.ndarray],
    res: dict[int, np.ndarray],
) -> dict[int, int]:
    """AF-Multimer anchor-based greedy chain assignment (AF3 §4.2).

    Unlike a per-pair Kabsch cost (which removes the global rotation and so
    scores *internal shape* similarity — near-zero and ambiguous for
    identical chains), this fits a **single global frame** from an anchor
    chain and then greedily matches each predicted chain to the spatially
    nearest ground-truth chain of the same entity. Spatial position — the
    thing the permutation must resolve — drives the assignment.

    Algorithm:
      1. Anchor entity = the entity with the fewest resolved chains (least
         ambiguous), tie-broken toward more resolved atoms; ``gt_anchor`` =
         its most-resolved chain.
      2. For each predicted chain of the anchor entity (the candidate global
         frames), Kabsch-align it onto ``gt_anchor``, apply that transform to
         every predicted chain centroid, and greedily assign each predicted
         chain to the nearest unused GT chain of its entity.
      3. Keep the frame with the smallest total matched squared distance.

    Per-chain inputs are keyed by chain id: ``cen_*`` are resolved-atom
    centroids (``None`` if fully unresolved); ``at_*`` / ``res`` are the
    per-chain resolved-atom coordinate blocks + boolean mask used only for
    the anchor Kabsch (chains of one entity share their atom layout, so the
    anchor pair's blocks line up row-for-row).
    """
    from collections import defaultdict

    import numpy as np

    assignment = {a: a for a in order}

    groups: dict[int, list[int]] = defaultdict(list)
    for a in order:
        groups[ent_of[a]].append(a)
    if not any(len(v) > 1 for v in groups.values()):
        return assignment  # no symmetric copies → nothing to permute

    resolved_chains = [a for a in order if cen_pred.get(a) is not None and cen_gt.get(a) is not None]
    if len(resolved_chains) < 2:
        return assignment

    ent_res: dict[int, list[int]] = defaultdict(list)
    for a in resolved_chains:
        ent_res[ent_of[a]].append(a)
    natoms = {a: int(res[a].sum()) for a in resolved_chains}

    anchor_entity = min(
        ent_res.keys(),
        key=lambda e: (len(ent_res[e]), -max(natoms[a] for a in ent_res[e])),
    )
    gt_anchor = max(ent_res[anchor_entity], key=lambda a: natoms[a])
    # Candidate global frames: predicted chains of the anchor entity whose
    # atom layout matches gt_anchor's (row-for-row Kabsch correspondence).
    candidates = [a for a in ent_res[anchor_entity] if res[a].shape[0] == res[gt_anchor].shape[0]]
    if not candidates:
        return assignment

    best: dict[int, int] | None = None
    best_score = np.inf
    for pa in candidates:
        both = res[pa] & res[gt_anchor]
        if int(both.sum()) < 3:
            continue
        r_mat, t_vec = _kabsch_transform_np(at_pred[pa][both], at_gt[gt_anchor][both])
        trial = {a: a for a in order}
        score = 0.0
        for chs in groups.values():
            gt_avail = [c for c in chs if cen_gt.get(c) is not None]
            used: set[int] = set()
            for pc in [c for c in chs if cen_pred.get(c) is not None]:
                pt = r_mat @ cen_pred[pc] + t_vec
                best_d: float | None = None
                best_g: int | None = None
                for gc in gt_avail:
                    if gc in used:
                        continue
                    d = float(((pt - cen_gt[gc]) ** 2).sum())
                    if best_d is None or d < best_d:
                        best_d = d
                        best_g = gc
                if best_g is not None:
                    used.add(best_g)
                    trial[pc] = best_g
                    score += best_d
        if score < best_score:
            best_score = score
            best = trial

    return best if best is not None else assignment


def compute_chain_permutation(
    pred_coords: torch.Tensor,  # [N_atoms, 3]
    gt_coords: torch.Tensor,  # [N_atoms, 3]
    gt_mask: torch.Tensor,  # [N_atoms] bool
    chains: np.ndarray,  # CHAIN_DTYPE structured array
) -> dict[int, int]:
    """Find the optimal pred-to-GT chain mapping for a multimer.

    Flat atom-layout adapter around :func:`_anchor_greedy_assignment`
    (AF-Multimer / AF3 §4.2): chains are grouped by ``entity_id``, a
    single global rigid frame is fitted from an anchor chain, and each
    predicted chain is greedily matched to the spatially nearest GT chain
    of the same entity over resolved heavy atoms. This resolves the
    spatial correspondence between symmetric copies — which a
    rotation-invariant per-pair Kabsch cost cannot, since it collapses to
    a near-constant matrix for identical chains and yields an arbitrary
    assignment.

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

    if len(chains) == 0:
        return {}

    order: list[int] = [int(r["asym_id"]) for r in chains]
    ent_of: dict[int, int] = {int(r["asym_id"]): int(r["entity_id"]) for r in chains}
    cen_pred: dict[int, np.ndarray | None] = {}
    cen_gt: dict[int, np.ndarray | None] = {}
    at_pred: dict[int, np.ndarray] = {}
    at_gt: dict[int, np.ndarray] = {}
    res: dict[int, np.ndarray] = {}

    with torch.no_grad():
        for row in chains:
            a = int(row["asym_id"])
            start = int(row["atom_idx"])
            num = int(row["atom_num"])
            end = start + num
            m = gt_mask[start:end].detach().cpu().numpy().astype(bool) if num > 0 else np.zeros(0, dtype=bool)
            if num == 0 or not m.any():
                cen_pred[a] = None
                cen_gt[a] = None
                continue
            pf = pred_coords[start:end].detach().cpu().numpy().astype(np.float64, copy=False)
            gf = gt_coords[start:end].detach().cpu().numpy().astype(np.float64, copy=False)
            at_pred[a] = pf
            at_gt[a] = gf
            res[a] = m
            cen_pred[a] = pf[m].mean(axis=0)
            cen_gt[a] = gf[m].mean(axis=0)

    return _anchor_greedy_assignment(order, ent_of, cen_pred, cen_gt, at_pred, at_gt, res)


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
    loss. Tokens are grouped by ``asym_id`` (each chain is one group) and
    chains into permutation classes by ``entity_id``. The pred → GT chain
    assignment is resolved by :func:`_anchor_greedy_assignment` — the
    AF-Multimer anchor-based greedy algorithm (AF3 §4.2): a single global
    frame is fitted from an anchor chain and each predicted chain is matched
    to the spatially nearest GT chain of the same entity. This resolves the
    spatial correspondence between symmetric copies that a per-pair Kabsch
    cost (rotation-invariant, hence blind to position) cannot.

    Returns ``{pred_asym_id: gt_asym_id}``. Identity for chains in singleton
    entity classes or when no valid anchor frame exists.
    """
    import numpy as np
    import torch

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

    ent_of: dict[int, int] = {}
    cen_pred: dict[int, np.ndarray | None] = {}
    cen_gt: dict[int, np.ndarray | None] = {}
    at_pred: dict[int, np.ndarray] = {}
    at_gt: dict[int, np.ndarray] = {}
    res: dict[int, np.ndarray] = {}

    with torch.no_grad():
        for asym in chain_order:
            idx = np.where(asym_ids_np == asym)[0]
            ent_of[asym] = int(entity_ids_np[idx[0]])
            tok_idx = torch.as_tensor(idx, dtype=torch.long, device=pred_pos.device)
            # Flatten (token × atom) so chains of one entity share a row layout.
            m = atom_mask[tok_idx].reshape(-1).detach().cpu().numpy().astype(bool)  # [T*M]
            if not m.any():
                cen_pred[asym] = None
                cen_gt[asym] = None
                continue
            pf = pred_pos[tok_idx].reshape(-1, 3).detach().cpu().numpy().astype(np.float64, copy=False)
            gf = gt_pos[tok_idx].reshape(-1, 3).detach().cpu().numpy().astype(np.float64, copy=False)
            at_pred[asym] = pf
            at_gt[asym] = gf
            res[asym] = m
            cen_pred[asym] = pf[m].mean(axis=0)
            cen_gt[asym] = gf[m].mean(axis=0)

    return _anchor_greedy_assignment(chain_order, ent_of, cen_pred, cen_gt, at_pred, at_gt, res)


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
