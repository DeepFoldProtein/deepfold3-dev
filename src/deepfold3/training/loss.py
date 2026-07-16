"""Training loss functions for DeepFold3.

Implements the multi-component loss from AF3 §5.3 (eq. 15)::

    L = alpha_diffusion * L_diff + alpha_distogram * L_disto
      + alpha_confidence * (L_plddt + L_pde + L_resolved + alpha_pae * L_pae)

All loss functions are vectorized over the diffusion sample dimension S
to avoid Python-level loops and unnecessary GPU synchronization.
"""

from __future__ import annotations

import math
from contextlib import nullcontext

import torch
import torch.nn.functional as F  # noqa: N812

from deepfold3.data.masking import MASK_FIXED, MASK_LOSS_WEIGHT, MASK_REGION
from deepfold3.model.config import TrainingConfig

# ---------------------------------------------------------------------------
# Weighted rigid alignment (AF3 Algorithm 28 / §3.7.1 eq. 2)
# ---------------------------------------------------------------------------


def weighted_rigid_align(
    gt_pos: torch.Tensor,  # [..., K, 3] — point set to transform
    pred_pos: torch.Tensor,  # [..., K, 3] — target
    weights: torch.Tensor,  # [..., K] or [K] — per-point weights (0 = ignore)
    eps: float = 1e-6,
    stop_gradient: bool = True,
) -> torch.Tensor:
    """Weighted Kabsch alignment — rotate+translate ``gt_pos`` onto ``pred_pos``.

    Implements AF3 eq. (2) in §3.7.1:
    ``{x^GT-aligned} = weighted_rigid_align({x^GT}, {x}, {w})``.

    Runs in fp32 with autocast disabled regardless of ambient precision —
    ``torch.linalg.svd`` on the 3×3 covariance does not support bf16 and
    orthogonality of the rotation is numerically critical (Protenix
    ``model/loss.py:1136`` documents the same constraint).

    When ``stop_gradient=True`` (AF3 default), the computation runs under
    ``no_grad`` and the output is detached — gradients in the MSE flow
    exclusively through ``pred_pos``, so the alignment acts as a rotation
    target rather than a differentiable module.

    Args:
        gt_pos: Point set to rotate/translate onto ``pred_pos`` ``[..., K, 3]``.
        pred_pos: Target point set ``[..., K, 3]``.
        weights: Per-point non-negative weights, broadcastable to the batch
            dims of ``gt_pos``. Zero weights drop the point entirely.
        eps: Floor for the weight sum to avoid division by zero when an
            entire batch is fully masked out.
        stop_gradient: If True, run under ``no_grad`` and detach the output.

    Returns:
        ``gt_pos`` rigidly transformed onto ``pred_pos`` ``[..., K, 3]``,
        in the input dtype of ``pred_pos``.
    """
    out_dtype = pred_pos.dtype
    grad_ctx = torch.no_grad() if stop_gradient else nullcontext()
    with grad_ctx, torch.amp.autocast(device_type=gt_pos.device.type, enabled=False):
        gt32 = gt_pos.float()
        pred32 = pred_pos.float()
        w32 = weights.float()

        # Broadcast weights to gt shape's leading dims.
        while w32.dim() < gt32.dim() - 1:
            w32 = w32.unsqueeze(0).expand(gt32.shape[: -1 - w32.dim()] + w32.shape)
        w_exp = w32.unsqueeze(-1)  # [..., K, 1]
        w_sum = w32.sum(dim=-1, keepdim=True).unsqueeze(-1).clamp(min=eps)  # [..., 1, 1]

        gt_mu = (gt32 * w_exp).sum(dim=-2, keepdim=True) / w_sum  # [..., 1, 3]
        pred_mu = (pred32 * w_exp).sum(dim=-2, keepdim=True) / w_sum

        gt_c = gt32 - gt_mu
        pred_c = pred32 - pred_mu

        # Covariance H = Σ w · (gt_c)·(pred_c)^T → [..., 3, 3]
        H = torch.matmul((gt_c * w_exp).transpose(-2, -1), pred_c)

        U, _S, Vh = torch.linalg.svd(H)
        U_T = U.transpose(-2, -1)
        V = Vh.transpose(-2, -1)
        # Reflection fix: R = V · diag(1, 1, sign(det(V·U^T))) · U^T.
        # ``sign`` (not ``det`` directly) guards against the fp32 SVD
        # noise that lets ``det(V·U^T)`` drift from exactly ±1 — matches
        # the numpy path in ``chain_perm.py:_kabsch_rmsd_np``.
        det = torch.linalg.det(torch.matmul(V, U_T))
        sign = torch.sign(det)
        ones = torch.ones_like(sign)
        diag = torch.diag_embed(torch.stack([ones, ones, sign], dim=-1))
        rot = torch.matmul(V, torch.matmul(diag, U_T))  # [..., 3, 3]

        # Apply: (gt - μ_gt) @ R^T + μ_pred
        gt_aligned = torch.matmul(gt_c, rot.transpose(-2, -1)) + pred_mu

    if stop_gradient:
        gt_aligned = gt_aligned.detach()
    return gt_aligned.to(out_dtype)


# ---------------------------------------------------------------------------
# Numerically safe self-cdist
# ---------------------------------------------------------------------------


class _SafeSelfCdist(torch.autograd.Function):
    """Self pairwise-distance matrix with a singularity-free backward.

    ``torch.cdist(x, x)`` has backward ``(x_i - x_j) / d_ij``, which is
    NaN/inf when two predicted points coincide. We keep the fused ``cdist``
    forward for memory, and compute the backward as
    ``(x_i - x_j) / sqrt(d_ij^2 + eps)`` so the Jacobian norm is bounded
    by ``1/sqrt(2)`` instead of diverging at ``d = 0``.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, eps: float) -> torch.Tensor:
        d = torch.cdist(x, x)
        ctx.save_for_backward(x, d)
        ctx.eps = eps
        return d

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, d = ctx.saved_tensors
        # d is symmetric; symmetrize the upstream grad so each unordered
        # pair contributes once to both x_i and x_j.
        sym_grad = grad_out + grad_out.transpose(-2, -1)
        d_safe = torch.sqrt(d * d + ctx.eps)
        diff = x.unsqueeze(-2) - x.unsqueeze(-3)  # [..., A, A, 3]
        scale = (sym_grad / d_safe).unsqueeze(-1)
        grad_x = (scale * diff).sum(-2)
        return grad_x, None


def _safe_self_cdist(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return _SafeSelfCdist.apply(x, eps)


class _SafeSparseNorm(torch.autograd.Function):
    """Euclidean norm over the last axis with a singularity-free backward.

    Sparse analogue of :class:`_SafeSelfCdist`: forward returns the true
    distance ``‖diff‖`` (so the loss value stays exact), backward uses
    ``diff / sqrt(‖diff‖² + eps)`` so the gradient is bounded when two predicted
    atoms coincide (``‖diff‖ = 0``). Operates on the already-gathered
    ``[..., 3]`` difference vectors of a sparse pair list, so it never
    materialises a dense ``[A, A, 3]`` buffer the way ``cdist``'s backward does.
    """

    @staticmethod
    def forward(ctx, diff: torch.Tensor, eps: float) -> torch.Tensor:
        d = torch.linalg.vector_norm(diff, dim=-1)
        ctx.save_for_backward(diff, d)
        ctx.eps = eps
        return d

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        diff, d = ctx.saved_tensors
        d_safe = torch.sqrt(d * d + ctx.eps)
        grad_diff = (grad_out / d_safe).unsqueeze(-1) * diff
        return grad_diff, None


def _safe_sparse_norm(diff: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return _SafeSparseNorm.apply(diff, eps)


# ---------------------------------------------------------------------------
# Single-representation magnitude regularization (IntFold §4.x)
# ---------------------------------------------------------------------------


def single_repr_reg_loss(
    single: torch.Tensor,  # [..., N, C]
    seq_mask: torch.Tensor,  # [..., N] (bool or float)
    p: int,  # 1 or 2
) -> torch.Tensor:
    """Per-token ``L_p`` magnitude of the single representation, seq-mask averaged.

    ``p=1`` returns ``mean_{valid(b,i)}  Σ_c |s[b,i,c]| / C``.
    ``p=2`` returns ``mean_{valid(b,i)}  Σ_c s[b,i,c]² / C`` (squared, no sqrt
    — the gradient scales with the activation, so the penalty grows
    quadratically when magnitudes explode, which is IntFold's failure mode).

    Normalising by the channel count ``C`` keeps the value ~O(1) under healthy
    training regardless of model width. Computed in fp32 regardless of ambient
    autocast — large activations can overflow under bf16 and this loss exists
    precisely to trigger on magnitude blow-ups.
    """
    if p not in (1, 2):
        raise ValueError(f"single_reg_p must be 1 or 2, got {p}")
    mask = seq_mask.float()
    n_valid = mask.sum().clamp(min=1)
    C = single.shape[-1]
    s32 = single.float()
    if p == 1:
        per_token = s32.abs().sum(dim=-1) / C
    else:
        per_token = (s32 * s32).sum(dim=-1) / C
    return (per_token * mask).sum() / n_valid


def pair_repr_reg_loss(
    pair: torch.Tensor,  # [..., N, N, C]
    pair_mask: torch.Tensor,  # [..., N, N] (bool or float)
    p: int,
) -> torch.Tensor:
    """Per-pair ``L_p`` magnitude of the pair representation, pair-mask averaged.

    Same shape as :func:`single_repr_reg_loss` but reduced over the pair grid.
    IntFold reports the activation explosion first appears in the Transition
    of later Pairformer layers before reaching the single track — watching
    this catches the failure mode one step earlier.
    """
    if p not in (1, 2):
        raise ValueError(f"pair_reg_p must be 1 or 2, got {p}")
    mask = pair_mask.float()
    n_valid = mask.sum().clamp(min=1)
    C = pair.shape[-1]
    p32 = pair.float()
    if p == 1:
        per_pair = p32.abs().sum(dim=-1) / C
    else:
        per_pair = (p32 * p32).sum(dim=-1) / C
    return (per_pair * mask).sum() / n_valid


def single_repr_log_rms_loss(
    single: torch.Tensor,  # [..., N, C]
    seq_mask: torch.Tensor,  # [..., N]
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """RMS magnitude of the single representation under ``seq_mask``, plus its log.

    ``RMS = sqrt(mean over (valid token, channel) of single²)``. Computed in
    fp32 regardless of ambient autocast — large activations can overflow under
    bf16 and this metric exists to track magnitude blow-ups.

    Returns ``(rms, log_rms)`` so the caller can log the RMS value while using
    ``log_rms`` as the loss term — the log keeps the gradient bounded even
    when activations are very large.
    """
    mask = seq_mask.float()
    n_valid = mask.sum().clamp(min=1)
    C = single.shape[-1]
    s32 = single.float()
    sq_per_token = (s32 * s32).sum(dim=-1) / C  # [..., N]
    mean_sq = (sq_per_token * mask).sum() / n_valid
    rms = mean_sq.sqrt()
    log_rms = (rms + eps).log()
    return rms, log_rms


def pair_repr_log_rms_loss(
    pair: torch.Tensor,  # [..., N, N, C]
    pair_mask: torch.Tensor,  # [..., N, N]
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pair-RMS analogue of :func:`single_repr_log_rms_loss`."""
    mask = pair_mask.float()
    n_valid = mask.sum().clamp(min=1)
    C = pair.shape[-1]
    p32 = pair.float()
    sq_per_pair = (p32 * p32).sum(dim=-1) / C  # [..., N, N]
    mean_sq = (sq_per_pair * mask).sum() / n_valid
    rms = mean_sq.sqrt()
    log_rms = (rms + eps).log()
    return rms, log_rms


# ---------------------------------------------------------------------------
# Smooth LDDT loss (diffusion)
# ---------------------------------------------------------------------------


def smooth_lddt_loss(
    pred_pos: torch.Tensor,  # [S, N, M, 3]
    gt_pos: torch.Tensor,  # [N, M, 3]
    atom_mask: torch.Tensor,  # [N, M] bool
    cutoffs: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0),
    inclusion_radius: float = 15.0,  # non-nucleotide pairs
    nucleotide_inclusion_radius: float = 30.0,  # nucleotide-nucleotide pairs
    min_distance: float = 1e-3,
    mol_type: torch.Tensor | None = None,  # [N] per-token mol type
) -> torch.Tensor:
    """Smooth LDDT loss between predicted and ground-truth atom positions.

    Vectorized over S samples — no Python loops over diffusion dimension.

    The within-radius neighbour list is derived from the ground-truth positions
    only (no gradient), so the dense O(A²) distance matrix lives just long
    enough to select the sparse pair set and is then freed. The differentiable
    sigmoid scoring runs over that sparse ``[S, P]`` pair list rather than a
    dense ``[S, A, A]`` tensor — identical result, ~16× lower peak memory at
    large crops (A≈8k), where within-radius pairs are only ~5% of A².

    Pair-inclusion radius (AlphaFold 3 supplement eq. 6, Protenix
    ``loss.py:457``): nucleotide-nucleotide atom pairs use the larger
    ``nucleotide_inclusion_radius`` (30 Å), all other pairs use
    ``inclusion_radius`` (15 Å). When ``mol_type`` is not provided, the
    smaller radius is used for every pair (the safer default).

    Args:
        pred_pos: Predicted positions [S, N, M, 3].
        gt_pos: Ground-truth positions [N, M, 3].
        atom_mask: Valid atom mask [N, M].
        cutoffs: Distance error thresholds in Angstroms.
        inclusion_radius: Pair inclusion radius for non-nucleotide pairs.
        nucleotide_inclusion_radius: Pair inclusion radius for
            nucleotide-nucleotide pairs.
        min_distance: Minimum GT distance for pair inclusion.
        mol_type: Per-token molecule type ``[N]`` (0=prot, 1=DNA, 2=RNA,
            3=ligand). Used to choose the per-pair inclusion radius.

    Returns:
        Scalar loss (1 - smooth_lddt), averaged over samples.
    """
    flat_mask = atom_mask.reshape(-1).bool()
    A = flat_mask.sum()
    if A < 2:
        return pred_pos.new_zeros(())

    S = pred_pos.shape[0]
    gt_flat = gt_pos.reshape(-1, 3)[flat_mask]  # [A, 3]

    # Neighbour list from ground truth only. Built under ``no_grad`` so the
    # dense O(A²) distance matrix is a short-lived buffer, freed before the
    # differentiable scoring runs. We keep only the sparse set of within-radius
    # pairs (both directions, mirroring the symmetric dense mask); the scoring
    # below then operates on ``[S, P]`` tensors instead of a dense ``[S, A, A]``.
    # At large crops (A≈8k) within-radius pairs are ~5% of A², cutting this
    # loss's peak memory ~16× (measured 23.6 GB → 1.4 GB at A=8192, S=8) — the
    # dominant term once the crop grows past ~640 tokens.
    with torch.no_grad():
        gt_dist = torch.cdist(gt_flat, gt_flat)  # [A, A]
        if mol_type is not None:
            # Broadcast per-token mol_type to per-atom (each token holds M atoms).
            N, M = atom_mask.shape
            per_atom_mol_type = mol_type.view(N, 1).expand(N, M).reshape(-1)[flat_mask]  # [A]
            is_nuc = (per_atom_mol_type == 1) | (per_atom_mol_type == 2)
            # AF3 Alg 27 line 6: the inclusion radius is set by the reference
            # atom alone (30 A if it is a nucleotide, else 15 A), not by
            # requiring BOTH atoms to be nucleotides. Key the radius on the
            # neighbour (column j) atom, matching _compute_per_atom_lddt_batched
            # (radius_j). The old AND-rule dropped protein/nucleic pairs in the
            # [15, 30) A band that AF3 includes when the reference is nucleic.
            radius = torch.where(
                is_nuc.unsqueeze(0),  # [1, A] -> radius[i, j] depends on atom j
                gt_dist.new_full((), nucleotide_inclusion_radius),
                gt_dist.new_full((), inclusion_radius),
            )
            pair_mask = (gt_dist < radius) & (gt_dist > min_distance)
        else:
            pair_mask = (gt_dist < inclusion_radius) & (gt_dist > min_distance)
        pair_i, pair_j = pair_mask.nonzero(as_tuple=True)  # [P], [P]
        gt_pair_dist = gt_dist[pair_i, pair_j]  # [P]
        n_pairs = max(pair_i.numel(), 1)
    del gt_dist, pair_mask  # free the O(A²) buffers before building the grad graph

    # Differentiable scoring over the sparse neighbour list only: [S, P].
    pred_flat = pred_pos.reshape(S, -1, 3)[:, flat_mask]  # [S, A, 3]
    diff_vec = pred_flat[:, pair_i] - pred_flat[:, pair_j]  # [S, P, 3]
    pred_pair_dist = _safe_sparse_norm(diff_vec)  # [S, P]
    dist_diff = (pred_pair_dist - gt_pair_dist.unsqueeze(0)).abs()  # [S, P]

    # Accumulate the sigmoid-scored fraction one cutoff at a time
    # (AF3 Alg. 27: sigmoid(threshold - |Δd|)).
    score_sum = pred_pos.new_zeros(S)  # [S]
    for cutoff in cutoffs:
        score_sum = score_sum + torch.sigmoid(cutoff - dist_diff).sum(dim=-1)
    # average over cutoffs → [S]
    lddt_per_sample = score_sum / (n_pairs * len(cutoffs))

    return (1.0 - lddt_per_sample).mean()


# ---------------------------------------------------------------------------
# Diffusion MSE loss with EDM weighting
# ---------------------------------------------------------------------------


def _inter_chain_bonds_from_features(
    features: dict[str, torch.Tensor],
) -> list[tuple[int, int]] | None:
    """Derive covalent inter-chain bonds (as ``(asym_a, asym_b)`` pairs) for the
    chain-permutation component grouping, from the token bond adjacency.

    Reads the ``bond_matrix`` feature ([N, N, 1], emitted only when the example
    has token-level bonds) + ``asym_id`` + ``mol_type``. A pair of tokens on
    *different* chains linked by a token bond groups those chains into one
    covalently-connected permutation component — but ONLY when at least one
    endpoint is a NON-POLYMER (glycan / covalent ligand). Inter-chain
    polymer–polymer links (disulfides) do not group chains: OF3 strips those in
    ``prefilter_bonds`` before the permutation, and requiring a non-polymer
    endpoint gives the same net grouping whether or not the stored bonds were
    prefiltered. Returns ``None`` when there is nothing to group (→ each chain is
    its own component, the entity-only behaviour).
    """
    bm = features.get("bond_matrix")
    asym = features.get("asym_id")
    if bm is None or asym is None:
        return None
    # Loss runs per example → expect [N, N, 1]; bail on any other layout.
    if bm.dim() != 3 or bm.shape[0] != asym.shape[0]:
        return None
    with torch.no_grad():
        idx = (bm[..., 0] > 0).nonzero(as_tuple=False)  # [K, 2] token pairs
    if idx.numel() == 0:
        return None
    asym_np = asym.detach().cpu().numpy()
    mol = features.get("mol_type")
    mol_np = mol.detach().cpu().numpy() if mol is not None else None
    MOL_NONPOLYMER = 3
    pairs: set[tuple[int, int]] = set()
    for i, j in idx.tolist():
        if i >= j:
            continue  # symmetric adjacency → keep one direction
        ai, aj = int(asym_np[i]), int(asym_np[j])
        if ai == aj:
            continue  # intra-chain bond
        if mol_np is not None and not (int(mol_np[i]) == MOL_NONPOLYMER or int(mol_np[j]) == MOL_NONPOLYMER):
            continue  # polymer–polymer inter-chain link (disulfide) → not a grouping bond
        pairs.add((ai, aj) if ai < aj else (aj, ai))
    return sorted(pairs) or None


def chain_perm_from_mini_rollout(
    pred: dict[str, torch.Tensor],
    features: dict[str, torch.Tensor],
    stability_gate_k: int = 1,
) -> dict[int, int] | None:
    """AF3 §4.3 chain permutation from the mini-rollout structure.

    The mini-rollout is a multi-step denoised prediction (``no_grad``) in
    the same frame as the un-augmented GT, so a single chain assignment
    computed here can be applied to every augmented GT sample consumed
    by the diffusion MSE. Returns ``None`` when the inputs needed for
    permutation aren't present (``mini_rollout_pos``, ``asym_id``,
    ``entity_id``) — callers fall back to the legacy per-sample compute
    on the noisy single-step denoised prediction.

    Stability gate (``stability_gate_k > 1``): resolve the assignment
    independently from each of the first ``K`` mini-rollout samples and
    apply it ONLY when all ``K`` agree on the whole-assembly assignment;
    otherwise return the identity ({}). Independent rollout draws only agree
    once the symmetric copies genuinely separate, so while a homo-oligomer is
    collapsed the gate keeps a deterministic (un-permuted) target instead of
    the oscillating coin-flip that stalls symmetry-breaking (see
    ``export/chain-perm-latch/``). Unanimity — not majority — is required.
    """
    mini_pos = pred.get("mini_rollout_pos")
    asym_id = features.get("asym_id")
    entity_id = features.get("entity_id")
    gt_pos = features.get("atom_gt_positions")
    atom_mask = features.get("atom_gt_mask")
    if mini_pos is None or asym_id is None or entity_id is None or gt_pos is None or atom_mask is None:
        return None

    from deepfold3.data.symmetry.chain_perm import compute_chain_permutation_tokens

    # The K-sample vote reads the dedicated gate rollout (``chain_perm_gate_pos``,
    # populated by ``forward_training`` when the gate is on); ``mini_rollout_pos``
    # itself stays at ``mini_rollout_samples`` for the other consumers. Default
    # (gate off / no gate positions) → single-sample from ``mini_rollout_pos``.
    vote_pos = pred.get("chain_perm_gate_pos")
    if vote_pos is None or int(stability_gate_k) <= 1:
        vote_pos = mini_pos

    def _assign(sample_idx: int) -> dict[int, int]:
        return compute_chain_permutation_tokens(
            vote_pos[sample_idx],
            gt_pos,
            atom_mask.bool(),
            asym_id,
            entity_id,
            token_to_rep_atom=features.get("token_to_rep_atom"),
            mol_type=features.get("mol_type"),
            seq_mask=features.get("seq_mask"),
            sym_token_index=features.get("sym_token_index"),
            inter_chain_bonds=_inter_chain_bonds_from_features(features),
        )

    k = min(int(stability_gate_k), int(vote_pos.shape[0]))
    if k <= 1:
        return _assign(0)

    # Stability gate: unanimous whole-assignment vote over K independent draws.
    a0 = _assign(0)
    for s in range(1, k):
        if _assign(s) != a0:
            return {}  # disagreement → identity (deterministic separating target)
    return a0


def resolve_gt_atom_symmetry(
    ref: torch.Tensor,  # [..., N, M, 3] — reference prediction (mini-rollout), matching gt's leading dims
    features: dict[str, torch.Tensor],
    gt: torch.Tensor,  # [..., N, M, 3] — GT to relabel (gt_pos [N,M,3] or gt_aug [S,N,M,3])
    atom_mask: torch.Tensor,  # [N, M] bool — GT resolved mask
) -> torch.Tensor:
    """Resolve intra-residue / ligand atom symmetry against a reference (AF3 §4.2).

    AF3 §4.2: after the chain permutation, resolve atom-naming ambiguities within
    each chemical component (ASP OD1/OD2, PHE ring, PO4 oxygens, nucleic OP1/OP2,
    ligand automorphisms) by permuting the GT to minimal RMSD to **the
    prediction** — the mini-rollout — and this one resolution is shared by every
    GT-based loss (diffusion, confidence, …). This aligns ``ref`` onto ``gt``
    (Kabsch weighted by the representative/token-center atoms only — CA / C1' are
    non-symmetric and anchor a stable frame; an all-atom weight would let the
    symmetric atoms being resolved skew their own alignment) and applies the same
    atom / ligand permutation ops. Works for either the un-augmented GT
    (``[N,M,3]``) or the per-sample augmented GT (``[S,N,M,3]``); ``ref`` must
    share ``gt``'s leading dims (broadcast the single mini-rollout across S).
    No-op when the symmetry tables are absent.
    """
    atom_swap = features.get("atom_swap_pairs")
    lig_swap = features.get("ligand_swap_pairs")
    lig_perms = features.get("ligand_perms")
    has_atom = atom_swap is not None and len(atom_swap) > 0
    has_lswap = lig_swap is not None and len(lig_swap) > 0
    has_lperm = lig_perms is not None and len(lig_perms) > 0
    if not (has_atom or has_lswap or has_lperm):
        return gt

    from deepfold3.data.symmetry.atom_perms import (
        apply_ligand_token_permutation,
        apply_ligand_token_swap,
        apply_token_atom_permutation,
    )

    n, m = gt.shape[-3], gt.shape[-2]
    lead = gt.shape[:-3]
    b = 1
    for d in lead:
        b *= d
    gt_b = gt.reshape(b, n, m, 3)
    ref_b = ref.reshape(b, n, m, 3)

    rep_idx = features.get("token_to_rep_atom")
    if rep_idx is not None:
        w2d = torch.zeros(n, m, device=gt.device, dtype=gt.dtype)
        w2d[torch.arange(n, device=gt.device), rep_idx.to(gt.device)] = 1.0
        w = (w2d.bool() & atom_mask).reshape(-1).to(gt.dtype)
    else:
        w = atom_mask.reshape(-1).to(gt.dtype)

    with torch.no_grad():
        mref = weighted_rigid_align(ref_b.reshape(b, -1, 3).detach(), gt_b.reshape(b, -1, 3), w).reshape(b, n, m, 3)
        mask_b = atom_mask.bool()
        out = gt_b
        if has_atom:
            out = apply_token_atom_permutation(mref, out, mask_b, atom_swap)
        if has_lswap:
            out = apply_ligand_token_swap(mref, out, mask_b, lig_swap)
        if has_lperm:
            out = apply_ligand_token_permutation(mref, out, mask_b, lig_perms)
    return out.reshape(*lead, n, m, 3)


def permute_gt_for_diffusion(
    pred: dict[str, torch.Tensor],
    features: dict[str, torch.Tensor],
    stability_gate_k: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resolve the chain + intra-residue atom/ligand symmetry ONCE from the
    mini-rollout and return the permuted ``(gt_positions, gt_mask)`` — so the
    diffusion noise is added to the SAME labelling the losses score against
    (AF3 §4.2/4.3 permute-then-noise; OpenFold3 `model.py`, Protenix
    `permute_label_to_match_mini_rollout`).

    Per-example (``[N, M, 3]`` / ``[N, M]``). The chain permutation reorders the
    token axis (reassigning each prediction slot to its matched GT chain — only
    same-entity chains are ever swapped, so the slot's conditioning stays valid),
    and the atom/ligand permutation swaps intra-token symmetric slots. No-op
    passthrough when the mini-rollout / symmetry tables are absent. The loss then
    re-derives the SAME permutation for the un-augmented shared GT and skips
    re-permuting ``gt_augmented`` (guarded on ``features["__gt_prepermuted__"]``).
    """
    gt_pos = features["atom_gt_positions"]
    atom_mask = features["atom_gt_mask"]

    chain_assign = chain_perm_from_mini_rollout(pred, features, stability_gate_k=stability_gate_k)
    asym_id_feat = features.get("asym_id")
    if chain_assign and asym_id_feat is not None and not all(int(k) == int(v) for k, v in chain_assign.items()):
        from deepfold3.data.symmetry.chain_perm import chain_permutation_index

        c_index, c_keep = chain_permutation_index(
            asym_id_feat, chain_assign, features.get("sym_token_index"), device=gt_pos.device
        )
        gt_pos = gt_pos.index_select(0, c_index)
        atom_mask = atom_mask.index_select(0, c_index) & c_keep.unsqueeze(-1)

    mini_ref = pred.get("mini_rollout_pos")
    if mini_ref is not None:
        gt_pos = resolve_gt_atom_symmetry(mini_ref[0], features, gt_pos, atom_mask)
    return gt_pos, atom_mask


def diffusion_mse_loss(
    pred_pos: torch.Tensor,  # [S, N, M, 3]
    gt_pos: torch.Tensor,  # [S, N, M, 3] (augmented GT per sample)
    atom_mask: torch.Tensor,  # [N, M]
    noise_levels: torch.Tensor,  # [S]
    sigma_data: float = 16.0,
    mol_type: torch.Tensor | None = None,  # [N] int64 (0=prot, 1=DNA, 2=RNA, 3=ligand)
    nucleotide_loss_weight: float = 5.0,
    ligand_loss_weight: float = 10.0,
    region_weights: torch.Tensor | None = None,  # [N] float — per-token loss weight from region labels
    atom_swap_pairs: torch.Tensor | None = None,  # [N_pairs, 3] (token, M_a, M_b)
    ligand_swap_pairs: torch.Tensor | None = None,  # [N_pairs, 3] atomized-residue inter-token swaps
    ligand_perms: torch.Tensor | None = None,  # [R, 4] (ligand_id, perm_id, dst, src) CCD automorphisms
    asym_id: torch.Tensor | None = None,  # [N] int — per-token chain id
    entity_id: torch.Tensor | None = None,  # [N] int — per-token entity id (for chain perm grouping)
    chain_assign: dict[int, int] | None = None,  # pre-computed pred→GT chain mapping
    atom_perm_pred: torch.Tensor | None = None,  # [N, M, 3] — mini-rollout reference for atom/ligand swap decisions
    token_to_rep_atom: torch.Tensor | None = None,  # [N] int — rep-atom slot (chain-perm token centers)
    seq_mask: torch.Tensor | None = None,  # [N] bool — real (non-padding) tokens for chain perm
    sym_token_index: torch.Tensor | None = None,  # [N] int — per-chain ordinal (crop-robust chain-perm key)
    apply_rigid_align: bool = True,
) -> torch.Tensor:
    """Weighted MSE between denoised predictions and augmented GT.

    Vectorized over S — single fused operation, no Python loop.

    Permutation alignments (AlphaFold 3 §4.2 / §4.3) are applied to the
    ground truth before the loss is computed:

    1. **Chain permutation** (when ``asym_id`` and ``entity_id`` are
       provided): chains in the same entity class are reassigned via
       Hungarian on per-chain centroid distances. The decision is made
       per diffusion sample because each augmented GT carries a different
       random rotation. Skipped for crops where same-entity chains have
       different token counts (the token-by-token correspondence is
       not well defined).
    2. **Atom permutation** (when ``atom_swap_pairs`` is provided):
       symmetry-equivalent side-chain atoms (ASP OD1/OD2, PHE CD1/CD2,
       etc.) are swapped per (sample, pair) to whichever ordering
       minimises that pair's MSE.

    Args:
        pred_pos: Denoised predictions [S, N, M, 3].
        gt_pos: Augmented ground truth [S, N, M, 3].
        atom_mask: Valid atoms [N, M].
        noise_levels: Noise level per sample [S].
        sigma_data: Data standard deviation.
        mol_type: Per-token molecule type [N].
        nucleotide_loss_weight: Weight multiplier for RNA/DNA atoms.
        ligand_loss_weight: Weight multiplier for ligand atoms.
        region_weights: Per-token loss multiplier from region labels.
        atom_swap_pairs: Optional ``[N_pairs, 3]`` int tensor produced by
            ``compute_token_atom_symmetry_pairs`` and routed through the
            featurizer (``"atom_swap_pairs"`` feature key).
        asym_id, entity_id: Per-token chain identifiers; both required
            to enable chain permutation alignment.

    Returns:
        Scalar loss.
    """
    if asym_id is not None and entity_id is not None and gt_pos.dim() >= 4:
        from deepfold3.data.symmetry.chain_perm import (
            apply_chain_permutation_tokens,
            compute_chain_permutation_tokens,
        )

        S = gt_pos.shape[0]
        atom_mask_bool = atom_mask.bool()
        # AF3 §4.3: a single chain assignment (from the multi-step
        # ``mini_rollout`` structure) is shared across all S augmented GT
        # samples, because augmentation is SE(3) and preserves chain
        # identity. Fall back to per-sample compute on the (noisy)
        # single-step denoised ``pred_pos`` only when the caller didn't
        # pre-compute an assignment — mainly legacy tests or inference
        # paths without mini-rollout.
        if chain_assign is not None:
            # ``chain_assign`` is shared across S (SE(3) augmentation preserves
            # chain identity), so the permuted mask is identical for every
            # sample. Reorder GT *and its mask* together: without permuting the
            # mask, slots resolved in the pred chain but not its assigned GT
            # chain (differentially-resolved homo-oligomer copies) keep
            # mask=valid over an unresolved (garbage) GT coordinate, so the MSE
            # scores predictions against garbage. Use the permuted mask below.
            permuted_chunks = []
            permuted_mask = atom_mask_bool
            for s in range(S):
                new_gt, permuted_mask = apply_chain_permutation_tokens(
                    gt_pos[s], atom_mask_bool, asym_id, chain_assign, sym_token_index=sym_token_index
                )
                permuted_chunks.append(new_gt)
            atom_mask = permuted_mask
            atom_mask_bool = permuted_mask
        else:
            # Fallback (no mini-rollout assignment): per-sample assignment means
            # the permuted mask differs per sample; the [N, M] downstream mask
            # can't represent that, so this legacy/inference path keeps the
            # original mask (pre-existing behavior).
            permuted_chunks = []
            for s in range(S):
                assign_s = compute_chain_permutation_tokens(
                    pred_pos[s],
                    gt_pos[s],
                    atom_mask_bool,
                    asym_id,
                    entity_id,
                    token_to_rep_atom=token_to_rep_atom,
                    mol_type=mol_type,
                    seq_mask=seq_mask,
                    sym_token_index=sym_token_index,
                )
                new_gt, _ = apply_chain_permutation_tokens(
                    gt_pos[s], atom_mask_bool, asym_id, assign_s, sym_token_index=sym_token_index
                )
                permuted_chunks.append(new_gt)
        gt_pos = torch.stack(permuted_chunks, dim=0)

    mask = atom_mask.float()  # [N, M]

    # Build per-atom loss weight: molecule-type upweighting, multiplied by
    # optional region weights. Shared by weighted_rigid_align and the final MSE
    # reduction so the alignment target and loss target agree on which atoms
    # matter. AF3 §3.7.1 eq. 4 is ADDITIVE over a base weight of 1:
    #   w_l = 1 + f_l^dna·α_dna + f_l^rna·α_rna + f_l^ligand·α_ligand
    # with α_dna = α_rna = ``nucleotide_loss_weight`` (5) and α_ligand =
    # ``ligand_loss_weight`` (10) → nucleotide atoms weigh 6, ligand atoms 11
    # (NOT 5/10 — the base-1 must not be overwritten).
    if mol_type is not None:
        token_weight = torch.ones(mask.shape[0], device=mask.device)
        token_weight = token_weight + (mol_type == 1).to(token_weight) * nucleotide_loss_weight  # DNA
        token_weight = token_weight + (mol_type == 2).to(token_weight) * nucleotide_loss_weight  # RNA
        token_weight = token_weight + (mol_type == 3).to(token_weight) * ligand_loss_weight  # ligand
        weight_2d = mask * token_weight.unsqueeze(-1)  # [N, M]
    else:
        weight_2d = mask
    if region_weights is not None:
        weight_2d = weight_2d * region_weights.unsqueeze(-1)  # [N, M]

    # AF3 §3.7.1 eq. 2: rigidly align GT onto pred per sample. This MUST run
    # BEFORE the symmetric-atom resolution below: a swap decision compares
    # |pred_a - gt_a|² vs |pred_a - gt_b|², whose sign is (pred_a-pred_b)·
    # (gt_a-gt_b). Across the arbitrary SE(3) augmentation gap that dot product
    # flips freely, so deciding swaps in the unaligned frame is dominated by
    # global misorientation rather than the ~1-2 Å local symmetry. Aligning
    # first puts pred and GT in a common frame so the local comparison is valid.
    # Alignment output is detached — gradients flow through pred only.
    if apply_rigid_align:
        S = pred_pos.shape[0]
        pred_flat = pred_pos.reshape(S, -1, 3)
        gt_flat = gt_pos.reshape(S, -1, 3)
        w_flat = weight_2d.reshape(-1)  # [N*M]
        gt_flat = weighted_rigid_align(gt_flat, pred_flat, w_flat)
        gt_pos = gt_flat.reshape_as(pred_pos)

    # Symmetric-atom / ligand resolution in the aligned frame (AF3 §4.3). The
    # decision is per diffusion sample against that sample's prediction; the
    # mini-rollout ``atom_perm_pred`` staging is retained only for the shared
    # chain assignment above (where it feeds ``chain_assign``).
    if atom_swap_pairs is not None and len(atom_swap_pairs) > 0:
        from deepfold3.data.symmetry.atom_perms import apply_token_atom_permutation

        gt_pos = apply_token_atom_permutation(pred_pos, gt_pos, atom_mask.bool(), atom_swap_pairs)

    if ligand_swap_pairs is not None and len(ligand_swap_pairs) > 0:
        from deepfold3.data.symmetry.atom_perms import apply_ligand_token_swap

        gt_pos = apply_ligand_token_swap(pred_pos, gt_pos, atom_mask.bool(), ligand_swap_pairs)

    if ligand_perms is not None and len(ligand_perms) > 0:
        from deepfold3.data.symmetry.atom_perms import apply_ligand_token_permutation

        gt_pos = apply_ligand_token_permutation(pred_pos, gt_pos, atom_mask.bool(), ligand_perms)

    weight_mask = weight_2d.unsqueeze(-1)  # [N, M, 1]
    # AF3 §5.3 eq. 3 / OF3 ``mse_loss`` / Protenix ``MSELoss``: divide by
    # *atom count* (not sum-of-weights) so the absolute scale tracks the
    # number of supervised atoms; the mol-type and region weights only
    # reweight the numerator. The leading 1/3 averages over the three
    # spatial coordinates.
    n_atoms = mask.sum().clamp(min=1)

    # Per-sample EDM weight [S]
    w = (noise_levels**2 + sigma_data**2) / (noise_levels * sigma_data) ** 2

    # Vectorized MSE: [S, N, M, 3] → [S]
    mse_per_sample = ((pred_pos - gt_pos) ** 2 * weight_mask.unsqueeze(0)).sum(dim=(-3, -2, -1)) / (3.0 * n_atoms)
    return (w * mse_per_sample).mean()


# ---------------------------------------------------------------------------
# Distogram loss
# ---------------------------------------------------------------------------


def distogram_loss(
    logits: torch.Tensor,  # [N, N, num_bins]
    gt_positions: torch.Tensor,  # [N, 3] representative atoms
    token_mask: torch.Tensor,  # [N]
    first_break: float = 2.3125,
    last_break: float = 21.6875,
    num_bins: int = 64,
) -> torch.Tensor:
    """Cross-entropy distogram loss."""
    dist = torch.cdist(gt_positions, gt_positions)  # [N, N]
    breaks = torch.linspace(first_break, last_break, num_bins - 1, device=dist.device)
    target = torch.bucketize(dist, breaks)  # [N, N] in [0, num_bins-1]

    pair_mask = (token_mask.unsqueeze(1) & token_mask.unsqueeze(0)).float()
    pair_mask.fill_diagonal_(0.0)  # drop trivial i==i self-pairs (distance 0 -> bin 0)
    n_pairs = pair_mask.sum().clamp(min=1)

    # Compute log_softmax in fp32 for numerical stability
    log_probs = F.log_softmax(logits.float(), dim=-1)
    nll = -log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)

    return (nll * pair_mask).sum() / n_pairs


def masked_msa_loss(
    logits: torch.Tensor,  # [S, N, C]
    true_msa: torch.Tensor,  # [S, N] int — original tokens
    bert_mask: torch.Tensor,  # [S, N] float — 1.0 at corrupted entries
    eps: float = 1e-8,
) -> torch.Tensor:
    """BERT-style masked-MSA cross-entropy.

    Per-entry cross-entropy between the predicted logits and the original MSA
    tokens, masked-averaged over the corrupted (``bert_mask`` = 1) entries.

    Uses ``F.cross_entropy`` (fused fp32 log-softmax + NLL, ``reduction="none"``)
    rather than an explicit ``log_softmax``+``gather`` — the fused kernel avoids
    materialising the full ``[S, N, C]`` log-prob tensor (≈50 MB fp32 at
    S=1024) and has a fused backward. The per-(row, token) ``bert_mask`` can't
    be expressed via CE's ``weight``/``ignore_index``, so the masked mean is
    applied here. When no entries are masked the loss is 0 (zero numerator;
    the ``eps`` floor guards the denominator).
    """
    C = logits.shape[-1]
    nll = F.cross_entropy(
        logits.float().reshape(-1, C),
        true_msa.long().reshape(-1),
        reduction="none",
    ).reshape(true_msa.shape)  # [S, N]
    mask = bert_mask.float()
    return (nll * mask).sum() / mask.sum().clamp_min(eps)


@torch.no_grad()
def distogram_entropy(
    logits: torch.Tensor,  # [N, N, num_bins]
    token_mask: torch.Tensor,  # [N]
) -> torch.Tensor:
    """Predictive entropy of the distogram softmax (logged-only metric).

    Returns ``H(p_θ) = -Σ_b p_b log p_b`` averaged over (i, j) pairs in the
    token mask. Pair masking and num-pairs normalisation mirror
    :func:`distogram_loss` so ``CE - entropy = KL(p_θ ‖ p_data)`` is exact.
    """
    pair_mask = (token_mask.unsqueeze(1) & token_mask.unsqueeze(0)).float()
    pair_mask.fill_diagonal_(0.0)  # drop trivial i==i self-pairs (distance 0 -> bin 0)
    n_pairs = pair_mask.sum().clamp(min=1)
    log_probs = F.log_softmax(logits.float(), dim=-1)
    entropy = -(log_probs.exp() * log_probs).sum(-1)
    return (entropy * pair_mask).sum() / n_pairs


# ---------------------------------------------------------------------------
# Confidence losses
# ---------------------------------------------------------------------------


def plddt_loss(
    logits: torch.Tensor,  # [S, N, M * num_bins]
    pred_pos: torch.Tensor,  # [S, N, M, 3]
    gt_pos: torch.Tensor,  # [N, M, 3]
    atom_mask: torch.Tensor,  # [N, M]
    num_bins: int = 50,
    mol_type: torch.Tensor | None = None,  # [N] int (0=prot, 1=DNA, 2=RNA, 3=lig)
    rep_idx: torch.Tensor | None = None,  # [N] int — rep atom slot in [0, M)
) -> torch.Tensor:
    """Per-atom pLDDT cross-entropy loss.

    AF3 §4.3.1 / OF3 ``all_atom_plddt_loss``: when ``mol_type`` and
    ``rep_idx`` are provided, the LDDT target is computed against
    representative atoms only (CA for protein, C1' for nucleic), with
    inclusion radius depending on j's molecule type — 15 Å for
    protein-rep j, 30 Å for nucleic-rep j; ligands contribute no j atoms.
    Without those features (legacy/test code paths), falls back to a flat
    30 Å radius over all atom pairs.

    Vectorized: computes per-atom LDDT for all S samples at once.
    """
    S, N = logits.shape[:2]
    M = atom_mask.shape[-1]
    logits_r = logits.reshape(S, N, M, num_bins)
    flat_mask = atom_mask.float()
    n_atoms = flat_mask.sum().clamp(min=1)

    # Compute per-atom LDDT for all samples at once [S, N, M]
    lddt_all = _compute_per_atom_lddt_batched(pred_pos, gt_pos, atom_mask, mol_type=mol_type, rep_idx=rep_idx)
    target = (lddt_all * num_bins).clamp(0, num_bins - 1).long()  # [S, N, M]

    log_probs = F.log_softmax(logits_r.float(), dim=-1)  # fp32 for stability
    nll = -log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)  # [S, N, M]
    return (nll * flat_mask.unsqueeze(0)).sum() / (n_atoms * S)


def _pairwise_binned_loss(
    logits: torch.Tensor,  # [S, N, N, num_bins]
    error: torch.Tensor,  # [S, N, N]
    pair_mask: torch.Tensor,  # [N, N]
    max_error: float = 31.0,
    num_bins: int = 64,
) -> torch.Tensor:
    """Shared NLL loss for pairwise error predictions (PAE and PDE)."""
    bin_edges = torch.linspace(0, max_error, num_bins - 1, device=logits.device)
    n_pairs = pair_mask.sum().clamp(min=1)

    target = torch.bucketize(error, bin_edges).clamp(max=num_bins - 1)
    log_probs = F.log_softmax(logits.float(), dim=-1)  # fp32 for stability
    nll = -log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    return (nll * pair_mask.unsqueeze(0)).sum() / (n_pairs * logits.shape[0])


def express_coordinates_in_frame(
    rep_pos: torch.Tensor,  # [..., N, 3] — representative atom per token
    frame_pos: torch.Tensor,  # [..., N, 3, 3] — 3 frame atoms (N/CA/C or C1'/C3'/C4')
    eps: float = 1e-8,
) -> torch.Tensor:
    """``expressCoordinatesInFrame`` from AF3 supplement Algorithm 29 (§4.3.2).

    For every ordered token pair ``(i, j)`` returns the coordinates of token
    ``j``'s representative atom in token ``i``'s local rigid frame, built from
    the three frame atoms ``(a, b, c)`` of token ``i`` (proteins: (N, CA, C);
    nucleic acids: (C1', C3', C4'); ``b`` is the origin).

    Args:
        rep_pos: Representative atom positions ``[..., N, 3]``.
        frame_pos: Frame atom positions ``[..., N, 3, 3]`` where the middle
            dim indexes the three frame atoms ``(a, b, c)``.
        eps: Stabilization constant for vector normalization.

    Returns:
        Local-frame coordinates ``[..., N, N, 3]`` where entry ``[i, j]`` is
        ``R_i^T @ (rep[j] - b_i)``.

    Gram-Schmidt and the frame-transform matmul run in fp32 regardless of
    ambient autocast precision — matches AF3's ``Precision.HIGHEST`` pattern
    for geometric transforms (references/alphafold3 rotation_matrix.py:87,93).
    Orthonormality of the rigid frame is numerically critical for PAE error
    and degrades visibly under bf16.
    """
    out_dtype = rep_pos.dtype
    rep_pos = rep_pos.float()
    frame_pos = frame_pos.float()

    a = frame_pos[..., 0, :]
    b = frame_pos[..., 1, :]
    c = frame_pos[..., 2, :]

    w1 = F.normalize(a - b, dim=-1, eps=eps)
    w2 = F.normalize(c - b, dim=-1, eps=eps)
    e1 = F.normalize(w1 + w2, dim=-1, eps=eps)
    e2 = F.normalize(w2 - w1, dim=-1, eps=eps)
    e3 = torch.cross(e1, e2, dim=-1)

    # Stack basis as rows → R_i^T. Broadcast over the j-axis so each
    # (i, j) pair multiplies rel[i, j] by its own R_i^T via a single
    # batched matmul (faster than einsum on cuBLAS/rocBLAS).
    r_transposed = torch.stack([e1, e2, e3], dim=-2)  # [..., N, 3, 3]
    rel = rep_pos.unsqueeze(-3) - b.unsqueeze(-2)  # [..., N, N, 3]
    # matmul: [..., N, 1, 3, 3] @ [..., N, N, 3, 1] → [..., N, N, 3, 1]
    result = torch.matmul(r_transposed.unsqueeze(-3), rel.unsqueeze(-1)).squeeze(-1)
    return result.to(out_dtype)


def _frame_angle_mask(
    frames: torch.Tensor,  # [..., N, 3, 3] — (a, b, c) atom positions
    angle_threshold: float = 25.0,  # degrees
    eps: float = 1e-8,
) -> torch.Tensor:
    """Per-token frame validity from co-linearity threshold.

    AF3 §4.3.2 / OF3 ``get_token_frame_atoms``: a frame is invalid when
    ∠(a, b, c) is within ``angle_threshold`` of 0° or 180°. Returns a
    bool mask of shape ``[..., N]`` that is ``True`` when the frame is
    well-conditioned (angle ∈ ``(threshold, 180 − threshold)``).
    """
    a = frames[..., 0, :]
    b = frames[..., 1, :]
    c = frames[..., 2, :]
    u = a - b
    v = c - b
    cos_angle = (u * v).sum(dim=-1) / (u.norm(dim=-1).clamp(min=eps) * v.norm(dim=-1).clamp(min=eps))
    cos_max = math.cos(math.radians(angle_threshold))
    cos_min = math.cos(math.radians(180.0 - angle_threshold))
    return (cos_angle < cos_max) & (cos_angle > cos_min)


def _tm_score_from_error(
    error: torch.Tensor,  # [S, N, N] per-pair distance (Å)
    mask: torch.Tensor,  # [S, N, N] float
    n_eff: int,
) -> torch.Tensor:
    """TM-like scalar in [0, 1] from a per-pair distance error tensor.

    Shares the d0 sizing (``1.24·(max(n,19)-15)^{1/3}-1.8``), the
    ``1/(1+(d/d0)²)`` TM kernel, the row-normalised j-sum, and the
    max-over-alignment-frames aggregation with the inference-time pTM
    (:func:`deepfold3.io.confidences.compute_ptm`), so its output is on the
    **same scale** as pTM and is directly comparable.

    It differs from pTM in *what the kernel is applied to*. pTM takes the
    **expectation of the kernel over the predicted PAE distribution**,
    ``Σ_b softmax(pae)_b · 1/(1+(center_b/d0)²)``
    (:func:`~deepfold3.io.confidences._compute_tm_adjusted_pae`), whereas this
    function applies the kernel to a **single per-pair distance**,
    ``1/(1+(d/d0)²)``. With ``error`` = the *true* frame-aligned error this
    yields ``tm_true`` — the deterministic ground-truth TM that pTM estimates;
    there is no distribution to integrate, so applying the kernel directly is
    exact.

    Caveat: feeding the PAE-derived **expected** distance ``E[d]`` here (a
    "tm_pred" shortcut) gives ``f(E[d])``, which is **not** equal to pTM's
    ``E[f(d)]`` — the kernel is non-linear, so the two differ by a Jensen gap.
    Match pTM's per-bin expectation if you need to reproduce the inference
    metric exactly.

    Args:
        error: per-pair distance, ``[S, N, N]``. Sample dim S is averaged.
        mask: per-pair valid mask in ``[0, 1]``. Padding / co-linear
              frames / cross-chain restriction all enter via this mask.
        n_eff: token count sizing ``d0`` (``1.24·(max(n,19)-15)^{1/3}-1.8``).
               A scalar → one **global** ``d0`` shared across all pairs; callers
               pass the total valid-token count ``|D|``.

               NOTE — ``itm_true`` vs inference ipTM: ``tm_true`` uses this
               global ``|D|``, matching AF3's pTM (global d0). The cross-chain
               ``itm_true`` diagnostic ALSO passes the global ``|D|`` here, but
               AF3's inference ipTM (:func:`deepfold3.io.confidences.compute_iptm`,
               which follows the AF3 reference code) sizes its interface d0
               **per pair** on ``|chain_i| + |chain_j|`` (AF3's own code diverges
               from its paper Eq. 18 / OpenFold-3, both of which use a global
               d0). So ``itm_true`` is a **global-d0 approximation** of ipTM, not
               byte-equivalent — they coincide only for two-chain complexes.
               Intentional: the per-pair interface d0 is deliberately not
               threaded into this diagnostic.
    """
    clipped = max(int(n_eff), 19)
    d0 = 1.24 * (clipped - 15) ** (1.0 / 3) - 1.8
    tm_per_pair = 1.0 / (1.0 + (error / d0) ** 2)  # [S, N, N]
    denom = mask.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    normed_mask = mask / denom
    per_alignment = (tm_per_pair * normed_mask).sum(dim=-1)  # [S, N]
    # max over i picks the best alignment frame; mean over S averages
    # across diffusion samples (S=1 for mini-rollout, S=num_samples for
    # the confidence-head PAE path).
    return per_alignment.max(dim=-1).values.mean()


_FAPE_ATOM_SET_SLOTS: dict[str, tuple[int, ...]] = {
    # Featurizer K=5 layout: 0=N, 1=CA, 2=C, 3=CB (rep), 4=O. Option
    # names spell out the protein atom set; non-protein tokens take the
    # corresponding frame-atom slots (e.g. slot 1 = C3' for nucleic).
    "CA": (1,),  # K=1 — frame centre only
    "NCAC+CB": (0, 1, 2, 3),  # K=4 — historical S3 (CB, no O)
    "NCACO": (0, 1, 2, 4),  # K=4 — AF2 §4.2.7 backbone FAPE
    "NCACO+CB": (0, 1, 2, 3, 4),  # K=5 — AF2 backbone + CB (current default)
}


def _apply_fape_atom_set_mask(
    mask: torch.Tensor,  # [N, K] bool
    atom_set: str,
) -> torch.Tensor:
    """Zero per-slot mask entries outside the named atom subset.

    Featurizer always emits K=5 slots; this restricts which contribute
    per ``cfg.fape_atom_set`` so the same checkpoint can run any option
    without re-featurising. See :class:`~deepfold3.model.config.TrainingConfig`.
    """
    try:
        keep = _FAPE_ATOM_SET_SLOTS[atom_set]
    except KeyError as exc:
        raise ValueError(f"Unknown fape_atom_set={atom_set!r}; expected one of {sorted(_FAPE_ATOM_SET_SLOTS)}") from exc
    K = mask.shape[-1]
    if len(keep) == K and all(k < K for k in keep):
        return mask
    keep_mask = torch.zeros(K, dtype=torch.bool, device=mask.device)
    for k in keep:
        if 0 <= k < K:
            keep_mask[k] = True
    return mask & keep_mask


def fape_loss(
    pred_pos: torch.Tensor,  # [S, N, M, 3]
    gt_pos: torch.Tensor,  # [N, M, 3]
    fape_token_index: torch.Tensor,  # [N, K] — token id per fape-atom slot
    fape_atom_index: torch.Tensor,  # [N, K] — atom slot within that token
    fape_atom_mask: torch.Tensor,  # [N, K] bool — per-slot atom presence
    frame_atom_index: torch.Tensor,  # [N, 3] — (a, b, c) frame atom slots
    frame_token_index: torch.Tensor,  # [N, 3] — token id per frame atom
    frames_mask: torch.Tensor,  # [N] bool (unused; pair_mask captures it)
    seq_mask: torch.Tensor,  # [N] bool (kept for signature parity)
    pair_mask: torch.Tensor,  # [N, N] bool — frame_i × token_j
    clip_distance: float = 10.0,
    normalize: bool = True,
    angle_threshold: float = 25.0,
    eps: float = 1e-4,
    atom_mask: torch.Tensor | None = None,  # [N, M] bool — required for all_atom_points
    all_atom_points: bool = False,  # AF2 §4.2.7 full: project every atom slot per token
    chunk_size: int = 0,  # chunk the frame (i) axis; 0 = all N frames at once
) -> torch.Tensor:
    """Frame-Aligned Point Error (AF2 §4.2.7), K-atoms-per-token form.

    For every (frame_i, token_j, fape_atom_k) triple projects atom
    ``(fape_token_index[j,k], fape_atom_index[j,k])`` into token ``i``'s
    local rigid frame, in both prediction and ground truth. Computes the
    L2 distance in frame coordinates, clamps it at ``clip_distance``,
    optionally divides by ``clip_distance`` for a dimensionless
    ``[0, 1]`` value, then averages per-sample over valid entries.

    K=5 in current featurizer: slots 0..2 are the frame's own (a, b, c)
    atoms (i.e. (N, CA, C) for protein, (C1', C3', C4') for nucleic),
    slot 3 is the rep atom (CB for protein, C1' for nucleic — when
    the rep duplicates a frame slot, the featurizer masks slot 3 out
    via ``fape_atom_mask`` to avoid double-counting), and slot 4 is
    the backbone carbonyl O (protein only, AF2 §4.2.7 backbone
    FAPE-equivalent). Slot 4 supervises the C=O direction (i.e. ψ
    basin commit), which is the defining geometry for β-strand
    cross-strand H-bond ladders and was the dominant missing signal
    in the K=4 layout.

    Single-rep predecessor (commit 96bc1ba, 2026-05-20) projected only
    the rep atom per token. For helix mirror images the chirality
    gradient there was diluted: under z→−z, local (x, y) coordinates
    are preserved and only local_z flips, so per-pair error scales
    with ``2·|gt_local_z|``. With rep=CB at local_z ≈ ±1.20 Å the
    signal exists but is only ~2.4 Å per pair (clipped). Stacking the
    frame's own (N, CA, C) atoms adds three more probe points per
    residue with much larger off-axis components, giving 3–4× more
    chirality signal per (frame_i, token_j) — comparable to AF2's
    backbone-FAPE.

    Frame normals (Gram-Schmidt from ``(a, b, c)``) encode chirality —
    a mirrored prediction gives left-handed frames and the per-pair
    error blows up regardless of pairwise distances. This is precisely
    the chirality signal AF2 trains with and AF3 dropped when it
    switched to diffusion MSE + smooth_lddt; the present run
    ``df3-chiral`` 2025-05-19→20 confirmed that with diffusion-MSE
    only (or even + Cβ chirality + backbone Cα-dihedral), the
    backbone helix remains mirrored. FAPE restores AF2's gradient.

    Args:
        pred_pos: ``[S, N, M, 3]`` predicted atom positions.
        gt_pos: ``[N, M, 3]`` ground-truth positions.
        fape_token_index: ``[N, K]`` token id of each fape-atom slot
            (typically the token itself for polymer; may cross for
            ligand atoms reused as frame atoms).
        fape_atom_index: ``[N, K]`` atom slot within the referenced
            token's dense ``[N, M]`` layout.
        fape_atom_mask: ``[N, K]`` bool — per-slot atom presence /
            de-duplication mask.
        frame_atom_index: ``[N, 3]`` (a, b, c) atom slots for the frame.
        frame_token_index: ``[N, 3]`` token id of each frame atom.
        frames_mask: ``[N]`` bool — frame validity (kept for API
            consistency; ``pair_mask`` already encodes frame validity).
        seq_mask: ``[N]`` bool — token validity (kept for API parity).
        pair_mask: ``[N, N]`` bool — frame_i × token_j pair mask
            (typically ``frames_mask[i] & seq_mask[j]``).
        clip_distance: outlier clamp in Å. Defaults to ``10.0``.
        normalize: divide by ``clip_distance`` so the loss is in
            ``[0, 1]`` regardless of clamp magnitude. Defaults ``True``.
        angle_threshold: drop frames within this many degrees of
            colinearity (AF3 §4.3.2 / OF3 ``pae_loss``). Defaults ``25``.
        eps: numerical stabiliser inside the norm (smooth ``sqrt``).
        atom_mask: ``[N, M]`` presence mask — used as the point set when
            ``all_atom_points=True`` (required in that mode).
        all_atom_points: AF2 §4.2.7 full FAPE. Project every atom slot of
            each token (masked by ``atom_mask``) into the frames instead of
            the K featurizer slots, so symmetric terminal atoms (OD/NH/ring)
            get a local-frame target and cannot collapse onto their pair
            midpoint. ``fape_token_index`` / ``fape_atom_index`` /
            ``fape_atom_mask`` are ignored in this mode.
        chunk_size: split the frame (i) axis into chunks of this size to
            bound peak memory (0 = all N frames at once). Exact: the
            (i, j, atom) reduction is a plain sum, so chunks accumulate.

    Returns:
        Scalar FAPE in pred dtype. Dimensionless when ``normalize=True``.
    """
    out_dtype = pred_pos.dtype
    if pair_mask.sum().item() == 0:
        return pred_pos.new_zeros(())

    with torch.amp.autocast(device_type=pred_pos.device.type, enabled=False):
        pred32 = pred_pos.float()
        gt32 = gt_pos.float()
        S, N = pred32.shape[0], pred32.shape[1]

        # Frame atoms (shared between Gram-Schmidt + colinearity filter)
        pred_frames = pred32[:, frame_token_index, frame_atom_index]  # [S, N, 3, 3]
        gt_frames = gt32[frame_token_index, frame_atom_index]  # [N, 3, 3]

        def _build_frame_basis(frames: torch.Tensor):
            # frames: [..., N, 3, 3] → (R_t, b) with R_t = [e1; e2; e3] rows.
            # Matches ``express_coordinates_in_frame`` exactly (inlined to
            # share the basis across all K projected atoms per token).
            a = frames[..., 0, :]
            b = frames[..., 1, :]
            c = frames[..., 2, :]
            w1 = F.normalize(a - b, dim=-1, eps=1e-8)
            w2 = F.normalize(c - b, dim=-1, eps=1e-8)
            e1 = F.normalize(w1 + w2, dim=-1, eps=1e-8)
            e2 = F.normalize(w2 - w1, dim=-1, eps=1e-8)
            e3 = torch.cross(e1, e2, dim=-1)
            R_t = torch.stack([e1, e2, e3], dim=-2)  # [..., N, 3, 3]
            return R_t, b

        pred_R_t, pred_b = _build_frame_basis(pred_frames)  # [S, N, 3, 3], [S, N, 3]
        gt_R_t, gt_b = _build_frame_basis(gt_frames)  # [N, 3, 3],    [N, 3]

        # Point set projected into every frame. K-atom mode gathers the
        # featurizer's [N, P=K] slots; all-atom mode (AF2 §4.2.7 full) uses
        # every atom slot of each token (masked by ``atom_mask``) so that
        # symmetric terminal atoms — absent from the K sets — get a
        # local-frame target and cannot collapse onto their pair midpoint.
        if all_atom_points:
            if atom_mask is None:
                raise ValueError("fape_loss(all_atom_points=True) requires atom_mask [N, M]")
            pred_atoms = pred32  # [S, N_j, P=M, 3]
            gt_atoms = gt32  # [N_j, P=M, 3]
            point_mask = (atom_mask > 0).float()  # [N_j, P]
        else:
            pred_atoms = pred32[:, fape_token_index, fape_atom_index]  # [S, N_j, P=K, 3]
            gt_atoms = gt32[fape_token_index, fape_atom_index]  # [N_j, P=K, 3]
            point_mask = fape_atom_mask.float()  # [N_j, P]

        # Frame-angle gate on the i-axis (frame-providing token), pred & GT.
        pred_angle_ok = _frame_angle_mask(pred_frames, angle_threshold).float()  # [S, N]
        gt_angle_ok = _frame_angle_mask(gt_frames, angle_threshold).float()  # [N]
        pmf = pair_mask.float()  # [N_i, N_j]

        # Chunk over the frame axis i to bound the [S, C, N_j, P, 3] peak
        # (all-atom P=M=24 is ~5× the K=5 path). The (i, j, atom) reduction is
        # a plain sum, so chunk sums accumulate to the exact unchunked result.
        cs = chunk_size if chunk_size and chunk_size > 0 else N
        err_per_sample = pred32.new_zeros(S)
        n_per_sample = pred32.new_zeros(S)
        for i0 in range(0, N, cs):
            i1 = min(i0 + cs, N)
            # rel[s, i, j, p, :] = atoms[s, j, p, :] - b[s, i, :]
            pred_rel = pred_atoms.unsqueeze(1) - pred_b[:, i0:i1].unsqueeze(2).unsqueeze(3)  # [S, C, N_j, P, 3]
            gt_rel = gt_atoms.unsqueeze(0) - gt_b[i0:i1].unsqueeze(1).unsqueeze(2)  # [C, N_j, P, 3]
            pred_local = torch.matmul(
                pred_R_t[:, i0:i1].unsqueeze(2).unsqueeze(2),  # [S, C, 1, 1, 3, 3]
                pred_rel.unsqueeze(-1),  # [S, C, N_j, P, 3, 1]
            ).squeeze(-1)  # [S, C, N_j, P, 3]
            gt_local = torch.matmul(
                gt_R_t[i0:i1].unsqueeze(1).unsqueeze(1),  # [C, 1, 1, 3, 3]
                gt_rel.unsqueeze(-1),  # [C, N_j, P, 3, 1]
            ).squeeze(-1)  # [C, N_j, P, 3]

            # Smooth norm: sqrt(d²+ε) keeps the backward bounded at d→0.
            error = (((pred_local - gt_local.unsqueeze(0)) ** 2).sum(-1) + eps).sqrt()  # [S, C, N_j, P]
            clamped = error.clamp(max=clip_distance)
            if normalize:
                clamped = clamped / clip_distance

            # Mask: pair_mask × frame-angle (i-axis) × per-point presence.
            mask_pair = (
                pmf[i0:i1].unsqueeze(0)  # [1, C, N_j]
                * pred_angle_ok[:, i0:i1].unsqueeze(-1)  # [S, C, 1]
                * gt_angle_ok[i0:i1].unsqueeze(0).unsqueeze(-1)  # [1, C, 1]
            )  # [S, C, N_j]
            mask = mask_pair.unsqueeze(-1) * point_mask.unsqueeze(0).unsqueeze(0)  # [S, C, N_j, P]

            err_per_sample = err_per_sample + (clamped * mask).sum(dim=(-1, -2, -3))  # [S]
            n_per_sample = n_per_sample + mask.sum(dim=(-1, -2, -3))  # [S]

        # Per-sample mean → mean across samples (matches ``pae_loss`` / OF3
        # convention; samples with sparse valid frames are not over-weighted).
        loss = (err_per_sample / n_per_sample.clamp(min=1)).mean()

    return loss.to(out_dtype)


def frame_aligned_error(
    pred_rep: torch.Tensor,  # [S, N, 3] representative atoms
    pred_frames: torch.Tensor,  # [S, N, 3, 3] frame atoms in pred space
    gt_rep: torch.Tensor,  # [N, 3] representative atoms
    gt_frames: torch.Tensor,  # [N, 3, 3] frame atoms in GT space
    angle_threshold: float = 25.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-pair frame-local alignment error + per-frame angle-ok masks.

    Shared by :func:`pae_loss` (CE target) and the ``tm_true`` diagnostic — both
    project token ``j`` into token ``i``'s local frame for pred and GT and take
    the L2 distance, then drop frames within ``angle_threshold`` of co-linearity.
    Computing it once lets FT3 reuse the result instead of running the O(N²)
    ``express_coordinates_in_frame`` twice.

    Returns ``(error [S, N, N], pred_angle_ok [S, N] bool, gt_angle_ok [N] bool)``.
    """
    pred_local = express_coordinates_in_frame(pred_rep, pred_frames)  # [S, N, N, 3]
    gt_local = express_coordinates_in_frame(gt_rep, gt_frames)  # [N, N, 3]
    error = (pred_local - gt_local.unsqueeze(0)).norm(dim=-1)  # [S, N, N]
    pred_angle_ok = _frame_angle_mask(pred_frames, angle_threshold)  # [S, N]
    gt_angle_ok = _frame_angle_mask(gt_frames, angle_threshold)  # [N]
    return error, pred_angle_ok, gt_angle_ok


def pae_loss(
    logits: torch.Tensor,  # [S, N, N, num_bins]
    pred_rep: torch.Tensor,  # [S, N, 3] representative atoms
    pred_frames: torch.Tensor,  # [S, N, 3, 3] frame atoms in pred space
    gt_rep: torch.Tensor,  # [N, 3] representative atoms
    gt_frames: torch.Tensor,  # [N, 3, 3] frame atoms in GT space
    pae_mask: torch.Tensor,  # [N, N] float — frames_mask[i] & pair_mask[i, j]
    num_bins: int = 64,
    max_error_bin: float = 31.0,
    angle_threshold: float = 25.0,
    frame_error: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Frame-aligned predicted aligned error (PAE) cross-entropy loss.

    AF3 §4.3.2. For every ordered pair ``(i, j)``, projects token
    ``j``'s representative atom into token ``i``'s local rigid frame for
    both prediction and ground truth, takes the L2 distance, bucketises into
    ``num_bins`` bins over ``[0, max_error_bin]`` (last bin is catch-all),
    and computes cross-entropy against the predicted bin logits.

    Frames whose ∠(a, b, c) lies within ``angle_threshold`` of co-linearity
    are dropped per-sample (separately on pred and GT), matching OF3
    ``pae_loss``'s ``angle_threshold=25`` filter — co-linear frames have
    ill-defined orientation and would otherwise inject noise into the
    alignment-error supervision.
    """
    if frame_error is None:
        error, pred_angle_ok, gt_angle_ok = frame_aligned_error(
            pred_rep, pred_frames, gt_rep, gt_frames, angle_threshold
        )
    else:
        error, pred_angle_ok, gt_angle_ok = frame_error

    bin_edges = torch.linspace(0, max_error_bin, num_bins - 1, device=logits.device)
    target = torch.bucketize(error, bin_edges).clamp(max=num_bins - 1)

    pred_angle_ok = pred_angle_ok.float()  # [S, N]
    gt_angle_ok = gt_angle_ok.float()  # [N]
    # Gate is on the i-axis (frame-providing token); j is unrestricted.
    mask_s = pae_mask.unsqueeze(0) * pred_angle_ok.unsqueeze(-1) * gt_angle_ok.unsqueeze(0).unsqueeze(-1)  # [S, N, N]

    log_probs = F.log_softmax(logits.float(), dim=-1)
    nll = -log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)  # [S, N, N]
    # Per-sample mean (OF3 convention: divide by per-sample valid count,
    # then average across samples) so samples with sparse valid frames
    # don't get over-weighted by global normalisation.
    nll_per_sample = (nll * mask_s).sum(dim=(-1, -2))  # [S]
    n_valid_per_sample = mask_s.sum(dim=(-1, -2)).clamp(min=1)  # [S]
    return (nll_per_sample / n_valid_per_sample).mean()


def pde_loss(
    logits: torch.Tensor,  # [S, N, N, num_bins]
    pred_pos: torch.Tensor,  # [S, N, 3] representative atoms
    gt_pos: torch.Tensor,  # [N, 3] representative atoms
    token_mask: torch.Tensor,  # [N]
    max_error: float = 31.0,
    num_bins: int = 64,
) -> torch.Tensor:
    """Predicted distance error cross-entropy loss."""
    pair_mask = (token_mask.unsqueeze(1) & token_mask.unsqueeze(0)).float()
    pair_mask.fill_diagonal_(0.0)  # drop trivial i==i self-pairs (distance 0 -> bin 0)
    gt_dist = torch.cdist(gt_pos, gt_pos)
    pred_dist = torch.cdist(pred_pos, pred_pos)  # [S, N, N]
    error = (pred_dist - gt_dist.unsqueeze(0)).abs()
    return _pairwise_binned_loss(logits, error, pair_mask, max_error, num_bins)


def exp_resolved_loss(
    logits: torch.Tensor,  # [S, N, M * 2]
    atom_mask: torch.Tensor,  # [N, M] bool — ground truth resolved mask
    valid_mask: torch.Tensor,  # [N, M] bool — atom slots that exist (canonical)
    fixed_mask: torch.Tensor | None = None,  # [N] float — token-level resolved fraction
) -> torch.Tensor:
    """Binary cross-entropy for experimentally-resolved prediction.

    Token-level resolved learning. Two cases:

    1. **Regular mmCIF (no ``fixed_mask``)**: target is per-atom ``atom_mask``
       (the GT resolved bool, 1 = atom resolved in mmCIF, 0 = missing).

    2. **Inpainted data (``fixed_mask`` provided, e.g. PATCHR ATLAS)**: every
       canonical slot is filled with positions, so per-atom ``atom_mask`` is
       degenerate (≈ ``valid_mask``). Instead, we use the token-level
       *experimental fraction* ``fixed_mask`` ∈ [0, 1] as the soft target,
       broadcast across all atoms in the token's canonical slots. The model
       learns to flag whole tokens as “experimental” vs “inpainted”.

    ``valid_mask`` is the supervision domain (canonical heavy-atom slots),
    used as the reduction denominator so padded slots are excluded from the
    mean entirely.
    """
    if fixed_mask is not None:
        # Token-level target broadcast to atom level. atom_mask is ignored
        # because inpainting fills it to ≈ valid_mask, losing the original
        # missing-residue signal.
        target = fixed_mask.unsqueeze(-1).expand_as(atom_mask).float()  # [N, M]
    else:
        target = atom_mask.float()

    M = atom_mask.shape[-1]
    logits_r = logits.reshape(*logits.shape[:2], M, 2)  # [S, N, M, 2]
    log_probs = F.log_softmax(logits_r.float(), dim=-1)  # [S, N, M, 2]

    p = target.unsqueeze(0).expand(logits_r.shape[:-1])  # [S, N, M]
    nll = -(p * log_probs[..., 1] + (1 - p) * log_probs[..., 0])  # [S, N, M]

    # Masked mean over valid (canonical) atom slots only.
    weight = valid_mask.float().unsqueeze(0).expand_as(nll)  # [S, N, M]
    return (nll * weight).sum() / weight.sum().clamp_min(1.0)


# ---------------------------------------------------------------------------
# Bond violation loss
# ---------------------------------------------------------------------------


def bond_loss(
    pred_pos: torch.Tensor,  # [S, N, M, 3]
    bond_atom_indices: torch.Tensor,  # [B_bonds, 4] — (token_1, atom_1, token_2, atom_2)
    gt_distances: torch.Tensor,  # [B_bonds] observed GT distances (AF3 Eq. 5)
    bond_mask: torch.Tensor,  # [B_bonds] bool
    sigma_data: float = 16.0,
    noise_levels: torch.Tensor | None = None,  # [S]
) -> torch.Tensor:
    """MSE loss on bonded atom pair distances vs ground-truth distances.

    AF3 Eq. 5: L_bond = mean_{(l,m)∈B} (||x_l - x_m|| - ||x_l^GT - x_m^GT||)²

    Vectorized over S diffusion samples.
    """
    if bond_atom_indices.shape[0] == 0:
        return pred_pos.new_zeros(())

    t1 = bond_atom_indices[:, 0].long()
    a1 = bond_atom_indices[:, 1].long()
    t2 = bond_atom_indices[:, 2].long()
    a2 = bond_atom_indices[:, 3].long()
    mask = bond_mask.float()
    n_bonds = mask.sum().clamp(min=1)

    # Vectorized gather: [S, B_bonds, 3]
    pos1 = pred_pos[:, t1, a1]
    pos2 = pred_pos[:, t2, a2]
    # eps inside the sqrt: bounded backward at coincident atoms (AF3 §5.3 Eq. 5).
    pred_dist = torch.sqrt(((pos1 - pos2) ** 2).sum(dim=-1) + 1e-6)  # [S, B_bonds]

    mse_per_sample = ((pred_dist - gt_distances.unsqueeze(0)) ** 2 * mask.unsqueeze(0)).sum(dim=-1) / n_bonds

    if noise_levels is not None:
        w = (noise_levels**2 + sigma_data**2) / (noise_levels * sigma_data) ** 2
        mse_per_sample = w * mse_per_sample

    return mse_per_sample.mean()


def _augment_bond_indices_with_symmetric_pairs(
    features: dict[str, torch.Tensor],
    gt_pos: torch.Tensor,  # [N, M, 3] resolved GT
    atom_mask: torch.Tensor,  # [N, M]
    bond_atom_indices: torch.Tensor | None,  # [B, 4] or None
    bond_gt_distances: torch.Tensor | None,  # [B] or None
    bond_loss_mask: torch.Tensor | None,  # [B] or None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Append the symmetry-swappable atom pairs to the bond-loss inputs.

    The pairs the per-residue symmetry renaming can swap (ASP OD1/OD2, PHE/TYR
    ring, ARG NH1/NH2, gem-dimethyls, nucleic OP1/OP2, atomized/ligand 2-cycles)
    collapse onto their pair midpoint because the renaming makes that ~loss-free
    under diffusion MSE. Feeding them into :func:`bond_loss` — whose target
    ``||gt_a - gt_b||`` is invariant to the a<->b labeling — adds a coin-flip-free,
    per-pair-normalised (un-diluted) separation penalty. Returns
    ``(indices [B', 4], gt_distances [B'], mask [B'])`` with the symmetric rows
    appended (originals unchanged when no symmetry pairs exist).
    """
    dev = gt_pos.device
    rows: list[torch.Tensor] = []
    # atom_swap_pairs [P, 3] = (token, M_a, M_b): intra-token protein / nucleic.
    asp = features.get("atom_swap_pairs")
    if asp is not None and len(asp) > 0:
        asp = asp.to(dev).long()
        asp = asp[asp[:, 1] != asp[:, 2]]  # drop collate self-pad rows
        if len(asp):
            rows.append(torch.stack([asp[:, 0], asp[:, 1], asp[:, 0], asp[:, 2]], dim=1))
    # ligand_swap_pairs [P, 2|3] = (token_a, token_b[, group]): inter-token, slot 0.
    lsp = features.get("ligand_swap_pairs")
    if lsp is not None and len(lsp) > 0:
        lsp = lsp.to(dev).long()
        lsp = lsp[lsp[:, 0] != lsp[:, 1]]
        if len(lsp):
            z = torch.zeros(lsp.shape[0], dtype=torch.long, device=dev)
            rows.append(torch.stack([lsp[:, 0], z, lsp[:, 1], z], dim=1))

    empty_idx = torch.zeros((0, 4), dtype=torch.long, device=dev)
    empty_1d = torch.zeros((0,), device=dev)
    if bond_atom_indices is not None and bond_atom_indices.shape[0] > 0:
        base_idx = bond_atom_indices.to(dev).long()
        base_gt = bond_gt_distances.to(dev)
        base_mask = bond_loss_mask.to(dev).bool()
    else:
        base_idx, base_gt, base_mask = empty_idx, empty_1d, empty_1d.bool()

    if not rows:
        return base_idx, base_gt, base_mask

    sym_idx = torch.cat(rows, dim=0)  # [P, 4]
    t1, a1, t2, a2 = sym_idx[:, 0], sym_idx[:, 1], sym_idx[:, 2], sym_idx[:, 3]
    sym_gt = (gt_pos[t1, a1] - gt_pos[t2, a2]).norm(dim=-1)  # [P] label-invariant
    sym_mask = atom_mask[t1, a1].bool() & atom_mask[t2, a2].bool()  # [P]
    return (
        torch.cat([base_idx, sym_idx], dim=0),
        torch.cat([base_gt, sym_gt], dim=0),
        torch.cat([base_mask, sym_mask], dim=0),
    )


# ---------------------------------------------------------------------------
# Chirality (sign-target) loss — AF3-spec deviation
# ---------------------------------------------------------------------------


def chirality_loss(
    pred_pos: torch.Tensor,  # [S, N, M, 3]
    frame_token_index: torch.Tensor,  # [N, 3] long
    frame_atom_index: torch.Tensor,  # [N, 3] long
    chirality_atom_idx: torch.Tensor,  # [N] long
    chirality_target: torch.Tensor,  # [N] float, in {-1, 0, +1}
) -> torch.Tensor:
    """Per-residue chirality quadratic on the Cβ projection onto the frame normal.

    Quadratic-toward-target form (2026-05-19 redesign):

      ``loss = mean over valid residues of (local_z − target)²``

    where ``local_z = (Cβ − CA) · e3``, ``e3`` is the (N, CA, C) frame
    normal, and ``target`` is the *signed* CCD ideal projection (≈ ±1.20
    Å for natural L/D amino acids, stored per-residue in
    ``features["chirality_target"]``). ``0.0`` masks the residue out.

    Physics rationale — the equilibrium CA → Cβ vector lies at
    ``|local_z| ≈ 1.20 ± 0.03 Å`` off the (N, CA, C) plane for any
    natural tetrahedral CA (measured: 122/122 residues of PDB 11CF).
    Penalising the squared distance to the *signed CCD value* (a) drives
    the correct sign, (b) drives the magnitude to the biological ideal
    rather than to a fictitious ``±1`` (the old hardcoded target had a
    17 % systematic underbias), and (c) keeps a smooth gradient
    everywhere (vs the older ``relu(margin − corr)`` hinge which had a
    flat dead zone above the margin and a sharp kink at it). Net per-residue
    range with ``|target| = 1.20``:

      * Correct (local_z = +1.20, target = +1.20) → 0
      * Cβ on the (N, CA, C) plane (local_z = 0) → 1.44
      * Mirror (local_z = −1.20, target = +1.20) → 5.76
      * Severe mispredict (local_z = −2) → 10.24

    The pre-quadratic baseline measured on af3spec ckpt 8500 ranged
    ~1.0–1.4 per residue (most predictions sat near the (N, CA, C)
    plane with |local_z| ≈ 0.01 Å — the Cβ stereocentre had collapsed
    rather than landed on the wrong side).

    Distance-based losses (smooth_lddt, distogram, bond) are invariant
    to reflection; the post-Kabsch diffusion MSE is the only other
    chirality signal but is dominated by global topology. Runs in fp32
    with autocast disabled — the cross-product orthogonality is
    noise-sensitive under bf16.

    Args:
        pred_pos: Per-sample predicted atom positions ``[S, N, M, 3]``.
        frame_token_index: Token index of each frame atom ``[N, 3]``
            (== self for polymers; see :func:`Featurizer.process`).
        frame_atom_index: Atom slot within the referenced token ``[N, 3]``.
        chirality_atom_idx: Probe atom slot per token ``[N]``. Value is
            ignored when ``chirality_target == 0``.
        chirality_target: Pre-computed sign ``[N]`` (``+1`` natural L-aa,
            ``-1`` D-aa, ``0`` mask out — Gly, missing, ambiguous).

    Returns:
        Scalar loss in the input dtype. Zero when no valid residues.
    """
    out_dtype = pred_pos.dtype
    target = chirality_target.float()
    valid_n = (target.abs() > 0).float()  # [N]
    if valid_n.sum().item() == 0:
        return pred_pos.new_zeros(())

    with torch.amp.autocast(device_type=pred_pos.device.type, enabled=False):
        pred32 = pred_pos.float()
        N = pred32.shape[1]
        arange_n = torch.arange(N, device=pred32.device)

        # Gather frame atoms via advanced indexing: result is [S, N, 3, 3]
        # (last two dims = (a, b, c) atom × xyz).
        frame_atoms = pred32[:, frame_token_index, frame_atom_index]  # [S, N, 3, 3]
        a = frame_atoms[..., 0, :]
        b = frame_atoms[..., 1, :]  # CA (or sugar C3' for nucleic acids)
        c = frame_atoms[..., 2, :]

        # Probe atom — Cβ (protein) or N1/N9 (nucleic acid, future).
        probe = pred32[:, arange_n, chirality_atom_idx]  # [S, N, 3]

        # Gram-Schmidt → e3 normal (matches express_coordinates_in_frame).
        w1 = F.normalize(a - b, dim=-1, eps=1e-8)
        w2 = F.normalize(c - b, dim=-1, eps=1e-8)
        e1 = F.normalize(w1 + w2, dim=-1, eps=1e-8)
        e2 = F.normalize(w2 - w1, dim=-1, eps=1e-8)
        e3 = torch.cross(e1, e2, dim=-1)

        local_z = ((probe - b) * e3).sum(-1)  # [S, N]
        # Quadratic-toward-target: penalises sign AND magnitude.
        loss_per = (local_z - target.unsqueeze(0)) ** 2  # [S, N]

        mask = valid_n.unsqueeze(0).expand_as(loss_per)  # [S, N]
        n_valid = mask.sum().clamp(min=1)
        loss = (loss_per * mask).sum() / n_valid

    return loss.to(out_dtype)


def _ca_pseudo_dihedral(
    p1: torch.Tensor,
    p2: torch.Tensor,
    p3: torch.Tensor,
    p4: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Dihedral angle around the (p2, p3) axis using atan2.

    Returns radians in ``[-π, π]``. Inputs broadcast like
    ``[..., 3]`` and the angle is reduced over the last dim. Numerically
    stable for the (rare) case ``|b2| ≈ 0``; otherwise this is the
    textbook signed dihedral formula.
    """
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3
    b2n = b2 / torch.linalg.norm(b2, dim=-1, keepdim=True).clamp(min=eps)
    n1 = torch.cross(b1, b2, dim=-1)
    n2 = torch.cross(b2, b3, dim=-1)
    m1 = torch.cross(n1, b2n, dim=-1)
    x = (n1 * n2).sum(-1)
    y = (m1 * n2).sum(-1)
    return torch.atan2(y, x)


def backbone_chirality_loss(
    pred_pos: torch.Tensor,  # [S, N, M, 3]
    gt_pos: torch.Tensor,  # [N, M, 3]
    atom_mask: torch.Tensor,  # [N, M] bool — gt atoms present
    seq_mask: torch.Tensor,  # [N] bool
    asym_id: torch.Tensor,  # [N] long
    mol_type: torch.Tensor,  # [N] int
    ca_slot: int = 1,
) -> torch.Tensor:
    """Backbone-trace chirality via Cα pseudo-dihedral sin matching.

    The Cβ-based :func:`chirality_loss` only fixes the *local*
    stereocentre — empirically the per-residue mirror_rate dropped from
    38 % to 5 % over 2,000 steps on ``df3-chiral-25e5-c10-g35`` but the
    *backbone trace* stayed mirror (``det(V·U^T) = -1.0`` for every
    inference checkpoint 8600-10500; Cα-pseudo-dihedrals at residues
    100-108 unchanged). The two chiralities are physically linked
    (steric clash in the wrong combination) but the link is energetic,
    not gradient-mediated under our loss menu.

    This term closes that gap: for every four consecutive same-chain
    protein Cα atoms it matches ``sin(pred_pseudo_dihedral)`` against
    ``sin(gt_pseudo_dihedral).detach()`` quadratically. ``sin`` weights
    α-helical residues most heavily (|sin(±50°)| ≈ 0.77) and naturally
    fades to ~0 for β-strands (``sin(±180°) ≈ 0``) and loops
    (``sin(near 0) ≈ 0``) — so the gradient targets exactly the
    handedness-bearing segments without over-constraining ambiguous
    backbone regions.

    Args:
        pred_pos: Predicted atom positions ``[S, N, M, 3]``.
        gt_pos: Ground-truth atom positions ``[N, M, 3]``.
        atom_mask: GT atom presence ``[N, M]`` bool.
        seq_mask: Per-token valid mask ``[N]``.
        asym_id: Per-token chain id ``[N]``; all four Cα of a
            quadruplet must share it.
        mol_type: ``[N]`` int; only ``MOL_PROTEIN`` tokens contribute.
        ca_slot: Atom slot index of CA within each token's dense atom
            layout. ``1`` for protein (``REF_ATOMS["ALA"] = [N, CA, C, O, CB]``).

    Returns:
        Scalar quadratic-mismatch loss in the input dtype. Zero when no
        valid 4-Cα quadruplet exists.
    """
    out_dtype = pred_pos.dtype
    N = pred_pos.shape[1]
    if N < 4:
        return pred_pos.new_zeros(())

    is_prot = mol_type == 0  # MOL_PROTEIN = 0
    # ``atom_mask`` may have been scaled by ``loss_weight_mask`` and is no
    # longer a strict bool; treat any nonzero weight as "atom present".
    ca_ok = atom_mask[:, ca_slot] > 0
    valid = seq_mask & is_prot & ca_ok
    # Same-chain quadruplet mask (all 4 share asym_id).
    quad_chain_ok = (asym_id[:-3] == asym_id[1:-2]) & (asym_id[1:-2] == asym_id[2:-1]) & (asym_id[2:-1] == asym_id[3:])
    quad_mask = valid[:-3] & valid[1:-2] & valid[2:-1] & valid[3:] & quad_chain_ok  # [N-3]
    if not bool(quad_mask.any().item()):
        return pred_pos.new_zeros(())

    with torch.amp.autocast(device_type=pred_pos.device.type, enabled=False):
        pred_ca = pred_pos.float()[:, :, ca_slot, :]  # [S, N, 3]
        gt_ca = gt_pos.float()[:, ca_slot, :]  # [N, 3]

        p1, p2, p3, p4 = pred_ca[:, :-3], pred_ca[:, 1:-2], pred_ca[:, 2:-1], pred_ca[:, 3:]
        g1, g2, g3, g4 = gt_ca[:-3], gt_ca[1:-2], gt_ca[2:-1], gt_ca[3:]

        pred_dih = _ca_pseudo_dihedral(p1, p2, p3, p4)  # [S, N-3]
        with torch.no_grad():
            gt_dih = _ca_pseudo_dihedral(g1, g2, g3, g4)  # [N-3]

        sin_pred = torch.sin(pred_dih)
        sin_gt = torch.sin(gt_dih).unsqueeze(0)  # [1, N-3]

        loss_per = (sin_pred - sin_gt) ** 2  # [S, N-3]
        mask = quad_mask.float().unsqueeze(0).expand_as(loss_per)
        n_valid = mask.sum().clamp(min=1)
        loss = (loss_per * mask).sum() / n_valid

    return loss.to(out_dtype)


@torch.no_grad()
def chirality_violation_rates(
    pred_pos: torch.Tensor,
    frame_token_index: torch.Tensor,
    frame_atom_index: torch.Tensor,
    chirality_atom_idx: torch.Tensor,
    chirality_target: torch.Tensor,
    margin: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Logged-only diagnostics for :func:`chirality_loss`.

    Returns:
        Dict with three scalars (sample- and residue-averaged over the
        validity mask):

        * ``chirality_mirror_rate`` — fraction of residues with
          ``correctness < 0`` (sign-flipped Cβ, i.e. definite mirror).
        * ``chirality_ambiguous_rate`` — fraction with ``correctness < margin``
          (mirror OR Cβ too close to the frame plane).
        * ``chirality_correctness_mean`` — mean signed projection
          ``local_z * target_sign`` in Å. Approaches the CCD ideal ~+1.0
          as chirality is learned.
    """
    target = chirality_target.float()
    valid_n = (target.abs() > 0).float()
    if valid_n.sum().item() == 0:
        z = pred_pos.new_zeros(())
        return {
            "chirality_mirror_rate": z,
            "chirality_ambiguous_rate": z,
            "chirality_correctness_mean": z,
        }

    with torch.amp.autocast(device_type=pred_pos.device.type, enabled=False):
        pred32 = pred_pos.float()
        N = pred32.shape[1]
        arange_n = torch.arange(N, device=pred32.device)
        frame_atoms = pred32[:, frame_token_index, frame_atom_index]
        a = frame_atoms[..., 0, :]
        b = frame_atoms[..., 1, :]
        c = frame_atoms[..., 2, :]
        probe = pred32[:, arange_n, chirality_atom_idx]

        w1 = F.normalize(a - b, dim=-1, eps=1e-8)
        w2 = F.normalize(c - b, dim=-1, eps=1e-8)
        e1 = F.normalize(w1 + w2, dim=-1, eps=1e-8)
        e2 = F.normalize(w2 - w1, dim=-1, eps=1e-8)
        e3 = torch.cross(e1, e2, dim=-1)

        local_z = ((probe - b) * e3).sum(-1)
        correctness = local_z * target.unsqueeze(0)

        mask = valid_n.unsqueeze(0).expand_as(correctness)
        n_valid = mask.sum().clamp(min=1)

        mirror = ((correctness < 0).float() * mask).sum() / n_valid
        ambiguous = ((correctness < margin).float() * mask).sum() / n_valid
        mean_corr = (correctness * mask).sum() / n_valid

    return {
        "chirality_mirror_rate": mirror,
        "chirality_ambiguous_rate": ambiguous,
        "chirality_correctness_mean": mean_corr,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_per_atom_lddt_batched(
    pred_pos: torch.Tensor,  # [S, N, M, 3]
    gt_pos: torch.Tensor,  # [N, M, 3]
    atom_mask: torch.Tensor,  # [N, M]
    cutoffs: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0),
    inclusion_radius: float = 30.0,
    mol_type: torch.Tensor | None = None,  # [N] int (0=prot, 1=DNA, 2=RNA, 3=lig)
    rep_idx: torch.Tensor | None = None,  # [N] int — rep atom slot in [0, M)
    inclusion_radius_protein: float = 15.0,
    inclusion_radius_nucleotide: float = 30.0,
) -> torch.Tensor:
    """Compute per-atom LDDT score in [0, 1], batched over S.

    With ``mol_type`` and ``rep_idx``, applies AF3 §4.3.1 conventions:
    j-axis restricted to representative atoms of standard polymer tokens
    (CA for protein, C1' for nucleic); inclusion radius is 15 Å for
    protein-rep j and 30 Å for nucleic-rep j; ligand tokens contribute
    no j atoms. Without those features, falls back to a flat
    ``inclusion_radius`` over all pairs.

    Returns [S, N, M] LDDT scores.
    """
    S = pred_pos.shape[0]
    N, M = atom_mask.shape
    result = torch.zeros(S, N, M, device=pred_pos.device, dtype=pred_pos.dtype)

    i_mask = atom_mask.reshape(-1).bool()  # [N*M]
    if i_mask.sum() < 1:
        return result

    if mol_type is None or rep_idx is None:
        # Legacy / fallback: flat radius, all-vs-all over resolved atoms.
        gt_flat = gt_pos.reshape(-1, 3)[i_mask]  # [A, 3]
        gt_dist = torch.cdist(gt_flat, gt_flat)  # [A, A]
        pair_mask = (gt_dist < inclusion_radius) & (gt_dist > 1e-3)
        n_pairs = pair_mask.sum(dim=-1).clamp(min=1)  # [A]

        pred_flat = pred_pos.reshape(S, -1, 3)[:, i_mask]  # [S, A, 3]
        pred_dist = torch.cdist(pred_flat, pred_flat)  # [S, A, A]
        dist_diff = (pred_dist - gt_dist.unsqueeze(0)).abs()

        scores = pred_pos.new_zeros(S, gt_flat.shape[0])
        for c in cutoffs:
            scores = scores + ((dist_diff < c).float() * pair_mask).sum(dim=-1) / n_pairs
        scores = scores / len(cutoffs)

        full = torch.zeros(S, atom_mask.numel(), device=pred_pos.device, dtype=pred_pos.dtype)
        full[:, i_mask] = scores
        return full.reshape(S, N, M)

    # OF3-style: per-pair-type radius, j restricted to rep atoms of
    # standard polymer tokens (protein/nucleic; ligands skipped).
    is_prot_tok = mol_type == 0  # MOL_PROTEIN
    is_nuc_tok = (mol_type == 1) | (mol_type == 2)  # MOL_DNA + MOL_RNA
    rep_clamped = rep_idx.clamp(min=0, max=M - 1)
    arange_M = torch.arange(M, device=atom_mask.device)
    is_rep_atom = (
        (arange_M.unsqueeze(0) == rep_clamped.unsqueeze(-1))
        & (rep_idx.unsqueeze(-1) >= 0)
        & (rep_idx.unsqueeze(-1) < M)
    )  # [N, M]

    is_prot_atom_j = is_prot_tok.unsqueeze(-1) & is_rep_atom & atom_mask.bool()  # [N, M]
    is_nuc_atom_j = is_nuc_tok.unsqueeze(-1) & is_rep_atom & atom_mask.bool()  # [N, M]
    j_mask = (is_prot_atom_j | is_nuc_atom_j).reshape(-1)  # [N*M]

    if j_mask.sum() < 1:
        return result

    gt_flat = gt_pos.reshape(-1, 3)
    gt_flat_i = gt_flat[i_mask]  # [Ai, 3]
    gt_flat_j = gt_flat[j_mask]  # [Aj, 3]
    gt_dist = torch.cdist(gt_flat_i, gt_flat_j)  # [Ai, Aj]

    is_prot_j_flat = is_prot_atom_j.reshape(-1)[j_mask]  # [Aj] bool
    radius_j = torch.where(
        is_prot_j_flat,
        gt_dist.new_full((), inclusion_radius_protein),
        gt_dist.new_full((), inclusion_radius_nucleotide),
    )  # [Aj]

    pair_mask = (gt_dist < radius_j.unsqueeze(0)) & (gt_dist > 1e-3)  # [Ai, Aj]
    n_pairs = pair_mask.sum(dim=-1).clamp(min=1)  # [Ai]

    pred_flat = pred_pos.reshape(S, -1, 3)
    pred_flat_i = pred_flat[:, i_mask]  # [S, Ai, 3]
    pred_flat_j = pred_flat[:, j_mask]  # [S, Aj, 3]
    pred_dist = torch.cdist(pred_flat_i, pred_flat_j)  # [S, Ai, Aj]
    dist_diff = (pred_dist - gt_dist.unsqueeze(0)).abs()  # [S, Ai, Aj]

    scores = pred_pos.new_zeros(S, gt_flat_i.shape[0])
    for c in cutoffs:
        scores = scores + ((dist_diff < c).float() * pair_mask).sum(dim=-1) / n_pairs
    scores = scores / len(cutoffs)

    full = torch.zeros(S, atom_mask.numel(), device=pred_pos.device, dtype=pred_pos.dtype)
    full[:, i_mask] = scores
    return full.reshape(S, N, M)


# Backward compat alias
def _compute_per_atom_lddt(
    pred_pos: torch.Tensor,
    gt_pos: torch.Tensor,
    atom_mask: torch.Tensor,
    cutoffs: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0),
    inclusion_radius: float = 30.0,
) -> torch.Tensor:
    """Compute per-atom LDDT score [N, M] (single sample)."""
    return _compute_per_atom_lddt_batched(pred_pos.unsqueeze(0), gt_pos, atom_mask, cutoffs, inclusion_radius).squeeze(
        0
    )


# ---------------------------------------------------------------------------
# Region → per-token weight mapping
# ---------------------------------------------------------------------------


def _region_to_weight(
    region_label: torch.Tensor,  # [N] int
    region_loss_weights: dict[int, float],
) -> torch.Tensor:
    """Convert region labels to per-token weight multipliers.

    Tokens with region_id 0 (unset) or not in the mapping get weight 1.0.
    """
    weights = torch.ones_like(region_label, dtype=torch.float32)
    for region_id, weight in region_loss_weights.items():
        weights[region_label == int(region_id)] = weight
    return weights


# ---------------------------------------------------------------------------
# Combined training loss
# ---------------------------------------------------------------------------


def diffusion_only_loss(
    pred: dict[str, torch.Tensor],
    features: dict[str, torch.Tensor],
    cfg: TrainingConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute just the diffusion-side losses for one (possibly chunked) batch.

    Used by ``DeepFold3.forward_training`` when ``diffusion_chunk_size > 0``:
    the diffusion loss is computed and back-propped per chunk so the chunk's
    activation graph can be freed before the next chunk's forward.

    Args:
        pred: Must contain ``denoised``, ``gt_augmented``, ``noise_levels``.
        features: Input features (need ``atom_gt_positions``, ``atom_gt_mask``,
            optionally ``mol_type``, swap pairs, bond features, mask channels).
        cfg: Training config.

    Returns:
        ``(total, components)`` where ``total`` is the scalar diffusion loss
        (already weighted by ``cfg.alpha_diffusion`` and bond weight) and
        ``components`` is a dict of unweighted scalars for logging.
    """
    gt_pos = features["atom_gt_positions"]
    atom_mask = features["atom_gt_mask"]
    seq_mask = features["seq_mask"].bool()
    gt_aug = pred["gt_augmented"]

    # AF3 §4.3: resolve the chain permutation ONCE (from the mini-rollout) and
    # apply it to the shared GT — the augmented GT the diffusion MSE scores AND
    # the un-augmented GT that smooth-lDDT / FAPE / backbone-chirality score.
    # Previously only the diffusion MSE saw the permutation, leaving the other
    # losses on un-permuted GT. Doing it here (single application point) means
    # the diffusion head's own chain perm is skipped below (``asym_id`` omitted).
    chain_assign = chain_perm_from_mini_rollout(
        pred, features, stability_gate_k=(cfg.chain_perm_gate_k if cfg.chain_perm_stability_gate else 1)
    )
    asym_id_feat = features.get("asym_id")
    if chain_assign and asym_id_feat is not None and not all(int(k) == int(v) for k, v in chain_assign.items()):
        from deepfold3.data.symmetry.chain_perm import chain_permutation_index

        c_index, c_keep = chain_permutation_index(
            asym_id_feat, chain_assign, features.get("sym_token_index"), device=gt_pos.device
        )
        gt_pos = gt_pos.index_select(0, c_index)
        atom_mask = atom_mask.index_select(0, c_index) & c_keep.unsqueeze(-1)
        seq_mask = seq_mask & c_keep
        # gt_aug is pre-permuted in forward_training when permute_gt_before_noise
        # is on (noise added to the permuted GT); re-permuting it here would double.
        if not features.get("__gt_prepermuted__"):
            gt_aug = gt_aug.index_select(-3, c_index)

    # AF3 §4.2: resolve intra-residue / ligand atom symmetry ONCE from the
    # mini-rollout, applied to the shared (un-augmented) GT that smooth-lDDT /
    # FAPE score. The same resolution goes onto the MSE's augmented GT below, so
    # every loss uses one consistent labelling.
    mini_ref = pred.get("mini_rollout_pos")
    if mini_ref is not None:
        gt_pos = resolve_gt_atom_symmetry(mini_ref[0], features, gt_pos, atom_mask)

    region_label = features.get(f"mask_{MASK_REGION}")
    region_weights: torch.Tensor | None = None
    if region_label is not None and cfg.region_loss_weights:
        region_weights = _region_to_weight(region_label, cfg.region_loss_weights)

    # Per-token loss-weight mask (1.0 keep, 0.0 drop). Set per-dataset via
    # ``DatasetConfig.exclude_loss_mol_types`` (e.g. PATCHR ligands carry 0.0
    # here). Equivalent to multiplying the per-atom ``atom_mask`` by the
    # broadcast token-level mask, which lets the existing loss functions
    # use it transparently — no signature changes needed.
    loss_weight_mask = features.get(f"mask_{MASK_LOSS_WEIGHT}")
    if loss_weight_mask is not None and bool((loss_weight_mask < 1.0).any()) is False:
        loss_weight_mask = None
    if loss_weight_mask is not None:
        atom_mask = atom_mask * loss_weight_mask.unsqueeze(-1)

    denoised = pred["denoised"]
    noise_levels = pred["noise_levels"]

    components: dict[str, torch.Tensor] = {}
    total = gt_pos.new_zeros(())

    if cfg.alpha_diffusion > 0:
        # Atom/ligand symmetry (AF3 §4.2). Default: resolve the augmented GT once
        # from the mini-rollout (same labelling as the shared GT above) and pass
        # no swap tables to the MSE. Opt-in: resolve per noise sample inside the
        # MSE (argmin vs each denoised sample). GT is already chain-permuted, so
        # the MSE skips its own chain perm (no ``asym_id``).
        if cfg.diffusion_mse_per_sample_symmetry and not features.get("__gt_prepermuted__"):
            mse_kwargs = dict(
                atom_swap_pairs=features.get("atom_swap_pairs"),
                ligand_swap_pairs=features.get("ligand_swap_pairs"),
                ligand_perms=features.get("ligand_perms"),
            )
        else:
            # Skip when pre-permuted in forward_training (gt_aug already atom-resolved).
            if mini_ref is not None and not features.get("__gt_prepermuted__"):
                gt_aug = resolve_gt_atom_symmetry(
                    mini_ref[0].unsqueeze(0).expand_as(gt_aug), features, gt_aug, atom_mask
                )
            mse_kwargs = {}
        l_mse = diffusion_mse_loss(
            denoised,
            gt_aug,
            atom_mask,
            noise_levels,
            mol_type=features.get("mol_type"),
            nucleotide_loss_weight=cfg.nucleotide_loss_weight,
            ligand_loss_weight=cfg.ligand_loss_weight,
            region_weights=region_weights,
            apply_rigid_align=cfg.use_rigid_align,
            **mse_kwargs,
        )
        components["diffusion_mse"] = l_mse.detach()

        if cfg.alpha_smooth_lddt is not None:
            l_lddt = smooth_lddt_loss(
                denoised,
                gt_pos,
                atom_mask,
                cutoffs=cfg.lddt_cutoffs,
                inclusion_radius=cfg.lddt_inclusion_radius,
                nucleotide_inclusion_radius=cfg.lddt_nucleotide_inclusion_radius,
                min_distance=cfg.lddt_min_distance,
                mol_type=features.get("mol_type"),
            )
            components["smooth_lddt"] = l_lddt.detach()
            total = total + cfg.alpha_diffusion * (l_mse + cfg.alpha_smooth_lddt * l_lddt)
        else:
            total = total + cfg.alpha_diffusion * l_mse

        if cfg.alpha_bond > 0 and ("bond_atom_indices" in features or cfg.bond_include_symmetric):
            b_idx = features.get("bond_atom_indices")
            b_gt = features.get("bond_gt_distances")
            b_mask = features.get("bond_loss_mask")
            if cfg.bond_include_symmetric:
                b_idx, b_gt, b_mask = _augment_bond_indices_with_symmetric_pairs(
                    features, gt_pos, atom_mask, b_idx, b_gt, b_mask
                )
            l_bond = bond_loss(denoised, b_idx, b_gt, b_mask, noise_levels=noise_levels)
            components["bond"] = l_bond.detach()
            total = total + cfg.alpha_diffusion * cfg.alpha_bond * l_bond

        if cfg.alpha_chirality > 0 and "chirality_target" in features:
            l_chir = chirality_loss(
                denoised,
                features["frame_token_index"],
                features["frame_atom_index"],
                features["chirality_atom_idx"],
                features["chirality_target"],
            )
            components["chirality"] = l_chir.detach()
            total = total + cfg.alpha_chirality * l_chir
            for k, v in chirality_violation_rates(
                denoised,
                features["frame_token_index"],
                features["frame_atom_index"],
                features["chirality_atom_idx"],
                features["chirality_target"],
                margin=cfg.chirality_margin,
            ).items():
                components[k] = v.detach()

        if cfg.alpha_backbone_chirality > 0 and "mol_type" in features:
            l_bbchir = backbone_chirality_loss(
                denoised,
                gt_pos,
                atom_mask,
                seq_mask,
                features["asym_id"],
                features["mol_type"],
            )
            components["backbone_chirality"] = l_bbchir.detach()
            total = total + cfg.alpha_backbone_chirality * l_bbchir

        if (
            cfg.alpha_fape > 0
            and "fape_atom_index" in features
            and "fape_token_index" in features
            and "fape_atom_mask" in features
            and "frame_token_index" in features
            and "frame_atom_index" in features
        ):
            seq_m = seq_mask
            frm_m = features.get("frames_mask")
            frm_m = frm_m.bool() if frm_m is not None else seq_m
            pair_m = features.get("pair_mask")
            if pair_m is None:
                pair_m = frm_m.unsqueeze(-1) & seq_m.unsqueeze(-2)
            else:
                pair_m = frm_m.unsqueeze(-1) & pair_m.bool()
            if cfg.fape_atom_set.upper() == "ALL":
                # AF2 §4.2.7 full: every atom slot is a point (the K-slot
                # ``fape_*`` point indices are ignored); ``atom_mask`` gives
                # presence. Frames still come from ``frame_*``.
                l_fape = fape_loss(
                    denoised,
                    gt_pos,
                    None,
                    None,
                    None,
                    features["frame_atom_index"],
                    features["frame_token_index"],
                    frm_m,
                    seq_m,
                    pair_m,
                    clip_distance=cfg.fape_clip_distance,
                    normalize=cfg.fape_normalize,
                    atom_mask=atom_mask,
                    all_atom_points=True,
                    chunk_size=cfg.fape_chunk_size,
                )
            else:
                fape_mask_local = _apply_fape_atom_set_mask(features["fape_atom_mask"].bool(), cfg.fape_atom_set)
                l_fape = fape_loss(
                    denoised,
                    gt_pos,
                    features["fape_token_index"],
                    features["fape_atom_index"],
                    fape_mask_local,
                    features["frame_atom_index"],
                    features["frame_token_index"],
                    frm_m,
                    seq_m,
                    pair_m,
                    clip_distance=cfg.fape_clip_distance,
                    normalize=cfg.fape_normalize,
                    chunk_size=cfg.fape_chunk_size,
                )
            components["fape"] = l_fape.detach()
            total = total + cfg.alpha_fape * l_fape

    return total, components


def training_loss(
    pred: dict[str, torch.Tensor],
    features: dict[str, torch.Tensor],
    cfg: TrainingConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the full training loss.

    Args:
        pred: Model predictions dict.
        features: Input features.
        cfg: Training config with loss weights and parameters.

    Returns:
        Tuple of ``(total_loss, loss_dict)``.
    """
    gt_pos = features["atom_gt_positions"]  # [N, M, 3]
    atom_mask = features["atom_gt_mask"]  # [N, M]
    seq_mask = features["seq_mask"].bool()  # [N] — pred-layout validity (kept raw for the reg)
    rep_idx = features.get("token_to_rep_atom")
    # pLDDT's neighbour set m∈R is the token *center* atom (CA/C1'), NOT the
    # distogram rep atom (CB/C4/C2) — AF3 §4.3.1. Fall back to rep_idx for
    # legacy features that predate token_to_center_atom.
    center_idx = features.get("token_to_center_atom")
    if center_idx is None:
        center_idx = rep_idx

    # AF3 §4.3: resolve the chain permutation ONCE (from the mini-rollout) and
    # score EVERY GT-based loss (smooth-lDDT, distogram, FAPE, pLDDT/PAE/PDE,
    # tm_true) against the permuted GT — previously only the diffusion MSE saw
    # it. Reorders the shared gt_pos / atom_mask so each pred chain lines up
    # with its assigned GT chain; ``gt_seq_mask`` additionally drops pred tokens
    # with no GT counterpart. ``seq_mask`` stays raw for the pred-side
    # representation regularisers below. ``c_index`` (if any) also permutes the
    # augmented GT in the non-chunked diffusion recompute path.
    chain_assign = chain_perm_from_mini_rollout(
        pred, features, stability_gate_k=(cfg.chain_perm_gate_k if cfg.chain_perm_stability_gate else 1)
    )
    asym_id_feat = features.get("asym_id")
    c_index: torch.Tensor | None = None
    gt_seq_mask = seq_mask
    if chain_assign and asym_id_feat is not None and not all(int(k) == int(v) for k, v in chain_assign.items()):
        from deepfold3.data.symmetry.chain_perm import chain_permutation_index

        c_index, c_keep = chain_permutation_index(
            asym_id_feat, chain_assign, features.get("sym_token_index"), device=gt_pos.device
        )
        gt_pos = gt_pos.index_select(0, c_index)
        atom_mask = atom_mask.index_select(0, c_index) & c_keep.unsqueeze(-1)
        gt_seq_mask = seq_mask & c_keep

    # AF3 §4.2: resolve intra-residue / ligand atom symmetry ONCE from the
    # mini-rollout, applied to the shared GT so the per-atom GT losses
    # (smooth-lDDT, pLDDT, exp_resolved) and the ligand-sensitive distogram /
    # tm_true (gt_rep) score against the symmetry-corrected GT. Same labelling
    # as the diffusion MSE (below / in diffusion_only_loss).
    mini_ref = pred.get("mini_rollout_pos")
    if mini_ref is not None:
        gt_pos = resolve_gt_atom_symmetry(mini_ref[0], features, gt_pos, atom_mask)

    losses: dict[str, torch.Tensor] = {}
    # Anchor total to the distogram prediction (pair features or logits)
    # so the grad path stays alive even when chunked diffusion detaches
    # its contribution.  Distogram is always present and graph-connected.
    _pair = pred["distogram_features"] if "distogram_features" in pred else pred["distogram"]
    total = (_pair.view(-1)[0] * 0).float()

    # --- Token mask channels ---
    fixed_mask = features.get(f"mask_{MASK_FIXED}")  # [N] float or None
    # Collation zero-fills mask channels for samples that lack them (mixed
    # datasets). Treat all-zero as absent so the default atom_mask path is used.
    if fixed_mask is not None and not fixed_mask.any():
        fixed_mask = None
    region_label = features.get(f"mask_{MASK_REGION}")  # [N] int or None
    region_weights: torch.Tensor | None = None
    if region_label is not None and cfg.region_loss_weights:
        region_weights = _region_to_weight(region_label, cfg.region_loss_weights)

    # Per-token loss-weight mask (set per-dataset via
    # ``DatasetConfig.exclude_loss_mol_types``; e.g. PATCHR drops ligand tokens).
    # Drops the masked tokens from per-atom / per-pair losses by zeroing out
    # the corresponding entries of ``atom_mask`` and ``seq_mask`` before
    # passing them to the loss functions. ``pair_reg`` / ``single_reg``
    # regularisers keep the originals so excluded tokens' activation
    # magnitudes stay bounded.
    loss_weight_mask = features.get(f"mask_{MASK_LOSS_WEIGHT}")
    if loss_weight_mask is not None and bool((loss_weight_mask < 1.0).any()) is False:
        loss_weight_mask = None
    if loss_weight_mask is not None:
        loss_atom_mask = atom_mask * loss_weight_mask.unsqueeze(-1)
        loss_seq_mask = gt_seq_mask & (loss_weight_mask >= 0.5)
    else:
        loss_atom_mask = atom_mask
        loss_seq_mask = gt_seq_mask

    # --- Single/pair-representation magnitude regularization ---
    # ``alpha_{single,pair}_reg is None`` disables entirely; 0.0 is
    # observation-only (compute under no_grad and log); >0 adds the loss
    # term to the total. ``cfg.reg_form`` selects between two forms:
    #   * "lp"      → loss = α · L_p,        log key = single_l{p} / pair_l{p}
    #   * "log_rms" → loss = α · log(RMS+ε), log key = single_rms / pair_rms
    #                 (the RMS value, not log(RMS), is logged)
    if cfg.alpha_single_reg is not None and "single" in pred:
        if cfg.reg_form == "log_rms":
            if cfg.alpha_single_reg > 0:
                rms, log_rms = single_repr_log_rms_loss(pred["single"], seq_mask)
                total = total + cfg.alpha_single_reg * log_rms
            else:
                with torch.no_grad():
                    rms, _ = single_repr_log_rms_loss(pred["single"], seq_mask)
            losses["single_rms"] = rms
        else:
            if cfg.alpha_single_reg > 0:
                l_sreg = single_repr_reg_loss(pred["single"], seq_mask, cfg.single_reg_p)
                total = total + cfg.alpha_single_reg * l_sreg
            else:
                with torch.no_grad():
                    l_sreg = single_repr_reg_loss(pred["single"], seq_mask, cfg.single_reg_p)
            losses[f"single_l{cfg.single_reg_p}"] = l_sreg

    if cfg.alpha_pair_reg is not None and "pair" in pred:
        pmask = features.get("pair_mask")
        if pmask is None:
            pmask = seq_mask.unsqueeze(-1) & seq_mask.unsqueeze(-2)
        if cfg.reg_form == "log_rms":
            if cfg.alpha_pair_reg > 0:
                rms, log_rms = pair_repr_log_rms_loss(pred["pair"], pmask)
                total = total + cfg.alpha_pair_reg * log_rms
            else:
                with torch.no_grad():
                    rms, _ = pair_repr_log_rms_loss(pred["pair"], pmask)
            losses["pair_rms"] = rms
        else:
            if cfg.alpha_pair_reg > 0:
                l_preg = pair_repr_reg_loss(pred["pair"], pmask, cfg.pair_reg_p)
                total = total + cfg.alpha_pair_reg * l_preg
            else:
                with torch.no_grad():
                    l_preg = pair_repr_reg_loss(pred["pair"], pmask, cfg.pair_reg_p)
            losses[f"pair_l{cfg.pair_reg_p}"] = l_preg

    # --- Diffusion loss ---
    # When ``forward_training`` ran the chunked-diffusion path, it already
    # back-propped each chunk's loss internally and stashed the (detached)
    # weighted total + per-component scalars on ``pred``. Inject them for
    # logging and add the detached weighted total to ``total`` so the
    # displayed loss reflects the full training objective (zero grad).
    if "__diffusion_logged__" in pred:
        losses.update(pred["__diffusion_logged__"])
        total = total + pred["__diffusion_total__"]
    elif "denoised" in pred and cfg.alpha_diffusion > 0:
        denoised = pred["denoised"]  # [S, N, M, 3]
        gt_aug = pred["gt_augmented"]  # [S, N, M, 3]
        noise_levels = pred["noise_levels"]  # [S]

        # GT already chain-permuted above → permute the augmented GT the same way
        # and skip the diffusion MSE's own chain perm (no ``asym_id``). Skip when
        # pre-permuted in forward_training (gt_aug noised from the permuted GT).
        if c_index is not None and not features.get("__gt_prepermuted__"):
            gt_aug = gt_aug.index_select(-3, c_index)

        # Atom/ligand symmetry (AF3 §4.2): default resolves gt_aug once from the
        # mini-rollout (same labelling as the shared GT); opt-in resolves per
        # noise sample inside the MSE.
        if cfg.diffusion_mse_per_sample_symmetry and not features.get("__gt_prepermuted__"):
            mse_kwargs = dict(
                atom_swap_pairs=features.get("atom_swap_pairs"),
                ligand_swap_pairs=features.get("ligand_swap_pairs"),
                ligand_perms=features.get("ligand_perms"),
            )
        else:
            # Skip when pre-permuted in forward_training (gt_aug already atom-resolved).
            if mini_ref is not None and not features.get("__gt_prepermuted__"):
                gt_aug = resolve_gt_atom_symmetry(
                    mini_ref[0].unsqueeze(0).expand_as(gt_aug), features, gt_aug, loss_atom_mask
                )
            mse_kwargs = {}
        l_mse = diffusion_mse_loss(
            denoised,
            gt_aug,
            loss_atom_mask,
            noise_levels,
            mol_type=features.get("mol_type"),
            nucleotide_loss_weight=cfg.nucleotide_loss_weight,
            ligand_loss_weight=cfg.ligand_loss_weight,
            region_weights=region_weights,
            apply_rigid_align=cfg.use_rigid_align,
            **mse_kwargs,
        )
        losses["diffusion_mse"] = l_mse

        if cfg.alpha_smooth_lddt is not None:
            l_lddt = smooth_lddt_loss(
                denoised,
                gt_pos,
                loss_atom_mask,
                cutoffs=cfg.lddt_cutoffs,
                inclusion_radius=cfg.lddt_inclusion_radius,
                nucleotide_inclusion_radius=cfg.lddt_nucleotide_inclusion_radius,
                min_distance=cfg.lddt_min_distance,
                mol_type=features.get("mol_type"),
            )
            losses["smooth_lddt"] = l_lddt
            total = total + cfg.alpha_diffusion * (l_mse + cfg.alpha_smooth_lddt * l_lddt)
        else:
            total = total + cfg.alpha_diffusion * l_mse

        # Bond violation loss (stage 2 fine-tuning)
        if cfg.alpha_bond > 0 and ("bond_atom_indices" in features or cfg.bond_include_symmetric):
            b_idx = features.get("bond_atom_indices")
            b_gt = features.get("bond_gt_distances")
            b_mask = features.get("bond_loss_mask")
            if cfg.bond_include_symmetric:
                b_idx, b_gt, b_mask = _augment_bond_indices_with_symmetric_pairs(
                    features, gt_pos, loss_atom_mask, b_idx, b_gt, b_mask
                )
            l_bond = bond_loss(denoised, b_idx, b_gt, b_mask, noise_levels=noise_levels)
            losses["bond"] = l_bond
            total = total + cfg.alpha_diffusion * cfg.alpha_bond * l_bond

        # Chirality hinge — explicit sign-target supervision on Cβ frame
        # projection. Off by default (alpha_chirality=0); enable to bound
        # the mirror-image saddle that distance-based losses cannot see.
        if cfg.alpha_chirality > 0 and "chirality_target" in features:
            l_chir = chirality_loss(
                denoised,
                features["frame_token_index"],
                features["frame_atom_index"],
                features["chirality_atom_idx"],
                features["chirality_target"],
            )
            losses["chirality"] = l_chir
            total = total + cfg.alpha_chirality * l_chir
            # Diagnostics: logged-only, no grad. ``chirality_margin`` here
            # is purely the threshold for the ``ambiguous_rate`` indicator,
            # not a loss hyperparameter (the loss is now quadratic).
            losses.update(
                chirality_violation_rates(
                    denoised,
                    features["frame_token_index"],
                    features["frame_atom_index"],
                    features["chirality_atom_idx"],
                    features["chirality_target"],
                    margin=cfg.chirality_margin,
                )
            )

        # Backbone-trace chirality: complements ``chirality_loss`` —
        # the local Cβ term cannot move the Cα handedness, see commit
        # log on ``df3-chiral`` 11CF inference (det(V·U^T) = -1 across
        # mirror_rate 38→5 % training).
        if cfg.alpha_backbone_chirality > 0 and "mol_type" in features:
            l_bbchir = backbone_chirality_loss(
                denoised,
                gt_pos,
                loss_atom_mask,
                gt_seq_mask,
                features["asym_id"],
                features["mol_type"],
            )
            losses["backbone_chirality"] = l_bbchir
            total = total + cfg.alpha_backbone_chirality * l_bbchir

        # AF2 §4.2.7 FAPE on the diffusion training output. Replaces the
        # ``chirality_loss`` + ``backbone_chirality_loss`` pair as a
        # single chirality+structure signal. Same Gram-Schmidt frame
        # projection as ``pae_loss`` but with gradient flowing to atoms
        # (PAE detaches via ``bucketize``).
        if (
            cfg.alpha_fape > 0
            and "fape_atom_index" in features
            and "fape_token_index" in features
            and "fape_atom_mask" in features
            and "frame_token_index" in features
            and "frame_atom_index" in features
        ):
            frm_m_feat = features.get("frames_mask")
            frm_m = frm_m_feat.bool() if frm_m_feat is not None else loss_seq_mask
            pair_m_feat = features.get("pair_mask")
            if pair_m_feat is None:
                pair_m = frm_m.unsqueeze(-1) & loss_seq_mask.unsqueeze(-2)
            else:
                pair_m = frm_m.unsqueeze(-1) & pair_m_feat.bool()
            if cfg.fape_atom_set.upper() == "ALL":
                # AF2 §4.2.7 full: every atom slot is a point (the K-slot
                # ``fape_*`` point indices are ignored); ``loss_atom_mask``
                # gives presence. Frames still come from ``frame_*``.
                l_fape = fape_loss(
                    denoised,
                    gt_pos,
                    None,
                    None,
                    None,
                    features["frame_atom_index"],
                    features["frame_token_index"],
                    frm_m,
                    loss_seq_mask,
                    pair_m,
                    clip_distance=cfg.fape_clip_distance,
                    normalize=cfg.fape_normalize,
                    atom_mask=loss_atom_mask,
                    all_atom_points=True,
                    chunk_size=cfg.fape_chunk_size,
                )
            else:
                fape_mask_local = _apply_fape_atom_set_mask(features["fape_atom_mask"].bool(), cfg.fape_atom_set)
                l_fape = fape_loss(
                    denoised,
                    gt_pos,
                    features["fape_token_index"],
                    features["fape_atom_index"],
                    fape_mask_local,
                    features["frame_atom_index"],
                    features["frame_token_index"],
                    frm_m,
                    loss_seq_mask,
                    pair_m,
                    clip_distance=cfg.fape_clip_distance,
                    normalize=cfg.fape_normalize,
                    chunk_size=cfg.fape_chunk_size,
                )
            losses["fape"] = l_fape
            total = total + cfg.alpha_fape * l_fape

    # Representative atom positions (shared by distogram + confidence)
    N = seq_mask.shape[0]
    if rep_idx is not None:
        arange_n = torch.arange(N, device=gt_pos.device)
        gt_rep = gt_pos[arange_n, rep_idx]  # [N, 3]
        # Gate the rep-atom coordinate losses (distogram / PAE / PDE) on GT
        # presence of the representative atom. ``loss_seq_mask`` alone is
        # CCD-derived (``rep_atom_present`` from the reference conformer) so it
        # stays True for a token whose rep atom (CB / C4 / C2 / CA) is unresolved
        # in the GT — which would otherwise inject a (0, 0, 0) rep coordinate into
        # the distance targets. Uses the chain-permuted ``atom_mask`` so it lines
        # up with the permuted ``gt_rep``. AF3 masks these tokens out of the
        # rep-atom losses; this is also the prerequisite for materializing
        # entirely-unresolved residues as tokens (structure-prepare item 4b).
        rep_present = atom_mask[arange_n, rep_idx].bool()  # [N]
    else:
        gt_rep = gt_pos[:, 1, :]  # fallback to CA
        rep_present = torch.ones(N, dtype=torch.bool, device=gt_pos.device)
    rep_seq_mask = loss_seq_mask & rep_present

    # --- Distogram loss (all samples, incl. distillation) ---
    # AF3 §5.2 Table 6: "Train structure and distogram" = True for stages 1–3
    # regardless of sample origin. Only confidence losses are PDB-only (§4.3).
    if cfg.alpha_distogram > 0:
        if "distogram_features" in pred:
            from deepfold3.training.fused_loss import (
                distogram_entropy_fused,
                distogram_loss_fused,
            )

            l_disto = distogram_loss_fused(
                pred["distogram_features"],
                pred["distogram_weight"],
                gt_rep,
                rep_seq_mask,
                chunk_n=cfg.fused_ce_chunk_n,
            )
            e_disto = None
            if cfg.log_distogram_entropy:
                with torch.no_grad():
                    e_disto = distogram_entropy_fused(
                        pred["distogram_features"],
                        pred["distogram_weight"],
                        rep_seq_mask,
                        chunk_n=cfg.fused_ce_chunk_n,
                    )
        elif "distogram" in pred:
            l_disto = distogram_loss(pred["distogram"], gt_rep, rep_seq_mask)
            e_disto = None
            if cfg.log_distogram_entropy:
                with torch.no_grad():
                    e_disto = distogram_entropy(pred["distogram"], rep_seq_mask)
        else:
            l_disto = None
            e_disto = None
        if l_disto is not None:
            losses["distogram"] = l_disto
            total = total + cfg.alpha_distogram * l_disto
        if e_disto is not None:
            losses["distogram_entropy"] = e_disto

    # --- Masked-MSA (BERT) loss (all samples, incl. distillation) ---
    # Gated on the head having run (``masked_msa_logits`` present ⇔
    # ``ModelConfig.masked_msa.enabled`` and training). Added unconditionally
    # at ``alpha_masked_msa`` (× 0 still reaches the head params, keeping DDP
    # ``static_graph`` happy when the weight is 0). Placed above the
    # distillation early-return so distillation MSAs contribute too.
    if "masked_msa_logits" in pred:
        l_mmsa = masked_msa_loss(pred["masked_msa_logits"], pred["true_msa"], pred["bert_mask"])
        losses["masked_msa"] = l_mmsa
        total = total + cfg.alpha_masked_msa * l_mmsa

    # --- TM-true / iTM-true (global accuracy from mini-rollout positions) ---
    # Mirrors the pTM kernel but uses the *true* alignment error instead of
    # the PAE-derived expected distance. Always computable when
    # ``mini_rollout_pos`` and frame features are present — no dependency
    # on the confidence head, so it fires in stage 0 too. Logged-only
    # (``with no_grad``); not added to ``total``. Cross-chain ``itm_true``
    # is emitted only when the batch contains at least one cross-chain pair
    # (DDP-safe via DistributedAverageMeters' per-key counts + key-set
    # union; single-chain ranks simply don't contribute that key).
    # Gated by ``cfg.log_tm_true`` to skip the O(N²) frame expansion in stages
    # 1–3. When ``alpha_pae > 0`` (FT3) it is computed regardless: the PAE loss
    # already builds the same express_coordinates_in_frame error from the same
    # mini-rollout frames, so tm_true is ~free on the margin there.
    mini_pos_for_tm = pred.get("mini_rollout_pos")  # [1, N, M, 3] or None
    frame_idx_feat = features.get("frame_atom_index")
    frame_tok_feat = features.get("frame_token_index")
    frames_mask_feat = features.get("frames_mask")
    # Shared frame-aligned error, computed once here and reused by the PAE loss
    # below (FT3) — see ``frame_aligned_error``. ``None`` when frames are absent
    # or both the tm_true metric and PAE are off.
    shared_frame_error = None
    if (
        (cfg.log_tm_true or cfg.alpha_pae > 0)
        and mini_pos_for_tm is not None
        and frame_idx_feat is not None
        and frames_mask_feat is not None
    ):
        with torch.no_grad():
            arange_n_tm = torch.arange(N, device=gt_pos.device)
            if rep_idx is not None:
                pred_rep_tm = mini_pos_for_tm[:, arange_n_tm, rep_idx]  # [1, N, 3]
            else:
                pred_rep_tm = mini_pos_for_tm[:, :, 1, :]
            if frame_tok_feat is None:
                frame_tok_feat = arange_n_tm.unsqueeze(-1).expand(N, 3)
            gt_frames_atoms_tm = gt_pos[frame_tok_feat, frame_idx_feat]  # [N, 3, 3]
            pred_frames_atoms_tm = mini_pos_for_tm[:, frame_tok_feat, frame_idx_feat]  # [1, N, 3, 3]

            true_error, pred_angle_ok_tm, gt_angle_ok_tm = frame_aligned_error(
                pred_rep_tm, pred_frames_atoms_tm, gt_rep, gt_frames_atoms_tm
            )
            shared_frame_error = (true_error, pred_angle_ok_tm, gt_angle_ok_tm)

            pair_mask_for_tm = features.get("pair_mask")
            if pair_mask_for_tm is None:
                pair_mask_for_tm = loss_seq_mask.unsqueeze(1) & loss_seq_mask.unsqueeze(0)
            # Gate on GT rep-atom presence (consistent with the distogram / PAE /
            # PDE losses) so materialized unresolved residues (item 4b, whose rep
            # atom is (0,0,0)) don't leak into the tm_true / itm_true diagnostic.
            pair_mask_for_tm = pair_mask_for_tm.bool() & rep_present.unsqueeze(0) & rep_present.unsqueeze(1)
            pae_mask_for_tm = (frames_mask_feat.bool().unsqueeze(-1) & pair_mask_for_tm.bool()).float()
            mask_s_tm = (
                pae_mask_for_tm.unsqueeze(0)
                * pred_angle_ok_tm.float().unsqueeze(-1)
                * gt_angle_ok_tm.float().unsqueeze(0).unsqueeze(-1)
            )  # [1, N, N]

            n_eff_tm = int(rep_seq_mask.sum().item())
            if mask_s_tm.sum() > 0 and n_eff_tm > 0:
                losses["tm_true"] = _tm_score_from_error(true_error, mask_s_tm, n_eff_tm)
                asym_id_for_tm = features.get("asym_id")
                if asym_id_for_tm is not None:
                    cc_mask_2d = (asym_id_for_tm.unsqueeze(0) != asym_id_for_tm.unsqueeze(1)).float()
                    cc_mask_s_tm = mask_s_tm * cc_mask_2d.unsqueeze(0)
                    if cc_mask_s_tm.sum() > 0:
                        losses["itm_true"] = _tm_score_from_error(true_error, cc_mask_s_tm, n_eff_tm)

    # --- Distillation: skip confidence losses (AF3 §4.3: PDB-only, res-gated) ---
    is_distillation = features.get("is_distillation")
    if is_distillation is not None and is_distillation.item():
        losses["total"] = total
        return total, losses

    # --- Confidence losses ---
    # Resolution gating uses a tensor multiply (not a Python branch) so the
    # autograd graph topology is identical across iterations. This is required
    # for DDP ``static_graph=True`` with ``find_unused_parameters=False``: the
    # set of parameters reached by autograd must be the same every step, even
    # for samples whose resolution falls outside the confidence-loss range.
    resolution = features.get("resolution")
    if resolution is None:
        conf_valid_mask = torch.ones((), device=gt_pos.device, dtype=gt_pos.dtype)
    else:
        res_t = resolution if torch.is_tensor(resolution) else torch.tensor(resolution, device=gt_pos.device)
        conf_valid_mask = ((res_t >= cfg.min_resolution) & (res_t <= cfg.max_resolution)).to(gt_pos.dtype)

    if cfg.alpha_confidence > 0:
        mini_pos = pred.get("mini_rollout_pos")  # [1, N, M, 3]
        if mini_pos is not None:
            if "plddt" in pred:
                l_plddt = plddt_loss(
                    pred["plddt"],
                    mini_pos,
                    gt_pos,
                    loss_atom_mask,
                    mol_type=features.get("mol_type"),
                    rep_idx=center_idx,
                )
                losses["plddt"] = l_plddt
                total = total + cfg.alpha_confidence * conf_valid_mask * l_plddt

            if rep_idx is not None:
                mini_rep = mini_pos[:, arange_n, rep_idx]  # [S, N, 3]
            else:
                mini_rep = mini_pos[:, :, 1, :]

            if "pde_features" in pred:
                from deepfold3.training.fused_loss import pde_loss_fused

                l_pde = pde_loss_fused(
                    pred["pde_features"],
                    pred["pde_weight"],
                    mini_rep,
                    gt_rep,
                    rep_seq_mask,
                    chunk_n=cfg.fused_ce_chunk_n,
                )
                losses["pde"] = l_pde
                total = total + cfg.alpha_confidence * conf_valid_mask * l_pde
            elif "pde" in pred:
                l_pde = pde_loss(pred["pde"], mini_rep, gt_rep, rep_seq_mask)
                losses["pde"] = l_pde
                total = total + cfg.alpha_confidence * conf_valid_mask * l_pde

            if "exp_resolved" in pred:
                # ``ref_mask`` (=canonical heavy-atom slots) is the supervision
                # domain — padded slots are excluded from the mean. ``atom_mask``
                # (=GT resolved mmCIF mask) is the binary target.
                ref_mask = features.get("ref_mask")
                if ref_mask is None:
                    ref_mask = loss_atom_mask  # fall back to GT mask if ref_mask absent
                else:
                    # Apply ligand-loss exclusion to the supervision domain
                    # (ref_mask) as well, otherwise excluded ligand atoms still
                    # enter the reduction denominator.
                    if loss_weight_mask is not None:
                        ref_mask = ref_mask * loss_weight_mask.unsqueeze(-1)
                l_resolved = exp_resolved_loss(
                    pred["exp_resolved"],
                    loss_atom_mask,
                    valid_mask=ref_mask,
                    fixed_mask=fixed_mask,
                )
                losses["exp_resolved"] = l_resolved
                total = total + cfg.alpha_confidence * conf_valid_mask * l_resolved

            if cfg.alpha_pae > 0:
                frame_idx = features.get("frame_atom_index")  # [N, 3] int — atom slot within referenced token
                frame_tok = features.get("frame_token_index")  # [N, 3] int — which token each frame atom comes from
                frames_mask = features.get("frames_mask")  # [N] bool
                if frame_idx is not None and frames_mask is not None:
                    if frame_tok is None:
                        # Back-compat for caches produced before §4.3.2 ligand frames:
                        # every frame atom came from the token's own slot layout.
                        frame_tok = torch.arange(N, device=gt_pos.device).unsqueeze(-1).expand(N, 3)
                    # When the tm_true block already built the frame-aligned error
                    # (FT3 hits this path), pae_loss reuses it via ``frame_error``
                    # and the raw frame atoms are unused — skip the gathers.
                    if shared_frame_error is None:
                        gt_frames_atoms = gt_pos[frame_tok, frame_idx]  # [N, 3, 3]
                        mini_frames_atoms = mini_pos[:, frame_tok, frame_idx]  # [S, N, 3, 3]
                    else:
                        gt_frames_atoms = mini_frames_atoms = None

                    pair_mask_feat = features.get("pair_mask")
                    if pair_mask_feat is None:
                        pair_mask_feat = loss_seq_mask.unsqueeze(1) & loss_seq_mask.unsqueeze(0)
                    elif loss_weight_mask is not None:
                        # AND-in the per-token exclusion mask so PAE doesn't
                        # supervise pairs involving an excluded token.
                        keep = loss_weight_mask >= 0.5
                        pair_mask_feat = pair_mask_feat.bool() & keep.unsqueeze(1) & keep.unsqueeze(0)
                    # Drop pairs whose frame/target token has an unresolved GT rep
                    # atom — the aligned error uses gt_rep (see rep_seq_mask).
                    pair_mask_feat = pair_mask_feat.bool() & rep_present.unsqueeze(0) & rep_present.unsqueeze(1)
                    pae_mask = (frames_mask.bool().unsqueeze(-1) & pair_mask_feat.bool()).float()

                    if "pae_features" in pred:
                        from deepfold3.training.fused_loss import pae_loss_fused

                        l_pae = pae_loss_fused(
                            pred["pae_features"],
                            pred["pae_weight"],
                            mini_rep,
                            mini_frames_atoms,
                            gt_rep,
                            gt_frames_atoms,
                            pae_mask,
                            chunk_n=cfg.fused_ce_chunk_n,
                            frame_error=shared_frame_error,
                        )
                    elif "pae" in pred:
                        l_pae = pae_loss(
                            pred["pae"],
                            mini_rep,
                            mini_frames_atoms,
                            gt_rep,
                            gt_frames_atoms,
                            pae_mask,
                            frame_error=shared_frame_error,
                        )
                    else:
                        l_pae = None
                    if l_pae is not None:
                        losses["pae"] = l_pae
                        total = total + cfg.alpha_confidence * cfg.alpha_pae * conf_valid_mask * l_pae

    losses["total"] = total
    return total, losses
