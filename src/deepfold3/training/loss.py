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
        # Reflection fix: R = V · diag(1, 1, sign(det(V·U^T))) · U^T
        det = torch.linalg.det(torch.matmul(V, U_T))
        ones = torch.ones_like(det)
        diag = torch.diag_embed(torch.stack([ones, ones, det], dim=-1))
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
        mol_type: Per-token molecule type ``[N]`` (0=prot, 1=RNA, 2=DNA,
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
    gt_dist = torch.cdist(gt_flat, gt_flat)  # [A, A]

    if mol_type is not None:
        # Broadcast per-token mol_type to per-atom (each token holds M atoms).
        N, M = atom_mask.shape
        per_atom_mol_type = mol_type.view(N, 1).expand(N, M).reshape(-1)[flat_mask]  # [A]
        is_nuc = (per_atom_mol_type == 1) | (per_atom_mol_type == 2)
        nuc_pair = is_nuc.unsqueeze(0) & is_nuc.unsqueeze(1)  # [A, A]
        radius = torch.where(
            nuc_pair,
            gt_dist.new_full((), nucleotide_inclusion_radius),
            gt_dist.new_full((), inclusion_radius),
        )
        pair_mask = (gt_dist < radius) & (gt_dist > min_distance)
    else:
        pair_mask = (gt_dist < inclusion_radius) & (gt_dist > min_distance)
    n_pairs = pair_mask.sum().clamp(min=1)

    # Batch over S: [S, A, 3]
    pred_flat = pred_pos.reshape(S, -1, 3)[:, flat_mask]
    pred_dist = _safe_self_cdist(pred_flat)  # [S, A, A]
    dist_diff = (pred_dist - gt_dist.unsqueeze(0)).abs()  # [S, A, A]

    # Vectorize over cutoffs: [C] tensor
    cutoff_t = pred_pos.new_tensor(cutoffs).reshape(-1, 1, 1, 1)  # [C, 1, 1, 1]
    # [C, S, A, A] → sigmoid-scored fractions
    scores = (0.5 * (1.0 + torch.tanh((cutoff_t - dist_diff.unsqueeze(0)) * 5.0)) * pair_mask).sum(dim=(-2, -1))
    # [C, S] → average over cutoffs → [S]
    lddt_per_sample = scores.sum(dim=0) / (n_pairs * len(cutoffs))

    return (1.0 - lddt_per_sample).mean()


# ---------------------------------------------------------------------------
# Diffusion MSE loss with EDM weighting
# ---------------------------------------------------------------------------


def chain_perm_from_mini_rollout(
    pred: dict[str, torch.Tensor],
    features: dict[str, torch.Tensor],
) -> dict[int, int] | None:
    """AF3 §4.3 chain permutation from the mini-rollout structure.

    The mini-rollout is a multi-step denoised prediction (``no_grad``) in
    the same frame as the un-augmented GT, so a single chain assignment
    computed here can be applied to every augmented GT sample consumed
    by the diffusion MSE. Returns ``None`` when the inputs needed for
    permutation aren't present (``mini_rollout_pos``, ``asym_id``,
    ``entity_id``) — callers fall back to the legacy per-sample compute
    on the noisy single-step denoised prediction.
    """
    mini_pos = pred.get("mini_rollout_pos")
    asym_id = features.get("asym_id")
    entity_id = features.get("entity_id")
    gt_pos = features.get("atom_gt_positions")
    atom_mask = features.get("atom_gt_mask")
    if mini_pos is None or asym_id is None or entity_id is None or gt_pos is None or atom_mask is None:
        return None

    from deepfold3.data.symmetry.chain_perm import compute_chain_permutation_tokens

    # mini-rollout is ``[S_mini, N, M, 3]`` — use the first sample.
    return compute_chain_permutation_tokens(mini_pos[0], gt_pos, atom_mask.bool(), asym_id, entity_id)


def diffusion_mse_loss(
    pred_pos: torch.Tensor,  # [S, N, M, 3]
    gt_pos: torch.Tensor,  # [S, N, M, 3] (augmented GT per sample)
    atom_mask: torch.Tensor,  # [N, M]
    noise_levels: torch.Tensor,  # [S]
    sigma_data: float = 16.0,
    mol_type: torch.Tensor | None = None,  # [N] int64 (0=prot, 1=RNA, 2=DNA, 3=ligand)
    nucleotide_loss_weight: float = 5.0,
    ligand_loss_weight: float = 10.0,
    region_weights: torch.Tensor | None = None,  # [N] float — per-token loss weight from region labels
    atom_swap_pairs: torch.Tensor | None = None,  # [N_pairs, 3] (token, M_a, M_b)
    ligand_swap_pairs: torch.Tensor | None = None,  # [N_pairs, 2] (token_a, token_b) ligand swaps
    asym_id: torch.Tensor | None = None,  # [N] int — per-token chain id
    entity_id: torch.Tensor | None = None,  # [N] int — per-token entity id (for chain perm grouping)
    chain_assign: dict[int, int] | None = None,  # pre-computed pred→GT chain mapping
    atom_perm_pred: torch.Tensor | None = None,  # [N, M, 3] — mini-rollout reference for atom/ligand swap decisions
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
            permuted_chunks = [
                apply_chain_permutation_tokens(gt_pos[s], atom_mask_bool, asym_id, chain_assign)[0] for s in range(S)
            ]
        else:
            permuted_chunks = []
            for s in range(S):
                assign_s = compute_chain_permutation_tokens(pred_pos[s], gt_pos[s], atom_mask_bool, asym_id, entity_id)
                new_gt, _ = apply_chain_permutation_tokens(gt_pos[s], atom_mask_bool, asym_id, assign_s)
                permuted_chunks.append(new_gt)
        gt_pos = torch.stack(permuted_chunks, dim=0)

    # Atom / ligand permutation use ``mini_rollout_pos`` (AF3 §4.3:
    # decisions come from the multi-step mini-rollout) when available,
    # falling back to the per-sample single-step ``pred_pos`` only when
    # the caller didn't stage a mini-rollout reference.
    if atom_perm_pred is not None:
        atom_ref = atom_perm_pred.unsqueeze(0).expand_as(pred_pos)
    else:
        atom_ref = pred_pos

    if atom_swap_pairs is not None and len(atom_swap_pairs) > 0:
        from deepfold3.data.symmetry.atom_perms import apply_token_atom_permutation

        gt_pos = apply_token_atom_permutation(atom_ref, gt_pos, atom_mask.bool(), atom_swap_pairs)

    if ligand_swap_pairs is not None and len(ligand_swap_pairs) > 0:
        from deepfold3.data.symmetry.atom_perms import apply_ligand_token_swap

        gt_pos = apply_ligand_token_swap(atom_ref, gt_pos, atom_mask.bool(), ligand_swap_pairs)

    mask = atom_mask.float()  # [N, M]

    # Build per-atom loss weight: molecule-type upweighting (DNA/RNA 5×, ligand 10×)
    # multiplied by optional region weights. Shared by weighted_rigid_align and
    # the final MSE reduction so the alignment target and loss target agree on
    # which atoms matter (AF3 §3.7.1 eq. 4).
    if mol_type is not None:
        token_weight = torch.ones(mask.shape[0], device=mask.device)
        token_weight[mol_type == 1] = nucleotide_loss_weight  # RNA
        token_weight[mol_type == 2] = nucleotide_loss_weight  # DNA
        token_weight[mol_type == 3] = ligand_loss_weight
        weight_2d = mask * token_weight.unsqueeze(-1)  # [N, M]
    else:
        weight_2d = mask
    if region_weights is not None:
        weight_2d = weight_2d * region_weights.unsqueeze(-1)  # [N, M]

    # AF3 §3.7.1 eq. 2: rigidly align GT onto pred per sample before MSE.
    # Alignment output is detached — gradients flow through pred only.
    if apply_rigid_align:
        S = pred_pos.shape[0]
        pred_flat = pred_pos.reshape(S, -1, 3)
        gt_flat = gt_pos.reshape(S, -1, 3)
        w_flat = weight_2d.reshape(-1)  # [N*M]
        gt_flat = weighted_rigid_align(gt_flat, pred_flat, w_flat)
        gt_pos = gt_flat.reshape_as(pred_pos)

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
    n_pairs = pair_mask.sum().clamp(min=1)

    # Compute log_softmax in fp32 for numerical stability
    log_probs = F.log_softmax(logits.float(), dim=-1)
    nll = -log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)

    return (nll * pair_mask).sum() / n_pairs


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
    S = logits.shape[0]

    pred_local = express_coordinates_in_frame(pred_rep, pred_frames)  # [S, N, N, 3]
    gt_local = express_coordinates_in_frame(gt_rep, gt_frames)  # [N, N, 3]
    error = (pred_local - gt_local.unsqueeze(0)).norm(dim=-1)  # [S, N, N]

    bin_edges = torch.linspace(0, max_error_bin, num_bins - 1, device=logits.device)
    target = torch.bucketize(error, bin_edges).clamp(max=num_bins - 1)

    pred_angle_ok = _frame_angle_mask(pred_frames, angle_threshold).float()  # [S, N]
    gt_angle_ok = _frame_angle_mask(gt_frames, angle_threshold).float()  # [N]
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
    gt_aug = pred["gt_augmented"]
    noise_levels = pred["noise_levels"]

    components: dict[str, torch.Tensor] = {}
    total = gt_pos.new_zeros(())

    if cfg.alpha_diffusion > 0:
        l_mse = diffusion_mse_loss(
            denoised,
            gt_aug,
            atom_mask,
            noise_levels,
            mol_type=features.get("mol_type"),
            nucleotide_loss_weight=cfg.nucleotide_loss_weight,
            ligand_loss_weight=cfg.ligand_loss_weight,
            region_weights=region_weights,
            atom_swap_pairs=features.get("atom_swap_pairs"),
            ligand_swap_pairs=features.get("ligand_swap_pairs"),
            asym_id=features.get("asym_id"),
            entity_id=features.get("entity_id"),
            chain_assign=chain_perm_from_mini_rollout(pred, features),
            atom_perm_pred=(pred["mini_rollout_pos"][0] if "mini_rollout_pos" in pred else None),
            apply_rigid_align=cfg.use_rigid_align,
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

        if cfg.alpha_bond > 0 and "bond_atom_indices" in features:
            l_bond = bond_loss(
                denoised,
                features["bond_atom_indices"],
                features["bond_gt_distances"],
                features["bond_loss_mask"],
                noise_levels=noise_levels,
            )
            components["bond"] = l_bond.detach()
            total = total + cfg.alpha_diffusion * cfg.alpha_bond * l_bond

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
    seq_mask = features["seq_mask"].bool()  # [N]
    rep_idx = features.get("token_to_rep_atom")

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
        loss_seq_mask = seq_mask & (loss_weight_mask >= 0.5)
    else:
        loss_atom_mask = atom_mask
        loss_seq_mask = seq_mask

    # --- Single-representation magnitude regularization ---
    # ``alpha_{single,pair}_reg is None`` disables entirely; 0.0 is
    # observation-only (compute under no_grad and log); >0 adds
    # ``alpha * L_p`` to the total.
    if cfg.alpha_single_reg is not None and "single" in pred:
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

        l_mse = diffusion_mse_loss(
            denoised,
            gt_aug,
            loss_atom_mask,
            noise_levels,
            mol_type=features.get("mol_type"),
            nucleotide_loss_weight=cfg.nucleotide_loss_weight,
            ligand_loss_weight=cfg.ligand_loss_weight,
            region_weights=region_weights,
            atom_swap_pairs=features.get("atom_swap_pairs"),
            ligand_swap_pairs=features.get("ligand_swap_pairs"),
            asym_id=features.get("asym_id"),
            entity_id=features.get("entity_id"),
            chain_assign=chain_perm_from_mini_rollout(pred, features),
            atom_perm_pred=(pred["mini_rollout_pos"][0] if "mini_rollout_pos" in pred else None),
            apply_rigid_align=cfg.use_rigid_align,
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
        if cfg.alpha_bond > 0 and "bond_atom_indices" in features:
            l_bond = bond_loss(
                denoised,
                features["bond_atom_indices"],
                features["bond_gt_distances"],
                features["bond_loss_mask"],
                noise_levels=noise_levels,
            )
            losses["bond"] = l_bond
            total = total + cfg.alpha_diffusion * cfg.alpha_bond * l_bond

    # Representative atom positions (shared by distogram + confidence)
    N = seq_mask.shape[0]
    if rep_idx is not None:
        arange_n = torch.arange(N, device=gt_pos.device)
        gt_rep = gt_pos[arange_n, rep_idx]  # [N, 3]
    else:
        gt_rep = gt_pos[:, 1, :]  # fallback to CA

    # --- Distogram loss (all samples, incl. distillation) ---
    # AF3 §5.2 Table 6: "Train structure and distogram" = True for stages 1–3
    # regardless of sample origin. Only confidence losses are PDB-only (§4.3).
    if cfg.alpha_distogram > 0:
        if "distogram_features" in pred:
            from deepfold3.training.fused_loss import distogram_loss_fused

            l_disto = distogram_loss_fused(
                pred["distogram_features"],
                pred["distogram_weight"],
                gt_rep,
                loss_seq_mask,
                chunk_n=cfg.fused_ce_chunk_n,
            )
        elif "distogram" in pred:
            l_disto = distogram_loss(pred["distogram"], gt_rep, loss_seq_mask)
        else:
            l_disto = None
        if l_disto is not None:
            losses["distogram"] = l_disto
            total = total + cfg.alpha_distogram * l_disto

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
                    rep_idx=rep_idx,
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
                    loss_seq_mask,
                    chunk_n=cfg.fused_ce_chunk_n,
                )
                losses["pde"] = l_pde
                total = total + cfg.alpha_confidence * conf_valid_mask * l_pde
            elif "pde" in pred:
                l_pde = pde_loss(pred["pde"], mini_rep, gt_rep, loss_seq_mask)
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
                    gt_frames_atoms = gt_pos[frame_tok, frame_idx]  # [N, 3, 3]
                    mini_frames_atoms = mini_pos[:, frame_tok, frame_idx]  # [S, N, 3, 3]

                    pair_mask_feat = features.get("pair_mask")
                    if pair_mask_feat is None:
                        pair_mask_feat = loss_seq_mask.unsqueeze(1) & loss_seq_mask.unsqueeze(0)
                    elif loss_weight_mask is not None:
                        # AND-in the per-token exclusion mask so PAE doesn't
                        # supervise pairs involving an excluded token.
                        keep = loss_weight_mask >= 0.5
                        pair_mask_feat = pair_mask_feat.bool() & keep.unsqueeze(1) & keep.unsqueeze(0)
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
                        )
                    elif "pae" in pred:
                        l_pae = pae_loss(
                            pred["pae"],
                            mini_rep,
                            mini_frames_atoms,
                            gt_rep,
                            gt_frames_atoms,
                            pae_mask,
                        )
                    else:
                        l_pae = None
                    if l_pae is not None:
                        losses["pae"] = l_pae
                        total = total + cfg.alpha_confidence * cfg.alpha_pae * conf_valid_mask * l_pae

    losses["total"] = total
    return total, losses
