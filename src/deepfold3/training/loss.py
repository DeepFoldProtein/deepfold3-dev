"""Training loss functions for DeepFold3.

Implements the multi-component loss from AF3 §5.3 (eq. 15)::

    L = alpha_diffusion * L_diff + alpha_distogram * L_disto
      + alpha_confidence * (L_plddt + L_pde + L_resolved + alpha_pae * L_pae)

All loss functions are vectorized over the diffusion sample dimension S
to avoid Python-level loops and unnecessary GPU synchronization.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812

from deepfold3.model.config import TrainingConfig

# ---------------------------------------------------------------------------
# Smooth LDDT loss (diffusion)
# ---------------------------------------------------------------------------


def smooth_lddt_loss(
    pred_pos: torch.Tensor,  # [S, N, M, 3]
    gt_pos: torch.Tensor,  # [N, M, 3]
    atom_mask: torch.Tensor,  # [N, M] bool
    cutoffs: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0),
    inclusion_radius: float = 30.0,
    min_distance: float = 1e-3,
) -> torch.Tensor:
    """Smooth LDDT loss between predicted and ground-truth atom positions.

    Vectorized over S samples — no Python loops over diffusion dimension.

    Args:
        pred_pos: Predicted positions [S, N, M, 3].
        gt_pos: Ground-truth positions [N, M, 3].
        atom_mask: Valid atom mask [N, M].
        cutoffs: Distance error thresholds in Angstroms.
        inclusion_radius: Only score pairs within this radius in GT.
        min_distance: Minimum GT distance for pair inclusion.

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
    pair_mask = (gt_dist < inclusion_radius) & (gt_dist > min_distance)  # [A, A]
    n_pairs = pair_mask.sum().clamp(min=1)

    # Batch over S: [S, A, 3]
    pred_flat = pred_pos.reshape(S, -1, 3)[:, flat_mask]
    pred_dist = torch.cdist(pred_flat, pred_flat)  # [S, A, A]
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


def diffusion_mse_loss(
    pred_pos: torch.Tensor,  # [S, N, M, 3]
    gt_pos: torch.Tensor,  # [S, N, M, 3] (augmented GT per sample)
    atom_mask: torch.Tensor,  # [N, M]
    noise_levels: torch.Tensor,  # [S]
    sigma_data: float = 16.0,
    mol_type: torch.Tensor | None = None,  # [N] int64 (0=prot, 1=RNA, 2=DNA, 3=ligand)
    nucleotide_loss_weight: float = 5.0,
    ligand_loss_weight: float = 10.0,
) -> torch.Tensor:
    """Weighted MSE between denoised predictions and augmented GT.

    Vectorized over S — single fused operation, no Python loop.

    Args:
        pred_pos: Denoised predictions [S, N, M, 3].
        gt_pos: Augmented ground truth [S, N, M, 3].
        atom_mask: Valid atoms [N, M].
        noise_levels: Noise level per sample [S].
        sigma_data: Data standard deviation.
        mol_type: Per-token molecule type [N].
        nucleotide_loss_weight: Weight multiplier for RNA/DNA atoms.
        ligand_loss_weight: Weight multiplier for ligand atoms.

    Returns:
        Scalar loss.
    """
    mask = atom_mask.float()  # [N, M]

    # Build per-token loss weight based on molecule type
    if mol_type is not None:
        token_weight = torch.ones(mask.shape[0], device=mask.device)
        token_weight[mol_type == 1] = nucleotide_loss_weight  # RNA
        token_weight[mol_type == 2] = nucleotide_loss_weight  # DNA
        token_weight[mol_type == 3] = ligand_loss_weight
        weight_mask = mask * token_weight.unsqueeze(-1)  # [N, M]
    else:
        weight_mask = mask

    weight_mask = weight_mask.unsqueeze(-1)  # [N, M, 1]
    n_weighted = weight_mask.sum().clamp(min=1)

    # Per-sample EDM weight [S]
    w = (noise_levels**2 + sigma_data**2) / (noise_levels * sigma_data) ** 2

    # Vectorized MSE: [S, N, M, 3] → [S]
    mse_per_sample = ((pred_pos - gt_pos) ** 2 * weight_mask.unsqueeze(0)).sum(dim=(-3, -2, -1)) / n_weighted
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
) -> torch.Tensor:
    """Per-atom pLDDT cross-entropy loss.

    Vectorized: computes per-atom LDDT for all S samples at once.
    """
    S, N = logits.shape[:2]
    M = atom_mask.shape[-1]
    logits_r = logits.reshape(S, N, M, num_bins)
    flat_mask = atom_mask.float()
    n_atoms = flat_mask.sum().clamp(min=1)

    # Compute per-atom LDDT for all samples at once [S, N, M]
    lddt_all = _compute_per_atom_lddt_batched(pred_pos, gt_pos, atom_mask)
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


def pae_loss(
    logits: torch.Tensor,  # [S, N, N, num_bins]
    pred_pos: torch.Tensor,  # [S, N, 3] representative atoms
    gt_pos: torch.Tensor,  # [N, 3] representative atoms
    token_mask: torch.Tensor,  # [N]
    max_error: float = 31.0,
    num_bins: int = 64,
) -> torch.Tensor:
    """Predicted aligned error cross-entropy loss."""
    pair_mask = (token_mask.unsqueeze(1) & token_mask.unsqueeze(0)).float()
    diff = pred_pos - gt_pos.unsqueeze(0)  # [S, N, 3]
    error = torch.cdist(diff, diff)  # [S, N, N]
    return _pairwise_binned_loss(logits, error, pair_mask, max_error, num_bins)


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
) -> torch.Tensor:
    """Binary cross-entropy for experimentally-resolved prediction."""
    M = atom_mask.shape[-1]
    logits_r = logits.reshape(*logits.shape[:2], M, 2)
    target = atom_mask.long().unsqueeze(0).expand(logits_r.shape[:-1])
    log_probs = F.log_softmax(logits_r.float(), dim=-1)  # fp32 for stability
    nll = -log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    return nll.mean()


# ---------------------------------------------------------------------------
# Bond violation loss
# ---------------------------------------------------------------------------


def bond_loss(
    pred_pos: torch.Tensor,  # [S, N, M, 3]
    bond_atom_indices: torch.Tensor,  # [B_bonds, 4] — (token_1, atom_1, token_2, atom_2)
    ideal_distances: torch.Tensor,  # [B_bonds]
    bond_mask: torch.Tensor,  # [B_bonds] bool
    sigma_data: float = 16.0,
    noise_levels: torch.Tensor | None = None,  # [S]
) -> torch.Tensor:
    """MSE loss on bonded atom pair distances vs target distances.

    Vectorized over S samples.
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
    pred_dist = (pos1 - pos2).norm(dim=-1)  # [S, B_bonds]

    mse_per_sample = ((pred_dist - ideal_distances.unsqueeze(0)) ** 2 * mask.unsqueeze(0)).sum(dim=-1) / n_bonds

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
) -> torch.Tensor:
    """Compute per-atom LDDT score in [0, 1], batched over S.

    Returns [S, N, M] LDDT scores.
    """
    S = pred_pos.shape[0]
    flat_mask = atom_mask.reshape(-1).bool()
    A = flat_mask.sum()
    result = torch.zeros(S, *atom_mask.shape, device=pred_pos.device, dtype=pred_pos.dtype)
    if A < 2:
        return result

    gt_flat = gt_pos.reshape(-1, 3)[flat_mask]  # [A, 3]
    gt_dist = torch.cdist(gt_flat, gt_flat)  # [A, A]
    pair_mask = (gt_dist < inclusion_radius) & (gt_dist > 1e-3)
    n_pairs = pair_mask.sum(dim=-1).clamp(min=1)  # [A]

    pred_flat = pred_pos.reshape(S, -1, 3)[:, flat_mask]  # [S, A, 3]
    pred_dist = torch.cdist(pred_flat, pred_flat)  # [S, A, A]
    dist_diff = (pred_dist - gt_dist.unsqueeze(0)).abs()

    # Per-atom scoring (hard threshold): [S, A]
    scores = pred_pos.new_zeros(S, A)
    for c in cutoffs:
        scores = scores + ((dist_diff < c).float() * pair_mask).sum(dim=-1) / n_pairs
    scores = scores / len(cutoffs)

    # Scatter back to [S, N*M]
    full = torch.zeros(S, atom_mask.numel(), device=pred_pos.device, dtype=pred_pos.dtype)
    full[:, flat_mask] = scores
    return full.reshape(S, *atom_mask.shape)


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
# Combined training loss
# ---------------------------------------------------------------------------


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
    total = gt_pos.new_zeros(())

    # --- Diffusion loss ---
    if "denoised" in pred and cfg.alpha_diffusion > 0:
        denoised = pred["denoised"]  # [S, N, M, 3]
        gt_aug = pred["gt_augmented"]  # [S, N, M, 3]
        noise_levels = pred["noise_levels"]  # [S]

        l_mse = diffusion_mse_loss(
            denoised,
            gt_aug,
            atom_mask,
            noise_levels,
            mol_type=features.get("mol_type"),
            nucleotide_loss_weight=cfg.nucleotide_loss_weight,
            ligand_loss_weight=cfg.ligand_loss_weight,
        )
        losses["diffusion_mse"] = l_mse

        l_lddt = smooth_lddt_loss(
            denoised,
            gt_pos,
            atom_mask,
            cutoffs=cfg.lddt_cutoffs,
            inclusion_radius=cfg.lddt_inclusion_radius,
            min_distance=cfg.lddt_min_distance,
        )
        losses["smooth_lddt"] = l_lddt

        total = total + cfg.alpha_diffusion * (l_mse + l_lddt)

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

    # --- Distillation: skip distogram + confidence losses ---
    is_distillation = features.get("is_distillation")
    if is_distillation is not None and is_distillation.item():
        losses["total"] = total
        return total, losses

    # Representative atom positions (shared by distogram + confidence)
    N = seq_mask.shape[0]
    if rep_idx is not None:
        arange_n = torch.arange(N, device=gt_pos.device)
        gt_rep = gt_pos[arange_n, rep_idx]  # [N, 3]
    else:
        gt_rep = gt_pos[:, 1, :]  # fallback to CA

    # --- Distogram loss ---
    if cfg.alpha_distogram > 0:
        if "distogram_features" in pred:
            from deepfold3.training.fused_loss import distogram_loss_fused

            l_disto = distogram_loss_fused(
                pred["distogram_features"],
                pred["distogram_weight"],
                gt_rep,
                seq_mask,
                chunk_n=cfg.fused_ce_chunk_n,
            )
        elif "distogram" in pred:
            l_disto = distogram_loss(pred["distogram"], gt_rep, seq_mask)
        else:
            l_disto = None
        if l_disto is not None:
            losses["distogram"] = l_disto
            total = total + cfg.alpha_distogram * l_disto

    # --- Confidence losses ---
    # Resolution gating: keep as tensor comparison to avoid GPU sync
    resolution = features.get("resolution")
    conf_valid = True
    if resolution is not None:
        # Compare on whatever device resolution lives on (may be CPU)
        conf_valid = bool((resolution >= cfg.min_resolution) & (resolution <= cfg.max_resolution))

    if conf_valid and cfg.alpha_confidence > 0:
        mini_pos = pred.get("mini_rollout_pos")  # [1, N, M, 3]
        if mini_pos is not None:
            if "plddt" in pred:
                l_plddt = plddt_loss(pred["plddt"], mini_pos, gt_pos, atom_mask)
                losses["plddt"] = l_plddt
                total = total + cfg.alpha_confidence * l_plddt

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
                    seq_mask,
                    chunk_n=cfg.fused_ce_chunk_n,
                )
                losses["pde"] = l_pde
                total = total + cfg.alpha_confidence * l_pde
            elif "pde" in pred:
                l_pde = pde_loss(pred["pde"], mini_rep, gt_rep, seq_mask)
                losses["pde"] = l_pde
                total = total + cfg.alpha_confidence * l_pde

            if "exp_resolved" in pred:
                l_resolved = exp_resolved_loss(pred["exp_resolved"], atom_mask)
                losses["exp_resolved"] = l_resolved
                total = total + cfg.alpha_confidence * l_resolved

            if cfg.alpha_pae > 0:
                if "pae_features" in pred:
                    from deepfold3.training.fused_loss import pae_loss_fused

                    l_pae = pae_loss_fused(
                        pred["pae_features"],
                        pred["pae_weight"],
                        mini_rep,
                        gt_rep,
                        seq_mask,
                        chunk_n=cfg.fused_ce_chunk_n,
                    )
                    losses["pae"] = l_pae
                    total = total + cfg.alpha_confidence * cfg.alpha_pae * l_pae
                elif "pae" in pred:
                    l_pae = pae_loss(pred["pae"], mini_rep, gt_rep, seq_mask)
                    losses["pae"] = l_pae
                    total = total + cfg.alpha_confidence * cfg.alpha_pae * l_pae

    losses["total"] = total
    return total, losses
