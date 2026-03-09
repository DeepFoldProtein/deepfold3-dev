"""Training loss functions for DeepFold3.

Implements the multi-component loss from AF3 SS3.7::

    L = alpha_diffusion * L_diff + alpha_distogram * L_disto
      + alpha_confidence * (L_plddt + L_pde + L_resolved + alpha_pae * L_pae)

All loss functions accept configurable parameters; nothing is hardcoded.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812

from deepfold3.model.config import TrainingConfig

# ---------------------------------------------------------------------------
# Smooth LDDT loss (diffusion)
# ---------------------------------------------------------------------------


def _lddt_pairwise(
    pred_pos: torch.Tensor,  # [N, M, 3]
    gt_pos: torch.Tensor,  # [N, M, 3]
    atom_mask: torch.Tensor,  # [N, M]
    inclusion_radius: float,
    min_distance: float = 1e-3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int] | None:
    """Shared setup for LDDT: flatten atoms, compute pairwise distances, build mask.

    Args:
        pred_pos: Predicted positions ``[N, M, 3]``.
        gt_pos: Ground-truth positions ``[N, M, 3]``.
        atom_mask: Valid atom mask ``[N, M]``.
        inclusion_radius: Only score pairs within this radius in GT.
        min_distance: Minimum GT distance for pair inclusion.

    Returns:
        ``None`` if fewer than 2 valid atoms, otherwise
        ``(dist_diff, gt_dist, pair_mask, flat_mask, A)`` where
        dist_diff is ``[A, A]``, pair_mask is ``[A, A]`` bool.
    """
    flat_mask = atom_mask.reshape(-1).bool()
    A = int(flat_mask.sum())
    if A < 2:
        return None

    gt_flat = gt_pos.reshape(-1, 3)[flat_mask]  # [A, 3]
    pred_flat = pred_pos.reshape(-1, 3)[flat_mask]  # [A, 3]

    gt_dist = torch.cdist(gt_flat, gt_flat)  # [A, A]
    pred_dist = torch.cdist(pred_flat, pred_flat)  # [A, A]
    pair_mask = (gt_dist < inclusion_radius) & (gt_dist > min_distance)
    dist_diff = (pred_dist - gt_dist).abs()

    return dist_diff, gt_dist, pair_mask, flat_mask, A


def smooth_lddt_loss(
    pred_pos: torch.Tensor,  # [S, N, M, 3]
    gt_pos: torch.Tensor,  # [N, M, 3]
    atom_mask: torch.Tensor,  # [N, M] bool
    cutoffs: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0),
    inclusion_radius: float = 30.0,
    min_distance: float = 1e-3,
) -> torch.Tensor:
    """Smooth LDDT loss between predicted and ground-truth atom positions.

    Computes average fraction of atom-pair distance errors below each cutoff
    threshold, scored with a sigmoid approximation, then averaged.

    Args:
        pred_pos: Predicted positions, shape [S, N, M, 3].
        gt_pos: Ground-truth positions, shape [N, M, 3].
        atom_mask: Valid atom mask, shape [N, M].
        cutoffs: Distance error thresholds in Angstroms.
        inclusion_radius: Only score pairs within this radius in GT.
        min_distance: Minimum GT distance for pair inclusion.

    Returns:
        Scalar loss (1 - smooth_lddt), averaged over samples.
    """
    S = pred_pos.shape[0]

    # GT pairwise setup (reused across samples)
    flat_mask = atom_mask.reshape(-1).bool()
    A = int(flat_mask.sum())
    if A < 2:
        return pred_pos.new_zeros(())

    gt_flat = gt_pos.reshape(-1, 3)[flat_mask]
    gt_dist = torch.cdist(gt_flat, gt_flat)
    pair_mask = (gt_dist < inclusion_radius) & (gt_dist > min_distance)
    n_pairs = pair_mask.sum().clamp(min=1)

    loss_sum = pred_pos.new_zeros(())
    for s in range(S):
        result = _lddt_pairwise(pred_pos[s], gt_pos, atom_mask, inclusion_radius)
        if result is None:
            continue
        dist_diff = result[0]

        # Score: average over cutoffs of sigmoid-scored fractions
        score = pred_pos.new_zeros(())
        for c in cutoffs:
            score = score + (0.5 * (1.0 + torch.tanh((c - dist_diff) * 5.0)) * pair_mask).sum() / n_pairs
        score = score / len(cutoffs)

        loss_sum = loss_sum + (1.0 - score)

    return loss_sum / S


# ---------------------------------------------------------------------------
# Diffusion MSE loss with EDM weighting
# ---------------------------------------------------------------------------


def diffusion_mse_loss(
    pred_pos: torch.Tensor,  # [S, N, M, 3]
    gt_pos: torch.Tensor,  # [S, N, M, 3] (augmented GT per sample)
    atom_mask: torch.Tensor,  # [N, M]
    noise_levels: torch.Tensor,  # [S]
    sigma_data: float = 16.0,
) -> torch.Tensor:
    """Weighted MSE between denoised predictions and augmented GT.

    EDM weighting: w(sigma) = (sigma^2 + sigma_data^2) / (sigma * sigma_data)^2

    Args:
        pred_pos: Denoised predictions [S, N, M, 3].
        gt_pos: Augmented ground truth [S, N, M, 3].
        atom_mask: Valid atoms [N, M].
        noise_levels: Noise level per sample [S].
        sigma_data: Data standard deviation.

    Returns:
        Scalar loss.
    """
    S = pred_pos.shape[0]
    mask = atom_mask.unsqueeze(0).unsqueeze(-1).float()  # [1, N, M, 1]
    n_atoms = mask.sum().clamp(min=1)

    # Per-sample EDM weight
    w = (noise_levels**2 + sigma_data**2) / (noise_levels * sigma_data) ** 2  # [S]

    loss = pred_pos.new_zeros(())
    for s in range(S):
        mse = ((pred_pos[s] - gt_pos[s]) ** 2 * mask.squeeze(0)).sum() / n_atoms
        loss = loss + w[s] * mse

    return loss / S


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
    """Cross-entropy distogram loss.

    Args:
        logits: Predicted distance bin logits [N, N, num_bins].
        gt_positions: Ground-truth representative atom positions [N, 3].
        token_mask: Valid token mask [N].
        first_break: First distance bin edge.
        last_break: Last distance bin edge.
        num_bins: Number of bins.

    Returns:
        Scalar loss.
    """
    # Compute GT pairwise distances
    dist = torch.cdist(gt_positions, gt_positions)  # [N, N]

    # Bin edges
    breaks = torch.linspace(first_break, last_break, num_bins - 1, device=dist.device)
    # Target bin indices: which bin does each distance fall into
    target = torch.bucketize(dist, breaks)  # [N, N] in [0, num_bins-1]

    # Pair mask
    pair_mask = (token_mask.unsqueeze(1) & token_mask.unsqueeze(0)).float()  # [N, N]
    n_pairs = pair_mask.sum().clamp(min=1)

    log_probs = F.log_softmax(logits, dim=-1)  # [N, N, num_bins]
    nll = -log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)  # [N, N]

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

    Bins true lddt into num_bins classes and computes cross-entropy.

    Args:
        logits: Predicted pLDDT logits [S, N, M * num_bins].
        pred_pos: Predicted positions [S, N, M, 3].
        gt_pos: Ground-truth positions [N, M, 3].
        atom_mask: Valid atom mask [N, M].
        num_bins: Number of pLDDT bins.

    Returns:
        Scalar loss averaged over samples and atoms.
    """
    S, N = logits.shape[:2]
    M = atom_mask.shape[-1]
    logits_r = logits.reshape(S, N, M, num_bins)
    flat_mask = atom_mask.float()

    n_atoms = flat_mask.sum().clamp(min=1)
    loss = logits.new_zeros(())

    for s in range(S):
        # Per-atom LDDT: compare distances to all other atoms
        lddt_per_atom = _compute_per_atom_lddt(pred_pos[s], gt_pos, atom_mask)  # [N, M]
        # Bin into [0, num_bins-1]
        target = (lddt_per_atom * num_bins).clamp(0, num_bins - 1).long()  # [N, M]
        log_probs = F.log_softmax(logits_r[s], dim=-1)  # [N, M, num_bins]
        nll = -log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)  # [N, M]
        loss = loss + (nll * flat_mask).sum() / n_atoms

    return loss / S


def _pairwise_binned_loss(
    logits: torch.Tensor,  # [S, N, N, num_bins]
    error: torch.Tensor,  # [S, N, N]
    pair_mask: torch.Tensor,  # [N, N]
    max_error: float = 31.0,
    num_bins: int = 64,
) -> torch.Tensor:
    """Shared NLL loss for pairwise error predictions (PAE and PDE).

    Args:
        logits: Predicted bin logits [S, N, N, num_bins].
        error: Per-sample pairwise error [S, N, N].
        pair_mask: Valid token-pair mask [N, N].
        max_error: Maximum error bin edge.
        num_bins: Number of bins.

    Returns:
        Scalar loss averaged over samples and pairs.
    """
    bin_edges = torch.linspace(0, max_error, num_bins - 1, device=logits.device)
    n_pairs = pair_mask.sum().clamp(min=1)

    target = torch.bucketize(error, bin_edges).clamp(max=num_bins - 1)  # [S, N, N]
    log_probs = F.log_softmax(logits, dim=-1)  # [S, N, N, num_bins]
    nll = -log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)  # [S, N, N]
    return (nll * pair_mask.unsqueeze(0)).sum() / (n_pairs * logits.shape[0])


def pae_loss(
    logits: torch.Tensor,  # [S, N, N, num_bins]
    pred_pos: torch.Tensor,  # [S, N, 3] representative atoms
    gt_pos: torch.Tensor,  # [N, 3] representative atoms
    token_mask: torch.Tensor,  # [N]
    max_error: float = 31.0,
    num_bins: int = 64,
) -> torch.Tensor:
    """Predicted aligned error cross-entropy loss.

    Args:
        logits: PAE logits [S, N, N, num_bins].
        pred_pos: Predicted representative atom positions [S, N, 3].
        gt_pos: Ground-truth representative atom positions [N, 3].
        token_mask: Valid token mask [N].
        max_error: Maximum error bin edge.
        num_bins: Number of bins.

    Returns:
        Scalar loss.
    """
    pair_mask = (token_mask.unsqueeze(1) & token_mask.unsqueeze(0)).float()
    # Per-sample alignment error: ||diff[j] - diff[i]|| where diff = pred - gt
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
    """Predicted distance error cross-entropy loss.

    Args:
        logits: PDE logits [S, N, N, num_bins].
        pred_pos: Predicted representative atom positions [S, N, 3].
        gt_pos: Ground-truth representative atom positions [N, 3].
        token_mask: Valid token mask [N].
        max_error: Maximum error bin edge.
        num_bins: Number of bins.

    Returns:
        Scalar loss.
    """
    pair_mask = (token_mask.unsqueeze(1) & token_mask.unsqueeze(0)).float()
    gt_dist = torch.cdist(gt_pos, gt_pos)  # [N, N]
    pred_dist = torch.cdist(pred_pos, pred_pos)  # [S, N, N] (batched cdist)
    error = (pred_dist - gt_dist.unsqueeze(0)).abs()  # [S, N, N]
    return _pairwise_binned_loss(logits, error, pair_mask, max_error, num_bins)


def exp_resolved_loss(
    logits: torch.Tensor,  # [S, N, M * 2]
    atom_mask: torch.Tensor,  # [N, M] bool — ground truth resolved mask
) -> torch.Tensor:
    """Binary cross-entropy for experimentally-resolved prediction.

    Args:
        logits: Predicted logits [S, N, M * 2] (2 classes per atom).
        atom_mask: Ground-truth resolved mask [N, M].

    Returns:
        Scalar loss.
    """
    M = atom_mask.shape[-1]
    logits_r = logits.reshape(*logits.shape[:2], M, 2)  # [S, N, M, 2]
    target = atom_mask.long().unsqueeze(0).expand(logits_r.shape[:-1])  # [S, N, M]
    log_probs = F.log_softmax(logits_r, dim=-1)
    nll = -log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)  # [S, N, M]
    return nll.mean()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_per_atom_lddt(
    pred_pos: torch.Tensor,  # [N, M, 3]
    gt_pos: torch.Tensor,  # [N, M, 3]
    atom_mask: torch.Tensor,  # [N, M]
    cutoffs: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0),
    inclusion_radius: float = 30.0,
) -> torch.Tensor:
    """Compute per-atom LDDT score in ``[0, 1]``.

    Args:
        pred_pos: Predicted positions ``[N, M, 3]``.
        gt_pos: Ground-truth positions ``[N, M, 3]``.
        atom_mask: Valid atom mask ``[N, M]``.
        cutoffs: Distance error thresholds in Angstroms.
        inclusion_radius: Only score pairs within this radius in GT.

    Returns:
        Per-atom LDDT scores ``[N, M]``.
    """
    result = torch.zeros_like(atom_mask, dtype=pred_pos.dtype)
    pw = _lddt_pairwise(pred_pos, gt_pos, atom_mask, inclusion_radius)
    if pw is None:
        return result
    dist_diff, _, pair_mask, flat_mask, A = pw

    # Per-atom scoring (hard threshold, not smooth)
    scores = torch.zeros(A, device=pred_pos.device, dtype=pred_pos.dtype)
    for c in cutoffs:
        contrib = ((dist_diff < c).float() * pair_mask).sum(dim=-1)
        n_pairs = pair_mask.sum(dim=-1).clamp(min=1)
        scores = scores + contrib / n_pairs
    scores = scores / len(cutoffs)

    # Scatter back to [N*M]
    full_scores = torch.zeros(atom_mask.numel(), device=pred_pos.device, dtype=pred_pos.dtype)
    full_scores[flat_mask] = scores
    return full_scores.reshape(atom_mask.shape)


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
        pred: Model predictions dict. Expected keys:

            - ``denoised`` ``[S, N, M, 3]``
            - ``gt_augmented`` ``[S, N, M, 3]``
            - ``noise_levels`` ``[S]``
            - ``distogram`` ``[N, N, 64]``
            - ``plddt`` ``[S, N, M*50]``
            - ``pae`` ``[S, N, N, 64]``
            - ``pde`` ``[S, N, N, 64]``
            - ``exp_resolved`` ``[S, N, M*2]``
            - ``mini_rollout_pos`` ``[1, N, M, 3]``

        features: Input features including ``atom_gt_positions``,
            ``atom_gt_mask``, ``seq_mask``, ``token_to_rep_atom``, etc.
        cfg: Training config with loss weights and parameters.

    Returns:
        Tuple of ``(total_loss, loss_dict)`` where ``loss_dict`` maps
        component names to their scalar loss tensors.
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

        l_mse = diffusion_mse_loss(denoised, gt_aug, atom_mask, noise_levels)
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

    # Representative atom positions (shared by distogram + confidence)
    N = seq_mask.shape[0]
    if rep_idx is not None:
        arange_n = torch.arange(N, device=gt_pos.device)
        gt_rep = gt_pos[arange_n, rep_idx]  # [N, 3]
    else:
        gt_rep = gt_pos[:, 1, :]  # fallback to CA

    # --- Distogram loss ---
    if "distogram" in pred and cfg.alpha_distogram > 0:
        l_disto = distogram_loss(pred["distogram"], gt_rep, seq_mask)
        losses["distogram"] = l_disto
        total = total + cfg.alpha_distogram * l_disto

    # --- Confidence losses ---
    # Only compute if resolution is within range
    resolution = features.get("resolution")
    conf_valid = True
    if resolution is not None:
        r = resolution.item() if resolution.dim() == 0 else resolution.mean().item()
        conf_valid = cfg.min_resolution <= r <= cfg.max_resolution

    if conf_valid and cfg.alpha_confidence > 0:
        # Use mini-rollout positions for confidence targets
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

            if "pde" in pred:
                l_pde = pde_loss(pred["pde"], mini_rep, gt_rep, seq_mask)
                losses["pde"] = l_pde
                total = total + cfg.alpha_confidence * l_pde

            if "exp_resolved" in pred:
                l_resolved = exp_resolved_loss(pred["exp_resolved"], atom_mask)
                losses["exp_resolved"] = l_resolved
                total = total + cfg.alpha_confidence * l_resolved

            if "pae" in pred and cfg.alpha_pae > 0:
                l_pae = pae_loss(pred["pae"], mini_rep, gt_rep, seq_mask)
                losses["pae"] = l_pae
                total = total + cfg.alpha_confidence * cfg.alpha_pae * l_pae

    losses["total"] = total
    return total, losses
