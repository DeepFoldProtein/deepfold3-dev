"""Confidence metric computation and JSON serialization.

Computes all AF3 confidence metrics from raw logits:
  - pLDDT (per-atom and per-token)
  - PAE (predicted aligned error)
  - PDE (predicted distance error)
  - pTM (predicted TM-score)
  - ipTM (interface predicted TM-score)
  - Experimentally resolved probability
  - Contact probabilities from distogram
  - Ranking score (0.8*ipTM + 0.2*pTM)
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from deepfold3.model.config import ConfidenceConfig
from deepfold3.model.model import DeepFold3Output, DistogramHead

_DEFAULT_CFG = ConfidenceConfig()


def _sanitize_for_json(obj: object) -> object:
    """Replace NaN/Inf float values with None for valid JSON output."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    return obj


def _pae_bin_centers(
    max_error_bin: float = _DEFAULT_CFG.max_error_bin, num_bins: int = _DEFAULT_CFG.num_distogram_bins
) -> torch.Tensor:
    """Compute PAE/PDE bin centers following AF3 convention.

    AF3: linspace(0, max_error_bin, num_bins-1) breaks, shift by half-step,
    then append one catch-all bin at the end.

    Returns [num_bins] tensor.
    """
    breaks = torch.linspace(0.0, max_error_bin, num_bins - 1)
    step = breaks[1] - breaks[0]
    centers = breaks + step / 2
    centers = torch.cat([centers, centers[-1:] + step])
    return centers


def compute_plddt(
    plddt_logits: torch.Tensor,
    atom_mask: torch.Tensor,
    num_bins: int = _DEFAULT_CFG.num_plddt_bins,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-atom and per-token pLDDT from logits.

    Args:
        plddt_logits: [S, N, M*num_bins] raw logits from confidence head.
        atom_mask: [N, M] float mask of real atoms.
        num_bins: Number of pLDDT bins (50).

    Returns:
        per_atom: [S, N, M] pLDDT in [0, 100].
        per_token: [S, N] average pLDDT per token (over present atoms).
    """
    S, N = plddt_logits.shape[:2]
    M = atom_mask.shape[-1]
    logits = plddt_logits.reshape(S, N, M, num_bins)
    probs = logits.softmax(dim=-1)
    bin_centers = (torch.arange(num_bins, device=probs.device).float() + 0.5) / num_bins
    per_atom = (probs * bin_centers).sum(-1) * 100.0  # [S, N, M]

    # Per-token: average over present atoms
    mask = atom_mask.unsqueeze(0)  # [1, N, M]
    denom = mask.sum(-1).clamp(min=1)  # [1, N]
    per_token = (per_atom * mask).sum(-1) / denom  # [S, N]

    return per_atom, per_token


def compute_pae(
    pae_logits: torch.Tensor,
    pair_mask: torch.Tensor,
    max_error_bin: float = _DEFAULT_CFG.max_error_bin,
    num_bins: int = _DEFAULT_CFG.num_distogram_bins,
) -> torch.Tensor:
    """Compute predicted aligned error from logits.

    Args:
        pae_logits: [S, N, N, num_bins] raw logits.
        pair_mask: [N, N] float mask.

    Returns:
        pae: [S, N, N] expected error in Angstroms.
    """
    centers = _pae_bin_centers(max_error_bin, num_bins).to(pae_logits.device)
    probs = pae_logits.softmax(dim=-1)
    pae = (probs * centers).sum(-1)  # [S, N, N]
    return pae * pair_mask.unsqueeze(0)


def compute_pde(
    pde_logits: torch.Tensor,
    pair_mask: torch.Tensor,
    max_error_bin: float = _DEFAULT_CFG.max_error_bin,
    num_bins: int = _DEFAULT_CFG.num_distogram_bins,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute predicted distance error from logits.

    Args:
        pde_logits: [S, N, N, num_bins] symmetrised logits.
        pair_mask: [N, N] float mask.

    Returns:
        pde: [S, N, N] expected distance error in Angstroms.
        average_pde: [S] global average PDE (scalar per sample).
    """
    centers = _pae_bin_centers(max_error_bin, num_bins).to(pde_logits.device)
    probs = pde_logits.softmax(dim=-1)
    pde = (probs * centers).sum(-1)  # [S, N, N]
    pde = pde * pair_mask.unsqueeze(0)

    # Average over valid pairs
    n_pairs = pair_mask.sum().clamp(min=1)
    average_pde = pde.sum(dim=(-2, -1)) / n_pairs  # [S]
    return pde, average_pde


def compute_exp_resolved(
    logits: torch.Tensor,
    M: int,
) -> torch.Tensor:
    """Compute experimentally resolved probability per atom.

    Args:
        logits: [S, N, M*2] raw logits (2-class per atom: not-resolved, resolved).
        M: max atoms per token.

    Returns:
        [S, N, M] probability of being resolved (softmax class 1).
    """
    S, N = logits.shape[:2]
    logits_2c = logits.reshape(S, N, M, 2)
    probs = logits_2c.softmax(dim=-1)
    return probs[..., 1]  # [S, N, M] resolved probability


def _compute_tm_adjusted_pae(
    pae_logits: torch.Tensor,
    num_tokens: int,
    max_error_bin: float = _DEFAULT_CFG.max_error_bin,
    num_bins: int = _DEFAULT_CFG.num_distogram_bins,
) -> torch.Tensor:
    """Compute TM-score-adjusted PAE values.

    For each PAE bin, computes 1 / (1 + (center/d0)^2) weighted by probs.

    Args:
        pae_logits: [N, N, num_bins] logits for a single sample.
        num_tokens: total number of tokens (for d0 computation).

    Returns:
        [N, N] TM-adjusted PAE values.
    """
    centers = _pae_bin_centers(max_error_bin, num_bins).to(pae_logits.device)

    # d0 from TM-score formula: Yang & Skolnick 2004
    clipped_num_res = max(num_tokens, 19)
    d0 = 1.24 * (clipped_num_res - 15) ** (1.0 / 3) - 1.8

    # TM-score term for every bin
    tm_per_bin = 1.0 / (1.0 + (centers / d0) ** 2)

    probs = pae_logits.softmax(dim=-1)  # [N, N, num_bins]
    return (probs * tm_per_bin).sum(-1)  # [N, N]


def compute_ptm(
    pae_logits: torch.Tensor,
    seq_mask: torch.Tensor,
    pair_mask: torch.Tensor,
    max_error_bin: float = _DEFAULT_CFG.max_error_bin,
    num_bins: int = _DEFAULT_CFG.num_distogram_bins,
) -> float:
    """Compute predicted TM-score (pTM) for a single sample.

    Args:
        pae_logits: [N, N, num_bins] logits.
        seq_mask: [N] float mask.
        pair_mask: [N, N] float mask.

    Returns:
        pTM score (scalar).
    """
    N = int(seq_mask.sum().item())
    tm_adj = _compute_tm_adjusted_pae(pae_logits, N, max_error_bin, num_bins)

    # Normalise row-wise
    mask = pair_mask
    denom = mask.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    normed_mask = mask / denom

    per_alignment = (tm_adj * normed_mask).sum(dim=-1)  # [N]
    # Mask out padding
    per_alignment = per_alignment * seq_mask
    return float(per_alignment.max().item())


def compute_iptm(
    pae_logits: torch.Tensor,
    seq_mask: torch.Tensor,
    pair_mask: torch.Tensor,
    asym_id: torch.Tensor,
    max_error_bin: float = _DEFAULT_CFG.max_error_bin,
    num_bins: int = _DEFAULT_CFG.num_distogram_bins,
) -> float:
    """Compute interface predicted TM-score (ipTM) for a single sample.

    Args:
        pae_logits: [N, N, num_bins] logits.
        seq_mask: [N] float mask.
        pair_mask: [N, N] float mask.
        asym_id: [N] int chain identifiers.

    Returns:
        ipTM score (scalar). NaN if single-chain.
    """
    # Cross-chain mask
    cross_chain = (asym_id.unsqueeze(-1) != asym_id.unsqueeze(-2)).float()
    interface_mask = pair_mask * cross_chain

    if interface_mask.sum() == 0:
        return float("nan")

    N = int(seq_mask.sum().item())
    tm_adj = _compute_tm_adjusted_pae(pae_logits, N, max_error_bin, num_bins)

    denom = interface_mask.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    normed_mask = interface_mask / denom

    per_alignment = (tm_adj * normed_mask).sum(dim=-1)  # [N]
    per_alignment = per_alignment * seq_mask
    return float(per_alignment.max().item())


def compute_ranking_score(ptm: float, iptm: float) -> float:
    """Compute AF3 ranking score: 0.8*ipTM + 0.2*pTM (or just pTM if single-chain)."""
    if iptm != iptm:  # NaN check
        return ptm
    return 0.8 * iptm + 0.2 * ptm


def compute_contact_probs(
    distogram_logits: torch.Tensor,
    threshold: float = 8.0,
    first_break: float = DistogramHead.FIRST_BREAK,
    last_break: float = DistogramHead.LAST_BREAK,
    num_bins: int = DistogramHead.NUM_BINS,
) -> torch.Tensor:
    """Compute contact probability map from distogram logits.

    Args:
        distogram_logits: [N, N, num_bins] logits.
        threshold: distance threshold in Angstroms (8.0 for contacts).

    Returns:
        [N, N] probability of distance < threshold.
    """
    device = distogram_logits.device
    breaks = torch.linspace(first_break, last_break, num_bins - 1, device=device)
    probs = distogram_logits.softmax(dim=-1)

    # Sum probability mass of bins whose upper edge <= threshold
    # Bin i spans [breaks[i-1], breaks[i]); bin 0 spans [0, breaks[0])
    last_gap = breaks[-1] - breaks[-2]
    bin_upper = torch.cat([breaks, breaks[-1:] + last_gap])  # [num_bins]
    contact_bins = (bin_upper <= threshold).float()
    return (probs * contact_bins).sum(-1)  # [N, N]


def save_confidences_json(
    path: str | Path,
    output: DeepFold3Output,
    features: dict[str, torch.Tensor],
    sample_idx: int = 0,
    cfg: ConfidenceConfig | None = None,
) -> dict:
    """Compute all confidence metrics and save as JSON.

    Args:
        path: Output JSON file path.
        output: DeepFold3Output from model.forward().
        features: Feature dict from featurizer.
        sample_idx: Which diffusion sample to export (default 0).
        cfg: ConfidenceConfig (uses defaults if None).

    Returns:
        The summary dict (subset of the full JSON).
    """
    if cfg is None:
        cfg = _DEFAULT_CFG
    max_error_bin = cfg.max_error_bin
    num_bins = cfg.num_distogram_bins
    num_plddt_bins = cfg.num_plddt_bins
    path = Path(path)
    device = output.atom_positions.device

    atom_mask = output.atom_mask[0].float()  # [N, M] — same for all samples
    seq_mask = features["seq_mask"].float().to(device)
    pair_mask = features["pair_mask"].float().to(device)
    asym_id = features["asym_id"].to(device)
    residue_index = features["residue_index"]
    N = seq_mask.shape[0]
    M = atom_mask.shape[-1]

    result: dict = {}

    # Chain IDs per token
    from deepfold3.io.mmcif import _asym_id_to_chain_name

    asym_cpu = asym_id.cpu()
    token_chain_ids = [_asym_id_to_chain_name(int(a)) for a in asym_cpu]
    result["token_chain_ids"] = token_chain_ids
    result["token_res_ids"] = residue_index.cpu().tolist()

    # Chain IDs per atom (expanded)
    ref_mask = features["ref_mask"].cpu()  # [N, M] bool
    atom_chain_ids = []
    for i in range(N):
        for j in range(M):
            if ref_mask[i, j]:
                atom_chain_ids.append(token_chain_ids[i])
    result["atom_chain_ids"] = atom_chain_ids

    # pLDDT
    if output.plddt is not None:
        per_atom_plddt, per_token_plddt = compute_plddt(output.plddt, atom_mask, num_plddt_bins)
        s = sample_idx

        # Flat atom list (only real atoms)
        atom_plddts = []
        pa = per_atom_plddt[s].cpu()
        for i in range(N):
            for j in range(M):
                if ref_mask[i, j]:
                    atom_plddts.append(round(float(pa[i, j]), 2))
        result["atom_plddts"] = atom_plddts
        result["token_plddts"] = [round(float(v), 2) for v in per_token_plddt[s].cpu()]
        mean_plddt = float((per_token_plddt[s].cpu() * seq_mask.cpu()).sum() / seq_mask.cpu().sum().clamp(min=1))
    else:
        mean_plddt = 0.0

    # PAE
    if output.pae is not None:
        pae_full = compute_pae(output.pae, pair_mask, max_error_bin, num_bins)
        pae_sample = pae_full[sample_idx].cpu()
        result["pae"] = [[round(float(v), 2) for v in row] for row in pae_sample]

        # pTM and ipTM
        pae_logits_s = output.pae[sample_idx]
        ptm = compute_ptm(pae_logits_s, seq_mask, pair_mask, max_error_bin, num_bins)
        iptm = compute_iptm(pae_logits_s, seq_mask, pair_mask, asym_id, max_error_bin, num_bins)
    else:
        ptm = 0.0
        iptm = float("nan")

    # PDE
    if output.pde is not None:
        pde_full, avg_pde = compute_pde(output.pde, pair_mask, max_error_bin, num_bins)
        pde_sample = pde_full[sample_idx].cpu()
        result["pde"] = [[round(float(v), 2) for v in row] for row in pde_sample]
        average_pde = float(avg_pde[sample_idx].item())
    else:
        average_pde = 0.0

    # Experimentally resolved
    if output.exp_resolved is not None:
        exp_res = compute_exp_resolved(output.exp_resolved, M)
        er = exp_res[sample_idx].cpu()
        atom_exp_resolved = []
        for i in range(N):
            for j in range(M):
                if ref_mask[i, j]:
                    atom_exp_resolved.append(round(float(er[i, j]), 3))
        result["atom_exp_resolved"] = atom_exp_resolved

    # Contact probabilities from distogram
    if output.distogram is not None:
        contact_probs = compute_contact_probs(output.distogram)
        result["contact_probs"] = [[round(float(v), 3) for v in row] for row in contact_probs.cpu()]

    # Ranking score
    ranking_score = compute_ranking_score(ptm, iptm)

    # Number of unique chains
    num_chains = int(asym_cpu.unique().numel())

    summary = {
        "ptm": round(ptm, 4),
        "iptm": round(iptm, 4) if iptm == iptm else None,
        "ranking_score": round(ranking_score, 4),
        "mean_plddt": round(mean_plddt, 2),
        "average_pde": round(average_pde, 2),
        "num_tokens": N,
        "num_chains": num_chains,
    }
    result["summary"] = summary

    # Write JSON (sanitize NaN/Inf → null for valid JSON)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_sanitize_for_json(result), indent=2))

    return summary
