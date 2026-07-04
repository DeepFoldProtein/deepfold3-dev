"""Confidence metric computation and JSON serialization.

Computes all AF3 confidence metrics from raw logits:
  - pLDDT (per-atom and per-token)
  - PAE (predicted aligned error)
  - PDE (predicted distance error)
  - pTM (predicted TM-score)
  - ipTM (interface predicted TM-score)
  - Experimentally resolved probability
  - Contact probabilities from distogram
  - has_clash (steric clash detection via KD-tree)
  - fraction_disordered (fraction of disordered protein residues)
  - Ranking score: 0.8*ipTM + 0.2*pTM + 0.5*fraction_disordered - 100*has_clash
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from deepfold3.model.config import ConfidenceConfig, DistogramConfig
from deepfold3.model.model import DeepFold3Output

_DEFAULT_CFG = ConfidenceConfig()

# Clash detection constants (AF3 confidences.py:131-182)
_CLASH_CUTOFF = 1.1  # Angstroms
_MIN_CLASHES = 100
_MIN_CLASH_FRACTION = 0.5

# Disorder detection constant (pLDDT-based proxy)
_DISORDER_PLDDT_THRESHOLD = 50.0


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


def _error_bin_centers(
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
    centers = _error_bin_centers(max_error_bin, num_bins).to(pae_logits.device)
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
    centers = _error_bin_centers(max_error_bin, num_bins).to(pde_logits.device)
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
    num_tokens: int | torch.Tensor,
    max_error_bin: float = _DEFAULT_CFG.max_error_bin,
    num_bins: int = _DEFAULT_CFG.num_distogram_bins,
) -> torch.Tensor:
    """Compute TM-score-adjusted PAE values.

    For each PAE bin, computes 1 / (1 + (center/d0)^2) weighted by probs.

    Args:
        pae_logits: [N, N, num_bins] logits for a single sample.
        num_tokens: total number of tokens (scalar for pTM) or [N, N] tensor
            of per-pair token counts (for ipTM with per-chain-pair d0).

    Returns:
        [N, N] TM-adjusted PAE values.
    """
    centers = _error_bin_centers(max_error_bin, num_bins).to(pae_logits.device)

    if isinstance(num_tokens, torch.Tensor):
        # Per-pair d0: num_tokens is [N, N]
        clipped = num_tokens.clamp(min=19).float()
        d0 = 1.24 * (clipped - 15) ** (1.0 / 3) - 1.8  # [N, N]
        # centers [B] → [1, 1, B], d0 [N, N] → [N, N, 1]
        tm_per_bin = 1.0 / (1.0 + (centers.unsqueeze(0).unsqueeze(0) / d0.unsqueeze(-1)) ** 2)
    else:
        # Scalar d0 (pTM path)
        clipped_num_res = max(num_tokens, 19)
        d0 = 1.24 * (clipped_num_res - 15) ** (1.0 / 3) - 1.8
        tm_per_bin = 1.0 / (1.0 + (centers / d0) ** 2)

    probs = pae_logits.softmax(dim=-1)  # [N, N, num_bins]
    return (probs * tm_per_bin).sum(-1)  # [N, N]


def compute_ptm(
    pae_logits: torch.Tensor,
    seq_mask: torch.Tensor,
    pair_mask: torch.Tensor,
    max_error_bin: float = _DEFAULT_CFG.max_error_bin,
    num_bins: int = _DEFAULT_CFG.num_distogram_bins,
    d0_seq_mask: torch.Tensor | None = None,
) -> float:
    """Compute predicted TM-score (pTM) for a single sample.

    Args:
        pae_logits: [N, N, num_bins] logits.
        seq_mask: [N] float mask gating the per-row alignment + max.
        pair_mask: [N, N] float mask for the row-wise j normalization.
        d0_seq_mask: [N] float mask sizing the d0 token count. AF3 sizes d0 on
            the count of *all valid tokens* (``seq_mask.sum()`` in
            ``confidence_head._get_tmscore_adjusted_pae``), while the j-sum is
            normalized over frame-bearing tokens (``pair_mask``). Pass the full
            token mask here when ``seq_mask``/``pair_mask`` are frame-restricted.
            Defaults to ``seq_mask`` (frame count) for backward compatibility.

    Returns:
        pTM score (scalar).
    """
    d0_mask = seq_mask if d0_seq_mask is None else d0_seq_mask
    N = int(d0_mask.sum().item())
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
    d0_seq_mask: torch.Tensor | None = None,
) -> float:
    """Compute interface predicted TM-score (ipTM) for a single sample.

    AF3 §5.9.1 Eq. 18::

        ipTM(D) = max_{i∈D, has_frame(i)} (1/|D-chain(i)|)
                    Σ_{j∈D-chain(i)} Σ_b p^{ij}_b / (1 + (Δ_b / d0(|D|))²)

    where ``D-chain(i)`` excludes tokens in ``i``'s chain. This is the
    **full-structure** ipTM (``D`` = all tokens), so ``d0`` is sized on
    ``|D|`` = the total valid-token count — a single **global** d0, identical
    to pTM's, NOT a per-chain-pair size. (Earlier this used a per-pair
    ``num_interface_tokens`` d0 that equals Eq. 18 only for two-chain complexes
    and diverged for >2 chains; also disagreeing with the training-time
    ``itm_true`` diagnostic, which already uses the global |D|.)

    Args:
        pae_logits: [N, N, num_bins] logits.
        seq_mask: [N] float mask gating the per-row alignment + max.
        pair_mask: [N, N] float mask for the row-wise (cross-chain) j-sum.
        asym_id: [N] int chain identifiers.
        d0_seq_mask: [N] float mask sizing ``|D|`` for d0 (all valid tokens).
            Defaults to ``seq_mask``; pass the full token mask when
            ``seq_mask``/``pair_mask`` are frame-restricted (matches pTM).

    Returns:
        ipTM score (scalar). NaN if single-chain.
    """
    same_chain = asym_id.unsqueeze(-1) == asym_id.unsqueeze(-2)  # [N, N] bool
    cross_chain = (~same_chain).float()
    interface_mask = pair_mask * cross_chain
    if interface_mask.sum() == 0:
        return float("nan")

    # d0(|D|): full-structure ipTM sizes d0 on the total valid-token count
    # (global, identical to pTM) — Eq. 18, not a per-chain-pair size.
    d0_mask = seq_mask if d0_seq_mask is None else d0_seq_mask
    N = int(d0_mask.sum().item())
    tm_adj = _compute_tm_adjusted_pae(pae_logits, N, max_error_bin, num_bins)

    denom = interface_mask.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    normed_mask = interface_mask / denom
    per_alignment = (tm_adj * normed_mask).sum(dim=-1)  # [N]
    per_alignment = per_alignment * seq_mask
    return float(per_alignment.max().item())


def has_clash(
    atom_positions: torch.Tensor,
    atom_mask: torch.Tensor,
    residue_index: torch.Tensor,
    asym_id: torch.Tensor,
    polymer_mask: torch.Tensor,
    cutoff: float = _CLASH_CUTOFF,
    min_clashes: int = _MIN_CLASHES,
    min_clash_fraction: float = _MIN_CLASH_FRACTION,
) -> bool:
    """Detect steric clashes in predicted structure using KD-tree.

    AF3 algorithm: for each polymer atom, find neighbors within cutoff_radius.
    An atom is clashing if any neighbor is >1 residue apart or on a different chain.
    A chain has a clash if num_clashes > min_clashes OR frac_clashes > min_clash_fraction.

    Args:
        atom_positions: [N, M, 3] predicted coordinates.
        atom_mask: [N, M] float mask of real atoms.
        residue_index: [N] int residue indices.
        asym_id: [N] int chain identifiers.
        polymer_mask: [N] bool mask of polymer tokens.
        cutoff: Distance threshold in Angstroms (1.1).
        min_clashes: Minimum clash count per chain to flag (100).
        min_clash_fraction: Minimum fraction of clashing atoms per chain (0.5).

    Returns:
        True if any chain has a clash.
    """
    try:
        from scipy.spatial import KDTree
    except ImportError:
        return False

    pos = atom_positions.detach().cpu()
    mask = atom_mask.detach().cpu()
    poly = polymer_mask.detach().cpu().bool()
    res_idx = residue_index.detach().cpu()
    chain_id = asym_id.detach().cpu()

    N, M = mask.shape

    # Collect polymer atom coords, token index, and chain id
    coords = []
    token_indices = []
    chain_ids = []
    for i in range(N):
        if not poly[i]:
            continue
        for j in range(M):
            if mask[i, j] > 0:
                coords.append(pos[i, j].numpy())
                token_indices.append(int(res_idx[i]))
                chain_ids.append(int(chain_id[i]))

    if len(coords) < 2:
        return False

    coords_arr = np.stack(coords)
    token_indices_arr = np.array(token_indices)
    chain_ids_arr = np.array(chain_ids)

    tree = KDTree(coords_arr)
    neighbors = tree.query_ball_tree(tree, cutoff)

    # Mark atoms as clashing
    is_clashing = np.zeros(len(coords), dtype=bool)
    for i, nbrs in enumerate(neighbors):
        for j in nbrs:
            if i == j:
                continue
            if chain_ids_arr[i] != chain_ids_arr[j] or abs(token_indices_arr[i] - token_indices_arr[j]) > 1:
                is_clashing[i] = True
                break

    # Per-chain check
    unique_chains = np.unique(chain_ids_arr)
    for c in unique_chains:
        chain_sel = chain_ids_arr == c
        n_atoms = int(chain_sel.sum())
        if n_atoms == 0:
            continue
        n_clashes = int(is_clashing[chain_sel].sum())
        if n_clashes > min_clashes or (n_clashes / n_atoms) > min_clash_fraction:
            return True

    return False


def fraction_disordered(
    per_token_plddt: torch.Tensor,
    polymer_mask: torch.Tensor,
    threshold: float = _DISORDER_PLDDT_THRESHOLD,
) -> float:
    """Compute fraction of disordered protein residues from pLDDT.

    Uses pLDDT < threshold as a proxy for disorder (standard AF2/AF3 convention).

    Args:
        per_token_plddt: [N] per-token pLDDT values.
        polymer_mask: [N] bool mask of polymer tokens.
        threshold: pLDDT below this is considered disordered (50.0).

    Returns:
        Fraction of polymer tokens that are disordered (0.0 to 1.0).
    """
    poly = polymer_mask.bool()
    n_polymer = int(poly.sum().item())
    if n_polymer == 0:
        return 0.0
    disordered = (per_token_plddt < threshold) & poly
    return float(disordered.sum().item()) / n_polymer


def compute_ranking_score(ptm: float, iptm: float, frac_disordered: float = 0.0, clash: bool = False) -> float:
    """Compute AF3 ranking score.

    Formula: 0.8*ipTM + 0.2*pTM + 0.5*fraction_disordered - 100*has_clash.
    Single-chain: pTM + 0.5*fraction_disordered - 100*has_clash.
    """
    if math.isnan(iptm):
        ptm_iptm_average = ptm
    else:
        ptm_iptm_average = 0.8 * iptm + 0.2 * ptm
    return ptm_iptm_average + 0.5 * frac_disordered - 100.0 * clash


def _distogram_bin_centers(
    first_break: float = DistogramConfig.first_break,
    last_break: float = DistogramConfig.last_break,
    num_bins: int = DistogramConfig.num_bins,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Distogram bin centers in Angstroms.

    The model emits ``num_bins`` bins delimited by ``num_bins - 1`` linearly
    spaced breaks in ``[first_break, last_break]``. Bin 0 covers ``[0, breaks[0])``
    and the final bin extends one step past ``breaks[-1]``. Centers are taken
    as midpoints (the leading and trailing bins use ``breaks[0] / 2`` and
    ``breaks[-1] + step / 2`` respectively).
    """
    breaks = torch.linspace(first_break, last_break, num_bins - 1, device=device)
    step = breaks[1] - breaks[0]
    centers = torch.empty(num_bins, device=breaks.device)
    centers[0] = breaks[0] / 2
    centers[1:-1] = (breaks[:-1] + breaks[1:]) / 2
    centers[-1] = breaks[-1] + step / 2
    return centers


def compute_distogram_expected(
    distogram_logits: torch.Tensor,
    first_break: float = DistogramConfig.first_break,
    last_break: float = DistogramConfig.last_break,
    num_bins: int = DistogramConfig.num_bins,
) -> torch.Tensor:
    """Per-pair expected distance from distogram logits.

    Args:
        distogram_logits: [N, N, num_bins] logits.

    Returns:
        [N, N] expected pairwise distance in Angstroms.
    """
    centers = _distogram_bin_centers(first_break, last_break, num_bins, device=distogram_logits.device)
    probs = distogram_logits.softmax(dim=-1)
    return (probs * centers).sum(-1)


def compute_contact_probs(
    distogram_logits: torch.Tensor,
    threshold: float = 8.0,
    first_break: float = DistogramConfig.first_break,
    last_break: float = DistogramConfig.last_break,
    num_bins: int = DistogramConfig.num_bins,
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
    seed: int | None = None,
) -> dict:
    """Compute all confidence metrics and save as JSON.

    Args:
        path: Output JSON file path.
        output: DeepFold3Output from model.forward().
        features: Feature dict from featurizer.
        sample_idx: Which diffusion sample to export (default 0).
        cfg: ConfidenceConfig (uses defaults if None).
        seed: Random seed used for diffusion sampling (saved in summary).

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

    # frames_mask: use for pTM/ipTM if available (excludes frameless tokens like ions/ligands)
    if "frames_mask" in features:
        tm_mask = features["frames_mask"].float().to(device)
        tm_pair_mask = (tm_mask.unsqueeze(-1) * tm_mask.unsqueeze(-2)) * pair_mask
    else:
        tm_mask = seq_mask
        tm_pair_mask = pair_mask

    result: dict = {}

    # Chain IDs per token
    from deepfold3.io.mmcif import asym_id_to_chain_name

    asym_cpu = asym_id.cpu()
    token_chain_ids = [asym_id_to_chain_name(int(a)) for a in asym_cpu]
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

        # pTM and ipTM: normalize the j-sum over frame-bearing tokens
        # (tm_mask/tm_pair_mask), but size d0 on all valid tokens (seq_mask/
        # pair_mask), matching AF3 confidence_head._get_tmscore_adjusted_pae.
        pae_logits_s = output.pae[sample_idx]
        ptm = compute_ptm(pae_logits_s, tm_mask, tm_pair_mask, max_error_bin, num_bins, d0_seq_mask=seq_mask)
        iptm = compute_iptm(
            pae_logits_s, tm_mask, tm_pair_mask, asym_id, max_error_bin, num_bins, d0_seq_mask=seq_mask
        )
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

    # Distogram (per-pair expected distance) and contact probabilities
    if output.distogram is not None:
        distogram_expected = compute_distogram_expected(output.distogram)
        result["distogram"] = [[round(float(v), 2) for v in row] for row in distogram_expected.cpu()]
        contact_probs = compute_contact_probs(output.distogram)
        result["contact_probs"] = [[round(float(v), 3) for v in row] for row in contact_probs.cpu()]

    # Clash detection
    polymer_mask = features["frames_mask"].to(device) if "frames_mask" in features else seq_mask.bool()
    sample_clash = has_clash(
        output.atom_positions[sample_idx],
        atom_mask,
        features["residue_index"].to(device),
        asym_id,
        polymer_mask,
    )

    # Fraction disordered
    if output.plddt is not None:
        frac_disorder = fraction_disordered(per_token_plddt[sample_idx].to(device), polymer_mask)
    else:
        frac_disorder = 0.0

    # Ranking score
    ranking_score = compute_ranking_score(ptm, iptm, frac_disorder, sample_clash)

    # Number of unique chains
    num_chains = int(asym_cpu.unique().numel())

    summary = {
        "ptm": round(ptm, 4),
        "iptm": round(iptm, 4) if not math.isnan(iptm) else None,
        "ranking_score": round(ranking_score, 4),
        "mean_plddt": round(mean_plddt, 2),
        "average_pde": round(average_pde, 2),
        "fraction_disordered": round(frac_disorder, 4),
        "has_clash": sample_clash,
        "num_tokens": N,
        "num_chains": num_chains,
        "seed": seed,
    }
    result["summary"] = summary

    # Write JSON (sanitize NaN/Inf → null for valid JSON)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_sanitize_for_json(result), indent=2))

    return summary


def save_ranking_scores_csv(path: str | Path, summaries: list[dict]) -> None:
    """Write ``ranking_scores.csv`` for a list of per-(seed, sample) summaries.

    Each summary dict should be the value returned by :func:`save_confidences_json`
    augmented with ``seed`` and ``sample`` keys. Rows are sorted by
    ``ranking_score`` descending. The first three columns
    (``seed, sample, ranking_score``) match the AlphaFold 3 output layout;
    extra columns are appended for richer downstream analysis.
    """
    import csv

    path = Path(path)
    sorted_rows = sorted(summaries, key=lambda r: r["ranking_score"], reverse=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "sample", "ranking_score", "ptm", "iptm", "fraction_disordered", "has_clash"])
        for row in sorted_rows:
            iptm = row.get("iptm")
            iptm_str = "" if iptm is None or (isinstance(iptm, float) and math.isnan(iptm)) else f"{iptm:.4f}"
            writer.writerow(
                [
                    row["seed"],
                    row["sample"],
                    f"{row['ranking_score']:.4f}",
                    f"{row['ptm']:.4f}",
                    iptm_str,
                    f"{row['fraction_disordered']:.4f}",
                    int(row["has_clash"]),
                ]
            )


def save_summary_confidences_json(path: str | Path, summaries: list[dict], job_name: str | None = None) -> None:
    """Write a top-level ``summary_confidences.json`` aggregating all predictions.

    Each entry in ``summaries`` is the value returned by
    :func:`save_confidences_json` augmented with ``seed`` and ``sample``.
    The output payload contains ``job_name``, ``num_predictions``,
    ``best`` (highest-ranking entry), and ``predictions`` (sorted descending).
    """
    path = Path(path)
    sorted_entries = sorted(summaries, key=lambda e: e["ranking_score"], reverse=True)
    payload = {
        "job_name": job_name,
        "num_predictions": len(sorted_entries),
        "best": sorted_entries[0] if sorted_entries else None,
        "predictions": sorted_entries,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_sanitize_for_json(payload), indent=2))
