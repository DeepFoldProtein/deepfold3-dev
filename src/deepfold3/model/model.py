"""Top-level DeepFold3 model.

Assembles the full pipeline:
    EvoformerConditioningEncoder → raw_target_feat concat → Evoformer trunk
    [recycles] → DiffusionHead → ConfidenceHead

target_feat = concat(raw_feat[N,63], evo_cond_token_act[N,384]) → [N,447]
msa_feat: [S, N, 34]  (one_hot(32) + has_deletion + deletion_value)
"""

from __future__ import annotations

import dataclasses

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepfold3.model.atom_cross_attn import EvoformerConditioningEncoder
from deepfold3.model.confidence import ConfidenceHead
from deepfold3.model.config import ModelConfig
from deepfold3.model.diffusion import DiffusionHead
from deepfold3.model.embedders import create_relative_encoding
from deepfold3.model.evoformer import Evoformer
from deepfold3.model.primitives import LinearNoBias

# Raw target_feat dimensions (before evoformer conditioning)
# aatype_1hot(31) + profile(31) + deletion_mean(1) = 63
_RAW_TARGET_FEAT_DIM = 63
# aatype token classes (31 polymer types including unknown and gap)
_AATYPE_NUM_CLASSES = 31
_MSA_NUM_CLASSES = 32  # POLYMER_TYPES_NUM_WITH_UNKNOWN_AND_GAP + 1 (gap)


@dataclasses.dataclass
class DeepFold3Output:
    atom_positions: torch.Tensor  # [num_samples, N, M, 3]
    atom_mask: torch.Tensor  # [N, M]
    plddt: torch.Tensor | None  # [num_samples, N, M * num_plddt_bins]
    pae: torch.Tensor | None  # [num_samples, N, N, num_pae_bins]
    pde: torch.Tensor | None  # [num_samples, N, N, 64]
    exp_resolved: torch.Tensor | None  # [num_samples, N, M * 2]
    distogram: torch.Tensor | None  # [N, N, 64] logits


class DistogramHead(nn.Module):
    """Distogram head: pair activations → symmetrised distance bin logits.

    AF3 scope: diffuser/distogram_head/

    Attribute map:
        half_logits ← half_logits
    """

    NUM_BINS: int = 64
    FIRST_BREAK: float = 2.3125
    LAST_BREAK: float = 21.6875

    def __init__(self, pair_channels: int = 128):
        super().__init__()
        self.half_logits = LinearNoBias(pair_channels, self.NUM_BINS)

    def forward(self, pair: torch.Tensor) -> torch.Tensor:
        """Symmetrised logits [..., N, N, 64]."""
        h = self.half_logits(pair)  # [..., N, N, 64]
        return h + h.transpose(-3, -2)  # symmetrise

    def expected_distance(self, pair: torch.Tensor) -> torch.Tensor:
        """Expected Cβ–Cβ distance in Å, shape [..., N, N]."""
        probs = self.forward(pair).softmax(dim=-1)
        device = pair.device
        breaks = torch.linspace(self.FIRST_BREAK, self.LAST_BREAK, self.NUM_BINS - 1, device=device)
        last_gap = breaks[-1] - breaks[-2]
        bin_lower = torch.cat([torch.zeros(1, device=device), breaks])
        bin_upper = torch.cat([breaks, breaks[-1:] + last_gap])
        bin_centers = (bin_lower + bin_upper) * 0.5  # [64]
        return (probs * bin_centers).sum(-1)  # [..., N, N]


class DeepFold3(nn.Module):
    """Full AlphaFold3-equivalent model in PyTorch.

    Haiku root scope: diffuser~/
    """

    def __init__(self, cfg: ModelConfig | None = None):
        super().__init__()
        if cfg is None:
            cfg = ModelConfig()
        self.cfg = cfg

        ec = cfg.evoformer
        pfc = cfg.pairformer
        dc = cfg.diffusion
        cc = cfg.confidence

        # Evoformer conditioning encoder: ref atoms → token acts [N, 384]
        # Maps ref structure to the per-token contribution of target_feat
        self.evo_cond_encoder = EvoformerConditioningEncoder(
            per_atom_channels=128,
            per_token_channels=384,
            per_atom_pair_channels=16,
            num_blocks=3,
            Q=dc.atom_cross_att.queries_subset_size,
            K=dc.atom_cross_att.keys_subset_size,
        )

        # Evoformer trunk (with optional template embedding)
        self.evoformer = Evoformer(ec, pfc, template_cfg=cfg.template, optim=cfg.optim)

        # Diffusion head
        # rel_feat_dim = rel_pos(66) + rel_token(66) + entity_same(1) + rel_chain(6) = 139
        rel_feat_dim = 2 * (2 * ec.max_relative_idx + 2) + 1 + (2 * ec.max_relative_chain + 2)
        self.diffusion_head = DiffusionHead(
            dc, c_single=ec.seq_channel, c_pair=ec.pair_channel, rel_feat_dim=rel_feat_dim, optim=cfg.optim
        )

        # Confidence head
        self.confidence_head = ConfidenceHead(cc)

        # Distogram head
        self.distogram_head = DistogramHead(ec.pair_channel)

    def _run_evoformer(
        self,
        features: dict[str, torch.Tensor],
        num_recycles: int,
        seed: int = 0,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run feature building + evoformer trunk with recycling.

        For each recycle, a fresh random subset of MSA rows is sampled
        (``msa_crop_size`` rows, always including the query row 0).
        This follows AF3 paper (S2.3).

        Returns:
            (embeddings, rel_feat, target_feat, seq_mask, pair_mask)
        """
        device = features["res_type"].device
        N = features["res_type"].shape[0]

        # Build relative position features
        rel_feat = create_relative_encoding(
            token_index=features.get("token_index", torch.arange(N, dtype=torch.long, device=device)),
            residue_index=features["residue_index"],
            asym_id=features["asym_id"],
            entity_id=features["entity_id"],
            sym_id=features.get("sym_id", torch.zeros(N, dtype=torch.long, device=device)),
            max_relative_idx=self.cfg.evoformer.max_relative_idx,
            max_relative_chain=self.cfg.evoformer.max_relative_chain,
        )

        # Build raw target_feat [N, 63]
        raw_feat = self._build_raw_target_feat(features)

        # Evoformer conditioning: ref atoms → token_act [N, 384]
        evo_cond_act = self.evo_cond_encoder(
            ref_pos=features["ref_pos"],
            ref_mask=features["ref_mask"].float(),
            ref_element=features["ref_element"],
            ref_charge=features["ref_charge"].float(),
            ref_atom_name_chars=features["ref_atom_name_chars"],
        )

        # Full target_feat [N, 447] = concat(raw[63], evo_cond[384])
        target_feat = torch.cat([raw_feat, evo_cond_act], dim=-1)

        seq_mask = features["seq_mask"].float()
        pair_mask = features["pair_mask"].float()

        # Full MSA tensors (up to 16,384 rows)
        full_msa = features["msa"]  # [S_full, N]
        full_msa_mask = features["msa_mask"]  # [S_full, N]
        full_del_matrix = features.get("deletion_matrix")

        # Initialize recycling buffers
        c_pair = self.cfg.evoformer.pair_channel
        c_single = self.cfg.evoformer.seq_channel
        prev_pair = torch.zeros(N, N, c_pair, device=device)
        prev_single = torch.zeros(N, c_single, device=device)

        # Extract template features (optional — T=0 tensors when no templates)
        template_kwargs: dict[str, torch.Tensor] = {}
        if "template_aatype" in features:
            template_kwargs["template_aatype"] = features["template_aatype"]
            template_kwargs["template_pseudo_beta"] = features["template_pseudo_beta"].float()
            template_kwargs["template_pseudo_beta_mask"] = features["template_pseudo_beta_mask"].float()
            template_kwargs["template_backbone_frame"] = features["template_backbone_frame"].float()
            template_kwargs["template_backbone_mask"] = features["template_backbone_mask"].float()

        # Bond matrix: polymer-ligand and ligand-ligand covalent bonds [N, N, 1]
        bond_matrix = features.get("bond_matrix")
        if bond_matrix is not None:
            bond_matrix = bond_matrix.to(device).float()

        # Multichain mask: same-chain pairs for template masking [N, N]
        asym_id_tensor = features["asym_id"].to(device)
        multichain_mask = (asym_id_tensor.unsqueeze(-1) == asym_id_tensor.unsqueeze(-2)).float()

        # Per-recycle MSA subsampling (AF3 §2.5.4)
        S_full = full_msa.shape[0]
        crop = self.cfg.msa_crop_size
        rng = torch.Generator(device="cpu").manual_seed(seed)

        # Recycle
        for recycle_idx in range(num_recycles + 1):
            # Sample MSA rows: always include query (row 0), shuffle the rest
            if S_full <= crop:
                msa_sub = full_msa
                msa_mask_sub = full_msa_mask
                del_sub = full_del_matrix
            else:
                perm = torch.randperm(S_full - 1, generator=rng) + 1  # indices 1..S_full-1
                sel = perm[: crop - 1]
                idx = torch.cat([torch.zeros(1, dtype=torch.long), sel]).sort().values
                msa_sub = full_msa[idx]
                msa_mask_sub = full_msa_mask[idx]
                del_sub = full_del_matrix[idx] if full_del_matrix is not None else None

            # Build MSA features for this recycle [S_crop, N, 34]
            msa_feat = self._build_msa_feat_from_tensors(msa_sub, del_sub)
            msa_mask = msa_mask_sub.float()

            embeddings = self.evoformer(
                target_feat=target_feat,
                msa_feat=msa_feat,
                rel_feat=rel_feat,
                pair_mask=pair_mask,
                seq_mask=seq_mask,
                msa_mask=msa_mask,
                prev_pair=prev_pair,
                prev_single=prev_single,
                bond_matrix=bond_matrix,
                multichain_mask=multichain_mask,
                **template_kwargs,
            )
            if recycle_idx < num_recycles:
                prev_pair = embeddings["pair"].detach()
                prev_single = embeddings["single"].detach()

        return embeddings, rel_feat, target_feat, seq_mask, pair_mask

    def forward(
        self,
        features: dict[str, torch.Tensor],
        seed: int = 0,
        num_recycles: int | None = None,
        num_samples: int | None = None,
    ) -> DeepFold3Output:
        """Run full forward pass.

        Args:
            features: dict from featurizer with keys:
                token_index, residue_index, asym_id, entity_id, sym_id,
                mol_type, res_type, ref_pos, ref_mask, ref_element, ref_charge,
                ref_atom_name_chars, atom_to_token, token_to_rep_atom,
                msa, msa_mask, profile, deletion_mean, pair_mask, seq_mask
            seed: random seed for diffusion sampling
            num_recycles: override default number of recycles
            num_samples: override default number of diffusion samples
        """
        if num_recycles is None:
            num_recycles = self.cfg.num_recycles

        # Temporarily override num_samples if provided, restoring after
        orig_num_samples = self.diffusion_head.cfg.num_samples
        if num_samples is not None:
            self.diffusion_head.cfg.num_samples = num_samples

        try:
            embeddings, rel_feat, target_feat, seq_mask, pair_mask = self._run_evoformer(
                features, num_recycles, seed=seed
            )

            # Diffusion sampling
            device = features["res_type"].device
            generator = torch.Generator(device=device).manual_seed(seed)

            sample_result = self.diffusion_head.sample(
                embeddings=embeddings,
                ref_pos=features["ref_pos"],
                ref_mask=features["ref_mask"].float(),
                ref_element=features["ref_element"],
                ref_charge=features["ref_charge"].float(),
                ref_atom_name_chars=features["ref_atom_name_chars"],
                seq_mask=seq_mask,
                rel_feat=rel_feat,
                generator=generator,
            )

            # Confidence estimates (per-sample, matching AF3)
            conf_out = self.confidence_head(
                pair=embeddings["pair"],
                single=embeddings["single"],
                pair_mask=pair_mask,
                seq_mask=seq_mask,
                target_feat=target_feat,
                atom_positions=sample_result["atom_positions"],
                rep_atom_index=features.get("token_to_rep_atom"),
            )

            disto_logits = self.distogram_head(embeddings["pair"])

            return DeepFold3Output(
                atom_positions=sample_result["atom_positions"],
                atom_mask=sample_result["mask"],
                plddt=conf_out["plddt_logits"],
                pae=conf_out["pae_logits"],
                pde=conf_out["pde_logits"],
                exp_resolved=conf_out["exp_resolved_logits"],
                distogram=disto_logits,
            )
        finally:
            self.diffusion_head.cfg.num_samples = orig_num_samples

    def _build_raw_target_feat(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        """Build raw token feature vector [N, 63].

        Matches AF3 create_target_feat(batch, append_per_atom_features=False):
            aatype_1hot(31) + profile(31) + deletion_mean(1) = 63
        """
        feat_parts = []

        # aatype one-hot: 31 classes (POLYMER_TYPES_NUM_WITH_UNKNOWN_AND_GAP)
        aatype_1hot = F.one_hot(features["res_type"], _AATYPE_NUM_CLASSES).float()
        feat_parts.append(aatype_1hot)

        # Profile: per-position amino acid frequency (31-dim)
        if "profile" in features:
            prof = features["profile"].float()
            # Profile may be 22-dim (legacy) or 31-dim; handle both
            if prof.shape[-1] < _AATYPE_NUM_CLASSES:
                pad = torch.zeros(*prof.shape[:-1], _AATYPE_NUM_CLASSES - prof.shape[-1], device=prof.device)
                prof = torch.cat([prof, pad], dim=-1)
            feat_parts.append(prof[..., :_AATYPE_NUM_CLASSES])
        else:
            feat_parts.append(aatype_1hot)  # use one-hot as profile for single-seq

        # Deletion mean: [N, 1]
        if "deletion_mean" in features:
            feat_parts.append(features["deletion_mean"].float().unsqueeze(-1))
        else:
            feat_parts.append(torch.zeros(features["res_type"].shape[0], 1, device=features["res_type"].device))

        return torch.cat(feat_parts, dim=-1)  # [N, 63]

    def _build_msa_feat(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        """Build MSA feature tensor [S, N, 34] from the full features dict."""
        return self._build_msa_feat_from_tensors(features["msa"], features.get("deletion_matrix"))

    def _build_msa_feat_from_tensors(
        self,
        msa: torch.Tensor,
        deletion_matrix: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build MSA feature tensor [S, N, 34].

        Matches AF3 create_msa_feat:
            msa_1hot(32) + has_deletion(1) + deletion_value(1) = 34
        """
        S, N = msa.shape
        # one-hot over 32 classes (POLYMER_TYPES_NUM_WITH_UNKNOWN_AND_GAP + 1)
        msa_1hot = F.one_hot(msa, _MSA_NUM_CLASSES).float()  # [S, N, 32]

        parts = [msa_1hot]
        if deletion_matrix is not None:
            del_mat = deletion_matrix.float()  # [S, N]
            has_del = del_mat.clamp(0.0, 1.0).unsqueeze(-1)  # [S, N, 1]
            del_val = (torch.arctan(del_mat / 3.0) * (2.0 / torch.pi)).unsqueeze(-1)
            parts.append(has_del)
            parts.append(del_val)
        else:
            # No MSA: deletions are zero
            parts.append(torch.zeros(S, N, 2, device=msa.device))

        return torch.cat(parts, dim=-1)  # [S, N, 34]

    @torch.no_grad()
    def compute_distogram(
        self,
        features: dict[str, torch.Tensor],
        num_recycles: int | None = None,
    ) -> torch.Tensor:
        """Run evoformer + distogram head only (no diffusion).

        Fast validation path — useful for checking whether evoformer
        embeddings are correct without paying the cost of 200-step diffusion.

        Returns distogram logits [N, N, 64].
        """
        if num_recycles is None:
            num_recycles = self.cfg.num_recycles

        embeddings, _, _, _, _ = self._run_evoformer(features, num_recycles)
        return self.distogram_head(embeddings["pair"])  # [N, N, 64]
