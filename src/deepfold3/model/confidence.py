"""Confidence head: _embed_features + 4-layer PairFormer → pLDDT, PAE, PDE, exp_resolved.

Haiku scope: diffuser~/confidence_head~/

The confidence pairformer uses full PairFormerIteration (with_single=True),
same as the trunk pairformer. Both pair and single tracks are updated.

Before the PairFormer, ``_embed_features`` adds to pair_act:
  1. Outer-sum of target_feat projections (left + right)
  2. Distogram from predicted representative-atom positions (39 bins)

After the PairFormer:
  - pLDDT:  LayerNorm → Linear on single track  [S, N, M*50]
  - PAE:    LayerNorm → Linear on pair track     [S, N, N, 64]
  - PDE:    LayerNorm → symmetrised half-logits   [S, N, N, 64]
  - Experimentally resolved: LayerNorm → Linear on single track [S, N, M*2]

Attribute names match the Haiku → PyTorch converter mapping.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from deepfold3.model.config import ConfidenceConfig, OptimConfig
from deepfold3.model.embedders import DGRAM_NUM_BINS, dgram_from_positions
from deepfold3.model.evoformer import PairFormerBlock
from deepfold3.model.primitives import LayerNorm, LinearNoBias


class ConfidenceHead(nn.Module):
    """Shallow 4-layer PairFormer (with single track) for confidence prediction.

    Takes trunk pair + single representations, target_feat, and predicted atom
    positions. Outputs per-sample pLDDT and PAE.

    Haiku scope: diffuser~/confidence_head~/
    Layer scope: __layer_stack_no_per_layer/confidence_pairformer/

    Attribute names (for converter):
        layers   → layers (ModuleList of PairFormerBlock)
        plddt_logits_ln → plddt_logits_ln (LayerNorm on single before pLDDT)
        plddt_logits → plddt_logits (LinearNoBias, single → max_atoms*num_plddt_bins)
        pae_logits_ln → pae_logits_ln (LayerNorm on pair before PAE)
        pae_logits   → pae_logits   (LinearNoBias, pair → num_distogram_bins)
        pde_logits_ln → pde_logits_ln (LayerNorm on pair before PDE)
        pde_half_logits → pde_half_logits (LinearNoBias, pair → num_distogram_bins)
        exp_resolved_ln → exp_resolved_ln (LayerNorm on single)
        exp_resolved_logits → exp_resolved_logits (LinearNoBias, single → max_atoms*2)
        left_target_feat_project → left_target_feat_project (LinearNoBias)
        right_target_feat_project → right_target_feat_project (LinearNoBias)
        distogram_feat_project → distogram_feat_project (LinearNoBias)
    """

    def __init__(self, cfg: ConfidenceConfig, target_feat_dim: int = 447, optim: OptimConfig | None = None):
        super().__init__()
        c_pair = cfg.pair_channel
        c_single = cfg.seq_channel

        # _embed_features: target_feat outer-sum + distogram from positions
        self.left_target_feat_project = LinearNoBias(target_feat_dim, c_pair)
        self.right_target_feat_project = LinearNoBias(target_feat_dim, c_pair)
        self.distogram_feat_project = LinearNoBias(DGRAM_NUM_BINS, c_pair)

        # 4-layer PairFormer with single track (same as trunk)
        self.layers = nn.ModuleList(
            [
                PairFormerBlock(
                    c_pair=c_pair,
                    c_single=c_single,
                    num_pair_heads=cfg.pair_attention_num_head,
                    num_single_heads=cfg.single_attention_num_head,
                    single_key_dim=c_single,
                    single_value_dim=c_single,
                    optim=optim,
                )
                for _ in range(cfg.num_layer)
            ]
        )

        # Output heads: LayerNorm before logit projections (matching AF3)
        max_atoms = cfg.max_atoms_per_token

        self.plddt_logits_ln = LayerNorm(c_single)
        self.plddt_logits = LinearNoBias(c_single, max_atoms * cfg.num_plddt_bins)

        self.pae_logits_ln = LayerNorm(c_pair)
        self.pae_logits = LinearNoBias(c_pair, cfg.num_distogram_bins)

        # PDE head: symmetrised half-logits on pair track
        self.pde_logits_ln = LayerNorm(c_pair)
        self.pde_half_logits = LinearNoBias(c_pair, cfg.num_distogram_bins)

        # Experimentally resolved head: per-atom 2-class on single track
        self.exp_resolved_ln = LayerNorm(c_single)
        self.exp_resolved_logits = LinearNoBias(c_single, max_atoms * 2)

    def _embed_features(
        self,
        target_feat: torch.Tensor,  # [N, target_feat_dim]
        positions: torch.Tensor,  # [N, 3] representative atom positions
        pair_mask: torch.Tensor,  # [N, N]
    ) -> torch.Tensor:
        """Compute additional pair features from target_feat and positions.

        Returns tensor to add to pair_act [N, N, c_pair].
        """
        # Target feat outer sum: left[i] + right[j] → [..., N, N, c_pair]
        out = self.left_target_feat_project(target_feat).unsqueeze(-2)  # [..., N, 1, c_pair]
        out = out + self.right_target_feat_project(target_feat).unsqueeze(-3)  # [..., 1, N, c_pair]

        # Distogram from predicted positions
        dgram = dgram_from_positions(positions)  # [N, N, num_bins]
        dgram = dgram * pair_mask[..., None]

        out = out + self.distogram_feat_project(dgram)

        return out

    def forward(
        self,
        pair: torch.Tensor,  # [N, N, c_pair]
        single: torch.Tensor,  # [N, c_single]
        pair_mask: torch.Tensor,  # [N, N]
        seq_mask: torch.Tensor,  # [N]
        target_feat: torch.Tensor,  # [N, target_feat_dim]
        atom_positions: torch.Tensor,  # [num_samples, N, M, 3]
        rep_atom_index: torch.Tensor | None = None,  # [N] index of representative atom per token
        return_pre_logits: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Run confidence head per diffusion sample.

        Args:
            return_pre_logits: If True, return post-LN features instead of
                projected logits for pair-scale heads (PAE, PDE).  Used with
                fused chunked linear+CE to avoid materializing [S,N,N,64] logits.
                pLDDT and exp_resolved always return logits (small tensors).

        Returns:
            When ``return_pre_logits=False`` (default):
                plddt_logits:        [num_samples, N, max_atoms * num_plddt_bins]
                pae_logits:          [num_samples, N, N, num_distogram_bins]
                pde_logits:          [num_samples, N, N, num_distogram_bins]
                exp_resolved_logits: [num_samples, N, max_atoms * 2]
            When ``return_pre_logits=True``:
                pae_logits → pae_features [num_samples, N, N, pair_channel]
                pde_logits → pde_features [num_samples, N, N, pair_channel]
        """
        num_samples = atom_positions.shape[0]
        N = pair.shape[-3]

        # Default: use atom index 1 (Cα for proteins) as representative
        if rep_atom_index is None:
            rep_atom_index = torch.ones(N, dtype=torch.long, device=pair.device)

        plddt_list = []
        pae_list = []
        pde_list = []
        exp_resolved_list = []

        for s in range(num_samples):
            # Extract representative atom positions for this sample
            positions = atom_positions[s, torch.arange(N, device=pair.device), rep_atom_index]  # [N, 3]

            # Start from trunk embeddings (shared across samples)
            pair_act = pair.clone()
            single_act = single.clone()

            # Add position-dependent features
            pair_act = pair_act + self._embed_features(target_feat, positions, pair_mask)

            # Run confidence PairFormer
            for layer in self.layers:
                pair_act, single_act = layer(pair_act, single_act, pair_mask, seq_mask)

            pair_act = pair_act.float()
            single_act = single_act.float()

            plddt_list.append(self.plddt_logits(self.plddt_logits_ln(single_act)))
            exp_resolved_list.append(self.exp_resolved_logits(self.exp_resolved_ln(single_act)))

            if return_pre_logits:
                # Return post-LN features for fused chunked loss (no logit materialization)
                pae_list.append(self.pae_logits_ln(pair_act))
                pde_list.append(self.pde_logits_ln(pair_act))
            else:
                pae_list.append(self.pae_logits(self.pae_logits_ln(pair_act)))
                # PDE: symmetrised half-logits
                pde_half = self.pde_half_logits(self.pde_logits_ln(pair_act))
                pde_list.append(pde_half + pde_half.transpose(-3, -2))

        result = {
            "plddt_logits": torch.stack(plddt_list, dim=0),
            "exp_resolved_logits": torch.stack(exp_resolved_list, dim=0),
        }
        if return_pre_logits:
            result["pae_features"] = torch.stack(pae_list, dim=0)
            result["pde_features"] = torch.stack(pde_list, dim=0)
        else:
            result["pae_logits"] = torch.stack(pae_list, dim=0)
            result["pde_logits"] = torch.stack(pde_list, dim=0)
        return result
