"""Top-level DeepFold3 model.

Assembles the full pipeline:
    EvoformerConditioningEncoder → raw_target_feat concat → Evoformer trunk
    [recycles] → DiffusionHead → ConfidenceHead

target_feat = concat(raw_feat[N,63], evo_cond_token_act[N,384]) → [N,447]
msa_feat: [S, N, 34]  (one_hot(32) + has_deletion + deletion_value)
"""

from __future__ import annotations

import dataclasses
from functools import partial as _partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _torch_checkpoint

from deepfold3.model.atom_cross_attn import EvoformerConditioningEncoder
from deepfold3.model.confidence import ConfidenceHead
from deepfold3.model.config import DistogramConfig, ModelConfig, OptimConfig
from deepfold3.model.diffusion import DiffusionHead
from deepfold3.model.embedders import compute_rel_feat_dim, create_relative_encoding
from deepfold3.model.evoformer import Evoformer
from deepfold3.model.precision import fp32_scope
from deepfold3.model.primitives import LinearNoBias

# Disable the determinism metadata check that compares saved vs recomputed
# tensor shapes.  View reshapes inside checkpointed functions (e.g.
# [1,384,384,128] → [147456,128]) trigger a spurious mismatch warning
# that floods the log.  The values are correct; only the view shape differs.
torch_checkpoint = _partial(_torch_checkpoint, use_reentrant=False, determinism_check="none")

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

    def __init__(
        self,
        pair_channels: int = 128,
        cfg: DistogramConfig | None = None,
        fp32: bool = False,
    ):
        super().__init__()
        if cfg is None:
            cfg = DistogramConfig()
        self.num_bins = cfg.num_bins
        self.first_break = cfg.first_break
        self.last_break = cfg.last_break
        self.fp32 = fp32
        # AF3: half_logits uses final_init (zeros)
        self.half_logits = LinearNoBias(pair_channels, self.num_bins, initializer="zeros")

    def forward(self, pair: torch.Tensor) -> torch.Tensor:
        """Symmetrised logits [..., N, N, num_bins]."""
        with fp32_scope(self.fp32):
            h = self.half_logits(pair.float() if self.fp32 else pair)
        return h + h.transpose(-3, -2)

    def expected_distance(self, pair: torch.Tensor) -> torch.Tensor:
        """Expected Cβ–Cβ distance in Å, shape [..., N, N]."""
        probs = self.forward(pair).softmax(dim=-1)
        device = pair.device
        breaks = torch.linspace(self.first_break, self.last_break, self.num_bins - 1, device=device)
        last_gap = breaks[-1] - breaks[-2]
        bin_lower = torch.cat([torch.zeros(1, device=device), breaks])
        bin_upper = torch.cat([breaks, breaks[-1:] + last_gap])
        bin_centers = (bin_lower + bin_upper) * 0.5
        return (probs * bin_centers).sum(-1)


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
            per_atom_channels=dc.atom_cross_att.per_atom_channels,
            per_token_channels=ec.seq_channel,
            per_atom_pair_channels=dc.atom_cross_att.per_atom_pair_channels,
            num_blocks=dc.atom_cross_att.atom_transformer.num_blocks,
            Q=dc.atom_cross_att.queries_subset_size,
            K=dc.atom_cross_att.keys_subset_size,
            num_element_classes=dc.atom_cross_att.num_element_classes,
            num_atom_name_chars=dc.atom_cross_att.num_atom_name_chars,
            atom_name_length=dc.atom_cross_att.atom_name_length,
        )

        # Evoformer trunk (with optional template embedding)
        self.evoformer = Evoformer(ec, pfc, template_cfg=cfg.template, optim=cfg.optim)

        # Diffusion head
        rel_feat_dim = compute_rel_feat_dim(ec.max_relative_idx, ec.max_relative_chain)
        self.diffusion_head = DiffusionHead(
            dc, c_single=ec.seq_channel, c_pair=ec.pair_channel, rel_feat_dim=rel_feat_dim, optim=cfg.optim
        )

        # Confidence head
        self.confidence_head = ConfidenceHead(cc, optim=cfg.optim, fp32=cfg.precision.fp32_confidence_head)

        # Distogram head
        self.distogram_head = DistogramHead(ec.pair_channel, cfg=cfg.distogram, fp32=cfg.precision.fp32_distogram_head)

    def state_dict(self, *args, **kwargs):
        """Return state dict without fused kernel parameters."""
        markers = OptimConfig.FUSED_STATE_DICT_MARKERS
        full = super().state_dict(*args, **kwargs)
        return {k: v for k, v in full.items() if not any(m in k for m in markers)}

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Load state dict and sync any fused kernel parameters."""
        # Fused kernel params are derived — relax strict if none appear in the incoming dict.
        markers = OptimConfig.FUSED_STATE_DICT_MARKERS
        if strict:
            has_fused = any(any(m in k for m in markers) for k in state_dict)
            if not has_fused:
                strict = False
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        self._sync_fused_kernels()
        return result

    def _sync_fused_kernels(self):
        """Sync all fused kernel parameters from their base module weights.

        Fused Triton modules hold copies of weights from their base (unfused)
        counterparts. When the top-level model.load_state_dict() loads weights,
        only the base params are updated — fused copies become stale. This
        method walks all modules and re-syncs every fused copy.
        """
        for mod in self.modules():
            # TriangleMul: fused gate (LN+packed GEMM), dual GEMM, center proj (LN+Linear)
            if hasattr(mod, "_fused_gate") and mod._fused_gate is not None:
                mod._sync_to_fused_gate()
            if hasattr(mod, "_fused_dual_gemm") and mod._fused_dual_gemm is not None:
                mod._sync_dual_gemm()
            if hasattr(mod, "_fused_center_proj") and mod._fused_center_proj is not None:
                mod._sync_center_proj()
            # AdaTransitionBlock: fused AdaLN+Linear
            if hasattr(mod, "_fused_ada_ln_linear") and mod._fused_ada_ln_linear is not None:
                mod._sync_ffn_to_fused()
            # FusedTransitionBlock: fused LN+Linear
            if hasattr(mod, "_sync_to_fused") and hasattr(mod, "_ln_linear") and mod._ln_linear is not None:
                mod._sync_to_fused()
            # GridSelfAttention: fused LN+transpose, fused pair bias (LN+proj+permute)
            if hasattr(mod, "_ln_transpose") and mod._ln_transpose is not None:
                mod._sync_ln_transpose()
            if hasattr(mod, "_fused_pair_bias") and mod._fused_pair_bias is not None:
                mod._sync_pair_bias()

    def run_evoformer(
        self,
        features: dict[str, torch.Tensor],
        num_recycles: int,
        seed: int = 0,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run feature building + evoformer trunk with recycling.

        Supports both unbatched ``[N, ...]`` and batched ``[B, N, ...]`` features.

        For each recycle, a fresh random subset of MSA rows is sampled
        (``msa_crop_size`` rows, always including the query row 0).
        This follows AF3 §3.3 (Algorithm 8, line 2).

        Returns:
            (embeddings, rel_feat, target_feat, seq_mask, pair_mask)
        """
        device = features["res_type"].device
        has_batch = features["res_type"].dim() >= 2
        N = features["res_type"].shape[-1]

        # Build fallback tensors matching input shape
        token_index = features.get("token_index")
        if token_index is None:
            token_index = torch.arange(N, dtype=torch.long, device=device)
            if has_batch:
                token_index = token_index.unsqueeze(0).expand_as(features["residue_index"])
        sym_id = features.get("sym_id")
        if sym_id is None:
            sym_id = torch.zeros_like(features["residue_index"])

        # Build relative position features
        rel_feat = create_relative_encoding(
            token_index=token_index,
            residue_index=features["residue_index"],
            asym_id=features["asym_id"],
            entity_id=features["entity_id"],
            sym_id=sym_id,
            max_relative_idx=self.cfg.evoformer.max_relative_idx,
            max_relative_chain=self.cfg.evoformer.max_relative_chain,
        )

        # Build raw target_feat [..., N, 63]
        raw_feat = self._build_raw_target_feat(features)

        # Evoformer conditioning: ref atoms → token_act (natively batched).
        ref_pos = features["ref_pos"]
        ref_mask = features["ref_mask"].float()
        ref_element = features["ref_element"]
        ref_charge = features["ref_charge"].float()
        ref_atom_name_chars = features["ref_atom_name_chars"]

        use_ckpt = torch.is_grad_enabled()
        if use_ckpt:
            evo_cond_act = torch_checkpoint(
                self.evo_cond_encoder,
                ref_pos,
                ref_mask,
                ref_element,
                ref_charge,
                ref_atom_name_chars,
            )
        else:
            evo_cond_act = self.evo_cond_encoder(
                ref_pos=ref_pos,
                ref_mask=ref_mask,
                ref_element=ref_element,
                ref_charge=ref_charge,
                ref_atom_name_chars=ref_atom_name_chars,
            )

        # Full target_feat [..., N, 447] = concat(raw[63], evo_cond[384])
        target_feat = torch.cat([raw_feat, evo_cond_act], dim=-1)

        seq_mask = features["seq_mask"].float()
        pair_mask = features["pair_mask"].float()

        # Full MSA tensors [..., S_full, N]
        full_msa = features["msa"]
        full_msa_mask = features["msa_mask"]
        full_del_matrix = features.get("deletion_matrix")

        # Initialize recycling buffers (or resume from features if warm-started)
        c_pair = self.cfg.evoformer.pair_channel
        c_single = self.cfg.evoformer.seq_channel
        # Match dtype of model parameters to avoid mixed-precision mismatch under autocast
        param_dtype = target_feat.dtype
        if "_prev_pair" in features:
            prev_pair = features["_prev_pair"]
            prev_single = features["_prev_single"]
        elif has_batch:
            B = features["res_type"].shape[0]
            prev_pair = torch.zeros(B, N, N, c_pair, device=device, dtype=param_dtype)
            prev_single = torch.zeros(B, N, c_single, device=device, dtype=param_dtype)
        else:
            prev_pair = torch.zeros(N, N, c_pair, device=device, dtype=param_dtype)
            prev_single = torch.zeros(N, c_single, device=device, dtype=param_dtype)

        # Extract template features (optional — T=0 tensors when no templates)
        template_kwargs: dict[str, torch.Tensor] = {}
        if "template_aatype" in features:
            template_kwargs["template_aatype"] = features["template_aatype"]
            template_kwargs["template_pseudo_beta"] = features["template_pseudo_beta"].float()
            template_kwargs["template_pseudo_beta_mask"] = features["template_pseudo_beta_mask"].float()
            template_kwargs["template_backbone_frame"] = features["template_backbone_frame"].float()
            template_kwargs["template_backbone_mask"] = features["template_backbone_mask"].float()

        # Bond matrix: polymer-ligand and ligand-ligand covalent bonds [..., N, N, 1]
        bond_matrix = features.get("bond_matrix")
        if bond_matrix is not None:
            bond_matrix = bond_matrix.to(device).float()

        # Multichain mask: same-chain pairs for template masking [..., N, N]
        asym_id_tensor = features["asym_id"].to(device)
        multichain_mask = (asym_id_tensor.unsqueeze(-1) == asym_id_tensor.unsqueeze(-2)).float()

        # Per-recycle MSA subsampling (AF3 §3.3, Algorithm 8)
        S_full = full_msa.shape[-2]  # MSA rows dimension
        crop = self.cfg.msa_crop_size
        rng = torch.Generator(device="cpu").manual_seed(seed)
        msa_rows_dim = full_msa.dim() - 2  # 0 for [S,N], 1 for [B,S,N]

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
                idx = torch.cat([torch.zeros(1, dtype=torch.long), sel]).sort().values.to(device)
                msa_sub = full_msa.index_select(msa_rows_dim, idx)
                msa_mask_sub = full_msa_mask.index_select(msa_rows_dim, idx)
                del_sub = full_del_matrix.index_select(msa_rows_dim, idx) if full_del_matrix is not None else None

            # Build MSA features for this recycle [..., S_crop, N, 34]
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

    @staticmethod
    def _ref_kwargs(features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Extract ref atom tensors needed by diffusion head."""
        return {
            "ref_pos": features["ref_pos"],
            "ref_mask": features["ref_mask"].float(),
            "ref_element": features["ref_element"],
            "ref_charge": features["ref_charge"].float(),
            "ref_atom_name_chars": features["ref_atom_name_chars"],
        }

    def forward_training(
        self,
        features: dict[str, torch.Tensor],
        num_recycles: int,
        diffusion_batch_size: int,
        mini_rollout_steps: int = 20,
        mini_rollout_samples: int = 1,
        seed: int = 0,
        fused_ce: bool = False,
        diffusion_chunk_size: int = 0,
        backward_fn=None,
        loss_cfg=None,
    ) -> dict[str, torch.Tensor]:
        """Training forward pass with recycling, diffusion, confidence, and distogram.

        Encapsulates the full training pipeline:

        1. Evoformer trunk with recycling (gradient only on last cycle).
        2. Single-step diffusion training (augment GT, add noise, denoise).
        3. Mini-rollout (no gradient, reduced steps/samples).
        4. Confidence head on mini-rollout positions.
        5. Distogram head.

        Args:
            features: Batched feature dict from the dataloader.
            num_recycles: Number of recycles (gradient flows through last only).
            diffusion_batch_size: Number of noise samples for diffusion training.
            mini_rollout_steps: Denoising steps for the confidence mini-rollout.
            mini_rollout_samples: Diffusion samples in the mini-rollout.
            seed: Random seed for MSA subsampling.
            fused_ce: Whether to defer cross-entropy projection (for fused loss).
            diffusion_chunk_size: If >0 and divides ``diffusion_batch_size``,
                run diffusion in chunks of this size, computing the per-chunk
                loss and back-propping it via ``backward_fn`` before the next
                chunk's forward. Reduces peak memory by ~``diffusion_batch_size
                / diffusion_chunk_size``×. Requires ``backward_fn`` and
                ``loss_cfg`` to be set.
            backward_fn: Callable ``(loss: Tensor) -> None`` used by the
                chunked path to back-prop each chunk's diffusion loss. The
                trainer wraps ``self.scaler.scale(loss / accumulate).backward``.
            loss_cfg: ``TrainingConfig`` instance — needed by the chunked path
                to compute per-chunk diffusion losses via ``diffusion_only_loss``.

        Returns:
            Dict. In the unchunked path, contains ``denoised``, ``gt_augmented``,
            ``noise_levels`` plus the confidence/distogram outputs. In the
            chunked path, the diffusion-loss components are stashed under
            ``__diffusion_logged__`` (already back-propped); ``denoised`` etc.
            are not returned.
        """
        ref_kwargs = self._ref_kwargs(features)

        # --- 1. Evoformer trunk (gradient only on the FINAL pass) ---
        # AF3 convention: ``num_recycles`` is the number of *additional*
        # trunk passes after the first. Total trunk forwards = num_recycles + 1.
        #   num_recycles=0 → 1 pass (just initial, no recycling)
        #   num_recycles=N → N warmup passes (no_grad) + 1 grad pass
        # Warmup passes update ``prev_pair``/``prev_single`` to seed the
        # final grad pass.
        if num_recycles > 0:
            with torch.no_grad():
                embeddings_det, _, _, _, _ = self.run_evoformer(features, num_recycles - 1, seed=seed)
            # Pass warm-start buffers via features for the final (grad-enabled) pass.
            # No .detach() needed — tensors from no_grad() context already lack grad_fn.
            features["_prev_pair"] = embeddings_det["pair"]
            features["_prev_single"] = embeddings_det["single"]

        embeddings, rel_feat, target_feat, seq_mask, pair_mask = self.run_evoformer(
            features, 0, seed=seed + num_recycles
        )
        # Clean up side-channel keys to avoid stale references
        features.pop("_prev_pair", None)
        features.pop("_prev_single", None)
        embeddings["target_feat"] = target_feat

        pred: dict[str, torch.Tensor] = {}
        # Expose the attached single / pair representations for the magnitude
        # regularizers / observers in ``training_loss``.  The reference alone
        # does not extend the autograd graph; only the loss call does.
        pred["single"] = embeddings["single"]
        pred["pair"] = embeddings["pair"]

        # --- 2. Diffusion training step ---
        chunked = (
            diffusion_chunk_size > 0
            and backward_fn is not None
            and loss_cfg is not None
            and diffusion_batch_size % diffusion_chunk_size == 0
        )
        if chunked:
            # Detach trunk outputs into leaves so each chunk's backward can
            # run with ``retain_graph=False`` and free its activations
            # without dragging the (much larger) trunk graph with it. After
            # the chunk loop, accumulated leaf grads are flowed back through
            # the trunk in one walk. ``retain_graph=True`` keeps the trunk
            # graph alive for the distogram backward (which still uses the
            # attached ``embeddings["pair"]``).
            from deepfold3.training.loss import diffusion_only_loss

            pair_real = embeddings["pair"]
            single_real = embeddings["single"]
            target_feat_real = target_feat

            pair_det = pair_real.detach().requires_grad_(True)
            single_det = single_real.detach().requires_grad_(True)
            target_feat_det = target_feat_real.detach().requires_grad_(True)

            emb_det = {
                "pair": pair_det,
                "single": single_det,
                "target_feat": target_feat_det,
            }

            num_chunks = diffusion_batch_size // diffusion_chunk_size
            logged: dict[str, torch.Tensor] = {}
            weighted_total: torch.Tensor = pair_real.new_zeros(())
            # ``diffusion_only_loss`` assumes unbatched ``[N, ...]`` features.
            # Always slice per batch element (batch dim is always kept).
            B_pair = pair_real.shape[0]  # [B, N, N, C]

            for _ in range(num_chunks):
                gt_aug_c, denoised_c, noise_levels_c = self.diffusion_head.sample_training(
                    gt_positions=features["atom_gt_positions"],
                    gt_mask=features["atom_gt_mask"],
                    embeddings=emb_det,
                    seq_mask=seq_mask,
                    rel_feat=rel_feat,
                    num_samples=diffusion_chunk_size,
                    **ref_kwargs,
                )
                # chunk_pred shapes:
                #   denoised, gt_augmented: [B, S_chunk, N, M, 3]
                #   noise_levels: [B, S_chunk]
                chunk_loss = pair_real.new_zeros(())
                chunk_components: dict[str, torch.Tensor] = {}
                for b in range(B_pair):
                    chunk_pred_b = {
                        "denoised": denoised_c[b],
                        "gt_augmented": gt_aug_c[b],
                        "noise_levels": noise_levels_c[b],
                    }
                    features_b = {
                        k: (v[b] if isinstance(v, torch.Tensor) and v.dim() >= 1 and v.shape[0] == B_pair else v)
                        for k, v in features.items()
                    }
                    loss_b, comp_b = diffusion_only_loss(chunk_pred_b, features_b, loss_cfg)
                    chunk_loss = chunk_loss + loss_b / B_pair
                    for k, v in comp_b.items():
                        chunk_components[k] = chunk_components.get(k, v.new_zeros(())) + v / B_pair
                # Each chunk contributes 1/num_chunks of the full diffusion
                # loss; the accumulated grads on emb_det / diffusion params
                # equal the unchunked equivalent.
                backward_fn(chunk_loss / num_chunks)
                weighted_total = weighted_total + chunk_loss.detach() / num_chunks
                for k, v in chunk_components.items():
                    logged[k] = logged.get(k, v.new_zeros(())) + v / num_chunks

            torch.autograd.backward(
                tensors=[pair_real, single_real, target_feat_real],
                grad_tensors=[pair_det.grad, single_det.grad, target_feat_det.grad],
                retain_graph=True,
            )

            pred["__diffusion_total__"] = weighted_total
            pred["__diffusion_logged__"] = logged
        else:
            gt_aug, denoised, noise_levels = self.diffusion_head.sample_training(
                gt_positions=features["atom_gt_positions"],
                gt_mask=features["atom_gt_mask"],
                embeddings=embeddings,
                seq_mask=seq_mask,
                rel_feat=rel_feat,
                num_samples=diffusion_batch_size,
                **ref_kwargs,
            )
            pred["denoised"] = denoised
            pred["gt_augmented"] = gt_aug
            pred["noise_levels"] = noise_levels

        # --- 3. Mini-rollout (no gradient) ---
        with torch.no_grad():
            diff_cfg = self.diffusion_head.cfg
            orig_steps, orig_samples = diff_cfg.steps, diff_cfg.num_samples
            diff_cfg.steps = mini_rollout_steps
            diff_cfg.num_samples = mini_rollout_samples
            try:
                mini_result = self.diffusion_head.sample(
                    embeddings=embeddings, seq_mask=seq_mask, rel_feat=rel_feat, **ref_kwargs
                )
            finally:
                diff_cfg.steps, diff_cfg.num_samples = orig_steps, orig_samples

        pred["mini_rollout_pos"] = mini_result["atom_positions"].detach()

        # --- 4. Confidence head ---
        # AF3 §3.7: confidence head trains over fixed (detached) trunk
        # outputs and the (already-detached) mini-rollout positions, so
        # confidence-loss gradients update only confidence-head parameters.
        # ConfidenceHead supports batched ``[B?, N, ...]`` inputs natively
        # (PairFormer + projection ops are all batch-aware) — pass straight
        # through whether ``B == 1`` or ``B > 1``.
        conf_out = self.confidence_head(
            pair=embeddings["pair"].detach(),
            single=embeddings["single"].detach(),
            pair_mask=pair_mask,
            seq_mask=seq_mask,
            target_feat=target_feat.detach(),
            atom_positions=pred["mini_rollout_pos"],
            rep_atom_index=features.get("token_to_rep_atom"),
            return_pre_logits=fused_ce,
        )

        pred["plddt"] = conf_out["plddt_logits"]
        pred["exp_resolved"] = conf_out["exp_resolved_logits"]

        if fused_ce:
            # Return pre-projection features + weight refs for chunked linear+CE
            pred["pae_features"] = conf_out["pae_features"]
            pred["pde_features"] = conf_out["pde_features"]
            pred["pae_weight"] = self.confidence_head.pae_logits.weight
            pred["pde_weight"] = self.confidence_head.pde_half_logits.weight
            pred["distogram_features"] = embeddings["pair"]
            pred["distogram_weight"] = self.distogram_head.half_logits.weight
        else:
            pred["pae"] = conf_out["pae_logits"]
            pred["pde"] = conf_out["pde_logits"]
            pred["distogram"] = self.distogram_head(embeddings["pair"])

        return pred

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
            embeddings, rel_feat, target_feat, seq_mask, pair_mask = self.run_evoformer(
                features, num_recycles, seed=seed
            )

            # Diffusion sampling
            device = features["res_type"].device
            generator = torch.Generator(device=device).manual_seed(seed)

            sample_result = self.diffusion_head.sample(
                embeddings=embeddings,
                seq_mask=seq_mask,
                rel_feat=rel_feat,
                generator=generator,
                **self._ref_kwargs(features),
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
            feat_parts.append(torch.zeros(*features["res_type"].shape, 1, device=features["res_type"].device))

        return torch.cat(feat_parts, dim=-1)  # [N, 63]

    def _build_msa_feat(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        """Build MSA feature tensor [S, N, 34] from the full features dict."""
        return self._build_msa_feat_from_tensors(features["msa"], features.get("deletion_matrix"))

    def _build_msa_feat_from_tensors(
        self,
        msa: torch.Tensor,
        deletion_matrix: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build MSA feature tensor [..., S, N, 34].

        Matches AF3 create_msa_feat:
            msa_1hot(32) + has_deletion(1) + deletion_value(1) = 34

        Supports both unbatched [S, N] and batched [B, S, N] input.
        """
        msa_1hot = F.one_hot(msa, _MSA_NUM_CLASSES).float()  # [..., S, N, 32]

        parts = [msa_1hot]
        if deletion_matrix is not None:
            del_mat = deletion_matrix.float()
            has_del = del_mat.clamp(0.0, 1.0).unsqueeze(-1)
            del_val = (torch.arctan(del_mat / 3.0) * (2.0 / torch.pi)).unsqueeze(-1)
            parts.append(has_del)
            parts.append(del_val)
        else:
            parts.append(torch.zeros(*msa.shape, 2, device=msa.device))

        return torch.cat(parts, dim=-1)

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

        embeddings, _, _, _, _ = self.run_evoformer(features, num_recycles)
        return self.distogram_head(embeddings["pair"])  # [N, N, 64]
