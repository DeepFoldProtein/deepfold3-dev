"""Evoformer trunk: MSA stack + PairFormer stack.

Haiku scope hierarchy (separator '~/'):
    diffuser~/evoformer~/
        __layer_stack_no_per_layer/msa_stack/layer_{i}/
            msa_attention1~/        → row_attn
            outer_product_mean~/    → outer_product
            triangle_multiplication_outgoing~/  → tri_mul_out
            triangle_multiplication_incoming~/  → tri_mul_in
            pair_attention1~/       → pair_attn_row
            pair_attention2~/       → pair_attn_col
            pair_transition~/       → pair_transition
            msa_transition~/        → msa_transition
        __layer_stack_no_per_layer_1/trunk_pairformer/layer_{i}/
            triangle_multiplication_outgoing~/  → tri_mul_out
            triangle_multiplication_incoming~/  → tri_mul_in
            pair_attention1~/       → pair_attn_row
            pair_attention2~/       → pair_attn_col
            pair_transition~/       → pair_transition
            single_pair_logits_norm~/           → single_to_pair_norm (on pair act)
            single_pair_logits_projection~/     → single_to_pair
            single_attention_~/                 → single_attn
            single_transition~/                 → single_transition

Key dimension facts from AF3 weight file:
    left_single / right_single / single_activations / extra_msa_target_feat:
        input dim = 447  (= 63 raw target_feat + 384 evoformer_conditioning token_act)
    msa_activations:
        input dim = 34   (= 32 msa_1hot + 1 has_deletion + 1 deletion_value)
"""

from __future__ import annotations

import math
from functools import partial as _partial

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as _torch_checkpoint

from deepfold3.model.attention import GridSelfAttention, MSAAttention, SingleAttention
from deepfold3.model.config import EvoformerConfig, OptimConfig, PairFormerConfig, TemplateEmbeddingConfig
from deepfold3.model.ops import OuterProductMean, TriangleMulIncoming, TriangleMulOutgoing
from deepfold3.model.optim import make_transition_block
from deepfold3.model.primitives import LayerNorm, LinearNoBias, sum_fusion_init_
from deepfold3.model.template_embedding import TemplateEmbedding
from deepfold3.model.triton.dropout_add import dropout_add_columnwise, dropout_add_rowwise

_checkpoint = _partial(_torch_checkpoint, use_reentrant=False, determinism_check="none")

# Full target_feat dimension fed into the Evoformer (raw_feat + evo_cond_token_act)
_TARGET_FEAT_DIM = 447
# MSA feature dimension: one_hot(32) + has_deletion(1) + deletion_value(1)
_MSA_FEAT_DIM = 34


class EvoformerBlock(nn.Module):
    """Single layer of the Evoformer MSA stack.

    Haiku: EvoformerIteration, name='msa_stack'
    """

    def __init__(
        self,
        c_msa: int,
        c_pair: int,
        num_msa_heads: int = 8,
        num_pair_heads: int = 4,
        outer_product_hidden: int = 32,
        msa_dropout: float = 0.15,
        pair_dropout: float = 0.25,
        optim: OptimConfig | None = None,
    ):
        super().__init__()
        self.row_attn = MSAAttention(c_msa, c_pair, num_heads=num_msa_heads, optim=optim)
        self.outer_product = OuterProductMean(c_msa, c_pair, c_hidden=outer_product_hidden)
        if optim is not None and optim.tiled_opm_chunk > 0:
            self.outer_product._chunk_size = optim.tiled_opm_chunk
        self.tri_mul_out = TriangleMulOutgoing(c_pair, optim=optim)
        self.tri_mul_in = TriangleMulIncoming(c_pair, optim=optim)
        self.pair_attn_row = GridSelfAttention(c_pair, num_heads=num_pair_heads, transpose=False, optim=optim)
        self.pair_attn_col = GridSelfAttention(c_pair, num_heads=num_pair_heads, transpose=True, optim=optim)
        self.pair_transition = make_transition_block(c_pair, config=optim)
        self.msa_transition = make_transition_block(c_msa, config=optim)
        self.msa_dropout = msa_dropout
        self.pair_dropout = pair_dropout
        self.use_fused_dropout = optim.use_fused_dropout_add if optim is not None else True

    def forward(
        self,
        msa: torch.Tensor,  # [B?, S, N, c_msa]
        pair: torch.Tensor,  # [B?, N, N, c_pair]
        msa_mask: torch.Tensor,  # [B?, S, N]
        pair_mask: torch.Tensor,  # [B?, N, N]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        p_pair, p_msa, training = self.pair_dropout, self.msa_dropout, self.training
        use_fused = self.use_fused_dropout
        pair = pair + self.outer_product(msa, msa_mask)
        msa = dropout_add_rowwise(msa, self.row_attn(msa, msa_mask, pair), p_msa, training, use_fused)
        msa = msa + self.msa_transition(msa)
        pair = dropout_add_rowwise(pair, self.tri_mul_out(pair, pair_mask), p_pair, training, use_fused)
        pair = dropout_add_rowwise(pair, self.tri_mul_in(pair, pair_mask), p_pair, training, use_fused)
        pair = dropout_add_rowwise(pair, self.pair_attn_row(pair, pair_mask), p_pair, training, use_fused)
        pair = dropout_add_columnwise(pair, self.pair_attn_col(pair, pair_mask), p_pair, training, use_fused)
        pair = pair + self.pair_transition(pair)
        return msa, pair


class PairFormerBlock(nn.Module):
    """Single layer of the PairFormer trunk (with single track).

    Haiku: PairFormerIteration with_single=True, name='trunk_pairformer'

    The pair_logits for single attention come from the pair representation
    (not from single), following AF3: single_pair_logits_projection(single_pair_logits_norm(pair)).
    """

    def __init__(
        self,
        c_pair: int,
        c_single: int,
        num_pair_heads: int = 4,
        num_single_heads: int = 16,
        single_key_dim: int = 384,
        single_value_dim: int = 384,
        pair_dropout: float = 0.25,
        optim: OptimConfig | None = None,
        deepnet_layer_scale: int | None = None,
    ):
        super().__init__()
        self.tri_mul_out = TriangleMulOutgoing(c_pair, optim=optim)
        self.tri_mul_in = TriangleMulIncoming(c_pair, optim=optim)
        self.pair_attn_row = GridSelfAttention(c_pair, num_heads=num_pair_heads, transpose=False, optim=optim)
        self.pair_attn_col = GridSelfAttention(c_pair, num_heads=num_pair_heads, transpose=True, optim=optim)
        self.pair_transition = make_transition_block(c_pair, config=optim)
        # Single track
        self.single_to_pair_norm = LayerNorm(c_pair)  # normalizes pair act
        self.single_to_pair = LinearNoBias(c_pair, num_single_heads)  # pair → per-head logits
        self.single_attn = SingleAttention(
            c_single,
            num_heads=num_single_heads,
            key_dim=single_key_dim,
            value_dim=single_value_dim,
            optim=optim,
        )
        self.single_transition = make_transition_block(c_single, config=optim)
        self.pair_dropout = pair_dropout
        self.use_fused_dropout = optim.use_fused_dropout_add if optim is not None else True
        # DeepNet-style residual scaling on both pair and single tracks.
        # Output projections retain zero-init; only the forward residual is
        # scaled. Per user spec:
        #   tri_mul_out / tri_mul_in / single_attn / single_transition: 1/sqrt(2L)
        #   pair_attn_row / pair_attn_col / pair_transition:            1/sqrt(L)
        if deepnet_layer_scale is not None and deepnet_layer_scale > 0:
            L = float(deepnet_layer_scale)
            self._tri_residual_scale = 1.0 / math.sqrt(2.0 * L)
            self._pair_attn_residual_scale = 1.0 / math.sqrt(L)
            self._pair_ffn_residual_scale = 1.0 / math.sqrt(L)
            self._attn_residual_scale = 1.0 / math.sqrt(2.0 * L)
            self._ffn_residual_scale = 1.0 / math.sqrt(2.0 * L)
            self._deepnet_active = True
        else:
            self._tri_residual_scale = 1.0
            self._pair_attn_residual_scale = 1.0
            self._pair_ffn_residual_scale = 1.0
            self._attn_residual_scale = 1.0
            self._ffn_residual_scale = 1.0
            self._deepnet_active = False

    def forward(
        self,
        pair: torch.Tensor,  # [B?, N, N, c_pair]
        single: torch.Tensor,  # [B?, N, c_single]
        pair_mask: torch.Tensor,  # [B?, N, N]
        seq_mask: torch.Tensor,  # [B?, N]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        p, training = self.pair_dropout, self.training
        use_fused = self.use_fused_dropout
        if self._deepnet_active:
            s_tri = self._tri_residual_scale
            s_pa = self._pair_attn_residual_scale
            s_pf = self._pair_ffn_residual_scale
            pair = dropout_add_rowwise(pair, s_tri * self.tri_mul_out(pair, pair_mask), p, training, use_fused)
            pair = dropout_add_rowwise(pair, s_tri * self.tri_mul_in(pair, pair_mask), p, training, use_fused)
            pair = dropout_add_rowwise(pair, s_pa * self.pair_attn_row(pair, pair_mask), p, training, use_fused)
            pair = dropout_add_columnwise(pair, s_pa * self.pair_attn_col(pair, pair_mask), p, training, use_fused)
            pair = pair + s_pf * self.pair_transition(pair)
            # Single update: pair → logits, then single attention
            pair_logits = self.single_to_pair(self.single_to_pair_norm(pair))  # [..., N, N, H]
            pair_logits = pair_logits.movedim(-1, -3)  # [..., H, N, N]
            single = single + self._attn_residual_scale * self.single_attn(single, seq_mask, pair_logits)
            single = single + self._ffn_residual_scale * self.single_transition(single)
        else:
            pair = dropout_add_rowwise(pair, self.tri_mul_out(pair, pair_mask), p, training, use_fused)
            pair = dropout_add_rowwise(pair, self.tri_mul_in(pair, pair_mask), p, training, use_fused)
            pair = dropout_add_rowwise(pair, self.pair_attn_row(pair, pair_mask), p, training, use_fused)
            pair = dropout_add_columnwise(pair, self.pair_attn_col(pair, pair_mask), p, training, use_fused)
            pair = pair + self.pair_transition(pair)
            # Single update: pair → logits, then single attention
            pair_logits = self.single_to_pair(self.single_to_pair_norm(pair))  # [..., N, N, H]
            pair_logits = pair_logits.movedim(-1, -3)  # [..., H, N, N]
            single = single + self.single_attn(single, seq_mask, pair_logits)
            single = single + self.single_transition(single)
        return pair, single


class Evoformer(nn.Module):
    """Full Evoformer trunk: embedders + MSA stack + PairFormer stack.

    Handles recycled pair/single embeddings and the full Evoformer + PairFormer stacks.

    Haiku scope: diffuser~/evoformer~/

    Dimensions (from AF3 weight file):
        target_feat input: 447-dim (raw 63 + evo conditioning 384)
        msa_feat input: 34-dim (one_hot(32) + has_deletion + deletion_value)
    """

    def __init__(
        self,
        cfg: EvoformerConfig,
        pf_cfg: PairFormerConfig,
        template_cfg: TemplateEmbeddingConfig | None = None,
        optim: OptimConfig | None = None,
    ):
        super().__init__()
        self._grad_checkpoint = bool(optim and optim.use_gradient_checkpointing)
        c_msa = cfg.msa_channel
        c_pair = cfg.pair_channel
        c_single = cfg.seq_channel

        # Template embedding (optional — None if not using templates)
        if template_cfg is not None:
            self.template_embedding: TemplateEmbedding | None = TemplateEmbedding(template_cfg, optim=optim)
        else:
            self.template_embedding = None

        # Input embeddings for single and MSA — dimensions from AF3 weight file
        self.msa_activations = LinearNoBias(_MSA_FEAT_DIM, c_msa)
        self.extra_msa_target_feat = LinearNoBias(_TARGET_FEAT_DIM, c_msa)

        # Pair embeddings from sequence (target_feat = 447-dim)
        self.left_single = LinearNoBias(_TARGET_FEAT_DIM, c_pair)
        self.right_single = LinearNoBias(_TARGET_FEAT_DIM, c_pair)

        # Bond embedding
        self.bond_embedding = LinearNoBias(1, c_pair)

        # Previous embedding projections (recycling)
        # AF3: prev_embedding and prev_single_embedding use final_init (zeros)
        self.prev_embedding_layer_norm = LayerNorm(c_pair)
        self.prev_embedding = LinearNoBias(c_pair, c_pair, initializer="zeros")
        self.prev_single_embedding_layer_norm = LayerNorm(c_single)
        self.prev_single_embedding = LinearNoBias(c_single, c_single, initializer="zeros")

        # Position activations for relative encoding
        # rel_feat_dim = rel_pos(66) + rel_token(66) + entity_same(1) + rel_chain(6) = 139
        rel_feat_dim = 2 * (2 * cfg.max_relative_idx + 2) + 1 + (2 * cfg.max_relative_chain + 2)
        self.position_activations = LinearNoBias(rel_feat_dim, c_pair)

        # Single activations from target feat (447-dim input)
        self.single_activations = LinearNoBias(_TARGET_FEAT_DIM, c_single)

        # Variance-matched init for sum-fusion sites (from-scratch training).
        # Default af3_linear_init_ gives ``stddev(W) = sqrt(1/fan_in)``, which at
        # init produces per-stream output magnitudes that vary by orders of
        # magnitude across heterogeneous streams (e.g. fan_in=1 bond indicator
        # vs. fan_in=447 target_feat). For sum-fusion we want each stream to
        # contribute ~ 1/sqrt(K) at init.
        #
        # Pair init: K=4 (left, right, position, bond); prev=0, templates optional.
        _pair_target = 0.5  # 1 / sqrt(4)
        sum_fusion_init_(self.left_single.weight, _pair_target, "dense")
        sum_fusion_init_(self.right_single.weight, _pair_target, "dense")
        sum_fusion_init_(self.position_activations.weight, _pair_target, "one_hot")
        sum_fusion_init_(self.bond_embedding.weight, _pair_target, "one_hot")
        # MSA init: K=2 (msa_activations, extra_msa_target_feat broadcast).
        _msa_target = 1.0 / math.sqrt(2)
        sum_fusion_init_(self.msa_activations.weight, _msa_target, "dense")
        sum_fusion_init_(self.extra_msa_target_feat.weight, _msa_target, "dense")
        # Single init (K=1 active at step 0: only single_activations; prev=0) —
        # already matches default init, no override.

        # MSA stack
        self.msa_stack = nn.ModuleList(
            [
                EvoformerBlock(
                    c_msa,
                    c_pair,
                    num_msa_heads=cfg.msa_attention.num_head,
                    num_pair_heads=cfg.pair_attention.num_head,
                    outer_product_hidden=cfg.outer_product_num_channel,
                    msa_dropout=cfg.msa_dropout,
                    pair_dropout=cfg.pair_dropout,
                    optim=optim,
                )
                for _ in range(cfg.num_layer)
            ]
        )

        # PairFormer stack
        _trunk_deepnet_L = pf_cfg.num_layer if pf_cfg.deepnet_residual_scale else None
        self.trunk_pairformer = nn.ModuleList(
            [
                PairFormerBlock(
                    c_pair,
                    c_single,
                    num_pair_heads=pf_cfg.pair_attention.num_head,
                    num_single_heads=pf_cfg.single_attention_num_head,
                    single_key_dim=pf_cfg.single_key_dim,
                    single_value_dim=pf_cfg.single_value_dim,
                    pair_dropout=pf_cfg.pair_dropout,
                    optim=optim,
                    deepnet_layer_scale=_trunk_deepnet_L,
                )
                for _ in range(pf_cfg.num_layer)
            ]
        )

    def forward(
        self,
        target_feat: torch.Tensor,  # [B?, N, 447]
        msa_feat: torch.Tensor,  # [B?, S, N, 34]
        rel_feat: torch.Tensor,  # [B?, N, N, rel_feat_dim]
        pair_mask: torch.Tensor,  # [B?, N, N]
        seq_mask: torch.Tensor,  # [B?, N]
        msa_mask: torch.Tensor,  # [B?, S, N]
        prev_pair: torch.Tensor,  # [B?, N, N, c_pair]
        prev_single: torch.Tensor,  # [B?, N, c_single]
        bond_matrix: torch.Tensor | None = None,  # [B?, N, N, 1]
        template_aatype: torch.Tensor | None = None,  # [T, N]
        template_pseudo_beta: torch.Tensor | None = None,  # [T, N, 3]
        template_pseudo_beta_mask: torch.Tensor | None = None,  # [T, N]
        template_backbone_frame: torch.Tensor | None = None,  # [T, N, 3, 3]
        template_backbone_mask: torch.Tensor | None = None,  # [T, N]
        multichain_mask: torch.Tensor | None = None,  # [N, N]
    ) -> dict[str, torch.Tensor]:
        # Build initial pair activations (negative indices work for both batched/unbatched)
        left = self.left_single(target_feat).unsqueeze(-2)  # [..., N, 1, c_pair]
        right = self.right_single(target_feat).unsqueeze(-3)  # [..., 1, N, c_pair]
        pair = left + right  # [..., N, N, c_pair]

        # Add recycling pair
        pair = pair + self.prev_embedding(self.prev_embedding_layer_norm(prev_pair))

        # Add relative position encoding
        pair = pair + self.position_activations(rel_feat)

        # Add bond embedding
        if bond_matrix is not None:
            pair = pair + self.bond_embedding(bond_matrix)

        # Template embedding (after bonds, before MSA stack — matches AF3 pipeline).
        # ``TemplateEmbedding`` and ``SingleTemplateEmbedding`` natively support
        # batched ``[B?, N, ...]`` inputs (template axis is the dim before N).
        if self.template_embedding is not None and template_aatype is not None:
            t_axis = 1 if template_aatype.dim() == 3 else 0  # [B,T,N] vs [T,N]
            if template_aatype.shape[t_axis] > 0:
                if multichain_mask is None:
                    N = pair.shape[-2]
                    if pair.dim() == 4:
                        multichain_mask = torch.ones(pair.shape[0], N, N, device=pair.device, dtype=pair.dtype)
                    else:
                        multichain_mask = torch.ones(N, N, device=pair.device, dtype=pair.dtype)
                pair = pair + self.template_embedding(
                    query_pair=pair,
                    template_aatype=template_aatype,
                    template_pseudo_beta=template_pseudo_beta,
                    template_pseudo_beta_mask=template_pseudo_beta_mask,
                    template_backbone_frame=template_backbone_frame,
                    template_backbone_mask=template_backbone_mask,
                    pair_mask=pair_mask,
                    multichain_mask=multichain_mask,
                )

        # Initial single activations
        single = self.single_activations(target_feat)

        # Add recycling single
        single = single + self.prev_single_embedding(self.prev_single_embedding_layer_norm(prev_single))

        # MSA embedding: msa_feat [..., S, N, 34] → [..., S, N, c_msa]
        msa_act = self.msa_activations(msa_feat)
        msa_act = msa_act + self.extra_msa_target_feat(target_feat).unsqueeze(-3)

        # MSA stack (optionally checkpointed per block during training).
        use_ckpt = self._grad_checkpoint and torch.is_grad_enabled()
        for layer in self.msa_stack:
            if use_ckpt:
                msa_act, pair = _checkpoint(layer, msa_act, pair, msa_mask, pair_mask)
            else:
                msa_act, pair = layer(msa_act, pair, msa_mask, pair_mask)

        # PairFormer stack (optionally checkpointed per block during training)
        for layer in self.trunk_pairformer:
            if use_ckpt:
                pair, single = _checkpoint(layer, pair, single, pair_mask, seq_mask)
            else:
                pair, single = layer(pair, single, pair_mask, seq_mask)

        return {
            "pair": pair,
            "single": single,
            "target_feat": target_feat,
        }
