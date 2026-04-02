"""Diffusion head for DeepFold3.

Implements the denoising diffusion model for atom position generation.

Haiku scope: diffuser~/diffusion_head~/

Key submodules:
    conditioning~/  → see _conditioning() in AF3
    transformer~/   → DiffusionTransformer (24 layers, AdaLN-zero)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from deepfold3.model.atom_cross_attn import AtomDecoder, AtomEncoder
from deepfold3.model.config import DiffusionConfig, OptimConfig
from deepfold3.model.optim import make_transition_block
from deepfold3.model.primitives import LinearNoBias


def noise_schedule(
    t: torch.Tensor,
    sigma_data: float = 16.0,
    smin: float = 0.0004,
    smax: float = 160.0,
    p: float = 7.0,
) -> torch.Tensor:
    """AF3 noise schedule: sigma = sigma_data * (smax^(1/p) + t*(smin^(1/p)-smax^(1/p)))^p"""
    return sigma_data * (smax ** (1 / p) + t * (smin ** (1 / p) - smax ** (1 / p))) ** p


def _random_augmentation_from_vectors(
    positions: torch.Tensor,  # [N, M, 3]
    mask: torch.Tensor,  # [N, M] bool or float
    v0: torch.Tensor,  # [3]
    v1: torch.Tensor,  # [3]
    translation: torch.Tensor,  # [3]
) -> torch.Tensor:
    """Apply random rigid augmentation using pre-sampled random vectors.

    Matches AF3 diffusion_head.random_augmentation() exactly.
    """
    mask_f = mask.float().unsqueeze(-1)  # [N, M, 1]
    total = mask_f.sum().clamp(min=1e-6)
    center = (positions * mask_f).sum(dim=(0, 1), keepdim=True) / total
    pos = positions - center

    # Random rotation via Gram-Schmidt
    e0 = v0 / v0.norm().clamp(min=1e-10)
    v1_orth = v1 - e0 * torch.dot(v1, e0)
    e1 = v1_orth / v1_orth.norm().clamp(min=1e-10)
    e2 = torch.linalg.cross(e0, e1)
    rot = torch.stack([e0, e1, e2], dim=0)

    pos = pos @ rot + translation
    return pos * mask_f


def _random_augmentation(
    positions: torch.Tensor,
    mask: torch.Tensor,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Apply random rigid augmentation (backward-compatible single-sample interface)."""
    device = positions.device
    v0 = torch.randn(3, device=device, generator=generator)
    v1 = torch.randn(3, device=device, generator=generator)
    translation = torch.randn(3, device=device, generator=generator)
    return _random_augmentation_from_vectors(positions, mask, v0, v1, translation)


def _augment_in_place(
    positions: torch.Tensor,  # [B?, S, N, M, 3]
    mask: torch.Tensor,  # [B?, N, M] bool
    generator: torch.Generator | None,
) -> None:
    """Apply random augmentation to each sample in-place.

    Pre-samples all random vectors in a single kernel call to reduce
    the number of small CUDA kernel launches from 3*B*S to 3.
    """
    assert positions.dim() in (4, 5), f"Expected 4D or 5D positions, got {positions.dim()}D"
    device = positions.device
    S = positions.shape[-4]

    if positions.dim() == 5:  # [B, S, N, M, 3]
        B = positions.shape[0]
        # Pre-sample all random vectors: 3 vectors per (b, s) pair
        v0_all = torch.randn(B, S, 3, device=device, generator=generator)
        v1_all = torch.randn(B, S, 3, device=device, generator=generator)
        trans_all = torch.randn(B, S, 3, device=device, generator=generator)
        for b in range(B):
            for s in range(S):
                positions[b, s] = _random_augmentation_from_vectors(
                    positions[b, s], mask[b], v0_all[b, s], v1_all[b, s], trans_all[b, s]
                )
    else:  # [S, N, M, 3]
        v0_all = torch.randn(S, 3, device=device, generator=generator)
        v1_all = torch.randn(S, 3, device=device, generator=generator)
        trans_all = torch.randn(S, 3, device=device, generator=generator)
        for s in range(S):
            positions[s] = _random_augmentation_from_vectors(positions[s], mask, v0_all[s], v1_all[s], trans_all[s])


class DiffusionTransformerBlock(nn.Module):
    """Single AdaLN-zero transformer block for the diffusion model.

    Haiku: self_attention + transition_block functions called repeatedly in a
    layer_stack under transformer~/

    The block uses:
    - AdaLN self-attention (conditioned on single_cond)
    - AdaLN transition block

    Attribute names (converter uses the module name, e.g. 'transformer'):
        self-attention parts (name=module_name, e.g. 'transformer'):
            {name}layer_norm              → attn_norm (no elementwise affine)
            {name}single_cond_layer_norm  → attn_cond_norm_w
            {name}single_cond_scale       → attn_cond_scale
            {name}single_cond_bias        → attn_cond_bias
            {name}q_projection            → to_q (w/ bias)
            {name}k_projection            → to_k (no bias)
            {name}v_projection            → to_v (no bias)
            {name}gating_query            → to_g
            {name}transition2             → out_proj
            {name}adaptive_zero_cond      → zero_cond (bias=-2)
        transition parts (name=module_name):
            {name}ffw_layer_norm          → ffn_norm (no elementwise affine)
            {name}ffw_single_cond_*       → ffn_cond_*
            {name}ffw_transition1         → ffn_linear1
            {name}ffw_transition2         → ffn_linear2
            {name}ffw_adaptive_zero_cond  → ffn_zero_cond (bias=-2)
    """

    def __init__(
        self,
        c: int,
        c_single_cond: int,
        num_heads: int = 16,
        key_dim: int | None = None,
        value_dim: int | None = None,
        num_intermediate_factor: int = 2,
        optim: OptimConfig | None = None,
    ):
        super().__init__()
        # AF3: key_dim/value_dim default to c (num_channels) when None
        key_dim = key_dim if key_dim is not None else c
        value_dim = value_dim if value_dim is not None else c
        assert key_dim % num_heads == 0
        assert value_dim % num_heads == 0
        self.num_heads = num_heads
        self.key_dim_per_head = key_dim // num_heads
        self.val_dim_per_head = value_dim // num_heads
        self.scale = self.key_dim_per_head**-0.5

        num_int = c * num_intermediate_factor

        # Self-attention with AdaLN-zero
        # Conditioning inputs are c_single_cond-dimensional, outputs are c-dimensional
        self.attn_norm = nn.LayerNorm(c, elementwise_affine=False)
        self.attn_cond_norm_w = nn.Parameter(torch.ones(c_single_cond))
        self.attn_cond_scale = nn.Linear(c_single_cond, c)
        self.attn_cond_bias = LinearNoBias(c_single_cond, c)
        self.to_q = nn.Linear(c, key_dim)  # q_projection has bias
        self.to_k = LinearNoBias(c, key_dim)
        self.to_v = LinearNoBias(c, value_dim)
        self.to_g = LinearNoBias(c, value_dim)  # gating_query no bias
        self.out_proj = LinearNoBias(value_dim, c)  # transition2 no bias
        self.zero_cond = nn.Linear(c_single_cond, c)

        # FFN with AdaLN-zero
        self.ffn_norm = nn.LayerNorm(c, elementwise_affine=False)
        self.ffn_cond_norm_w = nn.Parameter(torch.ones(c_single_cond))
        self.ffn_cond_scale = nn.Linear(c_single_cond, c)
        self.ffn_cond_bias = LinearNoBias(c_single_cond, c)
        self.ffn_linear1 = LinearNoBias(c, num_int * 2)
        self.ffn_linear2 = LinearNoBias(num_int, c)  # ffw_transition2 no bias
        self.ffn_zero_cond = nn.Linear(c_single_cond, c)

        # Initialize
        nn.init.zeros_(self.zero_cond.weight)
        nn.init.constant_(self.zero_cond.bias, -2.0)
        nn.init.zeros_(self.ffn_zero_cond.weight)
        nn.init.constant_(self.ffn_zero_cond.bias, -2.0)

        # Fused FFN kernels (optional)
        self._fused_ada_ln_linear = None
        self._fused_swiglu = None
        if optim is not None and optim.use_fused_ada_transition:
            try:
                from deepfold3.model.triton.ada_layernorm_linear import FusedAdaLayerNormLinear
                from deepfold3.model.triton.swiglu import FusedSwiGLU

                self._fused_ada_ln_linear = FusedAdaLayerNormLinear(c, num_int * 2)
                self._fused_swiglu = FusedSwiGLU()
                self._sync_ffn_to_fused()
            except ImportError:
                pass

        if optim is not None and optim.use_torch_compile:
            self._compute = torch.compile(self._compute)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Load state dict then sync to fused FFN parameters if present."""
        if self._fused_ada_ln_linear is not None and strict:
            # Relax strict if incoming state_dict lacks fused keys (they are derived).
            if not any(k.startswith(("_fused_ada_ln_linear.", "_fused_swiglu.")) for k in state_dict):
                strict = False
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        if self._fused_ada_ln_linear is not None:
            self._sync_ffn_to_fused()
        return result

    def _sync_ffn_to_fused(self):
        """Copy ffn_linear1 weights to fused AdaLN+Linear."""
        if self._fused_ada_ln_linear is not None:
            with torch.no_grad():
                self._fused_ada_ln_linear.linear_weight.data.copy_(self.ffn_linear1.weight.data.T)

    def _compute(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        single_cond: torch.Tensor,
        pair_logits: torch.Tensor | None,
    ) -> torch.Tensor:
        B, N, C = x.shape
        Cs = single_cond.shape[-1]
        H = self.num_heads
        Dk = self.key_dim_per_head
        Dv = self.val_dim_per_head

        # AdaLN-zero self-attention
        xn = F.layer_norm(x, [C])
        cn = F.layer_norm(single_cond, [Cs]) * self.attn_cond_norm_w
        scale = torch.sigmoid(self.attn_cond_scale(cn))
        bias = self.attn_cond_bias(cn)
        xn = scale * xn + bias

        q = self.to_q(xn).view(B, N, H, Dk).permute(0, 2, 1, 3).float()  # [B, H, N, Dk]
        k = self.to_k(xn).view(B, N, H, Dk).permute(0, 2, 1, 3).float()
        v = self.to_v(xn).view(B, N, H, Dv).permute(0, 2, 1, 3).to(x.dtype)

        bias_mask = 1e9 * (mask.float() - 1.0)  # [B, N]

        logits = torch.matmul(q * self.scale, k.transpose(-2, -1))  # [B, H, N, N]
        if pair_logits is not None:
            logits = logits + pair_logits.float()
        logits = logits + bias_mask.view(B, 1, 1, N)
        w = F.softmax(logits, dim=-1).to(x.dtype)

        out = torch.matmul(w, v)  # [B, H, N, Dv]
        out = out.permute(0, 2, 1, 3).reshape(B, N, H * Dv)

        g = torch.sigmoid(self.to_g(xn))
        out = out * g
        out = self.out_proj(out)
        gate = torch.sigmoid(self.zero_cond(single_cond))
        x = x + gate * out

        # AdaLN-zero FFN
        cn = F.layer_norm(single_cond, [Cs]) * self.ffn_cond_norm_w
        scale = torch.sigmoid(self.ffn_cond_scale(cn))
        bias = self.ffn_cond_bias(cn)

        if self._fused_ada_ln_linear is not None and x.is_cuda:
            h = self._fused_ada_ln_linear(x, scale, bias)
            ff = self._fused_swiglu(h)
        else:
            xn = F.layer_norm(x, [C])
            xn = scale * xn + bias
            a, b = self.ffn_linear1(xn).chunk(2, dim=-1)
            ff = F.silu(a) * b

        ff_out = self.ffn_linear2(ff)
        gate = torch.sigmoid(self.ffn_zero_cond(single_cond))
        return x + gate * ff_out

    def forward(
        self,
        x: torch.Tensor,  # [B?, N, c]
        mask: torch.Tensor,  # [B?, N]
        single_cond: torch.Tensor,  # [B?, N, c_single_cond]
        pair_logits: torch.Tensor | None,  # [B?, H, N, N]
    ) -> torch.Tensor:
        has_batch = x.dim() == 3
        if not has_batch:
            x = x.unsqueeze(0)
            mask = mask.unsqueeze(0)
            single_cond = single_cond.unsqueeze(0)
            if pair_logits is not None:
                pair_logits = pair_logits.unsqueeze(0)

        x = self._compute(x, mask, single_cond, pair_logits)

        if not has_batch:
            x = x.squeeze(0)
        return x


class DiffusionTransformer(nn.Module):
    """Stack of DiffusionTransformerBlocks with pair-biased logits.

    Haiku: Transformer with num_blocks=24, super_block_size=4.

    Pair logits are recomputed once per super-block using a per-super-block
    projection (AF3 layer_stack gives each super-block its own weights).

    Attribute names:
        pair_norm_w            → bare Parameter (scale for pair LN, no offset)
        pair_logits_proj       → ModuleList[num_super] of Linear(c_pair, sbs*H)
        layers                 → ModuleList of DiffusionTransformerBlock
        output_norm_w          → bare Parameter (scale for output LN, no offset)
    """

    def __init__(
        self,
        c: int,
        c_single_cond: int,
        num_blocks: int = 24,
        num_heads: int = 16,
        key_dim: int | None = None,
        value_dim: int | None = None,
        num_intermediate_factor: int = 2,
        super_block_size: int = 4,
        c_pair: int = 128,
        optim: OptimConfig | None = None,
    ):
        super().__init__()
        assert num_blocks % super_block_size == 0
        self.num_blocks = num_blocks
        self.super_block_size = super_block_size
        num_super = num_blocks // super_block_size

        # Scale-only LayerNorm for pair conditioning (create_offset=False in AF3)
        self.pair_norm_w = nn.Parameter(torch.ones(c_pair))
        # Per-super-block pair→logit projection (each super-block has own weights)
        self.pair_logits_proj = nn.ModuleList(
            [LinearNoBias(c_pair, super_block_size * num_heads) for _ in range(num_super)]
        )

        self.layers = nn.ModuleList(
            [
                DiffusionTransformerBlock(
                    c, c_single_cond, num_heads, key_dim, value_dim, num_intermediate_factor, optim=optim
                )
                for _ in range(num_blocks)
            ]
        )

        # Scale-only output norm (create_offset=False in AF3)
        self.output_norm_w = nn.Parameter(torch.ones(c))

    def forward(
        self,
        act: torch.Tensor,  # [B?, N, c]
        mask: torch.Tensor,  # [B?, N]  float
        single_cond: torch.Tensor,  # [B?, N, c]
        pair_cond: torch.Tensor,  # [B?, N, N, c_pair]
    ) -> torch.Tensor:
        H = self.layers[0].num_heads
        sbs = self.super_block_size
        num_super = self.num_blocks // sbs

        # Compute pair norm once (pair_cond is constant across super-blocks)
        pair_norm = F.layer_norm(pair_cond, [pair_cond.shape[-1]]) * self.pair_norm_w

        for super_idx in range(num_super):
            # Per-super-block pair logit projection
            pair_logits_all = self.pair_logits_proj[super_idx](pair_norm)  # [..., N, N, sbs*H]
            # Move last dim (sbs*H) before the two N dims: [..., sbs*H, N, N]
            pair_logits_all = pair_logits_all.movedim(-1, -3)

            for block_in_super in range(sbs):
                i = super_idx * sbs + block_in_super
                # Slice along the sbs*H dim (now at dim -3)
                start = block_in_super * H
                end = start + H
                pl = pair_logits_all[..., start:end, :, :]  # [..., H, N, N]
                act = self.layers[i](act, mask, single_cond, pl)

        act = F.layer_norm(act, [act.shape[-1]]) * self.output_norm_w
        return act


class DiffusionConditioningBlock(nn.Module):
    """Projects trunk embeddings to diffusion conditioning vectors.

    Haiku: DiffusionHead._conditioning()

    Attribute names (under diffusion_head~/):
        pair_cond_initial_norm       → pair_cond_norm   (no offset)
        pair_cond_initial_projection → pair_cond_proj
        pair_transition_0~/ffw_*     → pair_trans0.*
        pair_transition_1~/ffw_*     → pair_trans1.*
        single_cond_initial_norm     → single_cond_norm  (no offset)
        single_cond_initial_projection → single_cond_proj
        noise_embedding_initial_norm → noise_emb_norm   (no offset)
        noise_embedding_initial_projection → noise_emb_proj
        single_transition_0~/ffw_*   → single_trans0.*
        single_transition_1~/ffw_*   → single_trans1.*
    """

    def __init__(
        self,
        c_pair_in: int,
        c_single_in: int,
        rel_feat_dim: int,
        c_pair_out: int,
        c_single_out: int,
        target_feat_dim: int = 447,
        sigma_data: float = 16.0,
        fourier_dim: int = 256,
        num_intermediate_factor: int = 2,
        optim: OptimConfig | None = None,
    ):
        super().__init__()
        self.sigma_data = sigma_data
        # Pair conditioning
        pair_in = c_pair_in + rel_feat_dim
        self.pair_cond_norm_w = nn.Parameter(torch.ones(pair_in))
        self.pair_cond_proj = LinearNoBias(pair_in, c_pair_out)
        self.pair_trans0 = make_transition_block(
            c_pair_out, num_intermediate_factor=num_intermediate_factor, config=optim
        )
        self.pair_trans1 = make_transition_block(
            c_pair_out, num_intermediate_factor=num_intermediate_factor, config=optim
        )

        # Single conditioning: concat(single_embedding, target_feat)
        single_in = c_single_in + target_feat_dim  # 384 + 447 = 831
        self.single_cond_norm_w = nn.Parameter(torch.ones(single_in))
        self.single_cond_proj = LinearNoBias(single_in, c_single_out)

        # Noise embedding conditioning (Fourier weights loaded from safetensors)
        self.register_buffer("fourier_weight", torch.zeros(fourier_dim))
        self.register_buffer("fourier_bias", torch.zeros(fourier_dim))
        self.noise_emb_norm_w = nn.Parameter(torch.ones(fourier_dim))
        self.noise_emb_proj = LinearNoBias(fourier_dim, c_single_out)

        self.single_trans0 = make_transition_block(
            c_single_out, num_intermediate_factor=num_intermediate_factor, config=optim
        )
        self.single_trans1 = make_transition_block(
            c_single_out, num_intermediate_factor=num_intermediate_factor, config=optim
        )

    def _fourier_noise_embedding(self, sigma: torch.Tensor) -> torch.Tensor:
        """Fixed Fourier embedding of noise level using registered buffers.

        sigma: scalar or batch of noise levels.
        Returns: [..., fourier_dim]
        """
        x = (1 / 4) * torch.log(sigma)
        emb = x.unsqueeze(-1) * self.fourier_weight + self.fourier_bias
        return torch.cos(2 * math.pi * emb)

    def forward(
        self,
        single_emb: torch.Tensor,  # [N, c_single_in]
        pair_emb: torch.Tensor,  # [N, N, c_pair_in]
        target_feat: torch.Tensor,  # [N, c_single_in]
        rel_feat: torch.Tensor,  # [N, N, rel_feat_dim]
        noise_level: torch.Tensor,  # scalar
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Pair cond
        pair_in = torch.cat([pair_emb, rel_feat], dim=-1)
        pair_norm = F.layer_norm(pair_in, [pair_in.shape[-1]]) * self.pair_cond_norm_w
        pair_cond = self.pair_cond_proj(pair_norm)
        # Simple transition (no external conditioning for pair)
        # AF3 uses transition_block without single_cond for pair here
        # AdaTransitionBlock expects cond; we use pair_cond itself as self-cond
        pair_cond = pair_cond + self.pair_trans0(pair_cond)
        pair_cond = pair_cond + self.pair_trans1(pair_cond)

        # Single cond
        single_in = torch.cat([single_emb, target_feat], dim=-1)
        single_norm = F.layer_norm(single_in, [single_in.shape[-1]]) * self.single_cond_norm_w
        single_cond = self.single_cond_proj(single_norm)

        # Add noise embedding (broadcast to [..., N, 256])
        # noise_level may be scalar or [B] for per-sample noise in training
        noise_emb = self._fourier_noise_embedding(noise_level / self.sigma_data)  # [..., 256]
        while noise_emb.dim() < single_cond.dim():
            noise_emb = noise_emb.unsqueeze(-2)
        noise_expanded = noise_emb.expand(*single_cond.shape[:-1], -1)
        noise_norm = F.layer_norm(noise_expanded, [noise_emb.shape[-1]]) * self.noise_emb_norm_w
        single_cond = single_cond + self.noise_emb_proj(noise_norm)

        single_cond = single_cond + self.single_trans0(single_cond)
        single_cond = single_cond + self.single_trans1(single_cond)

        return single_cond, pair_cond


class DiffusionHead(nn.Module):
    """Full diffusion head: conditioning + atom encoder + transformer + atom decoder.

    Haiku scope: diffuser~/diffusion_head~/
    """

    def __init__(
        self,
        cfg: DiffusionConfig,
        c_single: int = 384,
        c_pair: int = 128,
        rel_feat_dim: int = 139,
        optim: OptimConfig | None = None,
    ):
        super().__init__()
        self.cfg = cfg
        c_diff = cfg.atom_cross_att.per_token_channels  # 768

        # Conditioning
        self.conditioning = DiffusionConditioningBlock(
            c_pair_in=c_pair,
            c_single_in=c_single,
            rel_feat_dim=rel_feat_dim,
            c_pair_out=cfg.conditioning_pair_channel,
            c_single_out=cfg.conditioning_seq_channel,
            sigma_data=cfg.sigma_data,
            fourier_dim=cfg.fourier_dim,
            num_intermediate_factor=cfg.conditioning_num_intermediate_factor,
            optim=optim,
        )

        # Atom encoder (per-atom conditioning → token activations)
        self.atom_encoder = AtomEncoder(
            per_atom_channels=cfg.atom_cross_att.per_atom_channels,
            per_token_channels=cfg.atom_cross_att.per_token_channels,
            per_atom_pair_channels=cfg.atom_cross_att.per_atom_pair_channels,
            atom_transformer_num_blocks=cfg.atom_cross_att.atom_transformer.num_blocks,
            Q=cfg.atom_cross_att.queries_subset_size,
            K=cfg.atom_cross_att.keys_subset_size,
            num_element_classes=cfg.atom_cross_att.num_element_classes,
            num_atom_name_chars=cfg.atom_cross_att.num_atom_name_chars,
            atom_name_length=cfg.atom_cross_att.atom_name_length,
        )

        # Project trunk single to diffusion act (scale-only LN + no-bias projection)
        self.single_cond_emb_norm_w = nn.Parameter(torch.ones(cfg.conditioning_seq_channel))
        self.single_cond_embedding_projection = LinearNoBias(cfg.conditioning_seq_channel, c_diff)

        # Main diffusion transformer
        self.transformer = DiffusionTransformer(
            c=c_diff,
            c_single_cond=cfg.conditioning_seq_channel,
            num_blocks=cfg.transformer.num_blocks,
            num_heads=cfg.transformer.num_head,
            key_dim=cfg.transformer.key_dim,
            value_dim=cfg.transformer.value_dim,
            num_intermediate_factor=cfg.transformer.num_intermediate_factor,
            super_block_size=cfg.transformer.super_block_size,
            c_pair=cfg.conditioning_pair_channel,
            optim=optim,
        )

        # Atom decoder
        self.atom_decoder = AtomDecoder(
            per_atom_channels=cfg.atom_cross_att.per_atom_channels,
            per_token_channels=c_diff,
            per_atom_pair_channels=cfg.atom_cross_att.per_atom_pair_channels,
            atom_transformer_num_blocks=cfg.atom_cross_att.atom_transformer.num_blocks,
        )

    def denoise(
        self,
        noisy_pos: torch.Tensor,  # [B?, N, M, 3]
        noise_level: torch.Tensor,  # scalar or [B]
        embeddings: dict[str, torch.Tensor],
        ref_pos: torch.Tensor,  # [B?, N, M, 3]
        ref_mask: torch.Tensor,  # [B?, N, M]
        ref_element: torch.Tensor,  # [B?, N, M] int64
        ref_charge: torch.Tensor,  # [B?, N, M]
        ref_atom_name_chars: torch.Tensor,  # [B?, N, M, 4] int64
        seq_mask: torch.Tensor,  # [B?, N]
        rel_feat: torch.Tensor,  # [B?, N, N, rel_feat_dim]
    ) -> torch.Tensor:
        """Run one denoising step. Returns denoised positions [B?, N, M, 3].

        Supports both unbatched and batched inputs. The atom encoder/decoder
        run per-sample (build_atom_subsets is sample-dependent), while the
        conditioning block and main transformer run batched.
        """
        sd = self.cfg.sigma_data
        has_batch = noisy_pos.dim() == 4  # [B, N, M, 3]

        # Reshape noise_level for broadcasting with position tensors
        nl = noise_level
        while nl.dim() < noisy_pos.dim():
            nl = nl.unsqueeze(-1)

        # Scale inputs
        act_pos = noisy_pos * ref_mask.unsqueeze(-1)
        act_pos = act_pos / (nl**2 + sd**2).sqrt()

        # Conditioning (already batch-aware)
        single_cond, pair_cond = self.conditioning(
            single_emb=embeddings["single"],
            pair_emb=embeddings["pair"],
            target_feat=embeddings["target_feat"],
            rel_feat=rel_feat,
            noise_level=noise_level,
        )

        # Atom encoder: per-sample (build_atom_subsets is sample-dependent).
        # enc_list must persist until the decoder loop because the decoder
        # reuses encoder conditioning (skip_connection, pair_cond, subsets).
        if has_batch:
            B = noisy_pos.shape[0]
            enc_list = []
            for b in range(B):
                enc_list.append(
                    self.atom_encoder(
                        ref_pos=ref_pos[b],
                        ref_mask=ref_mask[b],
                        ref_element=ref_element[b],
                        ref_charge=ref_charge[b],
                        ref_atom_name_chars=ref_atom_name_chars[b],
                        noisy_atom_pos=act_pos[b],
                        trunk_single=embeddings["single"][b],
                        trunk_pair=pair_cond[b],
                    )
                )
            act = torch.stack([e.token_act for e in enc_list]).float()
        else:
            enc = self.atom_encoder(
                ref_pos=ref_pos,
                ref_mask=ref_mask,
                ref_element=ref_element,
                ref_charge=ref_charge,
                ref_atom_name_chars=ref_atom_name_chars,
                noisy_atom_pos=act_pos,
                trunk_single=embeddings["single"],
                trunk_pair=pair_cond,
            )
            act = enc.token_act.float()

        # Add trunk single conditioning to token activations
        sc_norm = F.layer_norm(single_cond, [single_cond.shape[-1]]) * self.single_cond_emb_norm_w
        act = act + self.single_cond_embedding_projection(sc_norm)

        # Main transformer (already batch-aware)
        act = self.transformer(act, seq_mask, single_cond, pair_cond)

        # Atom decoder: per-sample. Pop from enc_list to free each encoder
        # output as soon as it's consumed (enables GC mid-loop).
        if has_batch:
            pos_updates = []
            for b in range(B):
                pos_updates.append(self.atom_decoder(act[b], enc_list[b]))
                enc_list[b] = None  # release encoder output for GC
            pos_update = torch.stack(pos_updates)
        else:
            pos_update = self.atom_decoder(act, enc)

        # Combine skip + update (AF3 EDM-style)
        skip_scaling = sd**2 / (nl**2 + sd**2)
        out_scaling = nl * sd / (nl**2 + sd**2).sqrt()
        result = skip_scaling * noisy_pos + out_scaling * pos_update
        return result * ref_mask.unsqueeze(-1)

    def _denoise_samples(
        self,
        noisy_pos: torch.Tensor,  # [B?, S, N, M, 3]
        noise_level: torch.Tensor,  # scalar or [B?, S]
        embeddings: dict[str, torch.Tensor],
        ref_pos: torch.Tensor,
        ref_mask: torch.Tensor,
        ref_element: torch.Tensor,
        ref_charge: torch.Tensor,
        ref_atom_name_chars: torch.Tensor,
        seq_mask: torch.Tensor,
        rel_feat: torch.Tensor,
    ) -> torch.Tensor:
        """Denoise each sample independently. Returns [B?, S, N, M, 3].

        Each denoise call is wrapped in ``torch.utils.checkpoint`` so that
        forward-pass intermediates (24 transformer blocks + atom encoder/decoder)
        are freed per-sample rather than accumulated across all S samples.
        This reduces peak memory from O(S) to O(1) w.r.t. transformer activations,
        at the cost of recomputing each sample's forward pass during backward.
        """
        S = noisy_pos.shape[-4]
        use_ckpt = torch.is_grad_enabled()
        results = []
        for s in range(S):
            nl_s = noise_level if noise_level.dim() == 0 else noise_level[..., s]
            if use_ckpt:
                result = torch_checkpoint(
                    self.denoise,
                    noisy_pos[..., s, :, :, :],
                    nl_s,
                    embeddings,
                    ref_pos,
                    ref_mask,
                    ref_element,
                    ref_charge,
                    ref_atom_name_chars,
                    seq_mask,
                    rel_feat,
                    use_reentrant=False,
                )
            else:
                result = self.denoise(
                    noisy_pos=noisy_pos[..., s, :, :, :],
                    noise_level=nl_s,
                    embeddings=embeddings,
                    ref_pos=ref_pos,
                    ref_mask=ref_mask,
                    ref_element=ref_element,
                    ref_charge=ref_charge,
                    ref_atom_name_chars=ref_atom_name_chars,
                    seq_mask=seq_mask,
                    rel_feat=rel_feat,
                )
            results.append(result)
        return torch.stack(results, dim=-4)

    @torch.no_grad()
    def sample(
        self,
        embeddings: dict[str, torch.Tensor],
        ref_pos: torch.Tensor,  # [B?, N, M, 3]
        ref_mask: torch.Tensor,  # [B?, N, M]
        ref_element: torch.Tensor,  # [B?, N, M]
        ref_charge: torch.Tensor,  # [B?, N, M]
        ref_atom_name_chars: torch.Tensor,  # [B?, N, M, 4]
        seq_mask: torch.Tensor,  # [B?, N]
        rel_feat: torch.Tensor,  # [B?, N, N, rel_feat_dim]
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        """Draw samples from the diffusion model (Euler-Maruyama sampler).

        Supports both unbatched and batched inputs. For unbatched inputs,
        returns 'atom_positions' [S, N, M, 3] and 'mask' [S, N, M].
        For batched inputs, returns [B, S, N, M, 3] and [B, S, N, M].
        """
        cfg = self.cfg
        N, M = ref_mask.shape[-2:]
        device = ref_mask.device
        batch_shape = ref_mask.shape[:-2]  # () or (B,)

        noise_levels = noise_schedule(
            torch.linspace(0, 1, cfg.steps + 1, device=device),
            sigma_data=cfg.sigma_data,
            smin=cfg.smin,
            smax=cfg.smax,
            p=cfg.noise_p,
        )

        # Pre-compute per-step gamma to avoid GPU→CPU sync inside the loop
        gammas = torch.where(noise_levels[1:] > cfg.gamma_min, cfg.gamma_0, 0.0)  # [steps]

        # Initialize from pure noise: [B?, S, N, M, 3]
        positions = torch.randn(*batch_shape, cfg.num_samples, N, M, 3, device=device, generator=generator)
        positions = positions * noise_levels[0]
        aug_mask = ref_mask.bool()

        for step in range(cfg.steps):
            sigma_prev = noise_levels[step]
            sigma_next = noise_levels[step + 1]

            # Random augmentation per sample — required for AF3 equivalence.
            _augment_in_place(positions, aug_mask, generator)

            # Stochastic gamma step (pre-computed, no GPU sync).
            gamma = gammas[step]
            sigma_hat = sigma_prev * (1 + gamma)

            # Add noise
            noise_scale = cfg.noise_scale * (sigma_hat**2 - sigma_prev**2).sqrt()
            positions_hat = positions + noise_scale * torch.randn_like(positions, generator=generator)

            # Denoise + Euler step
            denoised = self._denoise_samples(
                positions_hat,
                sigma_hat,
                embeddings,
                ref_pos,
                ref_mask,
                ref_element,
                ref_charge,
                ref_atom_name_chars,
                seq_mask,
                rel_feat,
            )
            d_t = sigma_next - sigma_hat
            positions = positions_hat + cfg.step_scale * d_t * (positions_hat - denoised) / sigma_hat

        mask_expanded = ref_mask.unsqueeze(-3).expand(*batch_shape, cfg.num_samples, N, M)
        return {
            "atom_positions": positions,
            "mask": mask_expanded,
        }

    def sample_training(
        self,
        gt_positions: torch.Tensor,  # [B?, N, M, 3]
        gt_mask: torch.Tensor,  # [B?, N, M]
        embeddings: dict[str, torch.Tensor],
        ref_pos: torch.Tensor,  # [B?, N, M, 3]
        ref_mask: torch.Tensor,  # [B?, N, M]
        ref_element: torch.Tensor,  # [B?, N, M]
        ref_charge: torch.Tensor,  # [B?, N, M]
        ref_atom_name_chars: torch.Tensor,  # [B?, N, M, 4]
        seq_mask: torch.Tensor,  # [B?, N]
        rel_feat: torch.Tensor,  # [B?, N, N, rel_feat_dim]
        num_samples: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Training forward: augment GT, add noise, denoise once (AF3 §3.7.1).

        Returns:
            gt_augmented: Augmented ground truth [B?, S, N, M, 3].
            denoised: Denoised predictions [B?, S, N, M, 3].
            noise_levels: Sampled noise levels [B?, S].
        """
        cfg = self.cfg
        batch_shape = gt_positions.shape[:-3]  # () or (B,)
        device = gt_positions.device

        # Augment ground truth: expand to S copies, augment each independently
        gt_aug = gt_positions.unsqueeze(-4).expand(*batch_shape, num_samples, *gt_positions.shape[-3:]).clone()
        _augment_in_place(gt_aug, gt_mask.bool(), generator=None)

        # Sample noise levels from log-normal: t = sigma_data * exp(p_mean + p_std * z)
        noise_levels = cfg.sigma_data * torch.exp(
            cfg.p_mean + cfg.p_std * torch.randn(*batch_shape, num_samples, device=device)
        )  # [B?, S]

        # Add noise
        nl = noise_levels
        while nl.dim() < gt_aug.dim():
            nl = nl.unsqueeze(-1)
        noisy_pos = gt_aug + torch.randn_like(gt_aug) * nl  # [B?, S, N, M, 3]

        # Denoise
        denoised = self._denoise_samples(
            noisy_pos,
            noise_levels,
            embeddings,
            ref_pos,
            ref_mask,
            ref_element,
            ref_charge,
            ref_atom_name_chars,
            seq_mask,
            rel_feat,
        )

        return gt_aug, denoised, noise_levels
