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


def _random_augmentation(
    positions: torch.Tensor,  # [N, M, 3]
    mask: torch.Tensor,  # [N, M] bool or float
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Apply random rigid augmentation (center → rotate → translate).

    Matches AF3 diffusion_head.random_augmentation() exactly.
    Applied at every denoising step so the model sees training-time distribution.
    """
    device = positions.device
    mask_f = mask.float().unsqueeze(-1)  # [N, M, 1]
    total = mask_f.sum().clamp(min=1e-6)
    center = (positions * mask_f).sum(dim=(0, 1), keepdim=True) / total  # [1, 1, 3]
    pos = positions - center

    # Random rotation via Gram-Schmidt (matches AF3 random_rotation)
    v0 = torch.randn(3, device=device, generator=generator)
    v1 = torch.randn(3, device=device, generator=generator)
    e0 = v0 / v0.norm().clamp(min=1e-10)
    v1 = v1 - e0 * torch.dot(v1, e0)
    e1 = v1 / v1.norm().clamp(min=1e-10)
    e2 = torch.linalg.cross(e0, e1)
    rot = torch.stack([e0, e1, e2], dim=0)  # [3, 3] rows = basis vectors

    # pos @ rot  matches AF3's einsum('...i,ij->...j', x, rot)
    pos = pos @ rot  # [N, M, 3]

    # Random translation ~ N(0, 1) for each coordinate (matches AF3)
    translation = torch.randn(3, device=device, generator=generator)
    pos = pos + translation

    return pos * mask_f


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
        xn = F.layer_norm(x, [C])
        cn = F.layer_norm(single_cond, [Cs]) * self.ffn_cond_norm_w
        scale = torch.sigmoid(self.ffn_cond_scale(cn))
        bias = self.ffn_cond_bias(cn)
        xn = scale * xn + bias

        a, b = self.ffn_linear1(xn).chunk(2, dim=-1)
        ff = F.silu(a) * b
        ff_out = self.ffn_linear2(ff)
        gate = torch.sigmoid(self.ffn_zero_cond(single_cond))
        x = x + gate * ff_out

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
                DiffusionTransformerBlock(c, c_single_cond, num_heads, key_dim, value_dim, num_intermediate_factor)
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
        optim: OptimConfig | None = None,
    ):
        super().__init__()
        # Pair conditioning
        pair_in = c_pair_in + rel_feat_dim
        self.pair_cond_norm_w = nn.Parameter(torch.ones(pair_in))
        self.pair_cond_proj = LinearNoBias(pair_in, c_pair_out)
        self.pair_trans0 = make_transition_block(c_pair_out, num_intermediate_factor=2, config=optim)
        self.pair_trans1 = make_transition_block(c_pair_out, num_intermediate_factor=2, config=optim)

        # Single conditioning: concat(single_embedding, target_feat)
        single_in = c_single_in + target_feat_dim  # 384 + 447 = 831
        self.single_cond_norm_w = nn.Parameter(torch.ones(single_in))
        self.single_cond_proj = LinearNoBias(single_in, c_single_out)

        # Noise embedding conditioning (Fourier weights loaded from safetensors)
        fourier_dim = 256
        self.register_buffer("fourier_weight", torch.zeros(fourier_dim))
        self.register_buffer("fourier_bias", torch.zeros(fourier_dim))
        self.noise_emb_norm_w = nn.Parameter(torch.ones(fourier_dim))
        self.noise_emb_proj = LinearNoBias(fourier_dim, c_single_out)

        self.single_trans0 = make_transition_block(c_single_out, num_intermediate_factor=2, config=optim)
        self.single_trans1 = make_transition_block(c_single_out, num_intermediate_factor=2, config=optim)

    def _fourier_noise_embedding(self, sigma: torch.Tensor) -> torch.Tensor:
        """Fixed Fourier embedding of noise level using registered buffers.

        sigma: scalar or batch of noise levels.
        Returns: [..., 256]
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
        noise_emb = self._fourier_noise_embedding(noise_level / 16.0)  # [256]
        noise_expanded = noise_emb.view(*([1] * (single_cond.dim() - 1)), -1)
        noise_expanded = noise_expanded.expand(*single_cond.shape[:-1], -1)
        noise_norm = F.layer_norm(noise_expanded, [noise_emb.shape[-1]]) * self.noise_emb_norm_w
        single_cond = single_cond + self.noise_emb_proj(noise_norm)

        single_cond = single_cond + self.single_trans0(single_cond)
        single_cond = single_cond + self.single_trans1(single_cond)

        return single_cond, pair_cond


class DiffusionHead(nn.Module):
    """Full diffusion head: conditioning + atom encoder + transformer + atom decoder.

    Haiku scope: diffuser~/diffusion_head~/
    """

    SIGMA_DATA = 16.0

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
        noisy_pos: torch.Tensor,  # [N, M, 3]
        noise_level: torch.Tensor,  # scalar
        embeddings: dict[str, torch.Tensor],
        ref_pos: torch.Tensor,  # [N, M, 3]
        ref_mask: torch.Tensor,  # [N, M]
        ref_element: torch.Tensor,  # [N, M] int64
        ref_charge: torch.Tensor,  # [N, M]
        ref_atom_name_chars: torch.Tensor,  # [N, M, 4] int64
        seq_mask: torch.Tensor,  # [N]
        rel_feat: torch.Tensor,  # [N, N, rel_feat_dim]
    ) -> torch.Tensor:
        """Run one denoising step. Returns denoised positions [N, M, 3]."""
        sd = self.SIGMA_DATA

        # Scale inputs
        act_pos = noisy_pos * ref_mask.unsqueeze(-1)
        act_pos = act_pos / (noise_level**2 + sd**2).sqrt()

        # Conditioning
        single_cond, pair_cond = self.conditioning(
            single_emb=embeddings["single"],
            pair_emb=embeddings["pair"],
            target_feat=embeddings["target_feat"],
            rel_feat=rel_feat,
            noise_level=noise_level,
        )

        # Atom encoder
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

        # Main transformer
        act = self.transformer(act, seq_mask, single_cond, pair_cond)

        # Atom decoder
        pos_update = self.atom_decoder(act, enc)  # [N, M, 3]

        # Combine skip + update (AF3 EDM-style)
        skip_scaling = sd**2 / (noise_level**2 + sd**2)
        out_scaling = noise_level * sd / (noise_level**2 + sd**2).sqrt()
        result = skip_scaling * noisy_pos + out_scaling * pos_update
        return result * ref_mask.unsqueeze(-1)

    @torch.no_grad()
    def sample(
        self,
        embeddings: dict[str, torch.Tensor],
        ref_pos: torch.Tensor,
        ref_mask: torch.Tensor,
        ref_element: torch.Tensor,
        ref_charge: torch.Tensor,
        ref_atom_name_chars: torch.Tensor,
        seq_mask: torch.Tensor,
        rel_feat: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        """Draw samples from the diffusion model (Euler-Maruyama sampler).

        Returns:
            dict with 'atom_positions' [num_samples, N, M, 3]
                  and 'mask' [num_samples, N, M]
        """
        cfg = self.cfg
        N, M = ref_mask.shape
        device = ref_mask.device

        noise_levels = noise_schedule(
            torch.linspace(0, 1, cfg.steps + 1, device=device),
            sigma_data=cfg.sigma_data,
            smin=cfg.smin,
            smax=cfg.smax,
            p=cfg.noise_p,
        )

        # Initialize from pure noise
        positions = torch.randn(cfg.num_samples, N, M, 3, device=device, generator=generator)
        positions = positions * noise_levels[0]

        noise_levels_prev = noise_levels[:-1]
        noise_levels_next = noise_levels[1:]

        for step in range(cfg.steps):
            sigma_prev = noise_levels_prev[step]
            sigma_next = noise_levels_next[step]

            # Random augmentation per sample — required for AF3 equivalence.
            # The model was trained with random center/rotate/translate at every
            # denoising step; without this the positions are in a fixed frame
            # and the denoiser produces garbage.
            for s in range(cfg.num_samples):
                positions[s] = _random_augmentation(positions[s], ref_mask.bool(), generator)

            # Stochastic gamma step.
            # AF3 checks sigma_next (the target level), not sigma_prev.
            gamma = cfg.gamma_0 if sigma_next > cfg.gamma_min else 0.0
            sigma_hat = sigma_prev * (1 + gamma)

            # Add noise
            noise_scale = cfg.noise_scale * (sigma_hat**2 - sigma_prev**2).sqrt()
            noise = noise_scale * torch.randn_like(positions, generator=generator)
            positions_hat = positions + noise

            # Denoise each sample
            denoised_list = []
            for s in range(cfg.num_samples):
                denoised = self.denoise(
                    noisy_pos=positions_hat[s],
                    noise_level=sigma_hat,
                    embeddings=embeddings,
                    ref_pos=ref_pos,
                    ref_mask=ref_mask,
                    ref_element=ref_element,
                    ref_charge=ref_charge,
                    ref_atom_name_chars=ref_atom_name_chars,
                    seq_mask=seq_mask,
                    rel_feat=rel_feat,
                )
                denoised_list.append(denoised)
            denoised_all = torch.stack(denoised_list, dim=0)  # [S, N, M, 3]

            # Euler step
            grad = (positions_hat - denoised_all) / sigma_hat
            d_t = sigma_next - sigma_hat
            positions = positions_hat + cfg.step_scale * d_t * grad

        mask_expanded = ref_mask.unsqueeze(0).expand(cfg.num_samples, -1, -1)
        return {
            "atom_positions": positions,
            "mask": mask_expanded,
        }
