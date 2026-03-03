"""Atom cross-attention encoder and decoder.

Maps between per-atom layout [N_tok, max_atoms, C] and per-token layout [N_tok, C]
via a mini cross-attention transformer over flat atom subsets.

Flat atom layout with rectangular block attention (matching AF3):
  All real atoms packed into [total_atoms], padded to multiple of Q=32.
  Non-overlapping query subsets of Q=32 atoms (spanning ~6 residues).
  Overlapping key windows of K=128 atoms centered on each query block.

The weights are identical to the per-token sliding window implementation —
only the indexing/layout logic changes.

Two distinct encoders exist in AF3:
  1. EvoformerConditioningEncoder  — produces target_feat contribution (no trunk cond)
  2. AtomEncoder (diffusion)       — conditions on trunk single/pair + noisy positions

Haiku scope prefixes:
    EvoformerConditioningEncoder: 'diffuser/evoformer_conditioning_*'
    AtomEncoder (diffusion):      'diffuser/~/diffusion_head/diffusion_*'
    AtomDecoder (diffusion):      'diffuser/~/diffusion_head/diffusion_*' (decoder part)
"""

from __future__ import annotations

import dataclasses

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepfold3.model.primitives import LinearNoBias


@dataclasses.dataclass
class AtomSubsets:
    """Precomputed indices for flat atom subset attention.

    Maps between token-atoms layout [N, M] and flat subset layout [S, Q] / [S, K].
    """

    num_subsets: int  # S = num_atoms_padded // Q
    Q: int  # 32 (atoms per query subset)
    K: int  # 128 (atoms per key window)
    N: int  # num_tokens
    M: int  # max_atoms_per_token (24)

    # Gather: token-atoms [N*M] → queries [S, Q]
    ta_to_q_idx: torch.Tensor  # [S, Q] long — indices into flattened [N*M]
    ta_to_q_mask: torch.Tensor  # [S, Q] bool

    # Gather: flat queries [S*Q] → keys [S, K]
    q_to_k_idx: torch.Tensor  # [S, K] long — indices into flattened [S*Q]
    q_to_k_mask: torch.Tensor  # [S, K] bool

    # Token index per query/key atom (for trunk_pair broadcast + offsets_valid)
    token_for_query: torch.Tensor  # [S, Q] long
    token_for_key: torch.Tensor  # [S, K] long

    # Scatter: queries [S*Q] → token-atoms [N, M]
    q_to_ta_idx: torch.Tensor  # [N, M] long — indices into flattened [S*Q]
    q_to_ta_mask: torch.Tensor  # [N, M] bool


def build_atom_subsets(ref_mask: torch.Tensor, Q: int = 32, K: int = 128) -> AtomSubsets:
    """Build flat atom subset indices from ref_mask [N, M].

    1. Flatten ref_mask → find real atom indices in [N*M]
    2. Pad to next multiple of Q → num_atoms_padded, S = padded // Q
    3. Build query/key gather indices and scatter-back indices
    """
    device = ref_mask.device
    N, M = ref_mask.shape

    # Find real atom indices in flattened [N*M] space
    flat_mask = ref_mask.reshape(-1).bool()  # [N*M]
    real_indices = flat_mask.nonzero(as_tuple=False).squeeze(-1)  # [num_real]
    num_real = real_indices.shape[0]

    # Pad to next multiple of Q
    num_padded = ((num_real + Q - 1) // Q) * Q
    S = num_padded // Q

    # ta_to_q_idx: [S*Q] indices into [N*M], padded slots point to 0
    padded_indices = torch.zeros(num_padded, dtype=torch.long, device=device)
    padded_indices[:num_real] = real_indices
    padded_mask = torch.zeros(num_padded, dtype=torch.bool, device=device)
    padded_mask[:num_real] = True

    ta_to_q_idx = padded_indices.reshape(S, Q)  # [S, Q]
    ta_to_q_mask = padded_mask.reshape(S, Q)  # [S, Q]

    # Key windows: for each subset s, center at (s*Q + Q/2), take K atoms
    # Key indices are into the flat query space [S*Q]
    # AF3 uses window shifting (not clamping) at boundaries to avoid duplicate atoms
    centers = torch.arange(Q // 2, num_padded, Q, device=device)  # [S]
    key_offsets = torch.arange(-(K // 2), K // 2, device=device)  # [K]
    key_flat_idx = centers.unsqueeze(1) + key_offsets.unsqueeze(0)  # [S, K]

    # Shift entire windows to stay within [0, num_real-1] (matching AF3 boundary handling)
    # This ensures each key position is a unique atom — no duplicate boundary atoms
    left_overflow = (-key_flat_idx[:, 0]).clamp(min=0)  # [S] — how much window extends below 0
    right_overflow = (key_flat_idx[:, -1] - (num_real - 1)).clamp(min=0)  # [S] — above num_real-1
    shift = left_overflow - right_overflow  # positive = shift right, negative = shift left
    key_flat_idx = key_flat_idx + shift.unsqueeze(1)
    key_flat_idx = key_flat_idx.clamp(0, num_padded - 1)  # safety clamp for degenerate K > num_real

    # q_to_k_mask: valid if the key position has a real atom
    q_to_k_mask = padded_mask[key_flat_idx]  # [S, K]
    q_to_k_idx = key_flat_idx  # [S, K] — indices into [S*Q] flat space

    # Token index for each query/key atom (for trunk pair broadcast + offsets_valid)
    flat_token_idx = torch.arange(N, device=device).unsqueeze(1).expand(N, M).reshape(-1)  # [N*M]
    token_for_query = flat_token_idx[ta_to_q_idx]  # [S, Q]
    # Key atoms: first map to [N*M] via ta_to_q_idx, then get token
    key_nm_idx = padded_indices[key_flat_idx]  # [S, K] indices into [N*M]
    token_for_key = flat_token_idx[key_nm_idx]  # [S, K]

    # Scatter back: for each real atom in [N*M], what's its position in [S*Q]?
    q_to_ta_idx = torch.zeros(N * M, dtype=torch.long, device=device)
    q_to_ta_mask = torch.zeros(N * M, dtype=torch.bool, device=device)
    q_positions = torch.arange(num_padded, device=device)  # position in [S*Q]
    q_to_ta_idx[padded_indices[:num_real]] = q_positions[:num_real]
    q_to_ta_mask[padded_indices[:num_real]] = True

    q_to_ta_idx = q_to_ta_idx.reshape(N, M)  # [N, M]
    q_to_ta_mask = q_to_ta_mask.reshape(N, M)  # [N, M]

    return AtomSubsets(
        num_subsets=S,
        Q=Q,
        K=K,
        N=N,
        M=M,
        ta_to_q_idx=ta_to_q_idx,
        ta_to_q_mask=ta_to_q_mask,
        q_to_k_idx=q_to_k_idx,
        q_to_k_mask=q_to_k_mask,
        token_for_query=token_for_query,
        token_for_key=token_for_key,
        q_to_ta_idx=q_to_ta_idx,
        q_to_ta_mask=q_to_ta_mask,
    )


def _gather_to_subsets(arr: torch.Tensor, idx: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Gather from token-atoms [N, M, *F] to subset layout [S, Q_or_K, *F].

    idx: [S, Q_or_K] indices into flattened [N*M]
    mask: [S, Q_or_K] bool
    """
    flat = arr.reshape(-1, *arr.shape[2:])  # [N*M, *F]
    out = flat[idx]  # [S, Q_or_K, *F]
    return out * mask.reshape(*mask.shape, *([1] * (out.ndim - mask.ndim)))


def _scatter_from_subsets(arr: torch.Tensor, idx: torch.Tensor, mask: torch.Tensor, N: int, M: int) -> torch.Tensor:
    """Scatter from subset layout [S, Q, *F] back to token-atoms [N, M, *F].

    idx: [N, M] indices into flattened [S*Q]
    mask: [N, M] bool
    """
    flat = arr.reshape(-1, *arr.shape[2:])  # [S*Q, *F]
    out = flat[idx]  # [N, M, *F]
    return out * mask.reshape(*mask.shape, *([1] * (out.ndim - mask.ndim)))


def _gather_q_to_k(arr: torch.Tensor, q_to_k_idx: torch.Tensor, q_to_k_mask: torch.Tensor) -> torch.Tensor:
    """Gather from queries [S, Q, *F] to keys [S, K, *F].

    q_to_k_idx: [S, K] indices into flattened [S*Q]
    q_to_k_mask: [S, K] bool
    """
    S, Q = arr.shape[:2]
    flat = arr.reshape(S * Q, *arr.shape[2:])  # [S*Q, *F]
    out = flat[q_to_k_idx]  # [S, K, *F]
    return out * q_to_k_mask.reshape(*q_to_k_mask.shape, *([1] * (out.ndim - q_to_k_mask.ndim)))


@dataclasses.dataclass
class AtomCrossAttEncoderOutput:
    token_act: torch.Tensor  # [N_tok, per_token_channels]
    skip_connection: torch.Tensor  # [S, Q, per_atom_channels]
    queries_mask: torch.Tensor  # [S, Q] float
    queries_single_cond: torch.Tensor  # [S, Q, per_atom_channels]
    keys_mask: torch.Tensor  # [S, K] float
    keys_single_cond: torch.Tensor  # [S, K, per_atom_channels]
    pair_cond: torch.Tensor  # [S, Q, K, pair_ch]
    subsets: AtomSubsets  # precomputed indices for decoder


class AtomPerAtomConditioning(nn.Module):
    """Compute per-atom single + pair conditioning from reference structure.

    Haiku names (prefix = name):
        {name}embed_ref_pos         → embed_ref_pos
        {name}embed_ref_mask        → embed_ref_mask
        {name}embed_ref_element     → embed_ref_element
        {name}embed_ref_charge      → embed_ref_charge
        {name}embed_ref_atom_name   → embed_ref_atom_name
        {name}single_to_pair_cond_row → single_to_pair_row
        {name}single_to_pair_cond_col → single_to_pair_col
        {name}embed_pair_offsets    → embed_pair_offsets
        {name}embed_pair_distances  → embed_pair_distances
        {name}pair_mlp_1            → pair_mlp1
        {name}pair_mlp_2            → pair_mlp2
        {name}pair_mlp_3            → pair_mlp3

    Note: single_to_pair_row/col, embed_pair_offsets/distances, pair_mlp1/2/3
    are exposed as attributes so that encoder modules can reuse them for
    cross-token pair construction (matching AF3's Haiku parameter sharing).
    """

    def __init__(self, per_atom_channels: int, per_atom_pair_channels: int):
        super().__init__()
        c = per_atom_channels
        cp = per_atom_pair_channels
        self.embed_ref_pos = LinearNoBias(3, c)
        self.embed_ref_mask = LinearNoBias(1, c)
        self.embed_ref_element = LinearNoBias(128, c)
        self.embed_ref_charge = LinearNoBias(1, c)
        self.embed_ref_atom_name = LinearNoBias(64 * 4, c)
        # pair conditioning from single (accessible from encoder for cross-token pairs)
        self.single_to_pair_row = LinearNoBias(c, cp)
        self.single_to_pair_col = LinearNoBias(c, cp)
        self.embed_pair_offsets = LinearNoBias(3, cp)
        self.embed_pair_distances = LinearNoBias(1, cp)
        # small MLP on pair (accessible from encoder for cross-token pairs)
        self.pair_mlp1 = LinearNoBias(cp, cp)
        self.pair_mlp2 = LinearNoBias(cp, cp)
        self.pair_mlp3 = LinearNoBias(cp, cp)

    def single_cond(
        self,
        ref_pos: torch.Tensor,  # [N, M, 3]
        ref_mask: torch.Tensor,  # [N, M] float
        ref_element: torch.Tensor,  # [N, M] int64
        ref_charge: torch.Tensor,  # [N, M] float
        ref_atom_name_chars: torch.Tensor,  # [N, M, 4] int64
    ) -> torch.Tensor:
        """Compute per-atom single conditioning [N, M, c]."""
        N, M, _ = ref_pos.shape
        act = self.embed_ref_pos(ref_pos)
        act = act + self.embed_ref_mask(ref_mask.unsqueeze(-1))
        elem_1hot = F.one_hot(ref_element, 128).float()
        act = act + self.embed_ref_element(elem_1hot)
        act = act + self.embed_ref_charge(torch.arcsinh(ref_charge).unsqueeze(-1))
        name_1hot = F.one_hot(ref_atom_name_chars, 64).float().reshape(N, M, -1)
        act = act + self.embed_ref_atom_name(name_1hot)
        act = act * ref_mask.unsqueeze(-1)
        return act

    def apply_pair_mlp(self, pair: torch.Tensor) -> torch.Tensor:
        """Apply residual pair MLP to pair activations of any shape [..., cp]."""
        p2 = self.pair_mlp1(F.relu(pair))
        p2 = self.pair_mlp2(F.relu(p2))
        return pair + self.pair_mlp3(F.relu(p2))

    def forward(
        self,
        ref_pos: torch.Tensor,  # [N, M, 3]
        ref_mask: torch.Tensor,  # [N, M] float
        ref_element: torch.Tensor,  # [N, M] int64
        ref_charge: torch.Tensor,  # [N, M] float
        ref_atom_name_chars: torch.Tensor,  # [N, M, 4] int64
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (atom_single [N,M,c], intra_pair [N,M,M,cp]).

        The intra_pair output is NOT used by the encoder (cross-token mode).
        It exists so that per_atom's pair parameters are initialised in the
        same Haiku scope as the encoder's pair MLP (matching AF3 weight names).
        """
        act = self.single_cond(ref_pos, ref_mask, ref_element, ref_charge, ref_atom_name_chars)

        # Intra-token pair (same-token atom pairs)
        row = self.single_to_pair_row(F.relu(act))  # [N, M, cp]
        col = self.single_to_pair_col(F.relu(act))  # [N, M, cp]
        pair_act = row.unsqueeze(2) + col.unsqueeze(1)  # [N, M, M, cp]
        offsets = ref_pos.unsqueeze(2) - ref_pos.unsqueeze(1)  # [N, M, M, 3]
        sq_dists = (offsets**2).sum(-1, keepdim=True)
        pair_act = pair_act + self.embed_pair_offsets(offsets)
        pair_act = pair_act + self.embed_pair_distances(1.0 / (1.0 + sq_dists))
        pair_act = self.apply_pair_mlp(pair_act)

        return act, pair_act


class AtomCrossAttTransformer(nn.Module):
    """Cross-attention transformer for flat atom subsets.

    Queries = Q atoms from a non-overlapping subset;
    Keys = K atoms from an overlapping window centered on that subset.

    At each block, keys_act is re-derived from current queries_act by gathering
    from the key window, matching AF3's CrossAttTransformer behaviour.

    Haiku: CrossAttTransformer
        pair_input_layer_norm → pair_norm  (scale-only, no offset)
        pair_logits_projection → pair_logits_proj
        layers: stacked with __layer_stack_with_per_layer

    Converter attribute names (all per-block, indexed by ModuleList):
        attn_cond_w[i]     ← qsingle_cond_layer_norm  scale
        attn_cond_scale[i] ← qsingle_cond_scale
        attn_cond_bias[i]  ← qsingle_cond_bias
        k_cond_w[i]        ← ksingle_cond_layer_norm  scale
        k_cond_scale[i]    ← ksingle_cond_scale
        k_cond_bias[i]     ← ksingle_cond_bias
        attn_blocks[i].to_q      ← q_projection
        attn_blocks[i].to_k      ← k_projection
        attn_blocks[i].to_v      ← v_projection
        attn_blocks[i].to_g      ← gating_query
        attn_blocks[i].out_proj  ← transition2
        attn_blocks[i].zero_cond ← adaptive_zero_cond
        ffn_cond_w[i]      ← ffw_single_cond_layer_norm scale
        ffn_cond_scale[i]  ← ffw_single_cond_scale
        ffn_cond_bias[i]   ← ffw_single_cond_bias
        ffn_blocks[i].linear1   ← ffw_transition1
        ffn_blocks[i].linear2   ← ffw_transition2
        ffn_blocks[i].zero_cond ← ffw_adaptive_zero_cond
    """

    def __init__(
        self,
        per_atom_channels: int,
        per_atom_pair_channels: int,
        num_blocks: int = 3,
        num_heads: int = 4,
        num_intermediate_factor: int = 2,
    ):
        super().__init__()
        c = per_atom_channels
        cp = per_atom_pair_channels
        H = num_heads
        assert c % H == 0
        self.num_heads = H
        self.head_dim = c // H
        self.num_blocks = num_blocks

        # Pair logits (scale-only LN + no-bias projection)
        self.pair_norm_w = nn.Parameter(torch.ones(cp))
        self.pair_logits_proj = LinearNoBias(cp, num_blocks * H)

        num_int = c * num_intermediate_factor
        self.attn_blocks = nn.ModuleList()
        self.ffn_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.attn_blocks.append(
                nn.ModuleDict(
                    {
                        "to_q": nn.Linear(c, c),  # q_projection has bias
                        "to_k": LinearNoBias(c, c),
                        "to_v": LinearNoBias(c, c),
                        "to_g": LinearNoBias(c, c),  # gating_query no bias
                        "out_proj": LinearNoBias(c, c),  # transition2 no bias
                        "zero_cond": nn.Linear(c, c),  # adaptive_zero_cond has bias
                    }
                )
            )
            self.ffn_blocks.append(
                nn.ModuleDict(
                    {
                        "linear1": LinearNoBias(c, num_int * 2),
                        "linear2": LinearNoBias(num_int, c),
                        "zero_cond": nn.Linear(c, c),
                    }
                )
            )
        # Initialise zero_cond to near-zero gate (sigmoid(-2) ≈ 0.12)
        for b in self.attn_blocks:
            nn.init.zeros_(b["zero_cond"].weight)
            nn.init.constant_(b["zero_cond"].bias, -2.0)
        for b in self.ffn_blocks:
            nn.init.zeros_(b["zero_cond"].weight)
            nn.init.constant_(b["zero_cond"].bias, -2.0)

        # Shared norms (parameter-free, elementwise_affine=False)
        self.attn_norms = nn.ModuleList([nn.LayerNorm(c, elementwise_affine=False) for _ in range(num_blocks)])
        self.ffn_norms = nn.ModuleList([nn.LayerNorm(c, elementwise_affine=False) for _ in range(num_blocks)])

        # q-side AdaLN conditioning
        self.attn_cond_w = nn.ParameterList([nn.Parameter(torch.ones(c)) for _ in range(num_blocks)])
        self.attn_cond_scale = nn.ModuleList([nn.Linear(c, c) for _ in range(num_blocks)])
        self.attn_cond_bias = nn.ModuleList([LinearNoBias(c, c) for _ in range(num_blocks)])

        # k-side AdaLN conditioning (keys/values conditioned separately from queries)
        self.k_cond_w = nn.ParameterList([nn.Parameter(torch.ones(c)) for _ in range(num_blocks)])
        self.k_cond_scale = nn.ModuleList([nn.Linear(c, c) for _ in range(num_blocks)])
        self.k_cond_bias = nn.ModuleList([LinearNoBias(c, c) for _ in range(num_blocks)])

        # FFN AdaLN conditioning (on queries only)
        self.ffn_cond_w = nn.ParameterList([nn.Parameter(torch.ones(c)) for _ in range(num_blocks)])
        self.ffn_cond_scale = nn.ModuleList([nn.Linear(c, c) for _ in range(num_blocks)])
        self.ffn_cond_bias = nn.ModuleList([LinearNoBias(c, c) for _ in range(num_blocks)])

    def forward(
        self,
        queries_act: torch.Tensor,  # [S, Q, c]
        queries_mask: torch.Tensor,  # [S, Q] float
        queries_single_cond: torch.Tensor,  # [S, Q, c]
        keys_single_cond: torch.Tensor,  # [S, K, c]
        keys_mask: torch.Tensor,  # [S, K] float
        pair_cond: torch.Tensor,  # [S, Q, K, cp]
        q_to_k_idx: torch.Tensor,  # [S, K] long — indices into [S*Q]
        q_to_k_mask: torch.Tensor,  # [S, K] bool
    ) -> torch.Tensor:  # [S, Q, c]
        S, Q, C = queries_act.shape
        H = self.num_heads
        D = self.head_dim
        K = keys_mask.shape[1]

        # Pre-compute pair logits for all blocks (scale-only LN)
        pair_normed = F.layer_norm(pair_cond, [pair_cond.shape[-1]]) * self.pair_norm_w
        pair_logits = self.pair_logits_proj(pair_normed)  # [S, Q, K, nb*H]
        pair_logits = pair_logits.permute(3, 0, 1, 2)  # [nb*H, S, Q, K]
        pair_logits = pair_logits.reshape(self.num_blocks, H, S, Q, K)

        for i, (attn, ffn) in enumerate(zip(self.attn_blocks, self.ffn_blocks, strict=True)):
            pl = pair_logits[i]  # [H, S, Q, K]

            # Re-gather keys from current queries_act (AF3 does this every block)
            keys_act = _gather_q_to_k(queries_act, q_to_k_idx, q_to_k_mask)  # [S, K, c]

            # --- q-side AdaLN ---
            xn_q = self.attn_norms[i](queries_act)  # [S, Q, c]  (no params)
            cn_q = F.layer_norm(queries_single_cond, [C]) * self.attn_cond_w[i]
            xn_q = torch.sigmoid(self.attn_cond_scale[i](cn_q)) * xn_q + self.attn_cond_bias[i](cn_q)

            # --- k-side AdaLN ---
            xn_k = F.layer_norm(keys_act, [C])  # [S, K, c]  (no params)
            cn_k = F.layer_norm(keys_single_cond, [C]) * self.k_cond_w[i]
            xn_k = torch.sigmoid(self.k_cond_scale[i](cn_k)) * xn_k + self.k_cond_bias[i](cn_k)

            # --- Cross-attention ---
            q = attn["to_q"](xn_q).view(S, Q, H, D).permute(0, 2, 1, 3)  # [S,H,Q,D]
            k = attn["to_k"](xn_k).view(S, K, H, D).permute(0, 2, 1, 3)  # [S,H,K,D]
            v = attn["to_v"](xn_k).view(S, K, H, D).permute(0, 2, 1, 3)  # [S,H,K,D]

            logits = torch.matmul(q * (D**-0.5), k.transpose(-2, -1))  # [S,H,Q,K]
            logits = logits + pl.permute(1, 0, 2, 3)  # [S,H,Q,K]
            key_mask = keys_mask.unsqueeze(1).unsqueeze(2)  # [S,1,1,K]
            logits = logits + 1e9 * (key_mask - 1.0)
            w = F.softmax(logits.float(), dim=-1).to(queries_act.dtype)  # [S,H,Q,K]
            out = torch.matmul(w, v).permute(0, 2, 1, 3).reshape(S, Q, C)  # [S,Q,C]

            g = torch.sigmoid(attn["to_g"](xn_q))  # gate queries
            out = out * g
            out = attn["out_proj"](out)
            gate = torch.sigmoid(attn["zero_cond"](queries_single_cond))
            queries_act = queries_act + gate * out

            # --- FFN on queries only ---
            xn = self.ffn_norms[i](queries_act)
            cn = F.layer_norm(queries_single_cond, [C]) * self.ffn_cond_w[i]
            xn = torch.sigmoid(self.ffn_cond_scale[i](cn)) * xn + self.ffn_cond_bias[i](cn)
            ff = ffn["linear1"](xn)
            a, b = ff.chunk(2, dim=-1)
            ff = F.silu(a) * b
            ff_out = ffn["linear2"](ff)
            gate = torch.sigmoid(ffn["zero_cond"](queries_single_cond))
            queries_act = queries_act + gate * ff_out

        return queries_act


class EvoformerConditioningEncoder(nn.Module):
    """Evoformer per-atom conditioning encoder.

    Produces token-level activations from reference atom structure features.
    These are concatenated with raw token features to form the full target_feat
    input to the Evoformer (total dim = 63 raw + 384 here = 447).

    No trunk conditioning (pair/single from evoformer are not yet available).
    No noisy atom positions (uses only reference structure).

    Haiku scope prefix: 'diffuser/evoformer_conditioning_'

    Attribute names (for converter):
        per_atom          → AtomPerAtomConditioning (embed_ref_*, single_to_pair_*, ...)
        single_to_pair_row_enc   ← single_to_pair_cond_row_1
        single_to_pair_col_enc   ← single_to_pair_cond_col_1
        enc_embed_pair_offsets   ← embed_pair_offsets_1
        enc_embed_pair_distances ← embed_pair_distances_1
        enc_embed_offsets_valid  ← embed_pair_offsets_valid
        atom_transformer  → AtomCrossAttTransformer (3 blocks)
        proj_for_aggr     ← project_atom_features_for_aggr  (Linear c→384, no bias)
    """

    def __init__(
        self,
        per_atom_channels: int = 128,
        per_token_channels: int = 384,
        per_atom_pair_channels: int = 16,
        num_blocks: int = 3,
        Q: int = 32,
        K: int = 128,
    ):
        super().__init__()
        c = per_atom_channels
        cp = per_atom_pair_channels
        ct = per_token_channels
        self.Q = Q
        self.K = K

        self.per_atom = AtomPerAtomConditioning(c, cp)

        # Encoder-specific additional pair conditioning (_1 variants in AF3)
        self.single_to_pair_row_enc = LinearNoBias(c, cp)
        self.single_to_pair_col_enc = LinearNoBias(c, cp)
        self.enc_embed_pair_offsets = LinearNoBias(3, cp)
        self.enc_embed_pair_distances = LinearNoBias(1, cp)
        self.enc_embed_offsets_valid = LinearNoBias(1, cp)

        # Cross-attention transformer
        self.atom_transformer = AtomCrossAttTransformer(c, cp, num_blocks=num_blocks)

        # Project atom features → token level (masked mean)
        self.proj_for_aggr = LinearNoBias(c, ct)

    def forward(
        self,
        ref_pos: torch.Tensor,  # [N, M, 3]
        ref_mask: torch.Tensor,  # [N, M] float
        ref_element: torch.Tensor,  # [N, M] int64
        ref_charge: torch.Tensor,  # [N, M] float
        ref_atom_name_chars: torch.Tensor,  # [N, M, 4] int64
    ) -> torch.Tensor:
        """Returns token_act [N, per_token_channels]."""
        N, M, _ = ref_pos.shape

        # Build flat atom subsets
        subsets = build_atom_subsets(ref_mask.bool(), self.Q, self.K)

        # Single conditioning in token-atoms space [N, M, c]
        atom_single = self.per_atom.single_cond(ref_pos, ref_mask, ref_element, ref_charge, ref_atom_name_chars)

        # Gather to subset layouts
        queries_single = _gather_to_subsets(atom_single, subsets.ta_to_q_idx, subsets.ta_to_q_mask)  # [S, Q, c]
        queries_mask = subsets.ta_to_q_mask.float()  # [S, Q]
        keys_single = _gather_q_to_k(queries_single, subsets.q_to_k_idx, subsets.q_to_k_mask)  # [S, K, c]
        keys_mask = subsets.q_to_k_mask.float()  # [S, K]

        # Gather ref_pos to subset layouts
        queries_ref_pos = _gather_to_subsets(ref_pos, subsets.ta_to_q_idx, subsets.ta_to_q_mask)  # [S, Q, 3]
        keys_ref_pos = _gather_q_to_k(queries_ref_pos, subsets.q_to_k_idx, subsets.q_to_k_mask)  # [S, K, 3]

        # offsets_valid: same-token atom pairs (same reference frame)
        offsets_valid = subsets.token_for_query.unsqueeze(2) == subsets.token_for_key.unsqueeze(1)  # [S, Q, K]

        # Build cross-subset pair_cond [S, Q, K, cp]
        # AF3 only uses _1 variant weights for cross-subset pair
        # (base per_atom pair from _per_atom_conditioning is discarded)
        row = self.single_to_pair_row_enc(F.relu(queries_single))  # [S, Q, cp]
        col = self.single_to_pair_col_enc(F.relu(keys_single))  # [S, K, cp]
        atom_pair = row.unsqueeze(2) + col.unsqueeze(1)  # [S, Q, K, cp]

        offsets = queries_ref_pos.unsqueeze(2) - keys_ref_pos.unsqueeze(1)  # [S, Q, K, 3]
        sq_dists = (offsets**2).sum(-1, keepdim=True)
        ov = offsets_valid.unsqueeze(-1)  # [S, Q, K, 1]
        atom_pair = atom_pair + self.enc_embed_pair_offsets(offsets) * ov
        atom_pair = atom_pair + self.enc_embed_pair_distances(1.0 / (1.0 + sq_dists)) * ov
        atom_pair = atom_pair + self.enc_embed_offsets_valid(ov.float())

        # Pair MLP applied LAST (after all additive terms, matching AF3)
        atom_pair = self.per_atom.apply_pair_mlp(atom_pair)

        # Cross-attention transformer
        queries_act = self.atom_transformer(
            queries_act=queries_single,
            queries_mask=queries_mask,
            queries_single_cond=queries_single,
            keys_single_cond=keys_single,
            keys_mask=keys_mask,
            pair_cond=atom_pair,
            q_to_k_idx=subsets.q_to_k_idx,
            q_to_k_mask=subsets.q_to_k_mask,
        )
        queries_act = queries_act * queries_mask.unsqueeze(-1)

        # Scatter back to token-atoms and aggregate to token level (masked mean)
        token_atoms_act = _scatter_from_subsets(
            self.proj_for_aggr(queries_act), subsets.q_to_ta_idx, subsets.q_to_ta_mask, N, M
        )  # [N, M, ct]
        denom = ref_mask.sum(-1, keepdim=True).clamp(min=1e-6)
        token_act = (F.relu(token_atoms_act) * ref_mask.unsqueeze(-1)).sum(-2) / denom

        return token_act  # [N, ct=384]


class AtomEncoder(nn.Module):
    """Atom cross-attention encoder: atoms → tokens (for diffusion head).

    Conditions on trunk single/pair representations + noisy atom positions.

    Haiku scope prefix: 'diffusion_*' (within diffuser/~/diffusion_head/)

    Attribute names (for converter):
        per_atom                    → AtomPerAtomConditioning
        lnorm_trunk_single_w        ← lnorm_trunk_single_cond  (scale only)
        embed_trunk_single          ← embed_trunk_single_cond
        lnorm_trunk_pair_w          ← lnorm_trunk_pair_cond    (scale only)
        embed_trunk_pair            ← embed_trunk_pair_cond
        single_to_pair_row_enc      ← single_to_pair_cond_row_1
        single_to_pair_col_enc      ← single_to_pair_cond_col_1
        enc_embed_pair_offsets      ← embed_pair_offsets_1
        enc_embed_pair_distances    ← embed_pair_distances_1
        enc_embed_offsets_valid     ← embed_pair_offsets_valid
        _pos_proj                   ← atom_positions_to_features (Linear(3,c))
        atom_transformer            → AtomCrossAttTransformer
        proj_for_aggr               ← project_atom_features_for_aggr
    """

    def __init__(
        self,
        per_atom_channels: int,
        per_token_channels: int,
        per_atom_pair_channels: int,
        atom_transformer_num_blocks: int = 3,
        Q: int = 32,
        K: int = 128,
    ):
        super().__init__()
        c = per_atom_channels
        cp = per_atom_pair_channels
        ct = per_token_channels
        self.Q = Q
        self.K = K

        self.per_atom = AtomPerAtomConditioning(c, cp)

        # Trunk single conditioning
        self.lnorm_trunk_single_w = nn.Parameter(torch.ones(384))
        self.embed_trunk_single = LinearNoBias(384, c)

        # Trunk pair conditioning
        self.lnorm_trunk_pair_w = nn.Parameter(torch.ones(128))
        self.embed_trunk_pair = LinearNoBias(128, cp)

        # Encoder-specific additional pair conditioning (_1 variants)
        self.single_to_pair_row_enc = LinearNoBias(c, cp)
        self.single_to_pair_col_enc = LinearNoBias(c, cp)
        self.enc_embed_pair_offsets = LinearNoBias(3, cp)
        self.enc_embed_pair_distances = LinearNoBias(1, cp)
        self.enc_embed_offsets_valid = LinearNoBias(1, cp)

        # Project noisy atom positions (3D) → per_atom_channels
        self._pos_proj = LinearNoBias(3, c)

        # Cross-attention transformer
        self.atom_transformer = AtomCrossAttTransformer(c, cp, num_blocks=atom_transformer_num_blocks)

        # Project atom features → token level
        self.proj_for_aggr = LinearNoBias(c, ct)

    def forward(
        self,
        ref_pos: torch.Tensor,  # [N, M, 3]
        ref_mask: torch.Tensor,  # [N, M] float
        ref_element: torch.Tensor,  # [N, M] int64
        ref_charge: torch.Tensor,  # [N, M] float
        ref_atom_name_chars: torch.Tensor,  # [N, M, 4] int64
        noisy_atom_pos: torch.Tensor | None,  # [N, M, 3] or None
        trunk_single: torch.Tensor,  # [N, c_single=384]
        trunk_pair: torch.Tensor,  # [N, N, c_pair=128]
    ) -> AtomCrossAttEncoderOutput:
        N, M, _ = ref_pos.shape

        # Build flat atom subsets
        subsets = build_atom_subsets(ref_mask.bool(), self.Q, self.K)

        # --- Single conditioning ---
        atom_single = self.per_atom.single_cond(
            ref_pos, ref_mask, ref_element, ref_charge, ref_atom_name_chars
        )  # [N, M, c]

        # Broadcast trunk single to all atoms of each token
        ts_norm = F.layer_norm(trunk_single, [trunk_single.shape[-1]]) * self.lnorm_trunk_single_w
        ts_proj = self.embed_trunk_single(ts_norm)  # [N, c]
        atom_single = atom_single + ts_proj.unsqueeze(1)

        # --- Gather to subset layouts ---
        queries_single = _gather_to_subsets(atom_single, subsets.ta_to_q_idx, subsets.ta_to_q_mask)  # [S, Q, c]
        queries_mask = subsets.ta_to_q_mask.float()  # [S, Q]
        keys_single = _gather_q_to_k(queries_single, subsets.q_to_k_idx, subsets.q_to_k_mask)  # [S, K, c]
        keys_mask = subsets.q_to_k_mask.float()  # [S, K]

        # Gather ref_pos to subset layouts
        queries_ref_pos = _gather_to_subsets(ref_pos, subsets.ta_to_q_idx, subsets.ta_to_q_mask)  # [S, Q, 3]
        keys_ref_pos = _gather_q_to_k(queries_ref_pos, subsets.q_to_k_idx, subsets.q_to_k_mask)  # [S, K, 3]

        # offsets_valid: same-token atom pairs (same reference frame)
        offsets_valid = subsets.token_for_query.unsqueeze(2) == subsets.token_for_key.unsqueeze(1)  # [S, Q, K]

        # --- Trunk pair conditioning broadcasted to cross-subset atom pairs ---
        tp_norm = F.layer_norm(trunk_pair, [trunk_pair.shape[-1]]) * self.lnorm_trunk_pair_w
        tp_proj = self.embed_trunk_pair(tp_norm)  # [N, N, cp]
        # Gather trunk_pair[token_for_query[s,q], token_for_key[s,k], :] for each (s, q, k)
        pair_idx = subsets.token_for_query.unsqueeze(2) * N + subsets.token_for_key.unsqueeze(1)  # [S, Q, K]
        trunk_flat = tp_proj.reshape(N * N, -1)  # [N*N, cp]
        trunk_pair_contrib = trunk_flat[pair_idx]  # [S, Q, K, cp]
        pair_valid = subsets.ta_to_q_mask.unsqueeze(2) & subsets.q_to_k_mask.unsqueeze(1)  # [S, Q, K]
        trunk_pair_contrib = trunk_pair_contrib * pair_valid.unsqueeze(-1)

        # --- Build cross-subset pair_cond [S, Q, K, cp] ---
        # AF3 only uses _1 variant weights for cross-subset pair
        # (base per_atom pair from _per_atom_conditioning is discarded)
        row = self.single_to_pair_row_enc(F.relu(queries_single))  # [S, Q, cp]
        col = self.single_to_pair_col_enc(F.relu(keys_single))  # [S, K, cp]
        atom_pair = row.unsqueeze(2) + col.unsqueeze(1)  # [S, Q, K, cp]

        offsets = queries_ref_pos.unsqueeze(2) - keys_ref_pos.unsqueeze(1)  # [S, Q, K, 3]
        sq_dists = (offsets**2).sum(-1, keepdim=True)
        ov = offsets_valid.unsqueeze(-1)  # [S, Q, K, 1]
        atom_pair = atom_pair + self.enc_embed_pair_offsets(offsets) * ov
        atom_pair = atom_pair + self.enc_embed_pair_distances(1.0 / (1.0 + sq_dists)) * ov
        atom_pair = atom_pair + self.enc_embed_offsets_valid(ov.float())
        # Add trunk pair contribution
        atom_pair = atom_pair + trunk_pair_contrib

        # Pair MLP applied LAST (after all additive terms, matching AF3)
        atom_pair = self.per_atom.apply_pair_mlp(atom_pair)

        # --- queries_act: start from queries_single, add projected noisy positions ---
        if noisy_atom_pos is not None:
            noisy_q = _gather_to_subsets(noisy_atom_pos, subsets.ta_to_q_idx, subsets.ta_to_q_mask)  # [S, Q, 3]
            queries_act = queries_single + self._pos_proj(noisy_q) * queries_mask.unsqueeze(-1)
        else:
            queries_act = queries_single

        # --- Cross-attention transformer ---
        queries_act = self.atom_transformer(
            queries_act=queries_act,
            queries_mask=queries_mask,
            queries_single_cond=queries_single,
            keys_single_cond=keys_single,
            keys_mask=keys_mask,
            pair_cond=atom_pair,
            q_to_k_idx=subsets.q_to_k_idx,
            q_to_k_mask=subsets.q_to_k_mask,
        )
        queries_act = queries_act * queries_mask.unsqueeze(-1)

        # Project and aggregate to token level
        atom_proj = _scatter_from_subsets(
            self.proj_for_aggr(queries_act), subsets.q_to_ta_idx, subsets.q_to_ta_mask, N, M
        )  # [N, M, ct]
        denom = ref_mask.sum(-1, keepdim=True).clamp(min=1e-6)
        token_act = (F.relu(atom_proj) * ref_mask.unsqueeze(-1)).sum(-2) / denom

        return AtomCrossAttEncoderOutput(
            token_act=token_act,
            skip_connection=queries_act,
            queries_mask=queries_mask,
            queries_single_cond=queries_single,
            keys_mask=keys_mask,
            keys_single_cond=keys_single,
            pair_cond=atom_pair,
            subsets=subsets,
        )


class AtomDecoder(nn.Module):
    """Atom cross-attention decoder: tokens → atom position updates.

    Haiku scope prefix: 'diffusion_*' (within diffuser/~/diffusion_head/)

    Attribute names (for converter):
        proj_tokens    ← project_token_features_for_broadcast
        atom_transformer → AtomCrossAttTransformer
        atom_norm_w    ← atom_features_layer_norm  (scale only, no offset)
        pos_update     ← atom_features_to_position_update
    """

    def __init__(
        self,
        per_atom_channels: int,
        per_token_channels: int,
        per_atom_pair_channels: int,
        atom_transformer_num_blocks: int = 3,
    ):
        super().__init__()
        c = per_atom_channels
        ct = per_token_channels
        cp = per_atom_pair_channels

        self.proj_tokens = LinearNoBias(ct, c)
        self.atom_transformer = AtomCrossAttTransformer(c, cp, num_blocks=atom_transformer_num_blocks)
        self.atom_norm_w = nn.Parameter(torch.ones(c))
        self.pos_update = LinearNoBias(c, 3)

    def forward(
        self,
        token_act: torch.Tensor,  # [N, ct]
        enc: AtomCrossAttEncoderOutput,
    ) -> torch.Tensor:  # [N, M, 3]
        N, M = enc.subsets.N, enc.subsets.M
        c = enc.skip_connection.shape[-1]

        # Project token features → per_atom_channels, broadcast to atoms
        token_proj = self.proj_tokens(token_act)  # [N, c]
        # Broadcast to token-atoms [N, M, c], then gather to queries [S, Q, c]
        token_atoms = token_proj.unsqueeze(1).expand(N, M, c)
        queries_act = _gather_to_subsets(token_atoms, enc.subsets.ta_to_q_idx, enc.subsets.ta_to_q_mask)
        queries_act = queries_act + enc.skip_connection
        queries_act = queries_act * enc.queries_mask.unsqueeze(-1)

        # Cross-attention transformer (reuses encoder's pair/single conditioning)
        queries_act = self.atom_transformer(
            queries_act=queries_act,
            queries_mask=enc.queries_mask,
            queries_single_cond=enc.queries_single_cond,
            keys_single_cond=enc.keys_single_cond,
            keys_mask=enc.keys_mask,
            pair_cond=enc.pair_cond,
            q_to_k_idx=enc.subsets.q_to_k_idx,
            q_to_k_mask=enc.subsets.q_to_k_mask,
        )
        queries_act = queries_act * enc.queries_mask.unsqueeze(-1)

        # Normalize and project to 3D position updates
        queries_act = F.layer_norm(queries_act, [c]) * self.atom_norm_w
        pos_update_q = self.pos_update(queries_act)  # [S, Q, 3]

        # Scatter back to token-atoms [N, M, 3]
        pos_update = _scatter_from_subsets(pos_update_q, enc.subsets.q_to_ta_idx, enc.subsets.q_to_ta_mask, N, M)
        return pos_update
