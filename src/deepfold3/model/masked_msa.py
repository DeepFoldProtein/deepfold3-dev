"""Masked-MSA (BERT) learning — masking transform + prediction head.

A BERT-style masked-language-model objective on the MSA, with a
*single-query MSA cross-attention* enrichment to the head:

    h_i        = Attn(q = s_i, k/v = m[:, i])     # token i attends its MSA column
    logits_ri  = Linear(MLP([m_ri, s_i, h_i]))    # per (row r, token i)

* ``apply_bert_mask`` corrupts a fraction of MSA entries before the MSA
  stack sees them and returns ``(bert_msa, true_msa, bert_mask)``.
* ``MaskedMSAHead`` predicts the original residue identity at every
  (row, token) from the final MSA-stack representation ``m`` and the trunk
  single representation ``s``.

The DF3 MSA vocabulary is 32 classes (``const.MSA_VOCAB_SIZE``); indices
0–30 are real tokens and index 31 (``const.MSA_MASK``) is the BERT mask
token, so corruption never grows the 32-wide one-hot input feature.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from deepfold3.data import const
from deepfold3.model.config import MaskedMSAConfig
from deepfold3.model.primitives import LayerNorm, LinearNoBias, af3_linear_init_


def _masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    """Softmax over ``dim`` that ignores ``mask == 0`` positions, NaN-safe.

    Computed in fp32. Invalid positions are pushed to ``-inf`` before the
    softmax, then explicitly zeroed and the row renormalised — so a row whose
    entries are *all* masked yields all-zero weights (rather than a uniform /
    NaN distribution) instead of leaking padding into the attention output.
    """
    logits = logits.float()
    mask = mask.bool()
    logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    probs = logits.softmax(dim=dim)
    probs = torch.where(mask, probs, torch.zeros_like(probs))
    denom = probs.sum(dim=dim, keepdim=True).clamp_min(1e-8)
    return probs / denom


def _random_aa_probs(vocab: int, device: torch.device) -> torch.Tensor:
    """Uniform-over-amino-acids replacement distribution.

    Puts 0.05 on each of the 20 standard protein amino-acid MSA ids (0–19) and
    0 elsewhere. Nucleic / UNK / gap columns get their replacement mass from the
    ``profile`` and ``same`` mixture components instead.
    """
    p = torch.zeros(vocab, dtype=torch.float32, device=device)
    p[:20] = 0.05
    return p


@torch.no_grad()
def apply_bert_mask(
    msa: torch.Tensor,  # [..., S, N] long
    msa_mask: torch.Tensor,  # [..., S, N] bool/float — valid (non-padding) entries
    profile: torch.Tensor | None,  # [..., N, P] float per-token residue frequency, or None
    cfg: MaskedMSAConfig,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """BERT-corrupt an MSA (32-class DF3 vocab).

    Builds the per-entry categorical replacement distribution

        p = uniform_prob · random_aa + profile_prob · profile + same_prob · one_hot(msa)

    with the remaining mass ``1 - (uniform + profile + same)`` assigned to the
    ``MSA_MASK`` token, samples a replacement per entry, and applies it at a
    random ``replace_fraction`` of *valid* entries.

    Args:
        msa: Integer MSA tokens ``[..., S, N]`` (values in 0..30).
        msa_mask: Valid-entry mask ``[..., S, N]`` — padding/absent entries are
            never selected for masking (and would not be supervised anyway).
        profile: Per-token residue frequency ``[..., N, P]`` (P≈31, aligned with
            MSA ids 0..30). ``None`` drops the profile component.
        cfg: Masking probabilities.
        generator: Optional device RNG for reproducible masking.

    Returns:
        ``(bert_msa, true_msa, bert_mask)``:
            * ``bert_msa``  ``[..., S, N]`` long — corrupted MSA fed to the model.
            * ``true_msa``  ``[..., S, N]`` long — original tokens (loss target).
            * ``bert_mask`` ``[..., S, N]`` float — 1.0 at corrupted entries.
    """
    vocab = const.MSA_VOCAB_SIZE
    device = msa.device

    same = F.one_hot(msa.long(), vocab).float()  # [..., S, N, vocab]
    probs = cfg.same_prob * same + cfg.uniform_prob * _random_aa_probs(vocab, device)

    if profile is not None and cfg.profile_prob > 0.0:
        prof = profile.float()
        if prof.shape[-1] < vocab:
            prof = F.pad(prof, (0, vocab - prof.shape[-1]))
        else:
            prof = prof[..., :vocab]
        # Broadcast per-token profile across the S MSA rows.
        probs = probs + cfg.profile_prob * prof.unsqueeze(-3)

    mask_prob = 1.0 - cfg.uniform_prob - cfg.profile_prob - cfg.same_prob
    # Remaining categorical mass → the MSA_MASK column (additive: all other
    # components leave column MSA_MASK at 0).
    probs = probs.clone()
    probs[..., const.MSA_MASK] = probs[..., const.MSA_MASK] + mask_prob

    # Sample one replacement token per entry. ``multinomial`` needs 2-D input
    # and renormalises rows itself, but clamp/normalise defensively for bf16
    # profiles whose rows may not sum to exactly 1.
    flat = probs.reshape(-1, vocab).clamp_min(0.0)
    flat = flat / flat.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    sampled = torch.multinomial(flat, 1, generator=generator).reshape(msa.shape)

    rand = torch.rand(msa.shape, device=device, generator=generator)
    mask_position = (rand < cfg.replace_fraction) & msa_mask.bool()

    bert_msa = torch.where(mask_position, sampled.to(msa.dtype), msa)
    # ``true_msa`` is the loss target — clone so a later in-place transform on
    # the source ``msa`` tensor (anywhere in the pipeline) can't perturb it.
    true_msa = msa.clone()
    return bert_msa, true_msa, mask_position.to(torch.float32)


class MaskedMSAHead(nn.Module):
    """Single-query MSA cross-attention head for masked-MSA prediction.

    For each token ``i`` the single representation ``s_i`` queries the token's
    MSA column ``m[:, i]`` (keys/values over the ``S`` MSA rows) to form a
    context vector ``h_i``; an MLP over the concatenation ``[m_ri, s_i, h_i]``
    then predicts the original residue identity at every (row ``r``, token
    ``i``).

    Inputs are LayerNorm-ed; the q/k/v + MLP projections are bias-free with AF3
    ``linear`` init (random, not zeros — the head learns from scratch on
    warm-start). The output logit projection optionally carries a bias
    (``cfg.out_bias``, zero-init) to absorb the strong MSA class prior.
    Padding / invalid MSA rows are excluded from the cross-attention via a
    NaN-safe masked softmax (see :func:`forward`).
    """

    def __init__(self, cfg: MaskedMSAConfig, c_msa: int, c_single: int):
        super().__init__()
        self.cfg = cfg
        self.num_heads = int(cfg.num_heads)
        self.head_dim = int(cfg.head_dim)
        d = self.num_heads * self.head_dim
        self._attn_scale = 1.0 / math.sqrt(self.head_dim)

        self.msa_norm = LayerNorm(c_msa)
        self.single_norm = LayerNorm(c_single)

        # Cross-attention: query from single (per token), key/value from MSA.
        self.q_proj = LinearNoBias(c_single, d)
        self.k_proj = LinearNoBias(c_msa, d)
        self.v_proj = LinearNoBias(c_msa, d)

        # MLP over [m_ri (c_msa), s_i (c_single), h_i (d)] → hidden → logits.
        mlp_in = c_msa + c_single + d
        self.mlp1 = LinearNoBias(mlp_in, cfg.hidden_dim, initializer="relu")
        self.mlp2 = LinearNoBias(cfg.hidden_dim, cfg.hidden_dim, initializer="relu")
        # Output head: optional zero-init bias captures the per-class MSA prior
        # (gap / UNK / residue frequencies). Weight uses the AF3 ``linear`` init
        # to match the bias-free projections.
        self.out = nn.Linear(cfg.hidden_dim, cfg.c_out, bias=cfg.out_bias)
        af3_linear_init_(self.out.weight, "linear")
        if cfg.out_bias:
            nn.init.zeros_(self.out.bias)

    def forward(
        self,
        msa: torch.Tensor,  # [..., S, N, c_msa] — final MSA-stack representation
        single: torch.Tensor,  # [..., N, c_single] — trunk single representation
        msa_mask: torch.Tensor | None = None,  # [..., S, N] — valid (non-padding) MSA entries
    ) -> torch.Tensor:
        """Return masked-MSA logits ``[..., S, N, c_out]``.

        ``msa_mask`` (when given) restricts the cross-attention key/value set to
        valid MSA entries per token, so padding rows never leak into ``h_i``.
        """
        m = self.msa_norm(msa)
        s = self.single_norm(single)

        H, Dh = self.num_heads, self.head_dim
        q = self.q_proj(s).unflatten(-1, (H, Dh))  # [..., N, H, Dh]
        k = self.k_proj(m).unflatten(-1, (H, Dh))  # [..., S, N, H, Dh]
        v = self.v_proj(m).unflatten(-1, (H, Dh))  # [..., S, N, H, Dh]

        # Attention over the MSA-row axis S, independently per token i.
        attn = torch.einsum("...nhd,...snhd->...nhs", q, k) * self._attn_scale  # [..., N, H, S]
        if msa_mask is not None:
            # [..., S, N] → [..., N, 1, S] (broadcast over the head axis).
            kv_mask = msa_mask.bool().transpose(-1, -2).unsqueeze(-2)
            attn = _masked_softmax(attn, kv_mask, dim=-1).to(v.dtype)
        else:
            attn = attn.float().softmax(dim=-1).to(v.dtype)
        h = torch.einsum("...nhs,...snhd->...nhd", attn, v)  # [..., N, H, Dh]
        h = h.flatten(-2)  # [..., N, H*Dh]

        # First MLP linear over [m_ri, s_i, h_i], computed *factored* so the
        # [..., S, N, c_msa+c_single+d] concat is never materialised (it is the
        # head's largest tensor — ~0.9 GB at B=2/S=1024/N=384). Exactly
        # ``Linear([m, s_b, h_b]) = m·Wₘᵀ + s·Wₛᵀ + h·W_hᵀ``; since ``s`` and
        # ``h`` are per-token, their projections are computed once at [..., N, ·]
        # and broadcast-added across the S rows (also fewer FLOPs than projecting
        # the S-broadcast copies). Slicing the single ``mlp1.weight`` keeps the
        # parameter (and checkpoint key) layout unchanged.
        cm, cs = m.shape[-1], s.shape[-1]
        w1 = self.mlp1.weight  # [hidden, c_msa + c_single + d]
        per_token = F.linear(s, w1[:, cm : cm + cs]) + F.linear(h, w1[:, cm + cs :])  # [..., N, hidden]
        x = F.linear(m, w1[:, :cm]) + per_token.unsqueeze(-3)  # [..., S, N, hidden]

        x = F.relu(x)
        x = F.relu(self.mlp2(x))
        return self.out(x)  # [..., S, N, c_out]
