"""Minimal ESM2 encoder, weight-compatible with ``facebook/esm2_*`` on the Hub.

Reimplemented rather than imported so we control the attention mask: the
experiment this exists for turns on whether the PLM may attend across chain
boundaries, and HF's ``EsmModel`` only accepts a 2D padding mask. Here chains
are packed into one sequence and ``sequence_id`` restricts attention to
block-diagonal — the same scheme ESMC uses (``sequence_id`` → varlen), so
"per-chain" and "joined" differ by one argument rather than by a code path.

Reference: ``references/biohub-transformers/src/transformers/models/esm/
modeling_esm.py``. Deviations from it are noted inline; there are none that
change the numerics.

Packing note: RoPE makes the attention logit a function of ``pos_i - pos_j``,
so a chain's absolute offset inside the packed sequence cancels. Positions
therefore do **not** need to restart per chain — block-diagonal masking alone
reproduces the per-chain result exactly.

Not implemented: the LM head, the contact-prediction head, and absolute /
relative-key position embeddings. ESM2 checkpoints are all ``rotary``.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepfold3.data import const

logger = logging.getLogger(__name__)

#: ESM2's vocabulary, positions 4.. (ESMC shares it).
ESM2_TOKENS = "LAGVSERTIDPKQNFYMHWCXBUZO"

_THREE_TO_ONE = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "UNK": "X",
}


@dataclass(frozen=True)
class PLMTokenizer:
    """Special-token ids plus the ``res_type`` → token-id table for one PLM.

    :class:`~deepfold3.model.plm.FrozenPLM` packs and scatters against this
    rather than a hard-coded vocabulary, so a backend with different special
    tokens — ESMC, or a complex-trained model of our own — is a drop-in
    replacement.

    ``cls_id`` and ``eos_id`` are optional: a model trained without sentinels
    gets its chains packed back to back. Chain separation never depends on
    them — ``sequence_id`` is built from chain membership while packing, not
    recovered from token values — so a sentinel-free tokenizer keeps exactly
    the same per-chain semantics.

    ``pad_id`` is required because examples of different length share a batch.
    A model with no reserved padding id can name any unused id here; those
    positions are masked out before attention either way.
    """

    pad_id: int
    unk_id: int
    mask_id: int
    res_type_to_id: torch.Tensor
    cls_id: int | None = None
    eos_id: int | None = None

    @property
    def num_res_types(self) -> int:
        return int(self.res_type_to_id.numel())


def esm2_tokenizer() -> PLMTokenizer:
    """ESM2/ESMC tokenizer: ``<cls>``=0, ``<pad>``=1, ``<eos>``=2, ``<unk>``=3, ``<mask>``=32.

    Non-protein ``res_type`` values map to ``<unk>``; they never reach the
    backend (the frontend keeps only ``MOL_PROTEIN`` tokens), so the value is
    only a safe default.
    """
    table = torch.full((len(const.RES_TYPE_ORDER),), 3, dtype=torch.long)
    for name, rt in const.RES_TYPE_ORDER.items():
        one = _THREE_TO_ONE.get(name)
        if one is not None:
            table[rt] = 4 + ESM2_TOKENS.index(one)
    return PLMTokenizer(pad_id=1, unk_id=3, mask_id=32, res_type_to_id=table, cls_id=0, eos_id=2)


@dataclass
class ESM2Config:
    hidden_size: int = 1280
    num_hidden_layers: int = 33
    num_attention_heads: int = 20
    intermediate_size: int = 5120
    vocab_size: int = 33
    layer_norm_eps: float = 1e-5
    pad_token_id: int = 1
    mask_token_id: int = 32
    token_dropout: bool = True
    emb_layer_norm_before: bool = False

    @classmethod
    def from_json(cls, path: str | Path) -> ESM2Config:
        raw = json.loads(Path(path).read_text())
        fields = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in raw.items() if k in fields})


def _gelu(x: torch.Tensor) -> torch.Tensor:
    """ESM's own GELU. ``F.gelu`` is subtly different and shifts the outputs."""
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``q``/``k``: ``[B, H, L, D]``."""
        t = torch.arange(k.shape[-2], device=k.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos, sin = emb.cos()[None, None], emb.sin()[None, None]
        return (
            (q * cos + _rotate_half(q) * sin).to(q.dtype),
            (k * cos + _rotate_half(k) * sin).to(k.dtype),
        )


class ESM2SelfAttention(nn.Module):
    def __init__(self, cfg: ESM2Config) -> None:
        super().__init__()
        self.num_heads = cfg.num_attention_heads
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads
        self.query = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.key = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.value = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.rotary_embeddings = RotaryEmbedding(self.head_dim)

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor | None) -> torch.Tensor:
        B, L, _ = x.shape
        shape = (B, L, self.num_heads, self.head_dim)
        q = self.query(x).view(shape).transpose(1, 2)
        k = self.key(x).view(shape).transpose(1, 2)
        v = self.value(x).view(shape).transpose(1, 2)
        # ESM scales the *query* before RoPE rather than scaling the logits
        # after; with rotary the two are not interchangeable.
        q = q * self.head_dim**-0.5
        q, k = self.rotary_embeddings(q, k)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias, scale=1.0)
        return out.transpose(1, 2).reshape(B, L, -1)


class ESM2Layer(nn.Module):
    def __init__(self, cfg: ESM2Config) -> None:
        super().__init__()
        self.attention_ln = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.self_attn = ESM2SelfAttention(cfg)
        self.attn_out = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.ffn_ln = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.fc1 = nn.Linear(cfg.hidden_size, cfg.intermediate_size)
        self.fc2 = nn.Linear(cfg.intermediate_size, cfg.hidden_size)

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor | None) -> torch.Tensor:
        h = self.self_attn(self.attention_ln(x), attn_bias)
        x = x + self.attn_out(h)
        h = _gelu(self.fc1(self.ffn_ln(x)))
        return x + self.fc2(h)


class ESM2Encoder(nn.Module):
    """ESM2 trunk returning every hidden state.

    Args:
        cfg: Architecture description; ``ESM2Config.from_json`` reads the Hub's
            ``config.json`` directly.
    """

    #: Vocabulary and ``res_type`` mapping belong to the model, not to the
    #: frontend that packs sequences for it.
    tokenizer = staticmethod(esm2_tokenizer)

    def __init__(self, cfg: ESM2Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.word_embeddings = nn.Embedding(cfg.vocab_size, cfg.hidden_size, padding_idx=cfg.pad_token_id)
        self.emb_ln_before = (
            nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps) if cfg.emb_layer_norm_before else None
        )
        self.layers = nn.ModuleList(ESM2Layer(cfg) for _ in range(cfg.num_hidden_layers))
        self.emb_ln_after = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)

    def _attn_bias(
        self,
        input_ids: torch.Tensor,
        sequence_id: torch.Tensor | None,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        """Additive ``[B, 1, L, L]`` mask: padding always blocked, chains too
        when ``sequence_id`` is given.

        ``sequence_id`` is the whole experiment: pass it and each chain sees
        only itself (ESMFold2's behaviour); omit it and attention spans the
        packed sequence.
        """
        L = input_ids.shape[1]
        pad = input_ids.ne(self.cfg.pad_token_id)  # [B, L]
        allowed = pad[:, None, :].expand(-1, L, -1)
        if sequence_id is not None:
            same = sequence_id[:, :, None] == sequence_id[:, None, :]
            allowed = allowed & same & (sequence_id[:, :, None] >= 0)
        # A padding *query* can end up with no permitted key, which is a row of
        # -inf and therefore a NaN row out of softmax. Current SDPA happens to
        # return zeros there, but that is not contractual — let such rows attend
        # to themselves. Their output is masked off downstream either way.
        eye = torch.eye(L, dtype=torch.bool, device=input_ids.device)
        allowed = allowed | (~allowed.any(-1, keepdim=True) & eye)
        return torch.zeros_like(allowed, dtype=dtype).masked_fill_(~allowed, float("-inf")).unsqueeze(1)

    def forward(
        self,
        input_ids: torch.Tensor,  # [B, L]
        sequence_id: torch.Tensor | None = None,  # [B, L], -1 for padding
    ) -> torch.Tensor:
        """Returns ``[B, L, num_hidden_layers + 1, hidden_size]``."""
        pad_mask = input_ids.ne(self.cfg.pad_token_id)
        x = self.word_embeddings(input_ids)

        if self.cfg.token_dropout:
            # ESM zeroes masked positions and rescales the rest, so that a
            # sample with no masks matches the training-time expectation. With
            # no mask tokens this is a constant 0.88 factor — it is not a
            # no-op, and dropping it would silently shift every embedding.
            x = x.masked_fill((input_ids == self.cfg.mask_token_id).unsqueeze(-1), 0.0)
            mask_ratio_train = 0.15 * 0.8
            src_lengths = pad_mask.sum(-1)
            observed = (input_ids == self.cfg.mask_token_id).sum(-1).float() / src_lengths
            x = (x * (1 - mask_ratio_train) / (1 - observed)[:, None, None]).to(x.dtype)

        if self.emb_ln_before is not None:
            x = self.emb_ln_before(x)
        x = (x * pad_mask.unsqueeze(-1)).to(x.dtype)

        attn_bias = self._attn_bias(input_ids, sequence_id, x.dtype)
        states = [x]
        for layer in self.layers:
            x = layer(x, attn_bias)
            states.append(x)
        states[-1] = self.emb_ln_after(x)
        return torch.stack(states, dim=-2)

    # -- weights ---------------------------------------------------------

    @torch.no_grad()
    def load_hf_safetensors(self, path: str | Path) -> None:
        """Load ``model.safetensors`` from a ``facebook/esm2_*`` snapshot."""
        from safetensors.torch import load_file

        sd = load_file(str(path))
        prefix = "esm." if any(k.startswith("esm.") for k in sd) else ""

        def take(name: str) -> torch.Tensor:
            return sd[f"{prefix}{name}"]

        self.word_embeddings.weight.copy_(take("embeddings.word_embeddings.weight"))
        if self.emb_ln_before is not None:
            self.emb_ln_before.weight.copy_(take("embeddings.layer_norm.weight"))
            self.emb_ln_before.bias.copy_(take("embeddings.layer_norm.bias"))
        self.emb_ln_after.weight.copy_(take("encoder.emb_layer_norm_after.weight"))
        self.emb_ln_after.bias.copy_(take("encoder.emb_layer_norm_after.bias"))

        for i, layer in enumerate(self.layers):
            p = f"encoder.layer.{i}"
            layer.attention_ln.weight.copy_(take(f"{p}.attention.LayerNorm.weight"))
            layer.attention_ln.bias.copy_(take(f"{p}.attention.LayerNorm.bias"))
            for src, dst in (
                ("query", layer.self_attn.query),
                ("key", layer.self_attn.key),
                ("value", layer.self_attn.value),
            ):
                dst.weight.copy_(take(f"{p}.attention.self.{src}.weight"))
                dst.bias.copy_(take(f"{p}.attention.self.{src}.bias"))
            layer.attn_out.weight.copy_(take(f"{p}.attention.output.dense.weight"))
            layer.attn_out.bias.copy_(take(f"{p}.attention.output.dense.bias"))
            layer.ffn_ln.weight.copy_(take(f"{p}.LayerNorm.weight"))
            layer.ffn_ln.bias.copy_(take(f"{p}.LayerNorm.bias"))
            layer.fc1.weight.copy_(take(f"{p}.intermediate.dense.weight"))
            layer.fc1.bias.copy_(take(f"{p}.intermediate.dense.bias"))
            layer.fc2.weight.copy_(take(f"{p}.output.dense.weight"))
            layer.fc2.bias.copy_(take(f"{p}.output.dense.bias"))
        logger.info("ESM2 weights loaded from %s", path)
