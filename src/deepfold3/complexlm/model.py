"""The complex language model.

A protein complex is a *set* of chains. Inside a chain there is a residue
order and a meaningful sequence distance; between chains there is neither, and
any 1-D coordinate laid over a serialisation invents one. The architecture
follows that split literally: chain-scoped layers see relative position through
RoPE, complex-scoped layers see none at all.

Both kinds of layer read the same packed hidden tensor. Only the cumulative
sequence-length vector handed to the attention kernel changes::

    packed tokens ──┬── chain_cu   -> chain-scoped attention, RoPE on chain index
                    └── complex_cu -> complex-scoped attention, no positions

Nothing is re-gathered, transposed or padded between the two, so the frequency
of global attention is free to be a hyperparameter (see
:attr:`~deepfold3.complexlm.config.ComplexLMConfig.layer_pattern`).

Relation to PPLM (Liu, Chen & Zhang, *Nat. Commun.* 2026), the closest prior
work: it also gives intra-chain pairs RoPE and inter-chain pairs none, but it
adds the two score matrices *inside one softmax* of every layer, gated by a
learned per-layer, per-head weight. Here they are separate layers, so the two
kinds of interaction never compete for one attention budget, the ratio between
them is tunable, and the construction extends to an arbitrary number of chains
rather than a pair. Which of those matters is what the ablations measure.

Block internals follow ESMC (bias-free projections, QK layer norm, SwiGLU at
expansion 8/3, residual scaling by ``sqrt(n_layers / 36)``), with RMSNorm by
default and no BOS/EOS.

Optionally under maximal-update parameterisation (``cfg.mup``), so the learning
rate can be tuned on a narrow proxy and transferred to the target width instead
of being guessed or swept at full size. With ``mup=False`` — the default —
every scale factor below is exactly 1 and the model is a standard-parameterised
transformer.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepfold3.complexlm.batch import IGNORE_INDEX, PackedBatch
from deepfold3.complexlm.config import ComplexLMConfig

logger = logging.getLogger(__name__)


class RMSNorm(nn.Module):
    """Root-mean-square norm, computed in fp32 and returned in the input dtype."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


def build_norm(cfg: ComplexLMConfig, dim: int) -> nn.Module:
    if cfg.norm == "rmsnorm":
        return RMSNorm(dim, cfg.norm_eps)
    return nn.LayerNorm(dim, eps=cfg.norm_eps, bias=cfg.bias)


class RotaryEmbedding(nn.Module):
    """RoPE over explicit position ids.

    Positions are always passed in, never inferred from the row index: the
    packed tensor holds many chains back to back, so ``arange`` would report a
    separation of one across a chain boundary. Under the main model the ids
    restart at zero for every chain, which is the whole point — the rotary
    tables only ever have to cover the longest *chain*, not the longest
    complex.
    """

    def __init__(self, head_dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``positions`` ``[T]`` -> ``cos``/``sin`` of shape ``[T, 1, head_dim]``."""
        freqs = positions.float()[:, None] * self.inv_freq[None, :]
        emb = torch.cat((freqs, freqs), dim=-1)[:, None, :]
        return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """``x`` ``[T, H, D]`` rotated by the per-token angles."""
    return (x.float() * cos + _rotate_half(x.float()) * sin).to(x.dtype)


class Attention(nn.Module):
    """One attention layer, scoped by its kind.

    Args:
        cfg: Architecture.
        kind: One of :data:`~deepfold3.complexlm.config.LAYER_KINDS`.
    """

    def __init__(self, cfg: ComplexLMConfig, kind: str) -> None:
        super().__init__()
        self.cfg = cfg
        self.kind = kind
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        # muP scales attention logits by 1/d_head rather than 1/sqrt(d_head),
        # expressed against the proxy so that the two agree at the base width.
        self.scale = (cfg.base_head_dim**0.5) / self.head_dim if cfg.mup else self.head_dim**-0.5
        self.norm = build_norm(cfg, cfg.d_model)
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=cfg.bias)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        # ESMC normalises the whole concatenated query/key, not each head.
        self.q_norm = build_norm(cfg, cfg.d_model) if cfg.qk_layernorm else nn.Identity()
        self.k_norm = build_norm(cfg, cfg.d_model) if cfg.qk_layernorm else nn.Identity()
        self.rotary = RotaryEmbedding(self.head_dim, cfg.rope_theta) if kind in ("c", "r") else None

    def _positions(self, batch: PackedBatch) -> torch.Tensor | None:
        if self.kind == "c":
            return batch.positions
        if self.kind == "r":
            return batch.global_positions
        return None

    def forward(
        self, x: torch.Tensor, batch: PackedBatch, return_attn: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        T = x.shape[0]
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.chunk(3, dim=-1)
        q, k = self.q_norm(q).to(v.dtype), self.k_norm(k).to(v.dtype)
        q, k, v = (t.view(T, self.n_heads, self.head_dim) for t in (q, k, v))
        if self.rotary is not None:
            cos, sin = self.rotary(self._positions(batch))
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        if self._use_varlen(v) and not return_attn:
            out = self._varlen(q, k, v, batch)
            attn = None
        else:
            out, attn = self._dense(q, k, v, batch, return_attn)
        return self.out_proj(out.reshape(T, -1)), attn

    # -- kernels ---------------------------------------------------------

    def _use_varlen(self, x: torch.Tensor) -> bool:
        """Whether the packed FlashAttention path can run on this tensor.

        CUDA and a half dtype are kernel requirements, not preferences, so the
        dense path takes over silently on CPU or in fp32 — that is what makes
        the tests, and any small debugging run, work without a config change.
        Kind ``x`` has no varlen form at all (a token attending to every chain
        *but* its own is not a contiguous segment).
        """
        return (
            self.cfg.attn_impl == "varlen"
            and self.kind != "x"
            and x.is_cuda
            and x.dtype in (torch.bfloat16, torch.float16)
        )

    def _varlen(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, batch: PackedBatch) -> torch.Tensor:
        from torch.nn.attention.varlen import varlen_attn

        if self.kind == "x":
            raise NotImplementedError(
                "cross-chain-only attention (kind 'x') is not a varlen mask; run it with attn_impl='sdpa'"
            )
        if self.kind == "c":
            cu, max_len = batch.chain_cu, batch.max_chain_len
        else:
            cu, max_len = batch.complex_cu, batch.max_complex_len
        return varlen_attn(q.contiguous(), k.contiguous(), v.contiguous(), cu, cu, max_len, max_len, scale=self.scale)

    def _allowed(self, batch: PackedBatch) -> torch.Tensor:
        """``[T, T]`` boolean mask of permitted (query, key) pairs."""
        if self.kind == "c":
            allowed = batch.chain_index[:, None] == batch.chain_index[None, :]
        else:
            allowed = batch.complex_index[:, None] == batch.complex_index[None, :]
            if self.kind == "x":
                allowed = allowed & (batch.chain_index[:, None] != batch.chain_index[None, :])
        # A query with no permitted key is a row of -inf, i.e. NaN out of
        # softmax. Under kind 'x' that is not an edge case but the norm — every
        # token of a single-chain complex has one — so let such rows attend to
        # themselves; the alternative silently poisons the whole batch.
        eye = torch.eye(allowed.shape[0], dtype=torch.bool, device=allowed.device)
        return allowed | (~allowed.any(-1, keepdim=True) & eye)

    def _dense(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, batch: PackedBatch, return_attn: bool
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        allowed = self._allowed(batch)[None, None]
        q, k, v = (t.transpose(0, 1)[None] for t in (q, k, v))  # [1, H, T, D]
        if not return_attn:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed, scale=self.scale)
            return out[0].transpose(0, 1), None
        scores = (q.float() @ k.float().transpose(-1, -2)) * self.scale
        scores = scores.masked_fill(~allowed, float("-inf"))
        weights = scores.softmax(dim=-1)
        out = (weights.to(v.dtype) @ v)[0].transpose(0, 1)
        return out, weights[0]


def swiglu_hidden(d_model: int, expansion_ratio: float) -> int:
    """ESMC's rule: expand, then round up to a multiple of 256."""
    return int((expansion_ratio * d_model + 255) // 256 * 256)


class SwiGLUFFN(nn.Module):
    def __init__(self, cfg: ComplexLMConfig) -> None:
        super().__init__()
        hidden = swiglu_hidden(cfg.d_model, cfg.expansion_ratio)
        self.norm = build_norm(cfg, cfg.d_model)
        self.w_in = nn.Linear(cfg.d_model, 2 * hidden, bias=cfg.bias)
        self.w_out = nn.Linear(hidden, cfg.d_model, bias=cfg.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.w_in(self.norm(x)).chunk(2, dim=-1)
        return self.w_out(F.silu(gate) * up)


class Block(nn.Module):
    def __init__(self, cfg: ComplexLMConfig, kind: str) -> None:
        super().__init__()
        self.kind = kind
        self.attn = Attention(cfg, kind)
        self.ffn = SwiGLUFFN(cfg)
        self.residue_scale = math.sqrt(cfg.n_layers / 36) if cfg.scale_residue else 1.0

    def forward(
        self, x: torch.Tensor, batch: PackedBatch, return_attn: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        delta, attn = self.attn(x, batch, return_attn)
        x = x + delta / self.residue_scale
        return x + self.ffn(x) / self.residue_scale, attn


class MLMHead(nn.Module):
    """ESM/ESMC-style prediction head: dense -> GELU -> norm -> vocabulary.

    Named submodules rather than a ``Sequential`` so that ``out`` is
    identifiable as the readout — muP treats it differently from every other
    matrix, and keying that off a positional index would break the moment the
    head changed shape.
    """

    def __init__(self, cfg: ComplexLMConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.norm = build_norm(cfg, cfg.d_model)
        self.out = nn.Linear(cfg.d_model, cfg.vocab_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.norm(F.gelu(self.dense(x))))


@dataclass
class ComplexLMOutput:
    logits: torch.Tensor
    hidden: list[torch.Tensor] | None = None
    attentions: list[torch.Tensor] | None = None


class ComplexLM(nn.Module):
    """Masked language model over protein complexes.

    Args:
        cfg: Architecture. ``cfg.layer_pattern`` decides which layers are
            chain-scoped and which are complex-scoped.
    """

    def __init__(self, cfg: ComplexLMConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.embed_dropout = nn.Dropout(cfg.embed_dropout) if cfg.embed_dropout else nn.Identity()
        # Zero-initialised: the model starts exactly at chain_signal="none" and
        # only pays for the signal if the gradient asks for it.
        self.chain_gate = nn.Parameter(torch.zeros(1)) if cfg.chain_signal == "random" else None
        self.blocks = nn.ModuleList(Block(cfg, kind) for kind in cfg.kinds)
        self.norm = build_norm(cfg, cfg.d_model)
        self.head = MLMHead(cfg)
        self.readout_mult = 1.0 / cfg.width_mult
        self.apply(self._init_weights)
        self._scale_init()
        logger.info(
            "ComplexLM: %.1fM params, %d layers (%s), d_model=%d, heads=%d",
            self.n_params / 1e6,
            cfg.n_layers,
            "".join(cfg.kinds),
            cfg.d_model,
            cfg.n_heads,
        )

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=self.cfg.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=self.cfg.init_std)

    def _scale_init(self) -> None:
        """muP: divide *hidden* init variance by the width multiple.

        Only hidden matrices. Both of their dimensions grow with width, which
        in the reference implementation is ``infshape.ninf() == 2`` — and
        ``mup.init`` scales the init by ``width_mult ** -0.5`` for exactly
        those and leaves everything with one or zero width dimensions alone.
        The embedding and the readout are ``[vocab, d_model]``: the vocabulary
        does not grow, so they keep a width-independent init, and the readout's
        width dependence is carried entirely by its ``1/width_mult`` output
        multiplier instead.
        """
        if not self.cfg.mup:
            return
        factor = self.cfg.width_mult**0.5
        with torch.no_grad():
            for name, param in self.named_parameters():
                if _mup_role(name, param) == "hidden":
                    param.div_(factor)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        batch: PackedBatch,
        return_hidden: bool = False,
        return_attn: bool = False,
    ) -> ComplexLMOutput:
        """Returns logits ``[T, vocab]`` over the packed token axis.

        ``return_attn`` forces the dense attention path (``[H, T, T]`` per
        layer), so it is for diagnostics on a single complex, not for training.
        """
        x = self.embed_dropout(self.embed(batch.tokens))
        if self.chain_gate is not None:
            # Resampled every forward on purpose: a *fixed* per-chain vector
            # would be an order, and the model would learn it. What is wanted
            # is only that copies of one entity are distinguishable.
            noise = torch.randn(batch.n_chains, x.shape[-1], device=x.device, dtype=x.dtype)
            x = x + self.chain_gate * noise[batch.chain_index.long()]
        hidden = [x] if return_hidden else None
        attentions: list[torch.Tensor] = []
        for block in self.blocks:
            x, attn = block(x, batch, return_attn)
            if hidden is not None:
                hidden.append(x)
            if attn is not None:
                attentions.append(attn)
        x = self.norm(x)
        if hidden is not None:
            hidden[-1] = x
        return ComplexLMOutput(self.head(x) * self.readout_mult, hidden, attentions or None)

    def param_groups(self, weight_decay: float) -> list[dict]:
        """Parameter groups with a decay rule and a muP learning-rate scale.

        Matmul weights decay; norm gains, biases and the embedding table do
        not — decaying those pulls a scale toward zero rather than
        regularising a direction.

        Under muP the *hidden* matrices additionally take Adam learning rate
        ``eta / width_mult``, and their weight decay is multiplied by
        ``width_mult`` to compensate: AdamW's update is ``lr * wd * p``, so
        without that the decay per step would shrink with width and the
        regularisation would silently become width-dependent. (This is
        ``MuAdam``'s ``decoupled_wd=False`` branch, which is the right one for
        ``torch.optim.AdamW`` because its decay term carries ``lr``.)

        The embedding and the readout keep ``eta``. In the reference this falls
        out of the same test as the init rule — ``MuAdam`` scales only
        ``infshape.ninf() == 2`` parameters, and ``[vocab, d_model]`` has one
        width dimension, not two. The readout's width dependence lives in its
        ``1/width_mult`` output multiplier alone.

        The trainer reads ``lr_scale`` off each group; with ``mup=False`` every
        scale is 1 and this reduces to the two-group split.
        """
        buckets: dict[tuple[str, bool], list] = {}
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            role = _mup_role(name, param)
            decays = param.dim() >= 2 and role != "embedding"
            buckets.setdefault((role, decays), []).append(param)

        width_mult = self.cfg.width_mult
        groups = []
        for (role, decays), params in buckets.items():
            scaled = role == "hidden"
            groups.append(
                {
                    "params": params,
                    "weight_decay": (weight_decay * width_mult if scaled else weight_decay) if decays else 0.0,
                    "lr_scale": 1.0 / width_mult if scaled else 1.0,
                    "role": role,
                }
            )
        return groups


def _mup_role(name: str, param: torch.Tensor) -> str:
    """muP taxonomy of a parameter: embedding, hidden, readout, or vector.

    The distinction that matters is **how many of the tensor's dimensions grow
    with width** — ``infshape.ninf()`` in the reference implementation. Only
    ``hidden`` has two (both its dimensions are widths), and only ``hidden``
    is therefore rescaled. ``embedding`` and ``readout`` are ``[vocab,
    d_model]`` with one; ``vector`` (norm gains, biases, gates) has none. The
    two are kept apart from each other, and from ``vector``, because the
    readout additionally carries the ``1/width_mult`` output multiplier and
    because naming the groups makes the logged optimiser state readable.

    Dimensionality decides first: anything 1-D is vector-like. Deciding by name
    instead misfiles whichever norm happens not to have "norm" in its path.
    """
    if param.dim() < 2:
        return "vector"
    if name.startswith("embed."):
        return "embedding"
    if name.startswith("head.out."):
        return "readout"
    return "hidden"


def mlm_loss(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Cross-entropy over scored positions; returns ``(loss, n_scored)``.

    The mean is over this rank's own masked positions. Ranks differ slightly in
    how many they have — packing is by token budget, not by mask count — so DDP
    averaging is a weighted mean with weights within a few percent of equal.
    Correcting it exactly would need a collective per micro-batch, which costs
    more than the bias it removes.
    """
    scored = labels != IGNORE_INDEX
    n = scored.sum()
    if n == 0:
        return logits.sum() * 0.0, n
    return F.cross_entropy(logits.float(), labels, ignore_index=IGNORE_INDEX), n
