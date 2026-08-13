"""What the model is actually doing, beyond one perplexity number.

Three questions, each with a measurement:

**Does the partner chain help?** Compare the MLM loss of a masked residue when
the model sees the whole complex against the loss when it sees only that
residue's own chain,

.. math:: \\Delta L = L_\\text{single} - L_\\text{full},

so :math:`\\Delta L > 0` means the partner carried information. This costs one
extra forward pass and *no* change to the data: a chain-only view is the same
packed tensor with ``complex_cu`` replaced by ``chain_cu``, which makes every
chain its own complex. Reported split by source, by homomer/heteromer and by
interface/non-interface, because the failure mode this exists to catch is a
large :math:`\\Delta L` on homodimers only — that is sequence copying, not
binding.

**Is the model order-independent?** Chain order is arbitrary, so re-ordering
the chains of a complex should change nothing. For the main architecture this
is exact up to floating point (``tests/test_complexlm.py`` pins it), so the
measurement is really a regression test on the *trained* model and the honest
comparison for the global-RoPE baseline, which cannot be invariant.

**Is there an attention sink?** Bidirectional protein MLMs are not causal LMs
and the sink literature does not transfer, so it is measured rather than
assumed: per layer and head, how concentrated incoming attention is, how much
of it lands on chain-start residues, and whether chain-scoped and global layers
differ. Only when a sink actually appears is a sink token worth adding.
"""

from __future__ import annotations

import dataclasses
import logging
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from deepfold3.complexlm.batch import IGNORE_INDEX, PackedBatch

logger = logging.getLogger(__name__)


def single_chain_view(batch: PackedBatch) -> PackedBatch:
    """The same batch with every chain as its own complex.

    Global layers then attend within a chain, so the model sees exactly the
    target chain and nothing else — the ``L_single`` side of partner
    utilisation, at the cost of one field.
    """
    return dataclasses.replace(
        batch,
        complex_cu=batch.chain_cu,
        complex_index=batch.chain_index,
        max_complex_len=batch.max_chain_len,
    )


def permute_chain_order(
    batch: PackedBatch, generator: torch.Generator | None = None
) -> tuple[PackedBatch, torch.Tensor]:
    """Shuffle chain order inside each complex.

    Returns:
        ``(permuted batch, index)`` where ``index[i]`` is the row of the
        permuted batch holding the token that was row ``i``, so outputs can be
        compared token by token.
    """
    device = batch.tokens.device
    chain_of_complex: dict[int, list[int]] = defaultdict(list)
    starts = batch.chain_cu[:-1].tolist()
    ends = batch.chain_cu[1:].tolist()
    for chain, start in enumerate(starts):
        chain_of_complex[int(batch.complex_index[start])].append(chain)

    order: list[int] = []
    for complex_id in sorted(chain_of_complex):
        chains = chain_of_complex[complex_id]
        shuffled = torch.randperm(len(chains), generator=generator).tolist()
        order.extend(chains[i] for i in shuffled)

    gather = torch.cat([torch.arange(starts[c], ends[c], device=device) for c in order])
    lengths = [ends[c] - starts[c] for c in order]
    chain_cu = torch.zeros(len(order) + 1, dtype=torch.int32, device=device)
    chain_cu[1:] = torch.tensor(np.cumsum(lengths), dtype=torch.int32, device=device)

    permuted = dataclasses.replace(
        batch,
        tokens=batch.tokens[gather],
        labels=batch.labels[gather],
        positions=batch.positions[gather],
        global_positions=batch.global_positions[gather],
        chain_cu=chain_cu,
        chain_index=torch.repeat_interleave(
            torch.arange(len(order), dtype=torch.int32, device=device),
            torch.tensor(lengths, device=device),
        ),
        complex_index=batch.complex_index[gather],
        entity_index=batch.entity_index[gather],
        iface=batch.iface[gather],
        max_chain_len=max(lengths),
    )
    # global_positions must be rebuilt: they index within the complex, and the
    # complex has been re-ordered. Chain-scoped positions are per chain and
    # therefore travel with their tokens unchanged.
    permuted.global_positions = _complex_positions(permuted)
    inverse = torch.empty_like(gather)
    inverse[gather] = torch.arange(gather.numel(), device=device)
    return permuted, inverse


def _complex_positions(batch: PackedBatch) -> torch.Tensor:
    starts = batch.complex_cu[:-1]
    offsets = torch.repeat_interleave(starts, batch.complex_cu[1:] - starts)
    return (torch.arange(batch.n_tokens, device=batch.tokens.device) - offsets).to(torch.int32)


def token_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Per-token cross-entropy, ``nan`` where the position is not scored."""
    loss = F.cross_entropy(logits.float(), labels.clamp(min=0), reduction="none")
    return torch.where(labels == IGNORE_INDEX, torch.full_like(loss, float("nan")), loss)


def complex_is_homomer(batch: PackedBatch) -> torch.Tensor:
    """``[T]`` bool: is this token's complex a homomer (one distinct entity)?"""
    n_complexes = batch.n_complexes
    counts = torch.zeros(n_complexes, dtype=torch.long, device=batch.tokens.device)
    seen: set[tuple[int, int]] = set()
    for complex_id, entity in zip(batch.complex_index.tolist(), batch.entity_index.tolist(), strict=True):
        key = (complex_id, entity)
        if key not in seen:
            seen.add(key)
            counts[complex_id] += 1
    return (counts == 1)[batch.complex_index.long()]


class GroupedMetrics:
    """Sums of per-token quantities, bucketed by named groups."""

    def __init__(self) -> None:
        self.sums: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    def add(self, group: str, values: dict[str, torch.Tensor], mask: torch.Tensor) -> None:
        n = int(mask.sum())
        if n == 0:
            return
        self.sums[group]["n"] += n
        for key, value in values.items():
            self.sums[group][key] += float(value[mask].nansum())

    def result(self, keys: tuple[str, ...]) -> dict[str, float]:
        out: dict[str, float] = {}
        for group, sums in self.sums.items():
            n = sums.get("n", 0.0)
            if not n:
                continue
            out[f"{group}/n"] = n
            for key in keys:
                out[f"{group}/{key}"] = sums[key] / n
        return out


@torch.no_grad()
def partner_utilisation(
    model: torch.nn.Module,
    batch: PackedBatch,
    source_names: list[str],
    metrics: GroupedMetrics | None = None,
) -> GroupedMetrics:
    """Accumulate losses, accuracies and the partner gap, by group."""
    metrics = metrics if metrics is not None else GroupedMetrics()
    full_logits = model(batch).logits
    single_logits = model(single_chain_view(batch)).logits
    full = token_loss(full_logits, batch.labels)
    single = token_loss(single_logits, batch.labels)
    values = {
        "loss_full": full,
        "loss_single": single,
        "delta": single - full,
        # Same scored set as the loss, so the two are directly comparable. It
        # includes the 10% of selected positions left unchanged by the 80/10/10
        # rule, which are nearly free — as in ESM2 and BERT.
        "accuracy": (full_logits.argmax(-1) == batch.labels).float(),
        "accuracy_single": (single_logits.argmax(-1) == batch.labels).float(),
    }

    scored = batch.labels != IGNORE_INDEX
    homomer = complex_is_homomer(batch)
    source_of_token = batch.source[batch.complex_index.long()]
    # Interface groups are restricted to sources that *have* interface labels.
    # A sequence-only source's flags are all zero, and counting those as
    # non-interface turns the interface comparison into a source comparison —
    # on the first run that was two thirds of the non-interface group.
    known = batch.iface_known
    groups: list[tuple[str, torch.Tensor]] = [
        ("all", scored),
        ("homomer", scored & homomer),
        ("heteromer", scored & ~homomer),
        ("interface", scored & known & batch.iface),
        ("non_interface", scored & known & ~batch.iface),
        ("unlabelled", scored & ~known),
        ("homomer_interface", scored & known & homomer & batch.iface),
        ("heteromer_interface", scored & known & ~homomer & batch.iface),
        ("heteromer_non_interface", scored & known & ~homomer & ~batch.iface),
    ]
    for i, name in enumerate(source_names):
        groups.append((f"source:{name}", scored & (source_of_token == i)))
    for name, mask in groups:
        metrics.add(name, values, mask)
    return metrics


@torch.no_grad()
def permutation_sensitivity(model: torch.nn.Module, batch: PackedBatch, seed: int = 0) -> dict[str, float]:
    """How much re-ordering the chains changes the loss and the representation."""
    generator = torch.Generator().manual_seed(seed)
    permuted, inverse = permute_chain_order(batch, generator)
    base = model(batch, return_hidden=True)
    other = model(permuted, return_hidden=True)

    base_loss = token_loss(base.logits, batch.labels)
    other_loss = token_loss(other.logits, permuted.labels)[inverse]
    scored = batch.labels != IGNORE_INDEX
    hidden_a = base.hidden[-1]
    hidden_b = other.hidden[-1][inverse]

    pooled_a = _pool(hidden_a, batch.complex_index, batch.n_complexes)
    pooled_b = _pool(hidden_b, batch.complex_index, batch.n_complexes)
    return {
        "perm/abs_delta_loss": float((other_loss[scored] - base_loss[scored]).abs().mean()),
        "perm/token_cosine": float(F.cosine_similarity(hidden_a.float(), hidden_b.float(), dim=-1).mean()),
        "perm/pooled_cosine": float(F.cosine_similarity(pooled_a.float(), pooled_b.float(), dim=-1).mean()),
    }


def _pool(hidden: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    """Mean over each complex's tokens — the pooled complex representation."""
    out = torch.zeros(n, hidden.shape[-1], device=hidden.device, dtype=hidden.dtype)
    out.index_add_(0, index.long(), hidden)
    counts = torch.zeros(n, device=hidden.device, dtype=hidden.dtype)
    counts.index_add_(0, index.long(), torch.ones_like(index, dtype=hidden.dtype))
    return out / counts[:, None].clamp(min=1)


@torch.no_grad()
def attention_stats(model: torch.nn.Module, batch: PackedBatch) -> dict[str, float]:
    """Per-layer attention concentration, on the dense path.

    Runs the whole model with ``[H, T, T]`` weights materialised, so it wants a
    single small complex.

    ``max_mass`` is the largest share of total incoming attention any one token
    receives. **It is not comparable between layer kinds on its own**: every
    query row sums to 1, so the uniform share is ``1/T`` for both, but a token
    in a chain-scoped layer can only be reached by its own chain's ``L_c``
    queries and so tops out at ``L_c/T``, while a complex-scoped token can
    reach 1. With ``k`` chains that is a factor of ``k`` of headroom before any
    behavioural difference. ``x_uniform`` (``max_mass * T``) and
    ``frac_of_ceiling`` are therefore logged alongside, and are what should be
    compared.

    ``top5_mass`` is the share held by the five most-attended tokens — the
    quantity that actually distinguishes a sink (tens of percent, on the same
    token in every layer) from ordinary mild concentration.
    ``chain_start_mass`` is how much lands on the first residue of a chain,
    which is where a *positional* sink would show up.
    """
    impl = model.cfg.attn_impl
    model.cfg.attn_impl = "sdpa"
    try:
        out = model(batch, return_attn=True)
    finally:
        model.cfg.attn_impl = impl
    if not out.attentions:
        return {}

    starts = batch.chain_cu[:-1].long()
    is_start = torch.zeros(batch.n_tokens, dtype=torch.bool, device=batch.tokens.device)
    is_start[starts] = True

    n_tokens = batch.n_tokens
    chain_len = (batch.chain_cu[1:] - batch.chain_cu[:-1]).float()
    reachable = chain_len[batch.chain_index.long()] / n_tokens  # per-token ceiling under 'c'

    stats: dict[str, float] = {"attn/n_tokens": float(n_tokens), "attn/n_chains": float(batch.n_chains)}
    for layer, weights in enumerate(out.attentions):
        kind = model.blocks[layer].kind
        incoming = weights.sum(dim=-2)  # [H, T] total attention received
        share = incoming / incoming.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        entropy = -(weights.clamp(min=1e-9) * weights.clamp(min=1e-9).log()).sum(-1).mean()
        peak = share.max()
        token = int(share.max(dim=0).values.argmax())
        ceiling = float(reachable[token]) if kind == "c" else 1.0
        top5 = share.mean(0).topk(min(5, n_tokens)).values.sum()
        stats[f"attn/l{layer}_{kind}/max_mass"] = float(peak)
        stats[f"attn/l{layer}_{kind}/x_uniform"] = float(peak) * n_tokens
        stats[f"attn/l{layer}_{kind}/frac_of_ceiling"] = float(peak) / max(ceiling, 1e-9)
        stats[f"attn/l{layer}_{kind}/top5_mass"] = float(top5)
        stats[f"attn/l{layer}_{kind}/mean_entropy"] = float(entropy)
        stats[f"attn/l{layer}_{kind}/chain_start_mass"] = float(share[:, is_start].sum(-1).mean())
    for kind in ("c", "g", "r", "x"):
        keys = [k for k in stats if f"_{kind}/max_mass" in k]
        if not keys:
            continue
        for metric in ("max_mass", "x_uniform", "frac_of_ceiling", "top5_mass", "chain_start_mass"):
            stats[f"attn/{kind}/{metric}"] = float(np.mean([stats[k.replace("max_mass", metric)] for k in keys]))
    return stats
