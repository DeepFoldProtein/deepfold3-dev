"""MLM masking policies.

The v1 objective is masked language modelling and nothing else — no contact
head, no distogram, no contrastive pair loss — so the masking policy is the
only place where a design choice about *what the model must learn* is
expressed. Three policies, all ablatable:

``uniform``
    Every residue at ``rate``. What ESM2 does, and what PPLM does for STRING
    pairs, which have no structure.
``interface``
    Interface residues at ``interface_rate``, the rest at ``rate``. PPLM's PDB
    policy (30% / 15%). Only available for a store with interface flags.
``homomer_paired``
    Composable with either: mask the same positions in every copy of an
    entity. Without it a homodimer hands the model a verbatim answer key in the
    other chain, and the partner-utilisation diagnostic measures copying rather
    than binding.

    The pairing is one draw shared by the copies, *not* the union of
    independent draws. The union looks equivalent and is not: it raises the
    marginal rate to ``1 - (1 - p)^k`` for a ``k``-mer, which is 0.28 for a
    homodimer at ``p = 0.15`` and 0.86 for a 12-mer, so the objective would
    silently change with copy number.

    The 80/10/10 draw is shared too. Sharing only the *selection* leaves the
    10%-unchanged and 10%-random branches to fire independently in each copy,
    which hands back the answer at ~19% of paired positions — a smaller
    version of exactly the shortcut the policy exists to remove.

Corruption of the selected positions is BERT's 80/10/10 — replace with
``<mask>``, replace with a random amino acid, or leave unchanged — which ESM2
and PPLM both inherit. Leaving 10% unchanged is what stops the model from
learning that an unmasked position is always correct.
"""

from __future__ import annotations

import numpy as np

from deepfold3.complexlm.config import MaskingConfig
from deepfold3.complexlm.vocab import MASK_ID, standard_token_ids

_STANDARD = standard_token_ids()


def select_positions(
    rng: np.random.Generator,
    seqs: list[np.ndarray],
    iface: list[np.ndarray] | None,
    entities: np.ndarray,
    cfg: MaskingConfig,
    interface_masking: bool,
    groups: list[list[int]] | None = None,
) -> list[np.ndarray]:
    """Boolean selection mask per chain — which positions are scored."""
    rates = []
    for slot, seq in enumerate(seqs):
        rate = np.full(seq.size, cfg.rate, dtype=np.float32)
        if interface_masking and iface is not None:
            rate[iface[slot].astype(bool)] = cfg.interface_rate
        rates.append(rate)

    if groups is None:
        groups = _draw_groups(entities, [s.size for s in seqs], cfg.homomer_paired)
    selected: list[np.ndarray] = [np.zeros(s.size, dtype=bool) for s in seqs]
    for slots in groups:
        # Copies share a crop window (see crop.py), so position p is the same
        # residue in each; averaging their rates keeps the expected number of
        # masked residues equal to the unpaired policy.
        rate = np.mean([rates[s] for s in slots], axis=0)
        draw = rng.random(rate.size) < rate
        for slot in slots:
            selected[slot] = draw.copy()
    return selected


def _draw_groups(entities: np.ndarray, lengths: list[int], homomer_paired: bool) -> list[list[int]]:
    """Slots that share one masking draw.

    Every slot on its own unless ``homomer_paired``, in which case copies of an
    entity are grouped — but only when their cropped lengths agree, since a
    shared draw over misaligned copies would mask different residues.
    """
    if not homomer_paired:
        return [[i] for i in range(len(lengths))]
    groups: list[list[int]] = []
    for entity in np.unique(entities):
        slots = np.flatnonzero(entities == entity).tolist()
        if len(slots) > 1 and len({lengths[s] for s in slots}) > 1:
            groups.extend([s] for s in slots)
        else:
            groups.append(slots)
    return groups


def corrupt(
    rng: np.random.Generator,
    seqs: list[np.ndarray],
    selected: list[np.ndarray],
    cfg: MaskingConfig,
    groups: list[list[int]] | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Apply 80/10/10 to the selected positions.

    Args:
        groups: Slots that share one corruption draw, as returned by
            :func:`select_positions` via :func:`apply_masking`. Copies of an
            entity share theirs so that a position masked in one copy is masked
            in all of them.

    Returns:
        ``(inputs, labels)``; labels hold the true token at selected positions
        and ``-1`` elsewhere (the collator maps that to
        :data:`~deepfold3.complexlm.batch.IGNORE_INDEX`).
    """
    inputs = [seq.astype(np.int64).copy() for seq in seqs]
    labels = [np.full(seq.size, -1, dtype=np.int64) for seq in seqs]
    for slots in groups if groups is not None else [[i] for i in range(len(seqs))]:
        idx = np.flatnonzero(selected[slots[0]])
        if not idx.size:
            continue
        roll = rng.random(idx.size)
        swap_to = rng.choice(_STANDARD, size=idx.size)
        masked = idx[roll < cfg.mask_token_frac]
        swapped = (roll >= cfg.mask_token_frac) & (roll < cfg.mask_token_frac + cfg.random_token_frac)
        for slot in slots:
            labels[slot][idx] = inputs[slot][idx]
            inputs[slot][masked] = MASK_ID
            inputs[slot][idx[swapped]] = swap_to[swapped]
    return inputs, labels


def apply_masking(
    rng: np.random.Generator,
    seqs: list[np.ndarray],
    iface: list[np.ndarray] | None,
    entities: np.ndarray,
    cfg: MaskingConfig,
    interface_masking: bool,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """:func:`select_positions` then :func:`corrupt`, sharing both draws."""
    groups = _draw_groups(entities, [s.size for s in seqs], cfg.homomer_paired)
    selected = select_positions(rng, seqs, iface, entities, cfg, interface_masking, groups)
    return corrupt(rng, seqs, selected, cfg, groups)
