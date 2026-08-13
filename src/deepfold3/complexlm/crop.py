"""Cropping a complex down to the length budget.

Two budgets, and they are not the same thing:

* ``max_chain_len`` bounds one chain, and is the only length RoPE has to cover.
  Restricting rotary positions to chains is what decouples the two — a global
  1-D encoding would need tables as long as the whole complex.
* ``max_complex_len`` bounds the packed complex, i.e. what the global layers
  attend over.

Chains are cropped to a window biased toward interface residues, because a
crop that drops the interface leaves an example that cannot teach anything
about binding — the same reason PPLM crops its PDB pairs with an interface
bias.

Copies of one entity are cropped to the *same* window. They have to be: the
homomer-paired masking policy masks position ``p`` in every copy at once, and
that is only meaningful if position ``p`` means the same residue in each. The
cost is that the window is chosen from the first copy's interface, which is not
always the best window for the others.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CropSpec:
    """Which slots survive, and the residue window kept from each."""

    slots: list[int]
    starts: list[int]
    ends: list[int]

    def apply(self, arrays: list[np.ndarray]) -> list[np.ndarray]:
        return [arrays[s][a:b] for s, a, b in zip(self.slots, self.starts, self.ends, strict=True)]


def _window(rng: np.random.Generator, length: int, budget: int, iface: np.ndarray | None) -> tuple[int, int]:
    """A window of ``min(length, budget)`` residues, anchored on an interface."""
    if length <= budget:
        return 0, length
    anchors = np.flatnonzero(iface) if iface is not None else np.empty(0, dtype=np.int64)
    if anchors.size:
        centre = int(rng.choice(anchors))
        start = int(np.clip(centre - budget // 2, 0, length - budget))
    else:
        start = int(rng.integers(0, length - budget + 1))
    return start, start + budget


def crop_complex(
    rng: np.random.Generator,
    lengths: list[int],
    entities: np.ndarray,
    iface: list[np.ndarray] | None,
    *,
    max_chain_len: int,
    max_complex_len: int,
    max_chains: int,
    min_chain_len: int = 8,
) -> CropSpec:
    """Choose the slots and windows that fit the budgets.

    Chain selection is anchored: one slot is drawn at random and kept, then
    others are added in the (already permuted) order they arrive. Dropping from
    the end of a *permuted* order is a uniform choice of which chains survive,
    so no chain position is systematically favoured.

    Args:
        lengths: Residue count per slot.
        entities: Entity id per slot; equal ids share a crop window.
        iface: Per-residue interface flags per slot, or ``None``.
        min_chain_len: A slot whose share of the budget falls below this is
            dropped rather than reduced to a stub.
    """
    n = len(lengths)
    keep = list(range(n))
    if n > max_chains:
        anchor = int(rng.integers(n))
        others = [i for i in keep if i != anchor]
        keep = sorted([anchor, *others[: max_chains - 1]])

    total = sum(lengths[i] for i in keep)
    # Proportional budgets: every chain is squeezed by the same factor rather
    # than the tail being truncated away, so a large complex keeps a piece of
    # every chain and the global layers still see the full chain count.
    while True:
        if total <= max_complex_len:
            budgets = {i: min(lengths[i], max_chain_len) for i in keep}
            if sum(budgets.values()) <= max_complex_len:
                break
        scale = max_complex_len / max(total, 1)
        budgets = {i: min(int(lengths[i] * scale), max_chain_len) for i in keep}
        short = [i for i in keep if budgets[i] < min_chain_len]
        if not short or len(keep) <= 1:
            break
        keep = [i for i in keep if i not in short[: max(1, len(short))]]
        total = sum(lengths[i] for i in keep)

    windows: dict[int, tuple[int, int]] = {}
    starts, ends = [], []
    for i in keep:
        entity = int(entities[i])
        if entity not in windows:
            flags = iface[i] if iface is not None else None
            windows[entity] = _window(rng, lengths[i], budgets[i], flags)
        start, end = windows[entity]
        # A later copy can be shorter than the first only if the store is
        # inconsistent; clamp rather than trust it.
        end = min(end, lengths[i])
        starts.append(min(start, max(end - 1, 0)))
        ends.append(end)
    return CropSpec(keep, starts, ends)
