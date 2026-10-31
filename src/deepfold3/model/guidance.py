"""Classifier-free guidance terms as given on the command line.

One ``--guidance`` value is ``W[@TARGET[@LO..HI]]``:

* ``W``: the term's weight (:class:`~deepfold3.model.diffusion.GuidanceTerm`).
* ``TARGET``: what its unconditional branch drops. ``full`` (default) is the
  ``cond_drop`` drop, ``all`` every interface, and ``A:B,AA:B`` those chain
  pairs only. Pairs are unordered, and the chain names are the output's.
  Either side may name residues instead of the whole chain:
  ``A/25-40:B`` drops only the pair blocks between residues 25..40 of chain A
  and chain B, and ``+`` joins several ranges (``A/25-40+88:B``). Residue
  numbers are the input's own numbering (``residue_index``, which for a
  sequence given in the job JSON is 1-based).
* ``LO..HI``: apply only where LO ≤ σ̂ ≤ HI, in the sampler's absolute σ̂ (the
  default schedule spans 0.0077..4608), not the dimensionless σ/σ_data.

Several values add up: ``D_c + Σ_k (w_k − 1)·(D_c − D_u,k)``.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from deepfold3.model.config import DiffusionConfig
    from deepfold3.model.diffusion import GuidanceTerm

FULL = "full"
ALL_INTERFACES = "all"


@dataclass(frozen=True)
class ChainSel:
    """One side of a guidance pair: a chain, or some residues of it.

    Attributes:
        chain: The chain name, as the output writes it.
        ranges: Inclusive ``(lo, hi)`` residue numbers; empty is the whole
            chain. The numbers are the input's own ``residue_index``.
    """

    chain: str
    ranges: tuple[tuple[int, int], ...] = ()

    def __str__(self) -> str:
        if not self.ranges:
            return self.chain
        return self.chain + "/" + "+".join(f"{lo}" if lo == hi else f"{lo}-{hi}" for lo, hi in self.ranges)


@dataclass(frozen=True)
class GuidanceSpec:
    """One parsed ``--guidance`` value, before chain names are resolved.

    Attributes:
        scale: The term's weight.
        pairs: Selector pairs to drop; empty unless ``target`` names them.
        target: ``"full"``, ``"all"`` or ``"pairs"``.
        sigma_range: ``(lo, hi)`` or ``None`` for every step.
    """

    scale: float
    target: str = FULL
    pairs: tuple[tuple[ChainSel, ChainSel], ...] = ()
    sigma_range: tuple[float, float] | None = None


def has_residue_ranges(specs: Sequence[GuidanceSpec]) -> bool:
    """True if any active term drops residues rather than whole chain pairs.

    Training only ever dropped whole inter-chain pair blocks
    (:func:`~deepfold3.model.diffusion.draw_interface_pair_keep`), so a residue
    subset is a finer drop than the model was trained on. The callers say so.
    """
    return any(sel.ranges for s in specs if s.scale != 1.0 for pair in s.pairs for sel in pair)


def _parse_ranges(text: str) -> tuple[tuple[int, int], ...]:
    ranges = []
    for item in text.split("+"):
        bounds = item.split("-")
        if len(bounds) > 2 or not all(b.isdigit() for b in bounds):
            raise ValueError(f"expected a residue number or LO-HI, got {item!r}")
        lo = int(bounds[0])
        hi = int(bounds[1]) if len(bounds) == 2 else lo
        if lo > hi:
            raise ValueError(f"residue range needs LO <= HI, got {item!r}")
        ranges.append((lo, hi))
    return tuple(ranges)


def _parse_sel(text: str) -> ChainSel:
    chain, slash, residues = text.partition("/")
    if not chain:
        raise ValueError(f"expected a chain name, got {text!r}")
    if slash and not residues:
        raise ValueError(f"expected residue numbers after '/', got {text!r}")
    return ChainSel(chain, _parse_ranges(residues) if residues else ())


def _parse_pairs(text: str) -> tuple[tuple[ChainSel, ChainSel], ...]:
    pairs = []
    for item in text.split(","):
        sides = item.split(":")
        if len(sides) != 2 or not all(sides):
            raise ValueError(f"expected a chain pair like A:B or A/25-40:B, got {item!r}")
        a, b = (_parse_sel(s) for s in sides)
        if a.chain == b.chain:
            raise ValueError(f"an interface needs two chains, got {item!r}")
        pairs.append((a, b))
    return tuple(pairs)


def _parse_sigma_range(text: str) -> tuple[float, float]:
    bounds = text.split("..")
    if len(bounds) != 2:
        raise ValueError(f"expected LO..HI, got {text!r}")
    lo, hi = float(bounds[0]), float(bounds[1])
    if not lo <= hi:
        raise ValueError(f"sigma range needs LO <= HI, got {text!r}")
    return lo, hi


def parse_guidance_spec(text: str) -> GuidanceSpec:
    """Parse ``W[@TARGET[@LO..HI]]``.

    Args:
        text: The option value.

    Returns:
        The parsed term.

    Raises:
        ValueError: If ``text`` does not follow the syntax.
    """
    fields = text.split("@")
    if len(fields) > 3:
        raise ValueError(f"expected W[@TARGET[@LO..HI]], got {text!r}")
    scale = float(fields[0])
    if not math.isfinite(scale):
        raise ValueError(f"guidance weight must be finite, got {fields[0]!r}")
    target = fields[1] if len(fields) > 1 else FULL
    sigma_range = _parse_sigma_range(fields[2]) if len(fields) > 2 else None
    if target in (FULL, ALL_INTERFACES):
        return GuidanceSpec(scale, target, sigma_range=sigma_range)
    return GuidanceSpec(scale, "pairs", _parse_pairs(target), sigma_range)


def check_guidance_training(specs: Sequence[GuidanceSpec], diffusion: DiffusionConfig) -> None:
    """Refuse terms whose unconditional branch the model was never trained on.

    Args:
        specs: The requested terms.
        diffusion: The diffusion config the checkpoint was trained with.

    Raises:
        ValueError: If an active term needs a drop the training did not do.
    """
    active = [s for s in specs if s.scale != 1.0]
    if active and diffusion.use_conditioning_prob >= 1.0:
        raise ValueError("this checkpoint was trained without conditioning drop (use_conditioning_prob = 1)")
    if any(s.target != FULL for s in active) and diffusion.cond_drop_interface_frac <= 0.0:
        raise ValueError("this checkpoint was trained without interface drop (cond_drop_interface_frac = 0)")


def _sel_mask(
    sel: ChainSel,
    asym_id: torch.Tensor,  # [N]
    chain_index: Mapping[str, int],
    residue_index: torch.Tensor | None,  # [N]
    n_tokens: int | None,
) -> torch.Tensor:
    """``[N]`` bool: the tokens one selector names, never bucket padding."""
    import torch

    if sel.chain not in chain_index:
        raise ValueError(f"unknown chain {sel.chain!r}, this input has {sorted(chain_index)}")
    mask = asym_id == chain_index[sel.chain]
    if sel.ranges:
        if residue_index is None:
            raise ValueError(f"residue ranges need the input's residue_index, got none for {sel}")
        in_range = torch.zeros_like(mask)
        for lo, hi in sel.ranges:
            in_range |= (residue_index >= lo) & (residue_index <= hi)
        mask = mask & in_range
    if n_tokens is not None:
        mask = mask.clone()
        mask[n_tokens:] = False
    if not bool(mask.any()):
        raise ValueError(f"{sel} selects no token of this input")
    return mask


def _pairs_keep(
    masks: Sequence[tuple[torch.Tensor, torch.Tensor]],
    labels: Sequence[str] = (),
    allow_overlap: bool = False,
) -> torch.Tensor:
    """``[N, N]`` bool: False on both blocks of each pair of token masks.

    The pairs of one term are a union: one mask, one unconditional branch, one
    more denoiser pass per step. Two pairs that share blocks would be dropped
    once either way, so the union says nothing about which one was meant —
    refuse it rather than let a typo pass as a wider selection.
    ``allow_overlap`` takes the union anyway, for a selection written as
    overlapping pieces on purpose.
    """
    import torch

    n = masks[0][0].numel()
    keep = torch.ones(n, n, dtype=torch.bool, device=masks[0][0].device)
    for k, (in_a, in_b) in enumerate(masks):
        drop = in_a.unsqueeze(-1) & in_b.unsqueeze(-2)
        drop = drop | drop.T
        if labels and not allow_overlap and bool((drop & ~keep).any()):
            earlier = ", ".join(labels[:k])
            raise ValueError(
                f"{labels[k]} drops pair blocks already dropped by {earlier} in the same term "
                "(--allow-guidance-overlap takes the union anyway)"
            )
        keep &= ~drop
    return keep


def _warn_on_overlapping_terms(terms: Sequence[GuidanceTerm]) -> None:
    """Warn where two active terms drop the same pair block over the same σ̂.

    Each term contributes its own ``(w_k − 1)·(D_c − D_u,k)``, so a block in
    two terms is pushed by the sum of their weights, which is neither of the
    two numbers that were asked for. ``full`` terms are exempt: dropping
    everything is what they are for.
    """

    def overlaps(a: GuidanceTerm, b: GuidanceTerm) -> bool:
        lo_a, hi_a = a.sigma_range or (float("-inf"), float("inf"))
        lo_b, hi_b = b.sigma_range or (float("-inf"), float("inf"))
        if hi_a < lo_b or hi_b < lo_a:
            return False
        return bool((~a.pair_keep & ~b.pair_keep).any())

    masked = [t for t in terms if t.scale != 1.0 and t.pair_keep is not None]
    for i, a in enumerate(masked):
        for b in masked[i + 1 :]:
            if overlaps(a, b):
                warnings.warn(
                    f"guidance terms w={a.scale:g} and w={b.scale:g} drop some of the same pair blocks over "
                    f"the same σ̂; those blocks get (w₁−1)+(w₂−1), not either weight",
                    stacklevel=3,
                )
                return


def build_guidance_terms(
    specs: Sequence[GuidanceSpec],
    asym_id: torch.Tensor,  # [N]
    chain_index: Mapping[str, int],
    n_tokens: int | None = None,
    residue_index: torch.Tensor | None = None,  # [N]
    allow_overlap: bool = False,
) -> tuple[GuidanceTerm, ...]:
    """Resolve the selectors into the sampler's guidance terms.

    Args:
        specs: The parsed terms.
        asym_id: Token chain ids, possibly bucket-padded.
        chain_index: Chain name to ``asym_id``, for the chains of this input.
        n_tokens: Tokens before bucket padding; the rest are never dropped.
        residue_index: Token residue numbers, needed only by a selector that
            names residues (``A/25-40:B``).
        allow_overlap: Keep selections that drop the same pair block twice,
            inside one term (a union) or across terms (summed weights), which
            are otherwise an error and a warning.

    Returns:
        One term per spec, in order.

    Raises:
        ValueError: If a term needs an interface the input does not have, or a
            selector names a chain or residues the input does not have.
    """
    from deepfold3.model.diffusion import GuidanceTerm, interface_pair_keep

    terms = []
    for spec in specs:
        pair_keep = None
        if spec.target != FULL:
            if len(chain_index) < 2:
                raise ValueError(f"interface guidance needs two chains, this input has {sorted(chain_index)}")
            if spec.target == ALL_INTERFACES:
                pair_keep = interface_pair_keep(asym_id)
            else:
                pair_keep = _pairs_keep(
                    [
                        tuple(_sel_mask(sel, asym_id, chain_index, residue_index, n_tokens) for sel in pair)
                        for pair in spec.pairs
                    ],
                    labels=[f"{a}:{b}" for a, b in spec.pairs],
                    allow_overlap=allow_overlap,
                )
            if n_tokens is not None:
                pair_keep[n_tokens:, :] = True
                pair_keep[:, n_tokens:] = True
        terms.append(GuidanceTerm(spec.scale, pair_keep, spec.sigma_range))
    if not allow_overlap:
        _warn_on_overlapping_terms(terms)
    return tuple(terms)
