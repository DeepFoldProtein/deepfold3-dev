"""Classifier-free guidance terms as given on the command line.

One ``--guidance`` value is ``W[@TARGET[@LO..HI]]``:

* ``W``: the term's weight (:class:`~deepfold3.model.diffusion.GuidanceTerm`).
* ``TARGET``: what its unconditional branch drops. ``full`` (default) is the
  ``cond_drop`` drop, ``all`` every interface, and ``A:B,AA:B`` those chain
  pairs only. Pairs are unordered, and the chain names are the output's.
* ``LO..HI``: apply only where LO ≤ σ̂ ≤ HI, in the sampler's absolute σ̂ (the
  default schedule spans 0.0077..4608), not the dimensionless σ/σ_data.

Several values add up: ``D_c + Σ_k (w_k − 1)·(D_c − D_u,k)``.
"""

from __future__ import annotations

import math
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
class GuidanceSpec:
    """One parsed ``--guidance`` value, before chain names are resolved.

    Attributes:
        scale: The term's weight.
        pairs: Chain-name pairs to drop; empty unless ``target`` names them.
        target: ``"full"``, ``"all"`` or ``"pairs"``.
        sigma_range: ``(lo, hi)`` or ``None`` for every step.
    """

    scale: float
    target: str = FULL
    pairs: tuple[tuple[str, str], ...] = ()
    sigma_range: tuple[float, float] | None = None


def _parse_pairs(text: str) -> tuple[tuple[str, str], ...]:
    pairs = []
    for item in text.split(","):
        names = item.split(":")
        if len(names) != 2 or not all(names):
            raise ValueError(f"expected a chain pair like A:B, got {item!r}")
        if names[0] == names[1]:
            raise ValueError(f"an interface needs two chains, got {item!r}")
        pairs.append((names[0], names[1]))
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


def _pairs_keep(asym_id: torch.Tensor, pairs: Sequence[tuple[int, int]]) -> torch.Tensor:
    """``[N, N]`` bool: False on both blocks of each ``(asym_a, asym_b)`` pair."""
    import torch

    keep = torch.ones(asym_id.numel(), asym_id.numel(), dtype=torch.bool, device=asym_id.device)
    for a, b in pairs:
        in_a, in_b = asym_id == a, asym_id == b
        drop = in_a.unsqueeze(-1) & in_b.unsqueeze(-2)
        keep &= ~(drop | drop.T)
    return keep


def build_guidance_terms(
    specs: Sequence[GuidanceSpec],
    asym_id: torch.Tensor,  # [N]
    chain_index: Mapping[str, int],
    n_tokens: int | None = None,
) -> tuple[GuidanceTerm, ...]:
    """Resolve chain names into the sampler's guidance terms.

    Args:
        specs: The parsed terms.
        asym_id: Token chain ids, possibly bucket-padded.
        chain_index: Chain name to ``asym_id``, for the chains of this input.
        n_tokens: Tokens before bucket padding; the rest are never dropped.

    Returns:
        One term per spec, in order.

    Raises:
        ValueError: If a term needs an interface the input does not have.
    """
    from deepfold3.model.diffusion import GuidanceTerm, interface_pair_keep

    terms = []
    for spec in specs:
        pair_keep = None
        if spec.target != FULL:
            if len(chain_index) < 2:
                raise ValueError(f"interface guidance needs two chains, this input has {sorted(chain_index)}")
            for name in (n for pair in spec.pairs for n in pair):
                if name not in chain_index:
                    raise ValueError(f"unknown chain {name!r}, this input has {sorted(chain_index)}")
            if spec.target == ALL_INTERFACES:
                pair_keep = interface_pair_keep(asym_id)
            else:
                pair_keep = _pairs_keep(asym_id, [(chain_index[a], chain_index[b]) for a, b in spec.pairs])
            if n_tokens is not None:
                pair_keep[n_tokens:, :] = True
                pair_keep[:, n_tokens:] = True
        terms.append(GuidanceTerm(spec.scale, pair_keep, spec.sigma_range))
    return tuple(terms)
