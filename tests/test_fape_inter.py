"""Inter-chain FAPE: the cross-chain half of the loss, split out by ``fape_mode``.

The property that motivates the term: with a 10 Å clamp, a chain docked far
from its site saturates every cross-chain pair, so the intra-only loss has
**zero gradient** with respect to the pose. These tests pin (a) that a rigid
displacement of one chain moves only the inter term, (b) that it keeps a
gradient where a 10 Å clamp would not, and (c) that the point's chain is read
off the point's own token, not off its column.
"""

from __future__ import annotations

import torch

from deepfold3.training.loss import fape_loss

M = 4  # atom slots per token in these fixtures


# Frame atoms must not be colinear or `_frame_angle_mask` drops every frame
# (25° threshold) and both terms come out 0.
_ATOM_OFFSETS = torch.tensor(
    [
        [0.0, 0.0, 0.0],  # slot 0 — frame atom a
        [1.0, 1.2, 0.0],  # slot 1 — frame atom b (also a projected point)
        [2.4, 0.5, 0.3],  # slot 2 — frame atom c
        [1.2, 2.2, 0.8],  # slot 3 — projected point
    ]
)


def _chain(n_tokens: int, offset: float, spacing: float = 3.8) -> torch.Tensor:
    """A chain of `n_tokens` tokens, 4 atom slots each, frames non-colinear."""
    pos = torch.zeros(n_tokens, M, 3)
    for i in range(n_tokens):
        base = torch.tensor([i * spacing + offset, 0.0, 0.0])
        pos[i] = base + _ATOM_OFFSETS
    return pos


def _fixture(n_a: int = 5, n_b: int = 5, gap: float = 30.0):
    """Two chains; returns (gt_pos, asym_id, index tensors, masks)."""
    gt = torch.cat([_chain(n_a, 0.0), _chain(n_b, gap)], dim=0)  # [N, M, 3]
    n = n_a + n_b
    asym = torch.cat([torch.zeros(n_a, dtype=torch.long), torch.ones(n_b, dtype=torch.long)])
    tok = torch.arange(n).unsqueeze(-1).expand(n, 3)  # frame atoms live on their own token
    frame_atom = torch.tensor([0, 1, 2]).unsqueeze(0).expand(n, 3)
    fape_tok = torch.arange(n).unsqueeze(-1).expand(n, 2).contiguous()
    fape_atom = torch.tensor([1, 3]).unsqueeze(0).expand(n, 2).contiguous()
    fape_mask = torch.ones(n, 2, dtype=torch.bool)
    seq_mask = torch.ones(n, dtype=torch.bool)
    pair_mask = torch.ones(n, n, dtype=torch.bool)
    return gt, asym, tok, frame_atom, fape_tok, fape_atom, fape_mask, seq_mask, pair_mask


def _call(pred, gt, asym, tok, frame_atom, fape_tok, fape_atom, fape_mask, seq_mask, pair_mask, **kw):
    kw.setdefault("mode", "both")
    return fape_loss(
        pred.unsqueeze(0),
        gt,
        fape_tok,
        fape_atom,
        fape_mask,
        frame_atom,
        tok,
        seq_mask,
        seq_mask,
        pair_mask,
        asym_id=asym,
        **kw,
    )


def test_displacing_one_chain_moves_only_the_inter_term():
    gt, asym, tok, fa, ftok, fat, fm, sm, pm = _fixture()
    intra0, inter0 = _call(gt.clone(), gt, asym, tok, fa, ftok, fat, fm, sm, pm)

    moved = gt.clone()
    moved[asym == 1] += torch.tensor([0.0, 12.0, 0.0])  # rigid shift of chain B
    intra1, inter1 = _call(moved, gt, asym, tok, fa, ftok, fat, fm, sm, pm)

    # Each chain's internal geometry is untouched, so the intra term cannot move.
    assert torch.allclose(intra0, intra1, atol=1e-5)
    assert inter1 > inter0 + 0.1


def test_inter_term_still_has_gradient_past_the_intra_clamp():
    """A 20 Å pose error is saturated at a 10 Å clamp (grad 0) but not at 30 Å."""
    gt, asym, tok, fa, ftok, fat, fm, sm, pm = _fixture()
    moved = gt.clone()
    moved[asym == 1] += torch.tensor([0.0, 20.0, 0.0])

    p_narrow = moved.clone().requires_grad_(True)
    _, inter_narrow = _call(p_narrow, gt, asym, tok, fa, ftok, fat, fm, sm, pm, clip_distance_inter=10.0)
    inter_narrow.backward()
    g_narrow = p_narrow.grad[asym == 1].abs().sum()

    p_wide = moved.clone().requires_grad_(True)
    _, inter_wide = _call(p_wide, gt, asym, tok, fa, ftok, fat, fm, sm, pm, clip_distance_inter=30.0)
    inter_wide.backward()
    g_wide = p_wide.grad[asym == 1].abs().sum()

    assert g_narrow == 0.0  # fully clamped: the pose gets no signal at all
    assert g_wide > 0.0


def test_single_chain_crop_has_no_inter_term():
    gt, asym, tok, fa, ftok, fat, fm, sm, pm = _fixture()
    one = torch.zeros_like(asym)  # everything on one chain
    intra, inter = _call(gt.clone() + 0.3, gt, one, tok, fa, ftok, fat, fm, sm, pm)
    assert float(inter) == 0.0
    assert float(intra) > 0.0


def test_point_chain_comes_from_the_points_own_token():
    """A K-slot may borrow a neighbouring token's atom; classifying by the
    column would then put a cross-chain pair in the intra term."""
    gt, asym, tok, fa, ftok, fat, fm, sm, pm = _fixture(n_a=5, n_b=5)
    # Make the last token of chain A carry a point that belongs to chain B.
    ftok = ftok.clone()
    ftok[4, 1] = 5  # row 4 (chain A), slot 1 -> token 5 (chain B)

    moved = gt.clone()
    moved[asym == 1] += torch.tensor([0.0, 12.0, 0.0])
    intra_borrow, inter_borrow = _call(moved, gt, asym, tok, fa, ftok, fat, fm, sm, pm)

    # Same displacement, but with every slot on its own token.
    _, _, _, _, ftok_plain, _, _, _, _ = _fixture(n_a=5, n_b=5)
    intra_plain, _ = _call(moved, gt, asym, tok, fa, ftok_plain, fat, fm, sm, pm)

    # The borrowed point is cross-chain for chain-A frames, so it must not land
    # in the intra term — which it would if the column index were used.
    assert torch.allclose(intra_borrow, intra_plain, atol=0.05)
    assert float(inter_borrow) > 0.0


# --- fape_mode ---------------------------------------------------------------


def test_pooled_is_the_default_and_ignores_chains():
    """The default mode must reproduce plain AF2 FAPE: one term, every pair."""
    gt, asym, tok, fa, ftok, fat, fm, sm, pm = _fixture()
    pred = gt.clone()
    pred[asym == 1] += torch.tensor([0.0, 6.0, 0.0])

    pooled, pooled_inter = _call(pred, gt, asym, tok, fa, ftok, fat, fm, sm, pm, mode="pooled")
    assert float(pooled_inter) == 0.0  # everything lands in the first term

    # Same call with the chain ids replaced by a single chain: pooled cannot
    # depend on them.
    one = torch.zeros_like(asym)
    pooled_one, _ = _call(pred, gt, one, tok, fa, ftok, fat, fm, sm, pm, mode="pooled")
    assert torch.allclose(pooled, pooled_one, atol=1e-6)

    # And it is not the same as the intra term, which drops the cross-chain pairs.
    intra, _ = _call(pred, gt, asym, tok, fa, ftok, fat, fm, sm, pm, mode="intra")
    assert not torch.allclose(pooled, intra, atol=1e-3)


def test_intra_mode_is_blind_to_the_pose():
    """mode="intra" drops cross-chain pairs, so moving a chain changes nothing."""
    gt, asym, tok, fa, ftok, fat, fm, sm, pm = _fixture()
    moved = gt.clone()
    moved[asym == 1] += torch.tensor([0.0, 12.0, 0.0])

    a_intra, a_inter = _call(gt.clone(), gt, asym, tok, fa, ftok, fat, fm, sm, pm, mode="intra")
    b_intra, b_inter = _call(moved, gt, asym, tok, fa, ftok, fat, fm, sm, pm, mode="intra")
    assert torch.allclose(a_intra, b_intra, atol=1e-5)
    assert float(a_inter) == float(b_inter) == 0.0


def test_inter_mode_scores_only_cross_chain_pairs():
    gt, asym, tok, fa, ftok, fat, fm, sm, pm = _fixture()
    pred = gt.clone()
    pred[asym == 1] += torch.tensor([0.0, 12.0, 0.0])

    intra, inter = _call(pred, gt, asym, tok, fa, ftok, fat, fm, sm, pm, mode="inter")
    assert float(intra) == 0.0
    both_intra, both_inter = _call(pred, gt, asym, tok, fa, ftok, fat, fm, sm, pm, mode="both")
    assert torch.allclose(inter, both_inter, atol=1e-6)  # same pairs, same clamp


def test_unknown_mode_and_missing_asym_id_raise():
    import pytest

    gt, asym, tok, fa, ftok, fat, fm, sm, pm = _fixture()
    with pytest.raises(ValueError, match="Unknown fape mode"):
        _call(gt.clone(), gt, asym, tok, fa, ftok, fat, fm, sm, pm, mode="cross")
    with pytest.raises(ValueError, match="requires asym_id"):
        fape_loss(gt.unsqueeze(0), gt, ftok, fat, fm, fa, tok, sm, sm, pm, mode="both", asym_id=None)
