"""Masked/partial diffusion + conditioning dropout — the three off-by-default knobs.

``DiffusionConfig.diffusion_mask_prob`` (motif scaffolding, written from scratch —
OpenFold-3 has no equivalent), ``DiffusionConfig.use_conditioning_prob`` and
``ConfidenceConfig.use_pair_emb_prob`` (both mirrored from OF3).

The property that matters most is the first one tested here: at their defaults
none of them draws from the RNG, so turning the feature *on in the code* cannot
perturb an existing run's random stream.
"""

from __future__ import annotations

import pytest
import torch

from deepfold3.model.config import ConfidenceConfig, DiffusionConfig
from deepfold3.model.diffusion import build_diffusion_clean_mask


def test_defaults_are_inert():
    """Off by default, and the guards short-circuit before any RNG draw."""
    d, c = DiffusionConfig(), ConfidenceConfig()
    assert d.diffusion_mask_prob == 0.0
    assert d.use_conditioning_prob == 1.0
    assert c.use_pair_emb_prob == 1.0
    # `prob <= 0` returns before touching torch.rand, so the caller's stream is
    # untouched — this is what keeps enabling the *code* from changing an
    # existing run's sampling.
    torch.manual_seed(0)
    before = torch.rand(4)
    torch.manual_seed(0)
    assert (
        build_diffusion_clean_mask(torch.zeros(1, 8, dtype=torch.long), torch.ones(1, 8, dtype=torch.bool), 4, prob=0.0)
        is None
    )
    assert torch.equal(before, torch.rand(4))


def test_chain_mode_keeps_a_proper_subset():
    """Always ≥1 clean chain and ≥1 noised chain: something to place, and
    something to place it against."""
    asym = torch.tensor([[0] * 6 + [1] * 6 + [2] * 6])
    seq_mask = torch.ones(1, 18, dtype=torch.bool)
    torch.manual_seed(3)
    mask = build_diffusion_clean_mask(asym, seq_mask, 16, prob=1.0, mode="chain")
    assert mask is not None and mask.shape == (1, 16, 18)
    for s in range(16):
        clean_chains = {int(c) for c in asym[0][mask[0, s]].unique()}
        assert 0 < len(clean_chains) < 3, clean_chains
        # a whole chain at a time — never a partial chain
        for c in clean_chains:
            assert bool(mask[0, s][asym[0] == c].all())


def test_span_mode_is_contiguous_and_proper():
    seq_mask = torch.ones(1, 20, dtype=torch.bool)
    torch.manual_seed(4)
    mask = build_diffusion_clean_mask(
        torch.zeros(1, 20, dtype=torch.long), seq_mask, 8, prob=1.0, mode="span", min_frac=0.2, max_frac=0.8
    )
    assert mask is not None
    for s in range(8):
        idx = mask[0, s].nonzero(as_tuple=True)[0]
        assert 0 < idx.numel() < 20  # never everything
        assert int(idx[-1] - idx[0]) == idx.numel() - 1  # contiguous


def test_single_chain_falls_back_to_span():
    """A single-chain crop has no proper chain subset; "chain" must not return
    an empty (or full) mask for it."""
    torch.manual_seed(5)
    mask = build_diffusion_clean_mask(
        torch.zeros(1, 16, dtype=torch.long), torch.ones(1, 16, dtype=torch.bool), 8, prob=1.0, mode="chain"
    )
    assert mask is not None
    counts = mask[0].sum(-1)
    assert bool(((counts > 0) & (counts < 16)).all())


def test_mask_respects_padding():
    """Padded tokens are never marked clean — their GT is meaningless."""
    seq_mask = torch.zeros(1, 12, dtype=torch.bool)
    seq_mask[0, :8] = True
    asym = torch.tensor([[0] * 4 + [1] * 4 + [9] * 4])
    torch.manual_seed(6)
    mask = build_diffusion_clean_mask(asym, seq_mask, 8, prob=1.0, mode="chain")
    assert mask is not None
    assert not bool(mask[0][:, 8:].any())


def test_prob_controls_the_fired_fraction():
    asym = torch.tensor([[0] * 8 + [1] * 8])
    seq_mask = torch.ones(1, 16, dtype=torch.bool)
    torch.manual_seed(7)
    mask = build_diffusion_clean_mask(asym, seq_mask, 4096, prob=0.1, mode="chain")
    fired = (mask[0].sum(-1) > 0).float().mean()
    assert 0.07 < float(fired) < 0.13, fired


# ---------------------------------------------------------------------------
# Diffusion MSE scores only the noised region
# ---------------------------------------------------------------------------


def _mse(pred, gt, atom_mask, noise, **kw):
    from deepfold3.training.loss import diffusion_mse_loss

    return diffusion_mse_loss(pred, gt, atom_mask, noise, apply_rigid_align=False, **kw)


def _case(n_tok=6, n_atom=3, n_samples=2):
    atom_mask = torch.ones(n_tok, n_atom, dtype=torch.bool)
    gt = torch.zeros(n_samples, n_tok, n_atom, 3)
    pred = torch.zeros(n_samples, n_tok, n_atom, 3)
    noise = torch.full((n_samples,), 4.0)
    return pred, gt, atom_mask, noise


def test_all_true_mask_matches_the_unmasked_reduction():
    """The default path must not shift when the feature exists but nothing fired."""
    pred, gt, atom_mask, noise = _case()
    pred[:, :3] = 5.0
    ref = _mse(pred, gt, atom_mask, noise)
    all_noised = _mse(pred, gt, atom_mask, noise, noised_mask=torch.ones(2, 6, dtype=torch.bool))
    assert torch.allclose(ref, all_noised)


def test_clean_region_is_not_scored():
    """Error confined to the clean anchor must not reach the loss — otherwise the
    model is rewarded for copying an input it was handed."""
    pred, gt, atom_mask, noise = _case()
    pred[:, :3] = 5.0  # error only where the input was clean
    clean = torch.zeros(2, 6, dtype=torch.bool)
    clean[:, :3] = True
    assert float(_mse(pred, gt, atom_mask, noise)) > 0.0
    assert float(_mse(pred, gt, atom_mask, noise, noised_mask=~clean)) == 0.0


def test_noised_region_is_scored():
    pred, gt, atom_mask, noise = _case()
    pred[:, 3:] = 5.0  # error only where the input was noise
    clean = torch.zeros(2, 6, dtype=torch.bool)
    clean[:, :3] = True
    masked = float(_mse(pred, gt, atom_mask, noise, noised_mask=~clean))
    unmasked = float(_mse(pred, gt, atom_mask, noise))
    assert masked > 0.0
    # half the atoms are scored, so the per-atom average is 2x the pooled one
    assert masked == pytest.approx(2.0 * unmasked, rel=1e-5)


def test_denominator_is_per_sample():
    """Same per-atom error under different clean fractions ⇒ same loss. A shared
    denominator would make the more-masked sample look better than it is."""
    for n_noised in (1, 3, 5):
        pred, gt, atom_mask, noise = _case(n_samples=1)
        noised = torch.zeros(1, 6, dtype=torch.bool)
        noised[0, 6 - n_noised :] = True
        pred[0, 6 - n_noised :] = 2.0
        got = float(_mse(pred, gt, atom_mask, noise, noised_mask=noised))
        # per-coordinate mean of (2.0)^2 over 3 coords = 4.0, times the EDM weight
        w = (4.0**2 + 16.0**2) / (4.0 * 16.0) ** 2
        assert got == pytest.approx(w * 4.0, rel=1e-5), (n_noised, got)
