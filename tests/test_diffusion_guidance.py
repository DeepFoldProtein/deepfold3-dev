"""Classifier-free guidance in ``DiffusionHead.sample`` and the ``cond_drop`` modes."""

from __future__ import annotations

import pytest
import torch

from deepfold3.model.config import DiffusionConfig, OptimConfig
from deepfold3.model.diffusion import (
    DiffusionHead,
    draw_interface_pair_keep,
    interface_pair_keep,
    mask_trunk_pair,
)

N, M = 3, 2


def _head(**kw) -> DiffusionHead:
    cfg = DiffusionConfig(steps=4, num_samples=2, **kw)
    head = DiffusionHead(cfg, c_single=384, c_pair=128, rel_feat_dim=8, optim=OptimConfig()).eval()
    # AF3 zero-inits the output projections, which would make every denoiser
    # output identical and every comparison below vacuous.
    torch.manual_seed(0)
    with torch.no_grad():
        for p in head.parameters():
            p.normal_(0.0, 0.05)
    return head


def _inputs(pair: torch.Tensor | None = None) -> dict:
    g = torch.Generator().manual_seed(1)
    single = torch.randn(1, N, 384, generator=g)
    drawn = torch.randn(1, N, N, 128, generator=g)  # always drawn: keeps the stream aligned
    return {
        "embeddings": {
            "single": single,
            "pair": drawn if pair is None else pair,
            "target_feat": torch.randn(1, N, 447, generator=g),
        },
        "ref_pos": torch.randn(1, N, M, 3, generator=g),
        "ref_mask": torch.ones(1, N, M),
        "ref_element": torch.zeros(1, N, M, dtype=torch.int64),
        "ref_charge": torch.zeros(1, N, M),
        "ref_atom_name_chars": torch.zeros(1, N, M, 4, dtype=torch.int64),
        "seq_mask": torch.ones(1, N),
        "rel_feat": torch.randn(1, N, N, 8, generator=g),
    }


def _sample(head: DiffusionHead, inputs: dict) -> torch.Tensor:
    return head.sample(**inputs, generator=torch.Generator().manual_seed(2))["atom_positions"]


def _static(head: DiffusionHead, inputs: dict, use_conditioning: bool):
    return head.compute_static(**inputs, use_conditioning=use_conditioning)


@pytest.mark.parametrize("mode", ["pair", "both"])
def test_cond_drop_zeroes_only_what_it_names(mode):
    head, x = _head(cond_drop=mode), _inputs()
    with torch.no_grad():
        c, u = _static(head, x, True), _static(head, x, False)
        zero_pair = _inputs(pair=torch.zeros(1, N, N, 128))
        ref = _static(head, zero_pair, True)
    assert torch.allclose(u.pair_cond, ref.pair_cond)
    assert not torch.allclose(u.pair_cond, c.pair_cond)
    same_single = torch.allclose(u.single_cond_pre_noise, c.single_cond_pre_noise)
    assert same_single == (mode == "pair")


def test_default_drop_is_pair():
    assert DiffusionConfig().cond_drop == "pair"


def test_scale_one_is_the_unguided_path(monkeypatch):
    head, x = _head(), _inputs()
    calls = []
    orig = head.compute_static

    def spy(*a, **kw):
        calls.append(kw.get("use_conditioning", True))
        return orig(*a, **kw)

    monkeypatch.setattr(head, "compute_static", spy)
    base = _sample(head, x)
    assert calls == [True]  # no unconditional static, no second forward

    head.cfg.guidance_scale = 2.0
    calls.clear()
    guided = _sample(head, x)
    assert calls == [True, False]
    assert not torch.allclose(base, guided)


def test_scale_zero_is_the_unconditional_model():
    head = _head(guidance_scale=0.0)
    guided = _sample(head, _inputs())
    head.cfg.guidance_scale = 1.0
    uncond = _sample(head, _inputs(pair=torch.zeros(1, N, N, 128)))
    assert torch.allclose(guided, uncond, atol=1e-5)


def test_sigma_range_outside_schedule_is_unguided():
    x = _inputs()
    base = _sample(_head(), x)
    off = _sample(_head(guidance_scale=3.0, guidance_sigma_range=(1e6, 1e7)), x)
    assert torch.equal(base, off)


def test_config_validation():
    with pytest.raises(ValueError):
        DiffusionConfig(cond_drop="single")
    with pytest.raises(ValueError):
        DiffusionConfig(guidance_sigma_range=(5.0, 1.0))
    assert DiffusionConfig(guidance_sigma_range=[1.0, 5.0]).guidance_sigma_range == (1.0, 5.0)


def _asym(*sizes: int, pad: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    ids = [c for c, n in enumerate(sizes) for _ in range(n)] + [0] * pad
    mask = [1] * sum(sizes) + [0] * pad
    return torch.tensor([ids]), torch.tensor([mask])


def test_interface_keep_drops_every_inter_chain_pair():
    asym, _ = _asym(2, 1)
    keep = interface_pair_keep(asym)[0]
    assert keep.tolist() == [[True, True, False], [True, True, False], [False, False, True]]


def test_drawn_interface_keep_is_a_proper_chain_pair_subset():
    asym, seq_mask = _asym(2, 2, 2, 1, pad=2)
    g = torch.Generator().manual_seed(0)
    n_dropped = set()
    for _ in range(200):
        keep = draw_interface_pair_keep(asym, seq_mask, generator=g)[0]
        assert torch.equal(keep, keep.T)
        assert bool(keep[interface_pair_keep(asym)[0] & seq_mask[0].bool()[:, None]].all())  # intra kept
        assert bool(keep[-2:].all() and keep[:, -2:].all())  # padding untouched
        # a whole chain-pair block at a time
        blocks = {(int(asym[0, i]), int(asym[0, j])): set() for i in range(7) for j in range(7)}
        for i in range(7):
            for j in range(7):
                blocks[(int(asym[0, i]), int(asym[0, j]))].add(bool(keep[i, j]))
        assert all(len(v) == 1 for v in blocks.values())
        dropped = sum(1 for (a, b), v in blocks.items() if a < b and v == {False})
        assert dropped >= 1
        n_dropped.add(dropped)
    assert 1 in n_dropped and 6 in n_dropped  # single interfaces through all of them


def test_drawn_interface_keep_needs_two_chains():
    asym, seq_mask = _asym(4, pad=2)
    assert draw_interface_pair_keep(asym, seq_mask) is None


def test_masked_pair_matches_the_full_drop_where_dropped():
    head, x = _head(), _inputs()
    asym, _ = _asym(2, 1)
    keep = interface_pair_keep(asym)
    with torch.no_grad():
        c, u = _static(head, x, True), _static(head, x, False)
        part = head.compute_static(**{**x, "embeddings": mask_trunk_pair(x["embeddings"], keep)})
    k = keep[0]
    assert torch.allclose(part.pair_cond[0][k], c.pair_cond[0][k])
    assert torch.allclose(part.pair_cond[0][~k], u.pair_cond[0][~k])
    assert torch.allclose(part.single_cond_pre_noise, c.single_cond_pre_noise)


def test_guidance_pair_keep_is_the_unconditional_branch():
    asym, _ = _asym(2, 1)
    keep = interface_pair_keep(asym)
    x = _inputs()
    head = _head(guidance_scale=0.0, guidance_drop="interface")
    guided = head.sample(**x, generator=torch.Generator().manual_seed(2), guidance_pair_keep=keep)["atom_positions"]
    head.cfg.guidance_scale = 1.0
    ref = _sample(head, {**x, "embeddings": mask_trunk_pair(x["embeddings"], keep)})
    assert torch.allclose(guided, ref, atol=1e-5)


def test_interface_guidance_needs_a_keep_mask():
    with pytest.raises(ValueError):
        _sample(_head(guidance_scale=2.0, guidance_drop="interface"), _inputs())


def test_guidance_false_is_unguided():
    x = _inputs()
    base = _sample(_head(), x)
    off = _head(guidance_scale=3.0).sample(**x, generator=torch.Generator().manual_seed(2), guidance=False)
    assert torch.equal(base, off["atom_positions"])


def test_interface_config_validation():
    with pytest.raises(ValueError):
        DiffusionConfig(guidance_drop="chain")
    with pytest.raises(ValueError):
        DiffusionConfig(cond_drop_interface_frac=1.5)


class _Stop(Exception):
    pass


def _tiny_model(**diff_kw):
    from deepfold3.model.model import DeepFold3
    from tests.test_batch import _tiny_config

    cfg = _tiny_config()
    for k, v in diff_kw.items():
        setattr(cfg.diffusion, k, v)
    return DeepFold3(cfg)


def _tiny_batch():
    from deepfold3.data.module.collate import collate_fn
    from tests.test_batch import _make_features

    f = _make_features(6)
    f["asym_id"] = torch.tensor([0, 0, 0, 1, 1, 2])
    f["atom_gt_positions"] = torch.randn(6, 24, 3)
    f["atom_gt_mask"] = f["ref_mask"].clone()
    return collate_fn([f])


def test_training_interface_drop_masks_only_inter_chain_pair():
    """End to end through ``forward_training``: an always-firing interface drop
    reaches the diffusion head as a masked trunk pair with conditioning on,
    and the mini-rollout runs unguided even with ``guidance_scale`` set."""
    model = _tiny_model(use_conditioning_prob=0.0, cond_drop_interface_frac=1.0, guidance_scale=2.0).train()
    batch = _tiny_batch()
    seen = {}
    sample = model.diffusion_head.sample

    def spy_sample(*a, **kw):
        seen.setdefault("rollout_guidance", kw.get("guidance", True))
        return sample(*a, **kw)

    def spy_training(*a, **kw):
        seen["pair"], seen["use_conditioning"] = kw["embeddings"]["pair"][0], kw["use_conditioning"]
        raise _Stop

    model.diffusion_head.sample = spy_sample
    model.diffusion_head.sample_training = spy_training
    torch.manual_seed(0)
    with pytest.raises(_Stop):
        model.forward_training(
            batch, num_recycles=0, diffusion_batch_size=2, mini_rollout_steps=2, compute_confidence=False
        )
    assert seen["rollout_guidance"] is False
    assert seen["use_conditioning"] is True
    zero = seen["pair"].abs().sum(-1) == 0
    intra = interface_pair_keep(batch["asym_id"])[0]
    assert not bool(zero[intra].any())  # intra-chain pair untouched
    assert bool(zero[~intra].any())  # at least one interface dropped
    assert torch.equal(zero, zero.T)


def test_inference_interface_guidance_passes_the_interface_mask():
    model = _tiny_model(guidance_scale=2.0, guidance_drop="interface").eval()
    batch = _tiny_batch()
    seen = {}
    sample = model.diffusion_head.sample

    def spy(*a, **kw):
        seen["keep"] = kw.get("guidance_pair_keep")
        return sample(*a, **kw)

    model.diffusion_head.sample = spy
    with torch.no_grad():
        model(batch, num_recycles=0)
    assert torch.equal(seen["keep"], interface_pair_keep(batch["asym_id"]))
