"""Classifier-free guidance in ``DiffusionHead.sample`` and the ``cond_drop`` modes."""

from __future__ import annotations

import pytest
import torch

from deepfold3.model.config import DiffusionConfig, OptimConfig
from deepfold3.model.diffusion import DiffusionHead

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
