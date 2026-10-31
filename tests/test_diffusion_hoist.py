"""``DiffusionConfig.hoist_sampler`` against the sampler it replaces."""

from __future__ import annotations

import pytest
import torch

import deepfold3.model.diffusion as diffusion
from deepfold3.model.atom_cross_attn import build_atom_subsets
from deepfold3.model.config import DiffusionConfig, OptimConfig
from deepfold3.model.diffusion import DiffusionHead, GuidanceTerm, interface_pair_keep

N, M = 40, 8
ATOL = 1e-3  # Å, fp32 on CPU: observed up to 3e-4 after 4 steps. GPU TF32 is coarser, see docs/optimizations.md


def _head(steps: int = 4, num_samples: int = 3, **kw) -> DiffusionHead:
    cfg = DiffusionConfig(steps=steps, num_samples=num_samples, **kw)
    head = DiffusionHead(cfg, c_single=384, c_pair=128, rel_feat_dim=8, optim=OptimConfig()).eval()
    # AF3 zero-inits the output projections, which would make every comparison vacuous.
    torch.manual_seed(0)
    with torch.no_grad():
        for p in head.parameters():
            p.normal_(0.0, 0.05)
    return head


def _inputs(batch: tuple[int, ...] = (), pad_tokens: int = 0) -> dict:
    g = torch.Generator().manual_seed(1)
    ref_mask = torch.zeros(*batch, N, M)
    ref_mask[..., : N - pad_tokens, :5] = 1.0  # most slots empty, as in a real protein
    seq_mask = torch.zeros(*batch, N)
    seq_mask[..., : N - pad_tokens] = 1.0
    return {
        "embeddings": {
            "single": torch.randn(*batch, N, 384, generator=g),
            "pair": torch.randn(*batch, N, N, 128, generator=g),
            "target_feat": torch.randn(*batch, N, 447, generator=g),
        },
        "ref_pos": torch.randn(*batch, N, M, 3, generator=g) * ref_mask.unsqueeze(-1),
        "ref_mask": ref_mask,
        "ref_element": torch.zeros(*batch, N, M, dtype=torch.int64),
        "ref_charge": torch.zeros(*batch, N, M),
        "ref_atom_name_chars": torch.zeros(*batch, N, M, 4, dtype=torch.int64),
        "seq_mask": seq_mask,
        "rel_feat": torch.randn(*batch, N, N, 8, generator=g),
    }


def _keep(batch: tuple[int, ...] = ()) -> torch.Tensor:
    asym = (torch.arange(N) >= N // 2).long().expand(*batch, N)
    return interface_pair_keep(asym)


def _terms(n: int, batch: tuple[int, ...] = ()) -> list[GuidanceTerm]:
    return [GuidanceTerm(1.5), GuidanceTerm(3.0, _keep(batch))][:n]


def _run(head: DiffusionHead, x: dict, terms, hoist: bool) -> torch.Tensor:
    head.cfg.hoist_sampler = hoist
    gen = torch.Generator().manual_seed(2)
    return head.sample(**x, generator=gen, guidance_terms=terms)["atom_positions"]


def test_flag_is_off_by_default_and_on_for_the_inference_cli():
    from deepfold3.scripts.inference import inference

    assert DiffusionConfig().hoist_sampler is False
    option = next(p for p in inference.params if p.name == "hoist_sampler")
    assert option.default is True


@pytest.mark.parametrize("n_terms", [0, 1, 2])
def test_matches_the_plain_sampler(n_terms):
    head, x = _head(), _inputs()
    a, b = _run(head, x, _terms(n_terms), False), _run(head, x, _terms(n_terms), True)
    assert a.abs().max() > 1.0  # not vacuous
    assert torch.allclose(a, b, atol=ATOL, rtol=0.0)


@pytest.mark.parametrize("n_terms", [0, 2])
def test_matches_with_batch_and_padding(n_terms):
    head, x = _head(), _inputs(batch=(2,), pad_tokens=6)
    terms = _terms(n_terms, batch=(2,))
    assert torch.allclose(_run(head, x, terms, False), _run(head, x, terms, True), atol=ATOL, rtol=0.0)


def test_matches_with_padded_tokens_unbatched():
    head, x = _head(), _inputs(pad_tokens=6)
    assert torch.allclose(_run(head, x, [], False), _run(head, x, [], True), atol=ATOL, rtol=0.0)


def test_matches_with_a_full_atom_layout():
    """No window to drop: the hoisting alone has to hold."""
    head, x = _head(), _inputs()
    x["ref_mask"] = torch.ones(N, M)
    assert torch.allclose(_run(head, x, [], False), _run(head, x, [], True), atol=ATOL, rtol=0.0)


def test_sigma_ranges_still_apply_and_out_of_range_costs_no_pass(monkeypatch):
    head, x = _head(steps=6), _inputs()
    calls = []
    orig = head._denoise_samples

    def spy(*a, **kw):
        calls.append(1)
        return orig(*a, **kw)

    monkeypatch.setattr(head, "_denoise_samples", spy)
    terms = [GuidanceTerm(2.0, sigma_range=(1e6, 1e7)), GuidanceTerm(3.0, _keep(), sigma_range=(1.0, 300.0))]
    a = _run(head, x, terms, False)
    plain_calls = len(calls)
    calls.clear()
    b = _run(head, x, terms, True)
    assert len(calls) == plain_calls
    assert head.cfg.steps < plain_calls < 2 * head.cfg.steps  # the 2nd term gates some steps
    assert torch.allclose(a, b, atol=ATOL, rtol=0.0)


def test_rng_stream_is_untouched(monkeypatch):
    """Every draw matches exactly, and each sample gets its own noise."""
    head, x = _head(), _inputs()
    draws: list[list[torch.Tensor]] = []
    randn_like = torch.randn_like
    randn = torch.randn

    def spy_like(*a, **kw):
        draws[-1].append(randn_like(*a, **kw))
        return draws[-1][-1]

    def spy_randn(*a, **kw):
        draws[-1].append(randn(*a, **kw))
        return draws[-1][-1]

    monkeypatch.setattr(diffusion.torch, "randn_like", spy_like)
    monkeypatch.setattr(diffusion.torch, "randn", spy_randn)
    gens = []
    for hoist in (False, True):
        draws.append([])
        head.cfg.hoist_sampler = hoist
        gens.append(torch.Generator().manual_seed(2))
        head.sample(**x, generator=gens[-1], guidance_terms=_terms(2))
    assert len(draws[0]) == len(draws[1]) > 0
    assert all(torch.equal(u, v) for u, v in zip(*draws, strict=True))
    assert torch.equal(gens[0].get_state(), gens[1].get_state())
    noise = [d for d in draws[1] if d.shape == (3, N, M, 3)]
    assert len(noise) == head.cfg.steps + 1
    for d in noise:
        assert not torch.equal(d[0], d[1]) and not torch.equal(d[1], d[2])


def test_trim_keeps_real_atoms_and_drops_empty_windows():
    head, x = _head(), _inputs()
    with torch.no_grad():
        x1 = {
            k: v.unsqueeze(0) if torch.is_tensor(v) else {a: b.unsqueeze(0) for a, b in v.items()} for k, v in x.items()
        }
        enc = head.compute_static(**x1).encoder_static
    trimmed = enc.subsets.trim_empty()
    assert enc.subsets.num_subsets == (N * M + 31) // 32
    assert trimmed.num_subsets == (N * 5 + 31) // 32
    assert int(trimmed.ta_to_q_mask.sum()) == N * 5
    assert int(trimmed.q_to_k_idx.max()) < trimmed.num_subsets * 32
    assert int(trimmed.q_to_ta_idx.max()) < trimmed.num_subsets * 32
    dropped = enc.subsets.q_to_k_idx[:, : trimmed.num_subsets] >= trimmed.num_subsets * 32
    assert bool(dropped.any()) and not bool(enc.subsets.q_to_k_mask[:, : trimmed.num_subsets][dropped].any())


def test_trim_of_an_empty_molecule_keeps_one_window():
    assert build_atom_subsets(torch.zeros(1, 4, 8)).trim_empty().num_subsets == 1


def test_hoisted_state_is_one_copy():
    head, x = _head(num_samples=5), _inputs()
    with torch.no_grad():
        x1 = {
            k: v.unsqueeze(0) if torch.is_tensor(v) else {a: b.unsqueeze(0) for a, b in v.items()} for k, v in x.items()
        }
        st = head.hoist_static(head.compute_static(**x1), 5)
    enc = st.encoder_static
    assert enc.atom_pair.shape[0] == 5 and enc.atom_pair.numel() == enc.atom_pair[0].numel() * 5
    assert enc.atom_pair[0].data_ptr() == enc.atom_pair[4].data_ptr()  # a view
    assert st.pair_logits[0].shape[0] == 1 and enc.encoder_hoist[0].q_scale.shape[0] == 1


def test_oom_propagates(monkeypatch):
    """Nothing in the path catches it and retries without hoisting."""
    head, x = _head(), _inputs()
    calls = []

    def boom(*a, **kw):
        calls.append(1)
        raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    monkeypatch.setattr(head.transformer, "forward_hoisted", boom)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        _run(head, x, [], True)
    assert calls == [1]


def test_training_rollout_stays_on_the_plain_path(monkeypatch):
    from tests.test_diffusion_guidance import _tiny_batch, _tiny_model

    model = _tiny_model().train()
    called = []
    monkeypatch.setattr(model.diffusion_head, "hoist_static", lambda *a: called.append(1))
    sample = model.diffusion_head.sample

    class Stop(Exception):
        pass

    def stop(*a, **kw):
        sample(*a, **kw)
        raise Stop

    monkeypatch.setattr(model.diffusion_head, "sample", stop)
    with pytest.raises(Stop):
        model.forward_training(
            _tiny_batch(), num_recycles=0, diffusion_batch_size=2, mini_rollout_steps=2, compute_confidence=False
        )
    assert called == []
