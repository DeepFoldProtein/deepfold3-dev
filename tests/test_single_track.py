"""Deleting the trunk's per-token track, and what that does downstream.

The claims the arm rests on, in the order they matter:

1. ``single`` is a *sink* in a PairFormer block — it reads the pair
   representation and never writes back — so removing it leaves the pair track
   bit-identical. If that were false the ablation would be measuring two things.
2. With the track gone the trunk's single output is a pointwise function of
   ``target_feat``: no token mixing, no dependence on MSA, templates, or the
   recycled pair. This is what makes the diffusion conditioning's
   ``cat([s_trunk, target_feat])`` a duplicate rather than a second view.
3. ``conditioning_use_trunk_single=False`` therefore drops a channel that
   carries nothing — and the module must genuinely ignore it, not merely
   down-weight it.
"""

from __future__ import annotations

import logging

import pytest
import torch

from deepfold3.model.config import EvoformerConfig, ModelConfig, PairFormerConfig
from deepfold3.model.diffusion import DiffusionConditioningBlock
from deepfold3.model.evoformer import Evoformer, PairFormerBlock
from tests._util import wake_zero_init

C_Z, C_S, N = 32, 16, 8
TARGET_FEAT_DIM = 447


def _block(use_single_track: bool) -> PairFormerBlock:
    torch.manual_seed(0)
    return (
        PairFormerBlock(
            C_Z,
            C_S,
            num_pair_heads=4,
            num_single_heads=2,
            single_key_dim=C_S,
            single_value_dim=C_S,
            pair_dropout=0.0,
        ).eval()
        if use_single_track
        else PairFormerBlock(
            C_Z,
            C_S,
            num_pair_heads=4,
            num_single_heads=2,
            single_key_dim=C_S,
            single_value_dim=C_S,
            pair_dropout=0.0,
            use_single_track=False,
        ).eval()
    )


@pytest.fixture
def acts():
    torch.manual_seed(1)
    return (
        torch.randn(1, N, N, C_Z),
        torch.randn(1, N, C_S),
        torch.ones(1, N, N),
        torch.ones(1, N),
    )


def test_deleting_the_track_removes_its_parameters(acts):
    pair, single, pair_mask, seq_mask = acts
    blk = wake_zero_init(_block(use_single_track=False))
    out_pair, out_single = blk(pair, single, pair_mask, seq_mask)
    named = [n for n, _ in blk.named_parameters()]
    assert not any(t in n for n in named for t in ("single_attn", "single_transition", "single_to_pair"))
    assert torch.equal(out_single, single), "single must pass through untouched"
    assert out_pair.shape == pair.shape and torch.isfinite(out_pair).all()
    assert not torch.equal(out_pair, pair), "the pair track must still do something"


def test_the_single_track_is_a_sink(acts):
    """Same pair weights, with and without the track -> identical pair output.

    This is the load-bearing claim: ``single_to_pair`` reads the pair
    representation to build the PairBias logits, and nothing on the single side
    is ever added back into ``pair``. So the deletion is a subtraction of a
    read-out, not of a pair-track input.
    """
    pair, single, pair_mask, seq_mask = acts
    with_track = wake_zero_init(_block(use_single_track=True))
    without = _block(use_single_track=False)
    # The single-track keys exist in the source dict and not in the target
    # module, so they surface as *unexpected*; nothing may be missing.
    missing, unexpected = without.load_state_dict(with_track.state_dict(), strict=False)
    assert not missing, f"the pair track must be fully transferable: {missing}"
    assert unexpected and all(
        any(t in k for t in ("single_attn", "single_transition", "single_to_pair")) for k in unexpected
    ), f"only single-track keys may be left over: {unexpected}"

    pair_a, _ = with_track(pair, single, pair_mask, seq_mask)
    pair_b, _ = without(pair, single, pair_mask, seq_mask)
    assert pair_a.abs().max() > 1e-3, "the comparison must not be zero vs zero"
    assert torch.equal(pair_a, pair_b)


def _trunk(use_single_track: bool) -> Evoformer:
    torch.manual_seed(0)
    evo = EvoformerConfig(msa_channel=8, pair_channel=C_Z, seq_channel=C_S, num_layer=1)
    evo.msa_attention.num_head = 2
    evo.pair_attention.num_head = 2
    pf = PairFormerConfig(
        num_layer=1,
        pair_channel=C_Z,
        seq_channel=C_S,
        single_attention_num_head=2,
        single_key_dim=C_S,
        single_value_dim=C_S,
        pair_dropout=0.0,
        use_single_track=use_single_track,
    )
    pf.pair_attention.num_head = 2
    return wake_zero_init(Evoformer(evo, pf, template_cfg=None).eval())


def _trunk_inputs(seed: int):
    torch.manual_seed(seed)
    return dict(
        msa_feat=torch.randn(2, N, 34),
        rel_feat=torch.randn(N, N, 139),
        prev_pair=torch.randn(N, N, C_Z),
    )


def test_trunk_single_is_pair_independent_when_the_track_is_off():
    """Everything except ``target_feat`` may change; the single output may not.

    A single representation that cannot see the MSA, the templates, or the
    recycled pair is not carrying trunk information — which is exactly why
    concatenating it onto ``target_feat`` downstream buys nothing.
    """
    trunk = _trunk(use_single_track=False)
    torch.manual_seed(7)
    target_feat = torch.randn(N, TARGET_FEAT_DIM)
    seq_mask, msa_mask = torch.ones(N), torch.ones(2, N)
    pair_mask = torch.ones(N, N)
    prev_single = torch.randn(N, C_S)

    outs = []
    for seed in (11, 12):
        v = _trunk_inputs(seed)
        with torch.no_grad():
            outs.append(
                trunk(
                    target_feat=target_feat,
                    msa_feat=v["msa_feat"],
                    rel_feat=v["rel_feat"],
                    pair_mask=pair_mask,
                    seq_mask=seq_mask,
                    msa_mask=msa_mask,
                    prev_pair=v["prev_pair"],
                    prev_single=prev_single,
                )
            )
    assert not torch.equal(outs[0]["pair"], outs[1]["pair"]), "the pair track must respond to the inputs"
    assert torch.equal(outs[0]["single"], outs[1]["single"])
    # Stronger: it is exactly s_init, with the recycle projection gone too.
    assert torch.equal(outs[0]["single"], trunk.single_activations(target_feat))
    assert not hasattr(trunk, "prev_single_embedding")


def test_trunk_single_does_depend_on_the_pair_track_when_the_track_is_on():
    """The control — without it the test above could pass on a broken forward."""
    trunk = _trunk(use_single_track=True)
    torch.manual_seed(7)
    target_feat = torch.randn(N, TARGET_FEAT_DIM)
    outs = []
    for seed in (11, 12):
        v = _trunk_inputs(seed)
        with torch.no_grad():
            outs.append(
                trunk(
                    target_feat=target_feat,
                    msa_feat=v["msa_feat"],
                    rel_feat=v["rel_feat"],
                    pair_mask=torch.ones(N, N),
                    seq_mask=torch.ones(N),
                    msa_mask=torch.ones(2, N),
                    prev_pair=v["prev_pair"],
                    prev_single=torch.randn(N, C_S),
                )
            )
    assert not torch.equal(outs[0]["single"], outs[1]["single"])


@pytest.mark.parametrize("use_trunk_single", [True, False])
def test_conditioning_input_width_follows_the_flag(use_trunk_single):
    cond = DiffusionConditioningBlock(
        c_pair_in=C_Z,
        c_single_in=C_S,
        rel_feat_dim=139,
        c_pair_out=C_Z,
        c_single_out=C_S,
        target_feat_dim=TARGET_FEAT_DIM,
        use_trunk_single=use_trunk_single,
    )
    expected = TARGET_FEAT_DIM + (C_S if use_trunk_single else 0)
    assert cond.single_cond_proj.weight.shape[1] == expected
    assert cond.single_cond_norm_w.numel() == expected


def test_conditioning_ignores_the_trunk_single_when_disabled():
    """Not down-weighted — ignored. Feed it noise and the output must not move."""
    cond = wake_zero_init(
        DiffusionConditioningBlock(
            c_pair_in=C_Z,
            c_single_in=C_S,
            rel_feat_dim=139,
            c_pair_out=C_Z,
            c_single_out=C_S,
            target_feat_dim=TARGET_FEAT_DIM,
            use_trunk_single=False,
        ).eval()
    )
    torch.manual_seed(3)
    target_feat = torch.randn(N, TARGET_FEAT_DIM)
    pair_emb, rel_feat = torch.randn(N, N, C_Z), torch.randn(N, N, 139)
    with torch.no_grad():
        a, _ = cond.compute_static(torch.randn(N, C_S), pair_emb, target_feat, rel_feat)
        b, _ = cond.compute_static(torch.randn(N, C_S) * 100, pair_emb, target_feat, rel_feat)
    assert a.abs().max() > 1e-3, "the comparison must not be zero vs zero"
    assert torch.equal(a, b)


def test_config_warns_when_the_duplicate_half_is_left_in(caplog):
    cfg = ModelConfig()
    cfg.pairformer.use_single_track = False
    with caplog.at_level(logging.WARNING, logger="deepfold3.model.config"):
        cfg.__post_init__()
    assert "conditioning_use_trunk_single" in caplog.text

    caplog.clear()
    cfg.diffusion.conditioning_use_trunk_single = False
    with caplog.at_level(logging.WARNING, logger="deepfold3.model.config"):
        cfg.__post_init__()
    assert caplog.text == ""


def test_defaults_are_unchanged():
    """AF3 behaviour is the default on both flags; only a config opts out."""
    assert PairFormerConfig().use_single_track is True
    assert ModelConfig().diffusion.conditioning_use_trunk_single is True


def test_the_flag_does_not_reach_the_confidence_head():
    """Trunk scope only.

    The confidence head runs its own PairFormer over trunk pair + the predicted
    distogram, and rebuilding a per-token summary there is the point of that
    stack — so it keeps its single track no matter what the trunk does. It also
    reads its own config class, which has no such field; this test pins the
    scope rather than trusting that.
    """
    from deepfold3.model.confidence import ConfidenceHead

    cfg = ModelConfig()
    cfg.pairformer.use_single_track = False
    assert not hasattr(cfg.confidence, "use_single_track")
    assert not hasattr(cfg.template.pairformer, "use_single_track")

    conf_cfg = cfg.confidence
    conf_cfg.num_layer, conf_cfg.pair_channel, conf_cfg.seq_channel = 1, C_Z, C_S
    conf_cfg.pair_attention_num_head, conf_cfg.single_attention_num_head = 2, 2
    head = ConfidenceHead(conf_cfg)
    assert any("single_attn" in n for n, _ in head.named_parameters())


def _tiny_pairmixer_config() -> ModelConfig:
    """Full model, minimal depth, with both flags off.

    Channel sizes stay at 384/128 — atom cross-attention hardcodes them.
    """
    from deepfold3.model.config import DiffusionConfig, DiffusionTransformerConfig

    cfg = ModelConfig(
        evoformer=EvoformerConfig(num_layer=1, use_triangle_attention=False),
        pairformer=PairFormerConfig(num_layer=1, use_triangle_attention=False, use_single_track=False),
        diffusion=DiffusionConfig(
            steps=2,
            num_samples=1,
            conditioning_use_trunk_single=False,
            transformer=DiffusionTransformerConfig(num_blocks=2, super_block_size=2),
        ),
        num_recycles=1,
        msa_crop_size=8,
    )
    return cfg


def test_full_model_runs_without_the_track(small_features):
    """The recycling loop still passes ``prev_single`` around; nothing may trip.

    ``model.py`` seeds and stores ``_prev_single`` regardless of the flag, so the
    plumbing survives while the projection that consumed it does not. This is
    the integration check that the two do not disagree.
    """
    from deepfold3.model.model import DeepFold3

    model = wake_zero_init(DeepFold3(_tiny_pairmixer_config()).eval())
    assert not any("prev_single_embedding" in n for n, _ in model.named_parameters())
    assert not any(
        n.startswith("evoformer.trunk_pairformer") and t in n
        for n, _ in model.named_parameters()
        for t in ("single_attn", "single_transition", "single_to_pair")
    )
    # The confidence head keeps its own single track.
    assert any(n.startswith("confidence_head") and "single_attn" in n for n, _ in model.named_parameters())

    with torch.no_grad():
        out = model(small_features)
    assert torch.isfinite(out.atom_positions).all()
    assert out.distogram is not None and torch.isfinite(out.distogram).all()
    assert out.plddt is not None and torch.isfinite(out.plddt).all()


def test_s_init_still_receives_gradient_without_the_track(small_features):
    """The atom encoder is what keeps ``single_activations`` alive. Pin it.

    With the track off and the conditioning concat dropped, the obvious reading
    is that nothing trains ``single_activations``: the confidence head takes its
    single detached, and the diffusion *conditioning* no longer reads it. That
    reading is wrong, and the way it is wrong matters — the diffusion **atom**
    encoder conditions per-atom activations on the trunk single
    (``AtomEncoder.compute_static(trunk_single=...)``), and it receives no
    ``target_feat`` of its own, so that is its only token-level identity
    channel.

    If a future change removes that consumer, this weight becomes an unused
    parameter and DDP (``find_unused_parameters=False``) raises on the first
    backward of a 96-rank run. This test fails first instead.
    """
    from deepfold3.model.model import DeepFold3

    model = wake_zero_init(DeepFold3(_tiny_pairmixer_config()))

    # The sampler in ``forward`` runs under no_grad, so exercise the piece that
    # actually consumes the trunk single: the noise-independent half of the
    # diffusion head, which builds both the conditioning and the atom-encoder
    # static state.
    embeddings, rel_feat, _target_feat, seq_mask, _pair_mask = model.run_evoformer(small_features, num_recycles=0)
    static = model.diffusion_head.compute_static(
        embeddings=embeddings,
        seq_mask=seq_mask,
        rel_feat=rel_feat,
        **model._ref_kwargs(small_features),
    )
    leaves = [t for t in vars(static.encoder_static).values() if torch.is_tensor(t) and t.is_floating_point()]
    assert leaves, "the atom encoder produced no float state to differentiate"
    sum(t.float().sum() for t in leaves).backward()

    grad = model.evoformer.single_activations.weight.grad
    assert grad is not None, "s_init has no consumer left — DDP would raise at step 1"
    assert grad.abs().sum() > 0

    # And the conditioning genuinely does not read it, in the same model.
    assert model.diffusion_head.conditioning.use_trunk_single is False
