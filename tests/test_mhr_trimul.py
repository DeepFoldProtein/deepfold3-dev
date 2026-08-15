"""MHR-TriMul: head partition, per-head norm, and the target-pair router.

The claims worth pinning down are the ones the design note rests on:

1. the head partition changes nothing in the contraction itself,
2. a zero-initialized router is *exactly* AF3 TriMul, so every cell of the
   ablation grid starts from the same function and shares state_dict keys,
3. the router really does reallocate across heads (it is not absorbed by the
   output gate), and
4. deleting triangle attention removes the branches without disturbing the rest
   of the block.
"""

from __future__ import annotations

import pytest
import torch

from deepfold3.model.config import EvoformerConfig, PairFormerConfig
from deepfold3.model.evoformer import EvoformerBlock, PairFormerBlock, _trimul_kwargs
from deepfold3.model.ops import TriangleMulIncoming, TriangleMulOutgoing

C_Z, H, N = 64, 8, 12
MHR = dict(num_heads=H, head_norm="head_rms", router=True)


@pytest.fixture
def pair():
    torch.manual_seed(0)
    return torch.randn(1, N, N, C_Z), torch.ones(1, N, N)


def test_head_reshape_leaves_the_contraction_unchanged():
    """einsum('ikhd,jkhd->ijhd') == einsum('ikc,jkc->ijc') under c = (h, d)."""
    torch.manual_seed(0)
    a, b = torch.randn(N, N, C_Z), torch.randn(N, N, C_Z)
    flat = torch.einsum("ikc,jkc->ijc", a, b)
    heads = torch.einsum("ikhd,jkhd->ijhd", a.unflatten(-1, (H, C_Z // H)), b.unflatten(-1, (H, C_Z // H)))
    assert torch.equal(flat, heads.flatten(-2))


@pytest.mark.parametrize("cls", [TriangleMulOutgoing, TriangleMulIncoming])
def test_zero_init_router_is_exactly_af3(cls, pair):
    """g = H·softmax(0) = 1, so arm A is function-identical at initialization."""
    z, mask = pair
    base = _wake(cls(C_Z).eval())
    routed = _wake(cls(C_Z, num_heads=H, head_norm="channel", router=True).eval())
    missing, unexpected = routed.load_state_dict(base.state_dict(), strict=False)
    assert unexpected == [] and set(missing) == {"router.weight"}
    out_base, out_routed = base(z, mask), routed(z, mask)
    assert out_base.abs().max() > 1e-3, "the comparison must not be zero vs zero"
    assert torch.allclose(out_base, out_routed, atol=1e-5)


def test_per_head_affine_shares_norm_center_keys(pair):
    """Every ablation cell has identical state_dict keys, so any warm-starts from any."""
    keys = {
        k
        for cell in ({}, dict(num_heads=H, head_norm="head_rms"), MHR)
        for k in TriangleMulOutgoing(C_Z, **cell).state_dict()
    }
    base_keys = set(TriangleMulOutgoing(C_Z).state_dict())
    assert keys - base_keys == {"router.weight"}


def test_per_head_norm_gives_each_head_unit_scale(pair):
    """RMSNorm_d equalizes head magnitudes — the signal the router then has to re-create."""
    z, mask = pair
    mod = TriangleMulOutgoing(C_Z, num_heads=H, head_norm="head_rms").eval()
    torch.nn.init.normal_(mod.out_proj.weight, std=0.05)
    with torch.no_grad():
        act = mod._mhr_head_mix(torch.randn(1, N, N, C_Z) * torch.arange(1, H + 1).repeat_interleave(C_Z // H), None)
    rms = act.unflatten(-1, (H, C_Z // H)).pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, rms[..., :1].expand_as(rms), atol=1e-2)


def test_router_is_not_absorbed_by_the_output_gate(pair):
    """A per-head pre-projection scale is outside the class of a post-projection gate."""
    z, mask = pair
    mod = TriangleMulOutgoing(C_Z, **MHR).eval()
    torch.nn.init.normal_(mod.out_proj.weight, std=0.05)
    torch.nn.init.normal_(mod.out_gate.weight, std=0.05)
    flat = mod(z, mask)
    with torch.no_grad():
        mod.router.weight.normal_(std=1.0)
    routed = mod(z, mask)
    # Not equal, and not a per-output-channel rescale of the uniform-router output
    # (which is all the post-W_o gate could ever produce).
    assert not torch.allclose(flat, routed, atol=1e-4)
    ratio = (routed / flat.clamp(min=1e-3)).flatten(0, 2)
    assert ratio.std(0).max() > 1e-2


def test_router_stats_report_an_inert_router_as_inert(pair):
    z, mask = pair
    mod = TriangleMulOutgoing(C_Z, **MHR).eval()
    mod.router_stats_enabled = True
    mod(z, mask)
    mod(z, mask)  # two forwards: the read-out is a window mean, not the last one
    stats = mod._flush_router_stats()
    assert stats["entropy"] == pytest.approx(1.0, abs=1e-4)
    assert stats["occupancy_max"] == pytest.approx(1.0 / H, abs=1e-4)
    assert mod._flush_router_stats() is None, "flush should reset the window"


@pytest.mark.parametrize("block_cls", ["pairformer", "evoformer"])
def test_triangle_attention_can_be_deleted(block_cls, pair):
    z, mask = pair
    seq_mask = torch.ones(1, N)
    if block_cls == "pairformer":
        blk = PairFormerBlock(C_Z, 32, num_pair_heads=4, use_triangle_attention=False, trimul_kwargs=MHR).eval()
        out_pair, out_single = blk(z, torch.randn(1, N, 32), mask, seq_mask)
    else:
        blk = EvoformerBlock(16, C_Z, num_pair_heads=4, use_triangle_attention=False, trimul_kwargs=MHR).eval()
        out_single, out_pair = blk(torch.randn(1, 3, N, 16), z, torch.ones(1, 3, N), mask)
    assert not any("pair_attn" in n for n, _ in blk.named_parameters())
    assert out_pair.shape == z.shape and torch.isfinite(out_pair).all()
    assert torch.isfinite(out_single).all()


def test_config_flags_reach_the_modules():
    cfg = PairFormerConfig(pair_channel=C_Z, trimul_num_heads=H, trimul_head_norm="head_rms", trimul_router=True)
    assert _trimul_kwargs(cfg) == dict(num_heads=H, head_norm="head_rms", router=True, router_kind="softmax")
    assert EvoformerConfig().use_triangle_attention is True


def test_invalid_head_config_is_rejected():
    with pytest.raises(ValueError, match="divisible"):
        TriangleMulOutgoing(C_Z, num_heads=7, head_norm="head_rms")
    with pytest.raises(ValueError, match="num_heads > 0"):
        TriangleMulOutgoing(C_Z, router=True)
    with pytest.raises(ValueError, match="head_norm"):
        TriangleMulOutgoing(C_Z, num_heads=H, head_norm="groupnorm")


@pytest.mark.parametrize("head_norm", ["channel", "head_rms", "head_ln"])
def test_every_parameter_receives_gradient(head_norm, pair):
    """No orphan parameters in any mode — DDP errors on unused parameters.

    ``head_rms`` is the one at risk: RMSNorm has no shift, so the mode is built
    with ``norm_center`` bias-free rather than leaving the parameter unconsumed.
    """
    z, mask = pair
    mod = TriangleMulOutgoing(C_Z, num_heads=H, head_norm=head_norm, router=True)
    for m in (mod.out_proj, mod.out_gate, mod.gate):
        torch.nn.init.normal_(m.weight, std=0.05)
    mod(z, mask).sum().backward()
    orphans = [n for n, p in mod.named_parameters() if p.grad is None or not p.grad.any()]
    assert orphans == [], f"{head_norm}: parameters with no gradient: {orphans}"


def test_head_rms_has_no_shift():
    """RMSNorm means gain-only — the shift is absent, not merely unused."""
    mod = TriangleMulOutgoing(C_Z, num_heads=H, head_norm="head_rms", router=True)
    assert mod.norm_center.bias is None
    assert TriangleMulOutgoing(C_Z, num_heads=H, head_norm="head_ln").norm_center.bias is not None


def test_router_stats_separate_conditional_routing_from_constant_rescale(pair):
    """``g_std`` is the statistic that matters; entropy alone is not enough.

    A router that is sharp but identical for every pair scales each column
    block of ``W_o`` by a constant — a reparameterization of ``W_o``, inside
    AF3's function class. Entropy cannot tell that apart from real routing;
    the across-pair spread can.
    """
    z, mask = pair
    mod = TriangleMulOutgoing(C_Z, **MHR).eval()
    mod.router_stats_enabled = True

    # (a) inert: zero-init.
    mod(z, mask)
    inert = mod._flush_router_stats()
    assert inert["entropy"] == pytest.approx(1.0, abs=1e-4)
    assert inert["g_std"] == pytest.approx(0.0, abs=1e-4)

    # (b) sharp but pair-independent. A constant input is not enough — LN maps
    # it to zero and the router goes uniform again — so give ``norm_left`` a
    # bias, which makes LN(z) a fixed nonzero vector for every pair.
    with torch.no_grad():
        mod.router.weight.normal_(std=3.0)
        mod.norm_left.bias.normal_(std=1.0)
    mod(torch.ones_like(z), mask)
    sharp = mod._flush_router_stats()
    assert sharp["entropy"] < 0.95, "should look 'active' to entropy"
    assert sharp["g_std"] == pytest.approx(0.0, abs=1e-3), "but carries no pair information"

    # (c) genuinely conditional: same weights, pair-varying input.
    mod(z, mask)
    assert mod._flush_router_stats()["g_std"] > 1e-2


def test_router_stats_are_a_window_mean_not_the_last_forward(pair):
    """Recycling means several forwards per step; the read-out averages them."""
    z, mask = pair
    mod = TriangleMulOutgoing(C_Z, **MHR).eval()
    mod.router_stats_enabled = True
    with torch.no_grad():
        mod.router.weight.normal_(std=3.0)
        mod.norm_left.bias.normal_(std=1.0)
    mod(z, mask)
    varying = mod._flush_router_stats()["g_std"]
    mod(z, mask)
    mod(torch.ones_like(z), mask)  # pair-independent: contributes g_std ~ 0
    assert mod._flush_router_stats()["g_std"] == pytest.approx(varying / 2, rel=0.05)
    assert mod._flush_router_stats() is None, "flush resets the window"


def _wake(mod):
    """Give the zero-initialized output path non-zero weights.

    Without this every comparison of two TriangleMul variants is vacuous:
    ``out_proj`` and ``out_gate`` are zero-init, so both sides are exactly
    zero and ``torch.equal`` passes whatever the module did in between. That is
    how a missing ``head_norm == "channel"`` branch survived — "channel" fell
    through to per-head LayerNorm and the identity test still passed.
    """
    torch.manual_seed(0)
    for lin in (mod.out_proj, mod.out_gate, mod.gate):
        torch.nn.init.normal_(lin.weight, std=0.05)
    return mod


@pytest.mark.parametrize("kind", ["softmax", "sigmoid"])
def test_both_router_kinds_are_exactly_af3_at_init(kind, pair):
    z, mask = pair
    base = _wake(TriangleMulOutgoing(C_Z).eval())
    routed = _wake(TriangleMulOutgoing(C_Z, num_heads=H, head_norm="channel", router=True, router_kind=kind).eval())
    routed.load_state_dict(base.state_dict(), strict=False)
    out_base, out_routed = base(z, mask), routed(z, mask)
    assert out_base.abs().max() > 1e-3, "the comparison must not be zero vs zero"
    assert torch.allclose(out_base, out_routed, atol=1e-5)


def test_channel_mode_normalizes_over_c_z_not_per_head(pair):
    """head_norm='channel' must be AF3's LayerNorm(c_z), leaving head magnitudes free.

    Per-head normalization forces every head's RMS to 1; the whole point of the
    'channel' cell of the ablation grid is that it does not.
    """
    act = torch.randn(1, 5, 5, C_Z)
    logits = torch.zeros(1, 5, 5, H)
    ch = TriangleMulOutgoing(C_Z, num_heads=H, head_norm="channel", router=True).eval()
    hn = TriangleMulOutgoing(C_Z, num_heads=H, head_norm="head_ln", router=True).eval()
    with torch.no_grad():
        rms_ch = ch._mhr_head_mix(act, logits).unflatten(-1, (H, C_Z // H)).pow(2).mean(-1).sqrt()
        rms_hn = hn._mhr_head_mix(act, logits).unflatten(-1, (H, C_Z // H)).pow(2).mean(-1).sqrt()
    assert torch.allclose(rms_hn, torch.ones_like(rms_hn), atol=1e-3), "head_ln should equalize heads"
    assert rms_ch.std() > 0.05, f"channel must leave head magnitudes free, got std {rms_ch.std():.4f}"


def test_sigmoid_heads_are_independent_where_softmax_heads_compete():
    """The point of the sigmoid variant: raising one head must not lower others.

    Under softmax, sum_h g = H is a hard budget, so complementary relation
    subspaces cannot both be turned up for the same pair. That constraint — not
    the collapse it produces — is the architectural claim being tested. Driven
    through ``_mhr_head_mix`` with explicit logits, since routing the boost
    through ``W_r`` would land on a LayerNorm output whose channel mean is zero.
    """
    torch.manual_seed(0)
    act = torch.randn(1, 6, 6, C_Z)
    logits = torch.zeros(1, 6, 6, H)
    logits[..., 0] = 3.0

    def per_head_gain(kind):
        # Ratio of per-head RMS, not an elementwise ratio: under channel
        # normalization individual entries pass through zero, and dividing by
        # those is what makes an elementwise ratio unusable here.
        mod = TriangleMulOutgoing(C_Z, num_heads=H, head_norm="channel", router=True, router_kind=kind).eval()
        with torch.no_grad():
            flat = mod._mhr_head_mix(act, torch.zeros_like(logits)).unflatten(-1, (H, C_Z // H))
            routed = mod._mhr_head_mix(act, logits).unflatten(-1, (H, C_Z // H))
        rms = lambda x: x.pow(2).mean(-1).sqrt().mean((0, 1, 2))  # noqa: E731
        return rms(routed) / rms(flat).clamp_min(1e-8)

    soft, sig = per_head_gain("softmax"), per_head_gain("sigmoid")
    assert soft[0] > 1.0 and sig[0] > 1.0, "the boosted head must go up in both"
    # softmax pays for it out of the other heads; sigmoid does not.
    assert (soft[1:] < 0.99).all(), f"softmax should suppress the others: {soft.tolist()}"
    assert torch.allclose(sig[1:], torch.ones_like(sig[1:]), atol=1e-3), f"sigmoid must not: {sig.tolist()}"


def test_g_mean_is_pinned_for_softmax_and_free_for_sigmoid(pair):
    """``g_mean`` is the drift diagnostic the sigmoid variant needs."""
    z, mask = pair
    out = {}
    for kind in ("softmax", "sigmoid"):
        mod = TriangleMulOutgoing(C_Z, num_heads=H, head_norm="channel", router=True, router_kind=kind).eval()
        with torch.no_grad():
            mod.router.weight.normal_(std=2.0)
        mod.router_stats_enabled = True
        mod(z, mask)
        out[kind] = mod._flush_router_stats()["g_mean"]
    assert out["softmax"] == pytest.approx(1.0, abs=1e-4), "softmax cannot change the branch scale"
    assert abs(out["sigmoid"] - 1.0) > 1e-3, "sigmoid should be free to drift"


def test_invalid_router_kind_is_rejected():
    with pytest.raises(ValueError, match="router_kind"):
        TriangleMulOutgoing(C_Z, num_heads=H, router=True, router_kind="gumbel")
