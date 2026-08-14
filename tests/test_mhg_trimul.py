"""MHG-TriMul: head partition, per-head norm, and the target-pair router.

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
MHG = dict(num_heads=H, head_norm="head_rms", router=True)


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
    base = cls(C_Z).eval()
    routed = cls(C_Z, num_heads=H, head_norm="channel", router=True).eval()
    missing, unexpected = routed.load_state_dict(base.state_dict(), strict=False)
    assert unexpected == [] and set(missing) == {"router.weight"}
    assert torch.equal(base(z, mask), routed(z, mask))


def test_per_head_affine_shares_norm_center_keys(pair):
    """Every ablation cell has identical state_dict keys, so any warm-starts from any."""
    keys = {
        k
        for cell in ({}, dict(num_heads=H, head_norm="head_rms"), MHG)
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
        act = mod._mhg_head_mix(torch.randn(1, N, N, C_Z) * torch.arange(1, H + 1).repeat_interleave(C_Z // H), None)
    rms = act.unflatten(-1, (H, C_Z // H)).pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, rms[..., :1].expand_as(rms), atol=1e-2)


def test_router_is_not_absorbed_by_the_output_gate(pair):
    """A per-head pre-projection scale is outside the class of a post-projection gate."""
    z, mask = pair
    mod = TriangleMulOutgoing(C_Z, **MHG).eval()
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
    mod = TriangleMulOutgoing(C_Z, **MHG).eval()
    mod.router_stats_enabled = True
    mod(z, mask)
    assert mod._router_stats["entropy"] == pytest.approx(1.0, abs=1e-4)
    assert mod._router_stats["occupancy_max"] == pytest.approx(1.0 / H, abs=1e-4)


@pytest.mark.parametrize("block_cls", ["pairformer", "evoformer"])
def test_triangle_attention_can_be_deleted(block_cls, pair):
    z, mask = pair
    seq_mask = torch.ones(1, N)
    if block_cls == "pairformer":
        blk = PairFormerBlock(C_Z, 32, num_pair_heads=4, use_triangle_attention=False, trimul_kwargs=MHG).eval()
        out_pair, out_single = blk(z, torch.randn(1, N, 32), mask, seq_mask)
    else:
        blk = EvoformerBlock(16, C_Z, num_pair_heads=4, use_triangle_attention=False, trimul_kwargs=MHG).eval()
        out_single, out_pair = blk(torch.randn(1, 3, N, 16), z, torch.ones(1, 3, N), mask)
    assert not any("pair_attn" in n for n, _ in blk.named_parameters())
    assert out_pair.shape == z.shape and torch.isfinite(out_pair).all()
    assert torch.isfinite(out_single).all()


def test_config_flags_reach_the_modules():
    cfg = PairFormerConfig(pair_channel=C_Z, trimul_num_heads=H, trimul_head_norm="head_rms", trimul_router=True)
    assert _trimul_kwargs(cfg) == dict(num_heads=H, head_norm="head_rms", router=True)
    assert EvoformerConfig().use_triangle_attention is True


def test_invalid_head_config_is_rejected():
    with pytest.raises(ValueError, match="divisible"):
        TriangleMulOutgoing(C_Z, num_heads=7, head_norm="head_rms")
    with pytest.raises(ValueError, match="num_heads > 0"):
        TriangleMulOutgoing(C_Z, router=True)
    with pytest.raises(ValueError, match="head_norm"):
        TriangleMulOutgoing(C_Z, num_heads=H, head_norm="groupnorm")
