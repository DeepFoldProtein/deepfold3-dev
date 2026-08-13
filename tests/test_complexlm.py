"""Complex language model: store, sampling, masking, and the architecture.

The architectural tests are the point of this file. The claim behind the model
is that a complex is a *set* of chains — so the tests check the two invariances
that claim implies, rather than only checking that tensors have the right
shape:

* a chain-scoped layer must not let information cross a chain boundary;
* the whole model must be equivariant to chain order, and to which other
  complexes happen to share the packed batch.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from deepfold3.complexlm.config import ComplexLMConfig, MaskingConfig
from deepfold3.complexlm.crop import crop_complex
from deepfold3.complexlm.dataset import Example, pack
from deepfold3.complexlm.masking import apply_masking, select_positions
from deepfold3.complexlm.model import ComplexLM
from deepfold3.complexlm.store import ComplexRecord, ComplexStore, ComplexStoreWriter
from deepfold3.complexlm.vocab import MASK_ID, ascii_to_id_table, decode, standard_token_ids

TOKENS = ascii_to_id_table()


def _seq(text: str) -> np.ndarray:
    return TOKENS[np.frombuffer(text.encode(), dtype=np.uint8)]


def _example(
    chains: list[str],
    entities: list[int] | None = None,
    mask_at: list[int] | None = None,
    iface_known: bool = False,
) -> Example:
    tokens = [_seq(c).astype(np.int64) for c in chains]
    labels = [np.full(t.size, -100, dtype=np.int64) for t in tokens]
    if mask_at:
        for slot in mask_at:
            labels[slot] = tokens[slot].copy()
    return Example(
        tokens=tokens,
        labels=labels,
        iface=[np.zeros(t.size, dtype=np.uint8) for t in tokens],
        entities=np.array(entities if entities is not None else range(len(chains))),
        source=0,
        index=0,
        iface_known=iface_known,
    )


def _model(pattern: str = "cg", **kwargs) -> ComplexLM:
    torch.manual_seed(0)
    cfg = ComplexLMConfig(d_model=64, n_heads=4, n_layers=4, layer_pattern=pattern, **kwargs)
    return ComplexLM(cfg).eval()


# ---------------------------------------------------------------- store


def test_store_round_trip_and_entity_identity(tmp_path):
    writer = ComplexStoreWriter(tmp_path, "unit", with_iface=True)
    a, b = _seq("MKTIIALSYIF"), _seq("GGGSGGGS")
    writer.add(ComplexRecord("homo", [a, a], [np.ones(a.size, np.uint8), np.zeros(a.size, np.uint8)], date=20200101))
    writer.add(ComplexRecord("het", [a, b], [np.zeros(a.size, np.uint8), np.zeros(b.size, np.uint8)]))
    summary = writer.close()
    assert summary["n_unique_chains"] == 2  # the homodimer's two chains dedupe to one row

    store = ComplexStore(tmp_path)
    homo, het = store.example(0), store.example(1)
    assert homo.is_homomer and not het.is_homomer
    assert homo.chain_rows[0] == homo.chain_rows[1]
    assert decode(homo.seqs[0]) == "MKTIIALSYIF"
    assert homo.date == 20200101 and het.date == 0
    # Interface flags are per slot: the two copies of one sequence differ.
    assert homo.iface[0].sum() == a.size and homo.iface[1].sum() == 0


# ---------------------------------------------------------------- masking


def test_homomer_pairing_keeps_the_marginal_rate():
    """A shared draw, not a union: the rate must not grow with copy number."""
    rng = np.random.default_rng(0)
    seqs = [_seq("A" * 4000)] * 6
    entities = np.zeros(6, dtype=np.int64)
    cfg = MaskingConfig(rate=0.15, homomer_paired=True)
    selected = select_positions(rng, seqs, None, entities, cfg, interface_masking=False)
    rate = np.mean([s.mean() for s in selected])
    assert 0.13 < rate < 0.17, rate
    for s in selected[1:]:
        np.testing.assert_array_equal(s, selected[0])


def test_interface_bias_uses_both_rates():
    rng = np.random.default_rng(0)
    seq = _seq("A" * 20000)
    iface = np.zeros(seq.size, dtype=np.uint8)
    iface[: seq.size // 2] = 1
    cfg = MaskingConfig(rate=0.15, interface_rate=0.30, homomer_paired=False)
    (selected,) = select_positions(rng, [seq], [iface], np.array([0]), cfg, interface_masking=True)
    assert 0.28 < selected[iface == 1].mean() < 0.32
    assert 0.13 < selected[iface == 0].mean() < 0.17


def test_corruption_is_80_10_10():
    rng = np.random.default_rng(0)
    seq = _seq("ACDEFGHIKL" * 4000)
    cfg = MaskingConfig(rate=1.0, homomer_paired=False)
    (tokens,), (labels,) = apply_masking(rng, [seq], None, np.array([0]), cfg, interface_masking=False)
    np.testing.assert_array_equal(labels, seq.astype(np.int64))  # every position scored
    assert 0.79 < (tokens == MASK_ID).mean() < 0.81
    unchanged = (tokens == seq).mean()
    assert 0.09 < unchanged - 1 / len(standard_token_ids()) * 0.1 < 0.12


# ---------------------------------------------------------------- crop


def test_crop_respects_both_budgets():
    rng = np.random.default_rng(0)
    lengths = [3000, 1500, 400]
    spec = crop_complex(
        rng,
        lengths,
        np.array([0, 1, 2]),
        None,
        max_chain_len=1024,
        max_complex_len=2048,
        max_chains=16,
    )
    kept = [b - a for a, b in zip(spec.starts, spec.ends, strict=True)]
    assert max(kept) <= 1024 and sum(kept) <= 2048


def test_crop_keeps_copies_of_an_entity_aligned():
    """Homomer-paired masking is only meaningful if the windows agree."""
    rng = np.random.default_rng(0)
    spec = crop_complex(
        rng,
        [900, 900],
        np.array([7, 7]),
        [np.r_[np.ones(100, np.uint8), np.zeros(800, np.uint8)]] * 2,
        max_chain_len=256,
        max_complex_len=512,
        max_chains=16,
    )
    assert spec.starts[0] == spec.starts[1] and spec.ends[0] == spec.ends[1]


def test_crop_prefers_interface_residues():
    rng = np.random.default_rng(0)
    iface = np.zeros(2000, dtype=np.uint8)
    iface[1500:1520] = 1
    spec = crop_complex(rng, [2000], np.array([0]), [iface], max_chain_len=200, max_complex_len=200, max_chains=4)
    kept = iface[spec.starts[0] : spec.ends[0]]
    assert kept.sum() > 0


# ---------------------------------------------------------------- packing


def test_pack_builds_nested_cu_seqlens():
    batch = pack([_example(["AAAA", "CCC"]), _example(["DD"])])
    assert batch.n_tokens == 9
    assert batch.chain_cu.tolist() == [0, 4, 7, 9]
    assert batch.complex_cu.tolist() == [0, 7, 9]
    assert batch.positions.tolist() == [0, 1, 2, 3, 0, 1, 2, 0, 1]
    assert batch.global_positions.tolist() == [0, 1, 2, 3, 4, 5, 6, 0, 1]
    assert batch.entity_index.tolist() == [0, 0, 0, 0, 1, 1, 1, 2, 2]


# ---------------------------------------------------------------- model


@pytest.mark.parametrize("pattern", ["cg", "ccg", "r"])
def test_varlen_matches_dense(pattern):
    if not torch.cuda.is_available():
        pytest.skip("varlen attention needs CUDA")
    batch = pack([_example(["MKTIIALSYIFCLVFA", "GGSGGSGG"]), _example(["ACDEFGHIKLMNPQ"])])
    batch = batch.to("cuda")
    model = _model(pattern).cuda().to(torch.bfloat16)
    with torch.no_grad():
        model.cfg.attn_impl = "varlen"
        fast = model(batch).logits.float()
        model.cfg.attn_impl = "sdpa"
        ref = model(batch).logits.float()
    assert (fast - ref).abs().max() < 5e-2


def test_chain_layer_does_not_leak_across_chains():
    """A chain-scoped layer is block-diagonal: chain B cannot change chain A."""
    model = _model("c")
    left = pack([_example(["MKTIIALSYIF", "GGGGGGGG"])])
    right = pack([_example(["MKTIIALSYIF", "CCCCCCCC"])])
    with torch.no_grad():
        a = model(left).logits[:11]
        b = model(right).logits[:11]
    torch.testing.assert_close(a, b)


def test_global_layer_does_leak_across_chains():
    """The complement of the previous test: without it, ``g`` would be inert."""
    model = _model("g")
    left = pack([_example(["MKTIIALSYIF", "GGGGGGGG"])])
    right = pack([_example(["MKTIIALSYIF", "CCCCCCCC"])])
    with torch.no_grad():
        delta = (model(left).logits[:11] - model(right).logits[:11]).abs().max()
    assert delta > 1e-4


@pytest.mark.parametrize("pattern", ["cg", "ccg", "cx"])
def test_chain_permutation_equivariance(pattern):
    """f(π(A,B,C))[π(i)] == f(A,B,C)[i] — the architecture's central claim.

    Nothing enforces this except the absence of an absolute position across
    chains. The global-RoPE baseline ``r`` deliberately fails it, which is the
    point of keeping it as a baseline.
    """
    model = _model(pattern)
    chains = ["MKTIIALSYIF", "GGSGGSGG", "ACDEFGHIKLMNPQRS"]
    forward = pack([_example(chains)])
    reversed_ = pack([_example(chains[::-1])])
    with torch.no_grad():
        a = model(forward).logits
        b = model(reversed_).logits
    sizes = [len(c) for c in chains]
    offsets = np.cumsum([0, *sizes])
    rev_offsets = np.cumsum([0, *sizes[::-1]])
    for i in range(3):
        first = a[offsets[i] : offsets[i + 1]]
        j = 2 - i
        second = b[rev_offsets[j] : rev_offsets[j + 1]]
        torch.testing.assert_close(first, second, atol=1e-5, rtol=1e-4)


def test_global_rope_baseline_is_not_permutation_equivariant():
    model = _model("r")
    chains = ["MKTIIALSYIF", "GGSGGSGG"]
    with torch.no_grad():
        a = model(pack([_example(chains)])).logits[:11]
        b = model(pack([_example(chains[::-1])])).logits[8:]
    assert (a - b).abs().max() > 1e-3


@pytest.mark.parametrize("pattern", ["cg", "ccg"])
def test_packing_does_not_couple_complexes(pattern):
    """Which complexes share a batch must not change any of their outputs."""
    model = _model(pattern)
    alone = pack([_example(["MKTIIALSYIF", "GGSGGSGG"])])
    together = pack([_example(["MKTIIALSYIF", "GGSGGSGG"]), _example(["ACDEFGHIKLMNPQRS"])])
    with torch.no_grad():
        a = model(alone).logits
        b = model(together).logits[: alone.n_tokens]
    torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-4)


def test_layer_pattern_selects_kinds():
    model = _model("ccg")
    assert [b.kind for b in model.blocks] == ["c", "c", "g", "c"]


@pytest.mark.parametrize("pattern,collapses", [("cg", True), ("ccg", True), ("cx", False), ("r", False)])
def test_exact_homomer_collapses_to_monomer(pattern, collapses):
    """Under global NoPE an exact homomer *is* its monomer, at any copy number.

    Softmax over a key set duplicated ``k`` times equals softmax over one copy
    — each key's weight is divided by ``k`` and counted ``k`` times — so if the
    copies carry identical hidden states (identical sequence, per-chain
    positions restarting at 0, and, with ``homomer_paired`` masking, identical
    corruption) the global layer returns exactly the monomer's output. The
    model then cannot tell a monomer from a dimer from a trimer.

    This is not a bug to fix silently: it says an exact homomer teaches the
    global layers nothing, which is a fact about the mixture (AFDB-multimer is
    ~96% homodimer) and the motivation for ``chain_signal``.
    """
    model = _model(pattern)
    seq = "MKTIIALSYIFCLVFA"
    with torch.no_grad():
        mono = model(pack([_example([seq], [0])])).logits
        dimer = model(pack([_example([seq, seq], [0, 0])])).logits[: len(seq)]
        trimer = model(pack([_example([seq, seq, seq], [0, 0, 0])])).logits[: len(seq)]
    if collapses:
        torch.testing.assert_close(dimer, mono, atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(trimer, mono, atol=1e-5, rtol=1e-4)
    else:
        assert (dimer - mono).abs().max() > 1e-3
    # Copy number is invisible whenever cross-chain attention has no positions:
    # 'x' too, which excludes self but still sees k identical copies.
    if pattern in ("cg", "ccg", "cx"):
        torch.testing.assert_close(trimer, dimer, atol=1e-5, rtol=1e-4)
    else:
        assert (trimer - dimer).abs().max() > 1e-3


def test_chain_signal_breaks_the_homomer_collapse():
    model = _model("cg", chain_signal="random")
    with torch.no_grad():
        model.chain_gate.fill_(1.0)
        torch.manual_seed(0)
        seq = "MKTIIALSYIFCLVFA"
        mono = model(pack([_example([seq], [0])])).logits
        dimer = model(pack([_example([seq, seq], [0, 0])])).logits[: len(seq)]
    assert (dimer - mono).abs().max() > 1e-3


def test_homomer_paired_masking_shares_the_corruption_draw():
    """Sharing the selection but not the 80/10/10 leaks the answer."""
    rng = np.random.default_rng(0)
    seq = _seq("ACDEFGHIKL" * 100)
    tokens, labels = apply_masking(rng, [seq, seq], None, np.array([0, 0]), MaskingConfig(), False)
    np.testing.assert_array_equal(tokens[0], tokens[1])
    np.testing.assert_array_equal(labels[0], labels[1])


def test_single_chain_view_isolates_chains():
    from deepfold3.complexlm.diagnostics import single_chain_view

    model = _model("cg")
    batch = pack([_example(["MKTIIALSYIF", "ACDEFGHIKLMNPQRS"])])
    with torch.no_grad():
        joint = model(batch).logits[:11]
        alone = model(single_chain_view(batch)).logits[:11]
        solo = model(pack([_example(["MKTIIALSYIF"])])).logits
    torch.testing.assert_close(alone, solo, atol=1e-5, rtol=1e-4)
    assert (joint - solo).abs().max() > 1e-4


def test_first_complex_slice_keeps_every_chain():
    from deepfold3.complexlm.trainer import _first_complex

    batch = pack([_example(["AAAA", "CCC"]), _example(["DD"])])
    sliced = _first_complex(batch)
    assert sliced.n_tokens == 7
    assert sliced.chain_cu.tolist() == [0, 4, 7]
    assert sliced.complex_cu.tolist() == [0, 7]


def test_permutation_keeps_complex_positions_contiguous():
    from deepfold3.complexlm.diagnostics import permute_chain_order

    batch = pack([_example(["AAAA", "CCC", "DD"]), _example(["EEEEE"])])
    permuted, inverse = permute_chain_order(batch, torch.Generator().manual_seed(3))
    assert permuted.global_positions[:9].tolist() == list(range(9))
    assert permuted.global_positions[9:].tolist() == [0, 1, 2, 3, 4]
    torch.testing.assert_close(permuted.tokens[inverse], batch.tokens)


# ---------------------------------------------------------------- muP


def _mup_pair(width: int, base: int = 256):
    torch.manual_seed(0)
    kw = dict(n_heads=8, n_layers=4, attn_impl="sdpa")
    plain = ComplexLM(ComplexLMConfig(d_model=width, **kw))
    torch.manual_seed(0)
    mup = ComplexLM(ComplexLMConfig(d_model=width, mup=True, mup_base_width=base, **kw))
    return plain, mup


def test_mup_is_identity_at_the_base_width():
    """At d_model == mup_base_width every muP factor is 1, so the two agree."""
    plain, mup = _mup_pair(256)
    for (name, a), (_, b) in zip(plain.named_parameters(), mup.named_parameters(), strict=True):
        torch.testing.assert_close(a, b, msg=lambda m, name=name: f"{name}: {m}")
    assert mup.readout_mult == 1.0
    assert mup.blocks[0].attn.scale == plain.blocks[0].attn.scale
    assert all(g["lr_scale"] == 1.0 for g in mup.param_groups(0.01))


def test_mup_scales_only_the_matrix_like_parameters():
    """Only tensors whose *both* dimensions grow with width are rescaled.

    That is ``infshape.ninf() == 2`` in microsoft/mup: ``mup.init`` scales
    those by ``width_mult ** -0.5`` and ``MuAdam`` divides their learning rate
    by ``width_mult`` (multiplying weight decay to match, since AdamW's decay
    term carries the learning rate). ``[vocab, d_model]`` has one width
    dimension, not two, so the embedding *and the readout* keep the base
    learning rate; the readout's width dependence is carried entirely by its
    ``1/width_mult`` output multiplier.
    """
    _, mup = _mup_pair(1024, base=256)
    m = 4.0
    assert mup.readout_mult == 1 / m
    # 1/d_head rather than 1/sqrt(d_head), pinned to the proxy at the base width
    assert mup.blocks[0].attn.scale == pytest.approx((256 / 8) ** 0.5 / (1024 / 8))
    groups = {g["role"]: g for g in mup.param_groups(0.01)}
    assert groups["hidden"]["lr_scale"] == pytest.approx(1 / m)
    assert groups["hidden"]["weight_decay"] == pytest.approx(0.01 * m)
    for role in ("embedding", "readout", "vector"):
        assert groups[role]["lr_scale"] == 1.0, role
    assert groups["readout"]["weight_decay"] == pytest.approx(0.01)


def test_mup_init_shrinks_hidden_matrices_only():
    narrow_plain, narrow = _mup_pair(256)
    wide_plain, wide = _mup_pair(1024, base=256)
    assert wide.blocks[0].attn.qkv.weight.std().item() == pytest.approx(
        wide_plain.blocks[0].attn.qkv.weight.std().item() / 2, rel=0.05
    )
    for name in ("embed.weight", "head.out.weight"):
        wide_std = dict(wide.named_parameters())[name].std().item()
        narrow_std = dict(narrow.named_parameters())[name].std().item()
        assert wide_std == pytest.approx(narrow_std, rel=0.05), name


def test_mup_roles_cover_every_parameter():
    from deepfold3.complexlm.model import _mup_role

    model = ComplexLM(ComplexLMConfig(d_model=64, n_heads=4, n_layers=2, bias=True))
    roles = {name: _mup_role(name, p) for name, p in model.named_parameters()}
    assert roles["embed.weight"] == "embedding"
    assert roles["head.out.weight"] == "readout"
    assert roles["blocks.0.attn.qkv.weight"] == "hidden"
    assert roles["head.norm.weight"] == "vector"
    assert all(v == "vector" for k, v in roles.items() if k.endswith(".bias"))
    assert sum(p.numel() for _, p in model.named_parameters()) == model.n_params


def test_interface_groups_exclude_unlabelled_sources():
    """A sequence-only source must not be counted as all-non-interface."""
    from deepfold3.complexlm.diagnostics import partner_utilisation

    model = _model("cg")
    labelled = _example(["MKTIIALSYIF", "GGSGGSGG"], mask_at=[0, 1], iface_known=True)
    unlabelled = _example(["ACDEFGHIKLMNPQ", "WWWWWWWW"], mask_at=[0, 1], iface_known=False)
    batch = pack([labelled, unlabelled])
    result = partner_utilisation(model, batch, ["structural", "sequence"]).result(("loss_full",))
    assert result["unlabelled/n"] == sum(len(c) for c in ("ACDEFGHIKLMNPQ", "WWWWWWWW"))
    assert result["non_interface/n"] == sum(len(c) for c in ("MKTIIALSYIF", "GGSGGSGG"))
    assert "interface/n" not in result  # nothing is flagged in this fixture
