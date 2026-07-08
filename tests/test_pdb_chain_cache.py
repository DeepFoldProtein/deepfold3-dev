"""Tests for the per-chain template cache and the crop-then-load helpers."""

from __future__ import annotations

import numpy as np

from deepfold3.data import const
from deepfold3.data.crop.cropper import compute_chain_msa_cols
from deepfold3.data.parse.pdb_chain_cache import (
    _NEG,
    ChainTemplate,
    PdbChainCache,
    _atomic_write_npz,
    _deserialize,
    _serialize,
    chain_cache_path,
    pdb_done_path,
    template_features_from_chain,
)

A = const.MAX_ATOMS_PER_TOKEN


def _make_chain(L: int = 5, seqres: str | None = "ACDEFG") -> ChainTemplate:
    rng = np.random.default_rng(0)
    lookup = {}
    for i in range(L):
        aa = int(rng.integers(0, 20))
        pos = rng.standard_normal((A, 3)).astype(np.float32)
        msk = rng.integers(0, 2, size=A).astype(bool)
        lookup[i] = (aa, pos, msk)
    return ChainTemplate(seqres=seqres, struct_seq="X" * L, lookup=lookup)


def test_serialize_roundtrip():
    ct = _make_chain(L=7)
    rt = _deserialize(_serialize(ct))
    assert rt.seqres == ct.seqres
    assert rt.struct_seq == ct.struct_seq
    assert len(rt.lookup) == len(ct.lookup)
    for i in range(len(ct.lookup)):
        a0, p0, m0 = ct.lookup[i]
        a1, p1, m1 = rt.lookup[i]
        assert a0 == a1
        assert np.array_equal(p0, p1)
        assert np.array_equal(m0, m1)


def test_serialize_roundtrip_no_seqres():
    ct = _make_chain(L=3, seqres=None)
    rt = _deserialize(_serialize(ct))
    assert rt.seqres is None
    assert len(rt.lookup) == 3


def test_chain_cache_path_sharded(tmp_path):
    p = chain_cache_path(tmp_path, "1ABC", "A")
    # shard(pdb) = first two chars of the lowercased id
    assert p == tmp_path / "1a" / "1abc" / "A.npz"


def test_disk_hit_and_known_miss(tmp_path):
    ct = _make_chain(L=4)
    # Write one chain + the per-PDB sentinel by hand (no gemmi needed).
    _atomic_write_npz(chain_cache_path(tmp_path, "9xyz", "A"), **_serialize(ct))
    pdb_done_path(tmp_path, "9xyz").parent.mkdir(parents=True, exist_ok=True)
    pdb_done_path(tmp_path, "9xyz").touch()

    cache = PdbChainCache(cache_dir=tmp_path, mmcif_dir=None)
    got = cache.get("9xyz", "A")
    assert got is not None
    assert got.struct_seq == ct.struct_seq
    # Second lookup served from the memory LRU (still correct).
    assert cache.get("9xyz", "A") is got
    # A chain absent from a parsed PDB (sentinel present) is a known miss.
    assert cache.get("9xyz", "B") is None
    # ... and negative-cached in memory.
    assert cache._mem[("9xyz", "B")] is _NEG


def test_unknown_pdb_without_mmcif_returns_none(tmp_path):
    # No sentinel, no mmcif mirror → unknown → None (cannot parse).
    cache = PdbChainCache(cache_dir=tmp_path, mmcif_dir=None)
    assert cache.get("0abc", "A") is None


def test_mem_lru_eviction(tmp_path):
    cache = PdbChainCache(cache_dir=None, mmcif_dir=None, mem_capacity=2)
    cache._mem_put(("p", "1"), _make_chain())
    cache._mem_put(("p", "2"), _make_chain())
    cache._mem_put(("p", "3"), _make_chain())  # evicts ("p","1")
    assert ("p", "1") not in cache._mem
    assert ("p", "2") in cache._mem and ("p", "3") in cache._mem


def test_template_features_from_chain_shapes():
    ct = _make_chain(L=6)
    # query positions 0,1,2 → seqres positions 0,1,2 (len match → direct map)
    q2s = {0: 0, 1: 1, 2: 2}
    feats = template_features_from_chain(ct, q2s, query_length=10, max_atoms=A)
    assert feats.aatype.shape == (10,)
    assert feats.atom_positions.shape == (10, A, 3)
    assert feats.atom_mask.shape == (10, A)
    # mapped positions carry the chain's aatype; uncovered positions read as GAP
    # (AF3 §2.8 — a template that doesn't cover a query position is a gap, not UNK).
    for qi in (0, 1, 2):
        assert feats.aatype[qi] == ct.lookup[qi][0]
    assert feats.aatype[5] == const.MSA_GAP


def test_compute_chain_msa_cols_basic():
    asym = np.array([0, 0, 0, 1, 1, 2])
    # chain 0: cols 0,1,2 ; chain 1: cols 0,1 ; chain 2: ligand (-1) → omitted
    msa_col_idx = np.array([0, 1, 2, 0, 1, -1])
    out = compute_chain_msa_cols(asym, msa_col_idx)
    assert set(out) == {0, 1}
    assert np.array_equal(out[0], np.array([0, 1, 2]))
    assert np.array_equal(out[1], np.array([0, 1]))


def test_compute_chain_msa_cols_dedup_order():
    # Expanded non-standard residues share a column → dedup, order-stable.
    asym = np.array([0, 0, 0, 0])
    msa_col_idx = np.array([2, 2, 0, 1])
    out = compute_chain_msa_cols(asym, msa_col_idx)
    assert np.array_equal(out[0], np.array([2, 0, 1]))


def test_compute_chain_msa_cols_none():
    assert compute_chain_msa_cols(np.array([0, 1]), None) == {}
