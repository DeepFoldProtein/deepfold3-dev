"""``format: of3pdb`` — columnar dataset cache, AF3 §2.5.1 weights, converter.

Covers the three places where reading OpenFold 3's ``pdb_training_set`` in place
can go silently wrong:

* the streaming JSON reader must reproduce ``json.load`` exactly, including at
  buffer boundaries (a truncated read would drop structures, not raise);
* the cache's vectorised sampling weights must equal what
  ``data/sample/weighted.py`` computes from the equivalent ``Record`` list —
  the two paths feed the same sampler and must agree;
* the structure converter must take OF3's ``is_atomized`` verbatim and turn
  unresolved NaN coordinates into DeepFold3's zero + ``is_present=False``.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from deepfold3.data import const
from deepfold3.data.prepare.of3_pdb_cache import (
    OF3PdbCache,
    build_of3_pdb_cache,
    iter_structure_entries,
)

# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


def _cache_json() -> dict:
    """A ``ClusteredDatasetCache`` covering every shape the reader must handle.

    ``4xyz`` is a protein homodimer + shared-cluster ligand, ``2abc`` a
    protein/RNA heteromer with a missing resolution, ``9zzz`` a bare monomer.
    """
    return {
        "_type": "ClusteredDatasetCache",
        "name": "PDB-weighted",
        "structure_data": {
            "4xyz": {
                "release_date": "2014-10-29",
                "resolution": 2.8,
                "chains": {
                    "1": {
                        "label_asym_id": "A",
                        "auth_asym_id": "A",
                        "entity_id": 1,
                        "molecule_type": "PROTEIN",
                        "reference_mol_id": None,
                        "alignment_representative_id": "5enn_A",
                        "template_ids": ["3ls8_A", "3ls8_B"],
                        "cluster_id": "32032",
                        "cluster_size": 2,
                    },
                    "2": {
                        "label_asym_id": "B",
                        "auth_asym_id": "B",
                        "entity_id": 1,
                        "molecule_type": "PROTEIN",
                        "reference_mol_id": None,
                        "alignment_representative_id": "5enn_A",
                        "template_ids": ["3ls8_A"],
                        "cluster_id": "32032",
                        "cluster_size": 2,
                    },
                    "3": {
                        "label_asym_id": "C",
                        "auth_asym_id": "A",
                        "entity_id": 2,
                        "molecule_type": "LIGAND",
                        "reference_mol_id": "HEM",
                        "alignment_representative_id": None,
                        "template_ids": None,
                        "cluster_id": "43997",
                        "cluster_size": 2,
                    },
                },
                "interfaces": {
                    "1_2": {"cluster_id": "32032_32032", "cluster_size": 1},
                    "1_3": {"cluster_id": "32032_43997", "cluster_size": 2},
                },
            },
            "2abc": {
                "release_date": "2020-01-02",
                "resolution": float("nan"),
                "chains": {
                    "1": {
                        "label_asym_id": "A",
                        "auth_asym_id": "A",
                        "entity_id": 1,
                        "molecule_type": "PROTEIN",
                        "reference_mol_id": None,
                        "alignment_representative_id": "2abc_A",
                        "template_ids": [],
                        "cluster_id": "10",
                        "cluster_size": 1,
                    },
                    "7": {
                        "label_asym_id": "G",
                        "auth_asym_id": "B",
                        "entity_id": 2,
                        "molecule_type": "RNA",
                        "reference_mol_id": None,
                        "alignment_representative_id": "2abc_G",
                        "template_ids": None,
                        "cluster_id": "11",
                        "cluster_size": 1,
                    },
                },
                "interfaces": {"1_7": {"cluster_id": "10_11", "cluster_size": 1}},
            },
            "9zzz": {
                "release_date": None,
                "resolution": 1.4,
                "chains": {
                    "1": {
                        "label_asym_id": "A",
                        "auth_asym_id": "A",
                        "entity_id": 1,
                        "molecule_type": "PROTEIN",
                        "reference_mol_id": None,
                        "alignment_representative_id": "43997_A",
                        "template_ids": None,
                        "cluster_id": "43997",
                        "cluster_size": 2,
                    }
                },
                "interfaces": {},
            },
        },
        "reference_molecule_data": {},
    }


@pytest.fixture
def cache_file(tmp_path):
    path = tmp_path / "training_cache_with_templates.json"
    path.write_text(json.dumps(_cache_json(), indent=4))
    return path


@pytest.fixture
def cache(tmp_path, cache_file):
    out = tmp_path / "df3_cache"
    build_of3_pdb_cache(cache_file, out)
    return OF3PdbCache(out)


# --------------------------------------------------------------------------
# Streaming JSON reader
# --------------------------------------------------------------------------


def test_streaming_reader_matches_json_load(cache_file):
    ref = json.loads(cache_file.read_text())["structure_data"]
    streamed = dict(iter_structure_entries(cache_file))
    assert set(streamed) == set(ref)
    for pdb_id, entry in ref.items():
        # NaN != NaN, so compare the JSON encodings rather than the dicts.
        assert json.dumps(streamed[pdb_id], sort_keys=True) == json.dumps(entry, sort_keys=True)


@pytest.mark.parametrize("chunk_size", [1, 3, 17, 512])
def test_streaming_reader_survives_chunk_boundaries(cache_file, monkeypatch, chunk_size):
    """A tiny read size splits every token; the reader must still be exact.

    This is the failure that would not raise on its own — a reader that gives up
    at a boundary just yields fewer structures.
    """
    import deepfold3.data.prepare.of3_pdb_cache as mod

    original = mod._JsonCursor.__init__
    monkeypatch.setattr(
        mod._JsonCursor,
        "__init__",
        lambda self, fh, chunk_size=chunk_size, _o=original: _o(self, fh, chunk_size),
    )
    assert len(dict(iter_structure_entries(cache_file))) == 3


def test_streaming_reader_rejects_missing_key(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"_type": "ClusteredDatasetCache", "name": "x"}')
    with pytest.raises(KeyError):
        list(iter_structure_entries(path))


# --------------------------------------------------------------------------
# Columnar cache
# --------------------------------------------------------------------------


def test_cache_roundtrip(cache):
    assert len(cache) == 3
    assert cache.has_templates

    i = cache.index_of("4xyz")
    assert i is not None
    assert cache.pdb_id(i) == "4xyz"
    assert cache.release_date(i) == "2014-10-29"
    assert cache.resolution(i) == pytest.approx(2.8)

    chains = cache.chains(i)
    assert [c.key for c in chains] == ["1", "2", "3"]
    assert [c.label_asym_id for c in chains] == ["A", "B", "C"]
    assert [c.mol_type for c in chains] == [const.MOL_PROTEIN, const.MOL_PROTEIN, const.MOL_NONPOLYMER]
    assert [c.aln_rep_id for c in chains] == ["5enn_A", "5enn_A", None]
    # Same cluster_id string → same interned code; different id → different code.
    assert chains[0].cluster_code == chains[1].cluster_code != chains[2].cluster_code

    lo, _ = cache.chain_slice(i)
    assert cache.template_ids(lo + 0) == ["3ls8_A", "3ls8_B"]
    assert cache.template_ids(lo + 1) == ["3ls8_A"]
    assert cache.template_ids(lo + 2) == []

    a, b = cache.iface_slice(i)
    assert [(int(cache.iface_a[j]), int(cache.iface_b[j])) for j in range(a, b)] == [(0, 1), (0, 2)]


def test_cache_missing_metadata(cache):
    """``null`` release date and ``NaN`` resolution come back as ``None``."""
    assert cache.resolution(cache.index_of("2abc")) is None
    assert cache.release_date(cache.index_of("9zzz")) is None
    assert cache.release_dates()[cache.index_of("9zzz")] == b""


def test_cluster_codes_are_global(cache):
    """A cluster shared across structures interns to one code.

    ``9zzz``'s only chain and ``4xyz``'s ligand are both cluster ``43997``; if
    the code were per-structure the AF3 cluster balance would silently double
    that cluster's draw mass.
    """
    ligand = cache.chains(cache.index_of("4xyz"))[2]
    monomer = cache.chains(cache.index_of("9zzz"))[0]
    assert ligand.cluster_code == monomer.cluster_code


def test_column_width_overflow_raises(tmp_path):
    data = _cache_json()
    data["structure_data"]["4xyz"]["chains"]["1"]["alignment_representative_id"] = "x" * 40
    path = tmp_path / "wide.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="chain_aln_rep"):
        build_of3_pdb_cache(path, tmp_path / "out")


def test_version_mismatch_raises(tmp_path, cache_file):
    out = tmp_path / "df3_cache"
    build_of3_pdb_cache(cache_file, out)
    meta = json.loads((out / "meta.json").read_text())
    meta["version"] += 1
    (out / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="version"):
        OF3PdbCache(out)


# --------------------------------------------------------------------------
# AF3 §2.5.1 weights: cache path vs Record path
# --------------------------------------------------------------------------


def _records_from_cache(cache) -> list:
    """The same metadata as ``Record`` objects, for the reference weight path."""
    from deepfold3.data.types import ChainInfo, InterfaceInfo, Record

    records = []
    for i in range(len(cache)):
        chains = [
            ChainInfo(
                chain_id=k,
                chain_name=c.label_asym_id,
                mol_type=c.mol_type,
                cluster_id=c.cluster_code,
                msa_id=c.aln_rep_id or "",
                num_residues=1,
                entity_id=c.entity_id,
            )
            for k, c in enumerate(cache.chains(i))
        ]
        lo, hi = cache.iface_slice(i)
        interfaces = [
            InterfaceInfo(chain_1=int(cache.iface_a[j]), chain_2=int(cache.iface_b[j])) for j in range(lo, hi)
        ]
        records.append(Record(id=cache.pdb_id(i), chains=chains, interfaces=interfaces))
    return records


def test_weights_match_record_path(cache):
    """Vectorised cache weights == ``compute_cluster_weights`` on equal Records.

    The two are alternative implementations of the same AF3 §2.5.1 marginal and
    feed the same ``MultiDatasetSampler``; a divergence would skew the training
    distribution with nothing to catch it.
    """
    from deepfold3.data.module.of3_pdb_dataset import of3_anchor_weights
    from deepfold3.data.sample.weighted import build_record_anchor_tables, compute_cluster_weights

    w_chain, w_iface, w_struct = of3_anchor_weights(cache)
    reference = compute_cluster_weights(_records_from_cache(cache))
    np.testing.assert_allclose(w_struct, reference, rtol=1e-12, atol=1e-12)

    # ...and the per-record anchor conditional agrees too, anchor for anchor.
    tables = build_record_anchor_tables(_records_from_cache(cache))
    for i in range(len(cache)):
        c_lo, c_hi = cache.chain_slice(i)
        i_lo, i_hi = cache.iface_slice(i)
        mine = np.concatenate([w_chain[c_lo:c_hi], w_iface[i_lo:i_hi]])
        mine = mine[mine > 0]
        theirs = np.asarray(tables[i][1])
        np.testing.assert_allclose(np.sort(mine), np.sort(theirs), rtol=1e-12, atol=1e-12)


def test_weights_respect_keep_mask(cache):
    """Excluded structures get zero weight and leave the cluster counts.

    A cutoff filter shrinks clusters, and AF3 balances against the *surviving*
    occupancy — reading upstream's frozen ``cluster_size`` instead would keep
    down-weighting chains whose cluster siblings are no longer sampled.
    """
    from deepfold3.data.module.of3_pdb_dataset import of3_anchor_weights

    keep = np.ones(len(cache), dtype=bool)
    keep[cache.index_of("9zzz")] = False
    w_chain, _, w_struct = of3_anchor_weights(cache, keep=keep)

    assert w_struct[cache.index_of("9zzz")] == 0.0
    assert w_struct.sum() == pytest.approx(1.0)

    # 4xyz's ligand shared cluster 43997 with 9zzz's chain (N_clust 2 -> 1),
    # so dropping 9zzz doubles the ligand anchor's weight.
    lo, _ = cache.chain_slice(cache.index_of("4xyz"))
    full_chain, _, _ = of3_anchor_weights(cache)
    assert w_chain[lo + 2] == pytest.approx(2.0 * full_chain[lo + 2])


# --------------------------------------------------------------------------
# Structure converter
# --------------------------------------------------------------------------


def _write_atom_array(path, **overrides):
    """A two-chain OF3 structure NPZ: protein with a modified residue + ligand."""
    res = [
        ("ALA", "1", 1, 0, ["N", "CA", "C", "O", "CB"], True),
        ("SEP", "1", 2, 0, ["N", "CA", "C", "O", "CB", "OG", "P"], False),
        ("GLY", "1", 3, 0, ["N", "CA", "C", "O"], True),
        ("HEM", "3", 1, 3, ["FE", "NA", "NB"], False),
    ]
    coord, chain_id, res_id, res_name, atom_name, element = [], [], [], [], [], []
    mol, atomized, occ = [], [], []
    for k, (name, cid, rid, mt, atoms, standard) in enumerate(res):
        for j, a in enumerate(atoms):
            # Residue 3 (GLY) is entirely unresolved: NaN coordinates, zero occupancy.
            unresolved = name == "GLY"
            coord.append([np.nan] * 3 if unresolved else [float(k), float(j), 0.0])
            occ.append(0.0 if unresolved else 1.0)
            chain_id.append(cid)
            res_id.append(rid)
            res_name.append(name)
            atom_name.append(a)
            element.append(a[0])
            mol.append(mt)
            atomized.append(not standard)
    data = {
        "coord": np.array(coord, dtype=np.float32),
        "chain_id": np.array(chain_id, dtype="<U21"),
        "res_id": np.array(res_id, dtype=np.int64),
        "ins_code": np.array([""] * len(coord), dtype="<U1"),
        "res_name": np.array(res_name, dtype="<U5"),
        "atom_name": np.array(atom_name, dtype="<U6"),
        "element": np.array(element, dtype="<U2"),
        "occupancy": np.array(occ, dtype=np.float64),
        "charge": np.zeros(len(coord), dtype=np.int64),
        "molecule_type_id": np.array(mol, dtype=np.int64),
        "is_atomized": np.array(atomized, dtype=bool),
        "entity_id": np.array([1 if c == "1" else 2 for c in chain_id], dtype=np.int64),
        "label_asym_id": np.array(["A" if c == "1" else "C" for c in chain_id], dtype="<U4"),
        "auth_asym_id": np.array(["A"] * len(coord), dtype="<U4"),
        "bonds": np.array([[0, 1, 1]], dtype=np.uint32),
    }
    data.update(overrides)
    np.savez(path, **data)
    return path


def test_converter_uses_of3_is_atomized(tmp_path):
    """``is_standard`` comes from OF3's own decision, not the residue name.

    SEP maps to SER in ``RES_TYPE_ORDER``, so a name-derived rule would call it
    standard and emit one SER token over SEP's atom set — which the featurizer
    then slots against SER's reference atoms.
    """
    from deepfold3.data.prepare.openfold3 import convert_of3_atom_array

    structure, _, chain_keys = convert_of3_atom_array(_write_atom_array(tmp_path / "s.npz"), pdb_id="test")
    names = [r["name"].decode() for r in structure.residues]
    standard = {n: bool(r["is_standard"]) for n, r in zip(names, structure.residues, strict=True)}

    assert standard == {"ALA": True, "SEP": False, "GLY": True, "HEM": False}
    assert chain_keys == ["1", "3"]


def test_converter_zeroes_unresolved_coordinates(tmp_path):
    """OF3 stores unresolved atoms as NaN; DeepFold3's convention is 0 + not-present.

    A NaN reaching the crop poisons every distance it takes part in, and the
    resulting loss is NaN with no indication of which atom caused it.
    """
    from deepfold3.data.prepare.openfold3 import convert_of3_atom_array

    structure, _, _ = convert_of3_atom_array(_write_atom_array(tmp_path / "s.npz"), pdb_id="test")
    xyz = np.stack([structure.atoms["x"], structure.atoms["y"], structure.atoms["z"]], axis=1)
    assert np.isfinite(xyz).all()

    gly = next(r for r in structure.residues if r["name"] == b"GLY")
    lo, n = int(gly["atom_idx"]), int(gly["atom_num"])
    assert not structure.atoms["is_present"][lo : lo + n].any()
    assert (xyz[lo : lo + n] == 0.0).all()
    assert not bool(gly["is_present"])
    # Every other residue stays present.
    assert structure.atoms["is_present"].sum() == len(structure.atoms) - n


def test_converter_chain_and_entity_layout(tmp_path):
    from deepfold3.data.prepare.openfold3 import convert_of3_atom_array

    structure, record, _ = convert_of3_atom_array(_write_atom_array(tmp_path / "s.npz"), pdb_id="test")
    assert [c["name"].decode() for c in structure.chains] == ["A", "C"]
    assert [int(c["mol_type"]) for c in structure.chains] == [const.MOL_PROTEIN, const.MOL_NONPOLYMER]
    assert [int(c["entity_id"]) for c in structure.chains] == [1, 2]
    assert [int(c["sym_id"]) for c in structure.chains] == [0, 0]
    # Chain atom/residue spans must partition the arrays for Structure to be valid.
    assert int(structure.chains["atom_num"].sum()) == len(structure.atoms)
    assert int(structure.chains["res_num"].sum()) == len(structure.residues)
    assert record.chains[1].res_name == "HEM"


def test_converter_regroups_interleaved_chains(tmp_path):
    """Interleaved chain atoms are regrouped, and bonds follow the atoms.

    ``Structure`` addresses a chain as a single atom slice, so a file whose
    chains are not contiguous would otherwise produce chains that silently claim
    each other's atoms — and bond indices that point at the wrong ones.
    """
    from deepfold3.data.prepare.openfold3 import convert_of3_atom_array

    path = _write_atom_array(tmp_path / "s.npz")
    data = dict(np.load(path).items())
    n_atoms = len(data["coord"])

    # Move the ligand chain's atoms to the front, keeping relative order within
    # each chain (an interleaving the converter's stable regroup must undo).
    is_ligand = data["chain_id"] == "3"
    order = np.argsort(~is_ligand, kind="stable")
    bonded_names = tuple(data["atom_name"][data["bonds"][:, :2]].tolist()[0])

    shuffled = {k: (v[order] if getattr(v, "shape", (0,))[:1] == (n_atoms,) else v) for k, v in data.items()}
    inverse = np.empty(n_atoms, dtype=np.int64)
    inverse[order] = np.arange(n_atoms)
    shuffled["bonds"] = np.stack(
        [inverse[data["bonds"][:, 0]], inverse[data["bonds"][:, 1]], data["bonds"][:, 2]], axis=1
    ).astype(np.uint32)
    np.savez(tmp_path / "shuffled.npz", **shuffled)

    structure, _, chain_keys = convert_of3_atom_array(tmp_path / "shuffled.npz", pdb_id="test")

    # Ligand first now, and each chain's slice holds only its own atoms.
    assert chain_keys == ["3", "1"]
    assert int(structure.chains["atom_num"].sum()) == len(structure.atoms)
    for chain, key in zip(structure.chains, chain_keys, strict=True):
        lo, n = int(chain["atom_idx"]), int(chain["atom_num"])
        expected = {"3": {"HEM"}, "1": {"ALA", "SEP", "GLY"}}[key]
        r_lo, r_n = int(chain["res_idx"]), int(chain["res_num"])
        assert {r["name"].decode() for r in structure.residues[r_lo : r_lo + r_n]} == expected
        assert sum(int(r["atom_num"]) for r in structure.residues[r_lo : r_lo + r_n]) == n
        assert int(structure.residues[r_lo]["atom_idx"]) == lo

    # The bond still joins the same two named atoms after the remap.
    bond = structure.bonds[0]
    got = (
        structure.atoms["name"][int(bond["atom_1"])].decode().strip(),
        structure.atoms["name"][int(bond["atom_2"])].decode().strip(),
    )
    assert got == bonded_names


def test_msa_columns_follow_res_id_at_alternate_sites(tmp_path):
    """Two residues modelled at one res_id share ONE alignment column.

    Crystal structures sometimes model alternates at a single SEQRES position —
    3l4j has PTR and TYR both at res_id 362, 6gpv has two such sites. The
    tokenizer's running per-chain counter advances once per *residue*, so it
    walks one past the end of the alignment and throws
    ``index 757 is out of bounds for axis 1 with size 757`` from inside a
    dataloader worker. OF3 indexes by ``res_id - 1`` instead, which maps both
    residues onto the shared column.
    """
    from deepfold3.data.module.of3_pdb_dataset import _msa_cols_from_res_id
    from deepfold3.data.prepare.openfold3 import convert_of3_atom_array
    from deepfold3.data.tokenize.tokenizer import Tokenizer
    from deepfold3.data.types import Input

    # ALA(1) — [SEP(2) / THR(2) alternates] — GLY(3): three SEQRES positions.
    res = [
        ("ALA", 1, ["N", "CA", "C", "O", "CB"]),
        ("SEP", 2, ["N", "CA", "C", "O", "CB", "OG", "P"]),
        ("THR", 2, ["N", "CA", "C", "O", "CB"]),
        ("GLY", 3, ["N", "CA", "C", "O"]),
    ]
    coord, chain_id, res_id, res_name, atom_name, element, atomized = [], [], [], [], [], [], []
    for k, (name, rid, atoms) in enumerate(res):
        for j, a in enumerate(atoms):
            coord.append([float(k), float(j), 0.0])
            chain_id.append("1")
            res_id.append(rid)
            res_name.append(name)
            atom_name.append(a)
            element.append(a[0])
            atomized.append(name == "SEP")
    n = len(coord)
    np.savez(
        tmp_path / "alt.npz",
        coord=np.array(coord, dtype=np.float32),
        chain_id=np.array(chain_id, dtype="<U21"),
        res_id=np.array(res_id, dtype=np.int64),
        ins_code=np.array([""] * n, dtype="<U1"),
        res_name=np.array(res_name, dtype="<U5"),
        atom_name=np.array(atom_name, dtype="<U6"),
        element=np.array(element, dtype="<U2"),
        occupancy=np.ones(n),
        charge=np.zeros(n, dtype=np.int64),
        molecule_type_id=np.zeros(n, dtype=np.int64),
        is_atomized=np.array(atomized, dtype=bool),
        entity_id=np.ones(n, dtype=np.int64),
        label_asym_id=np.array(["A"] * n, dtype="<U4"),
        auth_asym_id=np.array(["A"] * n, dtype="<U4"),
        bonds=np.zeros((0, 3), dtype=np.uint32),
    )

    structure, _, _ = convert_of3_atom_array(tmp_path / "alt.npz", pdb_id="alt", build_record=False)
    tokenized = Tokenizer().tokenize(Input(structure=structure, paired_msa={}, unpaired_msa={}))

    # Four residues over three SEQRES positions.
    assert len(structure.residues) == 4
    counter_cols = np.asarray(tokenized.msa_col_idx)
    assert counter_cols.max() == 3, "the running counter reaches a 4th column..."

    rebased = np.asarray(_msa_cols_from_res_id(tokenized).msa_col_idx)
    # ...but only three alignment columns exist, and the alternates share one.
    assert rebased.max() == 2
    assert set(rebased.tolist()) == {0, 1, 2}

    # Both alternates land on the same column, and every other token is unmoved.
    res_of_token = tokenized.tokens["residue_index"]
    assert set(rebased[res_of_token == 2].tolist()) == {1}


def test_msa_columns_unchanged_without_alternates(tmp_path):
    """The res_id rule is a no-op wherever res_id runs 1..N without duplicates.

    That is what makes it safe to apply unconditionally: measured over 304 live
    structures, only the four carrying an alternate site changed at all.
    """
    from deepfold3.data.module.of3_pdb_dataset import _msa_cols_from_res_id
    from deepfold3.data.prepare.openfold3 import convert_of3_atom_array
    from deepfold3.data.tokenize.tokenizer import Tokenizer
    from deepfold3.data.types import Input

    structure, _, _ = convert_of3_atom_array(_write_atom_array(tmp_path / "s.npz"), pdb_id="t")
    tokenized = Tokenizer().tokenize(Input(structure=structure, paired_msa={}, unpaired_msa={}))
    before = np.asarray(tokenized.msa_col_idx).copy()
    after = np.asarray(_msa_cols_from_res_id(tokenized).msa_col_idx)
    np.testing.assert_array_equal(before, after)


def test_unknown_dna_residue_type(tmp_path):
    """``DN`` is unknown-nucleic (30), not protein UNK (20).

    ``get_res_type`` reaching 20 would route every unknown DNA base through the
    atomized path and give it the protein UNK embedding.
    """
    assert const.get_res_type("DN", const.MOL_DNA) == 30
    assert const.get_res_type("N", const.MOL_RNA) == 30


# --------------------------------------------------------------------------
# Curator weights (DatasetConfig.curator_weights)
# --------------------------------------------------------------------------


def _dataset(cache, tmp_path, **kw):
    """An ``OF3PdbDataset`` over the synthetic cache — weights only, no I/O."""
    from deepfold3.data.module.of3_pdb_dataset import OF3PdbDataset

    return OF3PdbDataset(cache=cache, root=tmp_path, name="t", anchor_sampling=True, **kw)


def _weight_file(path, cache, weights):
    ids = np.array([cache.pdb_id(i) for i in range(len(cache))], dtype="<U8")
    np.savez(path, pdb_id=ids[: len(weights)], weight=np.asarray(weights, dtype=np.float64))
    return path


def test_curator_weights_baseline_is_bit_identical(cache, tmp_path):
    """``m ≡ 1`` must reproduce the untouched AF3 marginal exactly.

    The A/B design relies on the baseline arm being the *same code path* as every
    treatment arm — if ``m ≡ 1`` diverged at all, the control would be measuring
    the plumbing rather than the curation.
    """
    base = _dataset(cache, tmp_path).sampling_weights
    path = _weight_file(tmp_path / "one.npz", cache, np.ones(len(cache)))
    curated = _dataset(cache, tmp_path, curator_weights=path).sampling_weights
    np.testing.assert_array_equal(curated, base)


def test_curator_weights_scale_sampling_mass(cache, tmp_path):
    """A k× multiplier moves k× the relative sampling mass, and stays normalised."""
    base = _dataset(cache, tmp_path).sampling_weights
    mult = np.ones(len(cache))
    mult[0] = 10.0
    path = _weight_file(tmp_path / "w.npz", cache, mult)
    curated = _dataset(cache, tmp_path, curator_weights=path).sampling_weights

    assert curated.sum() == pytest.approx(1.0)
    ratio = (curated[0] / base[0]) / (curated[1:].sum() / base[1:].sum())
    assert ratio == pytest.approx(10.0)


def test_curator_weights_partial_coverage_keeps_baseline(cache, tmp_path):
    """Entries absent from the file keep ``m = 1``.

    A scoring pass that has not finished must degrade toward the baseline, not
    silently zero every entry it has not reached yet.
    """
    base = _dataset(cache, tmp_path).sampling_weights
    path = _weight_file(tmp_path / "part.npz", cache, [4.0])  # only the first pdb_id
    curated = _dataset(cache, tmp_path, curator_weights=path).sampling_weights

    ratio = (curated[0] / base[0]) / (curated[1:].sum() / base[1:].sum())
    assert ratio == pytest.approx(4.0)


def test_curator_weights_survive_config_normalisation(tmp_path):
    """``normalize_datasets`` rebuilds the entry dict field by field.

    A new ``DatasetConfig`` field that is not emitted there is dropped in
    silence — exactly how per-dataset ``crop_weights`` was lost before c0e5e55.
    """
    from deepfold3.config_loader import normalize_datasets

    out = normalize_datasets([{"name": "pdb", "data_dir": "/x", "format": "of3pdb", "curator_weights": "/w.npz"}])
    assert out[0]["curator_weights"] == "/w.npz"
