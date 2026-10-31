"""Input-JSON chain ids survive to the output.

``InferenceInput.from_job`` names its internal chains A, B, C, … in asym_id
order, so before this was plumbed through, a job asking for chains A/B/D/E got
A/B/C/D back: predicted "C" was really the caller's "D". OST-style scorers map by
sequence and were unaffected, but every name-keyed consumer (``chain_pair_iptm``,
the bench's per-interface chain ids, any downstream analysis) silently compared
the wrong pairs.

The load-bearing invariant is that ``job_chain_ids`` returns names in exactly the
order ``from_job`` assigns asym_ids. These tests pin that ordering — including
the protein → DNA → RNA → ligand grouping and multi-copy ``id`` lists — so the
two cannot drift apart.
"""

from __future__ import annotations

import json

import pytest

from deepfold3.data.module.dataset import job_chain_ids
from deepfold3.data.parse.json_input import parse_json_input
from deepfold3.io.mmcif import asym_id_to_chain_name, normalize_chain_names, resolve_chain_name


def _write(tmp_path, entry: dict):
    p = tmp_path / "job.json"
    p.write_text(json.dumps(entry))
    return p


def _job(tmp_path, sequences: list[dict]):
    entry = {"name": "t", "dialect": "deepfold3", "version": 3, "modelSeeds": [1], "sequences": sequences}
    return parse_json_input(_write(tmp_path, entry))[0]


def test_chain_ids_preserved_with_a_gap(tmp_path):
    """The regression case: ids A/B/D/E must not come back as A/B/C/D."""
    job = _job(
        tmp_path,
        [
            {"protein": {"id": "A", "sequence": "AAAA"}},
            {"protein": {"id": "B", "sequence": "CCCC"}},
            {"protein": {"id": "D", "sequence": "DDDD"}},
            {"protein": {"id": "E", "sequence": "EEEE"}},
        ],
    )
    assert job_chain_ids(job) == ["A", "B", "D", "E"]
    assert [resolve_chain_name(i, job_chain_ids(job)) for i in range(4)] == ["A", "B", "D", "E"]


def test_ordering_is_protein_then_dna_then_rna_then_ligand(tmp_path):
    """asym_id order is by molecule class, not by JSON order."""
    job = _job(
        tmp_path,
        [
            {"dna": {"id": "X", "sequence": "ACGT"}},
            {"protein": {"id": "P", "sequence": "AAAA"}},
            {"ligand": {"id": "L", "ccdCodes": ["ATP"]}},
            {"rna": {"id": "R", "sequence": "ACGU"}},
        ],
    )
    assert job_chain_ids(job) == ["P", "X", "R", "L"]


def test_multi_copy_id_list_expands_in_order(tmp_path):
    """One chain entry with several ids becomes one chain per id."""
    job = _job(tmp_path, [{"protein": {"id": ["H", "I", "J"], "sequence": "AAAA"}}])
    assert job_chain_ids(job) == ["H", "I", "J"]


def test_unnamed_dialect_falls_back_to_synthetic(tmp_path):
    """Protenix/AF-Server style input names no chains, so nothing changes."""
    entry = {
        "name": "t",
        "sequences": [{"proteinChain": {"sequence": "AAAA", "count": 2}}],
    }
    job = parse_json_input(_write(tmp_path, entry))[0]
    ids = job_chain_ids(job)
    assert ids == [None, None]
    assert [resolve_chain_name(i, ids) for i in range(2)] == ["A", "B"]


def test_multi_ccd_ligand_declines_to_guess(tmp_path):
    """from_job expands ccdCodes into one chain each; one id can't label them all."""
    job = _job(
        tmp_path, [{"protein": {"id": "A", "sequence": "AAAA"}}, {"ligand": {"id": "L", "ccdCodes": ["ATP", "MG"]}}]
    )
    assert job_chain_ids(job) == ["A", None, None]


def test_duplicate_resolved_names_are_rejected(tmp_path):
    """A supplied name colliding with another chain's synthetic name is unsafe."""
    # chain 0 -> "C" (supplied), chain 1 -> unnamed, whose synthetic name is also "B"... use a real clash:
    # chain 0 supplied "B", chain 1 unnamed -> synthetic "B" => duplicate.
    assert normalize_chain_names(["B", None], [0, 1]) is None
    # no clash: identity mapping is kept
    assert normalize_chain_names(["A", None], [0, 1]) == ["A", "B"]


@pytest.mark.parametrize("n", [0, 1, 25, 26, 27])
def test_synthetic_fallback_unchanged(n):
    """Historical naming is untouched when no ids are supplied."""
    assert resolve_chain_name(n, None) == asym_id_to_chain_name(n)


def test_ordering_matches_from_job_asym_ids(tmp_path):
    """The drift guard: featurized asym_id order must match ``job_chain_ids``.

    ``job_chain_ids`` mirrors ``from_job``'s grouping by hand, so if either side
    is reordered the names would attach to the wrong chains with nothing else
    failing. Anchor it on distinct sequence lengths: token counts per asym_id
    identify which chain each asym_id actually is. (Ligands are omitted so the
    test needs no CCD cache.)
    """
    from deepfold3.data.module.dataset import InferenceInput

    # length -> the id we asked for, with lengths distinct so they identify chains
    spec = [
        ({"protein": {"id": "P1", "sequence": "A" * 7}}, 7, "P1"),
        ({"protein": {"id": "P2", "sequence": "C" * 9}}, 9, "P2"),
        ({"dna": {"id": "DX", "sequence": "ACGTACGTACG"}}, 11, "DX"),
        ({"rna": {"id": "RY", "sequence": "ACGUACGUACGUA"}}, 13, "RY"),
    ]
    # deliberately shuffled in the JSON: asym order is by class, not JSON order
    sequences = [spec[2][0], spec[0][0], spec[3][0], spec[1][0]]
    job = _job(tmp_path, sequences)

    names = job_chain_ids(job)
    feats = InferenceInput.from_job(job)
    asym = feats["asym_id"]

    by_len = {n_tok: cid for _, n_tok, cid in spec}
    for a in asym.unique(sorted=True).tolist():
        n_tok = int((asym == a).sum())
        assert resolve_chain_name(int(a), names) == by_len[n_tok], (
            f"asym_id {a} has {n_tok} tokens (chain {by_len[n_tok]}) but was named "
            f"{resolve_chain_name(int(a), names)!r} — job_chain_ids drifted from from_job"
        )


def _template_cif(tmp_path, seq: str):
    """A template chain of ``seq`` with ``label_seq_id`` 1..len(seq)."""
    gemmi = pytest.importorskip("gemmi")
    st = gemmi.Structure()
    st.name = "TMPL"
    model = gemmi.Model("1")
    chain = gemmi.Chain("A")
    for i, aa in enumerate(seq):
        res = gemmi.Residue()
        res.name = gemmi.expand_one_letter(aa, gemmi.ResidueKind.AA)
        res.seqid = gemmi.SeqId(i + 1, " ")
        res.label_seq = i + 1
        res.entity_type = gemmi.EntityType.Polymer
        for j, (name, el) in enumerate([("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O"), ("CB", "C")]):
            atom = gemmi.Atom()
            atom.name = name
            atom.element = gemmi.Element(el)
            atom.pos = gemmi.Position(3.8 * i + j * 0.5, j * 1.1, 0.3 * j)
            res.add_atom(atom)
        chain.add_residue(res)
    model.add_chain(chain)
    st.add_model(model)
    st.setup_entities()
    path = tmp_path / "tmpl.cif"
    st.make_mmcif_document().write_file(str(path))
    return path


def test_json_template_lands_on_its_own_chain_with_zero_based_indices(tmp_path):
    """``queryIndices`` / ``templateIndices`` are 0-based and local to the chain
    declaring them. A hit on chain C used to land on chain A, one residue off,
    and every chain after the first templated one lost its hits."""
    from deepfold3.data.module.dataset import InferenceInput

    seqs = {"A": "MKTAYIAKQR", "B": "GWSVLDEPNH", "C": "FYHRTGLWKC"}
    cif = _template_cif(tmp_path, seqs["C"])
    idx = list(range(len(seqs["C"])))

    def protein(cid):
        seq = seqs[cid]
        p = {"id": cid, "sequence": seq, "unpairedMsa": f">q\n{seq}\n", "pairedMsa": "", "templates": []}
        if cid == "C":
            p["templates"] = [{"mmcifPath": str(cif), "queryIndices": idx, "templateIndices": idx}]
        return {"protein": p}

    entry = {
        "name": "t",
        "modelSeeds": [1],
        "dialect": "alphafold3",
        "version": 2,
        "sequences": [protein(c) for c in seqs],
    }
    feats = InferenceInput.from_job(parse_json_input(_write(tmp_path, entry))[0])
    mask = feats["template_pseudo_beta_mask"][0].bool()
    on_c = feats["asym_id"] == 2
    assert mask[on_c].all()
    assert not mask[~on_c].any()
    assert feats["template_aatype"][0][on_c].tolist() == feats["res_type"][on_c].tolist()
