"""Tests for AF3 JSON input extensions: modifications, bondedAtomPairs, userCCD."""

from __future__ import annotations

import json
import tempfile

from deepfold3.data.parse.json_input import parse_json_input


def _write_json(data) -> str:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(data, f)
    f.flush()
    return f.name


def test_af3_modifications_protein():
    """AF3 ``modifications`` on a protein chain populates chain.modifications."""
    job_data = {
        "dialect": "alphafold3",
        "name": "mods_protein",
        "sequences": [
            {
                "protein": {
                    "id": "A",
                    "sequence": "MKTIISER",
                    "modifications": [
                        {"ptmType": "SEP", "ptmPosition": 6},
                        {"ptmType": "TPO", "ptmPosition": 3},
                    ],
                }
            }
        ],
    }
    path = _write_json(job_data)
    jobs = parse_json_input(path)
    assert len(jobs) == 1
    job = jobs[0]
    assert len(job.protein_chains) == 1
    mods = job.protein_chains[0].modifications
    assert mods == [("SEP", 6), ("TPO", 3)]
    # Other extension fields default to empty / None.
    assert job.bonded_atom_pairs == []
    assert job.user_ccd is None


def test_af3_modifications_rna_and_dna():
    """AF3 ``modifications`` on RNA and DNA chains populate their modifications lists."""
    job_data = {
        "dialect": "alphafold3",
        "name": "mods_nucleotide",
        "sequences": [
            {
                "rna": {
                    "id": "B",
                    "sequence": "ACGU",
                    "modifications": [{"ptmType": "5MC", "ptmPosition": 2}],
                }
            },
            {
                "dna": {
                    "id": "C",
                    "sequence": "ACGT",
                    "modifications": [{"ptmType": "6MA", "ptmPosition": 4}],
                }
            },
        ],
    }
    jobs = parse_json_input(_write_json(job_data))
    job = jobs[0]
    assert len(job.rna_chains) == 1
    assert len(job.dna_chains) == 1
    assert job.rna_chains[0].modifications == [("5MC", 2)]
    assert job.dna_chains[0].modifications == [("6MA", 4)]


def test_af3_bonded_atom_pairs():
    """AF3 ``bondedAtomPairs`` parses into JobInput.bonded_atom_pairs."""
    job_data = {
        "dialect": "alphafold3",
        "name": "bonded",
        "sequences": [
            {"protein": {"id": "A", "sequence": "MKTIIC"}},
            {"ligand": {"id": "B", "ccdCodes": ["HEM"]}},
        ],
        "bondedAtomPairs": [
            [["A", 6, "SG"], ["B", 1, "FE"]],
            [["A", 1, "N"], ["A", 6, "SG"]],
        ],
    }
    jobs = parse_json_input(_write_json(job_data))
    job = jobs[0]
    assert job.bonded_atom_pairs == [
        (("A", 6, "SG"), ("B", 1, "FE")),
        (("A", 1, "N"), ("A", 6, "SG")),
    ]


def test_af3_user_ccd_inline():
    """AF3 inline ``userCCD`` text is stored verbatim on JobInput.user_ccd."""
    user_ccd_text = "data_LIG\n_chem_comp.id LIG\n_chem_comp.name 'test ligand'\n"
    job_data = {
        "dialect": "alphafold3",
        "name": "custom_ccd",
        "sequences": [{"protein": {"id": "A", "sequence": "MKTII"}}],
        "userCCD": user_ccd_text,
    }
    jobs = parse_json_input(_write_json(job_data))
    job = jobs[0]
    assert job.user_ccd == user_ccd_text


def test_af3_user_ccd_path(tmp_path):
    """AF3 ``userCCDPath`` is read from disk into JobInput.user_ccd."""
    ccd_path = tmp_path / "my_ccd.cif"
    ccd_path.write_text("data_FOO\n_chem_comp.id FOO\n")

    job_data = {
        "dialect": "alphafold3",
        "name": "custom_ccd_path",
        "sequences": [{"protein": {"id": "A", "sequence": "MKTII"}}],
        "userCCDPath": str(ccd_path),
    }
    jobs = parse_json_input(_write_json(job_data))
    job = jobs[0]
    assert job.user_ccd is not None
    assert "data_FOO" in job.user_ccd


def test_alphafoldserver_modifications(monkeypatch):
    """AlphaFold-Server dialect also reads ``modifications`` on each chain."""
    # Stub MSA server search so the test does not hit the network.
    from deepfold3.data.parse import json_input as jin

    monkeypatch.setattr(jin, "_run_msa_search", lambda *a, **kw: {})

    job_data = {
        "dialect": "alphafoldserver",
        "name": "afserver_mods",
        "sequences": [
            {
                "proteinChain": {
                    "sequence": "MKTIISER",
                    "count": 1,
                    "modifications": [{"ptmType": "SEP", "ptmPosition": 6}],
                }
            }
        ],
    }
    jobs = parse_json_input(_write_json(job_data))
    job = jobs[0]
    assert len(job.protein_chains) == 1
    assert job.protein_chains[0].modifications == [("SEP", 6)]


def test_deepfold3_modifications_bonds_user_ccd():
    """DeepFold3 dialect accepts all AF3 extensions regardless of key style."""
    job_data = {
        "dialect": "deepfold3",
        "name": "deepfold3_ext",
        "sequences": [
            {
                "protein": {
                    "id": "A",
                    "sequence": "MKTIISER",
                    "modifications": [{"ptmType": "SEP", "ptmPosition": 6}],
                }
            }
        ],
        "bondedAtomPairs": [[["A", 1, "N"], ["A", 2, "C"]]],
        "userCCD": "data_X\n_chem_comp.id X\n",
    }
    jobs = parse_json_input(_write_json(job_data))
    job = jobs[0]
    assert job.protein_chains[0].modifications == [("SEP", 6)]
    assert job.bonded_atom_pairs == [(("A", 1, "N"), ("A", 2, "C"))]
    assert job.user_ccd == "data_X\n_chem_comp.id X\n"


def test_deepfold3_modifications_protenix_keys():
    """DeepFold3 dialect parses modifications on Protenix-style keys too."""
    job_data = {
        "dialect": "deepfold3",
        "name": "deepfold3_protenix_mods",
        "sequences": [
            {
                "proteinChain": {
                    "sequence": "MKTIISER",
                    "count": 1,
                    "modifications": [{"ptmType": "SEP", "ptmPosition": 6}],
                }
            }
        ],
    }
    jobs = parse_json_input(_write_json(job_data))
    job = jobs[0]
    assert len(job.protein_chains) == 1
    assert job.protein_chains[0].modifications == [("SEP", 6)]


def test_modifications_absent_default_empty():
    """Missing modifications fields default to empty lists."""
    job_data = {
        "dialect": "alphafold3",
        "name": "no_mods",
        "sequences": [{"protein": {"id": "A", "sequence": "MKTII"}}],
    }
    jobs = parse_json_input(_write_json(job_data))
    job = jobs[0]
    assert job.protein_chains[0].modifications == []
    assert job.bonded_atom_pairs == []
    assert job.user_ccd is None


def test_user_ccd_prefers_inline_over_path(tmp_path):
    """When both userCCD and userCCDPath are set, userCCD wins."""
    ccd_path = tmp_path / "not_used.cif"
    ccd_path.write_text("data_FILE\n")
    job_data = {
        "dialect": "alphafold3",
        "name": "prefer_inline",
        "sequences": [{"protein": {"id": "A", "sequence": "MKTII"}}],
        "userCCD": "data_INLINE\n",
        "userCCDPath": str(ccd_path),
    }
    jobs = parse_json_input(_write_json(job_data))
    job = jobs[0]
    assert job.user_ccd == "data_INLINE\n"
