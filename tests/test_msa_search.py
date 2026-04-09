"""Tests for the MMseqs2 MSA search module."""

from __future__ import annotations

import gzip
import io
import json
import os
import tarfile
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from deepfold3.data.msa.mmseqs2 import (
    MsaResult,
    _build_query_fasta,
    _cache_key,
    _extract_a3m,
    _parse_fasta,
    _save_cache,
    _sequence_hash,
    _try_load_cache,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tar_gz(files: dict[str, str]) -> bytes:
    """Create an in-memory tar.gz with the given filename→content mapping."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, content in files.items():
            data = content.encode()
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_build_query_fasta():
    result = _build_query_fasta(["ACDE", "FGHI"])
    assert result == ">101\nACDE\n>102\nFGHI\n"


def test_build_query_fasta_single():
    result = _build_query_fasta(["MKTII"])
    assert result == ">101\nMKTII\n"


def test_parse_fasta():
    text = ">seq1 description\nACDE\nFGHI\n>seq2\nKLMN\n"
    entries = _parse_fasta(text)
    assert len(entries) == 2
    assert entries[0] == ("seq1", "ACDEFGHI")
    assert entries[1] == ("seq2", "KLMN")


def test_parse_fasta_empty():
    assert _parse_fasta("") == []
    assert _parse_fasta("\n\n") == []


def test_parse_fasta_round_trip():
    seqs = [("101", "ACDEFGHIKLMNPQRSTVWY"), ("102", "GGGGGGGG")]
    fasta = "".join(f">{name}\n{seq}\n" for name, seq in seqs)
    parsed = _parse_fasta(fasta)
    assert parsed == seqs


def test_sequence_hash():
    h1 = _sequence_hash(["ACDE", "FGHI"])
    h2 = _sequence_hash(["FGHI", "ACDE"])
    assert h1 == h2  # sorted, so order doesn't matter
    assert len(h1) == 16


def test_extract_a3m_single():
    """Single query, uniref only."""
    uniref_a3m = ">101\nACDEFG\n>hit1\nACDEFG\n>hit2\nACDE-G\n"
    tar_bytes = _make_tar_gz({"uniref.a3m": uniref_a3m})
    result = _extract_a3m(tar_bytes, n_chains=1, use_env=False, use_pairing=False)
    assert 0 in result.unpaired
    assert ">101" in result.unpaired[0]
    assert ">hit1" in result.unpaired[0]
    assert len(result.paired) == 0


def test_extract_a3m_multi():
    """Two queries separated by \\x00."""
    uniref_a3m = ">101\nACDE\n>hit1\nACDE\n\x00>102\nFGHI\n>hit2\nFGHI\n"
    tar_bytes = _make_tar_gz({"uniref.a3m": uniref_a3m})
    result = _extract_a3m(tar_bytes, n_chains=2, use_env=False, use_pairing=False)
    assert 0 in result.unpaired
    assert 1 in result.unpaired
    assert ">101" in result.unpaired[0]
    assert ">hit1" in result.unpaired[0]
    assert ">102" in result.unpaired[1]
    assert ">hit2" in result.unpaired[1]
    # Chain 0 should NOT contain chain 1's sequences
    assert "FGHI" not in result.unpaired[0]


def test_extract_a3m_env():
    """UniRef + environmental DB merged."""
    uniref_a3m = ">101\nACDE\n>uhit\nACDE\n"
    env_a3m = ">101\nACDE\n>ehit\nAC-E\n"
    tar_bytes = _make_tar_gz(
        {
            "uniref.a3m": uniref_a3m,
            "bfd.mgnify30.metaeuk30.smag30.a3m": env_a3m,
        }
    )
    result = _extract_a3m(tar_bytes, n_chains=1, use_env=True, use_pairing=False)
    text = result.unpaired[0]
    assert ">uhit" in text
    assert ">ehit" in text
    # Query should appear only from uniref, not duplicated from env
    assert text.count(">101") == 1
    # Env query sequence must not leak as a bare (headerless) line
    lines = text.strip().splitlines()
    for j, line in enumerate(lines):
        if not line.startswith(">"):
            assert j > 0 and lines[j - 1].startswith(">"), f"bare sequence at line {j}: {line!r}"


def test_extract_a3m_env_query_only():
    """Env block with only the query entry (no hits) should contribute nothing."""
    uniref_a3m = ">101\nACDE\n>uhit\nACDE\n"
    env_a3m = ">101\nACDE\n"
    tar_bytes = _make_tar_gz(
        {
            "uniref.a3m": uniref_a3m,
            "bfd.mgnify30.metaeuk30.smag30.a3m": env_a3m,
        }
    )
    result = _extract_a3m(tar_bytes, n_chains=1, use_env=True, use_pairing=False)
    # Output should be identical to uniref-only
    assert result.unpaired[0] == uniref_a3m


def test_extract_a3m_paired():
    """Paired A3M extraction."""
    pair_a3m = ">101\nACDE\n>phit\nACDE\n\x00>102\nFGHI\n>phit\nFGHI\n"
    tar_bytes = _make_tar_gz({"pair.a3m": pair_a3m})
    result = _extract_a3m(tar_bytes, n_chains=2, use_env=False, use_pairing=True)
    assert 0 in result.paired
    assert 1 in result.paired
    assert ">101" in result.paired[0]
    assert ">102" in result.paired[1]


def test_cache_round_trip():
    """Write and read MsaResult from cache."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        sequences = ["ACDE", "FGHI"]
        key = _cache_key(sequences, use_env=True, use_pairing=False)

        result = MsaResult(
            unpaired={0: ">101\nACDE\n", 1: ">102\nFGHI\n"},
            paired={},
            m8_text="101\t1abc_A\t95.0\t100\t5\t0\t1\t100\t10\t109\t1e-50\t500\n",
        )
        _save_cache(cache_dir, key, result)

        loaded = _try_load_cache(cache_dir, key, n_chains=2)
        assert loaded is not None
        assert loaded.unpaired[0] == result.unpaired[0]
        assert loaded.unpaired[1] == result.unpaired[1]
        assert loaded.m8_text == result.m8_text


def test_cache_key_includes_result_shaping_flags():
    sequences = ["ACDE", "FGHI"]

    base = _cache_key(sequences, use_env=True, use_pairing=True)
    assert _cache_key(sequences, use_env=True, use_pairing=True, use_filter=False) != base
    assert _cache_key(sequences, use_env=True, use_pairing=True, use_templates=True) != base
    assert _cache_key(sequences, use_env=True, use_pairing=True, use_env_pairing=True) != base
    assert _cache_key(sequences, use_env=True, use_pairing=True, pairing_strategy="complete") != base


def test_cache_miss():
    """Cache miss returns None."""
    with tempfile.TemporaryDirectory() as tmpdir:
        loaded = _try_load_cache(Path(tmpdir), "nonexistent", n_chains=1)
        assert loaded is None


def test_output_parseable():
    """MsaResult A3M text can be parsed by parse_a3m."""
    from deepfold3.data.parse.a3m import parse_a3m

    a3m_text = ">101\nMKTIIALSYIFCLVFA\n>hit1\nMKTIIALS-IFCLVFA\n>hit2\nMKTII-LSYIFCLVFA\n"
    msa = parse_a3m(a3m_text)
    assert msa.sequences.shape[0] == 3  # 3 sequences
    assert msa.sequences.shape[1] == 16  # 16 residues


def test_local_command_sequence():
    """Verify correct mmseqs command sequence for local monomer search."""
    commands: list[list[str]] = []

    def mock_run(cmd, **kwargs):
        commands.append([str(x) for x in cmd])
        result = mock.MagicMock()
        result.returncode = 0
        result.stderr = ""
        return result

    with tempfile.TemporaryDirectory() as tmpdir:
        work_dir = Path(tmpdir)
        db_dir = work_dir / "dbs"
        db_dir.mkdir()

        # Create fake database files
        (db_dir / "uniref30_2302_db.dbtype").touch()

        # Create fake output files that unpack_results expects
        for i in range(1):
            (work_dir / "work" / f"{i}.a3m").parent.mkdir(parents=True, exist_ok=True)

        with mock.patch("subprocess.run", side_effect=mock_run):
            # Create a fake work directory with expected outputs
            work = work_dir / "work"
            work.mkdir(exist_ok=True)
            (work / "0.a3m").write_text(">101\nACDE\n")

            from deepfold3.data.msa.mmseqs2 import _run_mmseqs, _search_monomer

            # Write query
            query_fasta = ">101\nACDE\n"
            query_file = work / "query.fas"
            query_file.write_text(query_fasta)
            _run_mmseqs("mmseqs", ["createdb", query_file, work / "qdb", "--shuffle", "0", "--dbtype", "1"])

            commands.clear()
            _search_monomer(
                "mmseqs", db_dir, work, use_env=False, use_templates=False, filter=True, threads=4, gpu=False, gpu_server=False
            )

            # Verify key commands in order
            modules = [cmd[1] for cmd in commands]
            assert "search" in modules
            assert "mvdb" in modules
            assert "lndb" in modules
            assert "expandaln" in modules
            assert "align" in modules
            assert "filterresult" in modules
            assert "result2msa" in modules
            assert "unpackdb" in modules

            # search should come before expandaln
            assert modules.index("search") < modules.index("expandaln")
            # result2msa should come before unpackdb
            assert modules.index("result2msa") < modules.index("unpackdb")


# ---------------------------------------------------------------------------
# Colon-separated multimer + dedup tests
# ---------------------------------------------------------------------------


def test_sequence_colon_split():
    """--sequence with colon splits into multiple chains."""
    from click.testing import CliRunner

    from deepfold3.scripts.msa_search import msa_search

    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmpdir:
        # Invoke with a mock that captures the sequences passed to search_server
        captured = {}

        def fake_search(sequences, **kwargs):
            captured["sequences"] = sequences
            return MsaResult(unpaired={i: f">{101 + i}\n{s}\n" for i, s in enumerate(sequences)})

        with mock.patch("deepfold3.scripts.msa_search.search_server", side_effect=fake_search):
            result = runner.invoke(msa_search, ["--sequence", "AAAA:BBBB:AAAA", "--output", tmpdir])

        assert result.exit_code == 0, result.output
        # Should deduplicate: only 2 unique sequences searched
        assert captured["sequences"] == ["AAAA", "BBBB"]
        # But 3 output files written (chain 2 = copy of chain 0)
        assert (Path(tmpdir) / "unpaired_0.a3m").exists()
        assert (Path(tmpdir) / "unpaired_1.a3m").exists()
        assert (Path(tmpdir) / "unpaired_2.a3m").exists()
        # Chain 0 and chain 2 should have the same content
        assert (Path(tmpdir) / "unpaired_0.a3m").read_text() == (Path(tmpdir) / "unpaired_2.a3m").read_text()


def test_sequence_single_no_colon():
    """--sequence without colon is a single monomer chain."""
    from click.testing import CliRunner

    from deepfold3.scripts.msa_search import msa_search

    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmpdir:
        captured = {}

        def fake_search(sequences, **kwargs):
            captured["sequences"] = sequences
            return MsaResult(unpaired={0: ">101\nACDE\n"})

        with mock.patch("deepfold3.scripts.msa_search.search_server", side_effect=fake_search):
            result = runner.invoke(msa_search, ["--sequence", "ACDE", "--output", tmpdir])

        assert result.exit_code == 0, result.output
        assert captured["sequences"] == ["ACDE"]
        assert (Path(tmpdir) / "unpaired_0.a3m").exists()


def test_skeleton_json_output():
    """--json flag writes an AF3-format skeleton JSON file."""
    from click.testing import CliRunner

    from deepfold3.scripts.msa_search import msa_search

    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmpdir:

        def fake_search(sequences, **kwargs):
            return MsaResult(
                unpaired={i: f">{101 + i}\n{s}\n" for i, s in enumerate(sequences)},
                paired={0: ">101\nAAAA\n>phit\nAAAA\n", 1: ">102\nBBBB\n>phit\nBBBB\n"},
            )

        with mock.patch("deepfold3.scripts.msa_search.search_server", side_effect=fake_search):
            result = runner.invoke(
                msa_search,
                ["--sequence", "AAAA:BBBB", "--output", tmpdir, "--json", "--name", "test_job", "--no-templates"],
            )

        assert result.exit_code == 0, result.output
        json_path = Path(tmpdir) / "test_job.json"
        assert json_path.exists(), f"JSON file not created; output:\n{result.output}"

        data = json.loads(json_path.read_text())
        assert data["name"] == "test_job"
        assert data["dialect"] == "deepfold"
        assert data["version"] == 3
        assert data["modelSeeds"] == [42]
        assert len(data["sequences"]) == 2

        # AF3 format uses "protein" key with "id" field
        chain0 = data["sequences"][0]["protein"]
        assert chain0["id"] == "A"
        assert chain0["sequence"] == "AAAA"
        assert chain0["unpairedMsaPath"] == "./unpaired_0.a3m"
        assert chain0["pairedMsaPath"] == "./paired_0.a3m"

        chain1 = data["sequences"][1]["protein"]
        assert chain1["id"] == "B"
        assert chain1["sequence"] == "BBBB"
        assert chain1["unpairedMsaPath"] == "./unpaired_1.a3m"
        assert chain1["pairedMsaPath"] == "./paired_1.a3m"


# ---------------------------------------------------------------------------
# Template m8 parsing tests
# ---------------------------------------------------------------------------


def test_parse_m8():
    """Parse pdb70.m8 text into per-chain hits."""
    from deepfold3.data.msa.mmseqs2 import parse_m8

    m8_text = (
        "101\t1abc_A\t95.0\t100\t5\t0\t1\t100\t10\t109\t1e-50\t500\n"
        "101\t2def_B\t80.0\t50\t10\t0\t5\t54\t20\t69\t1e-20\t200\n"
        "102\t3ghi_C\t90.0\t80\t8\t0\t1\t80\t1\t80\t1e-40\t400\n"
    )
    hits = parse_m8(m8_text)
    assert 0 in hits  # chain 0 (query 101)
    assert 1 in hits  # chain 1 (query 102)
    assert len(hits[0]) == 2
    assert len(hits[1]) == 1

    h = hits[0][0]
    assert h.pdb_id == "1ABC"
    assert h.chain_id == "A"
    assert h.qstart == 1
    assert h.qend == 100
    assert h.evalue == 1e-50


def test_extract_a3m_with_m8():
    """Tar.gz with pdb70.m8 — m8_text extracted in MsaResult."""
    m8_content = "101\t1abc_A\t95.0\t100\t5\t0\t1\t100\t10\t109\t1e-50\t500\n"
    uniref_a3m = ">101\nACDE\n>hit1\nACDE\n"
    tar_bytes = _make_tar_gz({"uniref.a3m": uniref_a3m, "pdb70.m8": m8_content})
    result = _extract_a3m(tar_bytes, n_chains=1, use_env=False, use_pairing=False)
    assert result.m8_text == m8_content
    assert ">101" in result.unpaired[0]


def test_skeleton_json_with_templates():
    """--json + --use-templates writes template references in skeleton JSON."""
    from click.testing import CliRunner

    from deepfold3.scripts.msa_search import msa_search

    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmpdir:
        m8_text = "101\t1abc_A\t95.0\t100\t5\t0\t1\t100\t10\t109\t1e-50\t500\n"

        def fake_search(sequences, **kwargs):
            return MsaResult(
                unpaired={0: ">101\nACDE\n"},
                m8_text=m8_text,
            )

        def fake_fetch(pdb_id, **kwargs):
            return f"data_{pdb_id}\n_entry.id {pdb_id}\n"

        with (
            mock.patch("deepfold3.scripts.msa_search.search_server", side_effect=fake_search),
            mock.patch("deepfold3.data.parse.templates.fetch_pdb_mmcif", side_effect=fake_fetch),
        ):
            result = runner.invoke(
                msa_search,
                ["--sequence", "ACDE", "--output", tmpdir, "--json", "--name", "tmpl_job", "--use-templates"],
            )

        assert result.exit_code == 0, result.output

        # Check pdb70.m8 was saved
        assert (Path(tmpdir) / "pdb70.m8").exists()

        # Check template mmCIF was downloaded
        assert (Path(tmpdir) / "templates" / "1abc_A.cif").exists()

        # Check JSON references templates
        data = json.loads((Path(tmpdir) / "tmpl_job.json").read_text())
        chain0 = data["sequences"][0]["protein"]
        assert "templatesPath" in chain0
        assert chain0["templatesPath"] == "./templates/templates_0.json"

        # Check template JSON hits file
        tmpl_hits = json.loads((Path(tmpdir) / "templates" / "templates_0.json").read_text())
        assert len(tmpl_hits) == 1
        assert tmpl_hits[0]["name"] == "1abc_A.cif"
        assert tmpl_hits[0]["chainId"] == "A"
        assert tmpl_hits[0]["queryIndices"] == list(range(0, 100))


def test_no_fetch_templates():
    """--no-fetch-templates skips mmCIF download but saves m8."""
    from click.testing import CliRunner

    from deepfold3.scripts.msa_search import msa_search

    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmpdir:
        m8_text = "101\t1abc_A\t95.0\t100\t5\t0\t1\t100\t10\t109\t1e-50\t500\n"

        def fake_search(sequences, **kwargs):
            return MsaResult(unpaired={0: ">101\nACDE\n"}, m8_text=m8_text)

        with mock.patch("deepfold3.scripts.msa_search.search_server", side_effect=fake_search):
            result = runner.invoke(
                msa_search,
                ["--sequence", "ACDE", "--output", tmpdir, "--use-templates", "--no-fetch-templates"],
            )

        assert result.exit_code == 0, result.output
        # m8 saved
        assert (Path(tmpdir) / "pdb70.m8").exists()
        # No templates directory created
        assert not (Path(tmpdir) / "templates").exists()


def test_link_templates():
    """--link-templates creates symlinks to PDB mirror files."""
    from click.testing import CliRunner

    from deepfold3.scripts.msa_search import msa_search

    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        output = tmpdir / "out"
        mirror = tmpdir / "mirror"

        # Create a fake PDB mirror with a gzipped CIF
        subdir = mirror / "ab"
        subdir.mkdir(parents=True)
        content = "data_1ABC\n_entry.id 1ABC\n"
        with gzip.open(subdir / "1abc.cif.gz", "wt") as f:
            f.write(content)

        m8_text = "101\t1abc_A\t95.0\t100\t5\t0\t1\t50\t10\t59\t1e-50\t500\n"

        def fake_search(sequences, **kwargs):
            return MsaResult(unpaired={0: ">101\nACDE\n"}, m8_text=m8_text)

        with mock.patch("deepfold3.scripts.msa_search.search_server", side_effect=fake_search):
            result = runner.invoke(
                msa_search,
                [
                    "--sequence",
                    "ACDE",
                    "--output",
                    str(output),
                    "--use-templates",
                    "--link-templates",
                    "--pdb-mmcif-dir",
                    str(mirror),
                ],
            )

        assert result.exit_code == 0, result.output

        # Check symlink was created
        link = output / "templates" / "1abc_A.cif.gz"
        assert link.is_symlink()
        assert link.resolve() == (subdir / "1abc.cif.gz").resolve()

        # Check template JSON was created
        tmpl_json = output / "templates" / "templates_0.json"
        assert tmpl_json.exists()
        hits = json.loads(tmpl_json.read_text())
        assert hits[0]["name"] == "1abc_A.cif.gz"
        assert hits[0]["chainId"] == "A"


# ---------------------------------------------------------------------------
# PDB mirror lookup tests
# ---------------------------------------------------------------------------


def test_read_pdb_mmcif_local_gzipped():
    """Find a gzip-compressed mmCIF in standard PDB mirror layout."""
    from deepfold3.data.parse.templates import read_pdb_mmcif_local

    with tempfile.TemporaryDirectory() as tmpdir:
        mirror = Path(tmpdir)
        # Standard layout: mmcif/3v/a3vb.cif.gz
        subdir = mirror / "3v"
        subdir.mkdir()
        content = "data_A3VB\n_entry.id A3VB\n"
        with gzip.open(subdir / "a3vb.cif.gz", "wt") as f:
            f.write(content)

        result = read_pdb_mmcif_local("A3VB", mirror)
        assert result is not None
        assert result == content


def test_read_pdb_mmcif_local_plain():
    """Find an uncompressed mmCIF in PDB mirror layout."""
    from deepfold3.data.parse.templates import read_pdb_mmcif_local

    with tempfile.TemporaryDirectory() as tmpdir:
        mirror = Path(tmpdir)
        subdir = mirror / "ab"
        subdir.mkdir()
        content = "data_1ABC\n_entry.id 1ABC\n"
        (subdir / "1abc.cif").write_text(content)

        result = read_pdb_mmcif_local("1ABC", mirror)
        assert result is not None
        assert result == content


def test_read_pdb_mmcif_local_flat():
    """Find mmCIF in flat layout (no mid2 subdirectory)."""
    from deepfold3.data.parse.templates import read_pdb_mmcif_local

    with tempfile.TemporaryDirectory() as tmpdir:
        mirror = Path(tmpdir)
        content = "data_2XYZ\n"
        with gzip.open(mirror / "2xyz.cif.gz", "wt") as f:
            f.write(content)

        result = read_pdb_mmcif_local("2XYZ", mirror)
        assert result is not None
        assert result == content


def test_read_pdb_mmcif_local_missing():
    """Return None when PDB ID not found."""
    from deepfold3.data.parse.templates import read_pdb_mmcif_local

    with tempfile.TemporaryDirectory() as tmpdir:
        result = read_pdb_mmcif_local("9ZZZ", Path(tmpdir))
        assert result is None


def test_fetch_pdb_mmcif_prefers_local():
    """fetch_pdb_mmcif uses local mirror before cache or download."""
    from deepfold3.data.parse.templates import fetch_pdb_mmcif

    with tempfile.TemporaryDirectory() as tmpdir:
        mirror = Path(tmpdir) / "mirror"
        subdir = mirror / "ab"
        subdir.mkdir(parents=True)
        content = "data_1ABC\n_entry.id 1ABC\n"
        with gzip.open(subdir / "1abc.cif.gz", "wt") as f:
            f.write(content)

        # Should find it locally without any network call
        result = fetch_pdb_mmcif("1ABC", pdb_mmcif_dir=mirror)
        assert result == content


@pytest.mark.skipif(
    os.environ.get("DEEPFOLD3_TEST_NETWORK") != "1",
    reason="Network test; set DEEPFOLD3_TEST_NETWORK=1 to run",
)
def test_server_search():
    """Integration test: actual search against ColabFold API."""
    from deepfold3.data.msa.mmseqs2 import search_server

    result = search_server(["MKTIIALSYIFCLVFA"], use_env=False, use_pairing=False, timeout=300)
    assert 0 in result.unpaired
    assert len(result.unpaired[0]) > 0
    assert ">101" in result.unpaired[0] or ">10" in result.unpaired[0]

    # Verify parseable
    from deepfold3.data.parse.a3m import parse_a3m

    msa = parse_a3m(result.unpaired[0])
    assert msa.sequences.shape[0] >= 1
    assert msa.sequences.shape[1] == 16
