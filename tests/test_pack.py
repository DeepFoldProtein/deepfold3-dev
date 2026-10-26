"""Tests for packed datasets (tar shards + sorted index) and pack-aware loaders."""

from __future__ import annotations

import gzip
import json
import multiprocessing as mp
import shutil
import tarfile
from pathlib import Path

import numpy as np
import pytest

from deepfold3.data import const, pack
from deepfold3.data.io_cache import read_npz
from deepfold3.data.prepare.pack_build import build, parse_size, verify


@pytest.fixture(autouse=True)
def _clean_registry():
    pack.clear_registry()
    yield
    pack.clear_registry()


def _df3_tree(root: Path) -> dict[str, bytes]:
    """A miniature df3 data dir; returns {relative path: bytes} of packable files."""
    files: dict[str, bytes] = {}
    rng = np.random.default_rng(0)
    for rid in ("1abc", "1abd", "2xyz", "9zzz"):
        p = root / "structures" / rid[:2] / f"{rid}.npz"
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(p, atoms=rng.standard_normal((10 + len(files), 3)))
        files[f"structures/{rid[:2]}/{rid}.npz"] = p.read_bytes()
    for h in ("ab" + "0" * 62, "cd" + "1" * 62):
        p = root / "msa" / "unpaired" / h[:2] / f"{h}.npz"
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(p, sequences=rng.integers(0, 20, (5, 7)).astype(np.int8))
        files[f"msa/unpaired/{h[:2]}/{h}.npz"] = p.read_bytes()
        t = root / "monomer_templates" / h[:2] / f"{h}.json"
        t.parent.mkdir(parents=True, exist_ok=True)
        t.write_text(json.dumps([{"template_pdb_id": "1xyz", "n": len(files)}]))
        files[f"monomer_templates/{h[:2]}/{h}.json"] = t.read_bytes()
    # A long key (> 100 chars) exercises the PAX header path.
    long = "structures/zz/" + "z" * 120 + ".npz"
    (root / long).parent.mkdir(parents=True, exist_ok=True)
    (root / long).write_bytes(b"long-name-member")
    files[long] = b"long-name-member"
    (root / "manifest.json").write_text('{"records": []}')
    return files


def _deploy(src: Path, dst: Path, **kw) -> dict:
    """Pack ``src`` into ``dst/_pack`` and copy the unpacked small files, like a transfer would."""
    meta = build(src, dst / "_pack", **kw)
    shutil.copy(src / "manifest.json", dst / "manifest.json")
    return meta


def test_parse_size():
    assert parse_size("8G") == 8 << 30
    assert parse_size("512M") == 512 << 20
    assert parse_size("1.5K") == 1536
    assert parse_size(123) == 123


def test_roundtrip_and_coverage(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    files = _df3_tree(src)
    meta = _deploy(src, dst, include=["structures", "msa", "monomer_templates"], shard_size=600, workers=2)
    assert meta["n_files"] == len(files)
    assert meta["n_shards"] > 1  # tiny shard size forces several shards
    assert meta["include"] == ["monomer_templates/", "msa/", "structures/"]
    assert not list((dst / "_pack" / "shards").glob(".*"))  # no temp files left behind

    assert pack.register(dst)
    for rel, data in files.items():
        assert pack.read_bytes(dst / rel) == data
        assert pack.exists(dst / rel) and pack.is_file(dst / rel)

    # Authoritative miss inside a covered prefix: no filesystem fallback.
    missing = dst / "msa" / "unpaired" / "ef" / ("ef" + "2" * 62 + ".npz")
    assert not pack.exists(missing)
    with pytest.raises(FileNotFoundError):
        pack.read_bytes(missing)
    assert not pack.exists(dst / "msa" / "paired" / "ab" / "x.npz")

    # Passthrough: uncovered paths are answered by the filesystem.
    assert pack.read_bytes(dst / "manifest.json") is None
    assert pack.exists(dst / "manifest.json")
    assert pack.read_file(dst / "manifest.json") == b'{"records": []}'
    assert not pack.exists(dst / "masks" / "1a" / "1abc.npz")

    # Directories that exist only inside the pack.
    assert pack.is_dir(dst / "structures") and pack.is_dir(dst / "msa" / "unpaired" / "ab")
    assert pack.is_dir(dst) and not pack.is_dir(dst / "templates")
    assert not (dst / "structures").exists()  # really nothing on disk

    # Loader-level helpers.
    with read_npz(dst / "structures" / "1a" / "1abc.npz") as z:
        assert z["atoms"].shape == (10, 3)
    tpl = next(r for r in files if r.startswith("monomer_templates/"))
    assert pack.read_json(dst / tpl)[0]["template_pdb_id"] == "1xyz"

    assert verify(dst / "_pack", src=src, sample=100, checksums=True, workers=1) == []


def test_shards_are_plain_tar(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    files = _df3_tree(src)
    _deploy(src, dst, include=["structures"], workers=1)
    with tarfile.open(dst / "_pack" / "shards" / pack.shard_name(0)) as tf:
        names = tf.getnames()
        assert sorted(names) == sorted(r for r in files if r.startswith("structures/"))
        assert tf.extractfile("structures/1a/1abc.npz").read() == files["structures/1a/1abc.npz"]


def test_verify_detects_corruption(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    _df3_tree(src)
    _deploy(src, dst, include=["structures"], workers=1)
    shard = dst / "_pack" / "shards" / pack.shard_name(0)
    data = bytearray(shard.read_bytes())
    first = np.load(dst / "_pack" / "loc.npy")[0]
    data[int(first["offset"]) + 10] ^= 0xFF  # inside the first member's data
    shard.write_bytes(bytes(data))
    problems = verify(dst / "_pack", src=src, sample=100, checksums=True, workers=1)
    assert any("checksum mismatch" in p for p in problems)
    assert any("bytes differ" in p for p in problems)


def test_resume_and_plan_guard(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    files = _df3_tree(src)
    out = dst / "_pack"
    assert build(src, out, include=["structures", "msa"], shard_size=600, workers=1, limit_shards=1) is None
    assert not (out / "meta.json").exists()
    meta = build(src, out, include=["structures", "msa"], shard_size=600, workers=1)
    assert meta is not None and meta["n_files"] == sum(r.startswith(("structures/", "msa/")) for r in files)
    # A different request against an unfinished build is refused.
    out2 = tmp_path / "d2" / "_pack"
    build(src, out2, include=["structures"], shard_size=600, workers=1, limit_shards=1)
    with pytest.raises(SystemExit):
        build(src, out2, include=["msa"], shard_size=600, workers=1)


def test_out_must_be_named_pack(tmp_path):
    src = tmp_path / "src"
    _df3_tree(src)
    with pytest.raises(ValueError):
        build(src, tmp_path / "packed", include=["structures"])


def _of3_tree(root: Path, ids: list[str]) -> None:
    rng = np.random.default_rng(1)
    for i, rid in enumerate(ids):
        d = root / rid
        d.mkdir(parents=True)
        np.savez(d / "structure.npz", x=rng.standard_normal(3))
        if i % 2 == 0:  # half lack an alignment
            np.savez(d / "alignment.npz", y=rng.standard_normal(2))
    (root / "ids.txt").write_text("\n".join(ids) + "\n")


def test_ids_mode_whole_tree_rules(tmp_path):
    ids = [f"MGYP{i:012d}" for i in range(6)]
    src, dst = tmp_path / "src", tmp_path / "dst"
    _of3_tree(src, ids)
    meta = build(
        src,
        dst / "_pack",
        ids_file=src / "ids.txt",
        members=["structure.npz", "alignment.npz", "template.npz"],
        workers=1,
    )
    assert meta["n_files"] == 6 + 3 and meta["missing_at_build"] == 6 + 3
    assert meta["include"] == []
    pack.register(dst)
    assert pack.exists(dst / ids[0] / "alignment.npz")
    assert not pack.exists(dst / ids[1] / "alignment.npz")  # covered: same first component
    assert not pack.exists(dst / ids[0] / "template.npz")
    assert pack.is_dir(dst / ids[3])
    # ids.txt is not packed and not covered -> filesystem.
    assert not pack.exists(dst / "ids.txt")
    shutil.copy(src / "ids.txt", dst / "ids.txt")
    assert pack.exists(dst / "ids.txt")


def test_discover_of3_records_from_pack(tmp_path):
    from deepfold3.data.module.of3_dataset import discover_of3_records

    ids = [f"MGYP{i:012d}" for i in range(5)]
    src, dst = tmp_path / "src", tmp_path / "dst"
    _of3_tree(src, ids)
    build(src, dst / "_pack", ids_file=src / "ids.txt", members=["structure.npz", "alignment.npz"], workers=1)
    recs = discover_of3_records(dst)  # no ids.txt in dst: listed from the pack
    assert [r.id for r in recs] == ids
    assert (dst / "ids.txt").read_text().split() == ids  # cached as usual


def _child_read(path: str) -> bytes | None:
    return pack.read_bytes(path)


def test_fork_and_spawn_workers(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    files = _df3_tree(src)
    _deploy(src, dst, include=["structures"], workers=1)
    pack.register(dst)
    rel = "structures/2x/2xyz.npz"
    assert pack.read_bytes(dst / rel) == files[rel]  # parent opens its descriptors first
    for method in ("fork", "spawn"):
        with mp.get_context(method).Pool(2) as pool:
            got = pool.map(_child_read, [str(dst / rel)] * 4)
        assert all(g == files[rel] for g in got), method


def test_pdb_chain_cache_reads_pack_and_never_writes(tmp_path):
    from deepfold3.data.parse.pdb_chain_cache import (
        ChainTemplate,
        PdbChainCache,
        _atomic_write_npz,
        _serialize,
        chain_cache_path,
        pdb_done_path,
    )

    A = const.MAX_ATOMS_PER_TOKEN
    lookup = {i: (i, np.zeros((A, 3), np.float32), np.ones(A, bool)) for i in range(4)}
    ct = ChainTemplate(seqres="ACDE", struct_seq="ACDE", lookup=lookup)
    src, dst = tmp_path / "cache_src", tmp_path / "cache"
    _atomic_write_npz(chain_cache_path(src, "1abc", "A"), **_serialize(ct))
    done = pdb_done_path(src, "1abc")
    done.touch()
    build(src, dst / "_pack", workers=1)

    cache = PdbChainCache(dst, mmcif_dir=None)
    got = cache.get("1abc", "A")
    assert got is not None and got.seqres == "ACDE" and len(got.lookup) == 4
    assert cache.get("1abc", "B") is None  # known miss via the packed _done sentinel
    assert cache._disk_writable is False
    assert sorted(p.name for p in dst.iterdir()) == ["_pack"]


def test_read_pdb_mmcif_local_from_pack(tmp_path):
    from deepfold3.data.parse.templates import read_pdb_mmcif_local

    src, dst = tmp_path / "divided_src", tmp_path / "divided"
    (src / "bc").mkdir(parents=True)
    with gzip.open(src / "bc" / "1bcd.cif.gz", "wt") as f:
        f.write("data_1BCD\n#\n")
    build(src, dst / "_pack", workers=1)
    read_pdb_mmcif_local.cache_clear()
    assert read_pdb_mmcif_local("1BCD", dst) == "data_1BCD\n#\n"
    assert read_pdb_mmcif_local("9xyz", dst) is None


def test_unregistered_paths_untouched(tmp_path):
    p = tmp_path / "plain.json"
    p.write_text("[1]")
    assert pack.read_bytes(p) is None
    assert pack.exists(p) and not pack.is_packed(p)
    assert pack.read_json(p) == [1]
    assert not pack.register(tmp_path)  # no _pack here


def test_exclude_and_size_bounds(tmp_path):
    from deepfold3.data.prepare.pack_build import _bounds_by_size

    src, dst = tmp_path / "src", tmp_path / "dst"
    files = _df3_tree(src)
    (src / "structures" / "1a.tar.gz").write_bytes(b"archive of shard 1a")
    meta = _deploy(src, dst, include=["structures"], exclude=["*.tar.gz"], workers=1)
    assert meta["n_files"] == sum(r.startswith("structures/") for r in files)
    pack.register(dst)
    assert not pack.exists(dst / "structures" / "1a.tar.gz")

    sizes = np.array([100, 5000, 100, 100, 9000, 10], dtype=np.int64)
    b = _bounds_by_size(sizes, 6000)
    assert b[0] == 0 and b[-1] == len(sizes) and np.all(np.diff(b) > 0)
    # each shard's tar footprint stays within one member of the target
    member = 512 + np.ceil(sizes / 512) * 512
    for lo, hi in zip(b[:-1], b[1:], strict=True):
        assert member[lo:hi].sum() <= 6000 + member[lo:hi].max()
