"""Build and verify packs (see :mod:`deepfold3.data.pack` for the format).

Building is three resumable phases, all state under ``{pack}/_build/``:

1. **plan** — list the files (a directory walk under ``include`` prefixes, or
   ``{id}/{member}`` from an id list for the flat OF3 sets), sort them, and
   fix the shard boundaries: by cumulative size for a walk (sizes come with
   the listing), uniformly from a size sample for an id list. Stored as
   ``plan.json`` + ``keys.npy`` + ``bounds.npy``, so a resumed build cuts
   exactly the same shards.
2. **shards** — each shard is an independent task: its slice of the plan is
   streamed into ``shards/{k:05d}.tar`` (tmp file + rename, SHA-256 computed
   while writing) and its index slice saved as ``_build/{k:05d}.idx.npz``.
   A shard with both files is done; re-running skips it. Files that vanish
   between plan and write are recorded and skipped.
3. **finalize** — concatenate the slices (already in global key order) into
   ``keys.npy`` / ``loc.npy``, write ``SHA256SUMS`` and, last, ``meta.json``
   (whose presence is what makes the pack visible to readers).

``_build/`` is only needed to resume or audit a build; it is not read at
train time and need not be copied with the pack.
"""

from __future__ import annotations

import fnmatch
import hashlib
import itertools
import json
import logging
import os
import random
import tarfile
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from deepfold3.data.pack import FORMAT, LOC_DTYPE, PACK_DIRNAME, VERSION, shard_name

logger = logging.getLogger(__name__)

_MAX_FILES_PER_SHARD = 1_000_000
_SIZE_SAMPLE = 4000
_BLOCK = tarfile.BLOCKSIZE


def parse_size(s: str | int) -> int:
    """``"8G"`` / ``"512M"`` / ``"1T"`` / plain bytes -> bytes (binary units)."""
    if isinstance(s, int):
        return s
    s = s.strip().upper().removesuffix("B").removesuffix("I")
    mult = {"K": 1 << 10, "M": 1 << 20, "G": 1 << 30, "T": 1 << 40}
    if s and s[-1] in mult:
        return int(float(s[:-1]) * mult[s[-1]])
    return int(s)


# ---------------------------------------------------------------------------
# Listing
# ---------------------------------------------------------------------------


def _walk(top: str, rel: str) -> list[tuple[str, int]]:
    """Recursive listing of regular files under ``top`` as ``(key, size)``.

    Symlinks are followed; ``rel`` is ``top``'s path relative to the source
    root. Hidden entries (in-flight temp files) and nested ``_pack`` dirs are
    skipped; a realpath guard stops symlink loops. Sizes come from the
    directory entries (NFS ``READDIRPLUS`` usually answers them without a
    per-file round trip).
    """
    out: list[tuple[str, int]] = []
    seen: set[str] = set()
    stack = [(top, rel)]
    while stack:
        d, r = stack.pop()
        real = os.path.realpath(d)
        if real in seen:
            continue
        seen.add(real)
        with os.scandir(d) as it:
            for e in it:
                if e.name.startswith(".") or e.name == PACK_DIRNAME:
                    continue
                child = f"{r}/{e.name}" if r else e.name
                if e.is_dir():
                    stack.append((e.path, child))
                elif e.is_file():
                    out.append((child, e.stat().st_size))
    return out


def _walk_star(a: tuple[str, str]) -> list[tuple[str, int]]:
    return _walk(*a)


def list_tree(src: Path, include: list[str], workers: int) -> list[tuple[str, int]]:
    """All ``(key, size)`` under ``src/{prefix}`` for each include prefix.

    Each prefix's first level of subdirectories is walked in parallel — the
    trees here are 256-way sharded, so that is the natural unit of work.
    """
    tasks: list[tuple[str, str]] = []
    found: list[tuple[str, int]] = []
    for inc in include:
        inc = inc.strip("/")
        p = src / inc
        if p.is_file():
            found.append((inc, p.stat().st_size))
            continue
        if not p.is_dir():
            raise FileNotFoundError(f"include {inc!r}: {p} is neither a file nor a directory")
        with os.scandir(p) as it:
            for e in it:
                if e.name.startswith(".") or e.name == PACK_DIRNAME:
                    continue
                child = f"{inc}/{e.name}" if inc else e.name
                if e.is_dir():
                    tasks.append((e.path, child))
                elif e.is_file():
                    found.append((child, e.stat().st_size))
    with ProcessPoolExecutor(max(1, workers)) as ex:
        for part in ex.map(_walk_star, tasks, chunksize=1):
            found.extend(part)
    return found


def list_ids(ids_file: Path, members: list[str]) -> np.ndarray:
    """``{id}/{member}`` keys for every id in ``ids_file`` (``#`` comments skipped)."""
    with open(ids_file) as f:
        ids = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    if any("\x00" in i or "/" in i for i in ids):
        raise ValueError(f"{ids_file}: ids must not contain '/' or NUL")
    arr = np.array([i.encode() for i in ids], dtype=np.bytes_)
    parts = [np.char.add(np.char.add(arr, b"/"), m.encode()) for m in members]
    return np.concatenate(parts)


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------


def _encode_keys(keys: list[str]) -> np.ndarray:
    bad = [k for k in keys if "\x00" in k]
    if bad:
        raise ValueError(f"keys contain NUL bytes: {bad[:3]}")
    return np.array([k.encode() for k in keys], dtype=np.bytes_)


def _member_bytes(size: np.ndarray | float) -> np.ndarray | float:
    """Bytes a member occupies in the tar: header block + data padded to a block."""
    return _BLOCK + np.ceil(np.asarray(size, dtype=np.float64) / _BLOCK) * _BLOCK


def _bounds_by_size(sizes: np.ndarray, shard_size: int) -> np.ndarray:
    """Shard boundaries ``[0, ..., n]`` cutting where the running tar size crosses ``shard_size``."""
    cum = np.cumsum(_member_bytes(sizes))
    cuts = np.searchsorted(cum, np.arange(shard_size, cum[-1], shard_size), side="right")
    bounds = np.unique(np.concatenate([[0], cuts, [len(sizes)]]))
    # Very small members: also cap the count so a shard's index slice stays small.
    out = [0]
    for b in bounds[1:]:
        while b - out[-1] > _MAX_FILES_PER_SHARD:
            out.append(out[-1] + _MAX_FILES_PER_SHARD)
        out.append(int(b))
    return np.array(out, dtype=np.int64)


def _bounds_by_sample(src: Path, keys: np.ndarray, shard_size: int, seed: int = 0) -> tuple[np.ndarray, float]:
    """Uniform boundaries from a size sample (id lists: no listing gives sizes)."""
    rng = random.Random(seed)
    idx = rng.sample(range(len(keys)), min(_SIZE_SAMPLE, len(keys)))
    sizes = []
    for i in idx:
        try:
            sizes.append(os.stat(src / keys[i].decode()).st_size)
        except FileNotFoundError:
            pass
    if not sizes:
        raise FileNotFoundError(f"none of {len(idx)} sampled files exist under {src}")
    mean = float(np.mean(_member_bytes(np.array(sizes))))
    fps = int(min(_MAX_FILES_PER_SHARD, max(1, shard_size // mean)))
    return np.unique(np.append(np.arange(0, len(keys), fps), len(keys))), mean


def make_plan(
    src: Path,
    out: Path,
    *,
    include: list[str] | None,
    ids_file: Path | None,
    members: list[str] | None,
    shard_size: int,
    workers: int,
    exclude: list[str] | None = None,
) -> dict[str, Any]:
    """Create (or reuse) ``{out}/_build/plan.json`` + ``keys.npy`` + ``bounds.npy``."""
    build = out / "_build"
    plan_path = build / "plan.json"
    request = {
        "src": str(src),
        "include": sorted(i.strip("/") for i in include) if include else [],
        "exclude": sorted(exclude) if exclude else [],
        "ids_file": str(ids_file) if ids_file else None,
        "members": list(members) if members else [],
        "shard_size": int(shard_size),
    }
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        if {k: plan.get(k) for k in request} != request:
            raise SystemExit(
                f"{plan_path} was made with different arguments:\n  plan:    "
                f"{ {k: plan.get(k) for k in request} }\n  request: {request}\n"
                "Use a new --out, or delete its _build/ to start over."
            )
        logger.info("Reusing plan: %d files, %d shards", plan["n_keys"], plan["n_shards"])
        return plan

    build.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if ids_file is not None:
        keys = np.unique(list_ids(ids_file, members or []))
        if keys.size == 0:
            raise ValueError("nothing to pack")
        bounds, mean = _bounds_by_sample(src, keys, shard_size)
        how = f"uniform, mean member {mean:.0f} B from a sample"
    else:
        listed = list_tree(src, request["include"] or [""], workers)
        if exclude:
            n0 = len(listed)
            listed = [kv for kv in listed if not any(fnmatch.fnmatchcase(kv[0], pat) for pat in exclude)]
            logger.info("Excluded %d files matching %s", n0 - len(listed), exclude)
        if not listed:
            raise ValueError("nothing to pack")
        listed.sort(key=lambda kv: kv[0].encode())
        keys = _encode_keys([k for k, _ in listed])
        if len(np.unique(keys)) != len(keys):
            raise ValueError("duplicate keys in listing (overlapping --include?)")
        sizes = np.array([s for _, s in listed], dtype=np.int64)
        bounds = _bounds_by_size(sizes, shard_size)
        how = f"by size, {sizes.sum() / 2**30:.1f} GiB listed"
    logger.info("Listed %d files in %.0fs", len(keys), time.time() - t0)
    for name, arr in (("keys.npy", keys), ("bounds.npy", bounds)):
        tmp = build / f".{name}.tmp.npy"
        np.save(tmp, arr)
        os.replace(tmp, build / name)
    plan = {
        **request,
        "n_keys": int(len(keys)),
        "n_shards": int(len(bounds) - 1),
        "sharding": how,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    tmpj = build / ".plan.json.tmp"
    tmpj.write_text(json.dumps(plan, indent=2))
    os.replace(tmpj, plan_path)
    logger.info("Plan: %d files -> %d shards (%s)", len(keys), plan["n_shards"], how)
    return plan


# ---------------------------------------------------------------------------
# Shards
# ---------------------------------------------------------------------------


def shard_done(out: Path, k: int) -> bool:
    return (out / "_build" / f"{k:05d}.idx.npz").exists() and (out / "shards" / shard_name(k)).exists()


def _read_member(path: str) -> tuple[bytes, float] | None:
    try:
        with open(path, "rb") as f:
            mtime = os.fstat(f.fileno()).st_mtime
            return f.read(), mtime
    except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
        return None


def write_shard(src: str, out: str, k: int, lo: int, hi: int, io_threads: int = 16) -> dict[str, Any]:
    """Write shard ``k`` from plan keys ``[lo, hi)``. Module-level for picklability.

    Source reads are latency-bound on NFS (one round trip per open/read), so
    ``io_threads`` threads read ahead in a bounded window while this thread
    writes members in key order. Members are written as standard PAX tar
    entries (``TarInfo.tobuf``), each file read whole with one ``read``.
    """
    srcp, outp = Path(src), Path(out)
    keys = np.load(outp / "_build" / "keys.npy", mmap_mode="r")[lo:hi]
    final = outp / "shards" / shard_name(k)
    tmp = outp / "shards" / f".{shard_name(k)}.tmp{os.getpid()}"
    got_keys: list[bytes] = []
    offsets: list[int] = []
    sizes: list[int] = []
    missing: list[bytes] = []
    h = hashlib.sha256()
    pos = 0

    def put(b: bytes) -> None:
        nonlocal pos
        f.write(b)
        h.update(b)
        pos += len(b)

    window = max(4, io_threads * 4)
    with open(tmp, "wb", buffering=16 << 20) as f, ThreadPoolExecutor(max(1, io_threads)) as pool:
        pending: deque = deque()
        it = iter(keys)
        for kb in itertools.islice(it, window):
            kb = bytes(kb)
            pending.append((kb, pool.submit(_read_member, str(srcp / kb.decode()))))
        while pending:
            kb, fut = pending.popleft()
            nxt = next(it, None)
            if nxt is not None:
                nb = bytes(nxt)
                pending.append((nb, pool.submit(_read_member, str(srcp / nb.decode()))))
            res = fut.result()
            if res is None:
                missing.append(kb)
                continue
            data, mtime = res
            ti = tarfile.TarInfo(kb.decode())
            ti.size = len(data)
            ti.mtime = int(mtime)
            ti.mode = 0o644
            put(ti.tobuf(tarfile.PAX_FORMAT, "utf-8", "surrogateescape"))
            got_keys.append(kb)
            offsets.append(pos)
            sizes.append(len(data))
            put(data)
            if len(data) % _BLOCK:
                put(b"\0" * (_BLOCK - len(data) % _BLOCK))
        put(b"\0" * (2 * _BLOCK))  # end-of-archive marker
        if pos % tarfile.RECORDSIZE:
            put(b"\0" * (tarfile.RECORDSIZE - pos % tarfile.RECORDSIZE))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, final)
    width = max((len(x) for x in got_keys), default=1)
    idx_tmp = outp / "_build" / f".{k:05d}.idx.tmp{os.getpid()}.npz"
    np.savez(
        idx_tmp,
        keys=np.array(got_keys, dtype=f"S{width}"),
        offset=np.array(offsets, dtype=np.uint64),
        size=np.array(sizes, dtype=np.uint64),
        missing=np.array(missing, dtype=np.bytes_) if missing else np.empty(0, dtype="S1"),
        sha256=np.array(h.hexdigest()),
        nbytes=np.array(pos, dtype=np.uint64),
    )
    os.replace(idx_tmp, outp / "_build" / f"{k:05d}.idx.npz")
    return {"shard": k, "files": len(got_keys), "missing": len(missing), "bytes": pos}


def _clear_stale_tmp(out: Path) -> None:
    """Remove temp files a killed run left behind (they never become shards)."""
    for d in (out / "shards", out / "_build"):
        if d.is_dir():
            for e in os.scandir(d):
                if e.name.startswith(".") and ".tmp" in e.name:
                    os.unlink(e.path)


def write_shards(
    src: Path, out: Path, plan: dict[str, Any], workers: int, limit: int | None = None, io_threads: int = 16
) -> None:
    (out / "shards").mkdir(parents=True, exist_ok=True)
    _clear_stale_tmp(out)
    bounds = np.load(out / "_build" / "bounds.npy")
    n_shards = plan["n_shards"]
    pending = [k for k in range(n_shards) if not shard_done(out, k)]
    todo = pending[:limit] if limit is not None else pending
    logger.info("Shards: %d total, %d done, %d to write now", n_shards, n_shards - len(pending), len(todo))
    if not todo:
        return
    t0 = time.time()
    done_bytes = done_files = n_missing = 0
    with ProcessPoolExecutor(max(1, workers)) as ex:
        futs = [
            ex.submit(write_shard, str(src), str(out), k, int(bounds[k]), int(bounds[k + 1]), io_threads) for k in todo
        ]
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            done_bytes += r["bytes"]
            done_files += r["files"]
            n_missing += r["missing"]
            dt = time.time() - t0
            logger.info(
                "[%d/%d] shard %05d: %d files%s, %.2f GiB | total %.1f GiB, %.0f files/s, %.0f MiB/s",
                i,
                len(todo),
                r["shard"],
                r["files"],
                f" ({r['missing']} missing)" if r["missing"] else "",
                r["bytes"] / 2**30,
                done_bytes / 2**30,
                done_files / dt,
                done_bytes / 2**20 / dt,
            )
    if n_missing:
        logger.warning("%d listed files were missing at write time (recorded in _build/*.idx.npz)", n_missing)


# ---------------------------------------------------------------------------
# Finalize
# ---------------------------------------------------------------------------


def finalize(out: Path, plan: dict[str, Any]) -> dict[str, Any]:
    """Merge shard index slices into ``keys.npy`` / ``loc.npy`` and write ``meta.json``."""
    n_shards = plan["n_shards"]
    pending = [k for k in range(n_shards) if not shard_done(out, k)]
    if pending:
        raise SystemExit(f"{len(pending)} shards not written yet (first: {pending[:5]}); run the build again")
    key_parts, loc_parts, sums, missing = [], [], [], []
    total = 0
    for k in range(n_shards):
        with np.load(out / "_build" / f"{k:05d}.idx.npz") as z:
            kk = z["keys"]
            loc = np.empty(len(kk), dtype=LOC_DTYPE)
            loc["shard"] = k
            loc["offset"] = z["offset"]
            loc["size"] = z["size"]
            key_parts.append(kk)
            loc_parts.append(loc)
            sums.append(f"{z['sha256']}  shards/{shard_name(k)}")
            total += int(z["nbytes"])
            missing.extend(x.decode() for x in z["missing"] if x)
    width = max(p.dtype.itemsize for p in key_parts)
    keys = np.concatenate([p.astype(f"S{width}") for p in key_parts])
    loc = np.concatenate(loc_parts)
    if len(keys) > 1 and not bool(np.all(keys[1:] > keys[:-1])):
        raise RuntimeError("merged keys are not strictly increasing — shard slices out of order?")
    for name, arr in (("keys.npy", keys), ("loc.npy", loc)):
        tmp = out / f".{name}.tmp.npy"
        np.save(tmp, arr)
        os.replace(tmp, out / name)
    (out / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    if missing:
        (out / "_build" / "missing.txt").write_text("\n".join(missing) + "\n")
    meta = {
        "format": FORMAT,
        "version": VERSION,
        "n_files": int(len(keys)),
        "n_shards": int(n_shards),
        "total_bytes": int(total),
        "key_width": int(width),
        "include": [p.rstrip("/") + "/" if p else "" for p in plan["include"]] if plan["include"] else [],
        "source": plan["src"],
        "ids_file": plan["ids_file"],
        "members": plan["members"],
        "missing_at_build": len(missing),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    # An empty-string include means "whole tree": store none so readers use the
    # first-path-component rule.
    meta["include"] = [p for p in meta["include"] if p]
    tmp = out / ".meta.json.tmp"
    tmp.write_text(json.dumps(meta, indent=2))
    os.replace(tmp, out / "meta.json")
    return meta


def build(
    src: Path,
    out: Path,
    *,
    include: list[str] | None = None,
    ids_file: Path | None = None,
    members: list[str] | None = None,
    shard_size: int | str = "8G",
    workers: int = 16,
    limit_shards: int | None = None,
    exclude: list[str] | None = None,
    io_threads: int = 16,
) -> dict[str, Any] | None:
    """Plan → shards → finalize. Returns ``meta`` once complete, else ``None``."""
    src, out = Path(src), Path(out)
    if out.name != PACK_DIRNAME:
        raise ValueError(f"--out must be a directory named {PACK_DIRNAME!r} (it sits next to the data it replaces)")
    if (out / "meta.json").exists():
        logger.info("%s is already finalized", out)
        return json.loads((out / "meta.json").read_text())
    plan = make_plan(
        src,
        out,
        include=include,
        ids_file=ids_file,
        members=members,
        shard_size=parse_size(shard_size),
        workers=workers,
        exclude=exclude,
    )
    write_shards(src, out, plan, workers, limit_shards, io_threads)
    if any(not shard_done(out, k) for k in range(plan["n_shards"])):
        logger.info("Stopped before all shards were written; re-run to continue")
        return None
    return finalize(out, plan)


# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------


def _sha256_file(path: str) -> tuple[str, str]:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(16 << 20):
            h.update(chunk)
    return path, h.hexdigest()


def verify(
    pack_dir: Path,
    *,
    src: Path | None = None,
    sample: int = 1000,
    checksums: bool = False,
    workers: int = 8,
    seed: int = 0,
) -> list[str]:
    """Check a pack; return a list of problems (empty = OK).

    Always: meta/index consistency, keys strictly sorted, every shard present
    and long enough, and for ``sample`` random entries the tar header right
    before the data names the key. With ``src``: those entries' bytes equal
    the source files. With ``checksums``: every shard's SHA-256 matches
    ``SHA256SUMS`` (reads the whole pack).
    """
    from deepfold3.data.pack import PackReader

    problems: list[str] = []
    rd = PackReader(pack_dir)
    keys = np.load(pack_dir / "keys.npy", mmap_mode="r")
    loc = np.load(pack_dir / "loc.npy", mmap_mode="r")
    n = len(keys)
    if n != rd.meta["n_files"] or len(loc) != n:
        problems.append(f"count mismatch: meta {rd.meta['n_files']}, keys {n}, loc {len(loc)}")
    step = 1 << 22
    for lo in range(0, n - 1, step):
        a = np.asarray(keys[lo : min(lo + step + 1, n)])
        if not bool(np.all(a[1:] > a[:-1])):
            problems.append(f"keys not strictly increasing near index {lo}")
            break
    n_shards = int(rd.meta["n_shards"])
    ends = np.zeros(n_shards, dtype=np.uint64)
    for lo in range(0, n, step):
        c = np.asarray(loc[lo : lo + step])
        if int(c["shard"].max(initial=0)) >= n_shards:
            problems.append("loc references a shard beyond n_shards")
            break
        np.maximum.at(ends, c["shard"], c["offset"] + c["size"])
    for k in range(n_shards):
        p = pack_dir / "shards" / shard_name(k)
        if not p.exists():
            problems.append(f"missing shard {p.name}")
        elif p.stat().st_size < int(ends[k]):
            problems.append(f"shard {p.name} is {p.stat().st_size} B, index needs {int(ends[k])}")

    rng = random.Random(seed)
    idx = sorted(rng.sample(range(n), min(sample, n))) if n else []
    for i in idx:
        key = bytes(keys[i]).decode()
        rec = loc[i]
        shard, off = int(rec["shard"]), int(rec["offset"])
        with open(pack_dir / "shards" / shard_name(shard), "rb") as f:
            f.seek(off - _BLOCK)
            hdr = f.read(_BLOCK)
        name = hdr[:100].rstrip(b"\x00").decode("utf-8", "replace")
        size_field = hdr[124:136].rstrip(b"\x00 ").decode("ascii", "replace")
        if not key.startswith(name) or not name:
            problems.append(f"{key}: tar header before data names {name!r}")
            continue
        try:
            if int(size_field, 8) != int(rec["size"]):
                problems.append(f"{key}: tar size {size_field} != index {int(rec['size'])}")
        except ValueError:
            pass  # base-256 / pax-overridden size; covered by the byte compare
        if src is not None:
            try:
                with open(src / key, "rb") as f:
                    want = f.read()
            except FileNotFoundError:
                problems.append(f"{key}: missing in source {src}")
                continue
            if rd.read_index(i) != want:
                problems.append(f"{key}: bytes differ from source")

    if checksums:
        expect = {}
        for ln in (pack_dir / "SHA256SUMS").read_text().splitlines():
            h, _, p = ln.partition("  ")
            expect[str(pack_dir / p)] = h
        with ProcessPoolExecutor(max(1, workers)) as ex:
            for path, h in ex.map(_sha256_file, sorted(expect)):
                if h != expect[path]:
                    problems.append(f"checksum mismatch: {path}")
    return problems
