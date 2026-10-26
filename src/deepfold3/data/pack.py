"""Packed datasets: many small files in a few large tar shards + a sorted index.

Parallel filesystems cap the number of files per account (Hangang's Lustre
``/scratch``: 5M per group) while the training sets are tens of millions of
small NPZ / JSON files (``long_monomers`` alone is 48M). A *pack* stores a
directory tree as a few thousand uncompressed tar shards and an index that
maps each file's path to ``(shard, offset, size)``, so a read is one binary
search plus one ``pread``. Members are the original bytes, untouched — an NPZ
is still the same NPZ, and every shard is a valid tar (``tar tf``/``tar xf``
work for inspection or recovery).

Layout of a pack for a data root ``{root}``::

    {root}/_pack/
    ├── meta.json          format, version, counts, ``include`` prefixes
    ├── keys.npy           sorted fixed-width ``S{W}`` paths, relative to {root}
    ├── loc.npy            per key: shard (u4), offset (u8), size (u8)
    ├── SHA256SUMS         per shard, written by the packer
    └── shards/{k:05d}.tar

**Path transparency.** Loaders keep building ordinary paths
(``structure_path(data_dir / "structures", id)``); the pack is consulted
underneath. A root is *registered* (``register(root)``) when a dataset is
constructed, and afterwards :func:`read_bytes` / :func:`exists` /
:func:`is_dir` answer for any path under it. Registered roots are mirrored
into ``$DEEPFOLD3_PACK_ROOTS`` so spawned DataLoader workers pick them up;
forked workers inherit the registry directly. Each reader reopens its file
descriptors after a fork.

**Coverage.** A path under a registered root that is not in the index is
either an *authoritative miss* (the pack covers that part of the tree, so the
file does not exist — no filesystem ``stat``) or a *passthrough* (the pack
does not cover it, so the filesystem answers). With ``include`` prefixes in
``meta.json`` (``["structures/", "msa/"]``) the prefixes decide; a whole-tree
pack (empty ``include``) covers a path when any packed key shares its first
path component. Small top-level files — ``manifest.json``, ``ids.txt``,
``df3_cache/`` — therefore stay ordinary files next to ``_pack/``.

Written by ``deepfold3 pack``; checked by ``deepfold3 pack-verify``.
"""

from __future__ import annotations

import io
import json
import os
import threading
from pathlib import Path
from typing import Any

import numpy as np

PACK_DIRNAME = "_pack"
FORMAT = "deepfold3-pack"
VERSION = 1
ENV_ROOTS = "DEEPFOLD3_PACK_ROOTS"

LOC_DTYPE = np.dtype([("shard", "<u4"), ("offset", "<u8"), ("size", "<u8")])

_MISS = -1  # authoritative miss: covered by the pack, not in it
_PASS = -2  # not covered: the filesystem answers

_HAS_FADVISE = hasattr(os, "posix_fadvise")


def shard_name(k: int) -> str:
    """File name of shard ``k`` inside ``{pack}/shards/``."""
    return f"{k:05d}.tar"


class PackReader:
    """Random-access reader over one pack. Cheap to construct; opens lazily.

    Index arrays are memory-mapped (shared page cache across workers on a
    node); shard file descriptors are opened on first use and reopened after
    a fork so workers never share a file offset or a stale descriptor.
    """

    def __init__(self, pack_dir: Path):
        self.pack_dir = Path(pack_dir)
        with open(self.pack_dir / "meta.json") as f:
            self.meta: dict[str, Any] = json.load(f)
        if self.meta.get("format") != FORMAT:
            raise ValueError(f"{self.pack_dir}: not a {FORMAT} (format={self.meta.get('format')!r})")
        if int(self.meta.get("version", 0)) > VERSION:
            raise ValueError(f"{self.pack_dir}: pack version {self.meta['version']} is newer than supported {VERSION}")
        self.include: tuple[str, ...] = tuple(self.meta.get("include") or ())
        self._pid: int | None = None
        self._keys: np.ndarray | None = None
        self._loc: np.ndarray | None = None
        self._fds: dict[int, int] = {}
        self._lock = threading.Lock()

    # -- index ------------------------------------------------------------

    def _ensure_open(self) -> None:
        pid = os.getpid()
        if self._pid == pid:
            return
        with self._lock:
            if self._pid == pid:
                return
            # After a fork the inherited descriptors belong to the parent's
            # bookkeeping; drop them (without closing — the parent still uses
            # them) and start fresh.
            self._fds = {}
            self._keys = np.load(self.pack_dir / "keys.npy", mmap_mode="r")
            self._loc = np.load(self.pack_dir / "loc.npy", mmap_mode="r")
            self._pid = pid

    @property
    def keys(self) -> np.ndarray:
        self._ensure_open()
        return self._keys  # type: ignore[return-value]

    def __len__(self) -> int:
        return int(self.keys.shape[0])

    def find(self, key: str) -> int:
        """Index of ``key`` in the pack, or ``-1``."""
        keys = self.keys
        k = key.encode()
        if len(k) > keys.dtype.itemsize or not k:
            return -1
        i = int(np.searchsorted(keys, k))
        if i < keys.shape[0] and keys[i] == k:
            return i
        return -1

    def has_prefix(self, prefix: str) -> bool:
        """Whether any key starts with ``prefix``."""
        keys = self.keys
        p = prefix.encode()
        i = int(np.searchsorted(keys, p))
        return i < keys.shape[0] and bytes(keys[i]).startswith(p)

    def covers(self, key: str) -> bool:
        """Whether ``key`` lies in the part of the tree this pack is authoritative for."""
        if self.include:
            return any(key == p.rstrip("/") or key.startswith(p if p.endswith("/") else p + "/") for p in self.include)
        first = key.split("/", 1)[0]
        return self.has_prefix(first + "/")

    def status(self, key: str) -> int:
        """Index of ``key``, ``_MISS`` (covered but absent) or ``_PASS`` (not covered)."""
        i = self.find(key)
        if i >= 0:
            return i
        return _MISS if self.covers(key) else _PASS

    def is_dir(self, key: str) -> bool:
        if key in ("", "."):
            return True
        return self.has_prefix(key.rstrip("/") + "/")

    # -- data -------------------------------------------------------------

    def _fd(self, shard: int) -> int:
        fd = self._fds.get(shard)
        if fd is None:
            fd = os.open(self.pack_dir / "shards" / shard_name(shard), os.O_RDONLY)
            self._fds[shard] = fd
        return fd

    def read_index(self, i: int, *, drop_cache: bool = False) -> bytes:
        """Bytes of entry ``i``."""
        self._ensure_open()
        rec = self._loc[i]  # type: ignore[index]
        shard, offset, size = int(rec["shard"]), int(rec["offset"]), int(rec["size"])
        fd = self._fd(shard)
        buf = bytearray(size)
        view = memoryview(buf)
        got = 0
        while got < size:  # pread may return short (and caps one call at ~2 GiB)
            n = os.preadv(fd, [view[got:]], offset + got)
            if n <= 0:
                raise OSError(f"short read in {self.pack_dir}/shards/{shard_name(shard)} at {offset + got}")
            got += n
        if drop_cache and _HAS_FADVISE:
            try:
                os.posix_fadvise(fd, offset, size, os.POSIX_FADV_DONTNEED)
            except OSError:
                pass
        return bytes(buf)

    def read(self, key: str, *, drop_cache: bool = False) -> bytes:
        i = self.find(key)
        if i < 0:
            raise FileNotFoundError(f"{key} not in pack {self.pack_dir}")
        return self.read_index(i, drop_cache=drop_cache)

    def iter_keys_with_suffix(self, suffix: str) -> np.ndarray:
        """All keys ending in ``suffix`` (vectorised; for one-off listings)."""
        keys = self.keys
        out = []
        step = 1 << 22
        s = suffix.encode()
        for lo in range(0, keys.shape[0], step):
            chunk = np.asarray(keys[lo : lo + step])
            out.append(chunk[np.char.endswith(chunk, s)])
        return np.concatenate(out) if out else np.empty(0, dtype=keys.dtype)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_readers: dict[str, PackReader] = {}  # abs root (no trailing slash) -> reader
_roots_sorted: list[str] = []  # longest first, so nested roots resolve innermost
_env_seen: str | None = None
_reg_lock = threading.Lock()


def _abs(path: str | os.PathLike) -> str:
    return os.path.abspath(os.fspath(path))


def register(root: str | os.PathLike | None) -> bool:
    """Register ``root`` if it holds a pack (``{root}/_pack/meta.json``).

    Idempotent and cheap (one ``stat`` per new root). Returns whether the
    root is packed.
    """
    if root is None:
        return False
    _load_env()
    r = _abs(root).rstrip("/")
    if r in _readers:
        return True
    pack_dir = os.path.join(r, PACK_DIRNAME)
    if not os.path.isfile(os.path.join(pack_dir, "meta.json")):
        return False
    with _reg_lock:
        if r not in _readers:
            _readers[r] = PackReader(Path(pack_dir))
            _roots_sorted[:] = sorted(_readers, key=len, reverse=True)
            env = [p for p in os.environ.get(ENV_ROOTS, "").split(os.pathsep) if p]
            if r not in env:
                env.append(r)
                os.environ[ENV_ROOTS] = os.pathsep.join(env)
    return True


def _load_env() -> None:
    """Pick up roots registered by a parent process (spawned workers)."""
    global _env_seen
    env = os.environ.get(ENV_ROOTS, "")
    if env == _env_seen:
        return
    _env_seen = env
    for r in env.split(os.pathsep):
        if r and r not in _readers:
            register(r)


def registered_roots() -> list[str]:
    _load_env()
    return list(_roots_sorted)


def clear_registry() -> None:
    """Forget every registered root (tests)."""
    global _env_seen
    with _reg_lock:
        _readers.clear()
        _roots_sorted.clear()
        os.environ.pop(ENV_ROOTS, None)
        _env_seen = None


def _resolve(path: str | os.PathLike) -> tuple[PackReader, str] | None:
    if not _readers and not os.environ.get(ENV_ROOTS):
        return None
    _load_env()
    p = _abs(path)
    for r in _roots_sorted:
        if p.startswith(r) and (len(p) == len(r) or p[len(r)] == "/"):
            return _readers[r], p[len(r) + 1 :]
    return None


def reader_for(root: str | os.PathLike) -> PackReader | None:
    """The reader registered for exactly ``root`` (after trying to register it)."""
    if not register(root):
        return None
    return _readers[_abs(root).rstrip("/")]


# ---------------------------------------------------------------------------
# Path-level API used by the loaders
# ---------------------------------------------------------------------------


def is_packed(path: str | os.PathLike) -> bool:
    """Whether ``path`` lies under a registered pack root."""
    return _resolve(path) is not None


def read_bytes(path: str | os.PathLike, *, drop_cache: bool = False) -> bytes | None:
    """Bytes of ``path`` from its pack, or ``None`` when no pack covers it.

    Raises:
        FileNotFoundError: the pack covers ``path`` and it is not in it.
    """
    hit = _resolve(path)
    if hit is None:
        return None
    rd, key = hit
    st = rd.status(key)
    if st == _PASS:
        return None
    if st == _MISS:
        raise FileNotFoundError(f"{path} (not in pack {rd.pack_dir})")
    return rd.read_index(st, drop_cache=drop_cache)


def read_file(path: str | os.PathLike) -> bytes:
    """Bytes of ``path`` from its pack or, when not packed, from the filesystem."""
    b = read_bytes(path)
    if b is not None:
        return b
    with open(path, "rb") as f:
        return f.read()


def read_json(path: str | os.PathLike) -> Any:
    """``json.load`` that reads through the pack."""
    return json.loads(read_file(path))


def open_binary(path: str | os.PathLike) -> io.BufferedIOBase:
    """A readable binary file object for ``path`` (in-memory when packed)."""
    b = read_bytes(path)
    if b is not None:
        return io.BytesIO(b)
    return open(path, "rb")


def exists(path: str | os.PathLike) -> bool:
    """``Path.exists`` that answers packed paths from the index (no ``stat``)."""
    hit = _resolve(path)
    if hit is None:
        return os.path.exists(path)
    rd, key = hit
    if key == "":
        return True
    st = rd.status(key)
    if st >= 0:
        return True
    if st == _MISS:
        return rd.is_dir(key)
    return os.path.exists(path) or rd.is_dir(key)


def is_file(path: str | os.PathLike) -> bool:
    hit = _resolve(path)
    if hit is None:
        return os.path.isfile(path)
    rd, key = hit
    st = rd.status(key)
    if st >= 0:
        return True
    return st == _PASS and os.path.isfile(path)


def is_dir(path: str | os.PathLike) -> bool:
    """``Path.is_dir`` that also sees directories that exist only inside a pack."""
    hit = _resolve(path)
    if hit is None:
        return os.path.isdir(path)
    rd, key = hit
    return rd.is_dir(key) or os.path.isdir(path)
