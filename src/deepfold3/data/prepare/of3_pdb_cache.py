"""OpenFold 3 ``pdb_training_set`` dataset cache → columnar DeepFold3 cache.

OpenFold 3 ships the weighted-PDB training set's metadata as one giant JSON
(``dataset_caches/training_cache_with_templates.json``, ~1.7 GB): per structure a
release date + resolution, per chain the molecule type / entity / MSA
representative / template list / sequence-cluster, and per interface its cluster.
Upstream reads it either whole (tens of GB resident) or through an **LMDB**
key-value store. Neither is usable here: every DDP rank × dataloader worker would
pay it, and we do not want an LMDB dependency in the data path.

This module replaces both with a one-shot offline conversion to a directory of
plain ``.npy`` **columns** which :class:`OF3PdbCache` opens with
``mmap_mode="r"``. Startup is a handful of ``open()`` calls, resident cost is
whatever pages are touched, and forked workers share those pages through the
page cache. The whole cache is ~50 MB of columns for the full set.

The JSON is parsed **streaming** (:func:`iter_structure_entries`) so the
conversion itself never holds more than one structure entry plus the growing
output columns in memory.

Layout written under ``out_dir``:

===========================  =======  ===========================================
file                         dtype    contents
===========================  =======  ===========================================
``meta.json``                --       counts, source path/size/mtime, version
``struct_id``                ``S12``  PDB id, lowercase
``struct_release``           ``S10``  ISO release date (``b""`` if unknown)
``struct_resolution``        ``f4``   NaN if unknown
``struct_chain_off``         ``i8``   [S+1] offsets into the chain columns
``struct_iface_off``         ``i8``   [S+1] offsets into the interface columns
``chain_key``                ``S8``   OF3 chain key (``b"1"``) — joins to the
                                      structure NPZ's ``chain_id`` annotation
``chain_label``              ``S8``   ``label_asym_id`` (chain letter)
``chain_entity``             ``i4``   entity id
``chain_moltype``            ``i1``   **DeepFold3** ``mol_type``
``chain_aln_rep``            ``S16``  ``alignment_representative_id`` (``b""``
                                      when the chain has no MSA: ligands, DNA)
``chain_cluster``            ``i4``   sequence-cluster code (interned)
``chain_tmpl_off``           ``i8``   [C+1] offsets into ``tmpl_id``
``tmpl_id``                  ``S16``  ``{pdb}_{chain}`` template ids
``iface_a`` / ``iface_b``    ``i4``   **structure-local** chain indices
===========================  =======  ===========================================

Only the *conversion logic* is original — no OpenFold 3 code is copied. The JSON
field names are treated as a data format specification.
"""

from __future__ import annotations

import json
import logging
from array import array
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import IO

import numpy as np

from deepfold3.data import const

logger = logging.getLogger(__name__)

CACHE_VERSION = 1

# OF3 ``MoleculeType`` name → DeepFold3 ``mol_type``. The dataset cache stores the
# *name* ("PROTEIN"), unlike the structure NPZ which stores the int.
_MOL_NAME_TO_DF3 = {
    "PROTEIN": const.MOL_PROTEIN,
    "RNA": const.MOL_RNA,
    "DNA": const.MOL_DNA,
    "LIGAND": const.MOL_NONPOLYMER,
}

# Fixed widths for the string columns. Overflow raises at build time rather than
# silently truncating an id — a truncated ``alignment_representative_id`` would
# point at the wrong MSA file and only show up as a quiet accuracy loss.
_W_STRUCT_ID = 12
_W_RELEASE = 10
_W_CHAIN_KEY = 8
_W_CHAIN_LABEL = 8
_W_ALN_REP = 16
_W_TMPL_ID = 16


# ---------------------------------------------------------------------------
# Streaming JSON reader
# ---------------------------------------------------------------------------


class _JsonCursor:
    """Incremental JSON reader over a text stream.

    Only what is needed here: seek to a top-level key, then decode the values of
    an object one at a time. ``json.JSONDecoder.raw_decode`` does the parsing;
    this class just keeps the buffer topped up so a value that straddles a read
    boundary is retried with more input instead of failing.
    """

    def __init__(self, fh: IO[str], chunk_size: int = 1 << 22):
        self._fh = fh
        self._chunk = chunk_size
        self._dec = json.JSONDecoder()
        self._buf = ""
        self._pos = 0
        self._eof = False

    def _fill(self) -> bool:
        """Append one chunk; drop the consumed prefix. False at EOF."""
        if self._pos:
            self._buf = self._buf[self._pos :]
            self._pos = 0
        if self._eof:
            return False
        data = self._fh.read(self._chunk)
        if not data:
            self._eof = True
            return False
        self._buf += data
        return True

    def peek(self) -> str:
        """Next non-whitespace character, or ``""`` at EOF. Does not consume."""
        while True:
            while self._pos < len(self._buf) and self._buf[self._pos] in " \t\r\n":
                self._pos += 1
            if self._pos < len(self._buf):
                return self._buf[self._pos]
            if not self._fill():
                return ""

    def take(self, char: str) -> None:
        c = self.peek()
        if c != char:
            raise ValueError(f"expected {char!r} in dataset cache JSON, got {c!r}")
        self._pos += 1

    def skip_if(self, char: str) -> bool:
        if self.peek() == char:
            self._pos += 1
            return True
        return False

    def decode(self):
        """Decode one JSON value at the cursor.

        ``raw_decode`` does not skip leading whitespace, so ``peek`` runs first —
        it both skips it and tops the buffer up. A ``ValueError`` means the value
        is only partially buffered; re-read and retry until EOF makes it real.
        """
        while True:
            self.peek()
            try:
                obj, end = self._dec.raw_decode(self._buf, self._pos)
            except ValueError:
                if not self._fill():
                    raise
                continue
            self._pos = end
            return obj

    def seek_key(self, key: str) -> None:
        """Advance to just past ``"key":`` at the top level of the root object."""
        needle = f'"{key}"'
        search_from = self._pos
        while True:
            found = self._buf.find(needle, search_from)
            if found >= 0:
                self._pos = found + len(needle)
                self.take(":")
                return
            # Resume len(needle)-1 bytes back so a needle split across the chunk
            # boundary is still found. ``_fill`` drops everything before
            # ``self._pos``, which shifts buffer offsets down by that much.
            resume = max(self._pos, len(self._buf) - len(needle) + 1)
            search_from = resume - self._pos
            if not self._fill():
                raise KeyError(f"{key!r} not found in dataset cache JSON")


def iter_structure_entries(cache_path: Path) -> Iterator[tuple[str, dict]]:
    """Yield ``(pdb_id, entry)`` from an OF3 dataset cache JSON, streaming.

    Memory stays bounded at one chunk plus one structure entry regardless of the
    file size. Bare ``NaN`` literals (which the OF3 caches contain for missing
    resolutions) decode fine — Python's decoder accepts them by default.
    """
    with open(cache_path, encoding="utf-8") as fh:
        cur = _JsonCursor(fh)
        cur.seek_key("structure_data")
        cur.take("{")
        if cur.skip_if("}"):
            return
        while True:
            pdb_id = cur.decode()
            cur.take(":")
            entry = cur.decode()
            yield str(pdb_id), entry
            if cur.skip_if(","):
                continue
            cur.take("}")
            return


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def _fixed(values: list[bytes], width: int, column: str) -> np.ndarray:
    too_long = [v for v in values if len(v) > width]
    if too_long:
        raise ValueError(
            f"{column}: {len(too_long)} value(s) exceed the {width}-byte column width "
            f"(e.g. {too_long[0]!r}). Widen the column in of3_pdb_cache.py."
        )
    return np.array(values, dtype=f"S{width}")


def build_of3_pdb_cache(
    cache_path: Path,
    out_dir: Path,
    *,
    keep_templates: bool = True,
    progress_every: int = 50_000,
) -> dict:
    """Convert an OF3 ``ClusteredDatasetCache`` JSON to the columnar cache.

    Args:
        cache_path: ``dataset_caches/training_cache_with_templates.json`` (or the
            validation / a subset cache — the ``ClusteredDatasetCache`` and
            ``ValidationDatasetCache`` layouts agree on every field read here).
        out_dir: Directory to write the columns into (created if absent).
        keep_templates: Write the per-chain template id lists. ``False`` skips
            them (smaller cache, template-free training).
        progress_every: Log a progress line every N structures.

    Returns:
        The ``meta.json`` dict that was written.
    """
    cache_path, out_dir = Path(cache_path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    struct_id: list[bytes] = []
    struct_release: list[bytes] = []
    struct_resolution = array("f")
    struct_chain_off = array("q", [0])
    struct_iface_off = array("q", [0])

    chain_key: list[bytes] = []
    chain_label: list[bytes] = []
    chain_entity = array("i")
    chain_moltype = array("b")
    chain_aln_rep: list[bytes] = []
    chain_cluster = array("i")
    chain_tmpl_off = array("q", [0])
    tmpl_id: list[bytes] = []

    iface_a = array("i")
    iface_b = array("i")

    cluster_codes: dict[str, int] = {}
    n_skipped = 0

    for n_done, (pdb_id, entry) in enumerate(iter_structure_entries(cache_path), start=1):
        chains = entry.get("chains") or {}
        if not chains:
            n_skipped += 1
            continue

        local_index: dict[str, int] = {}
        for key, cd in chains.items():
            local_index[str(key)] = len(chain_key) - struct_chain_off[-1]
            chain_key.append(str(key).encode("ascii", "replace"))
            chain_label.append(str(cd.get("label_asym_id") or key).encode("ascii", "replace"))
            chain_entity.append(int(cd.get("entity_id") or 0))
            chain_moltype.append(_MOL_NAME_TO_DF3.get(str(cd.get("molecule_type")), const.MOL_NONPOLYMER))
            rep = cd.get("alignment_representative_id")
            chain_aln_rep.append(b"" if rep is None else str(rep).encode("ascii", "replace"))

            cid = cd.get("cluster_id")
            cid = f"{pdb_id}_{key}" if cid is None else str(cid)
            code = cluster_codes.get(cid)
            if code is None:
                code = len(cluster_codes)
                cluster_codes[cid] = code
            chain_cluster.append(code)

            if keep_templates:
                for tid in cd.get("template_ids") or []:
                    tmpl_id.append(str(tid).encode("ascii", "replace"))
            chain_tmpl_off.append(len(tmpl_id))

        for iface_key in entry.get("interfaces") or {}:
            parts = str(iface_key).split("_")
            if len(parts) != 2:
                continue
            a, b = local_index.get(parts[0]), local_index.get(parts[1])
            if a is None or b is None:
                continue
            iface_a.append(a)
            iface_b.append(b)

        struct_id.append(str(pdb_id).lower().encode("ascii", "replace"))
        rel = entry.get("release_date")
        struct_release.append(b"" if rel is None else str(rel)[:_W_RELEASE].encode("ascii", "replace"))
        res = entry.get("resolution")
        struct_resolution.append(float("nan") if res is None else float(res))
        struct_chain_off.append(len(chain_key))
        struct_iface_off.append(len(iface_a))

        if progress_every and n_done % progress_every == 0:
            logger.info("of3-cache: %d structures, %d chains, %d interfaces", n_done, len(chain_key), len(iface_a))

    def _save(name: str, arr: np.ndarray) -> None:
        np.save(out_dir / f"{name}.npy", arr)

    _save("struct_id", _fixed(struct_id, _W_STRUCT_ID, "struct_id"))
    _save("struct_release", _fixed(struct_release, _W_RELEASE, "struct_release"))
    _save("struct_resolution", np.frombuffer(struct_resolution, dtype=np.float32).copy())
    _save("struct_chain_off", np.frombuffer(struct_chain_off, dtype=np.int64).copy())
    _save("struct_iface_off", np.frombuffer(struct_iface_off, dtype=np.int64).copy())
    _save("chain_key", _fixed(chain_key, _W_CHAIN_KEY, "chain_key"))
    _save("chain_label", _fixed(chain_label, _W_CHAIN_LABEL, "chain_label"))
    _save("chain_entity", np.frombuffer(chain_entity, dtype=np.int32).copy())
    _save("chain_moltype", np.frombuffer(chain_moltype, dtype=np.int8).copy())
    _save("chain_aln_rep", _fixed(chain_aln_rep, _W_ALN_REP, "chain_aln_rep"))
    _save("chain_cluster", np.frombuffer(chain_cluster, dtype=np.int32).copy())
    _save("chain_tmpl_off", np.frombuffer(chain_tmpl_off, dtype=np.int64).copy())
    _save("tmpl_id", _fixed(tmpl_id, _W_TMPL_ID, "tmpl_id"))
    _save("iface_a", np.frombuffer(iface_a, dtype=np.int32).copy())
    _save("iface_b", np.frombuffer(iface_b, dtype=np.int32).copy())

    stat = cache_path.stat()
    meta = {
        "version": CACHE_VERSION,
        "source": str(cache_path),
        "source_bytes": stat.st_size,
        "source_mtime": int(stat.st_mtime),
        "n_structures": len(struct_id),
        "n_chains": len(chain_key),
        "n_interfaces": len(iface_a),
        "n_templates": len(tmpl_id),
        "n_clusters": len(cluster_codes),
        "n_skipped": n_skipped,
        "has_templates": bool(keep_templates and tmpl_id),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    logger.info("of3-cache: wrote %s (%s)", out_dir, meta)
    return meta


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OF3ChainMeta:
    """One chain's row of the cache, decoded."""

    key: str
    label_asym_id: str
    entity_id: int
    mol_type: int
    aln_rep_id: str | None
    cluster_code: int


class OF3PdbCache:
    """Memory-mapped reader for a cache written by :func:`build_of3_pdb_cache`.

    Every column is opened with ``mmap_mode="r"``, so construction is O(number of
    columns) and the resident cost is only the pages actually touched. Safe to
    build once in the parent process and inherit across ``fork``ed dataloader
    workers — the mappings are read-only and file-backed.
    """

    _COLUMNS = (
        "struct_id",
        "struct_release",
        "struct_resolution",
        "struct_chain_off",
        "struct_iface_off",
        "chain_key",
        "chain_label",
        "chain_entity",
        "chain_moltype",
        "chain_aln_rep",
        "chain_cluster",
        "chain_tmpl_off",
        "tmpl_id",
        "iface_a",
        "iface_b",
    )

    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        meta_path = self.cache_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"{self.cache_dir} is not an OF3 columnar cache (no meta.json). "
                "Build it with `deepfold3 build-of3-cache`."
            )
        self.meta = json.loads(meta_path.read_text())
        if int(self.meta.get("version", 0)) != CACHE_VERSION:
            raise ValueError(
                f"{self.cache_dir}: cache version {self.meta.get('version')} != {CACHE_VERSION}; rebuild it."
            )
        for name in self._COLUMNS:
            setattr(self, name, np.load(self.cache_dir / f"{name}.npy", mmap_mode="r"))
        self._id_to_index: dict[str, int] | None = None

    def __len__(self) -> int:
        return int(self.struct_id.shape[0])

    @property
    def has_templates(self) -> bool:
        return bool(self.meta.get("has_templates"))

    # -- structure-level ----------------------------------------------------

    def pdb_id(self, i: int) -> str:
        return self.struct_id[i].decode("ascii")

    def release_date(self, i: int) -> str | None:
        v = self.struct_release[i].decode("ascii")
        return v or None

    def resolution(self, i: int) -> float | None:
        v = float(self.struct_resolution[i])
        return None if np.isnan(v) else v

    def index_of(self, pdb_id: str) -> int | None:
        if self._id_to_index is None:
            self._id_to_index = {b.decode("ascii"): i for i, b in enumerate(np.asarray(self.struct_id))}
        return self._id_to_index.get(pdb_id.lower())

    def chain_slice(self, i: int) -> tuple[int, int]:
        return int(self.struct_chain_off[i]), int(self.struct_chain_off[i + 1])

    def iface_slice(self, i: int) -> tuple[int, int]:
        return int(self.struct_iface_off[i]), int(self.struct_iface_off[i + 1])

    # -- chain-level --------------------------------------------------------

    def chains(self, i: int) -> list[OF3ChainMeta]:
        lo, hi = self.chain_slice(i)
        out = []
        for c in range(lo, hi):
            rep = self.chain_aln_rep[c].decode("ascii")
            out.append(
                OF3ChainMeta(
                    key=self.chain_key[c].decode("ascii"),
                    label_asym_id=self.chain_label[c].decode("ascii"),
                    entity_id=int(self.chain_entity[c]),
                    mol_type=int(self.chain_moltype[c]),
                    aln_rep_id=rep or None,
                    cluster_code=int(self.chain_cluster[c]),
                )
            )
        return out

    def template_ids(self, chain_row: int) -> list[str]:
        lo, hi = int(self.chain_tmpl_off[chain_row]), int(self.chain_tmpl_off[chain_row + 1])
        if hi <= lo:
            return []
        return [b.decode("ascii") for b in np.asarray(self.tmpl_id[lo:hi])]

    # -- derived ------------------------------------------------------------

    def release_dates(self) -> np.ndarray:
        """All release dates as ``S10`` (``b""`` = unknown) — for cutoff filtering."""
        return np.asarray(self.struct_release)

    def interface_cluster_codes(self) -> np.ndarray:
        """[I] int64 code of each interface's *unordered* chain-cluster pair.

        AF3 §2.5.3 clusters an interface by the pair of its chains' sequence
        clusters, matching ``data/sample/weighted.py``'s ``_interface_cluster_id``.
        Packing ``(min, max)`` into one int64 lets the cluster sizes come out of a
        single ``np.unique``.
        """
        struct_of_iface = np.repeat(np.arange(len(self), dtype=np.int64), np.diff(np.asarray(self.struct_iface_off)))
        base = np.asarray(self.struct_chain_off)[struct_of_iface]
        ca = np.asarray(self.chain_cluster)[base + np.asarray(self.iface_a)].astype(np.int64)
        cb = np.asarray(self.chain_cluster)[base + np.asarray(self.iface_b)].astype(np.int64)
        lo = np.minimum(ca, cb)
        hi = np.maximum(ca, cb)
        return (lo << np.int64(32)) | hi
