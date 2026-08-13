"""On-disk store of protein complexes as token sequences.

One format for every source — PDB, STRING, AFDB-multimer, Teddymer — so the
sampler, the masking policies and the diagnostics never learn where an example
came from. A source is a directory of flat ``.npy`` arrays opened with
``mmap_mode="r"``: random access costs one page fault, nothing is parsed at
read time, and a dataloader worker holds no per-example file handle.

Layout::

    {root}/meta.json           source name, counts, build arguments
    {root}/seq.npy             uint8  [R]    unique chain sequences, concatenated
    {root}/chain_off.npy       int64  [C+1]  offsets into seq
    {root}/chain_cluster.npy   int32  [C]    sequence-cluster id, -1 if unknown
    {root}/slot_chain.npy      int32  [S]    chain index of each complex-chain slot
    {root}/slot_iface.npy      uint8  [Rs]   per-residue interface flag (optional)
    {root}/cx_off.npy          int64  [N+1]  offsets into slot_chain
    {root}/cx_cluster.npy      int32  [N]    complex-level cluster id
    {root}/cx_date.npy         int32  [N]    release date YYYYMMDD, 0 if unknown
    {root}/ids.txt                            complex ids, one per line

Sequences are **deduplicated globally**, which is what makes chain identity
free: two slots of one complex are the same entity exactly when they point at
the same chain row. Homodimer detection, homomer-paired masking and the
``n_entity`` statistic all read off that, with no sequence comparison at
sampling time.

Interface flags are stored **per slot**, not per chain: the same sequence has
different interface residues in different complexes, so they cannot be shared.
Their offsets are the cumulative chain lengths over slots and are rebuilt at
open time rather than stored.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_ARRAYS = ("seq", "chain_off", "chain_cluster", "slot_chain", "cx_off", "cx_cluster")

#: Written since the first build, but read defensively: a store predating the
#: column must not silently become "every complex released on day zero", which
#: a date filter would read as "drop everything".
_OPTIONAL_ARRAYS = ("cx_date",)


@dataclass(frozen=True)
class StoredComplex:
    """One complex: a list of chains, each a ``uint8`` view into the store.

    Attributes:
        index: Row in the store.
        chain_rows: Chain index of each slot; equal entries mean equal sequence.
        seqs: Token ids per slot, views into the mmap (never write to these).
        iface: Per-residue interface flag per slot, or ``None`` for a
            sequence-only source.
        cluster: Complex-level cluster id, for cluster-balanced sampling.
        date: Release date as ``YYYYMMDD``, 0 when the source has none.
    """

    index: int
    chain_rows: np.ndarray
    seqs: list[np.ndarray]
    iface: list[np.ndarray] | None
    cluster: int
    date: int = 0

    @property
    def n_chains(self) -> int:
        return len(self.seqs)

    @property
    def n_entities(self) -> int:
        return int(np.unique(self.chain_rows).size)

    @property
    def n_res(self) -> int:
        return int(sum(s.size for s in self.seqs))

    @property
    def is_homomer(self) -> bool:
        return self.n_entities == 1


class ComplexStore:
    """Read-only mmap view of one source.

    Args:
        root: Directory written by :class:`ComplexStoreWriter`.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.meta = json.loads((self.root / "meta.json").read_text())
        for name in _ARRAYS:
            setattr(self, name, np.load(self.root / f"{name}.npy", mmap_mode="r"))
        path = self.root / "slot_iface.npy"
        self.slot_iface = np.load(path, mmap_mode="r") if path.exists() else None
        for name in _OPTIONAL_ARRAYS:
            path = self.root / f"{name}.npy"
            setattr(self, name, np.load(path, mmap_mode="r") if path.exists() else None)

        # Slot lengths and their offsets are pure functions of the chain table;
        # deriving them (a few million adds, once) beats storing two more arrays
        # that could fall out of sync with it.
        off = np.asarray(self.chain_off)
        lengths = (off[1:] - off[:-1])[np.asarray(self.slot_chain)]
        self.slot_len = lengths.astype(np.int64)
        self.slot_iface_off = np.zeros(self.slot_len.size + 1, dtype=np.int64)
        np.cumsum(self.slot_len, out=self.slot_iface_off[1:])

        cx_off = np.asarray(self.cx_off)
        self.cx_nchain = (cx_off[1:] - cx_off[:-1]).astype(np.int32)
        self.cx_nres = np.add.reduceat(self.slot_len, cx_off[:-1]).astype(np.int64)
        self.cx_nres[self.cx_nchain == 0] = 0
        self._ids: list[str] | None = None

    @property
    def name(self) -> str:
        return str(self.meta.get("source", self.root.name))

    def __len__(self) -> int:
        return int(self.cx_off.size - 1)

    def __repr__(self) -> str:
        return f"ComplexStore({self.name!r}, n={len(self)}, chains={self.chain_off.size - 1})"

    @property
    def ids(self) -> list[str]:
        """Complex ids, read on first use — diagnostics only, never a hot path."""
        if self._ids is None:
            self._ids = (self.root / "ids.txt").read_text().splitlines()
        return self._ids

    @property
    def has_iface(self) -> bool:
        return self.slot_iface is not None

    @property
    def cluster_run(self) -> str | None:
        """Token of the clustering run that produced ``cx_cluster``.

        Complex-cluster ids index one shared table built across every store in
        a run, so they are comparable between stores that share this token and
        meaningless between stores that do not.
        """
        return self.meta.get("cluster_run")

    def n_entities(self) -> np.ndarray:
        """``int32[N]`` distinct chains per complex — the homo/hetero axis.

        Computed by sorting each complex's slot rows, which is one pass over the
        slot table; the result is small enough to keep and is what the type
        sampler buckets on.
        """
        cx_off = np.asarray(self.cx_off)
        rows = np.asarray(self.slot_chain)
        out = np.ones(len(self), dtype=np.int32)
        for i in range(len(self)):
            lo, hi = cx_off[i], cx_off[i + 1]
            out[i] = np.unique(rows[lo:hi]).size if hi > lo else 0
        return out

    def example(self, index: int) -> StoredComplex:
        lo, hi = int(self.cx_off[index]), int(self.cx_off[index + 1])
        rows = np.asarray(self.slot_chain[lo:hi])
        seqs = [np.asarray(self.seq[self.chain_off[r] : self.chain_off[r + 1]]) for r in rows]
        iface = None
        if self.slot_iface is not None:
            iface = [
                np.asarray(self.slot_iface[self.slot_iface_off[s] : self.slot_iface_off[s + 1]]) for s in range(lo, hi)
            ]
        date = int(self.cx_date[index]) if self.cx_date is not None else 0
        return StoredComplex(index, rows, seqs, iface, int(self.cx_cluster[index]), date)


@dataclass
class ComplexRecord:
    """One complex on its way into a store."""

    id: str
    seqs: list[np.ndarray]
    iface: list[np.ndarray] | None = None
    cluster: int = -1
    chain_clusters: list[int] | None = None
    #: Release date as ``YYYYMMDD``. For PDB-derived sources this is the axis
    #: the train/validation split runs on, so it is carried per complex rather
    #: than filtered at build time — the cutoff is a training decision and
    #: moving it must not require rebuilding the store.
    date: int = 0


class ComplexStoreWriter:
    """Streaming writer; feed :class:`ComplexRecord` in, get a store out.

    Deduplication is by exact sequence bytes. That is the same equivalence the
    manifests express with ``msa_id`` (a SHA-256 of the sequence), so it costs
    nothing extra and is stricter than clustering: near-identical chains stay
    distinct entities, which is the conservative choice for homomer-aware
    masking.

    Args:
        root: Output directory, created if absent.
        source: Name recorded in ``meta.json``.
        with_iface: Whether records carry interface flags. A source that has
            them must supply them for *every* record — a partly-populated
            column would silently become "no interface anywhere" for the
            missing complexes.
    """

    def __init__(self, root: str | Path, source: str, with_iface: bool) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.source = source
        self.with_iface = with_iface
        self._seq = bytearray()
        self._iface = bytearray()
        self._chain_off: list[int] = [0]
        self._chain_cluster: list[int] = []
        self._lookup: dict[bytes, int] = {}
        self._slot_chain: list[int] = []
        self._cx_off: list[int] = [0]
        self._cx_cluster: list[int] = []
        self._cx_date: list[int] = []
        self._ids: list[str] = []

    def add(self, record: ComplexRecord) -> None:
        if self.with_iface and record.iface is None:
            raise ValueError(f"{record.id}: store declares interface flags but the record has none")
        clusters = record.chain_clusters or [-1] * len(record.seqs)
        for slot, seq in enumerate(record.seqs):
            seq = np.ascontiguousarray(seq, dtype=np.uint8)
            key = seq.tobytes()
            row = self._lookup.get(key)
            if row is None:
                row = len(self._chain_cluster)
                self._lookup[key] = row
                self._seq += key
                self._chain_off.append(len(self._seq))
                self._chain_cluster.append(int(clusters[slot]))
            self._slot_chain.append(row)
            if self.with_iface:
                flags = np.ascontiguousarray(record.iface[slot], dtype=np.uint8)
                if flags.size != seq.size:
                    raise ValueError(f"{record.id}: interface flags {flags.size} != sequence {seq.size}")
                self._iface += flags.tobytes()
        self._cx_off.append(len(self._slot_chain))
        self._cx_cluster.append(int(record.cluster))
        self._cx_date.append(int(record.date))
        self._ids.append(record.id)

    def close(self, **meta: object) -> dict:
        def dump(name: str, values, dtype) -> None:
            np.save(self.root / f"{name}.npy", np.asarray(values, dtype=dtype))

        dump("seq", np.frombuffer(bytes(self._seq), dtype=np.uint8), np.uint8)
        dump("chain_off", self._chain_off, np.int64)
        dump("chain_cluster", self._chain_cluster, np.int32)
        dump("slot_chain", self._slot_chain, np.int32)
        dump("cx_off", self._cx_off, np.int64)
        dump("cx_cluster", self._cx_cluster, np.int32)
        dump("cx_date", self._cx_date, np.int32)
        if self.with_iface:
            dump("slot_iface", np.frombuffer(bytes(self._iface), dtype=np.uint8), np.uint8)
        (self.root / "ids.txt").write_text("\n".join(self._ids))
        summary = {
            "source": self.source,
            "n_complexes": len(self._ids),
            "n_chain_slots": len(self._slot_chain),
            "n_unique_chains": len(self._chain_cluster),
            "n_residues_unique": len(self._seq),
            "n_residues_slots": int(sum(self._chain_off[r + 1] - self._chain_off[r] for r in self._slot_chain)),
            "has_iface": self.with_iface,
            "n_dated": int(sum(1 for d in self._cx_date if d > 0)),
            **meta,
        }
        (self.root / "meta.json").write_text(json.dumps(summary, indent=2, default=str))
        logger.info("wrote %s: %s", self.root, summary)
        return summary
