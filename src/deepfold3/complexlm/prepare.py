"""Build :mod:`deepfold3.complexlm.store` stores from the raw sources.

Two kinds of source, one output format:

* **structural** — a DeepFold3 data directory (``structures/{ab}/{id}.npz``).
  Chains come from the canonical residue table, so unresolved residues are kept
  as part of the sequence: the language model is trained on sequence, and a
  crystallographic gap is not a gap in the protein. Interface flags come from
  the coordinates that *are* there.
* **paired FASTA** — PPLM's ``trainset.{PDB,STRING}.txt`` plus its sequence
  FASTA. Sequence only; no interface flags, therefore uniform masking.

Release dates travel with the records. For anything PDB-derived the
train/validation split is temporal — a sequence-level split leaks, because the
same complex is re-deposited at better resolution for years — so every record
carries ``YYYYMMDD`` and the cutoff is applied when the store is *opened*, not
when it is built. Moving the cutoff must never require a rebuild.

Interface criterion: representative atom (Cβ, Cα for glycine) within 8 Å of a
representative atom of another chain. This is the convention already used by
``export/plm-interface-probe``, so interface-biased masking and the probes that
measure interface signal agree on what an interface residue is. It is one atom
per residue rather than all heavy atoms — 20× cheaper over 2M complexes, and
the 8 Å threshold on Cβ is calibrated for that.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from deepfold3.complexlm.store import ComplexRecord, ComplexStoreWriter
from deepfold3.complexlm.vocab import ascii_to_id_table, res_type_to_id_table
from deepfold3.data import const

logger = logging.getLogger(__name__)

_RES_TO_ID = res_type_to_id_table()
_ASCII_TO_ID = ascii_to_id_table()


@dataclass
class ExtractConfig:
    """Knobs shared by every structural source."""

    contact_threshold: float = 8.0
    min_chains: int = 2
    max_chains: int = 64
    min_chain_len: int = 8


@dataclass
class ExtractStats:
    """Why complexes were dropped — printed at the end of a build."""

    seen: int = 0
    kept: int = 0
    dropped: dict[str, int] = field(default_factory=dict)

    def drop(self, reason: str) -> None:
        self.dropped[reason] = self.dropped.get(reason, 0) + 1

    def merge(self, other: ExtractStats) -> None:
        self.seen += other.seen
        self.kept += other.kept
        for k, v in other.dropped.items():
            self.dropped[k] = self.dropped.get(k, 0) + v


# ---------------------------------------------------------------- structures


def _representative_atoms(residues: np.ndarray, atoms: np.ndarray) -> np.ndarray:
    """Index of each residue's representative atom, ``-1`` when it has none.

    Cβ when resolved, Cα otherwise — glycine has no Cβ, and a residue with only
    a backbone trace still has a usable position. Unresolved residues keep
    ``-1`` and are excluded from the contact search rather than being placed at
    the origin, which would fabricate contacts.
    """
    n_res = residues.size
    counts = residues["atom_num"].astype(np.int64)
    if int(counts.sum()) != atoms.size:
        # Non-contiguous residue→atom mapping; fall back to explicit indices.
        atom_res = np.full(atoms.size, -1, dtype=np.int64)
        for r in range(n_res):
            lo = int(residues["atom_idx"][r])
            atom_res[lo : lo + int(counts[r])] = r
    else:
        atom_res = np.repeat(np.arange(n_res, dtype=np.int64), counts)

    names, present = atoms["name"], atoms["is_present"]
    rep = np.full(n_res, -1, dtype=np.int64)
    for target in (b"CA  ", b"CB  "):  # Cβ last: it wins where both exist
        hit = np.flatnonzero((names == target) & present)
        if hit.size:
            valid = atom_res[hit] >= 0
            rep[atom_res[hit[valid]]] = hit[valid]
    return rep


def _interface_flags(coords: list[np.ndarray], threshold: float) -> list[np.ndarray]:
    """Per-residue cross-chain contact flags, one ``uint8`` array per chain.

    ``coords[i]`` is ``[L_i, 3]`` with NaN rows for residues that have no
    representative atom. A single KD-tree over every chain at once is one query
    instead of ``K(K-1)/2``, which matters for the 200-chain assemblies in the
    PDB set.
    """
    sizes = [c.shape[0] for c in coords]
    flat = np.concatenate(coords, axis=0) if coords else np.zeros((0, 3), np.float32)
    slot = np.repeat(np.arange(len(sizes)), sizes)
    finite = np.isfinite(flat).all(axis=1)
    flags = np.zeros(flat.shape[0], dtype=np.uint8)
    if finite.sum() >= 2 and len(sizes) >= 2:
        idx = np.flatnonzero(finite)
        pairs = cKDTree(flat[idx]).query_pairs(threshold, output_type="ndarray")
        if pairs.size:
            a, b = idx[pairs[:, 0]], idx[pairs[:, 1]]
            cross = slot[a] != slot[b]
            flags[a[cross]] = 1
            flags[b[cross]] = 1
    return list(np.split(flags, np.cumsum(sizes)[:-1]))


def extract_structure(path: Path, cfg: ExtractConfig) -> tuple[ComplexRecord | None, str]:
    """One ``structures/*.npz`` -> one record, or ``(None, reason)``."""
    try:
        with np.load(path) as d:
            chains, residues, atoms = d["chains"], d["residues"], d["atoms"]
    except Exception as exc:  # a truncated shard should not lose the build
        logger.warning("unreadable %s: %s", path, exc)
        return None, "unreadable"

    protein = np.flatnonzero(chains["mol_type"] == const.MOL_PROTEIN)
    if protein.size < cfg.min_chains:
        return None, "too_few_protein_chains"
    if protein.size > cfg.max_chains:
        return None, "too_many_chains"

    rep = _representative_atoms(residues, atoms)
    xyz = np.stack((atoms["x"], atoms["y"], atoms["z"]), axis=1).astype(np.float32)

    seqs, coords = [], []
    for ci in protein:
        lo = int(chains["res_idx"][ci])
        hi = lo + int(chains["res_num"][ci])
        if hi - lo < cfg.min_chain_len:
            continue
        seqs.append(_RES_TO_ID[residues["res_type"][lo:hi]])
        r = rep[lo:hi]
        c = np.full((hi - lo, 3), np.nan, dtype=np.float32)
        c[r >= 0] = xyz[r[r >= 0]]
        coords.append(c)
    if len(seqs) < cfg.min_chains:
        return None, "too_few_long_chains"

    return ComplexRecord(id=path.stem, seqs=seqs, iface=_interface_flags(coords, cfg.contact_threshold)), "ok"


def _extract_shard(
    shard: Path, cfg: ExtractConfig, keep: dict[str, int] | None
) -> tuple[list[ComplexRecord], ExtractStats]:
    out, stats = [], ExtractStats()
    for path in sorted(shard.glob("*.npz")):
        if keep is not None and path.stem not in keep:
            continue
        stats.seen += 1
        record, reason = extract_structure(path, cfg)
        if record is None:
            stats.drop(reason)
            continue
        record.date = keep.get(path.stem, 0) if keep else 0
        stats.kept += 1
        out.append(record)
    return out, stats


def build_structure_store(
    data_dir: Path,
    out_dir: Path,
    source: str,
    cfg: ExtractConfig,
    workers: int = 32,
    keep_ids: dict[str, int] | None = None,
) -> dict:
    """Scan ``{data_dir}/structures`` in parallel and write a store.

    Args:
        keep_ids: ``{complex id: release date}``. Restricts the build to these
            ids — how a manifest's own filtering is honoured without parsing a
            multi-gigabyte JSON in every worker — and stamps each record with
            its date. ``None`` takes every file on disk, undated.
    """
    shards = sorted(p for p in (data_dir / "structures").iterdir() if p.is_dir())
    logger.info("%s: %d shards under %s (workers=%d)", source, len(shards), data_dir, workers)
    writer = ComplexStoreWriter(out_dir, source, with_iface=True)
    stats = ExtractStats()
    fn = partial(_extract_shard, cfg=cfg, keep=keep_ids)
    with Pool(workers) as pool:
        for i, (records, shard_stats) in enumerate(pool.imap(fn, shards)):
            stats.merge(shard_stats)
            for record in records:
                writer.add(record)
            if (i + 1) % 32 == 0:
                logger.info("%s: %d/%d shards, %d complexes", source, i + 1, len(shards), stats.kept)
    return writer.close(
        data_dir=str(data_dir),
        seen=stats.seen,
        dropped=stats.dropped,
        contact_threshold=cfg.contact_threshold,
        min_chains=cfg.min_chains,
    )


_MANIFEST_SCAN = re.compile(rb'"id": "([^"]+)"|"released": (?:"(\d{4})-(\d{2})-(\d{2})"|null)')


def manifest_index(path: Path) -> dict[str, int]:
    """``{complex id: release date}`` from a manifest, without parsing it.

    The AFDB manifest is 1.8 GB of JSON whose only use here is its id list, the
    fact that it is the *filtered* set (the PDB manifest drops ~10k entries
    whose files remain on disk), and the release date. A regex over the raw
    bytes gets all three in seconds instead of minutes and several gigabytes of
    dicts. ``chain_id`` / ``msa_id`` / ``cluster_id`` do not collide with the
    pattern: it anchors on a quote immediately before ``id``.

    Dates are ``YYYYMMDD`` integers, 0 when the source has none (AFDB and
    Teddymer are predicted structures and carry no release date).
    """
    index: dict[str, int] = {}
    current: str | None = None
    for m in _MANIFEST_SCAN.finditer(path.read_bytes()):
        if m.group(1) is not None:
            current = m.group(1).decode()
            index[current] = 0
        elif current is not None and m.group(2) is not None:
            index[current] = int(m.group(2) + m.group(3) + m.group(4))
    return index


def pdb_release_dates(meta_json: Path) -> dict[str, int]:
    """``{pdb id: YYYYMMDD}`` from ``deepfold3 scan-mmcif``'s metadata cache.

    This is the canonical release date — the earliest revision date, matching
    AlphaFold 3's template convention — and it covers every PDB entry, not just
    the ones in a training manifest. PPLM's pair sets name their examples by
    PDB id, so this is the only way to date them.
    """
    raw = json.loads(Path(meta_json).read_text())
    out = {}
    for pdb_id, meta in raw.items():
        released = (meta or {}).get("released")
        out[pdb_id.lower()] = int(released.replace("-", "")) if released else 0
    return out


# ------------------------------------------------------------- paired FASTA


def read_fasta(path: Path) -> dict[str, np.ndarray]:
    """FASTA -> ``{id: token ids}``; the id is the first whitespace token."""
    seqs: dict[str, list[bytes]] = {}
    key = None
    with path.open("rb") as fh:
        for line in fh:
            if line.startswith(b">"):
                key = line[1:].split()[0].decode()
                seqs[key] = []
            elif key is not None:
                seqs[key].append(line.strip())
    return {k: _ASCII_TO_ID[np.frombuffer(b"".join(v), dtype=np.uint8)] for k, v in seqs.items() if v}


def _split_pair(name: str, known: dict[str, np.ndarray]) -> tuple[str, str] | None:
    """Split a PPLM pair id into its two sequence ids.

    Both naming schemes are ambiguous — PDB's ``{pdb}-assembly1_{c1}_{c2}`` has
    chain ids that may contain ``_``, and STRING's ``{id1}-{id2}`` has ids that
    may contain ``-`` — so the split is resolved against the sequence table
    rather than by counting separators.
    """
    for sep in ("_", "-"):
        positions = [i for i, ch in enumerate(name) if ch == sep]
        for i in reversed(positions):
            left, right = name[:i], name[i + 1 :]
            if sep == "_":  # PDB: the right side is a bare chain id
                prefix = left.rsplit("_", 1)[0] if "_" in left else None
                if prefix and left in known and f"{prefix}_{right}" in known:
                    return left, f"{prefix}_{right}"
            elif left in known and right in known:
                return left, right
    return None


def iter_pair_records(
    pair_list: Path, seqs: dict[str, np.ndarray], dates: dict[str, int] | None = None
) -> Iterator[tuple[ComplexRecord, str]]:
    """One record per pair; ``cluster`` is the line number of the pair list.

    PPLM ships its training set already clustered — one line per cluster, every
    pair on that line redundant with the others. Carrying the line number
    through as the cluster id is what lets the sampler draw uniformly over
    clusters instead of over pairs, which is how PPLM balances the set.
    """
    with pair_list.open() as fh:
        for cluster, line in enumerate(fh):
            for name in line.split():
                pair = _split_pair(name, seqs)
                if pair is None:
                    yield ComplexRecord(name, []), "unparsable_pair"
                    continue
                a, b = pair
                # PPLM names a PDB pair "{pdb}-assembly1_{c1}_{c2}"; the date
                # comes from the entry, not from the pair.
                date = dates.get(name.split("-")[0].lower(), 0) if dates else 0
                yield ComplexRecord(name, [seqs[a], seqs[b]], cluster=cluster, date=date), "ok"


def build_pair_store(
    fasta: Path,
    pair_list: Path,
    out_dir: Path,
    source: str,
    min_chain_len: int = 8,
    dates: dict[str, int] | None = None,
) -> dict:
    seqs = read_fasta(fasta)
    logger.info("%s: %d sequences from %s", source, len(seqs), fasta.name)
    writer = ComplexStoreWriter(out_dir, source, with_iface=False)
    stats = ExtractStats()
    for record, reason in iter_pair_records(pair_list, seqs, dates):
        stats.seen += 1
        if reason != "ok":
            stats.drop(reason)
            continue
        if any(s.size < min_chain_len for s in record.seqs):
            stats.drop("short_chain")
            continue
        stats.kept += 1
        writer.add(record)
    logger.info("%s: kept %d/%d pairs %s", source, stats.kept, stats.seen, stats.dropped)
    return writer.close(fasta=str(fasta), pair_list=str(pair_list), seen=stats.seen, dropped=stats.dropped)


def summarize(store_root: Path) -> dict:
    """Read back a store and report the statistics the mixture design needs."""
    from deepfold3.complexlm.store import ComplexStore

    store = ComplexStore(store_root)
    n_ent = store.n_entities()
    n_chain = store.cx_nchain
    n_res = store.cx_nres
    chain_len = np.diff(np.asarray(store.chain_off))
    quantiles = [50, 90, 95, 99, 99.9]
    summary = {
        "source": store.name,
        "n_complexes": len(store),
        "n_unique_chains": int(chain_len.size),
        "homomer_frac": float((n_ent == 1).mean()),
        "dimer_frac": float((n_chain == 2).mean()),
        "multimer_frac": float((n_chain > 2).mean()),
        "chain_count": {
            str(k): int(v) for k, v in zip(*np.unique(np.clip(n_chain, 0, 12), return_counts=True), strict=True)
        },
        "chain_len_pct": {str(q): float(np.percentile(chain_len, q)) for q in quantiles},
        "complex_len_pct": {str(q): float(np.percentile(n_res, q)) for q in quantiles},
        "chain_len_cover": {str(c): float((chain_len <= c).mean()) for c in (512, 1024, 2048, 4096)},
        "complex_len_cover": {str(c): float((n_res <= c).mean()) for c in (1024, 2048, 4096, 8192)},
        "has_iface": store.has_iface,
    }
    if store.cx_date is not None and (np.asarray(store.cx_date) > 0).any():
        dated = np.asarray(store.cx_date)
        dated = dated[dated > 0]
        summary["n_dated"] = int(dated.size)
        summary["date_pct"] = {str(q): int(np.percentile(dated, q)) for q in (1, 50, 99)}
        for cutoff in (20210930, 20230101):
            summary[f"frac_after_{cutoff}"] = float((dated > cutoff).mean())
    if store.has_iface:
        iface = np.asarray(store.slot_iface)
        summary["iface_frac"] = float(iface.mean())
    return summary


def write_summary(store_root: Path) -> dict:
    summary = summarize(store_root)
    (Path(store_root) / "stats.json").write_text(json.dumps(summary, indent=2))
    return summary
