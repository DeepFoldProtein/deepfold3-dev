"""Sequence clustering across every store, and the source-overlap report.

One clustering run over the union of all unique chains does three jobs at once:

1. **Cluster-balanced sampling.** Drawing a cluster and then a member, rather
   than drawing a complex, is what stops a 500-entry family from outweighing a
   singleton. Written back into each store as ``chain_cluster``, and lifted to
   the complex level as the sorted tuple of its chains' clusters — for a
   heterodimer that is the *pair* cluster, which is how PPLM balances its sets.
2. **Source overlap.** Teddymer is AFDB-derived, so "451k Teddymer examples
   plus 1.59M AFDB examples" is not 2.04M independent examples. The report
   measures how far from independent they are, at sequence, cluster and
   complex level.
3. **Leakage.** The same measurement between a training source and an
   evaluation source is the leakage check. An index holdout does not survive
   near-duplicates; a cluster holdout does.

Clustering is ``mmseqs easy-linclust`` at 40% identity and 80% coverage — the
wwPDB convention, and linear-time, which matters at ~3M sequences.
"""

from __future__ import annotations

import json
import logging
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np

from deepfold3.complexlm.store import ComplexStore
from deepfold3.complexlm.vocab import id_to_char_table

logger = logging.getLogger(__name__)

_ID_TO_CHAR = id_to_char_table()


def export_chains(stores: list[ComplexStore], out: Path) -> int:
    """Write every store's unique chains to one FASTA, headed ``source:row``."""
    written = 0
    with out.open("wb") as fh:
        for store in stores:
            seq = np.asarray(store.seq)
            offsets = np.asarray(store.chain_off)
            for row in range(offsets.size - 1):
                chunk = seq[offsets[row] : offsets[row + 1]]
                fh.write(f">{store.name}:{row}\n".encode())
                fh.write(_ID_TO_CHAR[chunk.astype(np.int64)].tobytes() + b"\n")
                written += 1
    logger.info("wrote %d chains to %s", written, out)
    return written


def run_linclust(fasta: Path, out_prefix: Path, min_seq_id: float, coverage: float, threads: int) -> Path:
    """Run ``mmseqs easy-linclust``; returns the ``*_cluster.tsv`` path."""
    tsv = Path(f"{out_prefix}_cluster.tsv")
    cmd = [
        "mmseqs", "easy-linclust", str(fasta), str(out_prefix), str(out_prefix.parent / "tmp"),
        "--min-seq-id", str(min_seq_id), "-c", str(coverage), "--cov-mode", "1",
        "--threads", str(threads),
    ]  # fmt: skip
    logger.info("running %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return tsv


def read_clusters(tsv: Path) -> dict[str, dict[int, int]]:
    """``{source: {chain row: cluster id}}`` from an mmseqs cluster TSV.

    The TSV is ``representative<TAB>member``; representatives are numbered by
    first appearance, so ids are dense and stable for a given file.
    """
    assignment: dict[str, dict[int, int]] = defaultdict(dict)
    ids: dict[str, int] = {}
    with tsv.open() as fh:
        for line in fh:
            rep, member = line.rstrip("\n").split("\t")
            cluster = ids.setdefault(rep, len(ids))
            source, row = member.rsplit(":", 1)
            assignment[source][int(row)] = cluster
    logger.info("%s: %d clusters over %d sources", tsv.name, len(ids), len(assignment))
    return assignment


def complex_clusters(
    store: ComplexStore, chain_cluster: np.ndarray, ids: dict[tuple[int, ...], int] | None = None
) -> np.ndarray:
    """Lift chain clusters to complexes: the sorted tuple of member clusters.

    Two complexes share a cluster when they are built from the same families in
    the same multiplicities — the natural generalisation of PPLM's pair-level
    clustering to an arbitrary number of chains.

    Args:
        ids: Shared key -> id table. Passing one across stores is what makes
            complex-cluster ids **comparable between stores**, which is what
            the cluster-disjoint evaluation filter needs: ``string`` and
            ``string_valid`` are different stores, and per-store numbering
            would silently make every comparison between them a miss.
    """
    cx_off = np.asarray(store.cx_off)
    rows = np.asarray(store.slot_chain)
    ids = {} if ids is None else ids
    out = np.full(len(store), -1, dtype=np.int32)
    for i in range(len(store)):
        key = tuple(sorted(chain_cluster[rows[cx_off[i] : cx_off[i + 1]]].tolist()))
        out[i] = ids.setdefault(key, len(ids))
    return out


def apply_clusters(
    store: ComplexStore,
    assignment: dict[int, int],
    complex_ids: dict[tuple[int, ...], int] | None = None,
    run: str | None = None,
) -> dict:
    """Write ``chain_cluster`` and ``cx_cluster`` back into a store.

    Args:
        run: Token identifying this clustering run, recorded in the store's
            metadata. Complex-cluster ids are only comparable between stores
            that carry the *same* token — they are positions in one shared
            table, so comparing across runs would match unrelated families.
    """
    n_chains = int(store.chain_off.size - 1)
    chain_cluster = np.full(n_chains, -1, dtype=np.int32)
    for row, cluster in assignment.items():
        chain_cluster[row] = cluster
    missing = int((chain_cluster < 0).sum())
    if missing:
        # mmseqs drops nothing silently, but a truncated TSV would; giving the
        # unassigned chains singleton clusters keeps them samplable instead of
        # letting them vanish from a cluster-balanced draw.
        chain_cluster[chain_cluster < 0] = np.arange(chain_cluster.max() + 1, chain_cluster.max() + 1 + missing)
        logger.warning("%s: %d chains missing from the clustering, made singletons", store.name, missing)
    cx_cluster = complex_clusters(store, chain_cluster, complex_ids)
    np.save(store.root / "chain_cluster.npy", chain_cluster)
    np.save(store.root / "cx_cluster.npy", cx_cluster)
    if run is not None:
        meta = json.loads((store.root / "meta.json").read_text())
        meta["cluster_run"] = run
        (store.root / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    summary = {
        "source": store.name,
        "n_chain_clusters": int(np.unique(chain_cluster).size),
        "n_complex_clusters": int(np.unique(cx_cluster).size),
        "n_complexes": len(store),
    }
    logger.info("clustered %s", summary)
    return summary


def overlap_report(stores: list[ComplexStore], assignment: dict[str, dict[int, int]]) -> dict:
    """Pairwise source overlap at sequence, cluster and complex level.

    ``jaccard`` is over clusters; ``covered`` is the asymmetric quantity that
    matters for leakage — the fraction of *this* source's clusters that also
    appear in the other one.
    """
    by_source: dict[str, dict[str, set]] = {}
    for store in stores:
        rows = assignment.get(store.name, {})
        clusters = {rows.get(r, -1 - r) for r in range(int(store.chain_off.size - 1))}
        seq = np.asarray(store.seq)
        offsets = np.asarray(store.chain_off)
        sequences = {seq[offsets[r] : offsets[r + 1]].tobytes() for r in range(offsets.size - 1)}
        chain_cluster = np.array([rows.get(r, -1 - r) for r in range(offsets.size - 1)], dtype=np.int64)
        cx_off = np.asarray(store.cx_off)
        slot = np.asarray(store.slot_chain)
        complexes = {tuple(sorted(chain_cluster[slot[cx_off[i] : cx_off[i + 1]]].tolist())) for i in range(len(store))}
        by_source[store.name] = {"sequence": sequences, "cluster": clusters, "complex": complexes}

    report: dict = {"per_source": {k: {level: len(v) for level, v in d.items()} for k, d in by_source.items()}}
    pairs = {}
    names = list(by_source)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            entry = {}
            for level in ("sequence", "cluster", "complex"):
                x, y = by_source[a][level], by_source[b][level]
                shared = len(x & y)
                entry[level] = {
                    "shared": shared,
                    "jaccard": shared / max(len(x | y), 1),
                    f"covered_{a}": shared / max(len(x), 1),
                    f"covered_{b}": shared / max(len(y), 1),
                }
            pairs[f"{a}|{b}"] = entry
    report["pairs"] = pairs
    return report


def cluster_all(
    store_roots: list[Path],
    work_dir: Path,
    min_seq_id: float = 0.4,
    coverage: float = 0.8,
    threads: int = 32,
    skip_mmseqs: bool = False,
) -> dict:
    """Export, cluster, write back, and report."""
    work_dir.mkdir(parents=True, exist_ok=True)
    stores = [ComplexStore(root) for root in store_roots]
    fasta = work_dir / "chains.fasta"
    prefix = work_dir / "clust"
    if not skip_mmseqs:
        export_chains(stores, fasta)
        run_linclust(fasta, prefix, min_seq_id, coverage, threads)
    assignment = read_clusters(Path(f"{prefix}_cluster.tsv"))

    complex_ids: dict[tuple[int, ...], int] = {}
    tsv = Path(f"{prefix}_cluster.tsv")
    run = f"linclust-{min_seq_id}-{coverage}-{tsv.stat().st_size}"
    summary = {
        "min_seq_id": min_seq_id,
        "coverage": coverage,
        "cluster_run": run,
        "stores": [apply_clusters(store, assignment.get(store.name, {}), complex_ids, run) for store in stores],
        "overlap": overlap_report(stores, assignment),
    }
    (work_dir / "clustering.json").write_text(json.dumps(summary, indent=2))
    return summary
