"""``deepfold3 build-pdb-cache`` — pre-build the per-chain template cache.

Training-time template extraction otherwise re-parses the *whole* template
mmCIF with gemmi on every hit (a query-size-independent cost that dominates
the periodic dataloader stalls / step-time spikes). This command walks the
template indices under a data dir, collects every referenced template PDB,
and parses each one ONCE in parallel, slicing it into per-chain features
written to a hierarchical sharded cache::

    {output}/{shard(pdb)}/{pdb}/{chain}.npz
    {output}/{shard(pdb)}/{pdb}/_done        # per-PDB completion sentinel

At train time ``--pdb-chain-cache-dir {output}`` then serves every template
chain from disk (or the per-worker memory LRU) — no gemmi parse on the hot
path. The build is idempotent: PDBs with a ``_done`` sentinel are skipped
unless ``--overwrite``.

Examples::

    deepfold3 build-pdb-cache --data-dir /data/distillation/patchr_training \\
        --mmcif-dir /scratch/pdb/mmCIF --output /data/pdb/cache --workers 64
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import click

from deepfold3.scripts import cli


def _scan_index(args: tuple[str, str]) -> set[str]:
    """Worker: extract template PDB ids referenced by one index JSON.

    Module-level for picklability. Tolerant of corrupt indices (returns the
    empty set rather than failing the whole scan)."""
    path_str, kind = args
    from deepfold3.data.prepare.template_search import (
        load_monomer_template_hits,
        load_template_hits,
    )

    out: set[str] = set()
    try:
        if kind == "multimer":
            for h in load_template_hits(Path(path_str)):
                out.add(h.template_pdb_id)
        else:
            for h in load_monomer_template_hits(Path(path_str)):
                out.add(h.template_pdb_id)
    except Exception:
        pass
    return out


def _build_one(args: tuple[str, str, str, bool]) -> tuple[str, int]:
    """Worker: parse one PDB and write all its chains. Returns (pdb_id, n_chains).

    ``n_chains`` is the number written (0 = already cached / missing / empty,
    ``-1`` = parse error). Module-level for picklability."""
    pdb_id, mmcif_dir, cache_root, overwrite = args
    from deepfold3.data.parse.pdb_chain_cache import build_pdb_cache_entry

    try:
        n = build_pdb_cache_entry(pdb_id, Path(mmcif_dir), Path(cache_root), overwrite=overwrite)
    except Exception:
        return pdb_id, -1
    return pdb_id, n


def _collect_pdb_ids(data_dir: Path, workers: int, log) -> set[str]:
    """Union of all template PDB ids referenced by the data dir's indices."""
    index_files: list[tuple[str, str]] = []
    for sub, kind in (("templates", "multimer"), ("monomer_templates", "monomer")):
        root = data_dir / sub
        if not root.is_dir():
            continue
        files = list(root.rglob("*.json"))
        index_files += [(str(p), kind) for p in files]
        log.info("Found %d %s template index files under %s", len(files), kind, root)
    if not index_files:
        return set()

    pdb_ids: set[str] = set()
    if workers <= 1:
        for item in index_files:
            pdb_ids |= _scan_index(item)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for i, s in enumerate(ex.map(_scan_index, index_files, chunksize=256), 1):
                pdb_ids |= s
                if i % 20000 == 0:
                    log.info("  scanned %d/%d indices, %d unique PDBs so far", i, len(index_files), len(pdb_ids))
    return pdb_ids


def _collect_mirror_ids(mmcif_dir: Path, log) -> set[str]:
    """Every PDB id present in the local mmCIF mirror ({mid2}/{id}.cif[.gz|.zst]).

    Any dataset can hit any PDB as a template, so caching the whole mirror makes
    template loading cold-parse-free regardless of which dataset references it.
    """
    pdb_ids: set[str] = set()
    for p in mmcif_dir.rglob("*.cif*"):
        name = p.name
        cut = name.find(".cif")
        if cut > 0:
            pdb_ids.add(name[:cut])
        if len(pdb_ids) % 50000 == 0 and pdb_ids:
            log.info("  walked mirror, %d unique PDBs so far", len(pdb_ids))
    log.info("Found %d PDB entries in mirror %s", len(pdb_ids), mmcif_dir)
    return pdb_ids


@cli.command("build-pdb-cache")
@click.option(
    "--data-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Training data dir to scan for templates/ and monomer_templates/ indices "
    "(cache only the PDBs that dir references). Omit with --all to cache the whole mirror.",
)
@click.option(
    "--all",
    "scan_all",
    is_flag=True,
    help="Cache EVERY PDB in --mmcif-dir (any dataset can hit any PDB as a template). "
    "Unions with --data-dir if both are given.",
)
@click.option(
    "--mmcif-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Local PDB mmCIF mirror ({mid2}/{id}.cif[.gz|.zst]).",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=Path("/data/pdb/cache"),
    show_default=True,
    help="Per-chain cache root (hierarchical sharded layout).",
)
@click.option("--workers", type=int, default=64, show_default=True, help="Parallel workers (CPU-bound gemmi parse).")
@click.option("--overwrite", is_flag=True, help="Re-parse and overwrite PDBs that already have a _done sentinel.")
@click.option("--limit", type=int, default=0, help="Process at most N PDBs (0 = all; for smoke tests).")
def build_pdb_cache(
    data_dir: Path | None,
    scan_all: bool,
    mmcif_dir: Path,
    output: Path,
    workers: int,
    overwrite: bool,
    limit: int,
) -> None:
    """Pre-build the per-chain template cache.

    Source of PDB ids: ``--data-dir`` (only the templates that dir references)
    and/or ``--all`` (every PDB in the mirror). At least one is required.
    """
    from deepfold3.scripts._common import setup_cli_logging

    log = setup_cli_logging(__name__)

    if data_dir is None and not scan_all:
        raise click.UsageError("Provide --data-dir and/or --all to select which PDBs to cache.")

    pdb_id_set: set[str] = set()
    if data_dir is not None:
        log.info("Scanning template indices under %s (workers=%d)...", data_dir, workers)
        pdb_id_set |= _collect_pdb_ids(data_dir, workers, log)
    if scan_all:
        log.info("Walking mmCIF mirror %s for all PDB ids...", mmcif_dir)
        pdb_id_set |= _collect_mirror_ids(mmcif_dir, log)
    pdb_ids = sorted(pdb_id_set)
    if limit and len(pdb_ids) > limit:
        pdb_ids = pdb_ids[:limit]
    if not pdb_ids:
        click.echo("No template PDB ids found — nothing to build.")
        return
    log.info("Building cache for %d unique template PDBs → %s", len(pdb_ids), output)

    output.mkdir(parents=True, exist_ok=True)
    todo = [(pid, str(mmcif_dir), str(output), overwrite) for pid in pdb_ids]

    total_chains = 0
    built = 0  # PDBs that wrote ≥1 chain
    skipped = 0  # already cached / missing / empty
    errors = 0
    if workers <= 1:
        results = (_build_one(item) for item in todo)
    else:
        ex = ProcessPoolExecutor(max_workers=workers)
        results = ex.map(_build_one, todo, chunksize=4)
    try:
        for done, (_pid, n) in enumerate(results, 1):
            if n < 0:
                errors += 1
            elif n == 0:
                skipped += 1
            else:
                built += 1
                total_chains += n
            if done % 5000 == 0:
                log.info(
                    "  %d/%d PDBs (built=%d, skipped=%d, errors=%d, chains=%d)",
                    done,
                    len(todo),
                    built,
                    skipped,
                    errors,
                    total_chains,
                )
    finally:
        if workers > 1:
            ex.shutdown()

    log.info(
        "Done: %d PDBs built (%d chains), %d skipped (cached/missing), %d errors.",
        built,
        total_chains,
        skipped,
        errors,
    )
    click.echo(f"Cached {total_chains} chains from {built} PDBs → {output} ({skipped} skipped, {errors} errors)")
