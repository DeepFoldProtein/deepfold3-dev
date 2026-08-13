"""``deepfold3 plm-prepare`` — build complex-LM stores from the raw sources.

Structural sources (a DeepFold3 data directory)::

    deepfold3 plm-prepare structure --data-dir /data/distillation/afdb_nvda \\
        --output /data/plm/store/afdb --source afdb --workers 64

Paired-FASTA sources (PPLM's release)::

    deepfold3 plm-prepare pairs --fasta /data/plm/raw/STRING_seqs.fasta \\
        --pairs /data/plm/raw/trainset.STRING.txt \\
        --output /data/plm/store/string --source string

Statistics for an existing store::

    deepfold3 plm-prepare stats --store /data/plm/store/afdb
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import click

from deepfold3.scripts import cli
from deepfold3.scripts._common import setup_cli_logging

logger = logging.getLogger(__name__)


@cli.group("plm-prepare")
def plm_prepare() -> None:
    """Build complex-language-model stores."""


@plm_prepare.command("structure")
@click.option("--data-dir", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--output", required=True, type=click.Path(path_type=Path))
@click.option("--source", required=True, help="Name recorded in the store metadata.")
@click.option("--manifest", type=click.Path(path_type=Path), help="Defaults to {data-dir}/manifest.json.")
@click.option("--no-manifest", is_flag=True, help="Take every file on disk instead of the manifest's ids.")
@click.option("--mmcif-meta", type=click.Path(path_type=Path), help="scan-mmcif cache, for PDB release dates.")
@click.option("--workers", default=32, show_default=True)
@click.option("--min-chains", default=2, show_default=True)
@click.option("--max-chains", default=64, show_default=True)
@click.option("--contact-threshold", default=8.0, show_default=True, help="Cβ–Cβ interface cutoff in Å.")
def structure_cmd(
    data_dir: Path,
    output: Path,
    source: str,
    manifest: Path | None,
    no_manifest: bool,
    mmcif_meta: Path | None,
    workers: int,
    min_chains: int,
    max_chains: int,
    contact_threshold: float,
) -> None:
    """Extract sequences and interface flags from a DeepFold3 data directory."""
    setup_cli_logging(__name__)
    from deepfold3.complexlm.prepare import (
        ExtractConfig,
        build_structure_store,
        manifest_index,
        pdb_release_dates,
        write_summary,
    )

    keep: dict[str, int] | None = None
    manifest = manifest or (data_dir / "manifest.json")
    if not no_manifest:
        if not manifest.exists():
            raise click.ClickException(f"{manifest} not found; pass --no-manifest to scan the directory instead")
        logger.info("scanning %s (%.1f GB)", manifest, manifest.stat().st_size / 1e9)
        keep = manifest_index(manifest)
        logger.info("manifest lists %d complexes, %d dated", len(keep), sum(1 for v in keep.values() if v))
    if mmcif_meta is not None:
        # The manifest's own dates can be absent (a store built before
        # enrich-manifest ran); the mmCIF cache is authoritative and covers
        # every PDB entry, so it wins where both exist.
        dates = pdb_release_dates(mmcif_meta)
        keep = {k: dates.get(k.lower(), v) for k, v in (keep or {}).items()}
        logger.info("stamped %d/%d complexes from %s", sum(1 for v in keep.values() if v), len(keep), mmcif_meta)

    cfg = ExtractConfig(contact_threshold=contact_threshold, min_chains=min_chains, max_chains=max_chains)
    build_structure_store(data_dir, output, source, cfg, workers=workers, keep_ids=keep)
    click.echo(json.dumps(write_summary(output), indent=2))


@plm_prepare.command("pairs")
@click.option("--fasta", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--pairs", "pair_list", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--output", required=True, type=click.Path(path_type=Path))
@click.option("--source", required=True)
@click.option("--mmcif-meta", type=click.Path(path_type=Path), help="scan-mmcif cache, for PDB release dates.")
def pairs_cmd(fasta: Path, pair_list: Path, output: Path, source: str, mmcif_meta: Path | None) -> None:
    """Build a store from a PPLM pair list plus its sequence FASTA."""
    setup_cli_logging(__name__)
    from deepfold3.complexlm.prepare import build_pair_store, pdb_release_dates, write_summary

    dates = pdb_release_dates(mmcif_meta) if mmcif_meta else None
    build_pair_store(fasta, pair_list, output, source, dates=dates)
    click.echo(json.dumps(write_summary(output), indent=2))


@plm_prepare.command("stats")
@click.option("--store", required=True, type=click.Path(exists=True, path_type=Path), multiple=True)
def stats_cmd(store: tuple[Path, ...]) -> None:
    """Recompute and print a store's statistics."""
    setup_cli_logging(__name__)
    from deepfold3.complexlm.prepare import write_summary

    for root in store:
        click.echo(json.dumps(write_summary(root), indent=2))


@plm_prepare.command("cluster")
@click.option("--store", required=True, type=click.Path(exists=True, path_type=Path), multiple=True)
@click.option("--work-dir", required=True, type=click.Path(path_type=Path))
@click.option("--min-seq-id", default=0.4, show_default=True)
@click.option("--coverage", default=0.8, show_default=True)
@click.option("--threads", default=32, show_default=True)
@click.option("--skip-mmseqs", is_flag=True, help="Reuse an existing *_cluster.tsv in --work-dir.")
def cluster_cmd(
    store: tuple[Path, ...], work_dir: Path, min_seq_id: float, coverage: float, threads: int, skip_mmseqs: bool
) -> None:
    """Cluster every store's chains together, write ids back, report overlap."""
    setup_cli_logging(__name__)
    from deepfold3.complexlm.cluster import cluster_all

    summary = cluster_all(list(store), work_dir, min_seq_id, coverage, threads, skip_mmseqs)
    click.echo(json.dumps(summary["overlap"], indent=2))
