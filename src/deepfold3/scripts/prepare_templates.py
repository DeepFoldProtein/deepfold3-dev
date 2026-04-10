"""``deepfold3 prepare-templates`` -- discover multimer templates for training.

Reads per-chain m8 files (MMseqs2 hits against PDB SEQRES) and discovers
PDB entries that serve as structural templates for multiple chains of each
training complex.  Writes a lightweight JSON index per complex.

Examples::

    deepfold3 prepare-templates \\
        --data-dir /data/training \\
        --m8-dir /data/msa/pdb/merged/m8 \\
        --max-templates 4 --workers 16
"""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import click

from deepfold3.scripts import cli

logger = logging.getLogger(__name__)


def _process_one(
    record_id: str,
    chain_msa_ids: list[str],
    m8_dir: Path,
    output_dir: Path,
    max_templates: int,
    overwrite: bool = False,
) -> str | None:
    """Process a single record.  Returns skip reason or None on success."""
    from deepfold3.data.paths import template_path
    from deepfold3.data.prepare.template_search import find_multimer_templates, save_template_hits

    out_path = template_path(output_dir, record_id)
    if out_path.exists() and not overwrite:
        return "exists"

    hits = find_multimer_templates(
        record_pdb_id=record_id,
        chain_msa_ids=chain_msa_ids,
        m8_dir=m8_dir,
        max_templates=max_templates,
    )

    if not hits:
        return "no_hits"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_template_hits(hits, out_path)
    return None


@cli.command("prepare-templates")
@click.option(
    "--data-dir",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Training data directory with manifest.json.",
)
@click.option(
    "--m8-dir",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Directory containing {msa_id}.m8 files.",
)
@click.option("--max-templates", default=4, show_default=True, help="Maximum templates per complex.")
@click.option("--workers", default=1, show_default=True, help="Parallel workers.")
@click.option("--overwrite", is_flag=True, help="Overwrite existing template files.")
def prepare_templates(
    data_dir: Path,
    m8_dir: Path,
    max_templates: int,
    workers: int,
    overwrite: bool,
) -> None:
    """Discover multimer templates from per-chain m8 files."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from deepfold3.data.prepare.manifest import discover_records

    records = discover_records(data_dir)
    if not records:
        logger.error("No records found in %s", data_dir)
        return

    from deepfold3.data.paths import template_path

    output_dir = data_dir / "templates"
    output_dir.mkdir(parents=True, exist_ok=True)

    work: list[tuple[str, list[str]]] = []
    for record in records:
        if not overwrite and template_path(output_dir, record.id).exists():
            continue
        msa_ids = [str(c.msa_id) for c in record.chains]
        work.append((record.id, msa_ids))

    logger.info("Processing %d records (%d already done)", len(work), len(records) - len(work))

    done = 0
    no_hits = 0
    errors = 0

    if workers <= 1:
        for record_id, msa_ids in work:
            try:
                result = _process_one(record_id, msa_ids, m8_dir, output_dir, max_templates, overwrite)
                if result == "no_hits":
                    no_hits += 1
                done += 1
            except Exception:
                logger.exception("Error processing %s", record_id)
                errors += 1
            if done % 1000 == 0:
                logger.info("Progress: %d/%d (no_hits=%d, errors=%d)", done, len(work), no_hits, errors)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_process_one, rid, mids, m8_dir, output_dir, max_templates, overwrite): rid
                for rid, mids in work
            }
            for future in as_completed(futures):
                rid = futures[future]
                try:
                    result = future.result()
                    if result == "no_hits":
                        no_hits += 1
                    done += 1
                except Exception:
                    logger.exception("Error processing %s", rid)
                    errors += 1
                if done % 1000 == 0:
                    logger.info("Progress: %d/%d (no_hits=%d, errors=%d)", done, len(work), no_hits, errors)

    logger.info("Done: %d processed, %d no_hits, %d errors", done, no_hits, errors)
