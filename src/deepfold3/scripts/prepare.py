"""``deepfold3 prepare`` -- preprocess PDB mmCIF into training data.

Examples:
    Process specific PDB IDs::

        deepfold3 prepare --mmcif-dir /scratch/pdb/mmCIF --output /data/training \\
            --pdb-ids 1pfe,8qy0,7bv2

    Process all structures (batch)::

        deepfold3 prepare --mmcif-dir /scratch/pdb/mmCIF --output /data/training \\
            --max-resolution 4.0 --workers 8
"""

from __future__ import annotations

import json
import logging
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import click

from deepfold3.scripts import cli


def _process_one(
    pdb_id: str,
    mmcif_dir: Path,
    output_dir: Path,
    use_assembly: bool,
    max_resolution: float | None,
) -> dict | None:
    """Process a single PDB entry. Returns record dict or None on failure."""
    from deepfold3.data.parse.templates import find_pdb_mmcif_path
    from deepfold3.data.prepare.filters import filter_structure
    from deepfold3.data.prepare.structure import process_mmcif

    cif_path = find_pdb_mmcif_path(pdb_id, mmcif_dir)
    if cif_path is None:
        return None

    try:
        structure, record = process_mmcif(cif_path, use_assembly=use_assembly)
    except Exception:
        traceback.print_exc()
        return None

    # Resolution filter (skip if no resolution available for non-X-ray)
    if max_resolution is not None and record.structure_info is not None:
        res = record.structure_info.resolution
        if res is not None and res > max_resolution:
            return None

    # Apply quality filters
    record = filter_structure(structure, record)

    # Skip if no valid chains remain
    if not any(c.valid for c in record.chains):
        return None

    # Save structure NPZ
    structure.save(output_dir / "structures" / f"{record.id}.npz")

    # Save individual record JSON
    record_dict = record.to_dict()
    with open(output_dir / "records" / f"{record.id}.json", "w") as f:
        json.dump(record_dict, f, indent=2)

    return record_dict


def _collect_pdb_ids(mmcif_dir: Path, limit: int | None = None) -> list[str]:
    """Scan mmCIF directory for all PDB IDs.

    Uses single-level glob (``*/*.cif.gz``) matching the standard PDB mirror layout.
    Falls back to recursive glob if no matches found.
    """
    ids = []
    # Standard PDB layout: {mid2}/{pdb_id}.cif.gz
    matches = list(mmcif_dir.glob("*/*.cif.gz"))
    if not matches:
        matches = list(mmcif_dir.glob("*.cif.gz"))
    for gz in matches:
        pdb_id = gz.stem.split(".")[0].lower()
        ids.append(pdb_id)
        if limit and len(ids) >= limit:
            break
    return ids


@cli.command()
@click.option("--mmcif-dir", required=True, type=click.Path(exists=True), help="PDB mmCIF mirror directory.")
@click.option("--output", required=True, type=click.Path(), help="Output directory for processed data.")
@click.option("--pdb-ids", default=None, help="Comma-separated PDB IDs to process (e.g. 1pfe,8qy0).")
@click.option("--limit", default=None, type=int, help="Max number of structures to process (for sampling).")
@click.option("--max-resolution", default=None, type=float, help="Max resolution in Å (e.g. 4.0).")
@click.option("--no-assembly", is_flag=True, help="Skip assembly expansion (use asymmetric unit).")
@click.option("--workers", default=1, type=int, help="Number of parallel workers.")
def prepare(
    mmcif_dir: str,
    output: str,
    pdb_ids: str | None,
    limit: int | None,
    max_resolution: float | None,
    no_assembly: bool,
    workers: int,
) -> None:
    """Preprocess PDB mmCIF files into training data."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log = logging.getLogger(__name__)

    mmcif_path = Path(mmcif_dir)
    output_path = Path(output)
    # Create output directories once upfront
    (output_path / "structures").mkdir(parents=True, exist_ok=True)
    (output_path / "records").mkdir(parents=True, exist_ok=True)
    use_assembly = not no_assembly

    # Determine PDB IDs to process
    if pdb_ids:
        ids = [pid.strip().lower() for pid in pdb_ids.split(",")]
    else:
        log.info("Scanning %s for mmCIF files...", mmcif_path)
        ids = _collect_pdb_ids(mmcif_path, limit=limit)

    log.info("Processing %d structures (assembly=%s, workers=%d)", len(ids), use_assembly, workers)

    records = []
    failed = 0

    if workers <= 1:
        for i, pdb_id in enumerate(ids):
            result = _process_one(pdb_id, mmcif_path, output_path, use_assembly, max_resolution)
            if result is not None:
                records.append(result)
                log.info(
                    "[%d/%d] %s: %d chains, %d interfaces",
                    i + 1,
                    len(ids),
                    pdb_id,
                    len(result["chains"]),
                    len(result["interfaces"]),
                )
            else:
                failed += 1
                log.info("[%d/%d] %s: skipped", i + 1, len(ids), pdb_id)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_process_one, pid, mmcif_path, output_path, use_assembly, max_resolution): pid
                for pid in ids
            }
            for i, future in enumerate(as_completed(futures)):
                pdb_id = futures[future]
                try:
                    result = future.result()
                except Exception:
                    result = None
                    traceback.print_exc()

                if result is not None:
                    records.append(result)
                    log.info("[%d/%d] %s: %d chains", i + 1, len(ids), pdb_id, len(result["chains"]))
                else:
                    failed += 1

    # Save manifest using the shared utility
    from deepfold3.data.prepare.manifest import save_manifest
    from deepfold3.data.types import Record

    manifest_records = [Record.from_dict(r) for r in records]
    manifest_path = output_path / "manifest.json"
    save_manifest(manifest_records, manifest_path)

    log.info("Done: %d processed, %d skipped/failed.", len(records), failed)


# Allow running as module
if __name__ == "__main__":
    prepare()
