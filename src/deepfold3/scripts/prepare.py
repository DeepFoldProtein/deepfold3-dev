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

    Tries standard PDB mirror layout (``{mid2}/{pdb_id}.cif.gz``) first,
    then flat ``.cif.gz``, then flat ``.cif`` files.
    Uses lazy iteration to avoid materializing 250k+ Path objects when
    ``limit`` is set.
    """
    # Try glob patterns in priority order; use generators to avoid
    # materializing the full directory listing upfront.
    patterns = ["*/*.cif.gz", "*.cif.gz", "*.cif"]
    matches = None
    for pat in patterns:
        gen = mmcif_dir.glob(pat)
        first = next(gen, None)
        if first is not None:
            from itertools import chain

            matches = chain([first], gen)
            break

    if matches is None:
        return []

    ids = []
    for p in matches:
        ids.append(p.stem.split(".")[0].lower())
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
@click.option(
    "--ccd",
    default=None,
    type=click.Path(exists=True),
    help="Path to components.cif(.gz) for bulk CCD lookup (avoids per-compound downloads).",
)
@click.option("--workers", default=1, type=int, help="Number of parallel workers.")
def prepare(
    mmcif_dir: str,
    output: str,
    pdb_ids: str | None,
    limit: int | None,
    max_resolution: float | None,
    no_assembly: bool,
    ccd: str | None,
    workers: int,
) -> None:
    """Preprocess PDB mmCIF files into training data."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log = logging.getLogger(__name__)

    # Load bulk CCD if provided
    if ccd:
        from deepfold3.data.ccd import set_ccd_components_path

        set_ccd_components_path(ccd)

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


@cli.command("fetch-ccd")
@click.option(
    "--output",
    required=True,
    type=click.Path(),
    help="Directory to save components.cif.gz.",
)
def fetch_ccd(output: str) -> None:
    """Download the full CCD components.cif.gz from wwPDB (~100 MB)."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    from deepfold3.data.ccd import fetch_components_cif

    path = fetch_components_cif(output)
    click.echo(f"CCD components saved to: {path}")


# Allow running as module
if __name__ == "__main__":
    prepare()
