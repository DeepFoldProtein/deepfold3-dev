"""``deepfold3 scan-mmcif`` -- precompute mmCIF metadata into a JSON cache.

Walks a local PDB mmCIF mirror and writes a ``{pdb_id: {resolution, method,
deposited, released}}`` JSON dictionary. The ``released`` field is the
earliest ``_pdbx_audit_revision_history.revision_date`` (= initial release
date), matching AlphaFold 3's template date-cutoff convention.

This is the only canonical source for the original release date — wwPDB's
``entries.idx`` carries deposition date but not release date, and the
``holdings/`` JSONs carry only the *last modified* timestamp.

The output JSON is consumed by ``deepfold3 enrich-manifest --mmcif-cache``.

Examples::

    deepfold3 scan-mmcif --data-dir /data/training \\
        --mmcif-dir ~/deepfold3/pdb/divided \\
        --output mmcif_meta.json --workers 32

    deepfold3 scan-mmcif --ids-file pdb_ids.txt \\
        --mmcif-dir ~/deepfold3/pdb/divided \\
        --output mmcif_meta.json --workers 32
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import click

from deepfold3.scripts import cli


def _scan_one(args: tuple[str, str]) -> tuple[str, dict | None]:
    """Worker: read metadata from one PDB entry's mmCIF.

    Tries the canonical wwPDB layout (``{mid2}/{pdb_id}.cif.gz``) directly
    so each entry is one ``stat`` instead of the 9-candidate scan in
    ``find_pdb_mmcif_path``. Falls back to the full lookup only on miss.

    Returns ``(pdb_id, fields | None)``. Module-level for picklability.
    """
    from deepfold3.data.prepare.structure import extract_mmcif_metadata

    pdb_id, mmcif_dir_str = args
    pdb_lower = pdb_id.lower()
    direct = Path(mmcif_dir_str) / pdb_lower[1:3] / f"{pdb_lower}.cif.gz"
    if direct.is_file():
        cif: Path | None = direct
    else:
        from deepfold3.data.parse.templates import find_pdb_mmcif_path

        cif = find_pdb_mmcif_path(pdb_id, Path(mmcif_dir_str))
    if cif is None:
        return pdb_id, None
    si = extract_mmcif_metadata(cif)
    if si is None:
        return pdb_id, None
    return pdb_id, {
        "resolution": si.resolution,
        "method": si.method,
        "deposited": si.deposited,
        "released": si.released,
    }


@cli.command("scan-mmcif")
@click.option(
    "--mmcif-dir",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Local PDB mmCIF mirror (e.g. ~/deepfold3/pdb/divided).",
)
@click.option(
    "--output",
    required=True,
    type=click.Path(path_type=Path),
    help="Path to write the JSON metadata cache.",
)
@click.option(
    "--data-dir",
    default=None,
    type=click.Path(exists=True, path_type=Path),
    help="Training data directory; PDB IDs are taken from its manifest.json. Mutually exclusive with --ids-file.",
)
@click.option(
    "--ids-file",
    default=None,
    type=click.Path(exists=True, path_type=Path),
    help="Text file with one PDB ID per line. Mutually exclusive with --data-dir.",
)
@click.option(
    "--workers",
    default=1,
    type=int,
    show_default=True,
    help="Parallel workers (CPU-bound gemmi parse).",
)
def scan_mmcif(
    mmcif_dir: Path,
    output: Path,
    data_dir: Path | None,
    ids_file: Path | None,
    workers: int,
) -> None:
    """Walk a PDB mmCIF mirror and dump per-entry metadata to JSON."""
    from deepfold3.scripts._common import setup_cli_logging

    log = setup_cli_logging(__name__)

    if data_dir is None and ids_file is None:
        click.echo("Error: pass --data-dir or --ids-file.", err=True)
        sys.exit(1)
    if data_dir is not None and ids_file is not None:
        click.echo("Error: --data-dir and --ids-file are mutually exclusive.", err=True)
        sys.exit(1)

    if data_dir is not None:
        from deepfold3.data.prepare.manifest import load_manifest
        from deepfold3.scripts._common import require_manifest

        records = load_manifest(require_manifest(data_dir))
        pdb_ids = [r.id for r in records]
    else:
        from deepfold3.data.prepare.manifest import load_id_list

        pdb_ids = sorted(load_id_list(ids_file))

    log.info("Scanning %d entries from %s (workers=%d)...", len(pdb_ids), mmcif_dir, workers)
    todo = [(pid, str(mmcif_dir)) for pid in pdb_ids]

    results: dict[str, dict] = {}
    missing = 0
    if workers <= 1:
        for i, item in enumerate(todo):
            pdb_id, fields = _scan_one(item)
            if fields is None:
                missing += 1
            else:
                results[pdb_id] = fields
            if (i + 1) % 5000 == 0:
                log.info("  parsed %d/%d (matched=%d, missing=%d)", i + 1, len(todo), len(results), missing)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            done = 0
            for pdb_id, fields in ex.map(_scan_one, todo, chunksize=256):
                done += 1
                if fields is None:
                    missing += 1
                else:
                    results[pdb_id] = fields
                if done % 5000 == 0:
                    log.info("  parsed %d/%d (matched=%d, missing=%d)", done, len(todo), len(results), missing)

    log.info("Done: %d matched, %d missing (file not found / parse failed).", len(results), missing)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, sort_keys=True))
    log.info("Wrote %d entries to %s", len(results), output)
    click.echo(f"Scanned {len(results)}/{len(pdb_ids)} entries → {output}")
