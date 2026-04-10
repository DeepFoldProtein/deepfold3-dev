"""``deepfold3 build-ligand-cache`` -- precompute ligand symmetry automorphisms.

Walks the manifest, finds every unique non-polymer (ligand) CCD code,
and runs RDKit's automorphism enumeration once per code. Results land in
a single JSON file (``ligand_symmetry.json`` by default, inside the
training data directory) keyed by CCD code.

The dataloader loads this file at start-up via
``ligand_perms.prime_automorphism_memo`` so the per-step ligand symmetry
lookup is a constant-time dict access — RDKit is never invoked during
training.

Examples::

    deepfold3 build-ligand-cache --data-dir /data/training
    deepfold3 build-ligand-cache --data-dir /data/training --workers 16
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import click

from deepfold3.scripts import cli


def _scan_record(args: tuple[str, str]) -> set[str]:
    """Worker: load one Structure NPZ and return its non-polymer CCD codes."""
    from deepfold3.data import const
    from deepfold3.data.paths import structure_path
    from deepfold3.data.types import Structure

    pdb_id, structures_dir_str = args
    try:
        struct = Structure.load(structure_path(Path(structures_dir_str), pdb_id))
    except Exception:
        return set()
    out: set[str] = set()
    for res in struct.residues:
        if int(res["mol_type"]) != const.MOL_NONPOLYMER:
            continue
        out.add(res["name"].decode("ascii", errors="replace").strip().upper())
    return out


def _compute_one(ccd_code: str) -> tuple[str, list[list[int]]]:
    """Worker: compute automorphisms for one CCD code."""
    from deepfold3.data.symmetry.ligand_perms import compute_ligand_automorphisms

    try:
        from deepfold3.data.ccd import load_ccd_molecule

        mol = load_ccd_molecule(ccd_code)
    except Exception:
        return ccd_code, []
    perms = compute_ligand_automorphisms(mol)
    return ccd_code, [list(p) for p in perms]


@cli.command("build-ligand-cache")
@click.option(
    "--data-dir",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Training data directory containing manifest.json and structures/.",
)
@click.option(
    "--output",
    default=None,
    type=click.Path(path_type=Path),
    help="Output JSON path. Defaults to ``{data_dir}/ligand_symmetry.json``.",
)
@click.option(
    "--ccd",
    default=None,
    type=click.Path(exists=True, path_type=Path),
    help="Optional bulk components.cif (or .cif.gz) — strongly recommended to "
    "avoid per-compound HTTP fetches.",
)
@click.option(
    "--workers",
    default=1,
    type=int,
    show_default=True,
    help="Parallel workers for the RDKit automorphism step.",
)
def build_ligand_cache(
    data_dir: Path,
    output: Path | None,
    ccd: Path | None,
    workers: int,
) -> None:
    """Build the ligand-automorphism cache for a training dataset."""
    from deepfold3.data.prepare.manifest import discover_records
    from deepfold3.scripts._common import setup_cli_logging

    log = setup_cli_logging(__name__)

    if ccd is not None:
        from deepfold3.data.ccd import set_ccd_components_path

        set_ccd_components_path(ccd)

    records = discover_records(data_dir)
    if not records:
        click.echo("Error: no records found in data dir.", err=True)
        sys.exit(1)

    output_path = output or (data_dir / "ligand_symmetry.json")

    # Pass 1: collect unique ligand CCD codes by walking every Structure NPZ.
    log.info("Scanning %d records for non-polymer CCD codes...", len(records))
    structures_dir = data_dir / "structures"
    work = [(r.id, str(structures_dir)) for r in records]

    seen: set[str] = set()
    if workers <= 1:
        for item in work:
            seen.update(_scan_record(item))
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for codes in ex.map(_scan_record, work, chunksize=128):
                seen.update(codes)

    log.info("Found %d unique ligand CCD codes.", len(seen))

    # Pass 2: compute automorphisms for each unique code.
    cache: dict[str, list[list[int]]] = {}
    n_with_sym = 0
    n_failed = 0

    sorted_codes = sorted(seen)
    if workers <= 1:
        for i, code in enumerate(sorted_codes, 1):
            _, perms = _compute_one(code)
            cache[code] = perms
            if len(perms) > 1:
                n_with_sym += 1
            elif not perms:
                n_failed += 1
            if i % 100 == 0:
                log.info("Computed %d/%d (with_sym=%d, failed=%d)", i, len(sorted_codes), n_with_sym, n_failed)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            done = 0
            for code, perms in ex.map(_compute_one, sorted_codes, chunksize=8):
                cache[code] = perms
                if len(perms) > 1:
                    n_with_sym += 1
                elif not perms:
                    n_failed += 1
                done += 1
                if done % 100 == 0:
                    log.info("Computed %d/%d (with_sym=%d, failed=%d)", done, len(sorted_codes), n_with_sym, n_failed)

    log.info(
        "Done: %d ligand codes processed, %d with non-trivial symmetry (>1 perm), %d failed.",
        len(cache),
        n_with_sym,
        n_failed,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(cache, separators=(",", ":"), sort_keys=True))
    log.info("Wrote %d entries to %s", len(cache), output_path)
    click.echo(f"Built ligand cache: {n_with_sym}/{len(cache)} symmetric, → {output_path}")
