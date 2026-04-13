"""``deepfold3 build-masks`` -- generate PATCHR ATLAS token masks.

For each record in the manifest, looks up the corresponding PATCHR ATLAS
inpainting metadata JSON and builds a per-token ``"fixed"`` mask channel:

- 1.0 = fully experimental (fixed in the inpainting)
- 0.0 = fully inpainted (predicted, not experimental)
- fraction = partially fixed

The masks are saved as NPZ files in ``{data_dir}/masks/{ab}/{record_id}.npz``
and loaded automatically by ``TrainingDataset`` to scale the ``exp_resolved``
loss target.

Usage::

    deepfold3 build-masks --data-dir data/pdb_training --json-dir data/patchr/json --workers 16
"""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path

import click
import numpy as np

from deepfold3.scripts import cli
from deepfold3.scripts._common import require_manifest, setup_cli_logging


def _build_one(args: tuple) -> str | None:
    """Build mask for a single record. Returns record_id on success, None on skip."""
    record_dict, data_dir, json_dir = args

    from deepfold3.data.masking import MASK_FIXED, TokenMask
    from deepfold3.data.paths import mask_path, structure_path
    from deepfold3.data.tokenize.tokenizer import Tokenizer
    from deepfold3.data.types import Input, Record, Structure

    record = Record.from_dict(record_dict)

    # Find the matching JSON: {pdb_id}.json (renamed from PATCHR ATLAS naming)
    pdb_id = record.id.lower()
    json_path = Path(json_dir) / f"{pdb_id}.json"
    if not json_path.exists():
        return None

    # Load structure and tokenize (needed for residue→token mapping)
    struct_path_val = structure_path(Path(data_dir) / "structures", record.id)
    if not struct_path_val.exists():
        return None
    structure = Structure.load(struct_path_val)
    tokenizer = Tokenizer()
    inp = Input(structure=structure, paired_msa={}, unpaired_msa={})
    tokenized = tokenizer.tokenize(inp)

    # Build the fixed mask
    fixed = TokenMask.fixed_from_patchr(
        json_path,
        tokenized.tokens,
        structure.chains,
        chain_infos=record.chains,
    )

    # Save
    out_path = mask_path(Path(data_dir) / "masks", record.id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(out_path), **{MASK_FIXED: fixed})

    return record.id


@cli.command("build-masks")
@click.option("--data-dir", required=True, type=click.Path(exists=True), help="Training data directory with manifest.")
@click.option("--json-dir", required=True, type=click.Path(exists=True), help="Directory of PATCHR inpainting JSONs.")
@click.option("--workers", default=1, type=int, help="Parallel workers.")
def build_masks(data_dir: str, json_dir: str, workers: int) -> None:
    """Build token masks from PATCHR ATLAS inpainting metadata."""
    log = setup_cli_logging(__name__)
    data_path = Path(data_dir)
    require_manifest(data_path)

    from deepfold3.data.prepare.manifest import load_manifest

    records = load_manifest(data_path / "manifest.json")
    log.info("Loaded %d records from manifest", len(records))

    # Serialize records to dicts for multiprocessing
    args_list = [(r.to_dict(), data_dir, json_dir) for r in records]

    built = 0
    skipped = 0
    if workers > 1:
        with mp.Pool(workers) as pool:
            for i, result in enumerate(pool.imap_unordered(_build_one, args_list, chunksize=64)):
                if result is not None:
                    built += 1
                else:
                    skipped += 1
                if (i + 1) % 5000 == 0:
                    log.info("  %d / %d processed (%d built, %d skipped)", i + 1, len(args_list), built, skipped)
    else:
        for i, args in enumerate(args_list):
            result = _build_one(args)
            if result is not None:
                built += 1
            else:
                skipped += 1
            if (i + 1) % 5000 == 0:
                log.info("  %d / %d processed (%d built, %d skipped)", i + 1, len(args_list), built, skipped)

    log.info("Done: %d masks built, %d skipped (no JSON or structure).", built, skipped)
