"""Re-apply quality filters to an existing prepared dataset.

Reads the manifest, loads each structure NPZ, re-runs the chain- and
record-level filters with new thresholds, and writes the updated
manifest back (atomic save, ``.old`` backup). Records whose chains are
all invalidated are dropped from the manifest; others are kept with
updated ``valid`` flags.

Primary use: cleaning up a dataset after adopting stricter filters
(tight clashes, UNK fraction) without re-running the whole prepare
pipeline.
"""

from __future__ import annotations

import logging
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import click

from deepfold3.scripts import cli
from deepfold3.scripts._common import setup_cli_logging

logger = logging.getLogger(__name__)


def _process(
    record_dict: dict,
    structures_dir: str,
    cfg_dict: dict,
) -> tuple[str, dict | None, dict[str, str], int | None]:
    """Worker: load structure, re-filter, return updated record dict.

    Returns ``(pdb_id, record_or_None, reasons, tight_clash_count)``.
    Record is None when the structure is missing (skipped, not dropped).
    """
    from deepfold3.data.paths import structure_path
    from deepfold3.data.prepare.filters import (
        StructureFilterConfig,
        count_tight_clashes,
        filter_structure,
    )
    from deepfold3.data.types import Record, Structure

    pdb_id = record_dict["id"]
    record = Record.from_dict(record_dict)
    p = structure_path(Path(structures_dir), pdb_id)
    if not p.exists():
        return pdb_id, record_dict, {}, None

    structure = Structure.load(p)
    # Preserve prepare-time invalidations: ``filter_structure`` only ever
    # flips ``valid`` True→False (never False→True), so by NOT resetting
    # ``ci.valid = True`` we keep prior verdicts intact. The new filters
    # can only ever invalidate more chains, never un-invalidate.
    cfg = StructureFilterConfig(**cfg_dict)
    n_tight = count_tight_clashes(structure, cfg.tight_clash_dist)
    record, reasons = filter_structure(structure, record, cfg)
    return pdb_id, record.to_dict(), reasons, n_tight


@cli.command("refilter")
@click.option(
    "--data-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Prepared dataset directory (contains manifest.json and structures/).",
)
@click.option(
    "--max-unk-fraction",
    default=None,
    type=float,
    help="Reject chains whose UNK fraction exceeds this (e.g. 0.1 = >10%% UNK).",
)
@click.option(
    "--max-tight-clashes",
    default=None,
    type=int,
    help="Reject records with more than this many sub-tight-clash-dist heavy-atom pairs.",
)
@click.option(
    "--tight-clash-dist",
    default=0.8,
    type=float,
    show_default=True,
    help="Distance (Å) for the tight-clash count.",
)
@click.option(
    "--drop-empty/--keep-empty",
    default=True,
    show_default=True,
    help="Drop records with no remaining valid chains from the manifest.",
)
@click.option("--workers", default=1, type=int, show_default=True, help="Parallel workers.")
@click.option(
    "--dry-run",
    is_flag=True,
    help="Report statistics without writing the manifest back.",
)
def refilter(
    data_dir: str,
    max_unk_fraction: float | None,
    max_tight_clashes: int | None,
    tight_clash_dist: float,
    drop_empty: bool,
    workers: int,
    dry_run: bool,
) -> None:
    """Re-apply quality filters to an existing manifest in-place."""
    setup_cli_logging()
    from deepfold3.data.prepare.manifest import load_manifest, save_manifest

    root = Path(data_dir)
    manifest_path = root / "manifest.json"
    structures_dir = root / "structures"

    records = load_manifest(manifest_path)
    logger.info("Loaded manifest: %d records", len(records))

    if max_unk_fraction is None and max_tight_clashes is None:
        logger.warning("Neither --max-unk-fraction nor --max-tight-clashes set — nothing to do.")
        return

    cfg_dict = dict(
        max_unk_fraction=max_unk_fraction,
        max_tight_clashes=max_tight_clashes,
        tight_clash_dist=tight_clash_dist,
        # Keep the other filters permissive so we only apply the new
        # gates; the existing per-chain filters already ran at prepare
        # time and we don't want to re-invalidate for e.g. length.
        min_residues=0,
        max_residues=10**9,
        max_ca_dist=10**9,
        clash_freq=1.01,  # disables filter_clashing_chains
    )

    record_dicts = [r.to_dict() for r in records]

    updated: list[dict] = []
    n_dropped = 0
    n_changed = 0
    tight_hist: list[int] = []
    reason_counter: Counter[str] = Counter()

    def _finalize(pdb_id: str, new_dict: dict | None, reasons: dict[str, str], n_tight: int | None):
        nonlocal n_dropped, n_changed
        if new_dict is None:
            # Missing structure — keep the original untouched.
            for rd in record_dicts:
                if rd["id"] == pdb_id:
                    updated.append(rd)
                    break
            return
        if n_tight is not None:
            tight_hist.append(n_tight)
        valid_chains = [c for c in new_dict["chains"] if c.get("valid", True)]
        # Bucket reasons for reporting.
        for r in reasons.values():
            key = r.split(" ")[0] if r else "other"
            if "tight-clash" in r:
                key = "tight-clash"
            elif "UNK" in r:
                key = "unk-fraction"
            reason_counter[key] += 1
        if not valid_chains and drop_empty:
            n_dropped += 1
            return
        if reasons:
            n_changed += 1
        updated.append(new_dict)

    if workers <= 1:
        for rd in record_dicts:
            pid, new_dict, reasons, n_tight = _process(rd, str(structures_dir), cfg_dict)
            _finalize(pid, new_dict, reasons, n_tight)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_process, rd, str(structures_dir), cfg_dict) for rd in record_dicts]
            for i, fut in enumerate(as_completed(futures)):
                pid, new_dict, reasons, n_tight = fut.result()
                _finalize(pid, new_dict, reasons, n_tight)
                if (i + 1) % 5000 == 0:
                    logger.info("  processed %d/%d", i + 1, len(record_dicts))

    if tight_hist:
        tight_hist.sort()
        n = len(tight_hist)
        logger.info(
            "tight-clash distribution (< %.2f Å, bonded excluded): min=%d p50=%d p95=%d p99=%d max=%d",
            tight_clash_dist,
            tight_hist[0],
            tight_hist[n // 2],
            tight_hist[max(0, int(n * 0.95) - 1)],
            tight_hist[max(0, int(n * 0.99) - 1)],
            tight_hist[-1],
        )
    logger.info("changed %d records, dropped %d (now %d total)", n_changed, n_dropped, len(updated))
    if reason_counter:
        logger.info("reasons: %s", dict(reason_counter.most_common(10)))

    if dry_run:
        logger.info("dry-run: manifest not written.")
        return

    from deepfold3.data.types import Record

    new_records = [Record.from_dict(rd) for rd in updated]
    save_manifest(new_records, manifest_path)
    logger.info("Wrote %s (%d records, .old backup kept).", manifest_path, len(new_records))


if __name__ == "__main__":
    refilter()  # type: ignore[no-value-for-parameter]
