"""Backfill ``release_date`` in existing template JSON files using mmcif_meta.json.

Problem: ``deepfold3 prepare-templates`` was run against PATCHR without the
``--mmcif-cache`` argument, so every hit ended up with ``release_date: null``.
At training time this makes the date-cutoff filter drop *all* templates for
every PATCHR record (the ``release_date is not None`` clause in
``dataset.py:_load_monomer_templates``) — training runs without templates.

Fix: walk the template JSON tree and patch the ``release_date`` on each hit
from a ``scan-mmcif`` cache. No m8 re-parsing, pure JSON edit.

Usage::

    uv run python scripts/backfill_template_dates.py \\
        --templates-dir /data/distillation/patchr_training/templates \\
        --mmcif-meta data/mmcif_meta.json

    # Dry-run (no writes, only counts):
    uv run python scripts/backfill_template_dates.py \\
        --templates-dir /data/distillation/patchr_training/templates \\
        --mmcif-meta data/mmcif_meta.json \\
        --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger("backfill_template_dates")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--templates-dir", type=Path, required=True, help="Root of ``templates/{ab}/{id}.json`` tree.")
    ap.add_argument("--mmcif-meta", type=Path, required=True, help="JSON from ``deepfold3 scan-mmcif``.")
    ap.add_argument("--dry-run", action="store_true", help="Scan + report; do not rewrite files.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not args.templates_dir.is_dir():
        logger.error("templates dir not found: %s", args.templates_dir)
        return 1
    if not args.mmcif_meta.is_file():
        logger.error("mmcif meta not found: %s", args.mmcif_meta)
        return 1

    logger.info("loading %s", args.mmcif_meta)
    meta = json.loads(args.mmcif_meta.read_text())
    dates: dict[str, str] = {k.upper(): v["released"] for k, v in meta.items() if v.get("released")}
    logger.info("loaded %d release dates", len(dates))

    files = sorted(args.templates_dir.rglob("*.json"))
    logger.info("found %d template files", len(files))

    n_files_with_change = 0
    n_hits_total = 0
    n_hits_patched = 0
    n_hits_missing = 0

    for p in files:
        try:
            hits = json.loads(p.read_text())
        except Exception as e:  # noqa: BLE001
            logger.warning("skip %s: %s", p, e)
            continue
        if not isinstance(hits, list):
            logger.warning("skip %s: not a list", p)
            continue

        changed = False
        for h in hits:
            n_hits_total += 1
            if h.get("release_date") is not None:
                continue
            pid_raw = h.get("template_pdb_id")
            if not pid_raw:
                continue
            rd = dates.get(pid_raw.upper())
            if rd is not None:
                h["release_date"] = rd
                changed = True
                n_hits_patched += 1
            else:
                n_hits_missing += 1

        if changed:
            n_files_with_change += 1
            if not args.dry_run:
                p.write_text(json.dumps(hits, indent=2))

    logger.info("files with updates: %d / %d", n_files_with_change, len(files))
    logger.info("hits total: %d", n_hits_total)
    logger.info("hits patched: %d", n_hits_patched)
    logger.info("hits still missing (pdb_id not in meta): %d", n_hits_missing)
    if args.dry_run:
        logger.info("dry-run: no files written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
