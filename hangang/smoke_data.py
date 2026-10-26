"""Data smoke test: build the training DataLoader exactly as ``deepfold3 train`` does and time it.

Run on a compute node against the packed tree before the first training job.
It catches what a transfer can get wrong without crashing training outright
— a pack that is not registered (every file "missing", every MSA a
self-MSA fallback, templates silently empty), a missing ``ids.txt``, a
config path that still says ``/data`` — and measures the dataloader rate on
Lustre.

    uv run python hangang/smoke_data.py --config $RUN_DIR/config.yaml \\
        --ccd $DATA/components.cif.gz --ccd-cache $DATA/ccd_cache.json \\
        --mmcif-dir $DATA/pdb/divided --batches 50 --workers 16

Per dataset it also loads a few samples directly and reports how many came
back with a real MSA (> 1 row) and with templates — both should be well
above zero for every set (short_monomers has ~2.5 % query-only MSAs).
"""

from __future__ import annotations

import argparse
import logging
import random
import time
from collections import Counter
from pathlib import Path

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ccd", required=True)
    ap.add_argument("--ccd-cache", required=True)
    ap.add_argument("--mmcif-dir", required=True)
    ap.add_argument("--batches", type=int, default=50)
    ap.add_argument("--workers", type=int, default=None, help="Override training.num_workers.")
    ap.add_argument("--per-dataset", type=int, default=8, help="Direct samples per dataset for the MSA/template check.")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("smoke")

    from deepfold3.config_loader import load_config
    from deepfold3.data import pack
    from deepfold3.data.ccd import fetch_ccd_block, set_ccd_components_path
    from deepfold3.model.config import DatasetConfig, ModelConfig, TrainingConfig
    from deepfold3.scripts.train import _apply_overrides
    from deepfold3.training.data import build_dataloaders
    from deepfold3.training.distributed import DistInfo

    set_ccd_components_path(args.ccd)
    fetch_ccd_block("HOH")

    train_cfg, model_cfg = TrainingConfig(), ModelConfig()
    overrides = load_config(args.config)
    for key, val in overrides.get("training", {}).items():
        if key in ("extra_datasets", "eval_datasets"):
            setattr(train_cfg, key, [DatasetConfig(**d) for d in val])
        elif hasattr(train_cfg, key):
            setattr(train_cfg, key, val)
    _apply_overrides(model_cfg, overrides.get("model", {}))
    train_cfg.mmcif_dir = args.mmcif_dir
    train_cfg.ccd_cache_path = args.ccd_cache
    if args.workers is not None:
        train_cfg.num_workers = args.workers

    # Every configured root should resolve to a pack (or be a plain dir).
    problems = 0
    roots = [d.data_dir for d in train_cfg.extra_datasets] + [train_cfg.pdb_chain_cache_dir, args.mmcif_dir]
    for root in roots:
        if not root:
            continue
        p = Path(root)
        if pack.register(p):
            rd = pack.reader_for(p)
            log.info("pack  %-70s %10d files %6d shards", root, len(rd), rd.meta["n_shards"])
        elif p.is_dir():
            log.warning("plain %-70s (no _pack: reading loose files)", root)
        else:
            log.error("MISSING %s", root)
            problems += 1
    if problems:
        raise SystemExit(f"{problems} configured data root(s) missing")

    dist = DistInfo(rank=0, world_size=1, local_rank=0, device=torch.device("cpu"))
    train_dl, _ = build_dataloaders(
        cfg=train_cfg, model_cfg=model_cfg, dist=dist, records=[], eval_records=[], distillation_records=[]
    )

    # Direct per-dataset check: real MSA rows and templates.
    ds = train_dl.dataset
    subsets = getattr(ds, "datasets", None) or [ds]
    for sub in subsets:
        name = getattr(sub, "name", type(sub).__name__)
        n = len(sub)
        idx = random.Random(0).sample(range(n), min(args.per_dataset, n))
        stats: Counter[str] = Counter()
        t0 = time.perf_counter()
        for i in idx:
            try:
                x = sub[i]
            except Exception as e:  # noqa: BLE001
                stats["error"] += 1
                log.error("%s[%d]: %s", name, i, e)
                continue
            msa, mmask = x.get("msa"), x.get("msa_mask")
            if msa is not None and mmask is not None and msa.numel():
                rows = msa[mmask.any(dim=-1)]
                # Pre-cropped MSAs repeat per recycle; a query-only MSA is one unique row.
                if rows.shape[0] and torch.unique(rows, dim=0).shape[0] > 1:
                    stats["msa>1"] += 1
            tm = x.get("template_mask")
            if tm is not None and bool(tm.sum() > 0):
                stats["templates"] += 1
        dt = (time.perf_counter() - t0) / max(1, len(idx))
        log.info("%-18s %9d records | %d samples: %s | %.0f ms/sample", name, n, len(idx), dict(stats), dt * 1e3)

    # Throughput through the real DataLoader (workers, collate, mixture sampler).
    t0 = time.perf_counter()
    first = None
    for b, _batch in enumerate(train_dl, 1):
        if first is None:
            first = time.perf_counter() - t0
        if b >= args.batches:
            break
    dt = time.perf_counter() - t0
    log.info(
        "DataLoader: %d batches in %.1fs (first after %.1fs) -> %.2f batches/s with %d workers",
        b,
        dt,
        first or 0.0,
        b / dt,
        train_cfg.num_workers,
    )


if __name__ == "__main__":
    main()
