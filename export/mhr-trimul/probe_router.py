"""Is the MHR router doing anything, and is what it does outside AF3's reach?

Two statistics, and only the second one settles it (see
``_record_router_stats`` in ``model/ops.py``):

* ``entropy`` = H(p)/log H averaged over pairs. 1.0 means the router never left
  uniform — inert, functionally AF3 TriMul.
* ``g_std``   = the across-pair standard deviation of ``g = H·p``. A router that
  is sharp but the *same* for every pair scales each column block of ``W_o`` by
  a constant, which is a reparameterization of ``W_o`` and therefore inside
  AF3's function class. Only ``g_std > 0`` is routing AF3 could not already do.

**This must be run on real inputs.** ``g_std`` is a property of the router
*composed with the input distribution*: feeding random ``z`` produces a
meaningless number, because the spread it measures would come from the noise
rather than from structure the router is responding to. So the probe pulls real
examples from the run's own eval dataset.

Usage:
    python export/mhr-trimul/probe_router.py --checkpoint runs/df3-stage0-r17/checkpoint_1500.pt
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from pathlib import Path

import torch

from deepfold3.data.module.collate import collate_fn
from deepfold3.data.module.dataset import TrainingDataset
from deepfold3.data.prepare.manifest import discover_records
from deepfold3.model.model import DeepFold3
from deepfold3.model.ops import collect_router_stats, set_router_stats_enabled

OUT = Path(__file__).resolve().parent
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("probe_router")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--n-examples", type=int, default=8)
    ap.add_argument("--recycles", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    blob = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_cfg, train_cfg = blob["model_cfg"], blob["train_cfg"]
    step = blob.get("effective_step", blob.get("global_step"))
    log.info("checkpoint step %s", step)

    model = DeepFold3(model_cfg)
    missing, unexpected = model.load_state_dict(
        {k.removeprefix("module."): v for k, v in blob["model"].items()}, strict=False
    )
    if missing or unexpected:
        log.warning("load: %d missing, %d unexpected", len(missing), len(unexpected))
    model = model.eval().to(args.device)

    ds_cfg = train_cfg.eval_datasets[0]
    records = discover_records(Path(ds_cfg.data_dir))
    log.info("eval dataset %s: %d records", ds_cfg.name, len(records))
    if ds_cfg.cutoff_date:
        records = [
            r
            for r in records
            if r.structure_info and r.structure_info.released and r.structure_info.released > ds_cfg.cutoff_date
        ]
        log.info("after cutoff %s: %d records", ds_cfg.cutoff_date, len(records))

    dataset = TrainingDataset(
        records=records[: args.n_examples * 4],
        data_dir=Path(ds_cfg.data_dir),
        max_tokens=train_cfg.max_tokens,
        max_atoms=train_cfg.max_atoms,
        max_msa_seqs=train_cfg.max_msa_seqs,
        name=ds_cfg.name,
    )

    set_router_stats_enabled(model, True)
    n_done = 0
    for i in range(len(dataset)):
        if n_done >= args.n_examples:
            break
        try:
            features = collate_fn([dataset[i]])
        except Exception as e:  # noqa: BLE001 — a bad record should not end the probe
            log.warning("example %d skipped: %s", i, e)
            continue
        features = {k: (v.to(args.device) if torch.is_tensor(v) else v) for k, v in features.items()}
        features.pop("_record_id", None)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            model.forward_training(
                features,
                num_recycles=args.recycles,
                diffusion_batch_size=1,
                mini_rollout_steps=0,
                mini_rollout_samples=1,
                seed=0,
                # The featurizer here is built without the run's n_msa_crops, so
                # the pre-cropped MSA pool would be short for multi-recycle. Let
                # the model crop internally instead — it changes which MSA rows
                # the trunk sees, not the pair statistics the router reads.
                msa_precropped=False,
            )
        n_done += 1
        log.info("example %d/%d  n_tokens=%d", n_done, args.n_examples, features["res_type"].shape[1])

    stats = collect_router_stats(model)
    summary = {k: v for k, v in stats.items() if "by_module" not in k}
    log.info("\n%s", json.dumps(summary, indent=2))

    rows = []
    for k, v in stats.items():
        m = re.match(r"router/(entropy|g_std)_by_module/(.+)", k)
        if m:
            rows.append(dict(module=m.group(2), metric=m.group(1), value=round(v, 5)))
    wide: dict[str, dict] = {}
    for r in rows:
        wide.setdefault(r["module"], {"module": r["module"]})[r["metric"]] = r["value"]

    def depth(name: str) -> int:
        m = re.search(r"\.(\d+)\.", name)
        return int(m.group(1)) if m else -1

    out = sorted(wide.values(), key=lambda r: ("pairformer" not in r["module"], depth(r["module"])))
    path = Path(args.out) if args.out else OUT / f"router_stats_step{step}.csv"
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["module", "entropy", "g_std"])
        w.writeheader()
        w.writerows(out)
    log.info("wrote %s (%d modules)", path, len(out))

    pf = [r for r in out if "pairformer" in r["module"]]
    if pf:
        log.info("\nPairFormer depth profile (every 8th layer)")
        log.info(f"{'module':58s} {'entropy':>8} {'g_std':>8}")
        for r in pf[::16]:
            log.info(f"{r['module']:58s} {r.get('entropy', float('nan')):8.4f} {r.get('g_std', float('nan')):8.4f}")


if __name__ == "__main__":
    main()
