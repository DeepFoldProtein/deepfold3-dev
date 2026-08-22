"""Cross-check ``probe_pose_sigma.py`` against the REAL eval code path.

``probe_pose_sigma.py`` hand-rolls the trunk / rollout / noising so it can sweep
σ on a grid. That freedom is only worth anything if the hand-rolled path agrees
with ``run_eval`` on the one quantity both compute: ``fape_inter`` at the
training σ draw. This script runs the untouched path —
``model.forward_training(...)`` with ``run_eval``'s exact kwargs, then
``training_loss(...)`` with the stage-2 loss config — and reports the mean the
way the trainer does (including single-chain crops, which contribute a hard 0.0
because ``fape_loss._reduce`` has no cross-chain pairs to average).

Prints both the diluted mean (comparable to the 0.1835 in the log) and the
multi-chain-only mean (comparable to the probe).

Usage: CUDA_VISIBLE_DEVICES=i uv run python scripts/probe_pose_sigma_validate.py --shard i --num-shards 8 --n 30
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch

from deepfold3.config_loader import load_config
from deepfold3.data.module.collate import collate_fn
from deepfold3.data.module.dataset import TrainingDataset
from deepfold3.data.prepare.manifest import discover_records
from deepfold3.model.model import DeepFold3
from deepfold3.training.collectives import slice_batch_dict
from deepfold3.training.data import _apply_cutoff
from deepfold3.training.loss import training_loss


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/df3-stage1-r14/checkpoint_34000.pt")
    ap.add_argument("--config", default="configs/stage2.yaml")
    ap.add_argument("--data-dir", default="/data/distillation/patchr_training")
    ap.add_argument("--cutoff-date", default="2021-09-30")
    ap.add_argument("--out", default="/tmp/pose_validate")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--seed", type=int, default=101)
    args = ap.parse_args()

    dev = "cuda:0"
    torch.manual_seed(1000 + args.shard)

    blob = torch.load(args.ckpt, map_location="cpu", weights_only=False, mmap=True)
    model_cfg, train_cfg = blob["model_cfg"], blob["train_cfg"]
    state = dict(blob["model"])
    if blob.get("ema") and blob["ema"].get("shadow"):
        state.update(blob["ema"]["shadow"])
    model = DeepFold3(model_cfg)
    model.load_state_dict(state, strict=False)
    del blob, state
    model.to(dev).eval()

    # The logged 0.1835 came from stage2's loss config running on these (stage1)
    # weights — apply the stage2 training overrides onto the ckpt's train_cfg.
    ycfg = load_config(args.config).get("training", {})
    applied = {}
    for k, v in ycfg.items():
        if hasattr(train_cfg, k) and not isinstance(v, (list, dict)):
            setattr(train_cfg, k, v)
            applied[k] = v
    print(f"[shard {args.shard}] stage2 overrides applied: {applied}", flush=True)

    ds = TrainingDataset(
        records=_apply_cutoff(discover_records(Path(args.data_dir)), args.cutoff_date, keep_after=True),
        data_dir=Path(args.data_dir), is_distillation=False, name="patchr_eval",
        max_tokens=train_cfg.max_tokens, max_atoms=train_cfg.max_atoms, max_msa_seqs=train_cfg.max_msa_seqs,
        mmcif_dir=Path(train_cfg.mmcif_dir) if train_cfg.mmcif_dir else None,
        max_templates=model_cfg.data.max_templates, template_cutoff_days=model_cfg.data.template_cutoff_days,
        max_template_date=model_cfg.data.max_template_date,
        default_template_cutoff_date=model_cfg.data.default_template_cutoff_date,
        prefer_monomer_templates=model_cfg.data.prefer_monomer_templates,
        max_neighborhood=model_cfg.data.max_neighborhood,
        contiguous_crop_weight=model_cfg.data.contiguous_crop_weight,
        spatial_crop_weight=model_cfg.data.spatial_crop_weight,
        interface_crop_weight=model_cfg.data.interface_crop_weight,
        interface_distance=model_cfg.data.interface_distance,
        max_chains=model_cfg.data.max_chains, atomize_prob=0.0,
        ccd_cache_path=train_cfg.ccd_cache_path or None,
        pdb_chain_cache_dir=train_cfg.pdb_chain_cache_dir or None,
        pdb_chain_cache_mem=train_cfg.pdb_chain_cache_mem,
        msa_crop_size=model_cfg.msa_crop_size,
        n_msa_crops=(train_cfg.max_recycles + 1) if train_cfg.msa_precrop else 0,
        template_struct_space=True,
    )
    order = np.random.default_rng(args.seed).permutation(len(ds)).tolist()[args.shard :: args.num_shards]

    out_path = Path(args.out) / f"validate_{args.shard:02d}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fh = out_path.open("w", newline="")
    wr = csv.writer(fh)
    wr.writerow(["idx", "n_chains", "sigma", "fape", "fape_inter", "tm_true", "itm_true"])

    done, t0 = 0, time.time()
    for idx in order:
        if done >= args.n:
            break
        try:
            batch = collate_fn([ds[idx]], bucket_pad=True, fixed_bucket_n=train_cfg.max_tokens)
        except Exception:  # noqa: BLE001
            continue
        features = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
        try:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                # verbatim run_eval kwargs (eval.py:160-173)
                pred = model.forward_training(
                    features,
                    num_recycles=min(train_cfg.max_recycles, 4),
                    diffusion_batch_size=1,
                    mini_rollout_steps=train_cfg.mini_rollout_steps,
                    mini_rollout_samples=1,
                    seed=0,
                    msa_precropped=train_cfg.msa_precrop,
                    chain_perm_gate_k=(train_cfg.chain_perm_gate_k if train_cfg.chain_perm_stability_gate else 1),
                    permute_gt_before_noise=train_cfg.permute_gt_before_noise,
                )
                _, ld = training_loss(slice_batch_dict(pred, 0, 1), slice_batch_dict(features, 0, 1), train_cfg)
            m = features["seq_mask"][0].bool()
            nch = int(features["asym_id"][0][m].unique().numel())
            # the σ the trainer actually drew for this example — lets the real
            # path be overlaid on the probe's σ grid instead of only compared in
            # aggregate (a mean difference is ambiguous between "wrong curve"
            # and "wrong σ distribution"; the overlay is not).
            sig = float(pred["noise_levels"].flatten()[0])
            wr.writerow([idx, nch, sig, float(ld["fape"]), float(ld["fape_inter"]),
                         float(ld.get("tm_true", float("nan"))), float(ld.get("itm_true", float("nan")))])
            fh.flush()
            done += 1
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001
            print(f"  [skip {idx}] {type(e).__name__}: {e}", flush=True)

    fh.close()
    print(f"[shard {args.shard}] {done} examples in {time.time() - t0:.0f}s → {out_path}", flush=True)


if __name__ == "__main__":
    main()
