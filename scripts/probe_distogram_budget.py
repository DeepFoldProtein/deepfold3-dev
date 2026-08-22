"""Where does the distogram's gradient budget actually go? (no gradient needed)

Sizes the proposed cross-chain reweight / focal on the distogram before any knob is
guessed. `distogram_loss` is a plain mean over all valid pairs
([`loss.py:843-851`](../src/deepfold3/training/loss.py#L843)), so per-pair:

    nll_ij   = -log p(target_bin)                      ← its share of the LOSS
    dL/dlogits_ij = (p_ij - onehot_ij) / n_pairs       ← its share of the GRADIENT

The head is a single linear map, so the gradient reaching the pair representation at
(i, j) is `W^T (p_ij - y_ij) / n_pairs`; ‖p − y‖₂ is therefore the exact per-pair
gradient weight up to a factor shared by every pair. Counting pairs is NOT the same
thing: far cross-chain pairs are numerous but sit in the saturated last bin, are
already well predicted, and contribute almost no gradient.

Decomposes both shares over {intra, cross} × {contact <8 Å, near 8-15, mid 15-21.7,
far ≥21.7}, split by rollout pose quality, and additionally reports what a focal
weight `(1 - p_t)^γ` would do to those shares for γ = 1, 2 — so γ and the category
weight are read off measured data instead of picked.

Usage (8 GPUs):
    for i in $(seq 0 7); do
      CUDA_VISIBLE_DEVICES=$i OMP_NUM_THREADS=8 uv run python scripts/probe_distogram_budget.py \
        --shard $i --num-shards 8 --n 40 --out /tmp/disto_budget &
    done; wait
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
from deepfold3.training.loss import fape_loss, permute_gt_for_diffusion

BANDS = [("contact", 0.0, 8.0), ("near", 8.0, 15.0), ("mid", 15.0, 21.6875), ("far", 21.6875, 1e9)]
FAPE_KW = dict(clip_distance=10.0, clip_distance_inter=30.0, normalize=True,
               all_atom_points=True, mode="both", chunk_size=64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/df3-stage1-r14/checkpoint_34000.pt")
    ap.add_argument("--config", default="configs/stage2.yaml")
    ap.add_argument("--data-dir", default="/data/distillation/patchr_training")
    ap.add_argument("--cutoff-date", default="2021-09-30")
    ap.add_argument("--out", default="/tmp/disto_budget")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--seed", type=int, default=101)
    args = ap.parse_args()

    dev = "cuda:0"
    torch.manual_seed(4000 + args.shard)

    blob = torch.load(args.ckpt, map_location="cpu", weights_only=False, mmap=True)
    model_cfg, train_cfg = blob["model_cfg"], blob["train_cfg"]
    state = dict(blob["model"])
    if blob.get("ema") and blob["ema"].get("shadow"):
        state.update(blob["ema"]["shadow"])
    model = DeepFold3(model_cfg)
    model.load_state_dict(state, strict=False)
    del blob, state
    model.to(dev).eval()
    for k, v in load_config(args.config).get("training", {}).items():
        if hasattr(train_cfg, k) and not isinstance(v, (list, dict)):
            setattr(train_cfg, k, v)

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

    R, gate_k, dh = train_cfg.max_recycles, train_cfg.chain_perm_gate_k, model.diffusion_head
    head = model.distogram_head
    breaks = torch.linspace(head.first_break, head.last_break, head.num_bins - 1, device=dev)

    out_path = Path(args.out) / f"budget_{args.shard:02d}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fh = out_path.open("w", newline="")
    wr = csv.writer(fh)
    wr.writerow(["idx", "record_id", "n_chains", "roll_fape_inter", "block", "band", "n",
                 "sum_nll", "sum_grad", "sum_fl1", "sum_fl2",
                 "paired_min", "paired_med", "unpaired_min", "unpaired_med", "n_chain_no_paired"])

    # Capture the per-chain MSA depths the dataset actually loaded for the chains
    # that survived the crop. Doing it from the featurized `msa` tensor is not
    # possible: the dense UNPAIRED block also spans several chains on one row
    # (featurizer.py:806-813), so "row covers >=2 chains" does NOT mean "paired".
    cap: dict = {}
    _orig_load_msa = ds._load_cropped_msa

    def _capture(record, tokenized, surviving, cols):
        pmsa, umsa = _orig_load_msa(record, tokenized, surviving, cols)
        cap.clear()
        cap["record_id"] = record.id
        cap["paired"] = {k: int(v.sequences.shape[0]) for k, v in pmsa.items()}
        cap["unpaired"] = {k: int(v.sequences.shape[0]) for k, v in umsa.items()}
        cap["surviving"] = sorted(int(c) for c in surviving)
        return pmsa, umsa

    ds._load_cropped_msa = _capture

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
                if R > 0:
                    emb_det, *_ = model.run_evoformer(features, R - 1, seed=0, msa_precropped=train_cfg.msa_precrop, crop_offset=0)
                    features["_prev_pair"], features["_prev_single"] = emb_det["pair"], emb_det["single"]
                embeddings, rel_feat, target_feat, seq_mask, _ = model.run_evoformer(
                    features, 0, seed=R, msa_precropped=train_cfg.msa_precrop, crop_offset=R
                )
                features.pop("_prev_pair", None)
                features.pop("_prev_single", None)
                embeddings["target_feat"] = target_feat
                ref_kw = model._ref_kwargs(features)

                orig = dh.cfg.steps, dh.cfg.num_samples
                dh.cfg.steps, dh.cfg.num_samples = train_cfg.mini_rollout_steps, max(train_cfg.mini_rollout_samples, gate_k)
                try:
                    mini = dh.sample(embeddings=embeddings, seq_mask=seq_mask, rel_feat=rel_feat, **ref_kw)["atom_positions"]
                finally:
                    dh.cfg.steps, dh.cfg.num_samples = orig

                f0 = slice_batch_dict(features, 0, 1)
                gt_pos, gt_mask, _, c_keep = permute_gt_for_diffusion(
                    {"mini_rollout_pos": mini[0, : train_cfg.mini_rollout_samples], "chain_perm_gate_pos": mini[0]},
                    f0, stability_gate_k=gate_k,
                )
                asym = f0["asym_id"]
                seq_m = f0["seq_mask"].bool()
                if c_keep is not None:
                    seq_m = seq_m & c_keep
                n_chains = int(asym[seq_m].unique().numel())
                if n_chains < 2:
                    continue

                atom_m = gt_mask.bool()
                rep = f0["token_to_rep_atom"]
                ar = torch.arange(rep.shape[0], device=dev)
                gt_rep = gt_pos[ar, rep].float()
                d_gt = torch.cdist(gt_rep, gt_rep)

                # `distogram_loss` masks on token_mask only (not rep-atom presence),
                # so mirror that exactly — this is a budget audit of the real loss.
                pm = (seq_m.unsqueeze(1) & seq_m.unsqueeze(0)).clone()
                pm.fill_diagonal_(False)

                logits = head(embeddings["pair"])[0].float()
                logp = torch.log_softmax(logits, -1)
                p = logp.exp()
                tgt = torch.bucketize(d_gt, breaks)
                nll = -logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)  # [N, N]
                p_t = p.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
                # ‖p − onehot‖₂ = sqrt(Σp² − 2p_t + 1): the exact per-pair gradient weight
                gnorm = (p.pow(2).sum(-1) - 2 * p_t + 1.0).clamp(min=0).sqrt()
                fl1 = (1.0 - p_t) * nll
                fl2 = (1.0 - p_t).pow(2) * nll

                fm = f0["frames_mask"].bool() & atom_m[f0["frame_token_index"], f0["frame_atom_index"]].all(-1)
                pmf = fm.unsqueeze(-1) & f0["pair_mask"].bool()
                mf = atom_m.float().unsqueeze(-1)
                gt_c = (gt_pos.float() - (gt_pos.float() * mf).sum((0, 1), keepdim=True) / mf.sum().clamp(min=1e-6)) * mf
                _, fi = fape_loss(mini[0, 0].unsqueeze(0).float(), gt_c, None, None, None,
                                  f0["frame_atom_index"], f0["frame_token_index"], fm, seq_m, pmf,
                                  atom_mask=atom_m, asym_id=asym, **FAPE_KW)

                surv = cap.get("surviving", [])
                pdep = [cap.get("paired", {}).get(c, 0) for c in surv]
                udep = [cap.get("unpaired", {}).get(c, 0) for c in surv]
                def med(a):
                    return float(np.median(a)) if a else float("nan")
                p_min, p_med = (min(pdep) if pdep else float("nan")), med(pdep)
                u_min, u_med = (min(udep) if udep else float("nan")), med(udep)
                n_nopair = int(sum(1 for d in pdep if d <= 1))
                rid = cap.get("record_id", f"idx{idx}")
                msa_cols = [rid, p_min, p_med, u_min, u_med, n_nopair]

                same = asym.unsqueeze(1) == asym.unsqueeze(0)
                rows = []
                for bname, blk in (("intra", pm & same), ("cross", pm & ~same)):
                    for band, lo, hi in BANDS:
                        m = blk & (d_gt >= lo) & (d_gt < hi)
                        if not m.any():
                            rows.append([idx, msa_cols[0], n_chains, float(fi), bname, band, 0,
                                         0.0, 0.0, 0.0, 0.0, *msa_cols[1:]])
                            continue
                        rows.append([idx, msa_cols[0], n_chains, float(fi), bname, band, int(m.sum()),
                                     float(nll[m].sum()), float(gnorm[m].sum()),
                                     float(fl1[m].sum()), float(fl2[m].sum()), *msa_cols[1:]])
            wr.writerows(rows)
            fh.flush()
            done += 1
            if done % 10 == 0:
                print(f"[shard {args.shard}] {done}/{args.n} ({(time.time() - t0) / done:.1f}s/ex)", flush=True)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001
            print(f"  [skip {idx}] {type(e).__name__}: {e}", flush=True)

    fh.close()
    print(f"[shard {args.shard}] wrote {done} → {out_path} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
