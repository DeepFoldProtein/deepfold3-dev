"""Does serving more MSA rows rescue the pose failures? (paired A/B, no gradient)

``ModelConfig.msa_crop_size`` is the number of MSA rows the Evoformer consumes
per recycle. r14 trains at 1024 (deliberate — throughput, plus AF3 §2.3's
random-``k`` mask makes the model row-count invariant) and, until the inference
default changed, was *served* at 1024 too. This measures what the model gains
when it is served the whole pool instead.

Both settings run on the **same crop, in the same pass** — the crop is redrawn on
every ``__getitem__``, so two separate runs would not be paired. The trunk is
re-run per setting (that is the whole point); the diffusion rollout follows.

Also records the pool depth actually available, because the A/B is a no-op for
any example whose MSA is shallower than the smaller crop — and per
``export/pose-decomposition/`` Result 7 the failing examples are exactly the
shallow ones, so the headline number must be read stratified.

Usage (8 GPUs):
    for i in $(seq 0 7); do
      CUDA_VISIBLE_DEVICES=$i OMP_NUM_THREADS=8 uv run python scripts/probe_msa_crop_ab.py \
        --shard $i --num-shards 8 --n 10 --out /tmp/msa_ab &
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

CROPS = (1024, 16384)  # trained crop vs full pool
MAX_CC = 40_000
FAPE_KW = dict(clip_distance=10.0, clip_distance_inter=30.0, normalize=True,
               all_atom_points=True, mode="both", chunk_size=64)


def ridge_r(x_tr, y_tr, x_te, y_te, lam: float = 10.0) -> float:
    if x_tr.shape[0] < 200 or x_te.shape[0] < 200:
        return float("nan")
    xt, xe = x_tr.double(), x_te.double()
    mu, sd = xt.mean(0, keepdim=True), xt.std(0, keepdim=True).clamp(min=1e-6)
    xt, xe = (xt - mu) / sd, (xe - mu) / sd
    xt = torch.cat([xt, torch.ones(xt.shape[0], 1, device=xt.device, dtype=torch.float64)], 1)
    xe = torch.cat([xe, torch.ones(xe.shape[0], 1, device=xe.device, dtype=torch.float64)], 1)
    a = xt.T @ xt
    a.diagonal().add_(lam)
    try:
        w = torch.linalg.solve(a, xt.T @ y_tr.double())
    except Exception:  # noqa: BLE001
        return float("nan")
    p = xe @ w
    if not torch.isfinite(p).all():
        return float("nan")
    p = p - p.mean()
    e = y_te.double() - y_te.double().mean()
    if float(p.norm()) < 1e-6 * float(e.norm().clamp(min=1e-12)):
        return float("nan")
    return max(-1.0, min(1.0, float((p @ e) / (p.norm() * e.norm()))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/df3-stage1-r14/checkpoint_34000.pt")
    ap.add_argument("--dataset-config", default="configs/stage2.yaml")
    ap.add_argument("--data-dir", default="/data/distillation/patchr_training")
    ap.add_argument("--cutoff-date", default="2021-09-30")
    ap.add_argument("--out", default="/tmp/msa_ab")
    ap.add_argument("--recycles", type=int, default=10)
    ap.add_argument("--rollout-steps", type=int, default=20)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--seed", type=int, default=101)
    args = ap.parse_args()

    dev = "cuda:0"
    torch.manual_seed(9000 + args.shard)
    gen = np.random.default_rng(9000 + args.shard)

    blob = torch.load(args.ckpt, map_location="cpu", weights_only=False, mmap=True)
    model_cfg, train_cfg = blob["model_cfg"], blob["train_cfg"]
    state = dict(blob["model"])
    if blob.get("ema") and blob["ema"].get("shadow"):
        state.update(blob["ema"]["shadow"])
    model = DeepFold3(model_cfg)
    model.load_state_dict(state, strict=False)
    del blob, state
    model.to(dev).eval()
    for k, v in load_config(args.dataset_config).get("training", {}).items():
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
        msa_crop_size=1024, n_msa_crops=0,  # n_msa_crops=0 => featurizer does NOT crop
        template_struct_space=True,
    )
    order = np.random.default_rng(args.seed).permutation(len(ds)).tolist()[args.shard :: args.num_shards]

    dh, gate_k, R = model.diffusion_head, train_cfg.chain_perm_gate_k, args.recycles
    out_path = Path(args.out) / f"msa_ab_{args.shard:02d}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fh = out_path.open("w", newline="")
    wr = csv.writer(fh)
    wr.writerow(["idx", "record_id", "n_chains", "n_tokens", "msa_pool_rows"]
                + [f"{m}_c{c}" for c in CROPS for m in ("fape_inter", "fape_intra", "r_cc_z")])

    done, t0 = 0, time.time()
    for idx in order:
        if done >= args.n:
            break
        try:
            batch = collate_fn([ds[idx]], bucket_pad=True, fixed_bucket_n=train_cfg.max_tokens)
        except Exception:  # noqa: BLE001
            continue
        rid = getattr(ds.records[idx], "id", f"idx{idx}")
        features = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
        pool = int(features["msa"].shape[-2])
        try:
            row, gt_cache = [], {}
            for crop in CROPS:
                model.cfg.msa_crop_size = crop
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    if R > 0:
                        emb_det, *_ = model.run_evoformer(features, R - 1, seed=0, crop_offset=0)
                        features["_prev_pair"], features["_prev_single"] = emb_det["pair"], emb_det["single"]
                    embeddings, rel_feat, target_feat, seq_mask, _ = model.run_evoformer(
                        features, 0, seed=R, crop_offset=R
                    )
                    features.pop("_prev_pair", None)
                    features.pop("_prev_single", None)
                    embeddings["target_feat"] = target_feat
                    ref_kw = model._ref_kwargs(features)

                    o = dh.cfg.steps, dh.cfg.num_samples
                    dh.cfg.steps, dh.cfg.num_samples = args.rollout_steps, gate_k
                    try:
                        mini = dh.sample(embeddings=embeddings, seq_mask=seq_mask, rel_feat=rel_feat, **ref_kw)[
                            "atom_positions"
                        ]
                    finally:
                        dh.cfg.steps, dh.cfg.num_samples = o

                    f0 = slice_batch_dict(features, 0, 1)
                    # Resolve the chain permutation from THIS setting's own rollout,
                    # exactly as training does; the GT labelling is per-rollout.
                    gt_pos, gt_mask, _, c_keep = permute_gt_for_diffusion(
                        {"mini_rollout_pos": mini[0, :1], "chain_perm_gate_pos": mini[0]},
                        f0, stability_gate_k=gate_k,
                    )
                    asym = f0["asym_id"]
                    seq_m = f0["seq_mask"].bool()
                    if c_keep is not None:
                        seq_m = seq_m & c_keep
                    n_chains = int(asym[seq_m].unique().numel())
                    if n_chains < 2:
                        raise RuntimeError("single-chain crop")
                    atom_m = gt_mask.bool()
                    mf = atom_m.float().unsqueeze(-1)
                    gt_c = (gt_pos.float() - (gt_pos.float() * mf).sum((0, 1), keepdim=True) / mf.sum().clamp(min=1e-6)) * mf
                    fm = f0["frames_mask"].bool() & atom_m[f0["frame_token_index"], f0["frame_atom_index"]].all(-1)
                    pm = fm.unsqueeze(-1) & f0["pair_mask"].bool()
                    fx, fi = fape_loss(mini[0, 0].unsqueeze(0).float(), gt_c, None, None, None,
                                       f0["frame_atom_index"], f0["frame_token_index"], fm, seq_m, pm,
                                       atom_mask=atom_m, asym_id=asym, **FAPE_KW)

                    rep = f0["token_to_rep_atom"]
                    ar = torch.arange(rep.shape[0], device=dev)
                    gt_rep = gt_pos[ar, rep].float()
                    d_gt = torch.cdist(gt_rep, gt_rep)
                    rep_ok = seq_m & atom_m[ar, rep]
                    vld = rep_ok.unsqueeze(1) & rep_ok.unsqueeze(0)
                    vld.fill_diagonal_(False)
                    cc = vld & (asym.unsqueeze(1) != asym.unsqueeze(0))
                    z = embeddings["pair"][0].float()
                    n_all = rep.shape[0]
                    side = torch.zeros(n_all, dtype=torch.bool, device=dev)
                    side[torch.as_tensor(gen.permutation(n_all)[: n_all // 2].copy(), device=dev)] = True

                    def sub(m, _gen=gen, _dev=dev):
                        q = m.nonzero(as_tuple=False)
                        if q.shape[0] > MAX_CC:
                            q = q[torch.as_tensor(_gen.permutation(q.shape[0])[:MAX_CC].copy(), device=_dev)]
                        return q

                    q_tr = sub(cc & side.unsqueeze(1) & side.unsqueeze(0))
                    q_te = sub(cc & (~side).unsqueeze(1) & (~side).unsqueeze(0))
                    rz = ridge_r(z[q_tr[:, 0], q_tr[:, 1]], d_gt[q_tr[:, 0], q_tr[:, 1]],
                                 z[q_te[:, 0], q_te[:, 1]], d_gt[q_te[:, 0], q_te[:, 1]])
                row += [float(fi), float(fx), rz]
                gt_cache["n_chains"], gt_cache["n_tok"] = n_chains, int(seq_m.sum())

            wr.writerow([idx, rid, gt_cache["n_chains"], gt_cache["n_tok"], pool, *row])
            fh.flush()
            done += 1
            if done % 3 == 0:
                print(f"[shard {args.shard}] {done}/{args.n} ({(time.time() - t0) / done:.1f}s/ex)", flush=True)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"  [skip {idx}] OOM", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  [skip {idx}] {type(e).__name__}: {e}", flush=True)

    fh.close()
    print(f"[shard {args.shard}] wrote {done} → {out_path} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
