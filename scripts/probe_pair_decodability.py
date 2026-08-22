"""Is the cross-chain distance LINEARLY DECODABLE from the trunk pair rep?

Closes the decomposition left open by
``export/pose-decomposition/2026-07-31-onestep-vs-rollout-pose.md``: 60% of pose
failures had a blind distogram *read-out* (cross-chain contact AP < 0.2), which is
consistent with either

  (a) the representation genuinely lacks the interface   → trunk-side work
  (b) the distogram head cannot read what pair_cond still carries → denoiser-side work

The r10 homodimer probe (``export/homodimer-collapse-probe/2026-07-21-pair_cond-linear-probe.md``)
showed exactly that dissociation on 3 targets — trunk z / pair_cond / attention bias
all retained r = 0.90 for the true 43 Å separation while E[d] read 0.20. This runs the
same linear probe on a broad multi-chain eval set, grouped by rollout pose quality.

Feature sets probed → true rep-atom distance (ridge, token-disjoint split):
  z         embeddings["pair"]            what the trunk holds
  pair_cond DiffusionStatic.pair_cond     what the diffusion conditioning carries
  bias      LN(pair_cond)·w @ W_pair_logits   the ONLY pairwise path into the denoiser
  rel       rel_feat                      CONTROL: positional only (index diff + chain id),
                                          no 3D — a probe that beats nothing beats this
  Ed        distogram expected distance   the saturated read-out baseline

Everything (distogram AP, rollout fape_inter, probes) is measured in ONE pass per
example, because ``TrainingDataset.__getitem__`` crops randomly — results from
separate runs cannot be joined by record id.

Usage (8 GPUs):
    for i in $(seq 0 7); do
      CUDA_VISIBLE_DEVICES=$i OMP_NUM_THREADS=8 uv run python scripts/probe_pair_decodability.py \
        --shard $i --num-shards 8 --n 40 --out /tmp/pairdec &
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
from deepfold3.model.primitives import layer_norm as _ln
from deepfold3.training.collectives import slice_batch_dict
from deepfold3.training.data import _apply_cutoff
from deepfold3.training.loss import fape_loss, permute_gt_for_diffusion

CONTACT_A = 8.0
MAX_PAIRS = 60_000  # subsample per split; ridge on 128 features needs far fewer
RIDGE_LAM = 10.0
FAPE_KW = dict(clip_distance=10.0, clip_distance_inter=30.0, normalize=True,
               all_atom_points=True, mode="both", chunk_size=64)


def auc_ap(score: np.ndarray, label: np.ndarray) -> tuple[float, float]:
    n_pos, n_neg = int(label.sum()), int((~label.astype(bool)).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan"), float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1)
    s = score[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    auc = (ranks[label.astype(bool)].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    desc = np.argsort(-score, kind="mergesort")
    lab = label.astype(bool)[desc]
    prec = np.cumsum(lab) / np.arange(1, len(lab) + 1)
    return float(auc), float((prec * lab).sum() / n_pos)


def ridge_r(x_tr, y_tr, x_te, y_te, lam=RIDGE_LAM) -> float:
    """Fit ridge on (x_tr, y_tr), return test Pearson r. fp64 — the Gram matrix of
    128 correlated trunk channels is ill-conditioned enough that fp32 loses digits."""
    if x_tr.shape[0] < 200 or x_te.shape[0] < 200:
        return float("nan")
    xt, xe = x_tr.double(), x_te.double()
    yt, ye = y_tr.double(), y_te.double()
    # Standardize on TRAIN statistics: rel_feat is a near-one-hot 139-dim block
    # whose Gram matrix is badly scaled, and an unstandardized ridge there returns
    # a solution with no meaningful test variance.
    mu, sd = xt.mean(0, keepdim=True), xt.std(0, keepdim=True).clamp(min=1e-6)
    xt, xe = (xt - mu) / sd, (xe - mu) / sd
    xt = torch.cat([xt, torch.ones(xt.shape[0], 1, device=xt.device, dtype=torch.float64)], 1)
    xe = torch.cat([xe, torch.ones(xe.shape[0], 1, device=xe.device, dtype=torch.float64)], 1)
    a = xt.T @ xt
    a.diagonal().add_(lam)
    try:
        w = torch.linalg.solve(a, xt.T @ yt)
    except Exception:  # noqa: BLE001
        return float("nan")
    p = xe @ w
    if not torch.isfinite(p).all():
        return float("nan")
    p = p - p.mean()
    e = ye - ye.mean()
    # A degenerate fit predicts a constant; its "correlation" with the target is
    # numerical noise divided by numerical noise, which is how a Pearson r of 1.23
    # gets printed. Require real predicted spread relative to the target's.
    if float(p.norm()) < 1e-6 * float(e.norm().clamp(min=1e-12)):
        return float("nan")
    r = float((p @ e) / (p.norm() * e.norm()))
    return max(-1.0, min(1.0, r))


def main() -> None:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--ckpt", default="runs/df3-stage1-r14/checkpoint_34000.pt")
    ap_.add_argument("--config", default="configs/stage2.yaml")
    ap_.add_argument("--data-dir", default="/data/distillation/patchr_training")
    ap_.add_argument("--cutoff-date", default="2021-09-30")
    ap_.add_argument("--out", default="/tmp/pairdec")
    ap_.add_argument("--n", type=int, default=40)
    ap_.add_argument("--shard", type=int, default=0)
    ap_.add_argument("--num-shards", type=int, default=1)
    ap_.add_argument("--seed", type=int, default=101)
    args = ap_.parse_args()

    dev = "cuda:0"
    torch.manual_seed(3000 + args.shard)
    gen = np.random.default_rng(3000 + args.shard)

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
    lower = torch.cat([torch.zeros(1, device=dev), breaks])
    upper = torch.cat([breaks, breaks[-1:] + (breaks[-1] - breaks[-2])])
    centers = (lower + upper) * 0.5
    contact_bins = (upper <= CONTACT_A).float()
    tr = dh.transformer
    w_cat = torch.cat([p.weight.detach().float() for p in tr.pair_logits_proj], 0)  # [K, c_pair]
    pnw = tr.pair_norm_w.detach().float()

    out_path = Path(args.out) / f"pairdec_{args.shard:02d}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fh = out_path.open("w", newline="")
    wr = csv.writer(fh)
    wr.writerow([
        "idx", "record_id", "n_chains", "n_tokens", "n_cc_train", "n_cc_test",
        "roll_fape_inter", "trunk_cc_ap", "trunk_cc_auc",
        "r_cc_z", "r_cc_paircond", "r_cc_bias", "r_cc_rel", "r_cc_Ed",
        "r_intra_z", "r_cross_from_intra_z",
        "paired_min", "paired_med", "unpaired_min", "unpaired_med", "n_chain_no_paired",
        "H_cc", "H_cc_contact", "H_intra", "H_intra_contact",
        "p_contact_at_gt", "confident_far_at_gt", "confident_wrong_frac",
    ])

    # Per-chain MSA depths as actually loaded for the chains surviving the crop.
    # Must come from the dataset, not the featurized `msa` tensor: the dense
    # UNPAIRED block also spans several chains on a single row
    # (featurizer.py:806-813), so "row covers >=2 chains" is NOT "paired".
    # Captured in the SAME pass as the probes because the crop is redrawn on
    # every __getitem__ — separate runs cannot be joined by record id.
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
                rep_ok = seq_m & atom_m[ar, rep]
                d_gt = torch.cdist(gt_rep, gt_rep)

                # ---- feature sets ------------------------------------------------
                z = embeddings["pair"][0].float()  # [N, N, 128]
                static = dh.compute_static(embeddings=embeddings, seq_mask=seq_mask, rel_feat=rel_feat, **ref_kw)
                pc = static.pair_cond[0].float()  # [N, N, c]
                bias = (_ln(pc, [pc.shape[-1]]) * pnw) @ w_cat.T  # [N, N, K] — the actual attention biases
                rel = rel_feat[0].float()  # [N, N, 139] positional control
                probs = head(embeddings["pair"])[0].float().softmax(-1)
                ed = (probs * centers).sum(-1, keepdim=True)  # [N, N, 1]
                p_contact = (probs * contact_bins).sum(-1)

                valid = rep_ok.unsqueeze(1) & rep_ok.unsqueeze(0)
                valid.fill_diagonal_(False)
                cc = valid & (asym.unsqueeze(1) != asym.unsqueeze(0))
                ic = valid & (asym.unsqueeze(1) == asym.unsqueeze(0))
                cc_auc, cc_ap = auc_ap(p_contact[cc].cpu().numpy(), (d_gt[cc] < CONTACT_A).cpu().numpy())

                # Is the trunk UNCERTAIN or CONFIDENTLY WRONG about the interface?
                # H = -Sum p log p over the 64 bins; max = log(64) = 4.159.
                # "confident_far": the model puts >0.5 on the saturated last bin
                # (d >= 21.7 A) at a pair that is a TRUE contact (< 8 A) — i.e. it
                # asserts "these are far apart" about residues that touch.
                logits_full = head(embeddings["pair"])[0].float()
                lp = torch.log_softmax(logits_full, -1)
                pr = lp.exp()
                H = -(pr * lp).sum(-1)  # [N, N]
                p_far = pr[..., -1]
                gt_cc_contact = cc & (d_gt < CONTACT_A)
                gt_ic_contact = ic & (d_gt < CONTACT_A)
                def mn(t, m):
                    return float(t[m].mean()) if m.any() else float("nan")
                H_cc, H_cc_c = mn(H, cc), mn(H, gt_cc_contact)
                H_ic, H_ic_c = mn(H, ic), mn(H, gt_ic_contact)
                p_at_gt = mn(p_contact, gt_cc_contact)
                conf_far = mn(p_far, gt_cc_contact)
                conf_wrong = (
                    float((p_far[gt_cc_contact] > 0.5).float().mean()) if gt_cc_contact.any() else float("nan")
                )

                # ---- token-disjoint split (pairs within one crop are triangle-
                # inequality correlated, so a random PAIR split leaks; splitting the
                # TOKENS and keeping only within-side pairs does not) --------------
                n_tok_all = rep.shape[0]
                side = torch.zeros(n_tok_all, dtype=torch.bool, device=dev)
                side[torch.as_tensor(gen.permutation(n_tok_all)[: n_tok_all // 2].copy(), device=dev)] = True
                tr_m = side.unsqueeze(1) & side.unsqueeze(0)
                te_m = (~side).unsqueeze(1) & (~side).unsqueeze(0)

                def take(mask, cap=MAX_PAIRS):
                    ij = mask.nonzero(as_tuple=False)
                    if ij.shape[0] > cap:
                        ij = ij[torch.as_tensor(gen.permutation(ij.shape[0])[:cap].copy(), device=dev)]
                    return ij

                cc_tr, cc_te = take(cc & tr_m), take(cc & te_m)
                ic_tr = take(ic & tr_m)
                n_tr, n_te = int(cc_tr.shape[0]), int(cc_te.shape[0])

                def gather(t, ij):
                    return t[ij[:, 0], ij[:, 1]]

                y_cc_tr, y_cc_te = gather(d_gt, cc_tr), gather(d_gt, cc_te)
                res = {}
                for nm, feat in (("z", z), ("paircond", pc), ("bias", bias), ("rel", rel), ("Ed", ed)):
                    res[nm] = ridge_r(gather(feat, cc_tr), y_cc_tr, gather(feat, cc_te), y_cc_te)
                # transfer: fit on INTRA pairs, test on CROSS pairs — asks whether the
                # cross-chain block is written in the same metric code as the intra block
                ic_te = take(ic & te_m)  # ONE subsample — calling take() per argument
                # would draw two different pair sets and misalign X against y
                r_intra = ridge_r(gather(z, ic_tr), gather(d_gt, ic_tr), gather(z, ic_te), gather(d_gt, ic_te))
                r_transfer = ridge_r(gather(z, ic_tr), gather(d_gt, ic_tr), gather(z, cc_te), y_cc_te)

                fm = f0["frames_mask"].bool() & atom_m[f0["frame_token_index"], f0["frame_atom_index"]].all(-1)
                pm = fm.unsqueeze(-1) & f0["pair_mask"].bool()
                mf = atom_m.float().unsqueeze(-1)
                gt_c = (gt_pos.float() - (gt_pos.float() * mf).sum((0, 1), keepdim=True) / mf.sum().clamp(min=1e-6)) * mf
                _, fi = fape_loss(mini[0, 0].unsqueeze(0).float(), gt_c, None, None, None,
                                  f0["frame_atom_index"], f0["frame_token_index"], fm, seq_m, pm,
                                  atom_mask=atom_m, asym_id=asym, **FAPE_KW)

            surv = cap.get("surviving", [])
            pdep = [cap.get("paired", {}).get(c, 0) for c in surv]
            udep = [cap.get("unpaired", {}).get(c, 0) for c in surv]
            wr.writerow([idx, cap.get("record_id", f"idx{idx}"), n_chains, int(seq_m.sum()), n_tr, n_te,
                         float(fi), cc_ap, cc_auc,
                         res["z"], res["paircond"], res["bias"], res["rel"], res["Ed"], r_intra, r_transfer,
                         min(pdep) if pdep else "", float(np.median(pdep)) if pdep else "",
                         min(udep) if udep else "", float(np.median(udep)) if udep else "",
                         int(sum(1 for d in pdep if d <= 1)),
                         H_cc, H_cc_c, H_ic, H_ic_c, p_at_gt, conf_far, conf_wrong])
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
