"""Aggregate ``probe_pose_sigma.py`` shards into the three read-outs.

1. Per-example distribution at the *training* σ draw — the thing the reduced
   ``eval/fape_inter`` scalar hides. If the mean sits far above the median,
   the metric is a mixture and "0.1835 = everything is 5.5 Å off" is wrong.
2. σ-resolved curves — where the one-shot denoiser actually decides the pose,
   and how much pose information the noised input still carries (``d_ratio``
   of ``noisy_input``).
3. σ-decomposition of the mean — importance-weighting the σ grid by the
   training log-normal density says what fraction of the reported number is
   produced in each σ decade. This is the number that decides whether
   ``sigma_mix_alpha`` is worth turning on.

Usage: uv run python scripts/probe_pose_sigma_agg.py <dir-with-pose_probe_*.csv>
"""

from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else float("nan")


def main() -> None:
    d = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/pose_probe")
    rows = []
    for f in sorted(d.glob("pose_probe_*.csv")):
        with f.open() as fh:
            rows += list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"no rows under {d}")

    def fl(r, k):
        v = r[k]
        return float(v) if v not in ("", None) else float("nan")

    ids = sorted({r["record_id"] + "|" + r["idx"] for r in rows})
    print(f"{len(rows)} rows | {len(ids)} examples | multi-chain only\n")

    nch = [int(r["n_chains"]) for r in rows if r["kind"] == "mini_rollout"]
    print(f"chains/example: median {np.median(nch):.0f}  range {min(nch)}-{max(nch)}\n")

    # ---- 1. per-example distribution at the training σ draw -----------------
    per_ex = defaultdict(list)
    for r in rows:
        if r["kind"] == "traindraw":
            per_ex[r["record_id"] + "|" + r["idx"]].append(fl(r, "fape_inter"))
    ex_mean = np.array([np.mean(v) for v in per_ex.values()])
    draws = np.array([x for v in per_ex.values() for x in v])
    print("=" * 78)
    print("1. fape_inter at the TRAINING σ draw  (eval log reports the mean: 0.1835)")
    print("=" * 78)
    for nm, a in (("per-example (mean of 4 draws)", ex_mean), ("per-draw (raw)", draws)):
        print(
            f"  {nm:<30} mean {a.mean():.4f} | p10 {pct(a,10):.4f}  p25 {pct(a,25):.4f}  "
            f"p50 {pct(a,50):.4f}  p75 {pct(a,75):.4f}  p90 {pct(a,90):.4f}  max {a.max():.4f}"
        )
    for thr in (0.3, 0.4, 0.5):
        print(f"  fraction of examples > {thr}: {float((ex_mean > thr).mean()):.3f}")
    print(f"  mean/median ratio: {ex_mean.mean() / np.median(ex_mean):.2f}   (1.0 = symmetric, >1.3 = mixture)")

    # ---- 2. σ-resolved curves ----------------------------------------------
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["kind"] in ("oneshot", "noisy_input"):
            by[r["kind"]][float(r["sigma"])].append(r)
    print()
    print("=" * 78)
    print("2. σ-resolved one-shot denoise  (median over examples; IQR in brackets)")
    print("=" * 78)
    print(f"  {'σ':>8} {'fape_inter':>20} {'fape_intra':>12} {'d_ratio':>9} {'rg_ratio':>9} {'in:d_ratio':>11}")
    sigmas = sorted(by["oneshot"])
    for s in sigmas:
        o = by["oneshot"][s]
        fi = np.array([fl(r, "fape_inter") for r in o])
        fx = np.array([fl(r, "fape_intra") for r in o])
        dr = np.array([fl(r, "d_ratio") for r in o])
        rg = np.array([fl(r, "rg_ratio") for r in o])
        nd = np.array([fl(r, "d_ratio") for r in by["noisy_input"][s]])
        print(
            f"  {s:>8.1f} {pct(fi,50):>8.3f} [{pct(fi,25):.3f},{pct(fi,75):.3f}] "
            f"{pct(fx,50):>12.3f} {np.nanmedian(dr):>9.3f} {np.nanmedian(rg):>9.3f} {np.nanmedian(nd):>11.3f}"
        )

    mini = [r for r in rows if r["kind"] == "mini_rollout"]
    mfi = np.array([fl(r, "fape_inter") for r in mini])
    mfx = np.array([fl(r, "fape_intra") for r in mini])
    mdr = np.array([fl(r, "d_ratio") for r in mini])
    mrg = np.array([fl(r, "rg_ratio") for r in mini])
    print(
        f"  {'ROLLOUT':>8} {pct(mfi,50):>8.3f} [{pct(mfi,25):.3f},{pct(mfi,75):.3f}] "
        f"{pct(mfx,50):>12.3f} {np.nanmedian(mdr):>9.3f} {np.nanmedian(mrg):>9.3f} {'—':>11}"
    )
    print(f"\n  rollout fape_inter: mean {mfi.mean():.4f} | p10 {pct(mfi,10):.3f} p50 {pct(mfi,50):.3f} p90 {pct(mfi,90):.3f}")
    for thr in (0.4, 0.6):
        print(f"  rollout pose-failure fraction (fape_inter > {thr}): {float((mfi > thr).mean()):.3f}")
    print(f"  rollout collapse fraction (d_ratio < 0.8): {float((mdr < 0.8).mean()):.3f}   (rg_ratio < 0.8: {float((mrg < 0.8).mean()):.3f})")

    # ---- 3. σ-decomposition of the reported mean ---------------------------
    # Training draw: ln σ ~ N(ln σ_data + p_mean, p_std) = N(1.573, 1.5).
    mu, sd = math.log(16.0) - 1.2, 1.5
    edges = [0.0] + [math.sqrt(sigmas[i] * sigmas[i + 1]) for i in range(len(sigmas) - 1)] + [float("inf")]

    def cdf(x):
        return 0.5 * (1 + math.erf((math.log(x) - mu) / (sd * math.sqrt(2)))) if x > 0 else 0.0

    w = np.array([cdf(edges[i + 1]) - cdf(edges[i]) for i in range(len(sigmas))])
    med = np.array([pct(np.array([fl(r, "fape_inter") for r in by["oneshot"][s]]), 50) for s in sigmas])
    contrib = w * med
    print()
    print("=" * 78)
    print("3. σ-decomposition of E[fape_inter] under the training σ density")
    print("=" * 78)
    print(f"  {'σ':>8} {'P(σ bin)':>10} {'median':>9} {'contrib':>9} {'share':>8}")
    for s, wi, mi, ci in zip(sigmas, w, med, contrib, strict=False):
        print(f"  {s:>8.1f} {wi:>10.3f} {mi:>9.3f} {ci:>9.4f} {ci / contrib.sum():>7.1%}")
    print(f"  {'TOTAL':>8} {w.sum():>10.3f} {'':>9} {contrib.sum():>9.4f}")
    print(f"\n  reconstructed E[fape_inter] = {contrib.sum():.4f}   (eval log: 0.1835)")
    hi = float(contrib[np.array(sigmas) >= 20].sum() / contrib.sum())
    print(f"  share of the metric produced at σ >= 20 (pose actually unknown): {hi:.1%}")
    print(f"  probability mass  at σ >= 20: {float(w[np.array(sigmas) >= 20].sum()):.1%}")


if __name__ == "__main__":
    main()
