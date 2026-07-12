"""Per-TARGET SS trajectory for one DSSP fold-class group (default β-rich).

Tracks each individual target's helix over-prediction (FP frac of predicted-α)
and strand recall across training, to see whether the β-rich helix-hallucination
is uniform or driven by a stuck subset. Reads cached ss_strings.json (fast).

Outputs to ss_analysis/:
  * ss_target_traj_{group}_{category}.csv   (target × step long form)
  * ss_target_traj_{group}_{category}.png   (heatmaps: helix-FP + strand recall,
    targets sorted by final helix-FP; worst at top)

Usage:
    python bench/scripts/ss_target_traj.py --series df3v1 --group beta
"""

from __future__ import annotations

import argparse
import csv
import glob
import re
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BENCH = REPO / "bench"
sys.path.insert(0, str(BENCH / "scripts"))

from ss_confusion import THREE, target_job  # noqa: E402


def per_target_step(series, category, step, targets, n_samples, seed, workers):
    sdir = BENCH / "runs" / series / "pred" / category / f"step_{step}"
    jobs = [(step, t, n_samples, seed, str(sdir)) for t in targets if (sdir / t).is_dir()]
    out = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(target_job, j) for j in jobs]
        for fut in as_completed(futs):
            pdb, _c8, c3s = fut.result()
            if c3s is None:
                continue
            c = defaultdict(int)
            for k, v in c3s.items():
                a, b = k.split("|")
                c[(a, b)] += v
            predA = sum(c.get((g, "α"), 0) for g in THREE)
            trueB = sum(c.get(("β", p), 0) for p in THREE)
            out[pdb] = {
                "helix_fp": (predA - c.get(("α", "α"), 0)) / predA if predA else float("nan"),
                "pred_a": predA,
                "strand_recall": c.get(("β", "β"), 0) / trueB if trueB else float("nan"),
                "n_res": sum(c.values()),
            }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", default="df3v1")
    ap.add_argument("--category", default="monomer_protein")
    ap.add_argument("--group", default="beta")
    ap.add_argument("--n-samples", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--steps", default="all")
    args = ap.parse_args()

    fold = {r["pdb_id"]: r["fold_class"] for r in csv.DictReader(open(BENCH / "inputs" / f"{args.category}_dssp.csv"))}
    targets = sorted(p for p, c in fold.items() if c == args.group)
    base = BENCH / "runs" / args.series / "pred" / args.category
    if args.steps == "all":
        steps = sorted(int(re.search(r"step_(\d+)", d).group(1)) for d in glob.glob(str(base / "step_*")))
    else:
        steps = [int(s) for s in args.steps.split(",")]
    print(f"[ss_tgt] group={args.group}: {len(targets)} targets, {len(steps)} steps")

    data = {}  # step -> {pdb -> metrics}
    for st in steps:
        data[st] = per_target_step(args.series, args.category, st, targets, args.n_samples, args.seed, args.workers)

    out_dir = BENCH / "runs" / args.series / "ss_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for st in steps:
        for pdb in targets:
            m = data[st].get(pdb)
            if m:
                rows.append({"step": st, "target": pdb, **m})
    out_csv = out_dir / f"ss_target_traj_{args.group}_{args.category}.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "target", "helix_fp", "pred_a", "strand_recall", "n_res"])
        w.writeheader()
        w.writerows(rows)
    print(f"[ss_tgt] -> {out_csv}")

    # rank by final-step helix_fp
    last = steps[-1]
    fp_last = {p: data[last].get(p, {}).get("helix_fp", float("nan")) for p in targets}
    order = sorted(targets, key=lambda p: -(fp_last[p] if fp_last[p] == fp_last[p] else -1))
    print(f"\nβ-rich targets by helix-FP @ step {last} (worst first):")
    print(f"  {'target':>18} {'hFP_first':>9} {'hFP_last':>9} {'Δ':>7} {'sRec_last':>9} {'pred_α':>7}")
    first = steps[0]
    for p in order:
        a = data[first].get(p, {}).get("helix_fp", float("nan"))
        b = fp_last[p]
        sr = data[last].get(p, {}).get("strand_recall", float("nan"))
        pa = data[last].get(p, {}).get("pred_a", 0)
        d = (b - a) if (a == a and b == b) else float("nan")
        print(f"  {p:>18} {a:>9.2f} {b:>9.2f} {d:>+7.2f} {sr:>9.2f} {pa:>7}")

    _plot(args, steps, targets, order, data, out_dir)


def _plot(args, steps, targets, order, data, out_dir):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return
    H = np.array([[data[s].get(p, {}).get("helix_fp", np.nan) for s in steps] for p in order])
    R = np.array([[data[s].get(p, {}).get("strand_recall", np.nan) for s in steps] for p in order])
    # Column x-edges at the midpoints between benchmarked steps (half-width cells
    # at the ends), so uneven grids (e.g. 1k/2k/5k) are drawn to scale instead of
    # imshow's uniform stretch — which would misplace step 2000 at x=3000.
    sx = np.asarray(steps, dtype=float)
    if len(sx) > 1:
        mid = (sx[:-1] + sx[1:]) / 2
        xedges = np.concatenate([[sx[0] - (mid[0] - sx[0])], mid, [sx[-1] + (sx[-1] - mid[-1])]])
    else:
        xedges = np.array([sx[0] - 0.5, sx[0] + 0.5])
    yedges = np.arange(len(order) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(16, 10))
    for a, M, title, cmap in (
        (ax[0], H, "helix over-prediction (FP frac of pred-α)", "Reds"),
        (ax[1], R, "strand recall", "Blues"),
    ):
        im = a.pcolormesh(xedges, yedges, M, cmap=cmap, vmin=0, vmax=1)
        a.invert_yaxis()  # order[0] (worst helix-FP) at the top, as with imshow
        a.set_yticks(np.arange(len(order)) + 0.5)
        a.set_yticklabels(order, fontsize=7)
        if len(steps) <= 12:
            a.set_xticks(steps)
        a.set_xlabel("training step")
        a.set_title(title)
        fig.colorbar(im, ax=a, fraction=0.046)
    fig.suptitle(
        f"{args.series} {args.category} — {args.group}-class targets (n={len(order)}), sorted by final helix-FP"
    )
    png = out_dir / f"ss_target_traj_{args.group}_{args.category}.png"
    fig.tight_layout()
    fig.savefig(png, dpi=130)
    print(f"[ss_tgt] plot -> {png}")


if __name__ == "__main__":
    main()
