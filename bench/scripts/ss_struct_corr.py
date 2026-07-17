"""Correlation between DSSP secondary-structure quality and structural accuracy.

Joins per-sample SS metrics (ss_f1.json: macro 3/8-state F1, per-class 3-state
F1; + helix-overprediction from ss_strings.json) with the OST structural scores
(eval/.../raw/{category}_ost.csv: lDDT / TM / GDT-TS), and reports:

  1. cross-sectional correlation at one step (default latest full): per-SAMPLE
     and per-TARGET (mean over samples) Pearson + Spearman of each SS metric vs
     each structural metric — does getting SS right predict getting structure right?
  2. trajectory correlation across steps: per-step mean SS (ss_f1 CSV) vs per-step
     mean structural (eval summary.csv) — does SS improvement track accuracy gain?

Plots scatter grid (per-target) + a Pearson heatmap to ss_analysis/.

Usage:
    python bench/scripts/ss_struct_corr.py --series df3v1 --category monomer_protein --step 100000
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
BENCH = REPO / "bench"
sys.path.insert(0, str(BENCH / "scripts"))

SS3 = {"H": "α", "G": "α", "I": "α", "P": "α", "E": "β", "B": "β", "T": "c", "S": "c", "-": "c"}
SS_METRICS = ["macro_3", "macro_8", "f1_helix", "f1_strand", "f1_coil", "helix_fp"]
ST_METRICS = ["lddt", "tm_score", "oligo_gdtts"]


def _pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan")
    return float(np.corrcoef(x[m], y[m])[0, 1])


def _spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x[m]))
    ry = np.argsort(np.argsort(y[m]))
    return float(np.corrcoef(rx, ry)[0, 1])


def _helix_fp(ss_path):
    try:
        d = json.load(open(ss_path))
    except (OSError, json.JSONDecodeError):
        return float("nan")
    predA = corr = 0
    for rc, pc in zip(d["r"], d["p"], strict=False):
        if SS3[pc] == "α":
            predA += 1
            if SS3[rc] == "α":
                corr += 1
    return (predA - corr) / predA if predA else float("nan")


def load_step(series, category, step, n_samples=5, seed=42):
    raw = BENCH / "runs" / series / "eval" / category / f"step_{step}" / "raw" / f"{category}_ost.csv"
    struct = {}
    for r in csv.DictReader(open(raw)):
        try:
            struct[(r["pdb_id"], int(r["sample"]))] = {
                "lddt": float(r["lddt"]),
                "tm_score": float(r["tm_score"]),
                "oligo_gdtts": float(r["oligo_gdtts"]),
            }
        except (ValueError, KeyError):
            pass
    pred_root = BENCH / "runs" / series / "pred" / category / f"step_{step}"
    recs = []
    for (pdb, smp), st in struct.items():
        sd = pred_root / pdb / f"seed-{seed}_sample-{smp}"
        try:
            f1 = json.load(open(sd / "ss_f1.json"))
        except (OSError, json.JSONDecodeError):
            continue
        rec = {
            "pdb": pdb,
            "sample": smp,
            "macro_3": f1.get("macro_3"),
            "macro_8": f1.get("macro_8"),
            "f1_helix": f1.get("f1_3", {}).get("helix"),
            "f1_strand": f1.get("f1_3", {}).get("strand"),
            "f1_coil": f1.get("f1_3", {}).get("coil"),
            "helix_fp": _helix_fp(sd / "ss_strings.json"),
            **st,
        }
        recs.append(rec)
    return recs


def corr_table(recs, label):
    print(f"\n=== {label}  (n={len(recs)}) — Pearson r (Spearman ρ) ===")
    hdr = "SS \\ struct"
    print(f"{hdr:>12} | " + " | ".join(f"{s:>16}" for s in ST_METRICS))
    for ss in SS_METRICS:
        cells = []
        for stm in ST_METRICS:
            r = _pearson([x[ss] for x in recs], [x[stm] for x in recs])
            rho = _spearman([x[ss] for x in recs], [x[stm] for x in recs])
            cells.append(f"{r:>+6.2f} ({rho:>+5.2f})")
        print(f"{ss:>12} | " + " | ".join(f"{c:>16}" for c in cells))


def per_target(recs):
    by = defaultdict(list)
    for r in recs:
        by[r["pdb"]].append(r)
    out = []
    for pdb, rs in by.items():
        rec = {"pdb": pdb}
        for k in SS_METRICS + ST_METRICS:
            vals = [x[k] for x in rs if x[k] is not None and np.isfinite(x[k])]
            rec[k] = float(np.mean(vals)) if vals else float("nan")
        out.append(rec)
    return out


def trajectory_corr(series, category):
    ssf = BENCH / "runs" / series / "ss_analysis" / f"ss_f1_{category}.csv"
    ss = {int(r["step"]): r for r in csv.DictReader(open(ssf))}
    steps, rows = [], []
    for sp in sorted(ss):
        summ = BENCH / "runs" / series / "eval" / category / f"step_{sp}" / "summary.csv"
        if not summ.exists():
            continue
        s = next(csv.DictReader(open(summ)))
        steps.append(sp)
        rows.append(
            {
                "macro_3": float(ss[sp]["macro_f1_3"]),
                "f1_helix": float(ss[sp]["f1_3_helix"]),
                "f1_strand": float(ss[sp]["f1_3_strand"]),
                "lddt": float(s["mean_lddt"]),
                "tm_score": float(s["mean_tm_score"]),
                "oligo_gdtts": float(s["mean_gdt_ts"]),
            }
        )
    print(f"\n=== TRAJECTORY (across {len(steps)} steps) — per-step mean SS vs mean struct, Pearson r ===")
    hdr = "SS \\ struct"
    print(f"{hdr:>12} | " + " | ".join(f"{s:>10}" for s in ST_METRICS))
    for ssm in ["macro_3", "f1_helix", "f1_strand"]:
        cells = [f"{_pearson([r[ssm] for r in rows], [r[stm] for r in rows]):>+10.2f}" for stm in ST_METRICS]
        print(f"{ssm:>12} | " + " | ".join(cells))
    return steps, rows


def _plot(series, category, tgt, traj_steps, traj_rows, out_dir):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig = plt.figure(figsize=(16, 10))
    pairs = [
        ("macro_3", "tm_score"),
        ("f1_strand", "tm_score"),
        ("helix_fp", "tm_score"),
        ("macro_3", "lddt"),
        ("f1_strand", "oligo_gdtts"),
        ("f1_helix", "lddt"),
    ]
    for i, (ss, stm) in enumerate(pairs):
        ax = fig.add_subplot(2, 3, i + 1)
        x = [t[ss] for t in tgt]
        y = [t[stm] for t in tgt]
        ax.scatter(x, y, s=10, alpha=0.5)
        r = _pearson(x, y)
        ax.set_xlabel(ss)
        ax.set_ylabel(stm)
        ax.set_title(f"{ss} vs {stm}  (r={r:+.2f}, per-target)")
        ax.grid(alpha=0.3)
    fig.suptitle(f"{series} {category} — SS quality vs structural accuracy (per-target, latest step)")
    png = out_dir / f"ss_struct_corr_{category}.png"
    fig.tight_layout()
    fig.savefig(png, dpi=130)
    plt.close(fig)
    print(f"[corr] scatter -> {png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", default="df3v1")
    ap.add_argument("--category", default="monomer_protein")
    ap.add_argument("--step", type=int, default=100000)
    args = ap.parse_args()

    recs = load_step(args.series, args.category, args.step)
    if not recs:
        sys.exit("no joined records")
    print(f"[corr] {args.series} {args.category} step {args.step}: {len(recs)} samples joined")
    corr_table(recs, f"PER-SAMPLE step {args.step}")
    tgt = per_target(recs)
    corr_table(tgt, f"PER-TARGET step {args.step}")
    traj_steps, traj_rows = trajectory_corr(args.series, args.category)

    out_dir = BENCH / "runs" / args.series / "ss_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"ss_struct_corr_pertarget_{args.category}_step{args.step}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pdb"] + SS_METRICS + ST_METRICS)
        w.writeheader()
        w.writerows(tgt)
    _plot(args.series, args.category, tgt, traj_steps, traj_rows, out_dir)


if __name__ == "__main__":
    main()
