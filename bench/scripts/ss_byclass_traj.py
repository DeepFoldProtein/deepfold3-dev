"""Per-fold-class SS trajectory: is residual helix-over-prediction confined
to β-rich targets?

Splits the FoldBench monomer targets by DSSP fold class
(bench/inputs/{category}_dssp.csv: alpha / beta / alpha_beta) and pools the
3-state SS confusion PER GROUP at each step, tracking strand recall, helix
precision, and helix over-prediction separately for each fold class.

Reads the per-sample ``ss_strings.json`` caches written by ss_confusion.py
(no DSSP if already cached → fast). Same engine via ss_confusion.target_job.

Usage:
    python bench/scripts/ss_byclass_traj.py --series df3v1 --category monomer_protein
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

from ss_confusion import THREE, _prf, target_job  # noqa: E402

GROUPS = ("alpha", "beta", "alpha_beta")


def load_fold_class(category):
    p = BENCH / "inputs" / f"{category}_dssp.csv"
    return {r["pdb_id"]: r["fold_class"] for r in csv.DictReader(open(p))}


def pool_step_bygroup(series, category, step, n_samples, seed, workers, fold):
    sdir = BENCH / "runs" / series / "pred" / category / f"step_{step}"
    jobs = [
        (step, t.name, n_samples, seed, str(sdir))
        for t in sorted(sdir.iterdir())
        if t.is_dir() and not t.name.startswith("_")
    ]
    conf = {g: defaultdict(int) for g in GROUPS}
    ntgt = {g: 0 for g in GROUPS}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(target_job, j) for j in jobs]
        for fut in as_completed(futs):
            pdb, _c8, c3s = fut.result()
            if c3s is None:
                continue
            g = fold.get(pdb)
            if g not in GROUPS:
                continue
            ntgt[g] += 1
            for k, v in c3s.items():
                a, b = k.split("|")
                conf[g][(a, b)] += v
    return conf, ntgt


def metrics(conf):
    prf = _prf(conf, THREE)
    predA = sum(conf.get((x, "α"), 0) for x in THREE)
    fpA = predA - conf.get(("α", "α"), 0)
    return {
        "bR": prf["β"][1],
        "bP": prf["β"][0],
        "bF1": prf["β"][2],
        "aR": prf["α"][1],
        "aP": prf["α"][0],
        "aF1": prf["α"][2],
        "helix_fp_frac": (fpA / predA) if predA else float("nan"),
        "helixified_strand": (conf.get(("β", "α"), 0) / predA) if predA else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", default="df3v1")
    ap.add_argument("--category", default="monomer_protein")
    ap.add_argument("--n-samples", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--steps", default="all")
    args = ap.parse_args()

    fold = load_fold_class(args.category)
    base = BENCH / "runs" / args.series / "pred" / args.category
    if args.steps == "all":
        steps = sorted(int(re.search(r"step_(\d+)", d).group(1)) for d in glob.glob(str(base / "step_*")))
    else:
        steps = [int(s) for s in args.steps.split(",")]

    rows = []
    for st in steps:
        conf, ntgt = pool_step_bygroup(args.series, args.category, st, args.n_samples, args.seed, args.workers, fold)
        line = [f"step {st:>6}:"]
        for g in GROUPS:
            if ntgt[g] == 0:
                continue
            m = metrics(conf[g])
            rows.append({"step": st, "group": g, "n_targets": ntgt[g], **m})
            line.append(f"{g[:1]}β(R={m['bR']:.2f} hxFP={m['helix_fp_frac']:.2f})")
        print(" ".join(line))

    out_dir = BENCH / "runs" / args.series / "ss_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"ss_byclass_traj_{args.category}.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"[ss_byclass] -> {out_csv}")
    _plot(args.series, args.category, rows, out_dir)


def _plot(series, category, rows, out_dir):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    steps = sorted({r["step"] for r in rows})
    col = {"alpha": "tab:red", "beta": "tab:blue", "alpha_beta": "tab:green"}
    panels = [
        ("strand recall (βR)", "bR"),
        ("helix over-prediction (FP frac of pred-α)", "helix_fp_frac"),
        ("helix precision (αP)", "aP"),
        ("pred-α truly β (helixified strand)", "helixified_strand"),
    ]
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    ax = ax.ravel()
    for i, (title, key) in enumerate(panels):
        for g in GROUPS:
            ys = [next((r[key] for r in rows if r["step"] == s and r["group"] == g), None) for s in steps]
            nlast = next((r["n_targets"] for r in rows[::-1] if r["group"] == g), 0)
            ax[i].plot(steps, ys, "-o", color=col[g], label=f"{g} (n={nlast})")
        ax[i].set_title(title)
        ax[i].set_xlabel("step")
        ax[i].set_ylim(-0.02, 1.02)
        ax[i].grid(alpha=0.3)
        ax[i].legend(fontsize=8)
    fig.suptitle(f"{series} {category} — SS by DSSP fold class (does helix-overpred concentrate in β-rich?)")
    png = out_dir / f"ss_byclass_traj_{category}.png"
    fig.tight_layout()
    fig.savefig(png, dpi=130)
    print(f"[ss_byclass] plot -> {png}")


if __name__ == "__main__":
    main()
