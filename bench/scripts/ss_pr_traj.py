"""Per-class SS precision/recall/F1 TRAJECTORY over training, FoldBench monomer.

Decomposes the ss_f1.py F1 trajectory into precision vs recall per 3-state
class, plus the helix-over-prediction signal, across every step. Answers:
is the strand-F1 rise recall-driven, and does helix over-prediction (α
precision) shrink with training or stay frozen?

Reuses ss_confusion.target_job (DSSP + NW, caches ss_strings.json per sample),
so after a first pass it is fast. Writes CSV + plot under bench/runs/{series}/.

Usage:
    python bench/scripts/ss_pr_traj.py --series df3v1 --category monomer_protein
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


def pool_step(series, category, step, n_samples, seed, workers):
    sdir = BENCH / "runs" / series / "pred" / category / f"step_{step}"
    jobs = [
        (step, t.name, n_samples, seed, str(sdir))
        for t in sorted(sdir.iterdir())
        if t.is_dir() and not t.name.startswith("_")
    ]
    conf3 = defaultdict(int)
    n_tgt = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(target_job, j) for j in jobs]
        for fut in as_completed(futs):
            _pdb, _c8, c3s = fut.result()
            if c3s is None:
                continue
            n_tgt += 1
            for k, v in c3s.items():
                a, b = k.split("|")
                conf3[(a, b)] += v
    return conf3, n_tgt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", default="df3v1")
    ap.add_argument("--category", default="monomer_protein")
    ap.add_argument("--n-samples", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--steps", default="all")
    args = ap.parse_args()

    base = BENCH / "runs" / args.series / "pred" / args.category
    if args.steps == "all":
        steps = sorted(int(re.search(r"step_(\d+)", d).group(1)) for d in glob.glob(str(base / "step_*")))
    else:
        steps = [int(s) for s in args.steps.split(",")]
    # only keep reasonably-covered steps (≥50% of max targets)
    print(f"[ss_pr] steps: {steps}")

    rows = []
    for st in steps:
        conf3, n_tgt = pool_step(args.series, args.category, st, args.n_samples, args.seed, args.workers)
        if not conf3:
            continue
        prf = _prf(conf3, THREE)
        predA = sum(conf3.get((g, "α"), 0) for g in THREE)
        fpA = predA - conf3.get(("α", "α"), 0)
        rec = {
            "step": st,
            "n_targets": n_tgt,
            "aP": prf["α"][0],
            "aR": prf["α"][1],
            "aF1": prf["α"][2],
            "bP": prf["β"][0],
            "bR": prf["β"][1],
            "bF1": prf["β"][2],
            "cP": prf["c"][0],
            "cR": prf["c"][1],
            "cF1": prf["c"][2],
            # helix over-prediction
            "helix_fp_frac": (fpA / predA) if predA else float("nan"),
            "helixified_strand": (conf3.get(("β", "α"), 0) / predA) if predA else float("nan"),
            "fp_from_coil": (conf3.get(("c", "α"), 0) / fpA) if fpA else float("nan"),
        }
        rows.append(rec)
        print(
            f"  step {st:>6} (n_tgt={n_tgt:>3}): "
            f"βF1={rec['bF1']:.3f} βP={rec['bP']:.3f} βR={rec['bR']:.3f} | "
            f"αF1={rec['aF1']:.3f} αP={rec['aP']:.3f} αR={rec['aR']:.3f} | "
            f"helix_FP={rec['helix_fp_frac']:.2f}"
        )

    out_dir = BENCH / "runs" / args.series / "ss_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"ss_pr_traj_{args.category}.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"[ss_pr] -> {out_csv}")
    _plot(args.series, args.category, rows)


def _plot(series, category, rows):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    s = [r["step"] for r in rows]
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.4))
    # panel 1: per-class F1
    for c, k, col in (("helix", "aF1", "tab:red"), ("strand", "bF1", "tab:blue"), ("coil", "cF1", "tab:gray")):
        ax[0].plot(s, [r[k] for r in rows], "-o", color=col, label=c)
    ax[0].set_title("per-class F1")
    ax[0].set_ylabel("F1")
    # panel 2: strand & helix P vs R
    ax[1].plot(s, [r["bR"] for r in rows], "-o", color="tab:blue", label="strand recall")
    ax[1].plot(s, [r["bP"] for r in rows], "--s", color="tab:blue", alpha=0.6, label="strand precision")
    ax[1].plot(s, [r["aR"] for r in rows], "-o", color="tab:red", label="helix recall")
    ax[1].plot(s, [r["aP"] for r in rows], "--s", color="tab:red", alpha=0.6, label="helix precision")
    ax[1].set_title("strand vs helix: precision (dashed) vs recall (solid)")
    # panel 3: helix over-prediction
    ax[2].plot(s, [r["helix_fp_frac"] for r in rows], "-o", color="tab:purple", label="helix FP frac (of pred-α)")
    ax[2].plot(
        s, [r["helixified_strand"] for r in rows], "-o", color="tab:orange", label="pred-α truly β (helixified strand)"
    )
    ax[2].set_title("helix over-prediction")
    for a in ax:
        a.set_xlabel("training step")
        a.set_ylim(-0.02, 1.02)
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.suptitle(f"{series} {category} — SS precision/recall trajectory (DSSP, FoldBench monomer)")
    png = BENCH / "runs" / series / "ss_analysis" / f"ss_pr_traj_{category}.png"
    fig.tight_layout()
    fig.savefig(png, dpi=130)
    print(f"[ss_pr] plot -> {png}")


if __name__ == "__main__":
    main()
