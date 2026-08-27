"""SS8 / SS3 secondary-structure F1 trajectory for FoldBench monomer targets.

OpenFold Fig-4a style: run DSSP on each predicted model.cif and its
ground-truth CIF, NW-align by sequence, and score per-residue secondary
structure as per-class precision/recall/F1 -- 8-state (DSSP H/E/G/I/T/S/B/P)
and the 3-state collapse (helix/strand/coil) -- as a function of training
step. The decision-relevant signal is the per-class split: whether strand
F1 keeps trailing helix F1 across training.

Reuses scripts/dssp_f1.py (run_dssp, per_class_metrics, DSSP_8) and
scripts/check_chirality.py (needleman_wunsch) -- the same engine as
ss_learning.py, adapted to the bench/ layout. Per-sample results cache to
ss_f1.json next to model.cif (resumable).

Usage:
    python bench/scripts/ss_f1.py --series df3v1 --category monomer_protein
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BENCH = REPO / "bench"
GT_ROOT = BENCH / "data" / "ground_truths"
sys.path.insert(0, str(REPO / "scripts"))

from check_chirality import needleman_wunsch  # noqa: E402
from dssp_f1 import DSSP_8, per_class_metrics, run_dssp  # noqa: E402

SS3_MAP = {
    "H": "helix",
    "G": "helix",
    "I": "helix",
    "P": "helix",
    "E": "strand",
    "B": "strand",
    "T": "coil",
    "S": "coil",
    "-": "coil",
}
THREE = ("helix", "strand", "coil")
ALL_STEPS = list(range(10000, 95000, 5000))


def measure(pred_cif: Path, ref_res) -> dict | None:
    pred_res = run_dssp(pred_cif, cache=False)
    if not pred_res:
        return None
    pairs = needleman_wunsch("".join(a for a, _ in ref_res), "".join(a for a, _ in pred_res), include_mismatches=False)
    if not pairs:
        return None
    ref_ss = [ref_res[i][1] for i, _ in pairs]
    pred_ss = [pred_res[j][1] for _, j in pairs]
    _, _, f1_8, m8 = per_class_metrics(pred_ss, ref_ss, DSSP_8)
    pred_3 = [SS3_MAP[c] for c in pred_ss]
    ref_3 = [SS3_MAP[c] for c in ref_ss]
    _, _, f1_3, m3 = per_class_metrics(pred_3, ref_3, THREE)
    return {"macro_8": m8, "macro_3": m3, "f1_8": f1_8, "f1_3": f1_3, "n": len(pairs)}


def _cmean(rows, key, sub=None):
    vals = []
    for r in rows:
        v = r[key][sub] if sub else r[key]
        if isinstance(v, (int, float)) and v == v:
            vals.append(v)
    return statistics.mean(vals) if vals else float("nan")


def target_job(args):
    # A few GT cifs have multi-char chain names that break run_dssp's
    # cif->pdb step; skip any target that errors rather than crash the pool.
    try:
        return _target_job_inner(args)
    except Exception:
        return (args[0], args[1], None)


def _target_job_inner(args):
    step, pdb, n_samples, seed, pred_root_str = args
    pred_root = Path(pred_root_str)
    tdir = pred_root / pdb
    sample_dirs = [tdir / f"seed-{seed}_sample-{i}" for i in range(n_samples)]
    pending = [d for d in sample_dirs if (d / "model.cif").exists() and not (d / "ss_f1.json").exists()]
    ref_res = run_dssp(GT_ROOT / f"{pdb}.cif", cache=False) if pending else None
    rows = []
    for d in sample_dirs:
        cif, cache = d / "model.cif", d / "ss_f1.json"
        if not cif.exists():
            continue
        if cache.exists():
            try:
                rows.append(json.load(open(cache)))
                continue
            except (json.JSONDecodeError, OSError):
                pass
        r = measure(cif, ref_res) if ref_res else None
        if r is not None:
            cache.write_text(json.dumps(r))
            rows.append(r)
    if not rows:
        return (step, pdb, None)
    agg = {"macro_8": _cmean(rows, "macro_8"), "macro_3": _cmean(rows, "macro_3")}
    for c in DSSP_8:
        agg[f"f1_8_{c}"] = _cmean(rows, "f1_8", c)
    for c in THREE:
        agg[f"f1_3_{c}"] = _cmean(rows, "f1_3", c)
    return (step, pdb, agg)


def run(series, category, n_samples, seed, workers, steps):
    pred_base = BENCH / "runs" / series / "pred" / category
    jobs = []
    for step in steps:
        sdir = pred_base / f"step_{step}"
        if not sdir.exists():
            continue
        for tdir in sorted(sdir.iterdir()):
            if tdir.is_dir() and not tdir.name.startswith("_"):
                jobs.append((step, tdir.name, n_samples, seed, str(sdir)))
    if not jobs:
        raise SystemExit(f"no predictions under {pred_base}")
    print(f"[ss_f1] {len(jobs)} (step,target) jobs on {workers} workers ...")
    per_step = {}
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(target_job, j) for j in jobs]
        for fut in as_completed(futs):
            step, _pdb, agg = fut.result()
            if agg is not None:
                per_step.setdefault(step, []).append(agg)
            done += 1
            if done % 500 == 0:
                print(f"  {done}/{len(jobs)}")
    fields = (
        ["step", "n_targets", "macro_f1_8", "macro_f1_3"] + [f"f1_8_{c}" for c in DSSP_8] + [f"f1_3_{c}" for c in THREE]
    )
    rows = []
    for step in sorted(per_step):
        ts = per_step[step]
        rec = {
            "step": step,
            "n_targets": len(ts),
            "macro_f1_8": round(_cmean(ts, "macro_8"), 4),
            "macro_f1_3": round(_cmean(ts, "macro_3"), 4),
        }
        for c in DSSP_8:
            rec[f"f1_8_{c}"] = round(_cmean(ts, f"f1_8_{c}"), 4)
        for c in THREE:
            rec[f"f1_3_{c}"] = round(_cmean(ts, f"f1_3_{c}"), 4)
        rows.append(rec)
    out_dir = BENCH / "runs" / series / "ss_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"ss_f1_{category}.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"[ss_f1] {len(rows)} steps -> {out_csv}")
    print(f"{'step':>7} {'macro3':>7} {'macro8':>7} {'helix':>7} {'strand':>7} {'coil':>7}")
    for r in rows:
        print(
            f"{r['step']:>7} {r['macro_f1_3']:>7} {r['macro_f1_8']:>7} "
            f"{r['f1_3_helix']:>7} {r['f1_3_strand']:>7} {r['f1_3_coil']:>7}"
        )
    _plot(series, category, rows)


def _plot(series, category, rows):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[ss_f1] matplotlib unavailable; skipping plot")
        return
    steps = [r["step"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    axes[0].plot(steps, [r["macro_f1_3"] for r in rows], "-o", color="k", lw=2.5, label="macro F1 (3-state)")
    axes[0].plot(steps, [r["macro_f1_8"] for r in rows], "-s", color="tab:gray", lw=2, label="macro F1 (8-state)")
    axes[0].set_title("SS macro-F1 vs step")
    axes[0].set_ylabel("macro F1")
    cols = {"helix": "tab:red", "strand": "tab:blue", "coil": "tab:gray"}
    for c in THREE:
        axes[1].plot(steps, [r[f"f1_3_{c}"] for r in rows], "-o", color=cols[c], label=c)
    axes[1].set_title("3-state per-class F1 vs step")
    axes[1].set_ylabel("F1")
    for ax in axes:
        ax.set_xlabel("training step")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    fig.suptitle(f"{series} -- FoldBench-small {category}  DSSP secondary-structure F1")
    png = BENCH / "runs" / series / "ss_analysis" / f"ss_f1_{category}.png"
    fig.tight_layout()
    fig.savefig(png, dpi=130)
    print(f"[ss_f1] plot -> {png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", default="df3v1")
    ap.add_argument("--category", default="monomer_protein")
    ap.add_argument("--n-samples", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--steps", default="all")
    args = ap.parse_args()
    if args.steps == "all":
        import glob
        import re

        sdirs = glob.glob(str(BENCH / "runs" / args.series / "pred" / args.category / "step_*"))
        steps = sorted(int(re.search(r"step_(\d+)", d).group(1)) for d in sdirs)
    else:
        steps = [int(s) for s in args.steps.split(",")]
    run(args.series, args.category, args.n_samples, args.seed, args.workers, steps)


if __name__ == "__main__":
    main()
