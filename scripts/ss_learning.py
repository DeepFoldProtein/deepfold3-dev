"""Secondary-structure learning analysis, OpenFold-paper-style.

For a validation set of targets and a sequence of training checkpoints,
this script reports two things:

  (1) Per-class DSSP F1 (one curve per 8-state SSE category, plus the
      classical 3-state collapse) as a function of training step.
      Fig 4a of Ahdritz et al., 2024 (OpenFold).

  (2) Helix recovery histogram: for every ground-truth α-helix, find the
      longest contiguous run of predicted 'H' residues at the same
      sequence positions; report the recovery fraction
      (``longest_pred_run / ground_truth_helix_len``) and bin it.
      Fig 4c of the same paper. Helices either are recovered or aren't —
      the histogram is bimodal.

Inputs are gathered using the same convention as ``dssp_f1.py`` /
``check_chirality.py``: per-target ``step_{N}_n30/seed-S_sample-i``
inference layout, reference mmCIF at the target root.

Usage:
    uv run python scripts/ss_learning.py \\
        --targets-file tmp/bench/targets.csv \\
        --bench-root tmp/bench \\
        --arm-dir df3-fapeS3-25e5-c10-g35 \\
        --steps 2000 4000 6000 8000 10000 \\
        --n-samples 30 \\
        --out-csv tmp/bench/ss_f1_S3.csv \\
        --helix-recovery-csv tmp/bench/helix_recovery_S3.csv

For ad-hoc targets outside the bench layout (e.g., 11CF, 9SZN), pass
``--target-spec PDB:REF_PATH:INFERENCE_DIR`` (repeatable).
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import Counter
from pathlib import Path

# Reuse machinery
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
from check_chirality import needleman_wunsch  # noqa: E402
from dssp_f1 import (  # noqa: E402
    DSSP_8,
    THREE_STATE,
    confusion_matrix,
    per_class_metrics,
    run_dssp,
)
from gdt_ts import gdt_ts_for_pair  # noqa: E402


def align_and_score(pred_cif: Path, ref_res):
    """Return (pred_ss list, ref_ss list, alignment indices) aligned by NW."""
    pred_res = run_dssp(pred_cif)
    if not pred_res:
        return None
    pred_seq = "".join(aa for aa, _ in pred_res)
    ref_seq = "".join(aa for aa, _ in ref_res)
    pairs = needleman_wunsch(ref_seq, pred_seq, include_mismatches=False)
    if not pairs:
        return None
    ref_idx = [i for i, _ in pairs]
    pred_idx = [j for _, j in pairs]
    ref_ss = [ref_res[i][1] for i in ref_idx]
    pred_ss = [pred_res[j][1] for j in pred_idx]
    return pred_ss, ref_ss, ref_idx, pred_idx


def find_contiguous_runs(ss: list[str], code: str) -> list[tuple[int, int]]:
    """Return ``[(start_idx, length)]`` for every contiguous run of ``code``."""
    runs = []
    i = 0
    n = len(ss)
    while i < n:
        if ss[i] != code:
            i += 1
            continue
        j = i
        while j < n and ss[j] == code:
            j += 1
        runs.append((i, j - i))
        i = j
    return runs


def helix_recovery_fractions(ref_ss: list[str], pred_ss: list[str]) -> list[tuple[int, float]]:
    """For each contiguous H run in ``ref_ss``, find the longest contiguous
    H run in ``pred_ss`` that *overlaps* the ground-truth run, normalize by
    the ground-truth length, return ``[(gt_len, recovery_fraction)]``.

    "Overlapping" is interpreted strictly — the longest pred run is taken
    from the same residue window as the ground-truth run (clipped pred[ss]
    on that window, find longest H block). This mirrors OpenFold paper's
    "longest contiguous fraction of the ground-truth helix recovered".
    """
    out = []
    for start, length in find_contiguous_runs(ref_ss, "H"):
        window = pred_ss[start:start + length]
        # Longest H run inside the window
        best = 0
        cur = 0
        for code in window:
            if code == "H":
                cur += 1
                best = max(best, cur)
            else:
                cur = 0
        out.append((length, best / length))
    return out


def _per_sample(cif: Path, ref_res, ref_cif: Path):
    """Run DSSP + alignment + per-class P/R/F1 + helix recovery + confusion
    matrix + GDT-TS for one sample. Pure function — safe to call from a
    thread / process pool. Returns ``None`` if the sample is unusable.
    """
    res = align_and_score(cif, ref_res)
    if res is None:
        return None
    pred_ss, ref_ss, _r, _p = res
    p_8, r_8, f1_8, m8 = per_class_metrics(pred_ss, ref_ss, DSSP_8)
    pred_3 = [THREE_STATE[c] for c in pred_ss]
    ref_3 = [THREE_STATE[c] for c in ref_ss]
    p_3, r_3, f1_3, m3 = per_class_metrics(pred_3, ref_3, ("α", "β", "c"))
    conf_3 = confusion_matrix(pred_3, ref_3, ("α", "β", "c"))
    gdt_ts_v = gdt_ts_for_pair(cif, ref_cif)
    return {
        "pred_ss": pred_ss,
        "ref_ss": ref_ss,
        "n_aligned": len(pred_ss),
        "prec_8": p_8, "rec_8": r_8, "f1_8": f1_8, "macro_8": m8,
        "prec_3": p_3, "rec_3": r_3, "f1_3": f1_3, "macro_3": m3,
        "confusion_3": conf_3,
        "helix_fractions": helix_recovery_fractions(ref_ss, pred_ss),
        "gdt_ts": gdt_ts_v if (gdt_ts_v is not None) else float("nan"),
    }


def measure_one(step_dir: Path, ref_res, n_samples: int, seed: int = 42,
                max_workers: int | None = None, ref_cif: Path | None = None):
    """Aggregate per-class F1 + helix recovery for one ckpt.

    Parallelises DSSP + alignment across samples via a thread pool —
    each call is dominated by ``mkdssp`` subprocess time, which
    releases the GIL.
    """
    from concurrent.futures import ThreadPoolExecutor

    f1_8 = {c: [] for c in DSSP_8}
    p_8 = {c: [] for c in DSSP_8}
    r_8 = {c: [] for c in DSSP_8}
    f1_3 = {c: [] for c in ("α", "β", "c")}
    p_3 = {c: [] for c in ("α", "β", "c")}
    r_3 = {c: [] for c in ("α", "β", "c")}
    conf_3_total = {(g, p): 0 for g in ("α", "β", "c") for p in ("α", "β", "c")}
    macro_8: list[float] = []
    macro_3: list[float] = []
    helix_fractions: list[tuple[int, float]] = []
    counts_pred = Counter()
    counts_ref = Counter()
    counts_ref_done = False
    n_aligned = 0

    cifs = [step_dir / f"seed-{seed}_sample-{i}" / "model.cif" for i in range(n_samples)]
    cifs = [p for p in cifs if p.exists()]
    if not cifs:
        return None
    with ThreadPoolExecutor(max_workers=max_workers or min(len(cifs), 32)) as ex:
        results = list(ex.map(lambda c: _per_sample(c, ref_res, ref_cif), cifs))
    gdt_vals: list[float] = []
    for r in results:
        if r is None:
            continue
        n_aligned = r["n_aligned"]
        counts_pred.update(r["pred_ss"])
        if not counts_ref_done:
            counts_ref.update(r["ref_ss"])
            counts_ref_done = True
        for c in DSSP_8:
            for store, src in ((f1_8, "f1_8"), (p_8, "prec_8"), (r_8, "rec_8")):
                v = r[src][c]
                if v == v:
                    store[c].append(v)
        for c in ("α", "β", "c"):
            for store, src in ((f1_3, "f1_3"), (p_3, "prec_3"), (r_3, "rec_3")):
                v = r[src][c]
                if v == v:
                    store[c].append(v)
        for k, v in r["confusion_3"].items():
            conf_3_total[k] += v
        macro_8.append(r["macro_8"])
        macro_3.append(r["macro_3"])
        helix_fractions.extend(r["helix_fractions"])
        gt = r.get("gdt_ts", float("nan"))
        if gt == gt:
            gdt_vals.append(gt)
    if not macro_8:
        return None
    def mean_or_nan(v):
        return statistics.mean(v) if v else float("nan")
    return {
        "n_samples": len(macro_8),
        "n_aligned": n_aligned,
        "macro_f1_8": statistics.mean(macro_8),
        "macro_f1_3": statistics.mean(macro_3),
        "f1_8":   {c: mean_or_nan(v) for c, v in f1_8.items()},
        "prec_8": {c: mean_or_nan(v) for c, v in p_8.items()},
        "rec_8":  {c: mean_or_nan(v) for c, v in r_8.items()},
        "f1_3":   {c: mean_or_nan(v) for c, v in f1_3.items()},
        "prec_3": {c: mean_or_nan(v) for c, v in p_3.items()},
        "rec_3":  {c: mean_or_nan(v) for c, v in r_3.items()},
        "confusion_3": conf_3_total,
        "counts_pred": counts_pred,
        "counts_ref": counts_ref,
        "helix_fractions": helix_fractions,
        "gdt_ts": mean_or_nan(gdt_vals),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_argument_group("target source (mutually exclusive)")
    src.add_argument("--targets-file", type=Path,
                     help="CSV with header including 'pdb' column.")
    src.add_argument("--bench-root", type=Path,
                     help="Root containing one {pdb}/_ref.cif.gz per target.")
    src.add_argument("--arm-dir", type=str, default="df3-fapeS3-25e5-c10-g35",
                     help="Arm subdirectory name under each target (default: df3-fapeS3-25e5-c10-g35)")
    src.add_argument("--target-spec", action="append", default=[],
                     help="PDB:REF_PATH:INFERENCE_DIR  (repeatable, overrides bench-root path heuristic for that PDB)")
    ap.add_argument("--pattern", default="step_{step}_n30")
    ap.add_argument("--steps", type=int, nargs="+", required=True)
    ap.add_argument("--n-samples", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-csv", type=Path, default=None,
                    help="Per-step / per-class F1 CSV.")
    ap.add_argument("--helix-recovery-csv", type=Path, default=None,
                    help="Per-helix recovery fractions CSV (every helix × every sample).")
    ap.add_argument("--out-prefix", type=Path, default=None,
                    help="If set, also write {prefix}_f1.{png,pdf,npz} and "
                         "{prefix}_helix.{png,pdf,npz} with the same aggregated "
                         "data shown in the F1 / helix-recovery CSV.")
    args = ap.parse_args()

    # Resolve target list
    targets = []  # [(pdb, ref_path, inference_dir)]
    if args.target_spec:
        for spec in args.target_spec:
            pdb, _ref, infer = spec.split(":")
            targets.append((pdb, Path(_ref), Path(infer)))
    if args.targets_file:
        with open(args.targets_file) as fh:
            for row in csv.DictReader(fh):
                pdb = row["pdb"]
                if any(t[0] == pdb for t in targets):
                    continue
                ref_path = args.bench_root / pdb / "_ref.cif.gz"
                infer = args.bench_root / pdb / args.arm_dir
                if ref_path.exists():
                    targets.append((pdb, ref_path, infer))
    if not targets:
        sys.exit("No targets resolved")

    print(f"Targets ({len(targets)}): {[t[0] for t in targets]}")
    print(f"Steps: {args.steps}")
    print()

    # Pre-DSSP refs (one per target)
    ref_dssp = {pdb: run_dssp(ref) for pdb, ref, _ in targets}
    for pdb, dssp_res in ref_dssp.items():
        if dssp_res:
            cnt = Counter(c for _, c in dssp_res)
            print(f"REF {pdb}: {len(dssp_res)} residues, "
                  f"H={cnt.get('H',0)} E={cnt.get('E',0)} G={cnt.get('G',0)} "
                  f"I={cnt.get('I',0)} T={cnt.get('T',0)} S={cnt.get('S',0)} "
                  f"B={cnt.get('B',0)} P={cnt.get('P',0)} -={cnt.get('-',0)}")
    print()

    rows = []  # for CSV
    helix_rows = []
    for step in args.steps:
        print(f"=== step {step} ===")
        agg_f1_8 = {c: [] for c in DSSP_8}
        agg_f1_3 = {c: [] for c in ("α", "β", "c")}
        agg_macro = []
        for pdb, _ref, infer in targets:
            step_dir = infer / args.pattern.format(step=step)
            if not step_dir.exists():
                continue
            r = measure_one(step_dir, ref_dssp[pdb], args.n_samples, args.seed,
                            ref_cif=_ref)
            if r is None:
                continue
            agg_macro.append(r["macro_f1_3"])
            for c in DSSP_8:
                v = r["f1_8"][c]
                if v == v:
                    agg_f1_8[c].append(v)
            for c in ("α", "β", "c"):
                v = r["f1_3"][c]
                if v == v:
                    agg_f1_3[c].append(v)
            row = {
                "step": step, "target": pdb, "macro_f1_8": r["macro_f1_8"],
                "macro_f1_3": r["macro_f1_3"],
                "gdt_ts": r.get("gdt_ts", float("nan")),
            }
            for c in DSSP_8:
                row[f"f1_{c}"] = r["f1_8"][c]
                row[f"prec_{c}"] = r["prec_8"][c]
                row[f"rec_{c}"] = r["rec_8"][c]
            for c in ("α", "β", "c"):
                row[f"f1_3_{c}"] = r["f1_3"][c]
                row[f"prec_3_{c}"] = r["prec_3"][c]
                row[f"rec_3_{c}"] = r["rec_3"][c]
            # 3x3 confusion matrix (counts) → flat keys
            for (gt, pred), cnt in r["confusion_3"].items():
                row[f"conf_gt{gt}_pred{pred}"] = cnt
            rows.append(row)
            for gt_len, frac in r["helix_fractions"]:
                helix_rows.append({"step": step, "target": pdb,
                                   "helix_len": gt_len, "recovery": frac})
        if not agg_macro:
            continue
        print(f"  macro F1 (3st): {statistics.mean(agg_macro):.4f}")
        gdt_at_step = [r["gdt_ts"] for r in rows
                       if r["step"] == step and r["gdt_ts"] == r["gdt_ts"]]
        if gdt_at_step:
            print(f"  GDT-TS:         {statistics.mean(gdt_at_step):.4f}")
        print("  per-class F1 (8st):")
        for c in DSSP_8:
            if not agg_f1_8[c]:
                continue
            print(f"    {c}: {statistics.mean(agg_f1_8[c]):.4f}  (n_targets={len(agg_f1_8[c])})")
        print("  per-class F1 (3st):")
        for c in ("α", "β", "c"):
            if agg_f1_3[c]:
                print(f"    {c}: {statistics.mean(agg_f1_3[c]):.4f}")
        print()

    if args.out_csv:
        cols = ["step", "target", "macro_f1_8", "macro_f1_3"]
        for c in DSSP_8:
            cols += [f"f1_{c}", f"prec_{c}", f"rec_{c}"]
        for c in ("α", "β", "c"):
            cols += [f"f1_3_{c}", f"prec_3_{c}", f"rec_3_{c}"]
        for g in ("α", "β", "c"):
            for p in ("α", "β", "c"):
                cols.append(f"conf_gt{g}_pred{p}")
        with open(args.out_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.out_csv}  ({len(rows)} rows)")

    if args.helix_recovery_csv:
        with open(args.helix_recovery_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["step", "target", "helix_len", "recovery"])
            w.writeheader()
            w.writerows(helix_rows)
        print(f"wrote {args.helix_recovery_csv}  ({len(helix_rows)} rows)")

    if args.out_prefix:
        _make_plots(rows, helix_rows, args.steps, args.out_prefix)


def _make_plots(rows: list[dict], helix_rows: list[dict],
                steps: list[int], prefix: Path) -> None:
    """Write {prefix}_f1.{png,pdf,npz} + {prefix}_helix.{png,pdf,npz}."""
    import matplotlib  # noqa: PLC0415
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    prefix.parent.mkdir(exist_ok=True, parents=True)

    # --- (1) F1 trajectory per class, aggregated across targets per step.
    # Means across targets at each step.
    classes_8 = list(DSSP_8)
    classes_3 = ["α", "β", "c"]
    f1_8_means = {c: [] for c in classes_8}
    f1_3_means = {c: [] for c in classes_3}
    macro_3 = []
    for step in steps:
        per_step = [r for r in rows if r["step"] == step]
        if not per_step:
            for c in classes_8:
                f1_8_means[c].append(float("nan"))
            for c in classes_3:
                f1_3_means[c].append(float("nan"))
            macro_3.append(float("nan"))
            continue
        macro_3.append(np.nanmean([r["macro_f1_3"] for r in per_step]))
        for c in classes_8:
            f1_8_means[c].append(np.nanmean([r[f"f1_{c}"] for r in per_step]))
        for c in classes_3:
            f1_3_means[c].append(np.nanmean([r[f"f1_3_{c}"] for r in per_step]))

    fig, (ax8, ax3) = plt.subplots(1, 2, figsize=(12, 5))
    ax8.set_title("DSSP 8-state per-class F1")
    for c in classes_8:
        if all(v != v for v in f1_8_means[c]):  # all NaN
            continue
        ax8.plot(steps, f1_8_means[c], marker="o", label=c)
    ax8.set_xlabel("training step")
    ax8.set_ylabel("F1")
    ax8.legend(fontsize=8, ncol=1, loc="center left", bbox_to_anchor=(1.02, 0.5))
    ax8.grid(alpha=0.3)

    ax3.set_title("DSSP 3-state F1 (α=H+G+I+P, β=E+B, c=T+S+coil)")
    for c, color in zip(classes_3, ("tab:red", "tab:blue", "tab:gray"), strict=True):
        ax3.plot(steps, f1_3_means[c], marker="o", color=color, label=c)
    ax3.plot(steps, macro_3, marker="s", color="dimgray", lw=1.5, label="macro 3-state")
    # GDT-TS overlay (target-mean, Kabsch-superposed CA, cutoffs {1,2,4,8} Å).
    # Plotted as the dominant black curve since it is the structure-accuracy
    # summary; SS-F1 curves are class-level diagnostics.
    gdt_means = []
    for step in steps:
        per_step = [r["gdt_ts"] for r in rows
                    if r["step"] == step and "gdt_ts" in r
                    and r["gdt_ts"] == r["gdt_ts"]]
        gdt_means.append(np.mean(per_step) if per_step else float("nan"))
    if any(v == v for v in gdt_means):
        ax3.plot(steps, gdt_means, marker="^", color="black", lw=3, label="GDT-TS")
    ax3.set_xlabel("training step")
    ax3.set_ylabel("F1  /  GDT-TS")
    ax3.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5))
    ax3.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(str(prefix) + "_f1.png", dpi=140)
    fig.savefig(str(prefix) + "_f1.pdf")
    plt.close(fig)
    np.savez(
        str(prefix) + "_f1.npz",
        steps=np.array(steps),
        classes_8=np.array(classes_8),
        f1_8=np.array([f1_8_means[c] for c in classes_8]),
        classes_3=np.array(classes_3),
        f1_3=np.array([f1_3_means[c] for c in classes_3]),
        macro_3=np.array(macro_3),
        gdt_ts=np.array(gdt_means),
    )
    print(f"wrote {prefix}_f1.png / .pdf / .npz")

    # --- (1b) Precision / Recall trajectories for the 3-state α and β.
    # Diagnoses whether high F1 comes from precision (FP-controlled) or
    # recall (FN-controlled), and exposes the "helix eats β" pattern
    # via the confusion matrix entry conf_gtβ_predα / total_GT_β.
    pr_3 = {c: ([], []) for c in classes_3}  # (prec, rec) lists per step
    helix_eats_strand = []  # fraction of GT-β predicted as α, per step
    strand_eats_helix = []  # fraction of GT-α predicted as β, per step
    for step in steps:
        per_step = [r for r in rows if r["step"] == step]
        if not per_step:
            for c in classes_3:
                pr_3[c][0].append(float("nan"))
                pr_3[c][1].append(float("nan"))
            helix_eats_strand.append(float("nan"))
            strand_eats_helix.append(float("nan"))
            continue
        for c in classes_3:
            pr_3[c][0].append(np.nanmean([r[f"prec_3_{c}"] for r in per_step]))
            pr_3[c][1].append(np.nanmean([r[f"rec_3_{c}"] for r in per_step]))
        # Confusion-matrix-derived "eats" metrics: fraction of GT-β residues
        # predicted as α across all (target, sample) pairs.
        beta_total = sum(r["conf_gtβ_predα"] + r["conf_gtβ_predβ"] + r["conf_gtβ_predc"] for r in per_step)
        alpha_total = sum(r["conf_gtα_predα"] + r["conf_gtα_predβ"] + r["conf_gtα_predc"] for r in per_step)
        b2a = sum(r["conf_gtβ_predα"] for r in per_step)
        a2b = sum(r["conf_gtα_predβ"] for r in per_step)
        helix_eats_strand.append(b2a / beta_total if beta_total else float("nan"))
        strand_eats_helix.append(a2b / alpha_total if alpha_total else float("nan"))
    fig_pr, (axp, axr, axe) = plt.subplots(1, 3, figsize=(16, 5))
    axp.set_title("Precision (3-state)")
    axr.set_title("Recall (3-state)")
    for c, color in zip(classes_3, ("tab:red", "tab:blue", "tab:gray"), strict=True):
        axp.plot(steps, pr_3[c][0], marker="o", color=color, label=c)
        axr.plot(steps, pr_3[c][1], marker="o", color=color, label=c)
    for a in (axp, axr):
        a.set_xlabel("training step")
        a.set_ylabel("metric")
        a.grid(alpha=0.3)
        a.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5))
    axe.set_title("Confusion: GT-β→pred-α (helix eats strand)\nand GT-α→pred-β")
    axe.plot(steps, helix_eats_strand, marker="o", color="tab:red", label="GT-β predicted as α")
    axe.plot(steps, strand_eats_helix, marker="s", color="tab:blue", label="GT-α predicted as β")
    axe.set_xlabel("training step")
    axe.set_ylabel("fraction of GT class")
    axe.grid(alpha=0.3)
    axe.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig_pr.tight_layout()
    fig_pr.savefig(str(prefix) + "_pr.png", dpi=140)
    fig_pr.savefig(str(prefix) + "_pr.pdf")
    plt.close(fig_pr)
    np.savez(
        str(prefix) + "_pr.npz",
        steps=np.array(steps),
        classes_3=np.array(classes_3),
        prec_3=np.array([pr_3[c][0] for c in classes_3]),
        rec_3=np.array([pr_3[c][1] for c in classes_3]),
        helix_eats_strand=np.array(helix_eats_strand),
        strand_eats_helix=np.array(strand_eats_helix),
    )
    print(f"wrote {prefix}_pr.png / .pdf / .npz")

    # --- (2) Helix recovery histogram per step (3D + 2D ridge).
    if helix_rows:
        from matplotlib.collections import PolyCollection  # noqa: PLC0415
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401, PLC0415

        bins = np.linspace(0.0, 1.0001, 21)  # 20 bins on [0, 1]
        bin_centers = (bins[:-1] + bins[1:]) / 2
        per_step_counts = {}
        for step in steps:
            recoveries = [r["recovery"] for r in helix_rows if r["step"] == step]
            counts, _ = np.histogram(recoveries, bins=bins)
            per_step_counts[step] = counts.astype(np.int64)

        # 3D ridge / waterfall: one filled curve per step, stacked along Y axis.
        # This is the OpenFold paper's Fig 4c presentation style — emphasises
        # the bimodal "all-or-nothing" mass at 0.0 and 1.0 across training.
        fig3d = plt.figure(figsize=(10, 6))
        ax3d = fig3d.add_subplot(111, projection="3d")
        ax3d.set_title("α-helix recovery histogram across training (3D ridge)")
        verts = []
        zs = []
        for step in steps:
            counts = per_step_counts[step].astype(float)
            total = counts.sum()
            if total == 0:
                pmf = np.zeros_like(counts)
            else:
                pmf = counts / total
            # Build polygon for this step's filled histogram in (x=bin, z=pmf) plane.
            verts.append([(0.0, 0.0)]
                         + list(zip(bin_centers, pmf, strict=True))
                         + [(1.0, 0.0)])
            zs.append(step)
        poly = PolyCollection(verts, facecolors=plt.cm.viridis(
            np.linspace(0, 1, len(steps))), edgecolors="black", linewidths=0.4, alpha=0.85)
        ax3d.add_collection3d(poly, zs=zs, zdir="y")
        ax3d.set_xlabel("recovery fraction")
        ax3d.set_ylabel("training step")
        ax3d.set_zlabel("probability mass")
        ax3d.set_xlim(0, 1)
        ax3d.set_ylim(min(zs), max(zs))
        # Auto-scale z based on observed max pmf so all curves fit.
        max_pmf = max((per_step_counts[s].astype(float) / max(per_step_counts[s].sum(), 1)).max()
                      for s in steps if per_step_counts[s].sum() > 0)
        ax3d.set_zlim(0, max_pmf * 1.05)
        ax3d.view_init(elev=25, azim=-65)
        fig3d.tight_layout()
        fig3d.savefig(str(prefix) + "_helix3d.png", dpi=140)
        fig3d.savefig(str(prefix) + "_helix3d.pdf")
        plt.close(fig3d)

        # Also keep the 2D overlay for quick scanning.
        fig2, ax = plt.subplots(figsize=(8, 5))
        ax.set_title("α-helix recovery histogram by training step")
        for step in steps:
            counts = per_step_counts[step]
            total = counts.sum()
            if total == 0:
                continue
            ax.plot(bin_centers, counts / total, marker="o",
                    label=f"step {step}  (n={total})")
        ax.set_xlabel("recovery fraction (longest pred-H run / GT helix length)")
        ax.set_ylabel("probability mass")
        ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5))
        ax.grid(alpha=0.3)
        fig2.tight_layout()
        fig2.savefig(str(prefix) + "_helix.png", dpi=140)
        fig2.savefig(str(prefix) + "_helix.pdf")
        plt.close(fig2)
        np.savez(
            str(prefix) + "_helix.npz",
            steps=np.array(steps),
            bin_centers=bin_centers,
            counts=np.array([per_step_counts[s] for s in steps]),
        )
        print(f"wrote {prefix}_helix3d.png / .pdf  +  {prefix}_helix.png / .pdf / .npz")


if __name__ == "__main__":
    main()
