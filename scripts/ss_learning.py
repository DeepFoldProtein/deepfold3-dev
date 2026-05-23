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
from dssp_f1 import DSSP_8, THREE_STATE, per_class_f1, run_dssp  # noqa: E402


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


def measure_one(step_dir: Path, ref_res, n_samples: int, seed: int = 42):
    """Aggregate per-class F1 + helix recovery for one ckpt."""
    f1_by_class_8 = {c: [] for c in DSSP_8}
    f1_by_class_3 = {c: [] for c in ("α", "β", "c")}
    macro_8 = []
    macro_3 = []
    helix_fractions: list[tuple[int, float]] = []
    counts_pred = Counter()
    counts_ref_done = False
    counts_ref = Counter()
    n_aligned = 0
    for i in range(n_samples):
        cif = step_dir / f"seed-{seed}_sample-{i}" / "model.cif"
        if not cif.exists():
            continue
        res = align_and_score(cif, ref_res)
        if res is None:
            continue
        pred_ss, ref_ss, _ref_idx, _pred_idx = res
        n_aligned = len(pred_ss)
        counts_pred.update(pred_ss)
        if not counts_ref_done:
            counts_ref.update(ref_ss)
            counts_ref_done = True
        # 8-state per-class F1
        per_class, m8 = per_class_f1(pred_ss, ref_ss, DSSP_8)
        for c, f in per_class.items():
            if f == f:  # skip NaN
                f1_by_class_8[c].append(f)
        macro_8.append(m8)
        # 3-state
        pred_3 = [THREE_STATE[c] for c in pred_ss]
        ref_3 = [THREE_STATE[c] for c in ref_ss]
        per_class_3_, m3 = per_class_f1(pred_3, ref_3, ("α", "β", "c"))
        for c, f in per_class_3_.items():
            if f == f:
                f1_by_class_3[c].append(f)
        macro_3.append(m3)
        # Helix recovery
        helix_fractions.extend(helix_recovery_fractions(ref_ss, pred_ss))
    if not macro_8:
        return None
    return {
        "n_samples": len(macro_8),
        "n_aligned": n_aligned,
        "macro_f1_8": statistics.mean(macro_8),
        "macro_f1_3": statistics.mean(macro_3),
        "f1_8": {c: (statistics.mean(v) if v else float("nan")) for c, v in f1_by_class_8.items()},
        "f1_3": {c: (statistics.mean(v) if v else float("nan")) for c, v in f1_by_class_3.items()},
        "counts_pred": counts_pred,
        "counts_ref": counts_ref,
        "helix_fractions": helix_fractions,
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
            r = measure_one(step_dir, ref_dssp[pdb], args.n_samples, args.seed)
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
            rows.append({
                "step": step, "target": pdb, "macro_f1_8": r["macro_f1_8"],
                "macro_f1_3": r["macro_f1_3"],
                **{f"f1_{c}": r["f1_8"][c] for c in DSSP_8},
                **{f"f1_3_{c}": r["f1_3"][c] for c in ("α", "β", "c")},
            })
            for gt_len, frac in r["helix_fractions"]:
                helix_rows.append({"step": step, "target": pdb,
                                   "helix_len": gt_len, "recovery": frac})
        if not agg_macro:
            continue
        print(f"  macro F1 (3st): {statistics.mean(agg_macro):.4f}")
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
        cols = ["step", "target", "macro_f1_8", "macro_f1_3"] \
               + [f"f1_{c}" for c in DSSP_8] \
               + [f"f1_3_{c}" for c in ("α", "β", "c")]
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
    ax8.legend(fontsize=8, ncol=2)
    ax8.grid(alpha=0.3)

    ax3.set_title("DSSP 3-state F1 (α=H+G+I+P, β=E+B, c=T+S+coil)")
    for c, color in zip(classes_3, ("tab:red", "tab:blue", "tab:gray"), strict=True):
        ax3.plot(steps, f1_3_means[c], marker="o", color=color, label=c)
    ax3.plot(steps, macro_3, marker="s", color="black", lw=2, label="macro 3-state")
    ax3.set_xlabel("training step")
    ax3.set_ylabel("F1")
    ax3.legend(fontsize=8)
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
    )
    print(f"wrote {prefix}_f1.png / .pdf / .npz")

    # --- (2) Helix recovery histogram per step.
    if helix_rows:
        bins = np.linspace(0.0, 1.0001, 21)  # 20 bins on [0, 1]
        bin_centers = (bins[:-1] + bins[1:]) / 2
        per_step_counts = {}
        for step in steps:
            recoveries = [r["recovery"] for r in helix_rows if r["step"] == step]
            counts, _ = np.histogram(recoveries, bins=bins)
            per_step_counts[step] = counts.astype(np.int64)
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
        ax.legend(fontsize=8)
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
        print(f"wrote {prefix}_helix.png / .pdf / .npz")


if __name__ == "__main__":
    main()
