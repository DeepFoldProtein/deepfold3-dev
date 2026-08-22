"""Structure-accuracy trajectory: GDT-TS / TM-score / lDDT.

Companion to ``ss_learning.py``. Where ss_learning tracks secondary-structure
F1, this tracks the three global structure-accuracy metrics over training,
reading the *per-sample* values already cached by the analysis pipeline:

  * GDT-TS  ← ``lddt.json["gdt_ts"]``         (OST --rigid-scores; falls back to
              the legacy ``ss_metrics.json["gdt_ts"]`` LGA value for old steps)
  * TM-score← ``tm_score.json["tm_ref"]``     (OST --tm-score)
  * lDDT    ← ``lddt.json["lddt"]``           (OST, no-stereochecks)

Each ``step_{N}_n30/seed-S_sample-i`` directory contributes one value per
metric, so a step has ``n_targets × n_samples`` samples (per-sample
population). The figure ``{prefix}_struct.png`` has:

  * one trend row per metric: per-sample mean + median lines and a shaded
    inter-quartile (25–75 %) band over *all* steps;
  * a bottom row of box plots at the most recent ``--recent-box`` checkpoints
    (full per-sample distribution).

Same target-resolution / parallelism conventions as ``ss_learning.py``
(``--targets-file``/``--bench-root``/``--arm-dir`` or ``--target-spec``;
``--workers`` fans the per-step cache reads across processes).

Usage:
    uv run python scripts/struct_accuracy.py \\
        --targets-file tmp/bench/targets.csv \\
        --bench-root tmp/bench \\
        --arm-dir df3-fapeK5-25e5-c10-g35 \\
        --steps $(seq 1000 500 75000) \\
        --workers 64 \\
        --out-prefix tmp/bench/ss_analysis_full/fapeK5 \\
        --out-csv tmp/bench/ss_analysis_full/fapeK5_struct.csv
"""

from __future__ import annotations

import argparse
import csv
import functools
import json
import sys
from pathlib import Path

# (metric name, [(cache filename, json key), ...]) — first source that yields a
# value wins. GDT-TS prefers the OST --rigid-scores value in lddt.json and falls
# back to the legacy LGA value in ss_metrics.json for steps scored before the
# OST-GDT-TS integration.
METRICS = (
    ("GDT-TS", (("lddt.json", "gdt_ts"), ("ss_metrics.json", "gdt_ts"))),
    ("TM", (("tm_score.json", "tm_ref"),)),
    ("lDDT", (("lddt.json", "lddt"),)),
)
COLORS = {"GDT-TS": "black", "TM": "tab:green", "lDDT": "tab:purple"}


def _read_value(sample_dir: Path, fname: str, key: str) -> float | None:
    """Read one cached metric value; ``None`` if missing/invalid."""
    p = sample_dir / fname
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    v = d.get(key)
    if v is None:
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if v == v else None  # drop NaN


def _process_step(step: int, targets: list, n_samples: int, seed: int, pattern: str) -> tuple:
    """Collect per-sample values for one ``step``.

    Picklable (module-level) so it can run in a process-pool worker.
    Returns ``(step, {metric_name: [value, ...]})``.
    """
    out: dict[str, list[float]] = {name: [] for name, _ in METRICS}
    for _pdb, _ref, infer in targets:
        step_dir = infer / pattern.format(step=step, n=n_samples)
        if not step_dir.exists():
            continue
        for i in range(n_samples):
            sd = step_dir / f"seed-{seed}_sample-{i}"
            if not sd.exists():
                continue
            for name, sources in METRICS:
                v = next((x for fname, key in sources if (x := _read_value(sd, fname, key)) is not None), None)
                if v is not None:
                    out[name].append(v)
    return step, out


def _make_plot(steps: list[int], per_step: dict[int, dict[str, list[float]]], prefix: Path, recent_box: int) -> None:
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    names = [n for n, _ in METRICS]
    fig = plt.figure(figsize=(15, 16))
    gs = fig.add_gridspec(len(names) + 1, len(names), height_ratios=[1] * len(names) + [1.1])

    # --- trend rows: per-sample mean + median + IQR band over all steps.
    top = None
    for j, name in enumerate(names):
        ax = fig.add_subplot(gs[j, :], sharex=top)
        top = top or ax
        color = COLORS.get(name, f"C{j}")
        vals = [per_step.get(s, {}).get(name, []) for s in steps]
        mean = np.array([np.mean(v) if v else np.nan for v in vals])
        med = np.array([np.median(v) if v else np.nan for v in vals])
        q25 = np.array([np.percentile(v, 25) if v else np.nan for v in vals])
        q75 = np.array([np.percentile(v, 75) if v else np.nan for v in vals])
        ax.fill_between(steps, q25, q75, color=color, alpha=0.18, label="IQR (25–75 %, per-sample)")
        ax.plot(steps, mean, color=color, lw=2, marker="o", ms=3, label="mean")
        ax.plot(steps, med, color=color, lw=1, ls="--", alpha=0.85, label="median")
        n_per = next((len(v) for v in vals if v), 0)
        ax.set_title(f"{name}  (per-step, n≈{n_per} samples/step)")
        ax.set_ylabel(name)
        ax.set_ylim(-0.02, 1.02)  # GDT/TM/lDDT all in [0,1] — fixed for absolute comparability
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")
    top.set_xlabel("training step")

    # --- bottom row: box plots at the most recent checkpoints.
    recent = [s for s in steps if per_step.get(s, {}).get(names[0])][-recent_box:]
    for j, name in enumerate(names):
        ax = fig.add_subplot(gs[len(names), j])
        color = COLORS.get(name, f"C{j}")
        ax.set_xlabel("step")
        if j == 0:
            ax.set_ylabel("per-sample distribution")
        ax.set_ylim(-0.02, 1.02)  # [0,1] score range — comparable across metrics/panels
        ax.grid(alpha=0.3, axis="y")
        # Drop checkpoints with no samples for this metric before plotting:
        # matplotlib's boxplot rejects a fully-empty X (and any labels/X length
        # mismatch) with "Dimensions of labels and X must be compatible".
        pairs = [(str(s), per_step.get(s, {}).get(name, [])) for s in recent]
        pairs = [(lbl, d) for lbl, d in pairs if d]
        ax.set_title(f"{name} — recent {len(pairs)} ckpts")
        if not pairs:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes, color="gray")
            continue
        labels, data = zip(*pairs, strict=True)
        bp = ax.boxplot(
            list(data),
            tick_labels=list(labels),
            showfliers=True,
            patch_artist=True,
            flierprops={"marker": ".", "ms": 2, "alpha": 0.3},
            medianprops={"color": "black"},
        )
        for patch in bp["boxes"]:
            patch.set(facecolor=color, alpha=0.35)
        ax.tick_params(axis="x", labelrotation=90, labelsize=7)

    fig.suptitle(f"Structure accuracy — {prefix.name}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(str(prefix) + "_struct.png", dpi=140)
    fig.savefig(str(prefix) + "_struct.pdf")
    plt.close(fig)
    print(f"wrote {prefix}_struct.png / .pdf")


def _write_csv(steps: list[int], per_step: dict[int, dict[str, list[float]]], out_csv: Path) -> None:
    import numpy as np  # noqa: PLC0415

    cols = ["step", "metric", "n", "mean", "median", "q25", "q75", "min", "max"]
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for s in steps:
            for name, _ in METRICS:
                v = per_step.get(s, {}).get(name, [])
                if not v:
                    continue
                a = np.array(v)
                w.writerow(
                    {
                        "step": s,
                        "metric": name,
                        "n": len(a),
                        "mean": round(float(a.mean()), 5),
                        "median": round(float(np.median(a)), 5),
                        "q25": round(float(np.percentile(a, 25)), 5),
                        "q75": round(float(np.percentile(a, 75)), 5),
                        "min": round(float(a.min()), 5),
                        "max": round(float(a.max()), 5),
                    }
                )
    print(f"wrote {out_csv}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--targets-file", type=Path, help="CSV with header including 'pdb' column.")
    ap.add_argument("--bench-root", type=Path, help="Root containing one {pdb}/_ref.cif.gz per target.")
    ap.add_argument(
        "--arm-dir", type=str, default="df3-fapeK5-25e5-c10-g35", help="Arm subdirectory name under each target."
    )
    ap.add_argument("--target-spec", action="append", default=[], help="PDB:REF_PATH:INFERENCE_DIR (repeatable).")
    # {n} expands to --n-samples so the per-sample suffix tracks the driver
    # (analysis.sh uses n5); a hardcoded _n30 silently matched nothing under n5.
    ap.add_argument("--pattern", default="step_{step}_n{n}")
    ap.add_argument("--steps", type=int, nargs="+", required=True)
    ap.add_argument("--n-samples", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Process-pool size for the per-step cache reads "
        "(steps independent). High values speed up full-history "
        "rebuilds; default 1 (sequential).",
    )
    ap.add_argument("--recent-box", type=int, default=12, help="Number of most-recent checkpoints to box-plot.")
    ap.add_argument("--out-prefix", type=Path, default=None, help="Write {prefix}_struct.{png,pdf}.")
    ap.add_argument("--out-csv", type=Path, default=None, help="Per-step / per-metric distribution-stats CSV.")
    args = ap.parse_args()

    targets = []  # [(pdb, ref_path, inference_dir)]
    for spec in args.target_spec:
        pdb, _ref, infer = spec.split(":")
        targets.append((pdb, Path(_ref), Path(infer)))
    if args.targets_file:
        with open(args.targets_file) as fh:
            for row in csv.DictReader(fh):
                pdb = row["pdb"]
                if any(t[0] == pdb for t in targets):
                    continue
                infer = args.bench_root / pdb / args.arm_dir
                targets.append((pdb, args.bench_root / pdb / "_ref.cif.gz", infer))
    if not targets:
        sys.exit("No targets resolved")
    print(f"Targets ({len(targets)}): {[t[0] for t in targets]}")
    print(f"Steps: {len(args.steps)} ({args.steps[0]}..{args.steps[-1]})")

    workers = min(max(1, args.workers), len(args.steps))
    step_fn = functools.partial(
        _process_step, targets=targets, n_samples=args.n_samples, seed=args.seed, pattern=args.pattern
    )
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor

        print(f"(step-parallel: {workers} workers)")
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(step_fn, args.steps))  # ex.map preserves order
    else:
        results = [step_fn(s) for s in args.steps]
    per_step = {s: d for s, d in results}

    for name, _ in METRICS:
        last = next((per_step[s][name] for s in reversed(args.steps) if per_step.get(s, {}).get(name)), [])
        if last:
            import statistics

            print(f"  {name:7s} latest step mean={statistics.mean(last):.4f}  n={len(last)}")

    if args.out_csv:
        _write_csv(args.steps, per_step, args.out_csv)
    if args.out_prefix:
        _make_plot(args.steps, per_step, args.out_prefix, args.recent_box)


if __name__ == "__main__":
    main()
