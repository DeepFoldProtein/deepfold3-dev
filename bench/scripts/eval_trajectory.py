"""FoldBench eval trajectory: mean/oracle lDDT / TM / GDT-TS over training.

Reads the per-step ``summary.csv`` aggregates written by ``eval_ost.py`` under
``bench/runs/{series}/eval/{category}/step_{N}/`` and emits, for one category:

  * ``trajectory.csv``       — one row per step (n_scored + mean/oracle metrics)
  * ``trajectory_bands.csv`` — first step each metric's mean crosses each 0.1
                               band, and the step-delta from the previous band
                               (how many steps a 0.1 gain cost)
  * ``trajectory.png/pdf``   — one panel per metric (mean solid + oracle dashed),
                               with an inflection marker and pre/post slopes

Also prints a pre-vs-post-``--inflection`` linear-slope table per metric — the
"is the trajectory steepening?" signal. Mean over ``n_scored`` targets (330 for
monomer_protein) is low-noise, so raw first-crossings are used (no smoothing).

Usage::

    uv run python bench/scripts/eval_trajectory.py --series df3v1 \\
        --category monomer_protein --inflection 110000
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# (display name, mean column, oracle column, plot color)
METRICS = [
    ("lDDT", "mean_lddt", "oracle_lddt", "tab:purple"),
    ("TM", "mean_tm_score", "oracle_tm_score", "tab:green"),
    ("GDT-TS", "mean_gdt_ts", "oracle_gdt_ts", "black"),
]
_BANDS = tuple(round(0.1 * k, 1) for k in range(1, 11))


def read_trajectory(eval_dir: Path) -> list[dict]:
    """One dict per step from every ``step_*/summary.csv`` (sorted by step)."""
    rows = []
    for sc in eval_dir.glob("step_*/summary.csv"):
        for r in csv.DictReader(open(sc)):
            try:
                rec = {"step": int(r["step"]), "n_scored": int(r.get("n_scored") or 0)}
            except (KeyError, ValueError):
                continue
            for _name, mcol, ocol, _c in METRICS:
                for col in (mcol, ocol):
                    v = r.get(col)
                    rec[col] = float(v) if v not in (None, "") else None
            rows.append(rec)
    rows.sort(key=lambda x: x["step"])
    return rows


def band_crossings(steps, ys, bands=_BANDS):
    """First step whose mean reaches each 0.1 band + delta from previous band."""
    out, prev = [], None
    for thr in bands:
        hit = next((s for s, y in zip(steps, ys, strict=True) if y is not None and y >= thr), None)
        if hit is None:
            break
        out.append((thr, hit, (hit - prev) if prev is not None else None))
        prev = hit
    return out


def slope_per_1k(steps, ys, lo, hi):
    """Least-squares slope (metric units per 1000 steps) over ``[lo, hi]``."""
    import numpy as np  # noqa: PLC0415

    xy = [(s, y) for s, y in zip(steps, ys, strict=True) if y is not None and lo <= s <= hi]
    if len(xy) < 2:
        return None, len(xy)
    xs, yv = np.array([p[0] for p in xy]), np.array([p[1] for p in xy])
    return float(np.polyfit(xs, yv, 1)[0] * 1000), len(xy)


def write_trajectory_csv(rows, out_csv):
    cols = ["step", "n_scored"] + [c for _n, m, o, _c in METRICS for c in (m, o)]
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()})
    print(f"wrote {out_csv}")


def write_bands_csv(rows, out_csv):
    steps = [r["step"] for r in rows]
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "threshold", "first_step", "delta_steps_from_prev_band"])
        for name, mcol, _o, _c in METRICS:
            ys = [r.get(mcol) for r in rows]
            for thr, s, d in band_crossings(steps, ys):
                w.writerow([name, thr, s, "" if d is None else d])
    print(f"wrote {out_csv}")


def make_plot(rows, prefix, inflection, series, category):
    import matplotlib  # noqa: PLC0415
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    steps = [r["step"] for r in rows]
    fig, axs = plt.subplots(len(METRICS), 1, figsize=(13, 9), sharex=True)
    for ax, (name, mcol, ocol, color) in zip(axs, METRICS, strict=True):
        mean = [r.get(mcol) for r in rows]
        orac = [r.get(ocol) for r in rows]
        ax.plot(steps, mean, color=color, lw=1.6, marker=".", ms=4, label="mean")
        if any(o is not None for o in orac):
            ax.plot(steps, orac, color=color, lw=1.0, ls="--", marker=".", ms=4, alpha=0.7,
                    label="oracle (best-of-N)")
        ax.axvline(inflection, color="red", ls="--", alpha=0.6)
        pre, npre = slope_per_1k(steps, mean, steps[0], inflection)
        post, npost = slope_per_1k(steps, mean, inflection, steps[-1])
        sub = []
        if pre is not None:
            sub.append(f"pre {pre:+.4f}/1k")
        if post is not None:
            sub.append(f"post {post:+.4f}/1k")
        if pre and post and pre != 0:
            sub.append(f"×{post / pre:.1f}")
        ax.set_title(f"{name}   ({'  '.join(sub)})")
        ax.set_ylabel(name)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)
    axs[0].text(inflection, 1.0, f" inflection @ {inflection}", color="red", va="top", fontsize=9)
    axs[0].set_title(f"{axs[0].get_title()}\n{series} / {category}  (mean over {rows[-1]['n_scored']} targets)")
    axs[-1].set_xlabel("training step")
    fig.tight_layout()
    fig.savefig(str(prefix) + ".png", dpi=140)
    fig.savefig(str(prefix) + ".pdf")
    plt.close(fig)
    print(f"wrote {prefix}.png / .pdf")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--series", default="df3v1")
    ap.add_argument("--category", default="monomer_protein")
    ap.add_argument("--eval-root", type=Path, default=REPO / "bench" / "runs")
    ap.add_argument("--inflection", type=int, default=110000,
                    help="Step to split pre/post slopes and mark on the figure.")
    args = ap.parse_args()

    eval_dir = args.eval_root / args.series / "eval" / args.category
    rows = read_trajectory(eval_dir)
    if not rows:
        raise SystemExit(f"no summary.csv under {eval_dir}")
    steps = [r["step"] for r in rows]
    print(f"{args.series}/{args.category}: {len(rows)} steps {steps[0]}..{steps[-1]}, "
          f"n_scored={rows[-1]['n_scored']}")

    write_trajectory_csv(rows, eval_dir / "trajectory.csv")
    write_bands_csv(rows, eval_dir / "trajectory_bands.csv")
    make_plot(rows, eval_dir / "trajectory", args.inflection, args.series, args.category)

    print(f"\npre/post-{args.inflection} mean-slope (per 1000 steps):")
    print(f"  {'metric':7s} | {'pre':>10} | {'post':>10} | {'×':>5} | {'band crossings (0.1)':s}")
    for name, mcol, _o, _c in METRICS:
        ys = [r.get(mcol) for r in rows]
        pre, _ = slope_per_1k(steps, ys, steps[0], args.inflection)
        post, _ = slope_per_1k(steps, ys, args.inflection, steps[-1])
        ratio = f"{post / pre:.1f}" if (pre and post and pre != 0) else "--"
        bands = "  ".join(f"{thr:.1f}@{s}" + (f"(+{d})" if d else "") for thr, s, d in band_crossings(steps, ys))
        pre_s = f"{pre:+.4f}" if pre is not None else "--"
        post_s = f"{post:+.4f}" if post is not None else "--"
        print(f"  {name:7s} | {pre_s:>10} | {post_s:>10} | {ratio:>5} | {bands}")


if __name__ == "__main__":
    main()
