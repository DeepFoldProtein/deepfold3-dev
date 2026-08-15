"""What does removing triangle attention actually cost — at matched quality?

r17 differs from r16 by two things (no triangle attention, plus the MHR router),
so the interesting output of the arm is not "which run is ahead" but *which
capabilities depend on what*. Comparing at the same step conflates the two
questions: r17 is behind on overall structure, so every downstream metric is
behind, and a deficit only means something if it is **larger than the overall
one**.

The control is to compare at matched quality rather than matched step. Plotting
a secondary-structure metric against TM turns "r17 is behind" into a single
shared curve; a capability that genuinely depends on triangle attention shows up
as r17 sitting *below* r16's curve at the same TM, not merely to the left of it.

This is only readable once both runs are out of the transition — TM goes 0.46 ->
0.73 in 500 steps, so interpolating inside that is meaningless. The script says
so rather than printing a number when the data cannot support one.

Usage:
    python export/mhr-trimul/ablation_map.py
"""

from __future__ import annotations

import csv
import logging
import math
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
BENCH = REPO / "tmp" / "bench" / "ss_analysis_full"
OUT = Path(__file__).resolve().parent
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("ablation")

RUNS = {"r16 (with TriAttn)": "v5p0r16", "r17 (no TriAttn + MHR)": "v5p0r17"}
METRICS = ["f1_3_β", "f1_3_α", "f1_E", "rec_E", "prec_E", "f1_H", "macro_f1_3"]


def per_step(tag: str) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    f1 = BENCH / f"{tag}_f1.csv"
    if not f1.exists():
        return out
    rows: dict[int, list[dict]] = {}
    for r in csv.DictReader(open(f1)):
        rows.setdefault(int(r["step"]), []).append(r)
    for s, rs in rows.items():
        d = {}
        for k in METRICS:
            v = [float(r[k]) for r in rs if r.get(k) not in (None, "", "nan") and not math.isnan(float(r[k]))]
            if v:
                d[k] = st.mean(v)
        d["n"] = len(rs)
        out[s] = d
    struct = BENCH / f"{tag}_struct.csv"
    if struct.exists():
        for r in csv.DictReader(open(struct)):
            if r["metric"] == "TM" and int(r["step"]) in out:
                out[int(r["step"])]["TM"] = float(r["mean"])
    return {s: d for s, d in out.items() if "TM" in d}


def main() -> None:
    data = {name: per_step(tag) for name, tag in RUNS.items()}
    for name, d in data.items():
        log.info("%s: %d steps with TM (%s)", name, len(d), sorted(d)[:3] + ["..."] if d else "none")

    ref = data["r17 (no TriAttn + MHR)"]
    usable = {s: d for s, d in ref.items() if d.get("TM", 0) > 0.80}
    if not usable:
        tm_max = max((d["TM"] for d in ref.values()), default=0.0)
        log.warning(
            "\nNOT YET READABLE. r17's best TM is %.3f; the matched-quality comparison needs it out of the\n"
            "transition (TM > 0.80), because TM moves 0.46 -> 0.73 in 500 steps and any interpolation inside\n"
            "that region is dominated by where the transition happened to land, not by capability.",
            tm_max,
        )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, metric in zip(axes, ["f1_3_β", "f1_3_α", "rec_E"], strict=True):
        for name, d in data.items():
            steps = sorted(d)
            xs = [d[s]["TM"] for s in steps if metric in d[s]]
            ys = [d[s][metric] for s in steps if metric in d[s]]
            ax.plot(xs, ys, "o-", ms=3.5, lw=1.3, label=name)
        ax.set_xlabel("TM (whole structure)")
        ax.set_ylabel(metric)
        ax.grid(alpha=0.3)
        ax.set_title(f"{metric} vs overall quality")
    axes[0].legend(fontsize=8)
    fig.suptitle("does the capability depend on triangle attention, or just track overall quality?", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "ablation_map.png", dpi=120)

    rows = []
    for name, d in data.items():
        for s in sorted(d):
            rows.append(dict(run=name, step=s, **{k: round(v, 4) for k, v in d[s].items()}))
    with open(OUT / "ablation_map.csv", "w", newline="") as fh:
        fields = sorted({k for r in rows for k in r})
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    log.info("wrote %s", OUT / "ablation_map.csv")

    common = sorted(set(data["r16 (with TriAttn)"]) & set(ref))
    informative = [s for s in common if ref[s].get("TM", 0) > 0.3]
    if informative:
        log.info("\nmatched-STEP deficits (relative), informative steps only")
        log.info("%6s %6s | %s", "step", "n", "  ".join(f"{m:>10}" for m in ["TM"] + METRICS))
        for s in informative:
            a, b = data["r16 (with TriAttn)"][s], ref[s]
            cells = []
            for m in ["TM"] + METRICS:
                if m in a and m in b and a[m] > 1e-6:
                    cells.append(f"{100 * (b[m] - a[m]) / a[m]:+9.1f}%")
                else:
                    cells.append(f"{'—':>10}")
            log.info("%6d %6d | %s", s, b.get("n", 0), "  ".join(cells))
        log.info(
            "\nA capability is *disproportionately* hurt when its relative deficit exceeds TM's.\n"
            "That is suggestive, not conclusive: at matched step r17 is simply earlier in the same\n"
            "trajectory, and metrics that saturate late will always look worse there."
        )


if __name__ == "__main__":
    main()
