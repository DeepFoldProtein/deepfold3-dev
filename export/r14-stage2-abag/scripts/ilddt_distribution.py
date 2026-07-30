"""Inter-chain lDDT distribution over training, AbAg vs ppi.

Two questions this answers, neither visible in a mean:

  * does the LOWER end of the distribution move (are the worst interfaces
    improving)? -> quantile fan, top row
  * what shape is the distribution? -> histogram, bottom row; ppi is bimodal
    (a "no interface at all" mode near 0 and a solved mode near 0.85), so its
    flat mean is two fixed populations, not a converged middle

The per-interface value is the mean iLDDT over that interface's 20 samples
(4 seeds x 5), matching how eval_ost aggregates DockQ.

Writes ilddt_quantiles.csv + ilddt_distribution.png next to the report.
Run from the repo root.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EV = Path("bench/runs/df3v4r14/eval")
OUT = Path("export/r14-stage2-abag")
STEPS = list(range(20000, 34000, 1000))
# Last checkpoint trained with stochastic atomization; 32k+ are the restarted
# atomization-off run (see report).
ATOMIZE_LAST = 31000
HIST_STEP = 33000

# Okabe-Ito blue / vermillion: a published colourblind-safe pair (the skill's
# validator needs node, which this box does not have).
CATS = [
    ("interface_antibody_antigen", "antibody-antigen", "#0072B2"),
    ("interface_protein_protein", "protein-protein", "#D55E00"),
]
INK, MUTED = "#1a1a1a", "#6b6b6b"


def load(cat: str, step: int) -> dict[tuple, float]:
    f = EV / cat / f"step_{step}" / "raw" / f"{cat}_ost.csv"
    if not f.exists():
        return {}
    by: dict[tuple, list[float]] = defaultdict(list)
    with open(f) as fh:
        for r in csv.DictReader(fh):
            try:
                v = float(r["ilddt"])
            except (TypeError, ValueError):
                continue
            by[(r["pdb_id"], r["ic1"], r["ic2"])].append(v)
    return {k: sum(v) / len(v) for k, v in by.items()}


def quantile(vals: list[float], p: float) -> float:
    s = sorted(vals)
    i = p * (len(s) - 1)
    lo = int(i)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (i - lo)


def main() -> None:
    data = {cat: {s: load(cat, s) for s in STEPS} for cat, _, _ in CATS}
    data = {c: {s: v for s, v in d.items() if v} for c, d in data.items()}
    # Same interfaces at every step, so the quantile trajectory is not a
    # changing-population artefact.
    keys = {c: sorted(set.intersection(*(set(v) for v in d.values()))) for c, d in data.items()}

    rows = []
    for cat, label, _ in CATS:
        for s in sorted(data[cat]):
            v = [data[cat][s][k] for k in keys[cat]]
            rows.append({
                "category": cat, "step": s, "n": len(v),
                "p10": round(quantile(v, 0.10), 4), "q1": round(quantile(v, 0.25), 4),
                "median": round(quantile(v, 0.50), 4), "q3": round(quantile(v, 0.75), 4),
                "p90": round(quantile(v, 0.90), 4), "mean": round(sum(v) / len(v), 4),
                "atomize": "on" if s <= ATOMIZE_LAST else "off",
            })
    with open(OUT / "ilddt_quantiles.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), gridspec_kw={"height_ratios": [1.25, 1]})
    for col, (cat, label, color) in enumerate(CATS):
        steps = sorted(data[cat])
        vals = {s: [data[cat][s][k] for k in keys[cat]] for s in steps}
        xs = [s / 1000 for s in steps]
        p10 = [quantile(vals[s], 0.10) for s in steps]
        q1 = [quantile(vals[s], 0.25) for s in steps]
        med = [quantile(vals[s], 0.50) for s in steps]
        q3 = [quantile(vals[s], 0.75) for s in steps]
        p90 = [quantile(vals[s], 0.90) for s in steps]

        ax = axes[0][col]
        ax.fill_between(xs, p10, p90, color=color, alpha=0.14, linewidth=0, label="p10–p90")
        ax.fill_between(xs, q1, q3, color=color, alpha=0.32, linewidth=0, label="IQR")
        ax.plot(xs, med, color=color, linewidth=2, label="median")
        ax.axvline((ATOMIZE_LAST + 500) / 1000, color=MUTED, linestyle=(0, (4, 3)), linewidth=1)
        ax.text((ATOMIZE_LAST + 700) / 1000, 0.035, "atomization off", fontsize=8, color=MUTED)
        ax.set_title(f"{label}  (n={len(keys[cat])} interfaces)", fontsize=11, color=INK, loc="left")
        ax.set_xlabel("training step (thousands)", fontsize=9, color=MUTED)
        ax.set_ylim(0, 1)
        if col == 0:
            ax.set_ylabel("inter-chain lDDT", fontsize=9, color=MUTED)
            ax.legend(loc="upper left", fontsize=8, frameon=False, ncol=3, columnspacing=1.2)

        ax = axes[1][col]
        ax.hist(vals[HIST_STEP], bins=[i / 20 for i in range(21)], color=color, alpha=0.75,
                edgecolor="white", linewidth=0.8)
        ax.set_title(f"distribution at step {HIST_STEP // 1000}k", fontsize=10, color=INK, loc="left")
        ax.set_xlabel("inter-chain lDDT", fontsize=9, color=MUTED)
        ax.set_xlim(0, 1)
        if col == 0:
            ax.set_ylabel("interfaces", fontsize=9, color=MUTED)

    for row in axes:
        for ax in row:
            ax.grid(axis="y", color="#d9d9d9", linewidth=0.6, alpha=0.6)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color("#cccccc")
            ax.tick_params(colors=MUTED, labelsize=8)

    fig.suptitle("r14: inter-chain lDDT is flat in both categories; ppi is bimodal",
                 fontsize=12, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "ilddt_distribution.png", dpi=160)
    print(f"wrote {OUT/'ilddt_quantiles.csv'} ({len(rows)} rows) and {OUT/'ilddt_distribution.png'}")


if __name__ == "__main__":
    main()
