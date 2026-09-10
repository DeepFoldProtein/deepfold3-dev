"""Per-category FoldBench-small comparison: AF3 vs a DF3 checkpoint vs K-Fold.

Every number is re-aggregated from the benched raw CSVs
(``bench/runs/<series>/eval/<cat>/step_<N>/raw/<cat>_ost.csv``) over the targets
**every** compared series scored, so a row's columns are on one target set. Three
per-target aggregates, as in eval_ost.py: ``mean`` over samples, ``rank-1`` (the
``ranking_score`` top sample, meaned over ties — eval_ost.conf_pick), ``oracle``
(best sample).

AF3 is the measured fold through this bench, not the paper: ``af3_ref_tmpl``
(re-folded with templates) where it exists, else ``af3_ref`` (template-free,
marked †). ``af3_ref`` folded the nucleic monomers before the water fix, so AF3
has no measured row there (summarize.REF_SERIES_EXCLUDE_CATEGORIES); the paper's
full-set number is printed in its own column and never plotted.

Protein–DNA / protein–RNA DockQ covers only the handful of interfaces OST's DockQ
returns for nucleic partners, identically for every series; whole-model and
interface lDDT cover every target and are the rows to read there.

Usage::

    python bench/scripts/compare_series.py                         # r19s1 @120k vs kfold
    python bench/scripts/compare_series.py --df3-series df3v5r19s1 --df3-step 120000 \
        --out-dir export/kfold-foldbench
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from eval_ost import DOCKQ_SUCCESS, INTERFACE_CATEGORIES, conf_pick  # sibling script
from summarize import AF3_INTERFACE_LDDT, AF3_MONOMER, AF3_SUCCESS_RATE, REF_SERIES_EXCLUDE_CATEGORIES

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "bench" / "runs"

CATEGORIES = (
    "monomer_protein",
    "monomer_rna",
    "monomer_dna",
    "interface_protein_protein",
    "interface_antibody_antigen",
    "interface_protein_peptide",
    "interface_protein_dna",
    "interface_protein_rna",
)
SHORT = {
    "monomer_protein": "protein",
    "monomer_rna": "RNA",
    "monomer_dna": "DNA",
    "interface_protein_protein": "protein–protein",
    "interface_antibody_antigen": "antibody–antigen",
    "interface_protein_peptide": "protein–peptide",
    "interface_protein_dna": "protein–DNA",
    "interface_protein_rna": "protein–RNA",
}
NUMERIC = ("dockq", "lddt", "bb_lddt", "ilddt", "tm_score", "oligo_gdtts", "ranking_score")
# OST's DockQ scores only a few nucleic-partner interfaces (4 protein-DNA, 0 protein-RNA at
# r19 @120k); below this many common interfaces the DockQ rows are dropped from the table
# and the figure, and iLDDT / lDDT carry the category.
MIN_DOCKQ_N = 10

# Palette: the categorical reference slots 1-3 (validated all-pairs). Ours stays orange
# and the external AF3 blue, matching export/foldbench-compare.
COLORS = {"af3": "#2a78d6", "df3": "#eb6834", "kfold": "#1baf7a"}
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load_raw(series: str, step: int, cat: str) -> list[dict] | None:
    p = RUNS / series / "eval" / cat / f"step_{step}" / "raw" / f"{cat}_ost.csv"
    if not p.exists():
        return None
    rows = []
    for r in csv.DictReader(open(p)):
        rows.append({**r, **{k: _f(r.get(k)) for k in NUMERIC}})
    return rows


def af3_series(cat: str) -> tuple[str, bool] | None:
    """(series, template_free) for AF3 on this category, or None when not comparable."""
    if (RUNS / "af3_ref_tmpl" / "eval" / cat / "step_0" / "raw").is_dir():
        return "af3_ref_tmpl", False
    if cat in REF_SERIES_EXCLUDE_CATEGORIES.get("af3_ref", set()):
        return None
    if (RUNS / "af3_ref" / "eval" / cat / "step_0" / "raw").is_dir():
        return "af3_ref", True
    return None


def per_target(rows: list[dict], key_fn, metric: str) -> dict:
    """key -> (mean, rank1, oracle, n_samples) over rows with a numeric ``metric``."""
    by = defaultdict(dict)
    for r in rows:
        if r[metric] is None:
            continue
        by[key_fn(r)][(r["seed"], r["sample"])] = r  # dedupe whole-model rows repeated per interface
    out = {}
    for k, samples in by.items():
        rs = list(samples.values())
        vals = [r[metric] for r in rs]
        out[k] = (sum(vals) / len(vals), conf_pick(rs, metric), max(vals), len(rs))
    return out


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def compare(cat: str, models: dict[str, list[dict]]) -> list[dict]:
    """Long-form rows: one per (metric, stat, model), on the common target set."""
    iface = cat in INTERFACE_CATEGORIES
    specs = (
        [
            ("dockq", lambda r: (r["pdb_id"], r["ic1"], r["ic2"])),
            ("ilddt", lambda r: r["pdb_id"]),
            ("lddt", lambda r: r["pdb_id"]),
        ]
        if iface
        else [
            ("lddt", lambda r: r["pdb_id"]),
            ("tm_score", lambda r: r["pdb_id"]),
            ("oligo_gdtts", lambda r: r["pdb_id"]),
        ]
    )
    out = []
    for metric, key_fn in specs:
        agg = {m: per_target(rows, key_fn, metric) for m, rows in models.items()}
        common = set.intersection(*(set(a) for a in agg.values())) if agg else set()
        if not common:
            continue
        for m, a in agg.items():
            vals = [a[k] for k in common]
            budget = sorted(v[3] for v in vals)[len(vals) // 2]
            stats = {
                "mean": _mean(v[0] for v in vals),
                "rank1": _mean(v[1] for v in vals),
                "oracle": _mean(v[2] for v in vals),
            }
            if metric == "dockq":
                for s, idx in (("mean", 0), ("rank1", 1), ("oracle", 2)):
                    ok = [v[idx] >= DOCKQ_SUCCESS for v in vals if v[idx] is not None]
                    stats[f"success_{s}"] = 100 * sum(ok) / len(ok) if ok else None
            for stat, value in stats.items():
                out.append(
                    {
                        "category": cat,
                        "metric": metric,
                        "stat": stat,
                        "model": m,
                        "value": value,
                        "n": len(common),
                        "samples": budget,
                    }
                )
    return out


def paper_value(cat: str, metric: str, stat: str) -> float | None:
    """FoldBench Supp. Table 3 AF3 (full set, mean over samples) where one exists."""
    if cat in AF3_MONOMER and stat == "mean":
        return AF3_MONOMER[cat].get({"lddt": "lddt", "tm_score": "tm", "oligo_gdtts": "gdt"}[metric])
    if metric == "dockq" and stat == "success_mean" and cat in AF3_SUCCESS_RATE:
        return 100 * AF3_SUCCESS_RATE[cat]
    if metric == "lddt" and stat == "mean" and cat in AF3_INTERFACE_LDDT:
        return AF3_INTERFACE_LDDT[cat]
    return None


# (metric, stat, label, is_percent) rows of the markdown table
TABLE_MONOMER = [
    ("lddt", "rank1", "lDDT rank-1", False),
    ("lddt", "mean", "lDDT mean", False),
    ("lddt", "oracle", "lDDT oracle", False),
    ("tm_score", "rank1", "TM-score rank-1", False),
    ("oligo_gdtts", "rank1", "GDT-TS rank-1", False),
]
TABLE_INTERFACE = [
    ("dockq", "success_rank1", "DockQ≥0.23 rank-1 %", True),
    ("dockq", "success_mean", "DockQ≥0.23 mean %", True),
    ("dockq", "success_oracle", "DockQ≥0.23 oracle %", True),
    ("dockq", "rank1", "DockQ rank-1", False),
    ("ilddt", "rank1", "iLDDT rank-1", False),
    ("lddt", "rank1", "lDDT rank-1", False),
]


def write_markdown(long_rows, labels, af3_src, out: Path) -> None:
    idx = {(r["category"], r["metric"], r["stat"], r["model"]): r for r in long_rows}
    models = list(labels)
    lines = [
        "| category | metric | n | " + " | ".join(labels[m] for m in models) + " | AF3 paper (full set) |",
        "|---|---|---:|" + "---:|" * len(models) + "---:|",
    ]
    for cat in CATEGORIES:
        table = TABLE_INTERFACE if cat in INTERFACE_CATEGORIES else TABLE_MONOMER
        for metric, stat, label, pct in table:
            cells, n = [], None
            vals = {m: idx.get((cat, metric, stat, m)) for m in models}
            present = [v["value"] for v in vals.values() if v and v["value"] is not None]
            best = max(present) if len(present) > 1 else None
            for m in models:
                r = vals[m]
                if not r or r["value"] is None:
                    cells.append("—")
                    continue
                n = r["n"]
                s = f"{r['value']:.1f}" if pct else f"{r['value']:.3f}"
                cells.append(f"**{s}**" if best is not None and r["value"] == best else s)
            if n is None or (metric == "dockq" and n < MIN_DOCKQ_N):
                continue
            pv = paper_value(cat, metric, stat)
            paper = "—" if pv is None else (f"{pv:.1f}" if pct else f"{pv:.2f}")
            src = af3_src.get(cat)
            tag = " †" if src and src[1] else ""
            lines.append(f"| {SHORT[cat]}{tag} | {label} | {n} | " + " | ".join(cells) + f" | {paper} |")
    budgets = defaultdict(dict)
    for r in long_rows:
        budgets[r["category"]][r["model"]] = r["samples"]
    lines += [
        "",
        "Samples per target (median, on the common set): "
        + "; ".join(
            f"{SHORT[c]} " + "/".join(f"{labels[m]} {budgets[c].get(m, '—')}" for m in models)
            for c in CATEGORIES
            if c in budgets
        ),
        "",
        "† AF3 row is `af3_ref` (template-free); unmarked interface rows use `af3_ref_tmpl`. "
        "Bold = best measured value in the row. Paper column = FoldBench Supp. Table 3, full set, not like-for-like.",
    ]
    out.write_text("\n".join(lines) + "\n")


# --- split by AtlasFold confidence ------------------------------------------------------
# K-Fold's only information source is the AtlasFold apo ensemble, so the per-category
# averages mix targets where AtlasFold already folded the chains with targets where it did
# not. Binning by AtlasFold's own pLDDT separates the two. A target's value is its
# least-confident protein chain: one unfolded partner is enough to lose an interface.
APO_BINS = ((0, 50), (50, 70), (70, 85), (85, 101))
APO_METRIC = {  # category -> (metric, is_success_rate), rank-1 on the common set
    "monomer_protein": ("lddt", False),
    "interface_protein_protein": ("dockq", True),
    "interface_antibody_antigen": ("dockq", True),
    "interface_protein_peptide": ("dockq", True),
}


def apo_plddt() -> dict[str, dict[str, float]]:
    """category -> pdb -> mean AtlasFold pLDDT of the target's least-confident protein chain.

    AtlasFold's ``ranking_score`` is its mean pLDDT; this averages the rank-1 structure of
    each apo seed. Empty when no K-Fold apo ensemble is on disk.
    """
    import json

    from kfold_bench import APO_SEEDS, WORK, load_units, protein_key

    units, index = load_units()
    cache: dict[str, float | None] = {}

    def chain_conf(seq: str) -> float | None:
        if seq not in cache:
            root = WORK / "prep" / "apo" / protein_key(seq) / "sequences-0"
            vals = [
                json.loads(p.read_text())["ranking_score"]
                for s in APO_SEEDS
                if (p := root / f"seed-{s}" / "rank_1.json").exists()
            ]
            cache[seq] = sum(vals) / len(vals) if vals else None
        return cache[seq]

    out: dict[str, dict[str, float]] = {}
    for cat, targets in index.items():
        out[cat] = {}
        for pdb, name in targets.items():
            confs = [c for kind, _, seq in units[name]["entities"] if kind == "protein" and (c := chain_conf(seq))]
            if confs:
                out[cat][pdb] = min(confs)
    return out


def compare_by_apo(cat: str, models: dict[str, list[dict]], conf: dict[str, float]) -> list[dict]:
    """Rank-1 headline metric per (AtlasFold pLDDT bin, model) on the common target set."""
    metric, success = APO_METRIC[cat]
    key_fn = (lambda r: (r["pdb_id"], r["ic1"], r["ic2"])) if metric == "dockq" else (lambda r: r["pdb_id"])
    agg = {m: per_target(rows, key_fn, metric) for m, rows in models.items()}
    common = set.intersection(*(set(a) for a in agg.values()))
    out = []
    for lo, hi in APO_BINS:
        keys = [k for k in common if lo <= conf.get(k[0] if isinstance(k, tuple) else k, -1) < hi]
        if not keys:
            continue
        for m, a in agg.items():
            vals = [a[k][1] for k in keys if a[k][1] is not None]
            if success:
                value = 100 * sum(v >= DOCKQ_SUCCESS for v in vals) / len(vals) if vals else None
            else:
                value = _mean(vals)
            out.append({"category": cat, "bin": f"{lo}–{min(hi, 100)}", "model": m, "value": value, "n": len(keys)})
    return out


def write_apo_markdown(rows: list[dict], labels: dict[str, str], out: Path) -> None:
    models = list(labels)
    idx = {(r["category"], r["bin"], r["model"]): r for r in rows}
    lines = [
        "| category | metric | AtlasFold pLDDT | n | " + " | ".join(labels[m] for m in models) + " |",
        "|---|---|---|---:|" + "---:|" * len(models),
    ]
    for cat, (_metric, success) in APO_METRIC.items():
        label = "DockQ≥0.23 rank-1 %" if success else "lDDT rank-1"
        for lo, hi in APO_BINS:
            b = f"{lo}–{min(hi, 100)}"
            vals = {m: idx.get((cat, b, m)) for m in models}
            n = next((v["n"] for v in vals.values() if v), None)
            if n is None:
                continue
            cells = [
                "—" if not v or v["value"] is None else (f"{v['value']:.1f}" if success else f"{v['value']:.3f}")
                for v in vals.values()
            ]
            lines.append(f"| {SHORT[cat]} | {label} | {b} | {n} | " + " | ".join(cells) + " |")
    out.write_text("\n".join(lines) + "\n")


def plot(long_rows, labels, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    idx = {(r["category"], r["metric"], r["stat"], r["model"]): r for r in long_rows}
    panels = [
        ("Monomer lDDT (rank-1)", [c for c in CATEGORIES if c not in INTERFACE_CATEGORIES], "lddt", "rank1", 1.0),
        (
            "Interface success, DockQ ≥ 0.23 (rank-1)",
            [c for c in CATEGORIES if c in INTERFACE_CATEGORIES],
            "dockq",
            "success_rank1",
            100.0,
        ),
        ("Interface lDDT, iLDDT (rank-1)", [c for c in CATEGORIES if c in INTERFACE_CATEGORIES], "ilddt", "rank1", 1.0),
    ]
    heights = [len(p[1]) for p in panels]
    fig, axes = plt.subplots(
        len(panels), 1, figsize=(8.2, 1.2 + 0.55 * sum(heights)), gridspec_kw={"height_ratios": heights}
    )
    fig.patch.set_facecolor(SURFACE)
    offsets = {m: (i - (len(labels) - 1) / 2) * 0.22 for i, m in enumerate(labels)}
    for ax, (title, cats, metric, stat, scale) in zip(axes, panels, strict=True):
        ax.set_facecolor(SURFACE)
        ticks = []
        for y, cat in enumerate(cats):
            rows = {m: idx.get((cat, metric, stat, m)) for m in labels}
            vals = {m: r["value"] / scale for m, r in rows.items() if r and r["value"] is not None}
            n = next((r["n"] for r in rows.values() if r), 0)
            if metric == "dockq" and n < MIN_DOCKQ_N:
                vals = {}  # a success rate over a handful of interfaces is not a comparison
            ticks.append(f"{SHORT[cat]}  (n={n})" if vals else f"{SHORT[cat]}  (n={n}, not shown)")
            for m, v in vals.items():
                ax.scatter(v, y + offsets[m], s=70, color=COLORS[m], edgecolors=SURFACE, linewidths=2, zorder=3)
        ax.set_yticks(range(len(cats)), ticks, color=INK2, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_ylim(len(cats) - 0.5, -0.5)
        ax.set_title(title, loc="left", color=INK, fontsize=10.5, pad=6)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", colors=MUTED, labelsize=8.5, length=0)
        ax.tick_params(axis="y", length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(AXIS)
        if scale == 100.0:
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    handles = [
        plt.Line2D(
            [], [], marker="o", ls="", markersize=8, markerfacecolor=COLORS[m], markeredgecolor=SURFACE, label=labels[m]
        )
        for m in labels
    ]
    fig.legend(
        handles=handles,
        loc="upper left",
        ncol=len(labels),
        frameon=False,
        fontsize=9.5,
        labelcolor=INK,
        bbox_to_anchor=(0.01, 1.0),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965), h_pad=1.6)
    fig.savefig(out, dpi=180, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--df3-series", default="df3v5r19s1")
    ap.add_argument("--df3-step", type=int, default=120000)
    ap.add_argument("--df3-label", default="DF3 r19 @120k")
    ap.add_argument("--out-dir", type=Path, default=REPO / "export" / "kfold-foldbench")
    ap.add_argument(
        "--after",
        default=None,
        metavar="YYYY-MM-DD",
        help="Keep only targets released after this date (training-cutoff control), from "
        "export/foldbench-compare/data/foldbench_target_dates.csv.",
    )
    ap.add_argument(
        "--exclude-ids",
        type=Path,
        default=None,
        help="File of PDB ids to drop (first token per line; chain / assembly suffixes ignored), e.g. "
        "references/kfold/assets/splits/v261230/validation_ids.txt as a selection-bias control.",
    )
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    keep = None
    if args.after:
        dates = REPO / "export" / "foldbench-compare" / "data" / "foldbench_target_dates.csv"
        keep = {r["pdb_id"] for r in csv.DictReader(open(dates)) if (r.get("released") or "") > args.after}
        print(f"[compare] --after {args.after}: {len(keep)} FoldBench targets released later")
    drop: set[str] = set()
    if args.exclude_ids:
        drop = {ln.split()[0].split("_")[0].lower() for ln in open(args.exclude_ids) if ln.strip()}
        print(f"[compare] --exclude-ids {args.exclude_ids}: {len(drop)} PDB ids")

    labels = {"af3": "AF3", "df3": args.df3_label, "kfold": "K-Fold"}
    long_rows, apo_rows, af3_src = [], [], {}
    conf = apo_plddt() if (RUNS / "kfold" / "work" / "prep" / "apo").is_dir() else {}
    for cat in CATEGORIES:
        models = {}
        src = af3_series(cat)
        af3_src[cat] = src
        if src and (rows := load_raw(src[0], 0, cat)):
            models["af3"] = rows
        if rows := load_raw(args.df3_series, args.df3_step, cat):
            models["df3"] = rows
        if rows := load_raw("kfold", 0, cat):
            models["kfold"] = rows
        if keep is not None or drop:

            def selected(pdb_id: str) -> bool:
                return (keep is None or pdb_id in keep) and pdb_id.split("-")[0].lower() not in drop

            models = {m: kept for m, rows in models.items() if (kept := [r for r in rows if selected(r["pdb_id"])])}
        if len(models) < 2:
            print(f"[compare] {cat}: only {list(models)} benched — skipped")
            continue
        # N-match: oracle is monotone in sample count and rank-1 weakly so. AF3 folded seeds
        # 42-45 in every category while df3 / K-Fold fold 42-43 outside protein-protein and
        # antibody-antigen, so every model is cut to the seeds all of them have.
        seeds = set.intersection(*({r["seed"] for r in rows} for rows in models.values()))
        models = {m: [r for r in rows if r["seed"] in seeds] for m, rows in models.items()}
        print(f"[compare] {cat}: seeds {sorted(seeds)}")
        long_rows += compare(cat, models)
        if cat in APO_METRIC and "kfold" in models and conf.get(cat):
            apo_rows += compare_by_apo(cat, models, conf[cat])
        print(f"[compare] {cat}: models={list(models)} af3={src[0] if src else 'none'}")

    csv_path = args.out_dir / "compare_by_category.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["category", "metric", "stat", "model", "value", "n", "samples"])
        w.writeheader()
        w.writerows(long_rows)
    write_markdown(long_rows, labels, af3_src, args.out_dir / "compare_by_category.md")
    plot(long_rows, labels, args.out_dir / "compare_by_category.png")
    print(f"[compare] -> {csv_path.parent}/compare_by_category.{{csv,md,png}}")
    if apo_rows:
        with open(args.out_dir / "compare_by_apo_plddt.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["category", "bin", "model", "value", "n"])
            w.writeheader()
            w.writerows(apo_rows)
        write_apo_markdown(apo_rows, labels, args.out_dir / "compare_by_apo_plddt.md")
        print(f"[compare] -> {csv_path.parent}/compare_by_apo_plddt.{{csv,md}}")


if __name__ == "__main__":
    main()
