"""Per-target diff of two checkpoints in one bench series.

Answers, for a pair of steps (e.g. the last stage-1 checkpoint vs the first
stage-2 one): which targets started passing, which started failing, and whether
a flip is a real change or the success threshold's knife edge.

Reads the per-sample rows ``eval_ost.py`` writes to
``bench/runs/{series}/eval/{category}/step_{N}/raw/{category}_ost.csv`` and
re-aggregates them the same way (interfaces keyed by ``(pdb_id, ic1, ic2)``;
mean / oracle = best-of-N / conf = ranking_score top-1).

Interface categories are compared on DockQ against the FoldBench success cut
(0.23); monomer categories on lDDT / TM, where ``--cut`` is only a reporting
band. Every flip is printed with its margin to the cut and its per-sample pass
count, because a mean crossing the cut by 0.002 with 15/20 -> 9/20 samples
passing is a different event from 20/20 -> 1/20. A paired bootstrap over
targets gives the success-rate delta a CI, since single-checkpoint bench
differences on a few hundred targets are routinely noise.

Usage::

    uv run python bench/scripts/step_compare.py --series df3v4r14 \\
        --category interface_protein_protein --steps 25000 26000
"""

from __future__ import annotations

import argparse
import csv
import random
import statistics as st
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DOCKQ_SUCCESS = 0.23  # eval_ost.DOCKQ_SUCCESS (FoldBench)
KNIFE_HALFWIDTH = 0.05  # +-0.05 around the cut = "could flip on resampling"


def _num(row: dict, key: str) -> float | None:
    try:
        return float(row[key])
    except (TypeError, ValueError, KeyError):
        return None


def load_step(eval_dir: Path, category: str, step: int, metric: str) -> dict[tuple, dict]:
    """Aggregate one step's per-sample rows into per-target stats.

    Targets are keyed by interface for interface categories and by ``pdb_id``
    for monomers — the same grouping ``eval_ost.py`` aggregates over.
    """
    path = eval_dir / category / f"step_{step}" / "raw" / f"{category}_ost.csv"
    if not path.exists():
        raise SystemExit(f"missing: {path}")
    by: dict[tuple, list[dict]] = defaultdict(list)
    with open(path) as f:
        for row in csv.DictReader(f):
            key = (row["pdb_id"], row["ic1"], row["ic2"]) if "ic1" in row else (row["pdb_id"],)
            by[key].append(row)

    out: dict[tuple, dict] = {}
    for key, rows in by.items():
        vals = [v for v in (_num(r, metric) for r in rows) if v is not None]
        if not vals:
            continue
        out[key] = {
            "mean": sum(vals) / len(vals),
            "oracle": max(vals),
            "conf": _conf_pick(rows, metric),
            "sd": st.pstdev(vals),
            "n": len(vals),
            "vals": vals,
            "ilddt": _mean_of(rows, "ilddt"),
        }
    return out


def _conf_pick(rows: list[dict], metric: str) -> float | None:
    """``metric`` of the highest-``ranking_score`` sample — ``eval_ost.conf_pick``.

    Kept strictly first-wins on ties: ``ranking_score`` is rounded in the CSV, so
    ties do occur, and breaking them by the metric would quietly turn the
    confidence pick into a mini-oracle and disagree with ``summary.csv``.
    """
    best = best_score = None
    for row in rows:
        val, score = _num(row, metric), _num(row, "ranking_score")
        if val is None or score is None:
            continue
        if best_score is None or score > best_score:
            best_score, best = score, val
    return best


def _mean_of(rows: list[dict], key: str) -> float | None:
    vals = [v for v in (_num(r, key) for r in rows) if v is not None]
    return sum(vals) / len(vals) if vals else None


def bootstrap_delta(a: dict, b: dict, keys: list, agg: str, cut: float, trials: int, seed: int) -> tuple:
    """Paired bootstrap CI (percentile) for the success-rate delta in points."""
    diffs = [int(b[k][agg] >= cut) - int(a[k][agg] >= cut) for k in keys]
    rng = random.Random(seed)
    draws = []
    for _ in range(trials):
        picked = [diffs[rng.randrange(len(diffs))] for _ in diffs]
        draws.append(100 * sum(picked) / len(picked))
    draws.sort()
    return 100 * sum(diffs) / len(diffs), draws[int(0.025 * trials)], draws[int(0.975 * trials)]


def _name(key: tuple) -> str:
    pdb = key[0].replace("-assembly1", "")
    return f"{pdb} {key[1]}-{key[2]}" if len(key) == 3 else pdb


def compare(a: dict, b: dict, keys: list, agg: str, cut: float, metric: str, trials: int, seed: int) -> None:
    obs, lo, hi = bootstrap_delta(a, b, keys, agg, cut, trials, seed)
    n_a = sum(1 for k in keys if a[k][agg] >= cut)
    n_b = sum(1 for k in keys if b[k][agg] >= cut)
    print(f"\n=== {agg}: {metric} >= {cut} on {n_a}/{len(keys)} -> {n_b}/{len(keys)} targets ({obs:+.2f} pt)")
    print(f"    paired bootstrap 95% CI [{lo:+.2f}, {hi:+.2f}] pt" + ("  (includes 0)" if lo <= 0 <= hi else ""))

    for label, flipped in (
        ("NEW PASS", [k for k in keys if a[k][agg] < cut <= b[k][agg]]),
        ("NEW FAIL", [k for k in keys if b[k][agg] < cut <= a[k][agg]]),
    ):
        print(f"  {label} ({len(flipped)}):")
        for k in sorted(flipped, key=lambda k: abs(b[k][agg] - a[k][agg]), reverse=True):
            x, y = a[k][agg], b[k][agg]
            margin = abs(y - cut)
            # Per-sample pass counts separate a threshold artifact (15/20 -> 9/20,
            # mean grazing the cut) from a real loss (20/20 -> 1/20).
            pa = sum(1 for v in a[k]["vals"] if v >= cut)
            pb = sum(1 for v in b[k]["vals"] if v >= cut)
            print(
                f"    {_name(k):26s} {x:.3f} -> {y:.3f} ({y - x:+.3f})  margin to cut {margin:.3f}"
                f"  samples {pa}/{a[k]['n']} -> {pb}/{b[k]['n']}"
                f"{'  <- knife edge' if margin < KNIFE_HALFWIDTH else ''}"
            )

    d = [b[k][agg] - a[k][agg] for k in keys]
    band = sum(1 for k in keys if abs(b[k][agg] - cut) < KNIFE_HALFWIDTH)
    print(
        f"  delta: mean {st.mean(d):+.4f}  median {st.median(d):+.4f}  |d|>0.05 on {sum(1 for x in d if abs(x) > 0.05)}"
    )
    print(f"  within +-{KNIFE_HALFWIDTH} of the cut at the later step: {band}/{len(keys)} targets")


def movers(a: dict, b: dict, keys: list, agg: str, top: int) -> None:
    print(f"\n=== biggest {agg} movers (sample spread + interface lDDT tell a flip from a collapse)")
    print(f"    {'target':26s} {'before':>7s} {'after':>7s} {'sd_b':>6s} {'sd_a':>6s} {'ilddt_b':>8s} {'ilddt_a':>8s}")
    for k in sorted(keys, key=lambda k: abs(b[k][agg] - a[k][agg]), reverse=True)[:top]:
        il_a, il_b = a[k]["ilddt"], b[k]["ilddt"]
        print(
            f"    {_name(k):26s} {a[k][agg]:7.3f} {b[k][agg]:7.3f} {a[k]['sd']:6.3f} {b[k]['sd']:6.3f} "
            f"{il_a if il_a is not None else float('nan'):8.3f} {il_b if il_b is not None else float('nan'):8.3f}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--series", default="df3v4r14")
    ap.add_argument("--category", default="interface_protein_protein")
    ap.add_argument("--steps", nargs=2, type=int, required=True, metavar=("BEFORE", "AFTER"))
    ap.add_argument("--eval-root", type=Path, default=REPO / "bench" / "runs")
    ap.add_argument("--metric", default=None, help="default: dockq for interfaces, lddt for monomers")
    ap.add_argument("--cut", type=float, default=None, help=f"default: {DOCKQ_SUCCESS} for dockq, 0.5 otherwise")
    ap.add_argument("--top", type=int, default=15, help="how many movers to list")
    ap.add_argument("--trials", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    is_interface = args.category.startswith("interface_")
    metric = args.metric or ("dockq" if is_interface else "lddt")
    cut = args.cut if args.cut is not None else (DOCKQ_SUCCESS if metric == "dockq" else 0.5)

    eval_dir = args.eval_root / args.series / "eval"
    before, after = (load_step(eval_dir, args.category, s, metric) for s in args.steps)
    keys = sorted(set(before) & set(after))
    if not keys:
        raise SystemExit("no targets in common between the two steps")

    print(f"{args.series} {args.category}: step {args.steps[0]} -> {args.steps[1]}, metric={metric}, cut={cut}")
    print(f"targets {len(before)} / {len(after)}, compared on {len(keys)} shared")
    print(f"samples per target: {sorted({before[k]['n'] for k in keys})} -> {sorted({after[k]['n'] for k in keys})}")

    for agg in ("oracle", "mean", "conf"):
        if any(before[k][agg] is None or after[k][agg] is None for k in keys):
            print(f"\n=== {agg}: skipped (no ranking_score in one of the steps)")
            continue
        compare(before, after, keys, agg, cut, metric, args.trials, args.seed)
    movers(before, after, keys, "mean", args.top)


if __name__ == "__main__":
    main()
