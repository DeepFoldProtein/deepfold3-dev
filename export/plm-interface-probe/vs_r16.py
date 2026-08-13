"""Does the PLM's interface signal touch the targets r16 actually fails on?

The probe established that a frozen ESM2 embedding predicts *which residues sit
at an interface* (PPI AUC ~0.70 against a 0.52 null). That is only useful if it
is informative where r16 is weak. If the targets whose interface the PLM reads
well are also the ones r16 already folds correctly, the signal is redundant and
the shim has nothing to add.

Per-target interface AUC is computed by 2-fold cross-fitting so every target
gets a score from a probe that never saw it. It is then correlated against
r16's per-target DockQ at a chosen step, using Spearman ρ (rank-based: neither
quantity is expected to be linearly related, and DockQ is bounded and skewed).

Usage:
    python export/plm-interface-probe/vs_r16.py --step 17000
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe import (  # noqa: E402
    Embedder,
    Target,
    auc,
    contact_matrix,
    fit_logistic,
    interface_fraction,
    load_target,
)

REPO = Path(__file__).resolve().parents[2]
TARGETS_ROOT = REPO / "bench" / "data" / "targets"
BENCH = REPO / "bench" / "runs"
OUT = Path(__file__).resolve().parent

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("vs_r16")


def spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Spearman ρ and a two-sided p-value from the t approximation."""

    def rank(v):
        order = np.argsort(v)
        r = np.empty(len(v), dtype=float)
        r[order] = np.arange(len(v), dtype=float)
        # average ties
        _, inv, counts = np.unique(v, return_inverse=True, return_counts=True)
        sums = np.zeros(len(counts))
        np.add.at(sums, inv, r)
        return (sums / counts)[inv]

    rx, ry = rank(x), rank(y)
    rho = float(np.corrcoef(rx, ry)[0, 1])
    n = len(x)
    if n < 4 or abs(rho) >= 1:
        return rho, float("nan")
    t = rho * math.sqrt((n - 2) / (1 - rho**2))
    # two-sided survival of Student-t via the incomplete beta identity
    df = n - 2
    x_b = df / (df + t * t)
    # regularized incomplete beta I_x(df/2, 1/2) by continued fraction (scipy-free)
    from math import lgamma

    def betacf(a, b, x):
        tiny, qab, qap, qam = 1e-30, a + b, a + 1.0, a - 1.0
        c, d = 1.0, 1.0 - qab * x / qap
        d = tiny if abs(d) < tiny else d
        d, h = 1.0 / d, 1.0 / d
        for m in range(1, 200):
            m2 = 2 * m
            aa = m * (b - m) * x / ((qam + m2) * (a + m2))
            d = 1.0 + aa * d
            c = 1.0 + aa / c
            d = tiny if abs(d) < tiny else d
            c = tiny if abs(c) < tiny else c
            d = 1.0 / d
            h *= d * c
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
            d = 1.0 + aa * d
            c = 1.0 + aa / c
            d = tiny if abs(d) < tiny else d
            c = tiny if abs(c) < tiny else c
            d = 1.0 / d
            delta = d * c
            h *= delta
            if abs(delta - 1.0) < 3e-7:
                break
        return h

    def betai(a, b, x):
        if x <= 0:
            return 0.0
        if x >= 1:
            return 1.0
        lb = lgamma(a + b) - lgamma(a) - lgamma(b) + a * math.log(x) + b * math.log(1 - x)
        front = math.exp(lb)
        return front * betacf(a, b, x) / a if x < (a + 1) / (a + b + 2) else 1 - math.exp(lb) * betacf(b, a, 1 - x) / b

    return rho, float(betai(df / 2, 0.5, x_b))


def per_target_auc(targets: list[Target], emb: Embedder, layer: int) -> dict[str, float]:
    """2-fold cross-fitted interface AUC, so no target scores its own probe."""
    cache = {t.pdb_id: emb(t, "per_chain", layer) for t in targets}
    labels = {}
    for t in targets:
        per_chain = []
        for i, ch in enumerate(t.chains):
            m = np.zeros(len(ch.seq), dtype=np.float32)
            for j in range(len(t.chains)):
                if i != j:
                    m = np.maximum(m, contact_matrix(ch, t.chains[j]).max(1))
            per_chain.append(m)
        labels[t.pdb_id] = per_chain

    half = len(targets) // 2
    folds = [(targets[:half], targets[half:]), (targets[half:], targets[:half])]
    out: dict[str, float] = {}
    for train, test in folds:
        x = np.concatenate([e for t in train for e in cache[t.pdb_id]])
        y = np.concatenate([e for t in train for e in labels[t.pdb_id]])
        w = fit_logistic(x, y)
        for t in test:
            tx = np.concatenate(cache[t.pdb_id])
            ty = np.concatenate(labels[t.pdb_id])
            a = auc(np.concatenate([tx, np.ones((len(tx), 1), np.float32)], 1) @ w, ty)
            if not np.isnan(a):
                out[t.pdb_id] = a
    return out


def r16_dockq(step: int, series: str = "df3v5r16") -> dict[str, float]:
    """Per-target mean DockQ over every seed, sample and annotated interface."""
    path = (
        BENCH
        / series
        / "eval"
        / "interface_protein_protein"
        / f"step_{step}"
        / "raw"
        / "interface_protein_protein_ost.csv"
    )
    vals: dict[str, list[float]] = defaultdict(list)
    for row in csv.DictReader(open(path)):
        try:
            vals[row["pdb_id"]].append(float(row["dockq"]))
        except (TypeError, ValueError):
            continue
    return {k: float(np.mean(v)) for k, v in vals.items() if v}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/data/plm/esm2_t33_650M_UR50D")
    ap.add_argument("--tag", default="esm2")
    ap.add_argument("--step", type=int, default=17000)
    ap.add_argument("--limit", type=int, default=80)
    ap.add_argument("--layer", type=int, default=33)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--min-iface", type=float, default=0.05)
    ap.add_argument("--max-iface", type=float, default=0.50)
    args = ap.parse_args()

    seen: set[str] = set()
    targets: list[Target] = []
    for row in csv.DictReader(open(TARGETS_ROOT / "interface_protein_protein.csv")):
        pid = row["pdb_id"]
        if pid in seen:
            continue
        seen.add(pid)
        t = load_target(pid, "ppi")
        if t is not None and args.min_iface <= interface_fraction(t) <= args.max_iface:
            targets.append(t)
        if len(targets) >= args.limit:
            break

    dockq = r16_dockq(args.step)
    targets = [t for t in targets if t.pdb_id in dockq]
    log.info("targets with both probe and r16 DockQ: %d", len(targets))

    emb = Embedder(Path(args.model_dir), args.device)
    aucs = per_target_auc(targets, emb, args.layer)

    pairs = [(aucs[t.pdb_id], dockq[t.pdb_id], t.pdb_id) for t in targets if t.pdb_id in aucs]
    x = np.array([p[0] for p in pairs])
    y = np.array([p[1] for p in pairs])
    rho, pval = spearman(x, y)
    log.info("n = %d", len(pairs))
    log.info("interface AUC : mean %.3f  sd %.3f", x.mean(), x.std(ddof=1))
    log.info("r16 DockQ     : mean %.3f  sd %.3f", y.mean(), y.std(ddof=1))
    log.info("Spearman rho = %+.3f  (p = %.3f)", rho, pval)

    lo = y < np.median(y)
    log.info(
        "probe AUC on r16-failing half %.3f vs succeeding half %.3f (delta %+.3f)",
        x[lo].mean(),
        x[~lo].mean(),
        x[~lo].mean() - x[lo].mean(),
    )

    with open(OUT / f"vs_r16_{args.tag}_step{args.step}.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["pdb_id", "interface_auc", "r16_mean_dockq"])
        for a, d, pid in sorted(pairs):
            writer.writerow([pid, round(a, 4), round(d, 4)])
    log.info("wrote %s", OUT / f"vs_r16_{args.tag}_step{args.step}.csv")


if __name__ == "__main__":
    main()
