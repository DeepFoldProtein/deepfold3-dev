"""Radius-of-gyration scaling as an order parameter for the learning phase
transition (DF3 analogue of OpenFold Extended Data Fig 7).

Real proteins obey Rg = a * N^b with b ~ 0.37 (between the Flory compact-globule
exponent 1/3 and a self-avoiding coil ~0.6). Early in training a structure module
first emits extended chains (large b), overshoots into an over-collapsed regime
(small b, Rg ~ N-independent), then slowly re-expands toward the experimental
exponent. Tracking b(step) makes that trajectory a single scalar order parameter.

Per training step we collect (N=n_residues, Rg) for every FoldBench-small target
(Rg averaged over that target's seed-42 diffusion samples, to damp sampling noise),
fit log Rg = log a + b log N by least squares, and plot:
  (A) log-log Rg vs N, reference (experimental) + a family of per-step fit lines
      whose slope rotates from extended -> collapsed -> toward experimental;
  (B) the exponent b(step) trajectory vs the experimental b.

Usage:
    python bench/scripts/rg_scaling.py --series df3v2 --category monomer_protein
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import gemmi
import numpy as np

BENCH = Path(__file__).resolve().parents[1]
GT_ROOT = BENCH / "data" / "ground_truths"

# Shared plot limits so monomer and multimer figures are directly comparable.
XLIM_N = (18, 1300)  # sequence length (log)
YLIM_RG = (3.5, 70)  # radius of gyration Å (log)
YLIM_B = (0.0, 0.7)  # scaling exponent b

# Literature compact-globule reference Rg = LIT_A * N^LIT_B (Cα, ~3.2·N^(1/3)),
# plotted alongside the least-squares fit to *these* FoldBench reference structures.
LIT_A = 3.2
LIT_B = 1.0 / 3.0


def rg_and_n(cif: Path) -> tuple[int, float] | None:
    """(n_residues, radius of gyration over representative CA / C1' atoms)."""
    try:
        model = gemmi.read_structure(str(cif))[0]
    except Exception:
        return None
    coords = []
    for chain in model:
        for res in chain:
            for atom in res:
                if atom.name in ("CA", "C1'"):
                    coords.append([atom.pos.x, atom.pos.y, atom.pos.z])
                    break
    if len(coords) < 10:
        return None
    c = np.asarray(coords)
    c = c - c.mean(0)
    return len(coords), float(np.sqrt((c**2).sum(1).mean()))


def fit_exponent(pairs: list[tuple[int, float]]) -> tuple[float, float, float, int] | None:
    """Least-squares (b, a, R^2, n) in log Rg = log a + b log N."""
    if len(pairs) < 5:
        return None
    n = np.array([p[0] for p in pairs], float)
    r = np.array([p[1] for p in pairs], float)
    x, y = np.log(n), np.log(r)
    mat = np.vstack([np.ones_like(x), x]).T
    (loga, b), *_ = np.linalg.lstsq(mat, y, rcond=None)
    resid = y - mat @ [loga, b]
    r2 = 1 - (resid**2).sum() / (((y - y.mean()) ** 2).sum() + 1e-12)
    return b, float(np.exp(loga)), r2, len(pairs)


STEP_CACHE = "_rg_cache.json"  # per-step cache of (n_res, Rg) target pairs, co-located in the step dir
GT_CACHE = GT_ROOT / "_rg_scaling_gt_cache.json"  # global experimental Rg cache, keyed by pdb id


def _load_json(path: Path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _ref_pairs(pdbs: set[str], force: bool) -> tuple[list[tuple[int, float]], int]:
    """Experimental (n, Rg) per target via a global GT cache. Returns (pairs, n_newly_computed)."""
    cache = {} if force else (_load_json(GT_CACHE) or {})
    n_new = 0
    for pdb in sorted(pdbs):
        if pdb in cache:
            continue
        g = GT_ROOT / f"{pdb}.cif"
        if g.exists():
            rn = rg_and_n(g)
            if rn:
                cache[pdb] = [rn[0], round(rn[1], 4)]
                n_new += 1
    if n_new:
        with open(GT_CACHE, "w") as f:
            json.dump(cache, f)
    pairs = [(int(cache[p][0]), float(cache[p][1])) for p in sorted(pdbs) if p in cache]
    return pairs, n_new


def _step_pairs(step_dir: Path, force: bool) -> tuple[list[tuple[int, float]], bool]:
    """Per-target (n, mean Rg over seed-42 samples) for one step, cached in the step dir.

    Returns (pairs, computed) — ``computed`` is False on a cache hit.
    """
    cache_file = step_dir / STEP_CACHE
    if not force:
        cached = _load_json(cache_file)
        if cached is not None:
            return [(int(n), float(r)) for n, r in cached["pairs"]], False
    pairs = []
    for pdb_dir in sorted(step_dir.iterdir()):
        if not pdb_dir.is_dir() or pdb_dir.name.startswith("_"):
            continue
        rgs, n_res = [], None
        for s in range(5):
            cif = pdb_dir / f"seed-42_sample-{s}" / "model.cif"
            if cif.exists():
                rn = rg_and_n(cif)
                if rn:
                    rgs.append(rn[1])
                    n_res = rn[0]
        if rgs and n_res:
            pairs.append((n_res, statistics.fmean(rgs)))
    with open(cache_file, "w") as f:
        json.dump({"pairs": [[n, round(r, 4)] for n, r in pairs]}, f)
    return pairs, True


def collect(series: str, category: str, force: bool = False):
    """Return (records, ref_fit, ref_pairs, n_new_steps, n_new_ref).

    Per-step target pairs are cached in each step directory (``_rg_cache.json``)
    and experimental GT pairs in a global cache, so only steps / targets not
    computed before are read off disk. ``--force`` rebuilds every cache.
    """
    pred_root = BENCH / "runs" / series / "pred" / category
    steps = sorted(int(p.name.split("_")[1]) for p in pred_root.glob("step_*") if p.name != "step_")
    pdbs: set[str] = set()
    for step in steps:
        d = pred_root / f"step_{step}"
        pdbs |= {p.name for p in d.iterdir() if p.is_dir() and not p.name.startswith("_")}

    ref_pairs, n_new_ref = _ref_pairs(pdbs, force)
    ref_fit = fit_exponent(ref_pairs)

    records, n_new_steps = [], 0
    for step in steps:
        pairs, computed = _step_pairs(pred_root / f"step_{step}", force)
        n_new_steps += int(computed)
        fit = fit_exponent(pairs)
        if fit:
            records.append({"step": step, "b": fit[0], "a": fit[1], "r2": fit[2], "n": fit[3], "pairs": pairs})
    return records, ref_fit, ref_pairs, n_new_steps, n_new_ref


def plot(records, ref_fit, ref_pairs, series, category, out_png):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import Normalize

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(15, 6))
    steps = [r["step"] for r in records]
    norm = Normalize(min(steps), max(steps))
    # Reversed so the LATEST step is purple (the most salient viridis end) and
    # earliest is yellow — the current model's curve reads at a glance.
    cmap = cm.viridis_r

    # --- Panel A: log-log Rg vs N ---
    # GT reference in a colour+marker the step colormap can never emit (black ✕),
    # so experimental points are never mistaken for an early-step scatter.
    rn = np.array(ref_pairs)
    axA.scatter(
        rn[:, 0],
        rn[:, 1],
        s=30,  # plain stroke x (no fill, no outline) — arms are thin lines, so it never blankets
        marker="x",  # non-viridis red + distinct shape; step points show through the gaps
        c="#d62728",
        linewidths=1.0,
        alpha=0.45,
        label="FoldBench reference (GT)",
        zorder=6,
    )
    nx = np.array(XLIM_N, dtype=float)
    axA.plot(nx, LIT_A * nx**LIT_B, "--", color="tab:green", lw=2.5, zorder=6, label=r"literature $R_g=3.2\,N^{1/3}$")
    axA.plot(
        nx,
        ref_fit[1] * nx ** ref_fit[0],
        "--",
        color="tab:blue",
        lw=2.5,
        zorder=5,
        label=rf"FoldBench fit $R_g={ref_fit[1]:.2f}\,N^{{{ref_fit[0]:.3f}}}$",
    )
    # per-step fit lines (slope rotates over training); scatter for first & last step
    for r in (records[-1],):
        col = cmap(norm(r["step"]))
        axA.plot(nx, r["a"] * nx ** r["b"], "-", color=col, lw=1.6, alpha=0.9, zorder=3)
    last_step = records[-1]["step"]
    for r in records:  # (records[0], records[-1]):
        p = np.array(r["pairs"])
        # Latest step drawn on top of everything (GT z=6, lines z≤6) so the
        # current model's cloud is never occluded; older steps stay beneath.
        z = 12 if r["step"] == last_step else 2
        axA.scatter(p[:, 0], p[:, 1], s=10, color=cmap(norm(r["step"])), alpha=0.5, edgecolor="none", zorder=z)
    axA.set_xscale("log")
    axA.set_yscale("log")
    axA.set_xlim(*XLIM_N)  # shared across categories so monomer/multimer compare directly
    axA.set_ylim(*YLIM_RG)
    axA.set_xlabel(r"sequence length  $N$")
    axA.set_ylabel(r"radius of gyration  $R_g$ (Å)")
    axA.set_title(rf"{category}: $R_g$ vs $N$ (log-log), per-step power-law fits")
    axA.grid(True, which="both", alpha=0.2)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=axA, pad=0.01)
    cb.set_label("training step")
    axA.legend(fontsize=8, loc="upper left")

    # --- Panel B: exponent b(step) ---
    axB.axhline(LIT_B, color="tab:green", ls="--", lw=2, label=r"literature $b=1/3$")
    axB.axhline(ref_fit[0], color="tab:blue", ls="--", lw=2, label=rf"FoldBench fit $b={ref_fit[0]:.3f}$")
    axB.plot(steps, [r["b"] for r in records], "-o", color="tab:red", lw=2, ms=5, label=r"DF3 $b$(step)")
    for r in records:  # annotate the overshoot extremes
        if r["step"] in (steps[0], steps[1] if len(steps) > 1 else steps[0], steps[-1]):
            axB.annotate(
                f"{r['b']:.2f}", (r["step"], r["b"]), textcoords="offset points", xytext=(0, 8), fontsize=8, ha="center"
            )
    axB.set_xlabel("training step")
    axB.set_ylabel(r"scaling exponent  $b$   ($R_g \propto N^b$)")
    axB.set_title("size-scaling order parameter: extended → over-collapsed → re-expanding")
    axB.set_ylim(*YLIM_B)  # shared across categories
    axB.grid(True, alpha=0.3)
    axB.legend(fontsize=9, loc="best")

    fig.suptitle(
        f"{series} — Rg scaling as a learning-phase order parameter  (DF3 analogue of OpenFold Ext. Data Fig 7)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--series", default="df3v2")
    ap.add_argument("--category", default="monomer_protein")
    ap.add_argument("--force", action="store_true", help="ignore caches and recompute every step / GT target")
    args = ap.parse_args()

    records, ref_fit, ref_pairs, n_new_steps, n_new_ref = collect(args.series, args.category, args.force)
    if not records or ref_fit is None:
        print(f"[rg-scaling] no data for {args.series}/{args.category}")
        return
    print(
        f"[rg-scaling] cache: {len(records) - n_new_steps} steps reused, {n_new_steps} computed; "
        f"GT {n_new_ref} new / {len(ref_pairs) - n_new_ref} reused"
    )
    out_root = BENCH / "runs" / args.series
    out_csv = out_root / f"rg_scaling_{args.category}.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["series", "category", "step", "b_exponent", "a_prefactor", "r2", "n_targets"])
        for r in records:
            w.writerow(
                [args.series, args.category, r["step"], round(r["b"], 4), round(r["a"], 3), round(r["r2"], 3), r["n"]]
            )
    print(f"experimental: Rg = {ref_fit[1]:.2f}·N^{ref_fit[0]:.3f}  (R2={ref_fit[2]:.2f}, n={ref_fit[3]})")
    for r in records:
        print(f"  step {r['step']:>6}: b={r['b']:.3f}  a={r['a']:.2f}  R2={r['r2']:.2f}")
    out_png = out_root / f"rg_scaling_{args.category}.png"
    plot(records, ref_fit, ref_pairs, args.series, args.category, out_png)
    print(f"[rg-scaling] -> {out_csv}\n[rg-scaling] plot -> {out_png}")


if __name__ == "__main__":
    main()
