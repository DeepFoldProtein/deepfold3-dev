"""Convergence tracker: overlay the two shape order parameters — the Rg-scaling
exponent b(step) and the shape anisotropy κ²(step) — against their physical
targets, to see whether the model is still normalising toward real-protein shape
or has plateaued at an over-compact / over-isotropic ceiling.

Lightweight: reads the per-step aggregates already written by rg_scaling.py
(b) and eigen_growth.py (κ²); only the experimental κ² is (re)computed from the
ground-truth structures (all-atom, once). b target = compact-globule 1/3;
κ² target = the experimental mean.

Usage:
    python bench/scripts/convergence_tracker.py --series df3v2 --category monomer_protein
"""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

import gemmi
import numpy as np

BENCH = Path(__file__).resolve().parents[1]
GT_ROOT = BENCH / "data" / "ground_truths"
B_TARGET = 1.0 / 3.0  # compact-globule Rg ∝ N^(1/3)


def _kappa2(cif: Path) -> float | None:
    try:
        model = gemmi.read_structure(str(cif))[0]
    except Exception:
        return None
    coords = [[a.pos.x, a.pos.y, a.pos.z] for ch in model for r in ch for a in r if a.element.atomic_number != 1]
    if len(coords) < 10:
        return None
    c = np.asarray(coords)
    c = c - c.mean(0)
    ev = np.sort(np.linalg.eigvalsh((c[:, :, None] * c[:, None, :]).mean(0)))[::-1]
    ev = np.maximum(ev, 1e-9)
    s = ev.sum()
    return 1 - 3 * (ev[0] * ev[1] + ev[1] * ev[2] + ev[0] * ev[2]) / (s * s)


def _read_series(csv_path: Path, col: str) -> dict[int, float]:
    out: dict[int, float] = {}
    if not csv_path.exists():
        return out
    for r in csv.DictReader(open(csv_path)):
        try:
            out[int(r["step"])] = float(r[col])
        except (ValueError, KeyError):
            continue
    return out


def experimental_kappa2(series: str, category: str) -> float | None:
    pred_root = BENCH / "runs" / series / "pred" / category
    pdbs: set[str] = set()
    for d in pred_root.glob("step_*"):
        pdbs |= {p.name for p in d.iterdir() if p.is_dir() and not p.name.startswith("_")}
    vals = []
    for pdb in sorted(pdbs):
        g = GT_ROOT / f"{pdb}.cif"
        if g.exists():
            k = _kappa2(g)
            if k is not None:
                vals.append(k)
    return statistics.fmean(vals) if vals else None


def plot(steps, bvals, kvals, b_target, k_target, series, category, out_png):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axL = plt.subplots(figsize=(9, 6))
    axR = axL.twinx()

    (lb,) = axL.plot(steps, bvals, "-o", color="tab:red", lw=2.2, ms=5, label=r"$b$  ($R_g \propto N^b$)")
    axL.axhline(b_target, color="tab:red", ls="--", lw=1.6, alpha=0.7, label=r"$b$ target $=1/3$")
    (lk,) = axR.plot(steps, kvals, "-s", color="tab:purple", lw=2.2, ms=5, label=r"$\kappa^2$ (anisotropy)")
    axR.axhline(k_target, color="tab:purple", ls="--", lw=1.6, alpha=0.7, label=rf"$\kappa^2$ target $={k_target:.3f}$")

    axL.set_xlabel("training step")
    axL.set_ylabel(r"Rg-scaling exponent  $b$", color="tab:red")
    axR.set_ylabel(r"shape anisotropy  $\kappa^2$", color="tab:purple")
    axL.tick_params(axis="y", labelcolor="tab:red")
    axR.tick_params(axis="y", labelcolor="tab:purple")
    axL.set_ylim(0, max(0.7, max(bvals) * 1.1))
    axR.set_ylim(0, max(k_target * 1.3, max(kvals) * 1.1))
    axL.grid(True, alpha=0.3)

    # % of target reached at the latest step (post-collapse recovery)
    b_pct = 100 * bvals[-1] / b_target
    k_pct = 100 * kvals[-1] / k_target
    axL.set_title(
        rf"{series} — {category}: shape convergence tracker"
        f"\nlatest: $b$={bvals[-1]:.3f} ({b_pct:.0f}% of 1/3),  "
        rf"$\kappa^2$={kvals[-1]:.3f} ({k_pct:.0f}% of exp)",
        fontsize=11,
    )
    axL.legend(handles=[lb, lk], loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--series", default="df3v2")
    ap.add_argument("--category", default="monomer_protein")
    args = ap.parse_args()

    root = BENCH / "runs" / args.series
    bmap = _read_series(root / f"rg_scaling_{args.category}.csv", "b_exponent")
    kmap = _read_series(root / f"eigen_growth_{args.category}.csv", "kappa2")
    steps = sorted(set(bmap) & set(kmap))
    if not steps:
        print(f"[convergence] need rg_scaling + eigen_growth CSVs for {args.series}/{args.category}")
        return
    bvals = [bmap[s] for s in steps]
    kvals = [kmap[s] for s in steps]
    k_target = experimental_kappa2(args.series, args.category) or 0.145
    print(f"targets: b=1/3={B_TARGET:.3f}  κ²_exp={k_target:.3f}")
    for s, b, k in zip(steps, bvals, kvals, strict=True):
        print(
            f"  step {s:>6}: b={b:.3f} ({100 * b / B_TARGET:.0f}% of 1/3)  "
            f"κ²={k:.3f} ({100 * k / k_target:.0f}% of exp)"
        )
    out_png = root / f"convergence_{args.category}.png"
    plot(steps, bvals, kvals, B_TARGET, k_target, args.series, args.category, out_png)
    print(f"[convergence] plot -> {out_png}")


if __name__ == "__main__":
    main()
