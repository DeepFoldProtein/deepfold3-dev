"""Gyration-tensor eigenvalue growth as a learning order parameter (DF3 analogue
of OpenFold's staggered 1D->2D->3D dimensional-growth picture, Ext. Data Fig 6).

For each predicted structure the all-atom (heavy-atom) gyration tensor has
eigenvalues λ1≥λ2≥λ3 (variances along its principal axes). Their normalised fractions f_k = λ_k / Σλ
are a size-independent shape fingerprint:
    rod (1D):    f ≈ (1.0, 0, 0)      disc (2D):  f ≈ (0.5, 0.5, 0)
    sphere:      f ≈ (1/3, 1/3, 1/3)  real protein ≈ (0.55, 0.30, 0.15)  (anisotropic)
Relative shape anisotropy κ² = 1 − 3(λ1λ2+λ2λ3+λ1λ3)/(Σλ)² ∈ [0,1]
    (0 sphere, 0.25 disc, 1 line; globular protein ~0.14, 2-chain complex ~0.27).

Averaging f_k / κ² over the target population per step is robust to per-sample
diffusion noise (unlike a single-structure PCA trajectory), so the dimensional
evolution shows up cleanly: early rod (f1≈1), a rapid over-isotropisation
(κ²→0), then a slow drift back toward the experimental anisotropy.

Usage:
    python bench/scripts/eigen_growth.py --series df3v2 --category monomer_protein
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
N_SAMPLES = 3  # seed-42 samples averaged per target (keeps the CIF read bounded)
YLIM_KAPPA = (0.0, 0.6)  # shared κ² axis so monomer/multimer compare directly (frac panel is fixed 0-1)


def eig(cif: Path) -> np.ndarray | None:
    """Sorted gyration-tensor eigenvalues λ1≥λ2≥λ3 over ALL heavy atoms."""
    try:
        model = gemmi.read_structure(str(cif))[0]
    except Exception:
        return None
    coords = [
        [atom.pos.x, atom.pos.y, atom.pos.z]
        for chain in model
        for res in chain
        for atom in res
        if atom.element.atomic_number != 1  # skip hydrogens if any
    ]
    if len(coords) < 10:
        return None
    c = np.asarray(coords)
    c = c - c.mean(0)
    ev = np.sort(np.linalg.eigvalsh((c[:, :, None] * c[:, None, :]).mean(0)))[::-1]
    return np.maximum(ev, 1e-9)


def descriptors(ev: np.ndarray) -> dict:
    s = ev.sum()
    f = ev / s
    kappa2 = 1 - 3 * (ev[0] * ev[1] + ev[1] * ev[2] + ev[0] * ev[2]) / (s * s)
    oblate = (ev[1] - ev[2]) > (ev[0] - ev[1])
    return {"f1": f[0], "f2": f[1], "f3": f[2], "kappa2": kappa2, "oblate": 1.0 if oblate else 0.0}


def mean_descr(evs: list[np.ndarray]) -> dict:
    ds = [descriptors(e) for e in evs]
    return {k: statistics.fmean(d[k] for d in ds) for k in ds[0]}


def collect(series: str, category: str):
    pred_root = BENCH / "runs" / series / "pred" / category
    steps = sorted(int(p.name.split("_")[1]) for p in pred_root.glob("step_*") if p.name != "step_")
    pdbs: set[str] = set()
    for step in steps:
        pdbs |= {p.name for p in (pred_root / f"step_{step}").iterdir() if p.is_dir() and not p.name.startswith("_")}

    ref_evs = []
    for pdb in sorted(pdbs):
        g = GT_ROOT / f"{pdb}.cif"
        if g.exists():
            e = eig(g)
            if e is not None:
                ref_evs.append(e)
    ref = mean_descr(ref_evs) if ref_evs else None

    records = []
    for step in steps:
        d = pred_root / f"step_{step}"
        per_target = []
        for pdb_dir in sorted(d.iterdir()):
            if not pdb_dir.is_dir() or pdb_dir.name.startswith("_"):
                continue
            evs = [eig(pdb_dir / f"seed-42_sample-{s}" / "model.cif") for s in range(N_SAMPLES)]
            evs = [e for e in evs if e is not None]
            if evs:
                per_target.append(np.mean(evs, axis=0))  # sample-averaged eigenvalues
        if per_target:
            rec = mean_descr(per_target)
            rec["step"] = step
            rec["n"] = len(per_target)
            records.append(rec)
    return records, ref


def plot(records, ref, series, category, out_png):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [r["step"] for r in records]
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(15, 6))

    # Panel A: eigenvalue fractions f1>f2>f3 (shape dimensionality)
    cols = {"f1": "tab:red", "f2": "tab:green", "f3": "tab:blue"}
    labs = {
        "f1": r"$\lambda_1/\Sigma\lambda$ (longest)",
        "f2": r"$\lambda_2/\Sigma\lambda$ (middle)",
        "f3": r"$\lambda_3/\Sigma\lambda$ (shortest)",
    }
    for k in ("f1", "f2", "f3"):
        axA.plot(steps, [r[k] for r in records], "-o", color=cols[k], lw=2, ms=4, label=labs[k])
        if ref:
            axA.axhline(ref[k], color=cols[k], ls="--", lw=1.4, alpha=0.8)
    axA.axhline(1 / 3, color="0.6", ls=":", lw=1, label=r"isotropic sphere $=1/3$")
    axA.set_xlabel("training step")
    axA.set_ylabel(r"eigenvalue fraction  $\lambda_k / \Sigma\lambda$")
    axA.set_title(rf"{category}: shape dimensionality (dashed = experimental)")
    axA.set_ylim(0, 1)
    axA.grid(True, alpha=0.3)
    axA.legend(fontsize=8, loc="upper right")

    # Panel B: anisotropy κ² + oblate fraction
    axB.plot(steps, [r["kappa2"] for r in records], "-o", color="tab:purple", lw=2, ms=5, label=r"$\kappa^2$ (DF3)")
    if ref:
        axB.axhline(
            ref["kappa2"], color="tab:purple", ls="--", lw=2, label=rf"experimental $\kappa^2={ref['kappa2']:.3f}$"
        )
    axB.axhline(0.25, color="0.6", ls=":", lw=1.2, label=r"ideal flat disc $\kappa^2=0.25$")
    for r in records:
        if r["step"] in (steps[0], steps[-1]):
            axB.annotate(
                f"{r['kappa2']:.2f}",
                (r["step"], r["kappa2"]),
                textcoords="offset points",
                xytext=(0, 8),
                fontsize=8,
                ha="center",
            )
    axB.set_xlabel("training step")
    axB.set_ylabel(r"relative shape anisotropy  $\kappa^2$")
    axB.set_title(
        r"anisotropy: rod ($\kappa^2\uparrow$) → over-isotropised ball ($\kappa^2\downarrow$) → drift toward exp"
    )
    axB.set_ylim(*YLIM_KAPPA)  # shared across categories
    axB.grid(True, alpha=0.3)
    axB2 = axB.twinx()
    axB2.plot(
        steps,
        [100 * r["oblate"] for r in records],
        "-s",
        color="tab:orange",
        lw=1.3,
        ms=4,
        alpha=0.7,
        label="% oblate (disc-like)",
    )
    axB2.set_ylabel("% oblate", color="tab:orange")
    axB2.set_ylim(0, 100)
    axB2.tick_params(axis="y", labelcolor="tab:orange")
    lines = axB.get_lines() + axB2.get_lines()
    axB.legend(lines, [ln.get_label() for ln in lines], fontsize=8, loc="upper right")

    fig.suptitle(
        f"{series} — gyration-tensor eigenvalue growth (DF3 analogue of OpenFold dimensional staggering)", fontsize=12
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--series", default="df3v2")
    ap.add_argument("--category", default="monomer_protein")
    args = ap.parse_args()

    records, ref = collect(args.series, args.category)
    if not records:
        print(f"[eigen-growth] no data for {args.series}/{args.category}")
        return
    out_root = BENCH / "runs" / args.series
    out_csv = out_root / f"eigen_growth_{args.category}.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["series", "category", "step", "f1", "f2", "f3", "kappa2", "pct_oblate", "n_targets"])
        for r in records:
            w.writerow(
                [
                    args.series,
                    args.category,
                    r["step"],
                    round(r["f1"], 4),
                    round(r["f2"], 4),
                    round(r["f3"], 4),
                    round(r["kappa2"], 4),
                    round(100 * r["oblate"], 1),
                    r["n"],
                ]
            )
    if ref:
        print(
            f"experimental: f=({ref['f1']:.3f},{ref['f2']:.3f},{ref['f3']:.3f})  κ²={ref['kappa2']:.3f}  "
            f"%oblate={100 * ref['oblate']:.0f}"
        )
    for r in records:
        print(
            f"  step {r['step']:>6}: f=({r['f1']:.3f},{r['f2']:.3f},{r['f3']:.3f})  κ²={r['kappa2']:.3f}  "
            f"%oblate={100 * r['oblate']:.0f}"
        )
    plot(records, ref, args.series, args.category, out_root / f"eigen_growth_{args.category}.png")
    print(f"[eigen-growth] -> {out_csv}\n[eigen-growth] plot -> {out_root / f'eigen_growth_{args.category}.png'}")


if __name__ == "__main__":
    main()
