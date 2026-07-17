"""Decompose multimer size scaling into folding (intra-chain Rg) and docking
(inter-chain distance), and scale each separately.

By the parallel-axis theorem the whole-complex gyration splits cleanly:
    Rg²_complex = Σ_c (n_c/N)·Rg²_intra,c   +   Σ_c (n_c/N)·‖ĉ_c − ĉ‖²
                 └── folding (chain compactness) ──┘   └── docking (chain spread) ──┘
so a low whole-complex Rg (chain collapse) can come from either under-folded chains
or chains packed too close. This separates them:

  (A) intra-chain Rg vs chain length n   → does each chain fold compactly? (b→1/3)
  (B) mean inter-chain centroid distance D vs complex size N → are chains placed
      at the right separation, or collapsed onto each other?  (D_pred/D_gt → 1?)

Per step, Rg_intra / D are averaged over seed-42 diffusion samples per target and
fit as power laws; the literature Rg=3.2·N^(1/3) and the ground-truth D(N) are the
references.

Usage:
    python bench/scripts/rg_decompose.py --series df3v2 --category interface_protein_protein
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from itertools import combinations
from pathlib import Path

import gemmi
import numpy as np

BENCH = Path(__file__).resolve().parents[1]
GT_ROOT = BENCH / "data" / "ground_truths"
N_SAMPLES = 3
LIT_A, LIT_B = 3.2, 1.0 / 3.0
XLIM_N = (18, 1300)


def chains_ca(cif: Path) -> list[np.ndarray]:
    """Per-chain representative-atom coords (>= 4 residues)."""
    try:
        model = gemmi.read_structure(str(cif))[0]
    except Exception:
        return []
    out = []
    for chain in model:
        cs = []
        for res in chain:
            for atom in res:
                if atom.name in ("CA", "C1'"):
                    cs.append([atom.pos.x, atom.pos.y, atom.pos.z])
                    break
        if len(cs) >= 4:
            out.append(np.asarray(cs))
    return out


def rg(coords: np.ndarray) -> float:
    c = coords - coords.mean(0)
    return float(np.sqrt((c**2).sum(1).mean()))


def decompose(chains: list[np.ndarray]):
    """Return (per-chain [(n, Rg_intra)], (N_total, mean pairwise centroid dist))."""
    intra = [(len(c), rg(c)) for c in chains]
    cents = [c.mean(0) for c in chains]
    n_total = sum(len(c) for c in chains)
    if len(cents) >= 2:
        dists = [float(np.linalg.norm(a - b)) for a, b in combinations(cents, 2)]
        d = statistics.fmean(dists)
    else:
        d = None
    return intra, (n_total, d)


def fit_exponent(pairs):
    if len(pairs) < 5:
        return None
    n = np.array([p[0] for p in pairs], float)
    r = np.array([p[1] for p in pairs], float)
    x, y = np.log(n), np.log(r)
    mat = np.vstack([np.ones_like(x), x]).T
    (loga, b), *_ = np.linalg.lstsq(mat, y, rcond=None)
    r2 = 1 - ((y - mat @ [loga, b]) ** 2).sum() / (((y - y.mean()) ** 2).sum() + 1e-12)
    return b, float(np.exp(loga)), r2, len(pairs)


STEP_CACHE = "_rg_decompose_cache.json"  # per-step cache of per-target folding/docking decomposition
GT_CACHE = GT_ROOT / "_rg_decompose_gt_cache.json"  # global GT decomposition cache, keyed by pdb id


def _load_json(path: Path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _pack(intra, n_total, d) -> dict:
    """Serialize one target's decomposition: per-chain (n, Rg), total N, mean inter-chain D."""
    return {"intra": [[n, round(r, 4)] for n, r in intra], "N": n_total, "D": None if d is None else round(d, 4)}


def _target_gt(pdb: str) -> dict | None:
    g = GT_ROOT / f"{pdb}.cif"
    if not g.exists():
        return None
    chains = chains_ca(g)
    if len(chains) < 2:
        return None
    intra, (n_total, d) = decompose(chains)
    return _pack(intra, n_total, d)


def _target_pred(pdb_dir: Path) -> dict | None:
    """Predicted decomposition for one target, averaged over seed-42 samples (chains aligned by order)."""
    samp_intra, samp_D, samp_N = [], [], None
    for s in range(N_SAMPLES):
        chains = chains_ca(pdb_dir / f"seed-42_sample-{s}" / "model.cif")
        if len(chains) < 2:
            continue
        intra, inter = decompose(chains)
        samp_intra.append(intra)
        samp_N = inter[0]
        if inter[1] is not None:
            samp_D.append(inter[1])
    if not samp_intra:
        return None
    n_chains = min(len(si) for si in samp_intra)
    intra = [(samp_intra[0][ci][0], statistics.fmean(si[ci][1] for si in samp_intra)) for ci in range(n_chains)]
    d = statistics.fmean(samp_D) if samp_D else None
    return _pack(intra, samp_N, d)


def _ref_decomp(pdbs, force):
    """Reference (GT) decomposition via a global cache. Returns (ref_intra, ref_inter, gt_D, n_new)."""
    cache = {} if force else (_load_json(GT_CACHE) or {})
    n_new = 0
    for pdb in sorted(pdbs):
        if pdb in cache:
            continue
        t = _target_gt(pdb)
        if t is not None:
            cache[pdb] = t
            n_new += 1
    if n_new:
        with open(GT_CACHE, "w") as f:
            json.dump(cache, f)
    ref_intra, ref_inter, gt_D = [], [], {}
    for pdb in sorted(pdbs):
        t = cache.get(pdb)
        if not t:
            continue
        ref_intra += [(int(n), float(r)) for n, r in t["intra"]]
        if t["D"] is not None:
            inter = (int(t["N"]), float(t["D"]))
            ref_inter.append(inter)
            gt_D[pdb] = inter
    return ref_intra, ref_inter, gt_D, n_new


def _step_decomp(step_dir: Path, force: bool):
    """Per-target predicted decomposition for one step, cached in the step dir. Returns (targets, computed)."""
    cache_file = step_dir / STEP_CACHE
    if not force:
        cached = _load_json(cache_file)
        if cached is not None:
            return cached["targets"], False
    targets = {}
    for pdb_dir in sorted(step_dir.iterdir()):
        if not pdb_dir.is_dir() or pdb_dir.name.startswith("_"):
            continue
        t = _target_pred(pdb_dir)
        if t is not None:
            targets[pdb_dir.name] = t
    with open(cache_file, "w") as f:
        json.dump({"targets": targets}, f)
    return targets, True


def collect(series: str, category: str, force: bool = False):
    """Per-step folding/docking decomposition.

    Per-step target decompositions are cached in each step directory
    (``_rg_decompose_cache.json``) and GT decompositions in a global cache, so
    only steps / targets not computed before are read off disk. ``--force``
    rebuilds every cache.
    """
    pred_root = BENCH / "runs" / series / "pred" / category
    steps = sorted(int(p.name.split("_")[1]) for p in pred_root.glob("step_*") if p.name != "step_")
    pdbs: set[str] = set()
    for step in steps:
        pdbs |= {p.name for p in (pred_root / f"step_{step}").iterdir() if p.is_dir() and not p.name.startswith("_")}

    ref_intra, ref_inter, gt_D, n_new_ref = _ref_decomp(pdbs, force)

    records, n_new_steps = [], 0
    for step in steps:
        targets, computed = _step_decomp(pred_root / f"step_{step}", force)
        n_new_steps += int(computed)
        intra_pairs, inter_pairs, dratios = [], [], []
        for name, t in targets.items():
            intra_pairs += [(int(n), float(r)) for n, r in t["intra"]]
            if t["D"] is not None:
                inter_pairs.append((int(t["N"]), float(t["D"])))
                if name in gt_D:
                    dratios.append(float(t["D"]) / max(gt_D[name][1], 1e-6))
        records.append(
            {
                "step": step,
                "intra_fit": fit_exponent(intra_pairs),
                "inter_fit": fit_exponent(inter_pairs),
                "intra_pairs": intra_pairs,
                "inter_pairs": inter_pairs,
                "d_ratio": statistics.median(dratios) if dratios else float("nan"),
            }
        )
    return records, fit_exponent(ref_intra), fit_exponent(ref_inter), ref_intra, ref_inter, n_new_steps, n_new_ref


def plot(records, ref_intra_fit, ref_inter_fit, ref_intra, ref_inter, series, category, out_png):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import Normalize

    steps = [r["step"] for r in records]
    norm = Normalize(min(steps), max(steps))
    cmap = cm.viridis
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(15, 6))
    nx = np.array(XLIM_N, dtype=float)

    # Panel A: intra-chain Rg vs chain length n (folding)
    ri = np.array(ref_intra)
    axA.scatter(ri[:, 0], ri[:, 1], s=12, c="tab:blue", alpha=0.3, edgecolor="none", label="FoldBench chains", zorder=2)
    axA.plot(nx, LIT_A * nx**LIT_B, "--", color="tab:green", lw=2.5, zorder=6, label=r"literature $R_g=3.2\,N^{1/3}$")
    if ref_intra_fit:
        axA.plot(
            nx,
            ref_intra_fit[1] * nx ** ref_intra_fit[0],
            "--",
            color="tab:blue",
            lw=2.5,
            zorder=5,
            label=rf"FoldBench fit $R_g={ref_intra_fit[1]:.2f}\,n^{{{ref_intra_fit[0]:.3f}}}$",
        )
    for r in records:
        if r["intra_fit"]:
            axA.plot(
                nx,
                r["intra_fit"][1] * nx ** r["intra_fit"][0],
                "-",
                color=cmap(norm(r["step"])),
                lw=1.5,
                alpha=0.9,
                zorder=3,
            )
    axA.set_xscale("log")
    axA.set_yscale("log")
    axA.set_xlim(*XLIM_N)
    axA.set_ylim(3.5, 40)
    axA.set_xlabel(r"chain length  $n$")
    axA.set_ylabel(r"intra-chain radius of gyration  $R_g^{\mathrm{intra}}$ (Å)")
    axA.set_title(rf"{category}: FOLDING — per-chain $R_g^{{\mathrm{{intra}}}}$ vs $n$")
    axA.grid(True, which="both", alpha=0.2)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=axA, pad=0.01).set_label("training step")
    axA.legend(fontsize=8, loc="upper left")

    # Panel B: inter-chain distance D vs complex N (docking)
    # D vs N: the per-step power-law fit is unstable (low R²), so show the GT
    # scaling (clean) + early/late predicted scatter — predictions sit BELOW GT
    # (chains too close). The D_pred/D_gt ratio (in the label) is the robust
    # collapse metric; per-step fit lines would just be noise.
    re = np.array(ref_inter)
    axB.scatter(
        re[:, 0], re[:, 1], s=16, c="tab:blue", alpha=0.4, edgecolor="none", label="GT (experimental docking)", zorder=2
    )
    if ref_inter_fit:
        axB.plot(
            nx,
            ref_inter_fit[1] * nx ** ref_inter_fit[0],
            "--",
            color="tab:blue",
            lw=2.5,
            zorder=6,
            label=rf"GT fit $D={ref_inter_fit[1]:.2f}\,N^{{{ref_inter_fit[0]:.3f}}}$",
        )
    for r in (records[0], records[-1]):
        if r["inter_pairs"]:
            p = np.array(r["inter_pairs"])
            axB.scatter(
                p[:, 0],
                p[:, 1],
                s=15,
                color=cmap(norm(r["step"])),
                alpha=0.6,
                edgecolor="none",
                zorder=3,
                label=rf"DF3 step {r['step']}  ($D/D_{{\mathrm{{GT}}}}$={r['d_ratio']:.2f})",
            )
    axB.set_xscale("log")
    axB.set_yscale("log")
    axB.set_xlim(*XLIM_N)
    axB.set_ylim(3, 90)
    axB.set_xlabel(r"complex size  $N$")
    axB.set_ylabel(r"mean inter-chain centroid distance  $D$ (Å)")
    axB.set_title(r"DOCKING — chain separation $D$ vs $N$ (collapse $\Rightarrow$ $D$ too small)")
    axB.grid(True, which="both", alpha=0.2)
    sm2 = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm2.set_array([])
    fig.colorbar(sm2, ax=axB, pad=0.01).set_label("training step")
    axB.legend(fontsize=8, loc="upper left")

    fig.suptitle(
        rf"{series} — {category}: size-scaling split into folding ($R_g^{{\mathrm{{intra}}}}$) vs "
        rf"docking ($D$)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--series", default="df3v2")
    ap.add_argument("--category", default="interface_protein_protein")
    ap.add_argument("--force", action="store_true", help="ignore caches and recompute every step / GT target")
    args = ap.parse_args()

    records, rif, ref_inter_fit, ref_intra, ref_inter, n_new_steps, n_new_ref = collect(
        args.series, args.category, args.force
    )
    if not records:
        print(f"[rg-decompose] no data for {args.series}/{args.category}")
        return
    print(
        f"[rg-decompose] cache: {len(records) - n_new_steps} steps reused, {n_new_steps} computed; "
        f"{n_new_ref} GT targets computed"
    )
    if rif:
        print(f"REF intra (folding):  Rg = {rif[1]:.2f}·n^{rif[0]:.3f}  (R2={rif[2]:.2f}, n_chains={rif[3]})")
    if ref_inter_fit:
        print(
            f"REF inter (docking):  D  = {ref_inter_fit[1]:.2f}·N^{ref_inter_fit[0]:.3f}  "
            f"(R2={ref_inter_fit[2]:.2f}, n={ref_inter_fit[3]})"
        )
    print(f"{'step':>6} | {'b_intra':>8} {'b_inter':>8} {'D_pred/D_gt':>12}")
    out_root = BENCH / "runs" / args.series
    with open(out_root / f"rg_decompose_{args.category}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["series", "category", "step", "b_intra", "b_inter", "d_pred_over_gt"])
        for r in records:
            bi = r["intra_fit"][0] if r["intra_fit"] else float("nan")
            bo = r["inter_fit"][0] if r["inter_fit"] else float("nan")
            print(f"{r['step']:>6} | {bi:>8.3f} {bo:>8.3f} {r['d_ratio']:>12.2f}")
            w.writerow([args.series, args.category, r["step"], round(bi, 4), round(bo, 4), round(r["d_ratio"], 4)])
    out_png = out_root / f"rg_decompose_{args.category}.png"
    plot(records, rif, ref_inter_fit, ref_intra, ref_inter, args.series, args.category, out_png)
    print(f"[rg-decompose] plot -> {out_png}")


if __name__ == "__main__":
    main()
