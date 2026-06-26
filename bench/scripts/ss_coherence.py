"""Coherence length (dRMSD-by-sequence-separation) — superposition-free topology diagnostic.

For each monomer target we align pred↔GT Cα chains (Needleman–Wunsch on the
1-letter sequence), then measure, as a function of sequence separation s, how
faithfully the predicted internal-distance map matches the native one:

    Dᵖ_ij = ‖xᵢ−xⱼ‖ (pred)   Dᵍ_ij = ‖yᵢ−yⱼ‖ (GT)
    dRMSD(s) = sqrt( mean_{0.7s ≤ |i−j| < 1.5s} (Dᵖ−Dᵍ)² )

dRMSD is rigid-motion invariant (internal distances only → no superposition).
The **coherence length** L is the smallest separation at which the curve first
crosses τ=3 Å (log-interpolated): below L the prediction is locally faithful,
above L it has drifted. Short L ≈ single SSE; sheet/domain coherence ≈ 16–48 res.

L correlates with lDDT as strongly as any SS metric (corr(log L, lDDT) ≈ +0.67
on df3v1@105k), and corr(log L, length) < 0 — i.e. large proteins lose long-range
coherence. The size-binned L trajectory is the direct test of whether a larger
training crop lifts long-range topology for big chains.

Per-step results cache to ss_analysis/coherence/L_step{N}.csv; the trajectory
plot reuses those caches (only uncached steps recompute).

Usage:
    python bench/scripts/ss_coherence.py --series df3v1 --category monomer_protein --step 110000 --workers 48
    python bench/scripts/ss_coherence.py --series df3v1 --category monomer_protein --steps all --workers 48
"""
from __future__ import annotations

import argparse
import csv
import glob
import re
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import gemmi
import numpy as np

REPO = Path(__file__).resolve().parents[2]
BENCH = REPO / "bench"
GT_DIR = BENCH / "data" / "ground_truths"

TAU = 3.0  # Å threshold defining coherence length
ANCHORS = [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384]
SIZE_BINS = [("small", 0, 200), ("medium", 200, 400), ("large", 400, 10**9)]


def ca_seq(cif: str) -> tuple[str, np.ndarray]:
    """Cα coordinates + 1-letter sequence for the (single) protein chain."""
    st = gemmi.read_structure(cif)
    st.setup_entities()
    seq, xyz = [], []
    for ch in st[0]:
        for res in ch:
            info = gemmi.find_tabulated_residue(res.name)
            if info is None or not info.is_amino_acid():
                continue
            ca = res.find_atom("CA", "*")
            if ca is None:
                continue
            code = info.one_letter_code.upper()
            seq.append(code if code.isalpha() else "X")
            xyz.append([ca.pos.x, ca.pos.y, ca.pos.z])
    return "".join(seq), np.asarray(xyz, dtype=np.float64)


def needleman_wunsch(s1: str, s2: str, match=2, mismatch=-1, gap=-2) -> list[tuple[int, int]]:
    """Global alignment → list of aligned (i, j) index pairs (gap columns dropped)."""
    n, m = len(s1), len(s2)
    score = np.zeros((n + 1, m + 1), dtype=np.int64)
    score[:, 0] = np.arange(n + 1) * gap
    score[0, :] = np.arange(m + 1) * gap
    tb = np.zeros((n + 1, m + 1), dtype=np.int8)
    tb[1:, 0] = 1
    tb[0, 1:] = 2
    a = np.frombuffer(s1.encode(), dtype=np.uint8)
    b = np.frombuffer(s2.encode(), dtype=np.uint8)
    for i in range(1, n + 1):
        sub = np.where(a[i - 1] == b, match, mismatch)
        prev = score[i - 1]
        for j in range(1, m + 1):
            diag = prev[j - 1] + sub[j - 1]
            up = prev[j] + gap
            left = score[i, j - 1] + gap
            best, move = diag, 0
            if up > best:
                best, move = up, 1
            if left > best:
                best, move = left, 2
            score[i, j] = best
            tb[i, j] = move
    pairs: list[tuple[int, int]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i == 0:
            j -= 1
            continue
        if j == 0:
            i -= 1
            continue
        t = tb[i, j]
        if t == 0:
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif t == 1:
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    return pairs


def spectrum(Xg: np.ndarray, Xp: np.ndarray) -> tuple[list[int], list[float], float]:
    """dRMSD-by-separation curve + coherence length L (sep where curve crosses τ)."""
    n = len(Xg)
    Dg = np.sqrt(((Xg[:, None] - Xg[None]) ** 2).sum(-1))
    Dp = np.sqrt(((Xp[:, None] - Xp[None]) ** 2).sum(-1))
    iu = np.triu_indices(n, 1)
    sep = iu[1] - iu[0]
    err = (Dp - Dg)[iu]
    xs, ys = [], []
    for s in ANCHORS:
        if s >= n:
            break
        mask = (sep >= s * 0.7) & (sep < s * 1.5)
        if mask.sum() < 20:
            continue
        xs.append(s)
        ys.append(float(np.sqrt((err[mask] ** 2).mean())))
    L = float("nan")
    if ys:
        if ys[0] >= TAU:
            L = float(xs[0])
        elif ys[-1] < TAU:
            L = float(xs[-1])
        else:
            for k in range(1, len(ys)):
                if ys[k] >= TAU > ys[k - 1]:
                    f = (TAU - ys[k - 1]) / (ys[k] - ys[k - 1])
                    L = float(xs[k - 1] * (xs[k] / xs[k - 1]) ** f)
                    break
    return xs, ys, L


def _one(args) -> dict | None:
    """One target: NW once (pred seq identical across samples), dRMSD per sample."""
    pdb, pred_dir, seed, n_samples = args
    gt = GT_DIR / f"{pdb}.cif"
    if not gt.exists():
        return None
    try:
        gs, gx = ca_seq(str(gt))
    except Exception:
        return None
    P = aln = None
    Ls, curves = [], []
    for smp in range(n_samples):
        pr = Path(pred_dir) / pdb / f"seed-{seed}_sample-{smp}" / "model.cif"
        if not pr.exists():
            continue
        try:
            ps, px = ca_seq(str(pr))
            if aln is None:
                P = needleman_wunsch(gs, ps)
                gi = [a for a, _ in P]
                pi = [b for _, b in P]
                if len(gi) < 40:
                    return None
                aln = (gi, pi)
            gi, pi = aln
            xs, ys, L = spectrum(gx[gi], px[pi])
            if np.isfinite(L):
                Ls.append(L)
                curves.append((xs, ys))
        except Exception:
            continue
    if not Ls:
        return None
    # representative curve = the sample with median L
    med_idx = int(np.argsort(Ls)[len(Ls) // 2])
    xs, ys = curves[med_idx]
    return {"pdb": pdb, "L": float(np.median(Ls)), "L_samples": Ls,
            "xs": xs, "ys": ys, "n_aln": len(aln[0])}


def load_lddt(series, category, step) -> dict[str, float]:
    """Per-target mean lDDT over samples from the OST csv."""
    raw = BENCH / "runs" / series / "eval" / category / f"step_{step}" / "raw" / f"{category}_ost.csv"
    by = defaultdict(list)
    if raw.exists():
        for r in csv.DictReader(open(raw)):
            try:
                by[r["pdb_id"]].append(float(r["lddt"]))
            except (ValueError, KeyError):
                pass
    return {p: float(np.mean(v)) for p, v in by.items() if v}


def load_dssp(category) -> dict[str, tuple[int, str]]:
    f = BENCH / "inputs" / f"{category}_dssp.csv"
    out = {}
    if f.exists():
        for r in csv.DictReader(open(f)):
            try:
                out[r["pdb_id"]] = (int(r["n_res"]), r["fold_class"])
            except (ValueError, KeyError):
                pass
    return out


def size_bin(n_res: int) -> str:
    if n_res <= 0:
        return "unknown"  # DSSP failed → no length
    for name, lo, hi in SIZE_BINS:
        if lo <= n_res < hi:
            return name
    return "large"


def compute_step(series, category, step, seed, n_samples, workers) -> list[dict]:
    """Compute (or load cached) per-target coherence length for one step."""
    out_dir = BENCH / "runs" / series / "ss_analysis" / "coherence"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir / f"L_step{step}.csv"
    dssp = load_dssp(category)
    lddt = load_lddt(series, category, step)
    if cache.exists():
        rows = list(csv.DictReader(open(cache)))
        for r in rows:
            r["L"] = float(r["L"])
            r["n_res"] = int(r["n_res"])
            r["lddt"] = float(r["lddt"]) if r["lddt"] not in ("", "nan") else float("nan")
            r["ys"] = [float(v) for v in r["ys"].split(";")] if r["ys"] else []
            r["xs"] = [int(v) for v in r["xs"].split(";")] if r["xs"] else []
        return rows

    pred_dir = BENCH / "runs" / series / "pred" / category / f"step_{step}"
    pdbs = sorted(d.name for d in pred_dir.iterdir() if d.is_dir() and not d.name.startswith("_"))
    tasks = [(p, str(pred_dir), seed, n_samples) for p in pdbs]
    results = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed([ex.submit(_one, t) for t in tasks]):
            r = fut.result()
            if r:
                results.append(r)
    rows = []
    for r in results:
        n_res, fold = dssp.get(r["pdb"], (r["n_aln"], "unknown"))
        rows.append({"pdb": r["pdb"], "n_res": n_res, "fold_class": fold,
                     "lddt": lddt.get(r["pdb"], float("nan")), "L": r["L"],
                     "xs": ";".join(str(v) for v in r["xs"]),
                     "ys": ";".join(f"{v:.3f}" for v in r["ys"])})
    rows.sort(key=lambda r: r["pdb"])
    with open(cache, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pdb", "n_res", "fold_class", "lddt", "L", "xs", "ys"])
        w.writeheader()
        w.writerows(rows)
    # rehydrate parsed fields for the in-memory return
    for r in rows:
        r["ys"] = [float(v) for v in r["ys"].split(";")] if r["ys"] else []
        r["xs"] = [int(v) for v in r["xs"].split(";")] if r["xs"] else []
    print(f"[coherence] step {step}: computed {len(rows)}/{len(pdbs)} targets -> {cache}")
    return rows


def _corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    return float(np.corrcoef(x[m], y[m])[0, 1]) if m.sum() >= 3 else float("nan")


def report_step(series, category, step, rows):
    L = np.array([r["L"] for r in rows], float)
    nr = np.array([r["n_res"] for r in rows], float)
    ld = np.array([r["lddt"] for r in rows], float)
    fin = np.isfinite(L)
    finN = fin & (nr > 0)
    print(f"\n=== coherence length  {series} {category} step {step}  (n={fin.sum()}) ===")
    print(f"  L (sep@dRMSD={TAU:g}Å): median={np.median(L[fin]):.1f}  "
          f"range {L[fin].min():.0f}-{L[fin].max():.0f}")
    print(f"  corr(log L, lDDT)={_corr(np.log(L[fin]), ld[fin]):+.2f}   "
          f"corr(log L, length)={_corr(np.log(L[finN]), np.log(nr[finN])):+.2f}")
    print(f"  {'size bin':>10} | {'n':>4} | {'med L':>6} | {'med lDDT':>8} | {'med len':>7}")
    for name, lo, hi in SIZE_BINS:
        m = finN & (nr >= lo) & (nr < hi)
        if m.sum():
            print(f"  {name:>10} | {m.sum():>4} | {np.median(L[m]):>6.1f} | "
                  f"{np.nanmedian(ld[m]):>8.3f} | {np.median(nr[m]):>7.0f}")


def plot_step(series, category, step, rows, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fin = [r for r in rows if np.isfinite(r["L"])]
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
    # (a) dRMSD curves coloured by size bin
    colors = {"small": "tab:green", "medium": "tab:orange", "large": "tab:red"}
    for r in fin:
        if not r["ys"]:
            continue
        ax[0].plot(r["xs"], r["ys"], "-", lw=0.5, alpha=0.25,
                   color=colors.get(size_bin(r["n_res"]), "gray"))
    ax[0].axhline(TAU, ls="--", color="k", lw=1, label=f"τ={TAU:g}Å")
    ax[0].set_xscale("log"); ax[0].set_xlabel("sequence separation s (res)")
    ax[0].set_ylabel("dRMSD(s)  (Å)"); ax[0].set_ylim(0, 25)
    ax[0].set_title(f"{series} step {step}: dRMSD spectra (green=small / orange=med / red=large)")
    ax[0].legend(); ax[0].grid(alpha=0.3)
    # (b) L vs lDDT
    L = np.array([r["L"] for r in fin]); ld = np.array([r["lddt"] for r in fin])
    nr = np.array([max(r["n_res"], 1) for r in fin])
    sc = ax[1].scatter(L, ld, c=np.log10(nr), cmap="viridis", s=14, alpha=0.7)
    ax[1].set_xscale("log"); ax[1].set_xlabel("coherence length L (res)")
    ax[1].set_ylabel("lDDT"); ax[1].grid(alpha=0.3)
    ax[1].set_title(f"corr(log L, lDDT) = {_corr(np.log(L), ld):+.2f}")
    fig.colorbar(sc, ax=ax[1], label="log10 length")
    fig.tight_layout()
    png = out_dir / f"coherence_step{step}_{category}.png"
    fig.savefig(png, dpi=130); plt.close(fig)
    print(f"[coherence] per-step figure -> {png}")


def plot_trajectory(series, category, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    cache_dir = out_dir / "coherence"
    steps = sorted(int(re.search(r"L_step(\d+)\.csv", p).group(1))
                   for p in glob.glob(str(cache_dir / "L_step*.csv")))
    if not steps:
        return
    series_L = {name: [] for name, _, _ in SIZE_BINS}
    series_L["all"] = []
    for sp in steps:
        rows = list(csv.DictReader(open(cache_dir / f"L_step{sp}.csv")))
        L = np.array([float(r["L"]) for r in rows], float)
        nr = np.array([int(r["n_res"]) for r in rows], float)
        fin = np.isfinite(L)
        series_L["all"].append(np.median(L[fin]) if fin.any() else np.nan)
        for name, lo, hi in SIZE_BINS:
            m = fin & (nr >= lo) & (nr < hi)
            series_L[name].append(np.median(L[m]) if m.any() else np.nan)
    fig, ax = plt.subplots(figsize=(10, 6))
    style = {"all": ("k", 2.5), "small": ("tab:green", 1.5),
             "medium": ("tab:orange", 1.5), "large": ("tab:red", 1.5)}
    labels = {"all": "all", "small": "small (<200 res)",
              "medium": "medium (200–400 res)", "large": "large (≥400 res)"}
    for name in ["all", "small", "medium", "large"]:
        c, lw = style[name]
        ax.plot(steps, series_L[name], "-o", color=c, lw=lw, ms=4, label=labels[name])
    ax.set_xlabel("training step"); ax.set_ylabel(f"median coherence length L (res, sep@dRMSD={TAU:g}Å)")
    ax.set_yscale("log")
    ax.set_title(f"{series} {category} — coherence-length trajectory by chain size")
    ax.legend(); ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    png = out_dir / f"coherence_trajectory_{category}.png"
    fig.savefig(png, dpi=130); plt.close(fig)
    print(f"[coherence] trajectory figure ({len(steps)} steps) -> {png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", default="df3v1")
    ap.add_argument("--category", default="monomer_protein")
    ap.add_argument("--step", type=int, default=None, help="single step (compute + per-step figure)")
    ap.add_argument("--steps", default=None, help="'all' or comma list (trajectory)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-samples", type=int, default=5)
    ap.add_argument("--workers", type=int, default=48)
    args = ap.parse_args()

    out_dir = BENCH / "runs" / args.series / "ss_analysis"
    pred_base = BENCH / "runs" / args.series / "pred" / args.category

    if args.steps is not None:
        if args.steps == "all":
            steps = sorted(int(re.search(r"step_(\d+)", d).group(1))
                           for d in glob.glob(str(pred_base / "step_*")))
        else:
            steps = [int(s) for s in args.steps.split(",")]
        for sp in steps:
            rows = compute_step(args.series, args.category, sp, args.seed, args.n_samples, args.workers)
            report_step(args.series, args.category, sp, rows)
        plot_trajectory(args.series, args.category, out_dir)
        return

    if args.step is None:
        sys.exit("pass --step N (single) or --steps all (trajectory)")
    rows = compute_step(args.series, args.category, args.step, args.seed, args.n_samples, args.workers)
    report_step(args.series, args.category, args.step, rows)
    plot_step(args.series, args.category, args.step, rows, out_dir / "coherence")
    plot_trajectory(args.series, args.category, out_dir)


if __name__ == "__main__":
    main()
