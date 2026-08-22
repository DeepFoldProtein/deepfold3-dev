"""Per-target structure-accuracy snapshot + distogram-vs-GT MAE.

Companion to ``struct_accuracy.py`` (which tracks the *pooled* trajectory over
training). This renders the **per-target** distribution at a single checkpoint:
how each benchmark target does individually, plus a distogram-head sanity
metric. Two things are produced for ``--step``:

1. Per-target lDDT / TM / GDT-TS box plots over the ``n_samples`` diffusion
   samples, read from the OST caches written by ``tmp/_ost_lddt.py``::

       lDDT   ← lddt.json["lddt"]      (OST, no-stereochecks)
       GDT-TS ← lddt.json["gdt_ts"]    (OST --rigid-scores → oligo_gdtts)
       TM     ← tm_score.json["tm_ref"](OST --tm-score)

2. Per-target **distogram MAE**: mean |predicted − ground-truth| Cα–Cα distance.
   The inference ``distogram`` field (confidences.json) is the distogram-head
   *expected* Cα–Cα distance [N_tok, N_tok] in Å; it is a trunk-only output and
   therefore identical across diffusion samples, so it is read once (sample-0).
   The ground truth is the reference Cα–Cα distance matrix on residues paired by
   Needleman–Wunsch (sequence). The distogram saturates at the last bin
   (``last_break`` = 21.6875 Å), so MAE is reported over pairs whose GT distance
   is inside that representable range; a tighter contact MAE (GT < 8 Å) and the
   raw whole-matrix MAE are also tabulated. Cached to
   ``{step}_n30/distogram_mae.json`` per target.

Reuses the chain-A single-chain reference convention (11CF is a homotetramer →
chain A) and the NW pairing shared by ``gdt_ts.py`` / ``dssp_f1.py``.

Usage::

    uv run python scripts/per_target_metrics.py \\
        --targets-file tmp/bench/targets.csv --bench-root tmp/bench \\
        --arm-dir df3-25e5-c10-g35 --step 126000 --n-samples 30 \\
        --out-prefix tmp/bench/ss_analysis_full/nofape \\
        --out-csv tmp/bench/ss_analysis_full/nofape_pertarget.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
from check_chirality import needleman_wunsch  # noqa: E402
from gdt_ts import load_ca_with_seq  # noqa: E402

# Distogram last break (model/config.py DistogramConfig.last_break). The
# expected-distance read-out saturates here, so GT pairs beyond it are not
# representable and are excluded from the primary MAE.
LAST_BREAK = 21.6875

# Targets excluded from pooled aggregates upstream (engineered fusion, not a
# single-domain monomer). Shown here but flagged with '*' on the axis.
EXCLUDE = {"9OT9"}

METRICS = (
    ("lDDT", ("lddt.json", "lddt"), "tab:purple"),
    ("TM", ("tm_score.json", "tm_ref"), "tab:green"),
    ("GDT-TS", ("lddt.json", "gdt_ts"), "black"),
)


def _read(sd: Path, fname: str, key: str):
    p = sd / fname
    if not p.exists():
        return None
    try:
        v = json.loads(p.read_text()).get(key)
        v = float(v)
        return v if v == v else None
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        return None


def distogram_mae(step_dir: Path, ref_cif: Path, chain_id: str, seed: int) -> dict | None:
    """Per-target distogram-vs-GT Cα–Cα MAE (cached in step_dir).

    Reads the trunk distogram (sample-0, identical across samples), pairs it to
    the reference by NW alignment, and returns MAE over the representable range
    plus contact (<8 Å) and whole-matrix MAE.
    """
    cache = step_dir / "distogram_mae.json"
    if cache.exists():
        try:
            return json.loads(cache.read_text())
        except json.JSONDecodeError:
            pass
    sd = step_dir / f"seed-{seed}_sample-0"
    conf, model = sd / "confidences.json", sd / "model.cif"
    if not conf.exists() or not model.exists():
        return None
    d = json.loads(conf.read_text())
    # Trunk distogram E[d]: per-seed codec npz (current layout), with a fallback
    # to the legacy inline 'distogram' field.
    from deepfold3.io.confidences import load_logit_maps_from_json

    maps = load_logit_maps_from_json(conf)
    if "distogram" in maps:
        dg = np.asarray(maps["distogram"], dtype=np.float64)  # [N, N] expected Cα–Cα dist
    elif "distogram" in d:
        dg = np.asarray(d["distogram"], dtype=np.float64)
    else:
        return None
    _, pred_seq = load_ca_with_seq(model, chain_id)  # token order == model CA order
    ref_ca, ref_seq = load_ca_with_seq(ref_cif, chain_id)
    if dg.shape[0] != len(pred_seq) or not ref_seq:
        return None
    pairs = needleman_wunsch(ref_seq, pred_seq, include_mismatches=False)
    if len(pairs) < 2:
        return None
    ref_idx = np.array([i for i, _ in pairs])
    pred_idx = np.array([j for _, j in pairs])
    rc = ref_ca[ref_idx]
    gt = np.linalg.norm(rc[:, None, :] - rc[None, :, :], axis=-1)
    pr = dg[np.ix_(pred_idx, pred_idx)]
    iu = np.triu_indices(len(pairs), k=1)  # upper triangle, no diagonal
    gt_u, ae = gt[iu], np.abs(pr[iu] - gt[iu])
    in_range = gt_u < LAST_BREAK
    contact = gt_u < 8.0
    rec = {
        "n_res_aligned": len(pairs),
        "n_pairs": int(iu[0].size),
        "frac_inrange": round(float(in_range.mean()), 4),
        "mae_inrange": round(float(ae[in_range].mean()), 4) if in_range.any() else None,
        "mae_contact8": round(float(ae[contact].mean()), 4) if contact.any() else None,
        "mae_all": round(float(ae.mean()), 4),
        "version": 1,
    }
    cache.write_text(json.dumps(rec))
    return rec


def collect(targets, bench, arm, step, n_samples, seed):
    rows = []
    for pdb, cls in targets:
        ref = bench / pdb / "_ref.cif"
        if not ref.exists():
            ref = bench / pdb / "_ref.cif.gz"
        step_dir = bench / pdb / arm / f"step_{step}_n{n_samples}"
        per = {name: [] for name, _, _ in METRICS}
        for i in range(n_samples):
            sd = step_dir / f"seed-{seed}_sample-{i}"
            if not sd.exists():
                continue
            for name, (fname, key), _ in METRICS:
                v = _read(sd, fname, key)
                if v is not None:
                    per[name].append(v)
        dm = distogram_mae(step_dir, ref, "A", seed) if step_dir.exists() else None
        rows.append({"pdb": pdb, "cls": cls, "per": per, "dmae": dm})
    return rows


def _stats(xs):
    if not xs:
        return {"n": 0, "mean": None, "median": None, "std": None}
    a = np.array(xs, dtype=np.float64)
    return {
        "n": len(a),
        "mean": round(float(a.mean()), 4),
        "median": round(float(np.median(a)), 4),
        "std": round(float(a.std()), 4),
    }


def write_csv(rows, out_csv, step):
    cols = ["pdb", "cls", "step"]
    for name, _, _ in METRICS:
        cols += [f"{name}_n", f"{name}_mean", f"{name}_median", f"{name}_std"]
    cols += ["dmae_inrange", "dmae_contact8", "dmae_all", "frac_inrange", "n_res_aligned"]
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            row = {"pdb": r["pdb"], "cls": r["cls"], "step": step}
            for name, _, _ in METRICS:
                s = _stats(r["per"][name])
                row[f"{name}_n"], row[f"{name}_mean"] = s["n"], s["mean"]
                row[f"{name}_median"], row[f"{name}_std"] = s["median"], s["std"]
            dm = r["dmae"] or {}
            row["dmae_inrange"] = dm.get("mae_inrange")
            row["dmae_contact8"] = dm.get("mae_contact8")
            row["dmae_all"] = dm.get("mae_all")
            row["frac_inrange"] = dm.get("frac_inrange")
            row["n_res_aligned"] = dm.get("n_res_aligned")
            w.writerow(row)
    print(f"wrote {out_csv}")


def make_plot(rows, prefix, step, arm):
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    labels = [f"{r['pdb']}{'*' if r['pdb'] in EXCLUDE else ''}\n{r['cls']}" for r in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    for ax, (name, _, color) in zip(axes.flat[:3], METRICS, strict=True):
        data = [r["per"][name] for r in rows]
        bp = ax.boxplot(
            data,
            positions=x,
            widths=0.6,
            showfliers=True,
            patch_artist=True,
            flierprops={"marker": ".", "ms": 3, "alpha": 0.4},
            medianprops={"color": "black"},
        )
        for patch in bp["boxes"]:
            patch.set(facecolor=color, alpha=0.35)
        # overlay per-sample points
        for xi, vals in zip(x, data, strict=True):
            if vals:
                ax.scatter(np.full(len(vals), xi), vals, s=8, color=color, alpha=0.5, zorder=3)
        means = [np.mean(v) if v else np.nan for v in data]
        ax.plot(x, means, "D", color="red", ms=5, zorder=4, label="mean")
        ax.set_title(f"{name} — per target")
        ax.set_ylabel(name)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8, rotation=0)
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=8, loc="upper right")

    # distogram MAE bar (one value per target)
    ax = axes.flat[3]
    inrng = [(r["dmae"] or {}).get("mae_inrange") or np.nan for r in rows]
    cont = [(r["dmae"] or {}).get("mae_contact8") or np.nan for r in rows]
    w = 0.4
    ax.bar(x - w / 2, inrng, w, color="tab:blue", alpha=0.8, label="MAE (GT<21.7 Å)")
    ax.bar(x + w / 2, cont, w, color="tab:orange", alpha=0.8, label="MAE (GT<8 Å, contacts)")
    for xi, v in zip(x, inrng, strict=True):
        if v == v:
            ax.text(xi - w / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_title("Distogram-head expected dist vs GT — Cα–Cα MAE")
    ax.set_ylabel("MAE (Å)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=0)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8, loc="upper right")

    fig.suptitle(f"Per-target metrics — {arm}  step {step}   (* excluded from pooled aggregates)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(str(prefix) + "_pertarget.png", dpi=140)
    fig.savefig(str(prefix) + "_pertarget.pdf")
    plt.close(fig)
    print(f"wrote {prefix}_pertarget.png / .pdf")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets-file", type=Path, default=Path("tmp/bench/targets.csv"))
    ap.add_argument("--bench-root", type=Path, default=Path("tmp/bench"))
    ap.add_argument("--arm-dir", default="df3-25e5-c10-g35")
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--n-samples", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-prefix", type=Path, default=None)
    ap.add_argument("--out-csv", type=Path, default=None)
    args = ap.parse_args()

    targets = [(r["pdb"], r.get("class", "")) for r in csv.DictReader(open(args.targets_file))]
    rows = collect(targets, args.bench_root, args.arm_dir, args.step, args.n_samples, args.seed)

    print(f"\nPer-target metrics — {args.arm_dir} step {args.step}")
    print(
        f"{'pdb':>6} {'class':>12} | {'lDDT':>6} {'TM':>6} {'GDT-TS':>6} | "
        f"{'dMAE<21.7':>9} {'dMAE<8':>7} {'frac_rng':>8}"
    )
    print("-" * 78)
    for r in rows:
        s = {name: _stats(r["per"][name]) for name, _, _ in METRICS}
        dm = r["dmae"] or {}

        def f(v, w=6, p=3):
            return f"{v:>{w}.{p}f}" if isinstance(v, (int, float)) else f"{'--':>{w}}"

        print(
            f"{r['pdb']:>6} {r['cls']:>12} | "
            f"{f(s['lDDT']['mean'])} {f(s['TM']['mean'])} {f(s['GDT-TS']['mean'])} | "
            f"{f(dm.get('mae_inrange'), 9):>9} {f(dm.get('mae_contact8'), 7):>7} "
            f"{f(dm.get('frac_inrange'), 8):>8}"
        )

    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        write_csv(rows, args.out_csv, args.step)
    if args.out_prefix:
        args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
        make_plot(rows, args.out_prefix, args.step, args.arm_dir)


if __name__ == "__main__":
    main()
