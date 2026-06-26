"""DSSP secondary-structure CONFUSION MATRIX for FoldBench monomer targets.

Companion to ``ss_f1.py`` (which tracks per-class F1 over training). This pools
the per-residue 3-state (helix/strand/coil) and 8-state (DSSP H/E/G/I/T/S/B/P)
confusion across every (target, sample) at a given step, so you can see WHERE
the model errs -- e.g. predicting helix on coil/strand positions
(helix over-prediction), or missing strands (low strand recall).

Same engine as ss_f1.py: ``run_dssp`` + ``needleman_wunsch``, bench/ layout,
GT at ``bench/data/ground_truths/{pdb}.cif``. Per-sample aligned SS strings
cache to ``ss_strings.json`` next to model.cif (resumable; reused across runs).

Outputs (per step) under ``bench/runs/{series}/``:
  * console: raw counts + recall-normalized + precision-normalized 3-state
    confusion, per-class P/R/F1, helix-overprediction breakdown, and the
    per-target helix-precision tail (which targets drive FP-helix).
  * ``ss_confusion_{category}_step{N}.json`` : the pooled matrices + per-target.

Usage:
    python bench/scripts/ss_confusion.py --series df3v1 --category monomer_protein --step 100000
    python bench/scripts/ss_confusion.py --series df3v1 --step 100000 --top-bad 15
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BENCH = REPO / "bench"
GT_ROOT = BENCH / "data" / "ground_truths"
sys.path.insert(0, str(REPO / "scripts"))

from check_chirality import needleman_wunsch  # noqa: E402
from dssp_f1 import DSSP_8, run_dssp  # noqa: E402

SS3_MAP = {"H": "α", "G": "α", "I": "α", "P": "α",
           "E": "β", "B": "β",
           "T": "c", "S": "c", "-": "c"}
THREE = ("α", "β", "c")


def _aligned_ss(pred_cif: Path, ref_res):
    """Return (ref_ss, pred_ss) 8-state strings NW-aligned by sequence."""
    pred_res = run_dssp(pred_cif, cache=False)
    if not pred_res:
        return None
    pairs = needleman_wunsch("".join(a for a, _ in ref_res),
                             "".join(a for a, _ in pred_res),
                             include_mismatches=False)
    if not pairs:
        return None
    ref_ss = "".join(ref_res[i][1] for i, _ in pairs)
    pred_ss = "".join(pred_res[j][1] for _, j in pairs)
    return ref_ss, pred_ss


def target_job(args):
    try:
        return _target_job_inner(args)
    except Exception:
        return (args[1], None, None)


def _target_job_inner(args):
    step, pdb, n_samples, seed, pred_root_str = args
    pred_root = Path(pred_root_str)
    tdir = pred_root / pdb
    sample_dirs = [tdir / f"seed-{seed}_sample-{i}" for i in range(n_samples)]
    pending = [d for d in sample_dirs
               if (d / "model.cif").exists() and not (d / "ss_strings.json").exists()]
    ref_res = run_dssp(GT_ROOT / f"{pdb}.cif", cache=False) if pending else None
    c8 = defaultdict(int)
    c3 = defaultdict(int)
    n_used = 0
    for d in sample_dirs:
        cif, cache = d / "model.cif", d / "ss_strings.json"
        if not cif.exists():
            continue
        ss = None
        if cache.exists():
            try:
                j = json.load(open(cache))
                ss = (j["r"], j["p"])
            except (json.JSONDecodeError, OSError, KeyError):
                ss = None
        if ss is None:
            if ref_res is None:
                ref_res = run_dssp(GT_ROOT / f"{pdb}.cif", cache=False)
            res = _aligned_ss(cif, ref_res) if ref_res else None
            if res is None:
                continue
            ss = res
            try:
                cache.write_text(json.dumps({"r": ss[0], "p": ss[1]}))
            except OSError:
                pass
        rss, pss = ss
        for rc, pc in zip(rss, pss):
            c8[(rc, pc)] += 1
            c3[(SS3_MAP[rc], SS3_MAP[pc])] += 1
        n_used += 1
    if n_used == 0:
        return (pdb, None, None)
    # serialize defaultdicts as "ref|pred" -> count
    return (pdb,
            {f"{a}|{b}": v for (a, b), v in c8.items()},
            {f"{a}|{b}": v for (a, b), v in c3.items()})


def _norm_rows(conf, classes):
    out = {}
    for g in classes:
        s = sum(conf.get((g, p), 0) for p in classes) or 1
        out[g] = {p: conf.get((g, p), 0) / s for p in classes}
    return out


def _norm_cols(conf, classes):
    out = {}
    for p in classes:
        s = sum(conf.get((g, p), 0) for g in classes) or 1
        out[p] = {g: conf.get((g, p), 0) / s for g in classes}
    return out


def _prf(conf, classes):
    out = {}
    for c in classes:
        tp = conf.get((c, c), 0)
        fp = sum(conf.get((g, c), 0) for g in classes if g != c)
        fn = sum(conf.get((c, p), 0) for p in classes if p != c)
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        f1 = 2 * prec * rec / (prec + rec) if (prec == prec and rec == rec and prec + rec) else float("nan")
        out[c] = (prec, rec, f1)
    return out


def run(series, category, step, n_samples, seed, workers, top_bad):
    sdir = BENCH / "runs" / series / "pred" / category / f"step_{step}"
    if not sdir.exists():
        raise SystemExit(f"no predictions at {sdir}")
    jobs = [(step, t.name, n_samples, seed, str(sdir))
            for t in sorted(sdir.iterdir()) if t.is_dir() and not t.name.startswith("_")]
    print(f"[ss_conf] {len(jobs)} targets, {n_samples} samples, {workers} workers, step {step} ...")

    conf3 = defaultdict(int)
    conf8 = defaultdict(int)
    per_target = {}  # pdb -> helix precision (FP-helix signal)
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(target_job, j) for j in jobs]
        for fut in as_completed(futs):
            pdb, c8s, c3s = fut.result()
            if c3s is None:
                done += 1
                continue
            tc3 = defaultdict(int)
            for k, v in c3s.items():
                a, b = k.split("|"); conf3[(a, b)] += v; tc3[(a, b)] += v
            for k, v in c8s.items():
                a, b = k.split("|"); conf8[(a, b)] += v
            predA = sum(tc3[(g, "α")] for g in THREE)
            per_target[pdb] = {
                "Pα": (tc3[("α", "α")] / predA) if predA else float("nan"),
                "pred_α": predA, "n_res": sum(tc3.values()),
                "α_from_β": tc3[("β", "α")], "α_from_c": tc3[("c", "α")],
            }
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(jobs)}")

    n_res = sum(conf3.values())
    print(f"\n=== {series} {category} step {step} — pooled residues n={n_res:,} "
          f"({len(per_target)} targets scored) ===\n")

    # 3-state matrices
    print("RAW COUNTS (row=ref, col=pred)")
    print(f"{'ref/pred':>9} |" + "".join(f"{c:>10}" for c in THREE) + f"{'Σref':>11}")
    for g in THREE:
        row = [conf3.get((g, p), 0) for p in THREE]
        print(f"{g:>9} |" + "".join(f"{x:>10,}" for x in row) + f"{sum(row):>11,}")
    print(f"{'Σpred':>9} |" + "".join(f"{sum(conf3.get((g, p), 0) for g in THREE):>10,}" for p in THREE))

    rn = _norm_rows(conf3, THREE)
    print("\nRECALL view (row-normalized: of true X, predicted as…)")
    print(f"{'ref/pred':>9} |" + "".join(f"{c:>8}" for c in THREE))
    for g in THREE:
        print(f"{g:>9} |" + "".join(f"{rn[g][p]:>8.1%}" for p in THREE))

    cn = _norm_cols(conf3, THREE)
    print("\nPRECISION view (col-normalized: of predicted X, truly…)")
    print(f"{'truth/pred':>10} |" + "".join(f"{c:>8}" for c in THREE))
    for g in THREE:
        print(f"{g:>10} |" + "".join(f"{cn[p][g]:>8.1%}" for p in THREE))

    prf3 = _prf(conf3, THREE)
    print("\nper-class  P / R / F1  (3-state)")
    for c in THREE:
        p, r, f = prf3[c]
        print(f"  {c}:  P={p:.3f}  R={r:.3f}  F1={f:.3f}")

    # helix over-prediction summary
    predA = sum(conf3.get((g, "α"), 0) for g in THREE)
    fpA = predA - conf3.get(("α", "α"), 0)
    if predA:
        print(f"\nhelix over-prediction: of {predA:,} predicted-α, {fpA:,} wrong ({fpA/predA:.1%}) "
              f"→ from coil={conf3.get(('c','α'),0)/fpA:.0%}, from β={conf3.get(('β','α'),0)/fpA:.0%}")

    # per-target helix-precision tail (which targets drive FP-helix)
    tail = sorted((v for v in per_target.items() if v[1]["pred_α"] >= 20),
                  key=lambda kv: kv[1]["Pα"])[:top_bad]
    print(f"\nworst {len(tail)} targets by helix precision (pred-α≥20 res):")
    print(f"  {'target':>18} {'Pα':>5} {'pred_α':>7} {'α←β':>5} {'α←coil':>7}")
    for pdb, v in tail:
        print(f"  {pdb:>18} {v['Pα']:5.2f} {v['pred_α']:>7} {v['α_from_β']:>5} {v['α_from_c']:>7}")

    # 8-state per-class P/R/F1
    prf8 = _prf(conf8, DSSP_8)
    print("\nper-class P / R / F1  (8-state)")
    for c in DSSP_8:
        p, r, f = prf8[c]
        print(f"  {c}:  P={p:.3f}  R={r:.3f}  F1={f:.3f}")

    out = {
        "series": series, "category": category, "step": step, "n_res": n_res,
        "n_targets": len(per_target),
        "conf3": {f"{a}|{b}": conf3[(a, b)] for a in THREE for b in THREE},
        "conf8": {f"{a}|{b}": conf8.get((a, b), 0) for a in DSSP_8 for b in DSSP_8},
        "prf3": {c: prf3[c] for c in THREE},
        "per_target": per_target,
    }
    out_dir = BENCH / "runs" / series / "ss_analysis" / "confusion"
    out_dir.mkdir(parents=True, exist_ok=True)
    outp = out_dir / f"ss_confusion_{category}_step{step}.json"
    outp.write_text(json.dumps(out, indent=2))
    print(f"\n[ss_conf] -> {outp}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", default="df3v1")
    ap.add_argument("--category", default="monomer_protein")
    ap.add_argument("--step", type=int, default=100000)
    ap.add_argument("--n-samples", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--top-bad", type=int, default=15)
    args = ap.parse_args()
    run(args.series, args.category, args.step, args.n_samples, args.seed,
        args.workers, args.top_bad)


if __name__ == "__main__":
    main()
