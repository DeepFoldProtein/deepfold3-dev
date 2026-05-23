"""DSSP-based per-residue secondary structure F1 score for inference outputs.

For each predicted ``model.cif`` and the reference structure:

  1. Run ``mkdssp`` to get the 8-state secondary-structure label per residue.
  2. Extract ``(one_letter_AA, SS_label)`` tuples from the DSSP output.
  3. Align pred-vs-ref residue sequences via Needleman-Wunsch (the same
     alignment used in ``check_chirality.py``) — drops mismatches by
     default, keeps exact identity matches.
  4. For each aligned position, compare DSSP labels → per-class
     confusion matrix → macro-F1 over the 8 DSSP states.

DSSP 8-state classes::

    H = alpha-helix
    E = extended (beta-strand)
    G = 3_10-helix
    I = pi-helix
    T = turn
    S = bend
    B = beta-bridge
    P = polyproline II      (DSSP v4 only)
    - = coil / loop         (canonical blank, mapped to '-' here)

Outputs per-checkpoint:
  * macro-F1 (mean F1 over all observed classes)
  * accuracy (fraction matching label)
  * 3-state collapsed F1 (H+G+I → α, E+B → β, others → coil) — this is
    the classical SS prediction benchmark scale.

Usage:
    uv run python scripts/dssp_f1.py \\
        --ref tmp/9szn/_ref.cif.gz \\
        --inference-dir tmp/9szn/df3-fapeS3-25e5-c10-g35 \\
        --pattern 'step_{step}_n30' \\
        --steps 5000 7000 9000 10000 \\
        --n-samples 30
"""
from __future__ import annotations

import argparse
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

# We re-use the Needleman-Wunsch implementation from check_chirality so the
# alignment is identical to the chirality / TM scripts.
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
from check_chirality import needleman_wunsch  # noqa: E402

MKDSSP = "/usr/bin/mkdssp"

# 8-state DSSP labels. Anything else (e.g. blank/coil) collapses to '-'.
DSSP_8 = ("H", "E", "G", "I", "T", "S", "B", "P", "-")

# Collapse to a 3-state SS code (the classical benchmark scale).
THREE_STATE = {
    "H": "α", "G": "α", "I": "α", "P": "α",  # all helix-like
    "E": "β", "B": "β",                       # all strand-like
    "T": "c", "S": "c", "-": "c",             # turn / bend / coil
}


# mkdssp v4 enforces a strict PDB record sequence; this stub provides the
# minimum set of pre-coordinate records (HEADER → CRYST1 → ATOM) that
# satisfies its parser. Content is dummy — only presence matters.
_PDB_STUB_HEADER = (
    "HEADER    PROTEIN                                 01-JAN-26   XXXX\n"
    "COMPND    MOL_ID: 1;\n"
    "SOURCE    MOL_ID: 1;\n"
    "KEYWDS    PROTEIN STRUCTURE\n"
    "EXPDTA    X-RAY DIFFRACTION\n"
    "AUTHOR    DEEPFOLD3\n"
    "REMARK   2 RESOLUTION.    1.00 ANGSTROMS.\n"
    "CRYST1    1.000    1.000    1.000  90.00  90.00  90.00 P 1           1\n"
)


def run_dssp(cif: Path, cache: bool = True) -> list[tuple[str, str]]:
    """Run ``mkdssp`` and return ``[(aa_one_letter, ss_code)]`` per residue.

    If ``cache`` is True (default), the DSSP text output is written next
    to the input CIF as ``{cif.stem}.dssp`` and reused on subsequent
    calls. mkdssp is the slow step (~300 ms / sample); caching makes
    repeated analyses (e.g. re-plotting with different windows) instant.

    mkdssp v4 is strict about PDB header sections; gemmi-written PDBs
    lack several (COMPND, SOURCE, KEYWDS, EXPDTA, AUTHOR) which makes
    mkdssp silently emit zero residues. We prepend a tiny stub header
    before handing the file off — content of those records doesn't
    matter, just their presence.
    """
    cache_path = cif.with_suffix(".dssp")
    if cache and cache_path.exists() and cache_path.stat().st_size > 0:
        lines = cache_path.read_text().splitlines()
    else:
        import gemmi  # noqa: PLC0415
        with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w", delete=False) as tf:
            pdb_path = Path(tf.name)
        with tempfile.NamedTemporaryFile(suffix=".dssp", mode="w", delete=False) as tf:
            out_path = Path(tf.name)
        try:
            st = gemmi.read_structure(str(cif))
            st.write_pdb(str(pdb_path))
            body = pdb_path.read_text()
            keep_prefixes = ("ATOM", "HETATM", "TER", "END", "MODEL", "ENDMDL")
            body_atoms = "\n".join(
                line for line in body.splitlines() if line.startswith(keep_prefixes)
            ) + "\n"
            pdb_path.write_text(_PDB_STUB_HEADER + body_atoms)
            subprocess.run([MKDSSP, str(pdb_path), str(out_path)],
                           capture_output=True, text=True, check=False)
            dssp_text = out_path.read_text()
            lines = dssp_text.splitlines()
            if cache:
                # Atomic write to avoid half-files if interrupted.
                tmp = cache_path.with_suffix(".dssp.tmp")
                tmp.write_text(dssp_text)
                tmp.replace(cache_path)
        finally:
            pdb_path.unlink(missing_ok=True)
            out_path.unlink(missing_ok=True)
    # Find header line "  #  RESIDUE AA STRUCTURE …"; data starts the next line.
    start = next((i for i, line in enumerate(lines)
                  if line.startswith("  #  RESIDUE")), -1)
    if start < 0:
        return []
    residues = []
    for line in lines[start + 1:]:
        if len(line) < 17:
            continue
        if line[13:14] == "!":
            # Chain break marker in DSSP output.
            continue
        aa = line[13:14]
        code = line[16:17]
        if code not in DSSP_8:
            code = "-"
        residues.append((aa, code))
    return residues


def per_class_f1(pred: list[str], gt: list[str], classes: tuple[str, ...]
                 ) -> tuple[dict[str, float], float]:
    """Per-class F1 + macro-F1 over the listed classes."""
    f1s = {}
    for c in classes:
        tp = sum(1 for p, g in zip(pred, gt, strict=True) if p == c and g == c)
        fp = sum(1 for p, g in zip(pred, gt, strict=True) if p == c and g != c)
        fn = sum(1 for p, g in zip(pred, gt, strict=True) if p != c and g == c)
        if tp + fp == 0 and tp + fn == 0:
            f1s[c] = float("nan")
            continue
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s[c] = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    observed = [v for v in f1s.values() if v == v]  # drop NaN
    macro = statistics.mean(observed) if observed else float("nan")
    return f1s, macro


def measure_pair(pred_cif: Path, ref_cif: Path) -> dict | None:
    pred_res = run_dssp(pred_cif)
    ref_res = run_dssp(ref_cif)
    if not pred_res or not ref_res:
        return None
    pred_seq = "".join(aa for aa, _ in pred_res)
    ref_seq = "".join(aa for aa, _ in ref_res)
    pairs = needleman_wunsch(ref_seq, pred_seq, include_mismatches=False)
    if not pairs:
        return None
    ref_ss = [ref_res[i][1] for i, _ in pairs]
    pred_ss = [pred_res[j][1] for _, j in pairs]
    # 8-state
    _, f1_8 = per_class_f1(pred_ss, ref_ss, DSSP_8)
    # 3-state collapse
    pred_3 = [THREE_STATE[c] for c in pred_ss]
    ref_3 = [THREE_STATE[c] for c in ref_ss]
    _, f1_3 = per_class_f1(pred_3, ref_3, ("α", "β", "c"))
    acc = sum(1 for p, g in zip(pred_ss, ref_ss, strict=True) if p == g) / len(pred_ss)
    acc_3 = sum(1 for p, g in zip(pred_3, ref_3, strict=True) if p == g) / len(pred_3)
    return {
        "n_aligned": len(pairs),
        "f1_8": f1_8,
        "f1_3": f1_3,
        "acc_8": acc,
        "acc_3": acc_3,
    }


def measure_step(step_dir: Path, ref_cif: Path, n_samples: int, seed: int = 42):
    rs = []
    for i in range(n_samples):
        cif = step_dir / f"seed-{seed}_sample-{i}" / "model.cif"
        if not cif.exists():
            continue
        r = measure_pair(cif, ref_cif)
        if r is not None:
            rs.append(r)
    if not rs:
        return None
    return {
        "n": len(rs),
        "f1_8_mean": statistics.mean(r["f1_8"] for r in rs),
        "f1_8_med": statistics.median(r["f1_8"] for r in rs),
        "f1_3_mean": statistics.mean(r["f1_3"] for r in rs),
        "f1_3_med": statistics.median(r["f1_3"] for r in rs),
        "acc_8_mean": statistics.mean(r["acc_8"] for r in rs),
        "acc_3_mean": statistics.mean(r["acc_3"] for r in rs),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref", required=True, type=Path)
    ap.add_argument("--inference-dir", required=True, type=Path)
    ap.add_argument("--pattern", default="step_{step}_n30")
    ap.add_argument("--steps", type=int, nargs="+", required=True)
    ap.add_argument("--n-samples", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"REF: {args.ref}")
    print(f"mkdssp: {MKDSSP}")
    print()
    print(f"{'step':>6} | {'n':>3} | "
          f"{'F1(8st) mean':>12} | {'med':>5} | "
          f"{'F1(3st) mean':>12} | {'med':>5} | "
          f"{'acc(8)':>6} | {'acc(3)':>6}")
    print("-" * 90)
    for step in args.steps:
        sub = args.inference_dir / args.pattern.format(step=step)
        if not sub.exists():
            print(f"{step:>6} | (missing)")
            continue
        r = measure_step(sub, args.ref, args.n_samples, args.seed)
        if r is None:
            print(f"{step:>6} | (no usable samples)")
            continue
        print(f"{step:>6} | {r['n']:>3} | "
              f"{r['f1_8_mean']:>12.4f} | {r['f1_8_med']:>5.3f} | "
              f"{r['f1_3_mean']:>12.4f} | {r['f1_3_med']:>5.3f} | "
              f"{r['acc_8_mean']:>6.3f} | {r['acc_3_mean']:>6.3f}")


if __name__ == "__main__":
    main()
