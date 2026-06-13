"""GDT-TS (Global Distance Test - Total Score) for inference samples.

Computes GDT-TS between a predicted ``model.cif`` and a reference
mmCIF on backbone Cα atoms, after Kabsch superposition. Score per
sample::

    GDT-TS = mean_{c ∈ {1, 2, 4, 8}} ( |{i : d_i ≤ c Å}| / N )

where ``d_i`` is the post-superposition Cα-Cα distance for the i-th
aligned residue pair. Residue pairing is by Needleman-Wunsch (same
alignment used by ``check_chirality.py`` / ``dssp_f1.py``).

The reference for the score definition is::

    def gdt(p1, p2, mask, cutoffs):
        n = mask.sum(-1)
        d = ((p1 - p2) ** 2).sum(-1).sqrt()
        return mean(sum((d <= c) * mask / n for c in cutoffs) / len(cutoffs))

    def gdt_ts(p1, p2, mask):
        return gdt(p1, p2, mask, [1., 2., 4., 8.])

Usage (standalone CLI, mirrors ``usalign_tm.py``)::

    uv run python scripts/gdt_ts.py \\
        --ref tmp/bench/9SZN/_ref.cif.gz \\
        --inference-dir tmp/bench/9SZN/df3-fapeK5-25e5-c10-g35 \\
        --steps 1000 1500 2000 --n-samples 30 --per-sample
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import gemmi
import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
from check_chirality import needleman_wunsch  # noqa: E402

# Standard 20 amino acids — used to skip ligands / waters / non-standard residues.
_AA_3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

GDT_TS_CUTOFFS = (1.0, 2.0, 4.0, 8.0)


def _read_structure(path: Path) -> gemmi.Structure:
    s = str(path)
    if s.endswith(".cif.gz") or s.endswith(".cif"):
        doc = gemmi.cif.read(s)
        return gemmi.make_structure_from_block(doc.sole_block())
    return gemmi.read_structure(s)


def load_ca_with_seq(path: Path, chain_id: str = "A"):
    """Return ``(ca_xyz [N, 3] float64, seq_str N)`` for std-AA residues
    with a CA atom. Only the first chain matching ``chain_id`` is read.
    """
    st = _read_structure(path)
    coords: list[tuple[float, float, float]] = []
    seq_chars: list[str] = []
    for model in st:
        for chain in model:
            if chain.name != chain_id:
                continue
            for r in chain:
                name = r.name.strip()
                if name not in _AA_3TO1:
                    continue
                ca = r.find_atom("CA", "*")
                if ca is None:
                    continue
                coords.append((ca.pos.x, ca.pos.y, ca.pos.z))
                seq_chars.append(_AA_3TO1[name])
        break  # first model only
    if not coords:
        return np.zeros((0, 3), dtype=np.float64), ""
    return np.asarray(coords, dtype=np.float64), "".join(seq_chars)


def _kabsch_rt(pred: np.ndarray, ref: np.ndarray):
    """Return ``(R, t)`` such that ``pred @ R.T + t`` aligns to ``ref``.

    Pure proper rotation (determinant +1) — reflection branch flipped.
    Both inputs ``[N, 3]``.
    """
    p_c = pred.mean(axis=0)
    r_c = ref.mean(axis=0)
    H = (pred - p_c).T @ (ref - r_c)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d if d != 0 else 1.0])
    R = Vt.T @ D @ U.T
    t = r_c - R @ p_c
    return R, t


def kabsch_superimpose(pred: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Return ``pred`` rigidly superimposed onto ``ref`` (least-squares)."""
    R, t = _kabsch_rt(pred, ref)
    return pred @ R.T + t


def _gdt_score_at_cutoff(pred: np.ndarray, ref: np.ndarray,
                          cutoff: float, seed_len: int = 5,
                          stride: int = 3, max_iter: int = 20,
                          min_subset: int = 3) -> float:
    """LGA-style multi-seed best-fit subset score for one cutoff ``d``.

    For every length-``seed_len`` contiguous-residue seed (sliding with
    ``stride``), Kabsch-fit the seed, then iteratively grow the subset
    by adding residues whose superposed Cα–Cα distance ≤ ``cutoff``
    and refitting, until the subset stabilises. The seed that grows
    the largest converged subset wins; the score is its size / ``N``.

    Standard LGA uses fragment-based seeding (here: contiguous size-5
    seeds) precisely because a single Kabsch over all N residues is
    sensitive to outliers — a few wildly displaced residues drag the
    centroid and bias every iteration toward a poor alignment. Sliding
    seeds find the locally-correct subset even when most of the chain
    is misplaced. Stride 3 / seed 5 keeps runtime ~60 ms / sample
    (vs. ~1 ms for naive single-init) while matching LGA-style scores.
    """
    n = pred.shape[0]
    if n < seed_len:
        return 0.0
    best = 0
    for start in range(0, n - seed_len + 1, stride):
        mask = np.zeros(n, dtype=bool)
        mask[start:start + seed_len] = True
        for _ in range(max_iter):
            if mask.sum() < min_subset:
                break
            R, t = _kabsch_rt(pred[mask], ref[mask])
            d = np.linalg.norm(pred @ R.T + t - ref, axis=-1)
            new_mask = d <= cutoff
            if new_mask.sum() < min_subset:
                break
            if np.array_equal(new_mask, mask):
                break
            mask = new_mask
        if mask.sum() > best:
            best = int(mask.sum())
    return float(best / n)


def compute_gdt_ts(pred_ca: np.ndarray, ref_ca: np.ndarray,
                   cutoffs: tuple[float, ...] = GDT_TS_CUTOFFS) -> float:
    """LGA-style GDT-TS on already-paired Cα arrays. Both ``[N, 3]``.

    GDT-TS = mean over the four cutoffs ``{1, 2, 4, 8} Å`` of the
    fraction of residues within that cutoff under the *best-fit
    subset alignment* for each cutoff (iterative LGA convention).
    """
    if pred_ca.shape[0] != ref_ca.shape[0] or pred_ca.shape[0] == 0:
        return float("nan")
    scores = [_gdt_score_at_cutoff(pred_ca, ref_ca, c) for c in cutoffs]
    return float(np.mean(scores))


def gdt_ts_for_pair(pred_cif: Path, ref_cif: Path,
                    chain_id: str = "A") -> float | None:
    """End-to-end: load both, NW-align by sequence, superimpose, GDT-TS.

    Returns ``None`` if either file is unusable or no aligned residues.
    """
    pred_ca, pred_seq = load_ca_with_seq(pred_cif, chain_id)
    ref_ca, ref_seq = load_ca_with_seq(ref_cif, chain_id)
    if pred_seq == "" or ref_seq == "":
        return None
    pairs = needleman_wunsch(ref_seq, pred_seq, include_mismatches=False)
    if not pairs:
        return None
    ref_aligned = ref_ca[[i for i, _ in pairs]]
    pred_aligned = pred_ca[[j for _, j in pairs]]
    return compute_gdt_ts(pred_aligned, ref_aligned)


def measure_step(step_dir: Path, ref: Path, n_samples: int,
                 seed: int = 42, chain_id: str = "A"):
    rs: list[float] = []
    for i in range(n_samples):
        cif = step_dir / f"seed-{seed}_sample-{i}" / "model.cif"
        if not cif.exists():
            continue
        v = gdt_ts_for_pair(cif, ref, chain_id)
        if v is not None and v == v:
            rs.append(v)
    if not rs:
        return None
    return {
        "n": len(rs),
        "mean": statistics.mean(rs),
        "median": statistics.median(rs),
        "max": max(rs),
        "min": min(rs),
        "samples": rs,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref", required=True, type=Path)
    ap.add_argument("--inference-dir", required=True, type=Path)
    ap.add_argument("--pattern", default="step_{step}_n30")
    ap.add_argument("--steps", type=int, nargs="+", required=True)
    ap.add_argument("--n-samples", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--per-sample", action="store_true")
    ap.add_argument("--chain", default="A")
    args = ap.parse_args()

    print(f"REF: {args.ref}")
    print()
    print(f"  {'step':>5} | {'n':>3} | {'mean GDT-TS':>11} | {'median':>6} | "
          f"{'max':>5} | {'min':>5}")
    print("-" * 70)
    for step in args.steps:
        sub = args.inference_dir / args.pattern.format(step=step)
        if not sub.exists():
            print(f"  {step:>5} | (missing)")
            continue
        r = measure_step(sub, args.ref, args.n_samples, args.seed, args.chain)
        if r is None:
            print(f"  {step:>5} | (no usable samples)")
            continue
        print(f"  {step:>5} | {r['n']:>3} | {r['mean']:>11.4f} | "
              f"{r['median']:>6.4f} | {r['max']:>5.3f} | {r['min']:>5.3f}")
        if args.per_sample:
            samples_str = " ".join(f"{v:.3f}" for v in r["samples"])
            print(f"           samples: {samples_str}")


if __name__ == "__main__":
    main()
