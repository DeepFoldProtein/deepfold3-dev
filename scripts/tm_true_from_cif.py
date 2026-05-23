"""Compute training-time ``tm_true`` for inference samples against a reference.

Mirrors :func:`deepfold3.training.loss._tm_score_from_error` but operates on
two mmCIFs (predicted + reference) outside torch — no model required.

The training-time ``tm_true`` projects the rep atom (CA) of each token j
into the local (N, CA, C) frame of each token i, then computes a per-pair
error ``||pred_local - gt_local||``. Aggregation:

    d0          = 1.24 * (max(N_eff, 19) - 15)^(1/3) - 1.8
    tm_per_pair = 1 / (1 + (error / d0)^2)              # [N, N]
    per_align   = sum_j (tm_per_pair[i, j] * mask[i, j] / sum_j mask[i, j])
    tm_true     = max_i per_align[i]

Compared to USalign:
    * USalign performs sequence-or-3D alignment then global Kabsch; reports
      a single score normalized by reference length.
    * tm_true is sequence-aligned (same residue → same residue) and uses
      local frame projection; identical math to what shows up in train.log.
    * The two should be highly correlated when sequence + numbering match;
      this script lets you sanity-check that correlation for any pred-ref
      pair.

Usage:
    uv run python scripts/tm_true_from_cif.py \\
        --ref tmp/11cf/_pdb_cache/11cf.cif.gz \\
        --inference-dir tmp/11cf/df3-fapeS3-25e5-c10-g35 \\
        --pattern 'step_{step}_n30' \\
        --steps 8500 9500 9800 \\
        --n-samples 30
"""
from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import gemmi
import numpy as np

STD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}


def read_structure(path: Path) -> gemmi.Structure:
    s = str(path)
    if s.endswith(".cif.gz") or s.endswith(".cif"):
        doc = gemmi.cif.read(s)
        return gemmi.make_structure_from_block(doc.sole_block())
    return gemmi.read_structure(s)


def get_backbone_with_label(path: Path, chain_id: str = "A"):
    """Return ``{label_seq: (n_xyz, ca_xyz, c_xyz)}`` for std-AA residues
    where all three backbone atoms are present.
    """
    st = read_structure(path)
    out: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for model in st:
        for chain in model:
            if chain.name != chain_id:
                continue
            for r in chain:
                if r.name.strip() not in STD_AA:
                    continue
                n_atom = r.find_atom("N", "*")
                ca_atom = r.find_atom("CA", "*")
                c_atom = r.find_atom("C", "*")
                if n_atom is None or ca_atom is None or c_atom is None:
                    continue
                out[r.label_seq] = (
                    np.array([n_atom.pos.x, n_atom.pos.y, n_atom.pos.z]),
                    np.array([ca_atom.pos.x, ca_atom.pos.y, ca_atom.pos.z]),
                    np.array([c_atom.pos.x, c_atom.pos.y, c_atom.pos.z]),
                )
        break
    return out


def _normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / max(n, eps)


def frame_basis(n: np.ndarray, ca: np.ndarray, c: np.ndarray):
    """Gram-Schmidt of (N, CA, C) → (R_t [3, 3] as row-stacked e1/e2/e3, origin=CA)."""
    w1 = _normalize(n - ca)
    w2 = _normalize(c - ca)
    e1 = _normalize(w1 + w2)
    e2 = _normalize(w2 - w1)
    e3 = np.cross(e1, e2)
    R_t = np.stack([e1, e2, e3], axis=0)
    return R_t, ca


def _frame_angle_ok(n: np.ndarray, ca: np.ndarray, c: np.ndarray,
                    angle_threshold_deg: float = 25.0, eps: float = 1e-8) -> bool:
    u = n - ca
    v = c - ca
    cos_a = float(np.dot(u, v) / max(np.linalg.norm(u) * np.linalg.norm(v), eps))
    cos_max = np.cos(np.radians(angle_threshold_deg))
    cos_min = np.cos(np.radians(180.0 - angle_threshold_deg))
    return (cos_a < cos_max) and (cos_a > cos_min)


def compute_tm_true(pred_bb: dict, ref_bb: dict) -> tuple[float, int]:
    """Return ``(tm_true, n_used)`` for the residues common to both maps."""
    keys = sorted(k for k in ref_bb if k in pred_bb)
    if not keys:
        return float("nan"), 0
    N = len(keys)

    pred_frames = [frame_basis(*pred_bb[k]) for k in keys]
    gt_frames = [frame_basis(*ref_bb[k]) for k in keys]
    pred_ca = np.array([pred_bb[k][1] for k in keys])
    gt_ca = np.array([ref_bb[k][1] for k in keys])

    pred_angle_ok = np.array([
        _frame_angle_ok(*pred_bb[k]) for k in keys
    ])
    gt_angle_ok = np.array([
        _frame_angle_ok(*ref_bb[k]) for k in keys
    ])

    # error[i, j] = || R_i_pred^T (CA_j_pred - CA_i_pred) - R_i_gt^T (CA_j_gt - CA_i_gt) ||
    error = np.zeros((N, N))
    for i in range(N):
        R_p, b_p = pred_frames[i]
        R_g, b_g = gt_frames[i]
        rel_p = pred_ca - b_p
        rel_g = gt_ca - b_g
        local_p = rel_p @ R_p.T
        local_g = rel_g @ R_g.T
        error[i] = np.linalg.norm(local_p - local_g, axis=-1)

    mask = pred_angle_ok[:, None].astype(float) * gt_angle_ok[None, :].astype(float)
    # Mirror loss.py: pred frame angle gates i-axis; gt angle gates j-axis (well, atom side).
    # Strictly the loss has pred_angle on i and gt_angle on j atom, so we just use product.

    n_eff = max(N, 19)
    d0 = 1.24 * (n_eff - 15) ** (1.0 / 3) - 1.8
    tm_per_pair = 1.0 / (1.0 + (error / d0) ** 2)

    denom = mask.sum(axis=-1, keepdims=True).clip(min=1e-8)
    per_alignment = (tm_per_pair * (mask / denom)).sum(axis=-1)
    tm_true = float(per_alignment.max())
    return tm_true, N


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

    ref_bb = get_backbone_with_label(args.ref, args.chain)
    print(f"REF: {len(ref_bb)} residues with complete N/CA/C, label_seq "
          f"{min(ref_bb):d}-{max(ref_bb):d}")
    print()

    print(f"{'step':>6} | {'n':>3} | {'mean tm_true':>12} | {'median':>7} | {'max':>5} | {'min':>5}")
    print("-" * 70)
    for step in args.steps:
        sub = args.inference_dir / args.pattern.format(step=step)
        if not sub.exists():
            print(f"{step:>6} | (missing)")
            continue
        tms = []
        for i in range(args.n_samples):
            cif = sub / f"seed-{args.seed}_sample-{i}" / "model.cif"
            if not cif.exists():
                continue
            pred_bb = get_backbone_with_label(cif, args.chain)
            tm, _ = compute_tm_true(pred_bb, ref_bb)
            if not np.isnan(tm):
                tms.append(tm)
        if not tms:
            print(f"{step:>6} | (no usable samples)")
            continue
        print(f"{step:>6} | {len(tms):>3} | {statistics.mean(tms):>12.4f} | "
              f"{statistics.median(tms):>7.4f} | {max(tms):>5.3f} | {min(tms):>5.3f}")
        if args.per_sample:
            print("           samples: " + " ".join(f"{t:.2f}" for t in tms))


if __name__ == "__main__":
    main()
