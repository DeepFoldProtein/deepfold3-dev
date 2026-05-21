"""Chirality analysis for 11CF-style mirror-image diagnostics.

Two metrics per inference:
  (a) Global Kabsch chirality — det(V·U^T) of the rotation matrix from
      pred-vs-ref alignment. ``+1`` = proper, ``-1`` = mirror.
  (b) Helix-level Cα pseudo-dihedral sign match — for each annotated
      helix region (from ``_struct_conf`` of the reference mmCIF), the
      fraction of 4-Cα-windows whose pseudo-dihedral sign matches ref.

Critical correctness fix vs earlier ad-hoc scripts: residues are
aligned by ``label_seq_id``, not by sequential index. The reference
11CF mmCIF starts at label_seq=7 (auth=76); prediction starts at
label_seq=1. Aligning by sequential index off-sets ref↔pred by 6
residues and produces systematically wrong RMSDs / chirality calls.

Usage:
    uv run python scripts/check_chirality.py \\
        --ref tmp/11cf/_pdb_cache/11cf.cif.gz \\
        --inference-dir tmp/11cf/df3-fapeS3-25e5-c10-g35 \\
        --pattern 'step_{step}_n30' \\
        --steps 2000 2200 2400 2600 2800 3000 \\
        --n-samples 30 \\
        --helix-min-match 0.8

Each ``step_N`` directory must contain ``seed-42_sample-{i}/model.cif``
files (the layout produced by ``deepfold3 inference``).
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import gemmi
import numpy as np

STD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}


def read_structure(path: Path) -> gemmi.Structure:
    """Load mmCIF (.cif or .cif.gz) or PDB into a gemmi structure."""
    s = str(path)
    if s.endswith(".cif.gz") or s.endswith(".cif"):
        doc = gemmi.cif.read(s)
        return gemmi.make_structure_from_block(doc.sole_block())
    return gemmi.read_structure(s)


def get_ca_with_label(path: Path, chain_id: str = "A") -> list[tuple[int, np.ndarray]]:
    """Return list of ``(label_seq_id, np.array([x, y, z]))`` for std AA CA atoms."""
    st = read_structure(path)
    out: list[tuple[int, np.ndarray]] = []
    for model in st:
        for chain in model:
            if chain.name != chain_id:
                continue
            for r in chain:
                if r.name.strip() not in STD_AA:
                    continue
                ca = r.find_atom("CA", "*")
                if ca is None:
                    continue
                out.append((r.label_seq, np.array([ca.pos.x, ca.pos.y, ca.pos.z])))
        break
    return out


def parse_helix_annotations(ref_path: Path) -> list[tuple[int, int]]:
    """Read ``_struct_conf`` from the reference mmCIF for HELX_P regions.

    Returns deduplicated list of ``(label_seq_begin, label_seq_end)``.
    """
    doc = gemmi.cif.read(str(ref_path))
    block = doc.sole_block()
    table = block.find("_struct_conf.", ["beg_label_seq_id", "end_label_seq_id", "conf_type_id"])
    helices: set[tuple[int, int]] = set()
    if table:
        for row in table:
            try:
                start, end, conf = int(row[0]), int(row[1]), row[2]
            except ValueError:
                continue
            if conf.startswith("HELX"):
                helices.add((start, end))
    return sorted(helices)


def kabsch_det(p: np.ndarray, q: np.ndarray) -> tuple[float, float, float]:
    """Return ``(det(V·U^T), RMSD_proper, RMSD_improper)``.

    ``det = +1`` means no reflection was absorbed (proper rotation);
    ``det = -1`` means the Kabsch alignment had to absorb a reflection,
    i.e. ``q`` is in the global mirror chirality of ``p``.
    """
    pc = p - p.mean(0)
    qc = q - q.mean(0)
    h = qc.T @ pc
    u, _, vt = np.linalg.svd(h)
    det_VUT = float(np.linalg.det(vt.T @ u.T))
    diag = np.eye(3)
    diag[2, 2] = np.sign(det_VUT) if det_VUT != 0 else 1.0
    r_proper = vt.T @ diag @ u.T
    rmsd_p = float(np.sqrt(((p - (qc @ r_proper.T + p.mean(0))) ** 2).sum() / len(p)))
    r_improper = vt.T @ u.T
    rmsd_i = float(np.sqrt(((p - (qc @ r_improper.T + p.mean(0))) ** 2).sum() / len(p)))
    return det_VUT, rmsd_p, rmsd_i


def ca_pseudo_dihedral(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> float:
    """Signed Cα-Cα-Cα-Cα pseudo-dihedral in degrees."""
    b1, b2, b3 = p2 - p1, p3 - p2, p4 - p3
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    m1 = np.cross(n1, b2 / np.linalg.norm(b2))
    return float(np.degrees(np.arctan2(np.dot(m1, n2), np.dot(n1, n2))))


def compute_helix_signs(coord_map: dict[int, np.ndarray], helices: list[tuple[int, int]]):
    """Per-helix dihedral windows: ``{(start, end): [(i, sign), ...]}``."""
    out = {}
    for start, end in helices:
        wins = []
        for i in range(start, end - 2):
            if all(j in coord_map for j in (i, i + 1, i + 2, i + 3)):
                d = ca_pseudo_dihedral(
                    coord_map[i], coord_map[i + 1], coord_map[i + 2], coord_map[i + 3]
                )
                wins.append((i, d))
        out[(start, end)] = wins
    return out


def measure_step(step_dir: Path, ref_map: dict[int, np.ndarray],
                 ref_helix_signs, helix_min_match: float, n_samples: int):
    """Aggregate stats for one ``step_N`` directory (n_samples inferences)."""
    proper = 0
    rmsds_p: list[float] = []
    rmsds_m: list[float] = []
    helix_avg_match: list[float] = []
    full_consistent = 0
    for i in range(n_samples):
        cif = step_dir / f"seed-42_sample-{i}" / "model.cif"
        if not cif.exists():
            continue
        pred = get_ca_with_label(cif)
        pred_map = {ls: xyz for ls, xyz in pred}
        keys = sorted(k for k in ref_map if k in pred_map)
        if not keys:
            continue
        ra = np.array([ref_map[k] for k in keys])
        pa = np.array([pred_map[k] for k in keys])
        det, rp, ri = kabsch_det(ra, pa)
        if det > 0:
            proper += 1
            rmsds_p.append(rp)
        else:
            rmsds_m.append(ri)
        # Per-helix sign match against ref
        matches_per_helix: list[float] = []
        for (_start, _end), ref_wins in ref_helix_signs.items():
            sample_match = []
            for ii, ref_d in ref_wins:
                if all(j in pred_map for j in (ii, ii + 1, ii + 2, ii + 3)):
                    pd = ca_pseudo_dihedral(
                        pred_map[ii], pred_map[ii + 1], pred_map[ii + 2], pred_map[ii + 3]
                    )
                    sample_match.append(np.sign(pd) == np.sign(ref_d))
            if sample_match:
                matches_per_helix.append(sum(sample_match) / len(sample_match))
        if matches_per_helix:
            helix_avg_match.append(sum(matches_per_helix) / len(matches_per_helix))
            if all(m >= helix_min_match for m in matches_per_helix):
                full_consistent += 1
    n_total = proper + len(rmsds_m)
    if n_total == 0:
        return None
    p_proper = proper / n_total
    se = math.sqrt(p_proper * (1 - p_proper) / n_total)
    best = rmsds_p + rmsds_m
    return {
        "n": n_total,
        "n_proper": proper,
        "p_proper": p_proper,
        "ci_lo": max(0.0, p_proper - 1.96 * se),
        "ci_hi": min(1.0, p_proper + 1.96 * se),
        "mean_rmsd": sum(best) / len(best),
        "proper_rmsd": sum(rmsds_p) / len(rmsds_p) if rmsds_p else float("nan"),
        "mirror_rmsd": sum(rmsds_m) / len(rmsds_m) if rmsds_m else float("nan"),
        "helix_mean_match": (sum(helix_avg_match) / len(helix_avg_match)) if helix_avg_match else float("nan"),
        "consistent": full_consistent,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref", required=True, type=Path,
                    help="Reference structure (mmCIF or PDB).")
    ap.add_argument("--inference-dir", required=True, type=Path,
                    help="Directory containing step_N subdirs.")
    ap.add_argument("--pattern", default="step_{step}_n30",
                    help="Subdir name template; {step} is substituted. Default: step_{step}_n30")
    ap.add_argument("--steps", type=int, nargs="+", required=True,
                    help="List of training steps to analyse.")
    ap.add_argument("--n-samples", type=int, default=30,
                    help="Samples per ckpt (must match --num-samples used at inference time).")
    ap.add_argument("--chain", default="A", help="Chain to compare. Default: A")
    ap.add_argument("--helix-min-match", type=float, default=0.8,
                    help="Per-helix sign-match threshold for the 'fully consistent' tally.")
    args = ap.parse_args()

    ref_cas = get_ca_with_label(args.ref, args.chain)
    ref_map = {ls: xyz for ls, xyz in ref_cas}
    helices = parse_helix_annotations(args.ref)
    print(f"REF: {len(ref_cas)} CA atoms, label_seq {min(ref_map):d}-{max(ref_map):d}")
    print(f"REF helices: {helices}")
    ref_helix_signs = compute_helix_signs(ref_map, helices)
    for (s, e), wins in ref_helix_signs.items():
        sig = "".join("+" if d > 0 else "-" for _, d in wins)
        print(f"  helix {s:>3}-{e:>3} (n={len(wins):d}): {sig}")
    print()

    print(f"{'step':>6} | {'global':>10} | {'95% CI':>17} | {'mean':>5} | "
          f"{'proper_RMSD':>11} | {'mirror_RMSD':>11} | {'helix_match':>11} | "
          f"{'consistent (≥' + str(args.helix_min_match) + ')':>20}")
    print("-" * 120)
    for step in args.steps:
        sub = args.inference_dir / args.pattern.format(step=step)
        if not sub.exists():
            print(f"{step:>6} | (missing: {sub})")
            continue
        r = measure_step(sub, ref_map, ref_helix_signs, args.helix_min_match, args.n_samples)
        if r is None:
            print(f"{step:>6} | (no samples found)")
            continue
        print(f"{step:>6} | {r['n_proper']:>2}/{r['n']:<2}      | "
              f"[{r['ci_lo']*100:>4.1f}%, {r['ci_hi']*100:>4.1f}%]  | "
              f"{r['mean_rmsd']:>5.2f} | {r['proper_rmsd']:>11.2f} | "
              f"{r['mirror_rmsd']:>11.2f} | {r['helix_mean_match']*100:>10.1f}% | "
              f"{r['consistent']:>3}/{r['n']:<3}")


if __name__ == "__main__":
    main()
