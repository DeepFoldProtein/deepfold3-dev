"""Compare r17 (no-TriAttn + MHR-TriMul) against r16 at iso-step, iso-wall-clock,
and on the one secondary-structure ratio that is flat across r16's whole run.

Writes compare_r16_r17.csv and prints the three tables in README section 10.

    python export/mhr-trimul/compare_r16_r17.py
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics as st
from pathlib import Path

SS_STATES = {
    "f1_8_E": "E strand",
    "f1_8_G": "G 3-10",
    "f1_8_I": "I pi",
    "f1_8_T": "T turn",
    "f1_8_S": "S bend",
    "f1_8_B": "B bridge",
    "f1_8_P": "P PPII",
}


def wall_clock(log: Path) -> dict[int, int]:
    """step -> seconds since the run started, from the trainer's `t=` field."""
    out: dict[int, int] = {}
    for line in log.read_text(errors="ignore").splitlines():
        m = re.search(r"step=(\d+) \|.*t=(\d+)s", line)
        if m:
            out.setdefault(int(m.group(1)), int(m.group(2)))
    return out


def by_step(path: Path) -> dict[int, dict[str, str]]:
    with path.open() as fh:
        return {int(r["step"]): r for r in csv.DictReader(fh)}


def interp(d: dict[float, float], x: float) -> float:
    """Linear interpolation over a dict keyed by a monotone axis."""
    ks = sorted(d)
    if x <= ks[0]:
        return d[ks[0]]
    if x >= ks[-1]:
        return d[ks[-1]]
    for a, b in zip(ks, ks[1:], strict=False):
        if a <= x <= b:
            return d[a] + (x - a) / (b - a) * (d[b] - d[a])
    raise AssertionError("unreachable")


def crossing(curve: dict[int, float], y: float) -> float | None:
    """First step at which `curve` reaches level y, linearly interpolated."""
    ks = sorted(curve)
    for a, b in zip(ks, ks[1:], strict=False):
        if curve[a] <= y <= curve[b]:
            return a + (y - curve[a]) / (curve[b] - curve[a]) * (b - a)
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="r16")
    ap.add_argument("--arm", default="r17")
    ap.add_argument("--runs", default="runs", type=Path)
    ap.add_argument("--bench", default="bench/runs", type=Path)
    ap.add_argument("--out", default="export/mhr-trimul/compare_r16_r17.csv", type=Path)
    a = ap.parse_args()

    t_ref = wall_clock(a.runs / f"df3-stage0-{a.ref}" / "train.log")
    t_arm = wall_clock(a.runs / f"df3-stage0-{a.arm}" / "train.log")
    b_ref = by_step(a.bench / f"df3v5{a.ref}" / "byclass_monomer_protein.csv")
    b_arm = by_step(a.bench / f"df3v5{a.arm}" / "byclass_monomer_protein.csv")
    s_ref = by_step(a.bench / f"df3v5{a.ref}" / "ss_analysis" / "ss_f1_monomer_protein.csv")
    s_arm = by_step(a.bench / f"df3v5{a.arm}" / "ss_analysis" / "ss_f1_monomer_protein.csv")

    rows: list[dict[str, object]] = []

    print("iso-step, FoldBench-small monomer_protein")
    print(
        f"  {'step':>6} {a.ref + ' TM':>9} {a.arm + ' TM':>9} {'dTM':>8} "
        f"{a.ref + ' lDDT':>10} {a.arm + ' lDDT':>10} {'dlDDT':>8}"
    )
    for s in sorted(set(b_ref) & set(b_arm)):
        tm_r, tm_a = float(b_ref[s]["tm_mean"]), float(b_arm[s]["tm_mean"])
        ld_r, ld_a = float(b_ref[s]["lddt_mean"]), float(b_arm[s]["lddt_mean"])
        print(f"  {s:6d} {tm_r:9.4f} {tm_a:9.4f} {tm_a - tm_r:+8.4f} {ld_r:10.4f} {ld_a:10.4f} {ld_a - ld_r:+8.4f}")
        rows.append(
            {"table": "iso_step", "step": s, "ref_tm": tm_r, "arm_tm": tm_a, "ref_lddt": ld_r, "arm_lddt": ld_a}
        )

    print("\nwall clock")
    for s in sorted(set(t_ref) & set(t_arm)):
        if s % 1000 or not s:
            continue
        print(
            f"  step {s:5d}  {a.ref} {t_ref[s] / s:6.2f} s/step   {a.arm} {t_arm[s] / s:6.2f} s/step   "
            f"{a.arm} faster by {100 * (1 - t_arm[s] / t_ref[s]):5.1f}%"
        )
        rows.append({"table": "wall_clock", "step": s, "ref_s_per_step": t_ref[s] / s, "arm_s_per_step": t_arm[s] / s})

    # Steps of credit: the arm's cheaper step buys extra optimizer steps per hour.
    inv_ref = {v: float(k) for k, v in t_ref.items()}
    print(f"\nsteps of credit ({a.arm} step N vs where {a.ref} was at the same wall clock)")
    for s in sorted(t_arm):
        if s % 1000 or not s:
            continue
        s_ref_at = interp(inv_ref, t_arm[s])
        print(f"  {a.arm} step {s:5d} at t={t_arm[s]:7d}s -> {a.ref} at step {s_ref_at:7.0f}  ({s - s_ref_at:+.0f})")
        rows.append({"table": "credit", "step": s, "wall_s": t_arm[s], "ref_step_at_wall": s_ref_at})

    print("\nquality-matched (interpolated -- weakest link when the curve is steep)")
    for label, key in (("TM", "tm_mean"), ("lDDT", "lddt_mean")):
        curve = {s: float(b_ref[s][key]) for s in b_ref}
        for s in sorted(b_arm):
            y = float(b_arm[s][key])
            s_x = crossing(curve, y)
            if s_x is None:
                continue
            t_x, t_a = interp(t_ref, s_x), t_arm[s]
            print(
                f"  {label:5s} {a.arm}@{s} = {y:.4f} -> {a.ref} at step {s_x:.0f} (t={t_x:.0f}s); "
                f"{a.arm} got there at t={t_a}s -> {100 * (1 - t_a / t_x):+.1f}% wall clock"
            )
            rows.append(
                {
                    "table": "quality_matched",
                    "metric": label,
                    "step": s,
                    "level": y,
                    "ref_step": s_x,
                    "ref_wall_s": t_x,
                    "arm_wall_s": t_a,
                    "arm_margin_pct": 100 * (1 - t_a / t_x),
                }
            )

    # The phase-invariant read. Ratios to helix F1 cancel the overall quality
    # level; only E/H is actually flat in the reference, so only E/H is a fair
    # comparison between two runs at different points on the curve.
    scored = [s for s in sorted(s_ref) if float(s_ref[s]["f1_8_H"]) > 0.1]
    print(f"\nX/H secondary-structure ratios ({len(scored)} {a.ref} checkpoints)")
    print(f"  {'state':<11} {'mean':>8} {'sd':>8} {'CV':>7} {'ref@3k':>8} {'arm@3k':>8} {'vs ref@3k':>10} {'z':>7}")
    for key, name in SS_STATES.items():
        vals = [float(s_ref[s][key]) / float(s_ref[s]["f1_8_H"]) for s in scored]
        m, sd = st.mean(vals), st.pstdev(vals)
        r_3k = float(s_ref[3000][key]) / float(s_ref[3000]["f1_8_H"])
        a_3k = float(s_arm[3000][key]) / float(s_arm[3000]["f1_8_H"])
        z = (a_3k - m) / sd if sd else float("nan")
        print(
            f"  {name:<11} {m:8.4f} {sd:8.4f} {100 * sd / m:6.1f}% {r_3k:8.4f} {a_3k:8.4f} "
            f"{100 * (a_3k - r_3k) / r_3k:+9.1f}% {z:7.1f}"
        )
        rows.append(
            {
                "table": "ss_ratio",
                "state": name,
                "ref_mean": m,
                "ref_sd": sd,
                "ref_cv_pct": 100 * sd / m,
                "ref_3k": r_3k,
                "arm_3k": a_3k,
                "rel_pct": 100 * (a_3k - r_3k) / r_3k,
                "z": z,
            }
        )
    print("\n  Only E/H is flat in the reference (CV 0.6%), so only E/H is safe to")
    print("  read across a phase offset. The rest are still climbing at step 3000.")

    fields = sorted({k for r in rows for k in r})
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
