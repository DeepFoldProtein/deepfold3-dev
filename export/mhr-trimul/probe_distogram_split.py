"""Split the r17-vs-r16 distogram deficit into sharpness and calibration.

``eval/distogram`` is a cross-entropy against one-hot distance bins and
``eval/distogram_entropy`` is ``H(p_theta)`` on the same batch, so::

    CE = H + (CE - H)          sharpness + calibration

separates "the distribution is broader" from "the confidence is wrong". The
script also regresses the gap on step to test whether it is still closing, and
reports eval spikes shared between runs -- a spike both runs hit at the same
step is an eval-draw artifact (the eval sampler is reseeded with
``set_epoch(effective_step)``), not an instability.

    python export/mhr-trimul/probe_distogram_split.py

See README section 11.
"""

from __future__ import annotations

import argparse
import re
import statistics as st
from pathlib import Path

EVAL_RE = re.compile(r"Eval at step (\d+): \{(.*)\}")
METRIC_RE = re.compile(r"'eval/([a-z_]+)': ([0-9.e+-]+)")


def eval_metrics(log: Path) -> dict[int, dict[str, float]]:
    """step -> the trainer's logged eval dict."""
    out: dict[int, dict[str, float]] = {}
    for line in log.read_text(errors="ignore").splitlines():
        m = EVAL_RE.search(line)
        if m:
            out[int(m.group(1))] = {k: float(v) for k, v in METRIC_RE.findall(m.group(2))}
    return out


def slope_per_1k(pts: list[tuple[int, float]]) -> float:
    """Least-squares slope of y on step, scaled to 'per 1000 steps'."""
    mx, my = st.mean(x for x, _ in pts), st.mean(y for _, y in pts)
    den = sum((x - mx) ** 2 for x, _ in pts)
    return 1000 * sum((x - mx) * (y - my) for x, y in pts) / den


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="r16")
    ap.add_argument("--arm", default="r17")
    ap.add_argument("--runs", default="runs", type=Path)
    ap.add_argument("--split-at", type=int, default=3400, help="step where the catch-up appears to end")
    a = ap.parse_args()

    arm = eval_metrics(a.runs / f"df3-stage0-{a.arm}" / "train.log")
    ref = eval_metrics(a.runs / f"df3-stage0-{a.ref}" / "train.log")
    shared = sorted(set(arm) & set(ref))

    print(f"CE = H + (CE - H).  d = {a.arm} minus {a.ref}.")
    print(f"  {'step':>6} {'dCE':>9} {'dH':>9} {'d(CE-H)':>9} {'dH/dCE':>8}")
    for s in shared:
        if s % 500 or not s:
            continue
        d_ce = arm[s]["distogram"] - ref[s]["distogram"]
        d_h = arm[s]["distogram_entropy"] - ref[s]["distogram_entropy"]
        frac = f"{100 * d_h / d_ce:7.0f}%" if abs(d_ce) > 1e-9 else "      --"
        print(f"  {s:6d} {d_ce:+9.4f} {d_h:+9.4f} {d_ce - d_h:+9.4f} {frac}")

    # A gap that shrinks is a lag; a gap that is flat while both curves still
    # move is a level difference. Those need different explanations.
    print(f"\nslopes per 1000 steps, before and after step {a.split_at}")
    print(f"  {'window':>13} {'metric':>18} {a.ref:>9} {a.arm:>9} {'gap':>9} {'gap mean':>10} {'sd':>7}")
    lo0 = min(s for s in shared if s >= 2500)
    for lo, hi in ((lo0, a.split_at), (a.split_at, max(shared))):
        for key in ("distogram", "distogram_entropy", "tm_true"):
            pts = [s for s in shared if lo <= s <= hi and s % 100 == 0]
            if len(pts) < 3:
                continue
            gap = [(s, arm[s][key] - ref[s][key]) for s in pts]
            print(
                f"  {f'{lo}-{hi}':>13} {key:>18} "
                f"{slope_per_1k([(s, ref[s][key]) for s in pts]):+9.4f} "
                f"{slope_per_1k([(s, arm[s][key]) for s in pts]):+9.4f} "
                f"{slope_per_1k(gap):+9.4f} "
                f"{st.mean(v for _, v in gap):+10.4f} {st.pstdev([v for _, v in gap]):7.4f}"
            )

    print("\neval spikes -- a spike at the same step in both runs is the eval draw, not the model")
    for s in shared:
        ma, mb = arm[s]["diffusion_mse"], ref[s]["diffusion_mse"]
        if max(ma, mb) > 0.5 and s > 2500:
            both = "  <- shared" if min(ma, mb) > 0.5 * max(ma, mb) else ""
            print(f"  step {s:6d}   {a.ref} {mb:9.4f}   {a.arm} {ma:9.4f}{both}")


if __name__ == "__main__":
    main()
