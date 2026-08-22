"""USalign TM-score aggregator for inference outputs.

For each ckpt directory of the form ``step_{N}{suffix}/seed-S_sample-i/model.cif``,
runs USalign against a reference structure and reports per-ckpt TM-score
distribution (mean / median / max / per-sample).

TM-score is normalized by the **reference** length (USalign convention:
"normalized by length of Structure_2"). This is the score for ranking
predictions against a known structure.

Usage:
    uv run python scripts/usalign_tm.py \\
        --ref tmp/9szn/_ref.cif.gz \\
        --inference-dir tmp/9szn/df3-fapeS3-25e5-c10-g35 \\
        --pattern 'step_{step}_n30' \\
        --steps 5000 5200 5400 5500 \\
        --n-samples 30
"""

from __future__ import annotations

import argparse
import re
import statistics
import subprocess
from pathlib import Path

USALIGN = "/home/jovyan/.local/bin/USalign"
# USalign prints both Structure_1 and Structure_2 normalized scores;
# we want the one normalized by length of Structure_2 (= reference).
_TM_PATTERN = re.compile(r"TM-score=\s*([0-9.]+)\s*\(normalized by length of Structure_2")


def run_usalign(pred: Path, ref: Path) -> float | None:
    """Run USalign(pred, ref); return TM-score normalized by ref length."""
    if not pred.exists():
        return None
    try:
        out = subprocess.run(
            [USALIGN, str(pred), str(ref)],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        ).stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None
    m = _TM_PATTERN.search(out)
    return float(m.group(1)) if m else None


def measure_step(step_dir: Path, ref: Path, n_samples: int, seed: int = 42):
    tms = []
    for i in range(n_samples):
        cif = step_dir / f"seed-{seed}_sample-{i}" / "model.cif"
        tm = run_usalign(cif, ref)
        if tm is not None:
            tms.append(tm)
    if not tms:
        return None
    return {
        "n": len(tms),
        "mean": statistics.mean(tms),
        "median": statistics.median(tms),
        "max": max(tms),
        "min": min(tms),
        "tms": tms,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref", required=True, type=Path)
    ap.add_argument("--inference-dir", required=True, type=Path)
    ap.add_argument("--pattern", default="step_{step}_n30")
    ap.add_argument("--steps", type=int, nargs="+", required=True)
    ap.add_argument("--n-samples", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--per-sample", action="store_true", help="Print each sample's TM-score in addition to summary.")
    args = ap.parse_args()

    print(f"REF: {args.ref}")
    print(f"USalign: {USALIGN}")
    print()
    print(f"{'step':>6} | {'n':>3} | {'mean TM':>7} | {'median':>6} | {'max':>5} | {'min':>5}")
    print("-" * 60)
    for step in args.steps:
        sub = args.inference_dir / args.pattern.format(step=step)
        if not sub.exists():
            print(f"{step:>6} | (missing)")
            continue
        r = measure_step(sub, args.ref, args.n_samples, args.seed)
        if r is None:
            print(f"{step:>6} | (no usable samples)")
            continue
        print(
            f"{step:>6} | {r['n']:>3} | {r['mean']:>7.4f} | {r['median']:>6.4f} | {r['max']:>5.3f} | {r['min']:>5.3f}"
        )
        if args.per_sample:
            tms_str = " ".join(f"{t:.2f}" for t in r["tms"])
            print(f"           samples: {tms_str}")


if __name__ == "__main__":
    main()
