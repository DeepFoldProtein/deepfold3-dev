"""Run all benchmarks and generate plots.

Usage:
    uv run python benchmarks/run_all.py
    uv run python benchmarks/run_all.py --dry-run
    uv run python benchmarks/run_all.py --only primitives,transition
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

BENCHMARKS = {
    "primitives": "benchmarks.bench_primitives",
    "transition": "benchmarks.bench_transition",
    "attention": "benchmarks.bench_attention",
    "pair_ops": "benchmarks.bench_pair_ops",
}


def run_benchmark(name: str, script: str, dry_run: bool, warmup: int, rep: int) -> bool:
    print(f"\n{'=' * 60}")
    print(f"Running: {name}")
    print(f"{'=' * 60}")

    cmd = [sys.executable, "-m", script]
    if dry_run:
        cmd.append("--dry-run")
    else:
        cmd.extend(["--warmup", str(warmup), "--rep", str(rep)])

    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    if result.returncode != 0:
        print(f"FAILED: {name} (exit code {result.returncode})")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Run all benchmarks")
    parser.add_argument("--dry-run", action="store_true", help="Smoke test with tiny shapes")
    parser.add_argument("--only", type=str, default=None, help="Comma-separated list of benchmarks to run")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--rep", type=int, default=200)
    parser.add_argument("--no-plot", action="store_true", help="Skip plotting")
    parser.add_argument("--no-verify", action="store_true", help="Skip numerical verification")
    args = parser.parse_args()

    # Run numerical verification first
    if not args.no_verify:
        print(f"\n{'=' * 60}")
        print("Running numerical verification")
        print(f"{'=' * 60}")
        verify_cmd = [sys.executable, "-m", "benchmarks.verify", "--verbose"]
        vr = subprocess.run(verify_cmd, cwd=Path(__file__).parent.parent)
        if vr.returncode != 0:
            print("Numerical verification FAILED — aborting benchmarks")
            sys.exit(1)

    benchmarks = BENCHMARKS
    if args.only:
        selected = {k.strip() for k in args.only.split(",")}
        benchmarks = {k: v for k, v in BENCHMARKS.items() if k in selected}
        if not benchmarks:
            print(f"No matching benchmarks. Available: {', '.join(BENCHMARKS)}")
            sys.exit(1)

    passed = 0
    failed = 0
    for name, script in benchmarks.items():
        ok = run_benchmark(name, script, args.dry_run, args.warmup, args.rep)
        if ok:
            passed += 1
        else:
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'=' * 60}")

    if not args.no_plot and passed > 0:
        print("\nGenerating plots...")
        subprocess.run([sys.executable, "-m", "benchmarks.plot_results"], cwd=Path(__file__).parent.parent)
        print("\nGenerating recommended PDF report...")
        subprocess.run([sys.executable, "-m", "benchmarks.plot_recommended"], cwd=Path(__file__).parent.parent)

    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
