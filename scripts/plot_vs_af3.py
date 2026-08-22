"""Plot a training run's spectral trajectory vs AF3 official weights as bound.

Reads `tmp/spectral/<prefix>_step_*.txt` (output of `spectral_quick.py` on
training checkpoints) and overlays the AF3 baseline reference from
`tmp/spectral/alphafold3_baseline.txt` (= `spectral_quick.py` on
`resources/alphafold3.safetensors`) as dashed horizontal lines.

Emits three figures:
  - bucket eff_rank trajectory (PF, diffusion, msa, atom_encoder, template, distogram)
  - PF per-submodule eff_rank trajectory (8 submods)
  - Bucket rms_p50 trajectory (weight magnitude)

Usage:
    # Pre-compute spectral (one-time):
    for STEP in 100 200 300; do
        uv run python scripts/spectral_quick.py runs/df3-af3r/checkpoint_${STEP}.pt \
            --device cuda --out-dir tmp/spectral > tmp/spectral/af3r_step_${STEP}.txt
    done
    # AF3 reference:
    uv run python scripts/spectral_quick.py resources/alphafold3.safetensors \
        --device cuda --out-dir tmp/spectral > tmp/spectral/alphafold3_baseline.txt

    # Plot:
    uv run python scripts/plot_vs_af3.py --prefix af3r --output tmp/spectral/af3r_vs_af3
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BUCKETS_PLOT_ORDER = (
    "pairformer",
    "diffusion",
    "msa",
    "atom_encoder",
    "template",
    "distogram",
)

PF_SUBMODS_PLOT_ORDER = (
    "single_transition",
    "single_attn",
    "pair_transition",
    "tri_mul_in",
    "tri_mul_out",
    "pair_attn_row",
    "pair_attn_col",
    "single_to_pair",
)


def _parse_bucket_table(text: str) -> dict[str, dict[str, float]]:
    """Parse the 'Per-bucket summary' table."""
    out: dict[str, dict[str, float]] = {}
    in_sec = False
    for line in text.splitlines():
        if "Per-bucket summary" in line:
            in_sec = True
            continue
        if not in_sec:
            continue
        if line.startswith("==="):
            break
        m = re.match(
            r"\s+(\w+)\s+\d+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)",
            line,
        )
        if m:
            b, rms, sr, er, cd = m.groups()
            out[b] = {
                "rms": float(rms),
                "stable_rank": float(sr),
                "eff_rank": float(er),
                "cond": float(cd),
            }
    return out


def _parse_pf_submod_table(text: str) -> dict[str, dict[str, float]]:
    """Parse the 'Per PF-submodule averages' table."""
    out: dict[str, dict[str, float]] = {}
    in_sec = False
    for line in text.splitlines():
        if "Per PF-submodule averages" in line:
            in_sec = True
            continue
        if not in_sec:
            continue
        if line.startswith("==="):
            break
        m = re.match(r"\s+(\w+)\s+\d+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", line)
        if m:
            sub, er, sr, rms = m.groups()
            out[sub] = {
                "eff_rank": float(er),
                "stable_rank": float(sr),
                "rms": float(rms),
            }
    return out


def load_run_series(src_dir: Path, prefix: str) -> list[tuple[int, dict, dict]]:
    """Load (step, bucket_table, submod_table) tuples for all matching files."""
    files = sorted(src_dir.glob(f"{prefix}_step_*.txt"))
    rows: list[tuple[int, dict, dict]] = []
    for f in files:
        m = re.search(r"step_0*(\d+)", f.name)
        if not m:
            continue
        text = f.read_text()
        bk = _parse_bucket_table(text)
        sb = _parse_pf_submod_table(text)
        if bk or sb:
            rows.append((int(m.group(1)), bk, sb))
    rows.sort(key=lambda r: r[0])
    return rows


def load_af3_reference(path: Path) -> tuple[dict, dict]:
    """Load AF3 baseline bucket + PF submodule tables."""
    text = path.read_text()
    return _parse_bucket_table(text), _parse_pf_submod_table(text)


def plot_bucket_trajectory(
    rows: list[tuple[int, dict, dict]],
    af3_bk: dict,
    af3_sb: dict,
    metric: str,
    title: str,
    out_path: Path,
    log_y: bool = False,
) -> None:
    """Plot per-bucket trajectory with AF3 dashed reference lines."""
    del af3_sb  # unused here
    fig, ax = plt.subplots(figsize=(10, 6))
    steps = [r[0] for r in rows]
    colors = plt.cm.tab10(np.linspace(0, 1, len(BUCKETS_PLOT_ORDER)))
    for bucket, col in zip(BUCKETS_PLOT_ORDER, colors, strict=False):
        ys = [r[1].get(bucket, {}).get(metric, np.nan) for r in rows]
        ax.plot(steps, ys, marker="o", markersize=6, color=col, label=bucket, linewidth=1.8)
        if bucket in af3_bk:
            y_ref = af3_bk[bucket][metric]
            ax.axhline(y_ref, color=col, linestyle="--", linewidth=1.2, alpha=0.7)
            # Annotate AF3 value at right side
            ax.text(
                steps[-1] + (steps[-1] - steps[0]) * 0.02 if len(steps) > 1 else 1.05,
                y_ref,
                f"AF3 {bucket} = {y_ref:.2f}",
                fontsize=7,
                color=col,
                va="center",
            )
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel(f"{metric} (p50 per bucket)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"saved {out_path}")
    plt.close(fig)


def plot_pf_submod_trajectory(
    rows: list[tuple[int, dict, dict]],
    af3_sb: dict,
    metric: str,
    title: str,
    out_path: Path,
) -> None:
    """Plot per-PF-submodule trajectory with AF3 dashed reference lines."""
    fig, ax = plt.subplots(figsize=(10, 6))
    steps = [r[0] for r in rows]
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(PF_SUBMODS_PLOT_ORDER), 10)))
    for sub, col in zip(PF_SUBMODS_PLOT_ORDER, colors, strict=False):
        ys = [r[2].get(sub, {}).get(metric, np.nan) for r in rows]
        ax.plot(steps, ys, marker="o", markersize=6, color=col, label=sub, linewidth=1.8)
        if sub in af3_sb:
            y_ref = af3_sb[sub][metric]
            ax.axhline(y_ref, color=col, linestyle="--", linewidth=1.2, alpha=0.7)
            ax.text(
                steps[-1] + (steps[-1] - steps[0]) * 0.02 if len(steps) > 1 else 1.05,
                y_ref,
                f"AF3 = {y_ref:.1f}",
                fontsize=7,
                color=col,
                va="center",
            )
    ax.set_xlabel("training step")
    ax.set_ylabel(f"PF {metric} (avg over 48 layers)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"saved {out_path}")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--prefix", default="af3r", help="File prefix in tmp/spectral/ (e.g. 'af3r' reads tmp/spectral/af3r_step_*.txt)"
    )
    p.add_argument("--src-dir", default="tmp/spectral", type=Path)
    p.add_argument("--af3-ref", default="tmp/spectral/alphafold3_baseline.txt", type=Path)
    p.add_argument("--output", default=None, help="Output path prefix (defaults to <src-dir>/<prefix>_vs_af3)")
    args = p.parse_args()

    rows = load_run_series(args.src_dir, args.prefix)
    if not rows:
        print(f"no spectral files found matching {args.src_dir}/{args.prefix}_step_*.txt")
        return
    print(f"Loaded {len(rows)} run snapshots: steps {[r[0] for r in rows]}")

    if not args.af3_ref.exists():
        print(f"AF3 reference not found at {args.af3_ref}")
        return
    af3_bk, af3_sb = load_af3_reference(args.af3_ref)
    print(f"AF3 reference loaded: {len(af3_bk)} buckets, {len(af3_sb)} PF submods")

    out_prefix = args.output or str(args.src_dir / f"{args.prefix}_vs_af3")

    plot_bucket_trajectory(
        rows,
        af3_bk,
        af3_sb,
        "eff_rank",
        f"{args.prefix} — bucket effective rank vs step (dashed = AF3 baseline)",
        Path(f"{out_prefix}_bucket_eff_rank.png"),
    )
    plot_bucket_trajectory(
        rows,
        af3_bk,
        af3_sb,
        "rms",
        f"{args.prefix} — bucket weight RMS p50 vs step (dashed = AF3 baseline)",
        Path(f"{out_prefix}_bucket_rms.png"),
        log_y=True,
    )
    plot_pf_submod_trajectory(
        rows,
        af3_sb,
        "eff_rank",
        f"{args.prefix} — PF submodule effective rank vs step (dashed = AF3 baseline)",
        Path(f"{out_prefix}_pf_submod_eff_rank.png"),
    )
    plot_pf_submod_trajectory(
        rows,
        af3_sb,
        "rms",
        f"{args.prefix} — PF submodule weight RMS vs step (dashed = AF3 baseline)",
        Path(f"{out_prefix}_pf_submod_rms.png"),
    )


if __name__ == "__main__":
    main()
