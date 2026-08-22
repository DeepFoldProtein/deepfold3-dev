"""Plot SingleTrackProbe trajectories for the A/B/C ablation arms.

Reads ``single_track_probe.jsonl`` (and ``stdout.log`` as a fallback) from
each arm under a run root and writes three PNG pages:

* ``overview.png`` — losses + ``single_l2`` / ``pair_l2`` time-series, plus
  three scatter decision plots (loss vs RMS(s), single_l2 vs RMS(s),
  cos(s,u) vs L_diff).
* ``depth.png`` — depth-profiled RMS(s) / RMS(γ·u) / cos(s,u) at L0 / mid /
  last block.
* ``wakeup.png`` — wake-up dynamics: ``W₂(single_trans)`` / ``Wo(single_attn)``
  Frobenius norms at L0 vs last, α_su mean / max, and per-step
  ``Σ γ_k α_k`` aligned-gain budget.

Usage:
    python scripts/plot_sts_probe.py runs/sts_<DATE>             # one-shot
    python scripts/plot_sts_probe.py runs/sts_<DATE> --arms A,B  # subset
    python scripts/plot_sts_probe.py runs/sts_<DATE> --watch 60  # live monitor

The same script is safe to run while training is in flight — JSONL appends
are line-atomic, so a partial last line is just dropped.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_ARM_COLORS = {
    "A": "#1f77b4",
    "B": "#d62728",
    "C": "#2ca02c",
    "D": "#9467bd",
}


def load_arm(arm_dir: Path) -> tuple[list[dict], list[dict]]:
    """Read probe JSONL and stdout step rows for one arm.

    Returns ``(probe_rows, stdout_rows)``. Stdout rows are only used as a
    fallback for older runs that didn't thread ``loss_dict`` into the probe.
    """
    probe_rows: list[dict] = []
    probe_path = arm_dir / "single_track_probe.jsonl"
    if probe_path.exists():
        for line in probe_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                probe_rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    stdout_rows: list[dict] = []
    log_path = arm_dir / "stdout.log"
    if log_path.exists():
        for line in log_path.read_text().splitlines():
            if "step=" not in line or "loss=" not in line:
                continue
            try:
                fields = line.split(" | ")
                row = {}
                for f in fields:
                    if "=" not in f:
                        continue
                    k, _, v = f.partition("=")
                    k = k.strip().split()[-1]
                    try:
                        row[k] = float(v.rstrip("Gks"))
                    except ValueError:
                        pass
                if "step" in row:
                    stdout_rows.append(row)
            except Exception:
                continue
    return probe_rows, stdout_rows


def _series(rows: list[dict], key: str) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for r in rows:
        if key in r and r[key] is not None and "step" in r:
            xs.append(r["step"])
            ys.append(r[key])
    return np.asarray(xs), np.asarray(ys)


def _xy(rows: list[dict], xkey: str, ykey: str) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for r in rows:
        x = r.get(xkey)
        y = r.get(ykey)
        if x is not None and y is not None:
            xs.append(x)
            ys.append(y)
    return np.asarray(xs), np.asarray(ys)


def _plot_series(ax, arms, arm_data, key, title, scale="linear"):
    for arm in arms:
        probe_rows, stdout_rows = arm_data[arm]
        xs, ys = _series(probe_rows, key)
        # Stdout fallback (e.g. ``loss``→``total`` aliasing in old runs).
        if len(xs) == 0:
            xs, ys = _series(stdout_rows, key)
        if len(xs):
            ax.plot(xs, ys, label=arm, color=_ARM_COLORS.get(arm, "gray"), lw=1.4)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("step")
    ax.set_yscale(scale)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)


def _plot_scatter(ax, arms, arm_data, xkey, ykey, title, xscale="linear", yscale="linear"):
    for arm in arms:
        probe_rows, _ = arm_data[arm]
        xs, ys = _xy(probe_rows, xkey, ykey)
        if len(xs):
            ax.plot(xs, ys, label=arm, color=_ARM_COLORS.get(arm, "gray"), lw=1.0, marker="o", ms=2)
    ax.set_title(title, fontsize=10)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)


def _plot_overlay(ax, arms, arm_data, keys, labels, title, scale="linear"):
    """Plot multiple keys per arm overlaid on the same axis."""
    linestyles = ["-", "--", ":"]
    for arm in arms:
        probe_rows, _ = arm_data[arm]
        for ki, (key, lab) in enumerate(zip(keys, labels, strict=False)):
            xs, ys = _series(probe_rows, key)
            if len(xs):
                ax.plot(
                    xs,
                    ys,
                    label=f"{arm}:{lab}",
                    color=_ARM_COLORS.get(arm, "gray"),
                    lw=1.2,
                    ls=linestyles[ki % len(linestyles)],
                )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("step")
    ax.set_yscale(scale)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=7)


def render_overview(run_root: Path, arms: list[str], arm_data: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), constrained_layout=True)
    _plot_series(axes[0, 0], arms, arm_data, "total", "loss (total)", "log")
    _plot_series(axes[0, 1], arms, arm_data, "diffusion_mse", "diffusion_mse", "log")
    _plot_series(axes[0, 2], arms, arm_data, "distogram", "distogram", "linear")
    _plot_series(axes[1, 0], arms, arm_data, "smooth_lddt", "smooth_lddt", "linear")
    _plot_series(axes[1, 1], arms, arm_data, "single_l2", "single_l2", "linear")
    _plot_series(axes[1, 2], arms, arm_data, "pair_l2", "pair_l2", "linear")
    _plot_scatter(
        axes[2, 0],
        arms,
        arm_data,
        "rms_s_l_last",
        "single_l2",
        "single_l2 vs RMS(s_l_last)",
    )
    _plot_scatter(
        axes[2, 1],
        arms,
        arm_data,
        "rms_s_l_last",
        "diffusion_mse",
        "L_diff vs RMS(s_l_last)",
        yscale="log",
    )
    _plot_scatter(
        axes[2, 2],
        arms,
        arm_data,
        "diffusion_mse",
        "cos_su_mean",
        "cos(s,u)_mean vs L_diff",
        xscale="log",
    )
    fig.suptitle(f"{run_root.name} — overview", fontsize=12)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def render_depth(run_root: Path, arms: list[str], arm_data: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), constrained_layout=True)
    _plot_series(axes[0, 0], arms, arm_data, "rms_s_l0", "RMS(s) @ L=0")
    _plot_series(axes[0, 1], arms, arm_data, "rms_s_l_mid", "RMS(s) @ L=mid")
    _plot_series(axes[0, 2], arms, arm_data, "rms_s_l_last", "RMS(s) @ L=last")
    _plot_series(axes[1, 0], arms, arm_data, "rms_gu_l0", "RMS(γ·u) @ L=0")
    _plot_series(axes[1, 1], arms, arm_data, "rms_gu_l_mid", "RMS(γ·u) @ L=mid")
    _plot_series(axes[1, 2], arms, arm_data, "rms_gu_l_last", "RMS(γ·u) @ L=last")
    _plot_series(axes[2, 0], arms, arm_data, "cos_su_l0", "cos(s,u) @ L=0")
    _plot_series(axes[2, 1], arms, arm_data, "cos_su_l_mid", "cos(s,u) @ L=mid")
    _plot_series(axes[2, 2], arms, arm_data, "cos_su_l_last", "cos(s,u) @ L=last")
    fig.suptitle(f"{run_root.name} — depth profile", fontsize=12)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def render_wakeup(run_root: Path, arms: list[str], arm_data: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    _plot_overlay(
        axes[0, 0],
        arms,
        arm_data,
        ["w_single_trans_l2_l0", "w_single_trans_l2_l_last"],
        ["L0", "Llast"],
        "‖W₂(single_trans)‖ Frobenius",
    )
    _plot_overlay(
        axes[0, 1],
        arms,
        arm_data,
        ["w_single_attn_out_l0", "w_single_attn_out_l_last"],
        ["L0", "Llast"],
        "‖Wo(single_attn)‖ Frobenius",
    )
    _plot_overlay(
        axes[1, 0],
        arms,
        arm_data,
        ["alpha_su_mean", "alpha_su_max"],
        ["mean", "max"],
        "α_su = (s·u)/‖s‖²",
    )
    _plot_series(
        axes[1, 1],
        arms,
        arm_data,
        "sum_gamma_alpha",
        "Σ_k γ_k·α_k (per-step aligned-gain budget)",
    )
    fig.suptitle(f"{run_root.name} — wake-up & gain", fontsize=12)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def render_all(run_root: Path, arms: list[str], plots_dir: Path) -> None:
    # Skip arms whose subdir doesn't exist (e.g. D before it's launched).
    arms = [a for a in arms if (run_root / a).is_dir()]
    arm_data = {arm: load_arm(run_root / arm) for arm in arms}
    plots_dir.mkdir(parents=True, exist_ok=True)
    render_overview(run_root, arms, arm_data, plots_dir / "overview.png")
    render_depth(run_root, arms, arm_data, plots_dir / "depth.png")
    render_wakeup(run_root, arms, arm_data, plots_dir / "wakeup.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_root", type=Path, help="Run root containing A/, B/, C/ subdirs.")
    ap.add_argument(
        "--arms",
        default="A,B,C,D",
        help="Comma-separated arm names. Arms with no probe.jsonl are silently skipped.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output dir (default: <run_root>/plots/).",
    )
    ap.add_argument(
        "--watch",
        type=int,
        default=0,
        help="If > 0, re-render every N seconds (live monitoring).",
    )
    args = ap.parse_args()
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    plots_dir = args.out or (args.run_root / "plots")

    while True:
        try:
            render_all(args.run_root, arms, plots_dir)
            print(f"Rendered overview / depth / wakeup → {plots_dir}")
        except Exception as e:
            print(f"Render failed: {e!r}")
        if args.watch <= 0:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
