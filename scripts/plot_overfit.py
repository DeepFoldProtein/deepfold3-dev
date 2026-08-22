"""Overlay loss-curve plot from multiple training runs' train.log files.

Usage:
    python scripts/plot_overfit.py [--out PNG] [run_dir[=label[=style]]] ...

Each positional arg can be:
    RUN_DIR                     — auto-label from directory name, auto-pick color/linestyle
    RUN_DIR=LABEL               — custom label
    RUN_DIR=LABEL=STYLE         — STYLE like "C0-" or "C1--" (matplotlib shorthand)

With no runs, defaults to the four standard overfit directories:
    runs/overfit_1sample, runs/overfit_5samples,
    runs/overfit_1sample_fp32, runs/overfit_5samples_fp32
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

KEYS = (
    "loss",
    "distogram",
    "diffusion_mse",
    "smooth_lddt",
    "grad_norm",
    "single_l2",
    "pair_l2",
    "single_rms",
    "pair_rms",
    "lr",
)
LOG_SCALE = {
    "loss",
    "distogram",
    "diffusion_mse",
    "grad_norm",
    "single_l2",
    "pair_l2",
    "single_rms",
    "pair_rms",
    "lr",
}
TITLES = {
    "loss": "Total loss",
    "distogram": "Distogram loss",
    "diffusion_mse": "Diffusion MSE",
    "smooth_lddt": "Smooth LDDT",
    "grad_norm": "Gradient norm",
    "single_l2": "Single repr L2",
    "pair_l2": "Pair repr L2",
    "single_rms": "Single repr RMS",
    "pair_rms": "Pair repr RMS",
    "lr": "Learning rate",
}
FIELD_RE = re.compile(r"([a-zA-Z_][a-zA-Z0-9_]*)=([\d.\-eE]+|nan|inf|-inf)")
STEP_RE = re.compile(r"step=(\d+)")

DEFAULT_RUNS = [
    "runs/overfit_1sample",
    "runs/overfit_5samples",
    "runs/overfit_1sample_fp32",
    "runs/overfit_5samples_fp32",
]


def load(dirpath: Path) -> dict[str, np.ndarray] | None:
    """Parse the latest train.log under ``dirpath`` into per-key arrays.

    Accepts both layouts:
      - ``dirpath/<timestamp>/train.log`` (timestamped subdir)
      - ``dirpath/train.log`` (flat)
    """
    logs = sorted(dirpath.glob("*/train.log"))
    flat = dirpath / "train.log"
    if flat.exists():
        logs.append(flat)
    if not logs:
        return None
    data: dict[str, list[float]] = {k: [] for k in KEYS}
    data["step"] = []
    for line in logs[-1].read_text().splitlines():
        m = STEP_RE.search(line)
        if not m:
            continue
        fields = {k: v for k, v in FIELD_RE.findall(line)}
        data["step"].append(float(m.group(1)))
        for k in KEYS:
            v = fields.get(k, "nan")
            data[k].append(float(v) if v not in ("inf", "-inf") else float("nan"))
    if not data["step"]:
        return None
    return {k: np.asarray(v) for k, v in data.items()}


def ema(x: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    finite = np.isfinite(x)
    out = np.empty_like(x, dtype=np.float64)
    if not finite.any():
        out[:] = np.nan
        return out
    first = np.argmax(finite)
    out[:first] = np.nan
    out[first] = x[first]
    for i in range(first + 1, len(x)):
        if finite[i]:
            out[i] = alpha * x[i] + (1 - alpha) * out[i - 1]
        else:
            out[i] = out[i - 1]
    return out


def parse_run_spec(spec: str, idx: int) -> tuple[Path, str, str, str]:
    parts = spec.split("=", maxsplit=2)
    path = Path(parts[0])
    label = parts[1] if len(parts) > 1 and parts[1] else path.name
    if len(parts) == 3 and parts[2]:
        style = parts[2]
        color = "".join(c for c in style if c in "C0123456789") or f"C{idx % 10}"
        linestyle = "".join(c for c in style if c in "-:.") or "-"
        if linestyle == "..":
            linestyle = ":"
    else:
        color = f"C{idx % 10}"
        linestyle = "-" if idx % 2 else "--"
    return path, label, color, linestyle


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="runs/overfit_progress.png")
    ap.add_argument(
        "--title",
        default="Overfit loss curves",
        help="Figure title.",
    )
    ap.add_argument("runs", nargs="*", help="Run specs, see module docstring.")
    args = ap.parse_args(argv)

    specs = args.runs or DEFAULT_RUNS
    loaded = []
    for i, spec in enumerate(specs):
        path, label, color, linestyle = parse_run_spec(spec, i)
        data = load(path)
        if data is None:
            print(f"[skip] no train.log under {path}")
            continue
        loaded.append((label, color, linestyle, data))

    if not loaded:
        print("No runs with data found.")
        return

    # Only keep panels for keys that at least one run actually logged.
    active_keys = [k for k in KEYS if any(k in d and np.isfinite(d[k]).any() for _, _, _, d in loaded)]
    fig, axes = plt.subplots(len(active_keys), 1, figsize=(10, 2.6 * len(active_keys)), sharex=True)
    if len(active_keys) == 1:
        axes = [axes]
    for label, color, ls, d in loaded:
        for ax, k in zip(axes, active_keys, strict=False):
            if k not in d:
                continue
            y = d[k]
            ax.plot(d["step"], y, alpha=0.18, color=color, linewidth=0.6, linestyle=ls)
            ax.plot(d["step"], ema(y), color=color, linewidth=1.5, linestyle=ls, label=label)
    for ax, k in zip(axes, active_keys, strict=False):
        ax.set_ylabel(TITLES.get(k, k))
        ax.grid(True, which="both", alpha=0.3)
        if k in LOG_SCALE:
            ax.set_yscale("log")
    axes[0].set_title(args.title)
    axes[-1].set_xlabel("step")
    axes[0].legend(loc="best", fontsize=8, ncols=4)
    # Drop finite floor on log axes to avoid "no positive values" errors.
    for ax, k in zip(axes, active_keys, strict=False):
        if k in LOG_SCALE:
            vals = [c.get_ydata() for c in ax.lines]
            flat = np.concatenate([v[np.isfinite(v) & (v > 0)] for v in vals]) if vals else np.array([])
            if flat.size:
                ax.set_ylim(bottom=flat.min() * 0.5)
    plt.tight_layout()
    plt.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"Saved: {args.out}")

    for label, _, _, d in loaded:
        step = d["step"]
        lines = [
            f"{label:40s}",
            f"  steps {int(step[0])}..{int(step[-1])}  n={len(step)}",
        ]
        for k in ("loss", "distogram", "diffusion_mse", "grad_norm", "single_l2"):
            if k in d and np.isfinite(d[k]).any():
                v = d[k]
                finite = v[np.isfinite(v)]
                lines.append(f"  {k:14s} last={v[-1]:9.3f}  max={finite.max():9.2f}  median={np.median(finite):.3f}")
        print("\n".join(lines))


if __name__ == "__main__":
    main()
