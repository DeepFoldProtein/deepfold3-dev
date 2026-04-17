"""Plot overfit loss curves: baseline bf16 runs vs fp32 reruns."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
RUNS = [
    ("1-sample bf16 (baseline)", REPO / "runs/overfit_1sample", "C0", "--"),
    ("5-sample bf16 (baseline)", REPO / "runs/overfit_5samples", "C1", "--"),
    ("1-sample fp32 (live)", REPO / "runs/overfit_1sample_fp32", "C0", "-"),
    ("5-sample fp32 (live)", REPO / "runs/overfit_5samples_fp32", "C1", "-"),
]
KEYS = ["loss", "distogram", "diffusion_mse", "smooth_lddt", "grad_norm"]
LOG_SCALE = {"loss", "distogram", "diffusion_mse", "grad_norm"}
TITLES = {
    "loss": "Total loss",
    "distogram": "Distogram loss",
    "diffusion_mse": "Diffusion MSE",
    "smooth_lddt": "Smooth LDDT",
    "grad_norm": "Gradient norm",
}
PAT = re.compile(
    r"step=(\d+).*?loss=([\d.\-eE]+).*?diffusion_mse=([\d.\-eE]+)"
    r".*?distogram=([\d.\-eE]+).*?smooth_lddt=([\d.\-eE]+).*?grad_norm=([\d.\-eE]+)"
)


def load(dirpath: Path) -> dict[str, np.ndarray] | None:
    logs = sorted(dirpath.glob("*/train.log"))
    if not logs:
        return None
    rows = []
    for line in logs[-1].read_text().splitlines():
        m = PAT.search(line)
        if m:
            rows.append([float(x) for x in m.groups()])
    if not rows:
        return None
    a = np.asarray(rows)
    return {
        "step": a[:, 0],
        "loss": a[:, 1],
        "diffusion_mse": a[:, 2],
        "distogram": a[:, 3],
        "smooth_lddt": a[:, 4],
        "grad_norm": a[:, 5],
    }


def ema(x: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    out = np.empty_like(x)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1 - alpha) * out[i - 1]
    return out


def main(out_path: Path) -> None:
    fig, axes = plt.subplots(len(KEYS), 1, figsize=(10, 14), sharex=True)
    any_data = False
    for name, path, color, ls in RUNS:
        data = load(path)
        if data is None:
            continue
        any_data = True
        for ax, key in zip(axes, KEYS, strict=False):
            ax.plot(data["step"], data[key], alpha=0.18, color=color, linewidth=0.6, linestyle=ls)
            ax.plot(data["step"], ema(data[key]), color=color, linewidth=1.5, linestyle=ls, label=name)
    if not any_data:
        print("No runs found to plot", file=sys.stderr)
        return

    for ax, key in zip(axes, KEYS, strict=False):
        ax.set_ylabel(TITLES[key])
        ax.grid(True, which="both", alpha=0.3)
        if key in LOG_SCALE:
            ax.set_yscale("log")

    axes[0].set_title("Overfit: bf16 baseline vs fp32 precision")
    axes[-1].set_xlabel("step")
    axes[0].legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "runs/overfit_progress.png"
    main(out)
