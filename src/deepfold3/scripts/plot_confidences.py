"""Plot PAE, PDE, and contact probability maps from a confidences JSON file.

Usage
-----
uv run python -m deepfold3.scripts.plot_confidences \\
    --input  output/prediction_confidences.json \\
    --output output/confidences.png
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import numpy as np


@click.command("plot")
@click.option(
    "--input",
    "input_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Input confidences JSON file",
)
@click.option(
    "--output", default="confidences.png", type=click.Path(path_type=Path), show_default=True, help="Output PNG path"
)
@click.option("--dpi", default=150, show_default=True, help="Output DPI")
def plot(input_path: Path, output: Path, dpi: int) -> None:
    """Plot PAE, PDE, and contact probabilities from a confidences JSON."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("matplotlib is required: pip install matplotlib") from e

    from deepfold3.model.config import ConfidenceConfig

    cfg = ConfidenceConfig()

    data = json.loads(input_path.read_text())
    summary = data.get("summary", {})
    N = summary.get("num_tokens", 0)

    # Determine which panels to show
    has_pae = "pae" in data
    has_pde = "pde" in data
    has_contact = "contact_probs" in data

    panels = []
    if has_pae:
        panels.append(("PAE", np.array(data["pae"]), 0.0, cfg.max_error_bin, "Greens", "Å"))
    if has_pde:
        panels.append(("PDE", np.array(data["pde"]), 0.0, cfg.max_error_bin, "Greens", "Å"))
    if has_contact:
        panels.append(("Contact Probability (d < 8 Å)", np.array(data["contact_probs"]), 0.0, 1.0, "Reds", ""))

    if not panels:
        click.echo("[plot_confidences] No plottable data found in JSON.")
        return

    n_panels = len(panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    for ax, (title, matrix, vmin, vmax, cmap, unit) in zip(axes, panels, strict=True):
        im = ax.imshow(matrix, origin="upper", cmap=cmap, aspect="equal", vmin=vmin, vmax=vmax)
        ax.set_title(f"{title}  |  N={N}", fontsize=11)
        ax.set_xlabel("Token index")
        ax.set_ylabel("Token index")
        label = unit if unit else None
        plt.colorbar(im, ax=ax, shrink=0.82, label=label)

    # Build suptitle from summary
    parts = []
    if "mean_plddt" in summary:
        parts.append(f"pLDDT={summary['mean_plddt']:.1f}")
    if "ptm" in summary:
        parts.append(f"pTM={summary['ptm']:.3f}")
    if summary.get("iptm") is not None:
        parts.append(f"ipTM={summary['iptm']:.3f}")
    if "ranking_score" in summary:
        parts.append(f"ranking={summary['ranking_score']:.3f}")
    if "average_pde" in summary:
        parts.append(f"gPDE={summary['average_pde']:.2f}")

    suptitle = f"DeepFold3 Confidences — {input_path.name}"
    if parts:
        suptitle += f"\n{', '.join(parts)}"
    plt.suptitle(suptitle, fontsize=12)
    plt.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output), dpi=dpi, bbox_inches="tight")
    click.echo(f"[plot_confidences] saved → {output.resolve()}")


if __name__ == "__main__":
    plot()
