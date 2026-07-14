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

from deepfold3.scripts import cli


def plot_confidences_file(
    input_path: Path,
    output: Path,
    dpi: int = 150,
    logits_npz: Path | None = None,
    distogram_npz: Path | None = None,
) -> bool:
    """Render a PAE/PDE/contact-probability figure from a confidences JSON.

    The dense pairwise maps live in the compressed logit ``.npz`` files, not the
    JSON. ``logits_npz`` (per-sample PAE/PDE) and ``distogram_npz`` (per-seed)
    are auto-discovered next to ``input_path`` when not passed explicitly.

    Returns True if the figure was written, False if there was nothing plottable
    (e.g. confidence head disabled / no logit npz found).
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("matplotlib is required: pip install matplotlib") from e

    from deepfold3.io.confidences import discover_logit_npz, load_logit_maps
    from deepfold3.model.config import ConfidenceConfig, DistogramConfig

    cfg = ConfidenceConfig()
    dgram_cfg = DistogramConfig()

    data = json.loads(input_path.read_text())
    summary = data.get("summary", {})
    N = summary.get("num_tokens", 0)

    if logits_npz is None and distogram_npz is None:
        logits_npz, distogram_npz = discover_logit_npz(input_path)
    maps = load_logit_maps(logits_npz=logits_npz, distogram_npz=distogram_npz, cfg=cfg)

    panels = []
    if "pae" in maps:
        panels.append(("PAE", maps["pae"], 0.0, cfg.max_error_bin, "Greens", "Å"))
    if "pde" in maps:
        panels.append(("PDE", maps["pde"], 0.0, cfg.max_error_bin, "Greens", "Å"))
    if "distogram" in maps:
        panels.append(("Distogram (E[d])", maps["distogram"], 0.0, float(dgram_cfg.last_break), "viridis_r", "Å"))
    elif "contact_probs" in maps:
        panels.append(("Contact Probability (d < 8 Å)", maps["contact_probs"], 0.0, 1.0, "Reds", ""))

    if not panels:
        return False

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
    plt.close(fig)
    return True


@cli.command("plot")
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
    if plot_confidences_file(input_path, output, dpi=dpi):
        click.echo(f"[plot_confidences] saved → {output.resolve()}")
    else:
        click.echo("[plot_confidences] No plottable data found in JSON.")


if __name__ == "__main__":
    plot()
