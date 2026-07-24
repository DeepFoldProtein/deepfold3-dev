"""Plot PAE, PDE, distogram, and contact-probability maps from a prediction.

The dense pairwise maps live in the compressed logit ``.npz`` files written by
``deepfold3 inference``; ``confidences.json`` only carries the scalar summary
and per-token metadata. Any of those artifacts — or the directory holding them —
can be handed to this command, and the companions are discovered around it.

Usage
-----
deepfold3 plot --input output/seed-42_sample-0/                    # whole sample dir
deepfold3 plot --input output/seed-42_sample-0/logits.npz          # PAE + PDE (+ distogram)
deepfold3 plot --input output/seed-42_distogram.npz                # trunk distogram only
deepfold3 plot --input output/pred_confidences.json --panels pae,contact
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import click

from deepfold3.scripts import cli

# Panels that can be drawn, in --panels selection order. "pae"/"pde" come from
# the per-sample logits npz; "distogram"/"contact" from the per-seed one.
PANEL_KEYS = ("pae", "pde", "distogram", "contact")

# Selected when --panels auto: the distogram supersedes contact probabilities
# (it is the same head, unthresholded) and is dropped when unavailable.
_AUTO_PANELS = ("pae", "pde", "distogram")


def _parse_panels(panels: str | Sequence[str]) -> tuple[str, ...] | None:
    """Normalize the --panels selection; None means "auto"."""
    if isinstance(panels, str):
        if panels.strip().lower() in ("auto", ""):
            return None
        wanted = [p.strip().lower() for p in panels.split(",") if p.strip()]
    else:
        wanted = [str(p).strip().lower() for p in panels]
    unknown = [p for p in wanted if p not in PANEL_KEYS]
    if unknown:
        raise ValueError(f"unknown panel(s) {unknown}; choose from {list(PANEL_KEYS)}")
    return tuple(dict.fromkeys(wanted))


def _summary_line(summary: dict) -> str:
    """One-line scalar digest from a confidences.json summary block."""
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
    return ", ".join(parts)


def _provenance_line(attrs: dict) -> str:
    """Fallback digest for npz-only inputs: seed / sample from the codec attrs."""
    parts = []
    if attrs.get("seed") is not None:
        parts.append(f"seed={int(attrs['seed'])}")
    if attrs.get("sample") is not None:
        parts.append(f"sample={int(attrs['sample'])}")
    return ", ".join(parts)


def plot_confidences_file(
    input_path: str | Path | Sequence[str | Path],
    output: str | Path,
    dpi: int = 150,
    logits_npz: str | Path | None = None,
    distogram_npz: str | Path | None = None,
    panels: str | Sequence[str] = "auto",
    contact_threshold: float = 8.0,
) -> list[str]:
    """Render the pairwise confidence maps of one prediction sample.

    Args:
        input_path: A ``confidences.json``, a per-sample ``*logits.npz``, a
            per-seed ``*_distogram.npz``, or a directory holding them. A
            sequence may be passed to combine sources (e.g. a logits npz plus a
            distogram npz from another directory); earlier entries win.
        output: Output image path.
        dpi: Output resolution.
        logits_npz: Explicit PAE/PDE npz, overriding whatever was discovered.
        distogram_npz: Explicit distogram npz, overriding discovery.
        panels: ``"auto"`` or a comma-separated / sequence selection of
            :data:`PANEL_KEYS`, drawn in the given order.
        contact_threshold: Distance cutoff (Å) for the contact-probability panel.

    Returns:
        The panel keys actually drawn — empty (falsy) if there was nothing
        plottable, in which case no file is written.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("matplotlib is required: pip install matplotlib") from e

    from deepfold3.io.confidences import LogitSources, load_logit_maps, resolve_logit_sources
    from deepfold3.io.logit_codec import load_attrs
    from deepfold3.model.config import ConfidenceConfig, DistogramConfig

    cfg = ConfidenceConfig()
    dgram_cfg = DistogramConfig()
    wanted = _parse_panels(panels)

    inputs = [input_path] if isinstance(input_path, str | Path) else list(input_path)
    sources = LogitSources()
    for path in inputs:
        sources = sources.merge(resolve_logit_sources(path))
    if logits_npz is not None:
        sources.logits_npz = Path(logits_npz)
    if distogram_npz is not None:
        sources.distogram_npz = Path(distogram_npz)

    maps = load_logit_maps(
        logits_npz=sources.logits_npz,
        distogram_npz=sources.distogram_npz,
        cfg=cfg,
        contact_threshold=contact_threshold,
    )
    if not maps:
        return []

    available = {
        "pae": ("PAE", 0.0, cfg.max_error_bin, "Greens", "Å"),
        "pde": ("PDE", 0.0, cfg.max_error_bin, "Greens", "Å"),
        "distogram": ("Distogram (E[d])", 0.0, float(dgram_cfg.last_break), "viridis_r", "Å"),
        "contact": (f"Contact Probability (d < {contact_threshold:g} Å)", 0.0, 1.0, "Reds", ""),
    }
    key_to_map = {"pae": "pae", "pde": "pde", "distogram": "distogram", "contact": "contact_probs"}

    order = wanted if wanted is not None else _AUTO_PANELS
    drawn = [k for k in order if key_to_map[k] in maps]
    if wanted is None and "distogram" not in drawn and "contact_probs" in maps:
        drawn.append("contact")
    if not drawn:
        return []

    # ------------------------------------------------------------------
    # Header: scalar summary from the JSON when present, provenance from the
    # codec attrs otherwise (npz-only input).
    # ------------------------------------------------------------------
    summary: dict = {}
    if sources.conf_json is not None and Path(sources.conf_json).exists():
        summary = json.loads(Path(sources.conf_json).read_text()).get("summary", {})
    n_tokens = summary.get("num_tokens") or int(maps[key_to_map[drawn[0]]].shape[0])

    label_path = sources.conf_json or sources.logits_npz or sources.distogram_npz
    subtitle = _summary_line(summary)
    if not subtitle:
        attrs_src = sources.logits_npz or sources.distogram_npz
        subtitle = _provenance_line(load_attrs(attrs_src)) if attrs_src is not None else ""

    fig, axes = plt.subplots(1, len(drawn), figsize=(5.5 * len(drawn), 5))
    if len(drawn) == 1:
        axes = [axes]

    for ax, key in zip(axes, drawn, strict=True):
        title, vmin, vmax, cmap, unit = available[key]
        im = ax.imshow(maps[key_to_map[key]], origin="upper", cmap=cmap, aspect="equal", vmin=vmin, vmax=vmax)
        ax.set_title(f"{title}  |  N={n_tokens}", fontsize=11)
        ax.set_xlabel("Token index")
        ax.set_ylabel("Token index")
        plt.colorbar(im, ax=ax, shrink=0.82, label=unit or None)

    suptitle = f"DeepFold3 Confidences — {Path(label_path).name}"
    if subtitle:
        suptitle += f"\n{subtitle}"
    plt.suptitle(suptitle, fontsize=12)
    plt.tight_layout()

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return drawn


def _default_output(first_input: Path) -> Path:
    """Where to write when --output is omitted: alongside the input."""
    if first_input.is_dir():
        return first_input / "confidences.png"
    return first_input.with_suffix(".png")


@cli.command("plot")
@click.option(
    "--input",
    "inputs",
    required=True,
    multiple=True,
    type=click.Path(exists=True, path_type=Path),
    help="Confidences JSON, logits/distogram .npz, or a prediction directory. Repeatable to combine sources.",
)
@click.option(
    "--output",
    default=None,
    type=click.Path(path_type=Path),
    help="Output PNG path  [default: alongside the first --input]",
)
@click.option(
    "--panels",
    default="auto",
    show_default=True,
    help=f"Comma-separated panels to draw ({','.join(PANEL_KEYS)}), or 'auto'.",
)
@click.option(
    "--contact-threshold",
    default=8.0,
    show_default=True,
    help="Distance cutoff (Å) for the contact-probability panel.",
)
@click.option("--dpi", default=150, show_default=True, help="Output DPI")
def plot(inputs: tuple[Path, ...], output: Path | None, panels: str, contact_threshold: float, dpi: int) -> None:
    """Plot PAE, PDE, distogram, and contact maps from a prediction.

    Accepts the confidences JSON, the raw logit .npz files, or the directory
    holding them — the pairwise maps are reconstructed from whichever logit
    .npz files are found next to the input.
    """
    output = output or _default_output(inputs[0])
    try:
        drawn = plot_confidences_file(
            inputs,
            output,
            dpi=dpi,
            panels=panels,
            contact_threshold=contact_threshold,
        )
    except ValueError as e:
        raise click.BadParameter(str(e), param_hint="--panels") from e

    if drawn:
        click.echo(f"[plot_confidences] {', '.join(drawn)} → {output.resolve()}")
    else:
        click.echo(f"[plot_confidences] No plottable logits found for {inputs[0]}.")


if __name__ == "__main__":
    plot()
