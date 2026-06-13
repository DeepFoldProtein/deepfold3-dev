"""Plot effective-rank evolution across training lineage.

Parses tmp/spectral/lineage_step_*.txt outputs (from `spectral_quick.py`)
and emits line plots of bucket-level eff_rank/stable_rank p50 vs step.
Optionally overlay reference models (AF3, Protenix, Boltz-1/2) as horizontal
guides.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_BUCKET_RE = re.compile(
    r"\s+(\w+)\s+\d+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+\d+\s*$"
)


def parse_spectral_txt(path: Path) -> dict[str, dict[str, float]]:
    """Parse `spectral_quick.py` output → {bucket: {metric: value}}."""
    result: dict[str, dict[str, float]] = {}
    lines = path.read_text().splitlines()
    in_bucket = False
    for line in lines:
        if "Per-bucket summary" in line:
            in_bucket = True
            continue
        if in_bucket:
            if line.startswith("==="):
                break
            m = _BUCKET_RE.match(line)
            if not m:
                continue
            bucket, rms, sr, er, cond = m.groups()
            try:
                result[bucket] = {
                    "rms_p50": float(rms),
                    "stable_rank_p50": float(sr),
                    "eff_rank_p50": float(er),
                    "cond_p50": float(cond),
                }
            except ValueError:
                continue
    return result


def load_series(src_dir: Path, prefix: str) -> list[tuple[int, dict]]:
    files = sorted(src_dir.glob(f"{prefix}_step_*.txt"))
    out: list[tuple[int, dict]] = []
    for f in files:
        m = re.search(r"step_(\d+)", f.name)
        if not m:
            continue
        out.append((int(m.group(1)), parse_spectral_txt(f)))
    out.sort()
    return out


def main() -> None:
    src_dir = Path("tmp/spectral")
    fapeS3_rows = load_series(src_dir, "fapeS3")
    rsf_rows = []  # load_series(src_dir, "rsf")
    if not (fapeS3_rows or rsf_rows):
        print("No spectral files found (expected fapeS3_step_*.txt / rsf_step_*.txt)")
        return

    fapeS3_steps = [r[0] for r in fapeS3_rows]
    rsf_steps = [r[0] for r in rsf_rows]
    print(f"Loaded {len(fapeS3_rows)} fapeS3 (current; +AF3-spec MSA/tmpl fix + fapeS3ity hinge): {fapeS3_steps}")
    # print(f"Loaded {len(rsf_rows)} rsf (prev run, df3-rsf-25e5-c10-g35; no spec fix): {rsf_steps}")

    # Reference model values (from references/ static analysis).
    ref = {
        "AF3": {"pairformer": 62, "diffusion": 140, "msa": 57, "atom_encoder": 19, "template": 28},
        "Protenix": {"pairformer": 56, "diffusion": 32, "msa": 33, "atom_encoder": 1.7, "template": 25},
        "Boltz-1": {"pairformer": 65, "diffusion": 51, "msa": 54, "atom_encoder": 36},
        "Boltz-2": {"pairformer": 67, "diffusion": 78, "msa": 48, "atom_encoder": 36, "template": 30},
    }
    buckets = ["pairformer", "diffusion", "msa", "atom_encoder", "template"]
    # Architectural notes (current arm = `fapeS3`: AF3 baseline + refined SF +
    # AF3-spec MSA/template fixes — NO uniros, NO RMSN). All buckets are
    # architecturally comparable with the AF3/PX/Boltz reference lines.
    # Lineage / s035 series may differ — see series-level legend.
    arch_note = dict.fromkeys(buckets, "✓ same arch as references")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), squeeze=False)
    for i, b in enumerate(buckets):
        ax = axes[i // 3, i % 3]
        if rsf_rows:
            ys = [r[1].get(b, {}).get("eff_rank_p50", float("nan")) for r in rsf_rows]
            ax.plot(
                rsf_steps, ys, marker="s", color="tab:red",
                label="rsf (previous run, df3-rsf-25e5-c10-g35; no spec fix)",
                linewidth=0.8, markersize=2.5, markeredgewidth=0, alpha=0.7,
            )
        if fapeS3_rows:
            ys = [r[1].get(b, {}).get("eff_rank_p50", float("nan")) for r in fapeS3_rows]
            ax.plot(
                fapeS3_steps, ys, marker="o", color="tab:blue",
                label="fapeS3 (current; +AF3-spec MSA/tmpl fix + fapeS3ity hinge α=2)",
                linewidth=1.0, markersize=2.5, markeredgewidth=0,
            )
        # Reference horizontal lines
        ref_colors = {"AF3": "red", "Protenix": "purple", "Boltz-1": "darkgreen", "Boltz-2": "brown"}
        for name, vals in ref.items():
            if b in vals:
                ax.axhline(
                    vals[b], color=ref_colors[name], linestyle="--",
                    linewidth=1, alpha=0.6, label=name,
                )
        ax.set_xlabel("training step")
        ax.set_ylabel("eff_rank p50")
        note = arch_note.get(b, "")
        ax.set_title(f"{b}\n{note}", fontsize=9)
        ax.grid(True, alpha=0.3)
    # Hide last subplot but reuse its area for the shared legend so the
    # in-axes plots are never obstructed.
    legend_ax = axes[1, 2]
    legend_ax.axis("off")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    legend_ax.legend(handles, labels, fontsize=9, loc="center", frameon=False)
    fig.suptitle(
        "Bucket-level effective rank p50 evolution\n"
        "Current arm (fapeS3) is AF3-architecturally equivalent to AF3/PX/Boltz reference lines.",
        fontsize=11,
    )
    fig.tight_layout()
    out = src_dir / "fapeS3_eff_rank.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"saved {out}")

    # Save CSV for the current arm only.
    csv_path = src_dir / "fapeS3_eff_rank.csv"
    primary_rows = fapeS3_rows or rsf_rows
    with csv_path.open("w") as f:
        w = csv.writer(f)
        w.writerow(["step", *buckets])
        for step, per_bucket in primary_rows:
            w.writerow([step, *[per_bucket.get(b, {}).get("eff_rank_p50", "") for b in buckets]])
    print(f"saved {csv_path}")


if __name__ == "__main__":
    main()
