"""Plot PF collapse trajectory — per-submodule, per-layer, and final heatmap.

Reads `tmp/spectral/lineage_step_*.txt` (output of `spectral_quick.py`) and
emits:
  - pf_per_submod.png  : 8 submods × eff_rank vs step
  - pf_per_layer.png   : 48 layers × eff_rank vs step
  - pf_layer_submod_endpoints.png : (layer, submod) heatmap at first/last step

Surfaces where PF is collapsing: which submodule? which layer?
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _parse_layer_section(text: str, header: str, n_rows: int) -> dict[int, float]:
    """Parse a section of `spectral_quick.py` output that follows `header`.

    Returns mapping from first column (layer index) → eff_rank average column.
    """
    out: dict[int, float] = {}
    in_sec = False
    rows_parsed = 0
    for line in text.splitlines():
        if header in line:
            in_sec = True
            continue
        if not in_sec:
            continue
        # Skip blank / header lines
        if "layer" in line and "eff_rank" in line:
            continue
        if line.startswith("===") or rows_parsed >= n_rows:
            break
        m = re.match(r"\s+(\d+)\s+\d+\s+([\d.]+)\s+([\d.]+)", line)
        if m:
            idx, eff, _ = m.group(1), m.group(2), m.group(3)
            out[int(idx)] = float(eff)
            rows_parsed += 1
    return out


def _parse_submod_section(text: str) -> dict[str, float]:
    """Parse the Per PF-submodule averages section."""
    out: dict[str, float] = {}
    in_sec = False
    for line in text.splitlines():
        if "Per PF-submodule averages" in line:
            in_sec = True
            continue
        if not in_sec:
            continue
        if "submod" in line and "eff_rank" in line:
            continue
        if line.startswith("==="):
            break
        m = re.match(r"\s+(\w+)\s+\d+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", line)
        if m:
            sub, eff, _, _ = m.groups()
            out[sub] = float(eff)
    return out


def parse_one(path: Path) -> tuple[int, dict[int, float], dict[str, float]]:
    text = path.read_text()
    m = re.search(r"step_(\d+)", path.name)
    step = int(m.group(1)) if m else 0
    layer_data = _parse_layer_section(text, "Per PF-layer effective rank", 48)
    submod_data = _parse_submod_section(text)
    return step, layer_data, submod_data


def main() -> None:
    src = Path("tmp/spectral")
    files = sorted(src.glob("lineage_step_*.txt"))
    if not files:
        print("no lineage files found")
        return

    rows: list[tuple[int, dict, dict]] = []
    for f in files:
        step, lyr, sub = parse_one(f)
        if not lyr and not sub:
            continue
        rows.append((step, lyr, sub))
    rows.sort(key=lambda r: r[0])
    steps = [r[0] for r in rows]
    print(f"Loaded {len(rows)} lineage points: {steps}")

    # --- Per-layer trajectory ---
    all_layers = sorted({L for _, lyr, _ in rows for L in lyr})
    if all_layers:
        fig, ax = plt.subplots(figsize=(13, 7))
        cmap = plt.cm.viridis(np.linspace(0, 1, len(all_layers)))
        for L, col in zip(all_layers, cmap, strict=False):
            ys = [r[1].get(L, np.nan) for r in rows]
            ax.plot(steps, ys, color=col, linewidth=0.9, alpha=0.85)
        # Annotate first and last layer
        for L in (all_layers[0], all_layers[-1]):
            y0 = rows[-1][1].get(L, np.nan)
            if not np.isnan(y0):
                ax.annotate(f"L{L}", (steps[-1], y0), fontsize=8, xytext=(5, 0), textcoords="offset points")
        # Phase shading (consistent with plot_spectral_evolution.py)
        max_step = max(steps)
        ax.axvspan(0, 2600, color="lightgray", alpha=0.15)
        ax.axvspan(2600, 3500, color="orange", alpha=0.07)
        ax.axvspan(3500, 5000, color="green", alpha=0.07)
        ax.axvspan(5000, 16000, color="blue", alpha=0.07)
        ax.axvspan(16000, max(max_step, 16001), color="magenta", alpha=0.07)
        ax.set_xlabel("training step")
        ax.set_ylabel("PF effective rank (avg over 30 weights per layer)")
        ax.set_title("PF eff_rank trajectory — per-layer (48 layers; color = layer index 0→47)")
        ax.grid(True, alpha=0.3)
        # Colorbar
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=0, vmax=47))
        cbar = fig.colorbar(sm, ax=ax, pad=0.01)
        cbar.set_label("layer index")
        fig.tight_layout()
        out = src / "pf_per_layer.png"
        fig.savefig(out, dpi=140, bbox_inches="tight")
        print(f"saved {out}")

    # --- Per-submod trajectory ---
    all_submods = sorted({s for _, _, sub in rows for s in sub})
    if all_submods:
        fig, ax = plt.subplots(figsize=(11, 6))
        cmap = plt.cm.tab10(np.linspace(0, 1, max(len(all_submods), 10)))
        for s, col in zip(all_submods, cmap, strict=False):
            ys = [r[2].get(s, np.nan) for r in rows]
            ax.plot(steps, ys, marker="o", markersize=3, label=s, color=col, linewidth=1.4)
        max_step = max(steps)
        ax.axvspan(0, 2600, color="lightgray", alpha=0.15)
        ax.axvspan(2600, 3500, color="orange", alpha=0.07)
        ax.axvspan(3500, 5000, color="green", alpha=0.07)
        ax.axvspan(5000, 16000, color="blue", alpha=0.07)
        ax.axvspan(16000, max(max_step, 16001), color="magenta", alpha=0.07)
        ax.set_xlabel("training step")
        ax.set_ylabel("PF effective rank (avg over 48 layers)")
        ax.set_title("PF eff_rank trajectory — per-submodule")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="best", ncol=2)
        fig.tight_layout()
        out = src / "pf_per_submod.png"
        fig.savefig(out, dpi=140, bbox_inches="tight")
        print(f"saved {out}")

    # --- Endpoint comparison: first vs last per-layer ---
    if rows and all_layers:
        first_step, first_layer, _ = rows[0]
        last_step, last_layer, _ = rows[-1]
        # Find a useful intermediate (around peak step 8000)
        mid_idx = min(range(len(rows)), key=lambda i: abs(rows[i][0] - 8000))
        mid_step, mid_layer, _ = rows[mid_idx]

        fig, ax = plt.subplots(figsize=(13, 5))
        xs = all_layers
        ys_first = [first_layer.get(L, np.nan) for L in xs]
        ys_mid = [mid_layer.get(L, np.nan) for L in xs]
        ys_last = [last_layer.get(L, np.nan) for L in xs]
        ax.plot(xs, ys_first, marker="o", markersize=4, label=f"step {first_step}", color="tab:gray", linewidth=1.2)
        ax.plot(
            xs, ys_mid, marker="s", markersize=4, label=f"step {mid_step} (PF peak)", color="tab:orange", linewidth=1.2
        )
        ax.plot(xs, ys_last, marker="^", markersize=4, label=f"step {last_step}", color="tab:red", linewidth=1.2)
        ax.set_xlabel("PF layer index")
        ax.set_ylabel("avg eff_rank")
        ax.set_title("PF per-layer eff_rank — endpoint comparison")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        fig.tight_layout()
        out = src / "pf_layer_endpoints.png"
        fig.savefig(out, dpi=140, bbox_inches="tight")
        print(f"saved {out}")

        # Print top dying layers (largest Δ from peak to last)
        deltas = [
            (
                L,
                mid_layer.get(L, np.nan),
                last_layer.get(L, np.nan),
                last_layer.get(L, np.nan) - mid_layer.get(L, np.nan),
            )
            for L in xs
        ]
        deltas.sort(key=lambda d: d[3])  # most negative first
        print(f"\nTop 10 layers by largest eff_rank drop ({mid_step} → {last_step}):")
        print(f"  {'layer':>5}  {'peak':>8}  {'last':>8}  {'Δ':>8}")
        for L, p, q, d in deltas[:10]:
            print(f"  {L:>5}  {p:>8.2f}  {q:>8.2f}  {d:>+8.2f}")
        print("\nTop 10 layers least drop / gained:")
        for L, p, q, d in deltas[-10:][::-1]:
            print(f"  {L:>5}  {p:>8.2f}  {q:>8.2f}  {d:>+8.2f}")


if __name__ == "__main__":
    main()
