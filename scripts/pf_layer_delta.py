"""Per-PairFormer-layer weight delta across checkpoints.

For each consecutive checkpoint pair, computes the relative Frobenius
change of every parameter, aggregated per trunk PairFormer layer
(0..47) and per submodule family (tri_mul, attention, transition).
Reveals which layer the optimizer is moving most across a given
training window.

Usage::

    uv run python scripts/pf_layer_delta.py \\
        --arm-dir runs/df3-fapeK5-25e5-c10-g35 \\
        --steps 7000 11000 13000 15500

Output: one row per layer with delta per step-window, plus a per-
submodule breakdown for the final window. ``--plot`` writes a heat-map.
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

# Trunk PairFormer: `evoformer.trunk_pairformer.{layer}.{submod}.{...}`
_PF_PAT = re.compile(r"^evoformer\.trunk_pairformer\.(\d+)\.([^.]+)\.")

# Submodule families (grouped by role).
_SUBMOD_FAMILY = {
    "tri_mul_in": "tri_mul",
    "tri_mul_out": "tri_mul",
    "tri_attn_start": "tri_attn",
    "tri_attn_end": "tri_attn",
    "pair_transition": "transition",
    "pair_attn_row": "pair_attn",
    "pair_attn_col": "pair_attn",
    "single_attn": "single_attn",
    "single_transition": "single_transition",
}


def _load_pf_state(ckpt_path: Path) -> dict[str, torch.Tensor]:
    """Return {param_name: tensor (fp32 on cpu)} restricted to trunk PF."""
    blob = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    sd = blob["model"] if "model" in blob else blob
    out = {}
    for k, v in sd.items():
        if not k.startswith("evoformer.trunk_pairformer."):
            continue
        # Skip 1D scale/shift/bias — focus on matmul weight movement
        if v.dim() < 2:
            continue
        out[k] = v.detach().float().cpu()
    return out


def _layer_subfamily(name: str) -> tuple[int | None, str | None]:
    m = _PF_PAT.match(name)
    if not m:
        return None, None
    return int(m.group(1)), _SUBMOD_FAMILY.get(m.group(2), m.group(2))


def _aggregate_delta(
    sd_a: dict[str, torch.Tensor],
    sd_b: dict[str, torch.Tensor],
) -> dict[tuple[int, str], tuple[float, float]]:
    """{(layer, family): (sum |Δ|², sum |W_a|²)}."""
    acc: dict[tuple[int, str], list[float]] = defaultdict(lambda: [0.0, 0.0])
    for k, w_a in sd_a.items():
        if k not in sd_b:
            continue
        w_b = sd_b[k]
        if w_a.shape != w_b.shape:
            continue
        layer, fam = _layer_subfamily(k)
        if layer is None:
            continue
        delta_sq = float(((w_b - w_a) ** 2).sum())
        orig_sq = float((w_a**2).sum())
        acc[(layer, fam)][0] += delta_sq
        acc[(layer, fam)][1] += orig_sq
    return {kk: tuple(vv) for kk, vv in acc.items()}


def _rel_change(d_sq: float, w_sq: float) -> float:
    if w_sq <= 0:
        return float("nan")
    return float(np.sqrt(d_sq / w_sq))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm-dir", required=True, help="Path to runs/<arm> directory.")
    ap.add_argument("--steps", type=int, nargs="+", required=True, help="Sequence of ckpt steps to diff consecutively.")
    ap.add_argument("--plot", type=Path, default=None, help="Optional: write per-layer × window heatmap.")
    args = ap.parse_args()

    arm = Path(args.arm_dir)
    states = {}
    for s in args.steps:
        ckpt = arm / f"checkpoint_{s}.pt"
        print(f"loading {ckpt.name}")
        states[s] = _load_pf_state(ckpt)
    print()

    # Compute per consecutive pair
    windows = list(zip(args.steps[:-1], args.steps[1:]))
    deltas = {}
    for a, b in windows:
        deltas[(a, b)] = _aggregate_delta(states[a], states[b])

    n_layers = 1 + max(layer for (layer, _fam) in next(iter(deltas.values())).keys())

    # Per-layer rel change per window
    layer_rel = {}  # (layer, window) -> rel_change
    for win, dd in deltas.items():
        per_layer_sq = defaultdict(lambda: [0.0, 0.0])
        for (layer, fam), (d_sq, w_sq) in dd.items():
            per_layer_sq[layer][0] += d_sq
            per_layer_sq[layer][1] += w_sq
        for layer in range(n_layers):
            d_sq, w_sq = per_layer_sq.get(layer, (0.0, 0.0))
            layer_rel[(layer, win)] = _rel_change(d_sq, w_sq)

    # === Per-layer trajectory ===
    print(f"=== Per-layer ||ΔW|| / ||W|| across windows ===")
    header = f"{'layer':>5} | " + "  ".join(f"{a}-{b}".rjust(11) for a, b in windows)
    print(header)
    print("-" * len(header))
    # Compute mean rel-change per window for context line
    for layer in range(n_layers):
        cells = []
        for win in windows:
            r = layer_rel[(layer, win)]
            cells.append(f"{r * 100:>10.2f}%")
        print(f"{layer:>5} | " + "  ".join(cells))
    print("-" * len(header))
    mean_row = []
    for win in windows:
        mean = float(np.mean([layer_rel[(l, win)] for l in range(n_layers)]))
        mean_row.append(f"{mean * 100:>10.2f}%")
    print(f"{'mean':>5} | " + "  ".join(mean_row))

    # === Top-changing layers in the FINAL window ===
    final_win = windows[-1]
    print()
    print(f"=== Top 10 layers by rel-change in window {final_win[0]}→{final_win[1]} ===")
    ranked = sorted(((layer, layer_rel[(layer, final_win)]) for layer in range(n_layers)), key=lambda x: -x[1])
    for layer, r in ranked[:10]:
        print(f"  layer {layer:>2}  : {r * 100:>6.2f}%")

    # === Submodule breakdown across ALL windows ===
    print()
    print(f"=== Per-family rel-change across windows (normalized by 1k steps) ===")
    # Collect family rel-changes per window
    win_family = {}  # win -> {fam: rel_change}
    all_families = set()
    for win in windows:
        family_sq = defaultdict(lambda: [0.0, 0.0])
        for (layer, fam), (d_sq, w_sq) in deltas[win].items():
            family_sq[fam][0] += d_sq
            family_sq[fam][1] += w_sq
        win_family[win] = {fam: _rel_change(d_sq, w_sq) for fam, (d_sq, w_sq) in family_sq.items()}
        all_families.update(family_sq.keys())
    fams = sorted(all_families)
    header = f"{'family':<20} | " + " ".join(f"{a}-{b}".rjust(11) for a, b in windows) + "  | rate-last"
    print(header)
    print("-" * len(header))
    for fam in fams:
        cells = []
        for win in windows:
            r = win_family[win].get(fam, float("nan"))
            cells.append(f"{r * 100:>10.2f}%")
        # per-step rate of FINAL window (normalized per 1k step)
        last_a, last_b = windows[-1]
        last_r = win_family[windows[-1]].get(fam, float("nan"))
        rate = last_r * 100 / ((last_b - last_a) / 1000)
        print(f"{fam:<20} | " + " ".join(cells) + f"  | {rate:>5.2f}%/k")

    # === Optional plot ===
    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # heatmap: rows=layers, cols=windows; cell = rel change %
        z = np.zeros((n_layers, len(windows)), dtype=np.float32)
        for j, win in enumerate(windows):
            for i in range(n_layers):
                z[i, j] = layer_rel[(i, win)] * 100
        fig, ax = plt.subplots(figsize=(1.5 + 0.6 * len(windows), 12))
        im = ax.imshow(z, aspect="auto", cmap="viridis", interpolation="nearest")
        ax.set_yticks(range(n_layers))
        ax.set_yticklabels([str(i) for i in range(n_layers)], fontsize=7)
        ax.set_xticks(range(len(windows)))
        ax.set_xticklabels([f"{a}→{b}" for a, b in windows], rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("PairFormer layer (0=input, 47=output)")
        ax.set_title("Per-layer relative weight change  ||ΔW|| / ||W|| (%)")
        cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cbar.set_label("%")
        for i in range(n_layers):
            for j in range(len(windows)):
                ax.text(
                    j,
                    i,
                    f"{z[i, j]:.1f}",
                    ha="center",
                    va="center",
                    fontsize=5,
                    color="white" if z[i, j] < z.max() * 0.5 else "black",
                )
        plt.tight_layout()
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.plot, dpi=120, bbox_inches="tight")
        print(f"\nwrote {args.plot}")


if __name__ == "__main__":
    main()
