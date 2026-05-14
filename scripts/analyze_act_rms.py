"""Per-layer activation-RMS / ρ analysis for an active or completed training run.

Fetches the ``act_rms/*`` series from the WandB run, organises them by
layer / track / branch, and prints a compact summary that captures what
960 raw scalar series cannot convey by eyeballing:

* per-layer ρ distribution at the latest step (median / p90 / p99 / max)
* count of layers breaching the Karras ``ρ ≲ c/√L`` ceiling (default
  ``c = 1``, so ``c/√L = 0.144`` for L = 48)
* top-K outlier layer indices per branch + per track
* recent log-slope of stream RMS per layer (fast-growing layers float to
  the top)

Usage::

    uv run scripts/analyze_act_rms.py --run df3-rsf-18e3-w1k
    uv run scripts/analyze_act_rms.py --run df3-rsf-18e3-w1k --last 100 --top 5
    uv run scripts/analyze_act_rms.py --run df3-rsf-18e3-w1k --csv /tmp/act.csv
    uv run scripts/analyze_act_rms.py --entity DeepFoldProtein --project deepfold3-stage0 --run df3-rsf-18e3-w1k
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

_KEY_RE = re.compile(
    r"^act_rms/pf_"
    r"(?P<layer>in|\d{2})"
    r"_(?P<rest>.+)$"
)

# Branch buckets that ``ActivationRMSMeter.flush_to_dict`` writes.
_BRANCH_NAMES = (
    "tri_out",
    "tri_in",
    "pair_attn_row",
    "pair_attn_col",
    "pair_ffn",
    "single_attn",
    "single_ffn",
)


@dataclass
class ParsedKey:
    """Decoded WandB key under ``act_rms/``.

    ``kind`` is one of: ``"stream"`` (post-block stream RMS),
    ``"block_branch"`` (block-level Δ RMS), ``"block_rho"``,
    ``"branch_<name>"`` (per-sub-branch RMS), or ``"branch_<name>_rho"``.
    ``layer`` is ``"in"`` for the PF input point, else 0-padded layer
    index. ``track`` is ``"single"`` or ``"pair"`` for stream / block /
    single+pair sub-branches; ``None`` for ambiguous (shouldn't happen).
    """

    kind: str
    layer: str  # "in" or "00".."47"
    track: str | None


def _parse_key(k: str) -> ParsedKey | None:
    m = _KEY_RE.match(k)
    if not m:
        return None
    layer = m["layer"]
    rest = m["rest"]
    if rest in ("single", "pair"):
        return ParsedKey("stream", layer, rest)
    if rest in ("single_branch", "pair_branch"):
        return ParsedKey("block_branch", layer, rest.replace("_branch", ""))
    if rest in ("single_rho", "pair_rho"):
        return ParsedKey("block_rho", layer, rest.replace("_rho", ""))
    # Per-branch keys: b_<branch>[_rho]
    if rest.startswith("b_"):
        body = rest[2:]
        is_rho = body.endswith("_rho")
        if is_rho:
            body = body[: -len("_rho")]
        if body not in _BRANCH_NAMES:
            return None
        # Track of the sub-branch is determined by its name prefix.
        if body.startswith("single_"):
            track = "single"
        elif body.startswith("pair_") or body.startswith("tri_"):
            track = "pair"
        else:
            track = None
        kind = f"branch_{body}_rho" if is_rho else f"branch_{body}"
        return ParsedKey(kind, layer, track)
    return None


def _percentiles(values: list[float], qs: Iterable[float]) -> list[float]:
    if not values:
        return [float("nan")] * sum(1 for _ in qs)
    s = sorted(values)
    n = len(s)
    out = []
    for q in qs:
        if n == 1:
            out.append(s[0])
            continue
        idx = q * (n - 1)
        lo = int(idx)
        hi = min(lo + 1, n - 1)
        frac = idx - lo
        out.append(s[lo] * (1 - frac) + s[hi] * frac)
    return out


def _log_slope(steps: list[int], values: list[float]) -> float:
    """Least-squares slope of ``ln(v) ~ slope · t``. NaN if too few points."""
    if len(values) < 5:
        return float("nan")
    ln_v = [math.log(v) for v in values if v > 0]
    if len(ln_v) < 5:
        return float("nan")
    n = len(ln_v)
    matching_steps = [steps[i] for i, v in enumerate(values) if v > 0]
    mean_t = sum(matching_steps) / n
    mean_y = sum(ln_v) / n
    num = sum((matching_steps[i] - mean_t) * (ln_v[i] - mean_y) for i in range(n))
    den = sum((matching_steps[i] - mean_t) ** 2 for i in range(n))
    if den == 0:
        return float("nan")
    return num / den


def _fmt(v: float, width: int = 7, prec: int = 3) -> str:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return f"{'—':>{width}}"
    return f"{v:>{width}.{prec}f}"


def _fetch_rows(entity: str, project: str, run_name: str, last_steps: int) -> tuple[list[dict], dict]:
    """Pull recent rows from WandB.

    Strategy: list runs matching ``display_name``, take the most recent
    matching run, then ``scan_history`` filtered to ``act_rms/*`` keys.
    Returns ``(rows, meta)`` where each row has at least ``train/step``
    plus the ``act_rms`` values.
    """
    import wandb

    api = wandb.Api()
    path = f"{entity}/{project}" if entity else project
    runs = list(api.runs(path=path, filters={"display_name": run_name}, order="-created_at"))
    if not runs:
        raise SystemExit(f"No WandB run found matching display_name={run_name!r} under {path}")
    run = runs[0]
    meta = {
        "id": run.id,
        "name": run.name,
        "state": run.state,
        "created": str(run.created_at),
    }
    # `summary` gives the latest step; pull a history slab around it.
    last_step = run.summary.get("train/step", 0)
    if not isinstance(last_step, (int, float)):
        last_step = 0
    last_step = int(last_step)
    min_step = max(0, last_step - last_steps + 1)
    # Build the keys list — wandb wants explicit keys for efficient filtering.
    # We don't know the exact key set ahead of time, but `scan_history()`
    # without keys returns everything. Then we filter to ``act_rms/*``
    # locally.
    rows: list[dict] = []
    for row in run.scan_history(min_step=min_step, max_step=last_step + 1):
        rows.append(row)
    return rows, meta


def _organize(rows: list[dict]) -> dict:
    """Group rows by series → list of (step, value)."""
    series: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for row in rows:
        s = row.get("train/step")
        if s is None:
            continue
        s = int(s)
        for k, v in row.items():
            if not k.startswith("act_rms/") or v is None:
                continue
            if not isinstance(v, (int, float)):
                continue
            series[k].append((s, float(v)))
    for k in series:
        series[k].sort(key=lambda p: p[0])
    return series


def _per_layer_at_latest(series: dict, predicate) -> dict[int, float]:
    """For all series matching ``predicate(parsed_key)``, return ``{layer_idx: latest value}``.

    Skips the PF-input point (``layer == "in"``) since it has no
    corresponding "layer index".
    """
    out: dict[int, float] = {}
    for k, vals in series.items():
        pk = _parse_key(k)
        if pk is None or pk.layer == "in" or not predicate(pk):
            continue
        if not vals:
            continue
        out[int(pk.layer)] = vals[-1][1]
    return out


def _print_distribution(name: str, values_by_layer: dict[int, float], ceiling: float | None) -> None:
    """One-line distribution summary + breach count."""
    if not values_by_layer:
        print(f"  {name:<32} (no data)")
        return
    vals = list(values_by_layer.values())
    qs = _percentiles(vals, [0.5, 0.9, 0.99, 1.0])
    parts = [
        f"  {name:<32}",
        f"N={len(vals):3d}",
        f"med={_fmt(qs[0])}",
        f"p90={_fmt(qs[1])}",
        f"p99={_fmt(qs[2])}",
        f"max={_fmt(qs[3])}",
    ]
    if ceiling is not None:
        n_breach = sum(1 for v in vals if v > ceiling)
        parts.append(f"breach(>={ceiling:.3f})={n_breach}/{len(vals):3d}")
    print(" ".join(parts))


def _collect_values_per_layer(
    series: dict, predicate, num_layers: int = 48
) -> list[list[float]]:
    """Build ``[layer_0_values, layer_1_values, ..., layer_{L-1}_values]``.

    For every series matching ``predicate``, append all values (across the
    full fetched step window) to the layer's bucket. PF-input point is
    skipped.
    """
    buckets: list[list[float]] = [[] for _ in range(num_layers)]
    for k, vals in series.items():
        pk = _parse_key(k)
        if pk is None or pk.layer == "in" or not predicate(pk):
            continue
        li = int(pk.layer)
        if 0 <= li < num_layers:
            buckets[li].extend(v for _, v in vals)
    return buckets


def _render_boxplots(
    series: dict,
    output_path: str,
    ceiling: float,
    latest_step: int,
    num_layers: int = 48,
) -> None:
    """Write a per-layer box-plot PNG covering 9 ρ panels.

    Layout (3×3 grid):

    ::

        block-ρ single    block-ρ pair       b_tri_out ρ
        b_tri_in ρ        b_pair_attn_row    b_pair_attn_col
        b_pair_ffn ρ      b_single_attn ρ    b_single_ffn ρ

    Each panel shows 48 box-and-whiskers — one per layer — covering the
    distribution of that layer's ρ across the fetched step window. The
    Karras ``c/√L`` ceiling is drawn as a horizontal reference line.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [
        (r"block $\rho$ single", lambda pk: pk.kind == "block_rho" and pk.track == "single"),
        (r"block $\rho$ pair", lambda pk: pk.kind == "block_rho" and pk.track == "pair"),
        (r"$\rho$ tri_out", lambda pk: pk.kind == "branch_tri_out_rho"),
        (r"$\rho$ tri_in", lambda pk: pk.kind == "branch_tri_in_rho"),
        (r"$\rho$ pair_attn_row", lambda pk: pk.kind == "branch_pair_attn_row_rho"),
        (r"$\rho$ pair_attn_col", lambda pk: pk.kind == "branch_pair_attn_col_rho"),
        (r"$\rho$ pair_ffn", lambda pk: pk.kind == "branch_pair_ffn_rho"),
        (r"$\rho$ single_attn", lambda pk: pk.kind == "branch_single_attn_rho"),
        (r"$\rho$ single_ffn", lambda pk: pk.kind == "branch_single_ffn_rho"),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(18, 13), sharex=True)
    fig.suptitle(
        rf"act_rms $\rho$ per layer @ step {latest_step}  (Karras ceiling $c/\sqrt{{L}}$ = {ceiling:.3f})",
        fontsize=13,
    )
    for ax, (title, pred) in zip(axes.flat, panels, strict=False):
        buckets = _collect_values_per_layer(series, pred, num_layers=num_layers)
        n_points = sum(len(b) for b in buckets)
        if n_points == 0:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center")
            ax.set_title(title)
            continue
        # Replace empty layer buckets with [nan] so boxplot positions stay aligned.
        boxdata = [b if b else [float("nan")] for b in buckets]
        ax.boxplot(
            boxdata,
            positions=list(range(num_layers)),
            widths=0.65,
            showfliers=False,
            patch_artist=True,
            boxprops=dict(facecolor="#4f93ce", alpha=0.55, linewidth=0.6),
            medianprops=dict(color="#cc3333", linewidth=1.0),
            whiskerprops=dict(linewidth=0.6),
            capprops=dict(linewidth=0.6),
        )
        ax.axhline(ceiling, color="#cc6600", linestyle="--", linewidth=0.9, alpha=0.85)
        ax.set_title(f"{title}  (N={n_points})", fontsize=10)
        ax.set_xlim(-1, num_layers)
        ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.6)
        # Tick every 4 layers to keep the axis legible
        ax.set_xticks(list(range(0, num_layers, 4)))
        ax.tick_params(axis="both", labelsize=8)
    for ax in axes[-1, :]:
        ax.set_xlabel("layer index", fontsize=9)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=110)
    plt.close(fig)


def _print_top_outliers(name: str, values_by_layer: dict[int, float], top_k: int) -> None:
    if not values_by_layer:
        return
    items = sorted(values_by_layer.items(), key=lambda p: -p[1])
    head = items[:top_k]
    line = ", ".join(f"L{li:02d}={_fmt(v, 5, 3)}" for li, v in head)
    print(f"    top-{top_k:<2}                          {line}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--entity", default="DeepFoldProtein", help="WandB entity (default: DeepFoldProtein)")
    p.add_argument("--project", default="deepfold3-stage0", help="WandB project (default: deepfold3-stage0)")
    p.add_argument("--run", required=True, help="WandB display name of the run")
    p.add_argument("--last", type=int, default=200, help="Number of recent steps to fetch (default: 200)")
    p.add_argument("--top", type=int, default=5, help="Top-K outlier layers per metric (default: 5)")
    p.add_argument("--depth", type=int, default=48, help="PF layer count for Karras ceiling c/√L (default: 48)")
    p.add_argument("--c", type=float, default=1.0, help="Karras ρ ceiling numerator (default: 1.0; ρ ≲ c/√L)")
    p.add_argument(
        "--slope-window",
        type=int,
        default=200,
        help="Window in optimizer steps for log-slope of stream RMS (default: 200)",
    )
    p.add_argument("--csv", help="Optional path: dump all per-step per-key values to CSV (long format)")
    p.add_argument(
        "--boxplot",
        help=(
            "Path for the per-layer box-plot PNG. Each layer gets one box "
            "summarising the distribution of its recent ρ values over the "
            "fetched window. With --watch, the PNG is overwritten on each "
            "refresh."
        ),
    )
    p.add_argument(
        "--watch",
        type=int,
        default=0,
        metavar="SECONDS",
        help=(
            "If > 0, re-fetch + redraw every SECONDS and loop. Pair with "
            "--boxplot so each iteration overwrites the PNG. Stop with Ctrl-C."
        ),
    )
    args = p.parse_args()

    ceiling = args.c / math.sqrt(args.depth)
    iteration = 0
    while True:
        iteration += 1
        if args.watch > 0:
            import time as _time

            print(f"\n[iteration {iteration} @ {_time.strftime('%Y-%m-%d %H:%M:%S')}]")
        print(f"Fetching last {args.last} steps from WandB run {args.entity}/{args.project}/{args.run}...")
        try:
            rows, meta = _fetch_rows(args.entity, args.project, args.run, args.last)
        except Exception as exc:
            print(f"  fetch failed: {exc}")
            if args.watch > 0:
                _sleep_with_msg(args.watch)
                continue
            raise
        print(f"  run id={meta['id']}  state={meta['state']}  created={meta['created']}")
        print(f"  fetched {len(rows)} rows")
        if not rows:
            if args.watch > 0:
                _sleep_with_msg(args.watch)
                continue
            sys.exit("No rows returned.")

        series = _organize(rows)
        if not series:
            if args.watch > 0:
                print("  no act_rms keys yet — meter may not have flushed")
                _sleep_with_msg(args.watch)
                continue
            sys.exit("No act_rms keys found in the fetched rows.")
        latest_step = max(int(r["train/step"]) for r in rows if r.get("train/step") is not None)
        print(f"  latest step = {latest_step}")
        print(f"  Karras ceiling c/√L = {args.c:.2f}/√{args.depth} = {ceiling:.4f}\n")
        _run_one(args, series, ceiling, latest_step)
        if args.watch <= 0:
            break
        _sleep_with_msg(args.watch)


def _sleep_with_msg(interval: int) -> None:
    import time as _time

    print(f"\n  sleeping {interval}s …")
    try:
        _time.sleep(interval)
    except KeyboardInterrupt:
        print("\n  interrupted — exiting")
        sys.exit(0)


def _run_one(args, series, ceiling, latest_step):

    # ============================================================
    # Section 1: stream RMS distribution at the latest step
    # ============================================================
    print(f"=== Stream RMS @ step {latest_step} ===")
    for track in ("single", "pair"):
        vals_by_layer = _per_layer_at_latest(
            series, lambda pk, t=track: pk.kind == "stream" and pk.track == t
        )
        _print_distribution(f"stream {track} RMS", vals_by_layer, ceiling=None)

    # ============================================================
    # Section 2: block-level ρ at the latest step (one ρ per layer per track)
    # ============================================================
    print(f"\n=== Block-level ρ @ step {latest_step} (ρ ≲ {ceiling:.3f}) ===")
    for track in ("single", "pair"):
        vals_by_layer = _per_layer_at_latest(
            series, lambda pk, t=track: pk.kind == "block_rho" and pk.track == t
        )
        _print_distribution(f"block ρ {track}", vals_by_layer, ceiling=ceiling)
        _print_top_outliers(f"block ρ {track}", vals_by_layer, top_k=args.top)

    # ============================================================
    # Section 3: per-branch ρ at the latest step (7 sub-branches × per-layer)
    # ============================================================
    print(f"\n=== Per-branch ρ @ step {latest_step} (ρ ≲ {ceiling:.3f}) ===")
    for branch in _BRANCH_NAMES:
        vals_by_layer = _per_layer_at_latest(
            series, lambda pk, b=branch: pk.kind == f"branch_{b}_rho"
        )
        _print_distribution(f"ρ {branch:<14}", vals_by_layer, ceiling=ceiling)
        _print_top_outliers(f"ρ {branch}", vals_by_layer, top_k=args.top)

    # ============================================================
    # Section 4: recent log-slope of stream RMS per layer
    # ============================================================
    print(f"\n=== Stream RMS log-slope (last {args.slope_window} steps) ===")
    slope_window_start = latest_step - args.slope_window
    for track in ("single", "pair"):
        slopes_by_layer: dict[int, float] = {}
        for k, vals in series.items():
            pk = _parse_key(k)
            if pk is None or pk.kind != "stream" or pk.track != track or pk.layer == "in":
                continue
            sub = [(s, v) for s, v in vals if s >= slope_window_start]
            if len(sub) < 5:
                continue
            steps = [s for s, _ in sub]
            vs = [v for _, v in sub]
            slope = _log_slope(steps, vs)
            if not math.isnan(slope):
                slopes_by_layer[int(pk.layer)] = slope
        if not slopes_by_layer:
            print(f"  {track} slope: (no data)")
            continue
        vals = list(slopes_by_layer.values())
        qs = _percentiles(vals, [0.5, 0.9, 0.99, 1.0])
        # Express as ratio per 50 steps so it lines up with our existing
        # block-window analysis vocabulary.
        ratio50 = [math.exp(q * 50) for q in qs]
        print(
            f"  {track:<6} slope/step "
            f"med={_fmt(qs[0], 8, 5)}  "
            f"p90={_fmt(qs[1], 8, 5)}  "
            f"p99={_fmt(qs[2], 8, 5)}  "
            f"max={_fmt(qs[3], 8, 5)}  "
            f"(ratio/50step: med={ratio50[0]:.2f}× / max={ratio50[3]:.2f}×)"
        )
        items = sorted(slopes_by_layer.items(), key=lambda p: -p[1])
        head = items[: args.top]
        print(
            "    top fastest-growing layers: "
            + ", ".join(f"L{li:02d}={_fmt(s, 7, 5)}" for li, s in head)
        )

    # ============================================================
    # Optional CSV dump
    # ============================================================
    if args.csv:
        import csv

        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", "key", "kind", "layer", "track", "value"])
            for k, vals in series.items():
                pk = _parse_key(k)
                if pk is None:
                    continue
                for s, v in vals:
                    w.writerow([s, k, pk.kind, pk.layer, pk.track or "", v])
        print(f"\nCSV dump → {args.csv}")

    # ============================================================
    # Optional box-plot rendering (3x3 ρ panels)
    # ============================================================
    if args.boxplot:
        _render_boxplots(series, args.boxplot, ceiling, latest_step, num_layers=args.depth)
        print(f"\nBox-plot PNG → {args.boxplot}")


if __name__ == "__main__":
    main()
