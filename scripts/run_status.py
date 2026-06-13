"""Quick run-status snapshot for a deepfold3 training run.

Parses ``train.log`` and prints:
  * latest step + key metrics
  * sampled trajectory at landmark steps (warmup + post-warmup)
  * grad_norm distribution quantiles per window
  * speed (dt, tok/s) recent average
  * optional baseline comparison against another run

Usage:
    uv run python scripts/run_status.py                          # latest run by mtime
    uv run python scripts/run_status.py RUN_NAME                 # specific run
    uv run python scripts/run_status.py --run-dir runs/RUN_NAME  # explicit path
    uv run python scripts/run_status.py RUN --vs BASELINE_RUN    # diff with baseline
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent

# Trainer prints ``Starting training: ...`` at the top of every fresh
# session (after init / resume). When a run dir survives multiple
# sessions — JobSet retries, intentional re-launch into the same
# RUN_NAME, or an aborted attempt that left log lines behind — only the
# tail belongs to the *current* session, so ``parse_log`` discards
# everything before the last occurrence of this marker.
_STARTING_RE = re.compile(r"Starting training:")

# Match a single training step log line. Tolerant to additional fields between
# the captured ones — every key is matched independently against the line.
_STEP_RE = re.compile(r"step=(\d+) ")
_FIELD_RES = {
    "loss": re.compile(r"\bloss=([\d.eE+\-nainf]+)"),
    "diffusion_mse": re.compile(r"\bdiffusion_mse=([\d.eE+\-nainf]+)"),
    "distogram": re.compile(r"\bdistogram=([\d.eE+\-nainf]+)"),
    "smooth_lddt": re.compile(r"\bsmooth_lddt=([\d.eE+\-nainf]+)"),
    "grad_norm": re.compile(r"\bgrad_norm=([\d.eE+\-nainf]+)"),
    "lr": re.compile(r"\blr=([\d.eE+\-nainf]+)"),
    # Reg metric: any of single_l1/2, single_rms (and pair_*) — pick whichever exists.
    "single_reg": re.compile(r"\bsingle_(?:l[12]|rms)=([\d.eE+\-nainf]+)"),
    "pair_reg": re.compile(r"\bpair_(?:l[12]|rms)=([\d.eE+\-nainf]+)"),
    # Reg key NAME (so we can label the column correctly).
    "single_reg_key": re.compile(r"\b(single_(?:l[12]|rms))="),
    "pair_reg_key": re.compile(r"\b(pair_(?:l[12]|rms))="),
    "tok_per_s": re.compile(r"\btok/s=([\d.]+)([km]?)"),
    "dt": re.compile(r"\bdt=([\d.]+)s"),
    "eta": re.compile(r"\beta=(\S+)"),
    # TM-like accuracy/calibration metrics in [0, 1]. ``*_true`` is computed
    # from the true alignment error (= ground-truth structural accuracy);
    # ``*_pred`` is the model's PAE-derived self-estimate (= classical pTM).
    # Cross-chain variants (``itm_*``) only fire on multi-chain inputs. The
    # ``p`` prefix is dropped so that ``*_true`` / ``*_pred`` distinguishes
    # ground-truth from prediction without colliding with pTM's "p".
    "tm_true": re.compile(r"\btm_true=([\d.eE+\-nainf]+)"),
    "itm_true": re.compile(r"\bitm_true=([\d.eE+\-nainf]+)"),
    "tm_pred": re.compile(r"\btm_pred=([\d.eE+\-nainf]+)"),
    "itm_pred": re.compile(r"\bitm_pred=([\d.eE+\-nainf]+)"),
}

LANDMARK_STEPS = [1, 10, 50, 100, 200, 300, 500, 700, 1000, 1500, 2000, 3000, 5000, 7000, 10000, 15000, 20000]
LANDMARK_STEPS.sort()


def _windows(size: int, max_step: int = 50000) -> list[tuple[int, int]]:
    """Return ``[(lo, hi), ...]`` covering [0, max_step) at stride ``size``."""
    n = max(1, max_step // size + 1)
    return [(i * size, (i + 1) * size) for i in range(n)]


def _parse_float(s: str) -> float:
    if s in ("nan", "inf", "-inf"):
        return float(s)
    return float(s)


def _parse_tok_per_s(match: re.Match[str] | None) -> float | None:
    if match is None:
        return None
    val = float(match.group(1))
    unit = match.group(2)
    if unit == "k":
        val *= 1000
    elif unit == "m":
        val *= 1_000_000
    return val


def parse_log(path: Path) -> tuple[list[dict], dict[str, str]]:
    """Return (rows, meta) where rows are per-step dicts and meta has reg-key labels.

    Only step lines that follow the *last* ``Starting training:`` marker are
    returned; lines from prior aborted sessions in the same log file are
    discarded so that downstream stats reflect the current session's geometry
    (world_size, batch geometry, lr/clip etc. may have changed).
    """
    rows: list[dict] = []
    meta: dict[str, str] = {}
    if not path.exists():
        return rows, meta
    with path.open() as f:
        for line in f:
            if _STARTING_RE.search(line):
                # New session boundary — discard everything we've accumulated
                # so far. Hold on to nothing from the prior attempt.
                rows = []
                meta = {}
                continue
            sm = _STEP_RE.search(line)
            if sm is None:
                continue
            row: dict = {"step": int(sm.group(1))}
            for k, rg in _FIELD_RES.items():
                m = rg.search(line)
                if m is None:
                    continue
                if k == "tok_per_s":
                    row[k] = _parse_tok_per_s(m)
                elif k in ("eta", "single_reg_key", "pair_reg_key"):
                    row[k] = m.group(1)
                    if k.endswith("_key"):
                        meta[k] = m.group(1)
                else:
                    try:
                        row[k] = _parse_float(m.group(1))
                    except ValueError:
                        pass
            rows.append(row)
    return rows, meta


def find_latest_run(runs_dir: Path) -> Path | None:
    candidates = [d for d in runs_dir.iterdir() if d.is_dir() and (d / "train.log").exists()]
    if not candidates:
        return None
    return max(candidates, key=lambda d: (d / "train.log").stat().st_mtime)


def resolve_run(arg: str | None, run_dir: Path | None) -> Path | None:
    if run_dir is not None:
        return Path(run_dir)
    runs_root = _REPO / "runs"
    if arg is None:
        return find_latest_run(runs_root)
    p = Path(arg)
    if p.exists() and p.is_dir():
        return p
    return runs_root / arg


def fmt_num(v: float | None, width: int = 8, prec: int = 3) -> str:
    if v is None:
        return f"{'-':>{width}}"
    if v != v:  # NaN
        return f"{'NaN':>{width}}"
    return f"{v:>{width}.{prec}g}"


def show_landmark_trajectory(rows: list[dict], meta: dict[str, str], tail: int | None = None) -> None:
    sk = meta.get("single_reg_key", "single_*")
    pk = meta.get("pair_reg_key", "pair_*")
    by_step = {r["step"]: r for r in rows}
    seen_steps = sorted(set(by_step) & set(LANDMARK_STEPS))
    if not seen_steps:
        return
    truncated = tail is not None and len(seen_steps) > tail
    if truncated:
        seen_steps = seen_steps[-tail:]
    title = f"\n## Sampled trajectory  ({sk}, {pk})"
    if truncated:
        title += f"  [last {tail}; --full for all]"
    print(title)
    print(f"  {'step':>6}  {'lr':>9}  {sk:>12}  {pk:>10}  {'smooth_lddt':>11}  {'grad_norm':>10}")
    for s in seen_steps:
        r = by_step[s]
        print(
            f"  {s:>6d}  {fmt_num(r.get('lr'), 9, 3)}  "
            f"{fmt_num(r.get('single_reg'), 12, 4)}  {fmt_num(r.get('pair_reg'), 10, 4)}  "
            f"{fmt_num(r.get('smooth_lddt'), 11, 4)}  {fmt_num(r.get('grad_norm'), 10, 4)}"
        )


def show_grad_norm_windows(
    rows: list[dict],
    clip_threshold: float | None = None,
    tail: int | None = None,
    window_size: int = 250,
) -> None:
    rows_per_window = []
    max_step = max((r["step"] for r in rows), default=0) + window_size
    for lo, hi in _windows(window_size, max_step):
        gns = [
            r["grad_norm"]
            for r in rows
            if "grad_norm" in r and lo <= r["step"] < hi and r["grad_norm"] == r["grad_norm"]
        ]
        if not gns:
            continue
        rows_per_window.append((lo, hi, gns))
    if not rows_per_window:
        return
    truncated = tail is not None and len(rows_per_window) > tail
    if truncated:
        rows_per_window = rows_per_window[-tail:]
    title = "\n## grad_norm distribution by window"
    if truncated:
        title += f"  [last {tail}; --full for all]"
    print(title)
    header = f"  {'window':<13} {'n':>5}  {'p50':>7}  {'p90':>8}  {'p99':>9}  {'max':>10}  {'mean':>8}  {'m/p50':>5}"
    if clip_threshold is not None:
        header += f"  {f'>{int(clip_threshold)}':>6}"
    print(header)
    for lo, hi, gns in rows_per_window:
        gns.sort()
        n = len(gns)
        median = gns[n // 2]
        mean = sum(gns) / n
        ratio = mean / median if median > 0 else float("nan")
        line = (
            f"  {lo:>5}-{hi:<7d} {n:>5d}  "
            f"{median:>7.2f}  {gns[9 * n // 10]:>8.2f}  "
            f"{gns[min(n - 1, 99 * n // 100)]:>9.2f}  {gns[-1]:>10.2f}  "
            f"{mean:>8.2f}  {ratio:>5.2f}"
        )
        if clip_threshold is not None:
            frac_over = sum(1 for g in gns if g > clip_threshold) / n
            line += f"  {frac_over:>5.1%}"
        print(line)


def show_loss_windows(
    rows: list[dict],
    meta: dict[str, str],
    tail: int | None = None,
    window_size: int = 250,
) -> None:
    sk = meta.get("single_reg_key", "single_*")
    pk = meta.get("pair_reg_key", "pair_*")
    keys = [
        ("loss", "loss"),
        ("diffusion_mse", "diff_mse"),
        ("distogram", "distogram"),
        ("smooth_lddt", "smooth_lddt"),
        ("single_reg", sk),
        ("pair_reg", pk),
        ("tm_true", "tm_true"),
        ("itm_true", "itm_true"),
        ("tm_pred", "tm_pred"),
        ("itm_pred", "itm_pred"),
    ]
    max_step = max((r["step"] for r in rows), default=0) + window_size
    windows = _windows(window_size, max_step)
    for key, label in keys:
        rows_per_window = []
        for lo, hi in windows:
            vals = [r[key] for r in rows if key in r and lo <= r["step"] < hi and r[key] == r[key]]
            if not vals:
                continue
            rows_per_window.append((lo, hi, vals))
        if not rows_per_window:
            continue
        truncated = tail is not None and len(rows_per_window) > tail
        if truncated:
            rows_per_window = rows_per_window[-tail:]
        title = f"\n## {label} distribution by window"
        if truncated:
            title += f"  [last {tail}; --full for all]"
        print(title)
        print(
            f"  {'window':<13} {'n':>5}  {'min':>9}  {'p25':>9}  {'p50':>9}  {'p75':>9}  "
            f"{'p90':>9}  {'p99':>9}  {'max':>10}  {'mean':>9}"
        )
        for lo, hi, vals in rows_per_window:
            vals.sort()
            n = len(vals)
            print(
                f"  {lo:>5}-{hi:<7d} {n:>5d}  "
                f"{vals[0]:>9.4g}  {vals[n // 4]:>9.4g}  {vals[n // 2]:>9.4g}  "
                f"{vals[3 * n // 4]:>9.4g}  {vals[9 * n // 10]:>9.4g}  "
                f"{vals[min(n - 1, 99 * n // 100)]:>9.4g}  {vals[-1]:>10.4g}  "
                f"{sum(vals) / n:>9.4g}"
            )


def show_recent_speed(rows: list[dict], n_recent: int = 50) -> None:
    recent = [r for r in rows if "dt" in r][-n_recent:]
    if not recent:
        return
    dts = [r["dt"] for r in recent]
    tps = [r["tok_per_s"] for r in recent if r.get("tok_per_s") is not None]
    last = rows[-1]
    print(f"\n## Speed (last {len(recent)} steps)")
    print(f"  dt:     mean={sum(dts) / len(dts):.1f}s  min={min(dts):.1f}s  max={max(dts):.1f}s")
    if tps:
        print(
            f"  tok/s:  mean={sum(tps) / len(tps) / 1000:.2f}k  min={min(tps) / 1000:.2f}k  max={max(tps) / 1000:.2f}k"
        )
    if "eta" in last:
        print(f"  ETA:    {last['eta']}  (latest line)")


def show_latest(rows: list[dict], meta: dict[str, str]) -> None:
    if not rows:
        print("(no parsed steps yet)")
        return
    last = rows[-1]
    sk = meta.get("single_reg_key", "single_*")
    pk = meta.get("pair_reg_key", "pair_*")
    print(f"## Latest step: {last['step']}")
    # (display label, dict key)
    cols = [
        ("loss", "loss"),
        ("diff_mse", "diffusion_mse"),
        ("distogram", "distogram"),
        (sk, "single_reg"),
        (pk, "pair_reg"),
        ("smooth_lddt", "smooth_lddt"),
        ("tm_true", "tm_true"),
        ("itm_true", "itm_true"),
        ("tm_pred", "tm_pred"),
        ("itm_pred", "itm_pred"),
        ("grad_norm", "grad_norm"),
        ("lr", "lr"),
    ]
    parts = []
    ncols = 5
    for i, (label, key) in enumerate(cols):
        end = "\n" if (ncols - i - 1) % ncols == 0 else ""
        v = last.get(key)
        if v is None:
            continue
        if label == "lr":
            parts.append(f"{label}={v:.2e}{end}")
        else:
            parts.append(f"{label}={v:.4f}{end}")
    print("  " + "  ".join(parts))


def show_baseline_comparison(
    rows: list[dict],
    baseline_rows: list[dict],
    baseline_meta: dict[str, str],
    tail: int | None = None,
) -> None:
    """Show baseline metric values at the same landmark steps. If reg keys
    differ (e.g. baseline single_l2 vs current single_rms), print the sqrt
    conversion of the L2 form so the comparison is apples-to-apples in
    RMS units.
    """
    base_by_step = {r["step"]: r for r in baseline_rows}
    cur_by_step = {r["step"]: r for r in rows}
    common = sorted(set(LANDMARK_STEPS) & set(base_by_step) & set(cur_by_step))
    if not common:
        print("\n## Baseline comparison\n  (no overlapping landmark steps)")
        return
    truncated = tail is not None and len(common) > tail
    if truncated:
        common = common[-tail:]

    bsk = baseline_meta.get("single_reg_key", "single")
    convert = bsk.endswith("l2")
    label = f"{bsk}^{{1/2}}" if convert else bsk
    title = f"\n## Baseline comparison @ landmark steps  ({label} → RMS scale)"
    if truncated:
        title += f"  [last {tail}; --full for all]"
    print(title)
    print(
        f"  {'step':>6}  {'cur single_rms':>14}  {'base single_rms':>15}  {'cur smooth_lddt':>11}  {'base smooth_lddt':>12}"
    )
    for s in common:
        cur = cur_by_step[s]
        base = base_by_step[s]
        cv = cur.get("single_reg")
        bv = base.get("single_reg")
        if convert and bv is not None:
            bv = bv**0.5
        print(
            f"  {s:>6d}  {fmt_num(cv, 14, 4)}  {fmt_num(bv, 15, 4)}  "
            f"{fmt_num(cur.get('smooth_lddt'), 11, 4)}  {fmt_num(base.get('smooth_lddt'), 12, 4)}"
        )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("run", nargs="?", help="run name (under runs/) or absolute path")
    p.add_argument("--run-dir", type=Path, help="explicit run directory")
    p.add_argument("--vs", help="baseline run name to compare against")
    p.add_argument("--clip", type=float, default=None, help="clip threshold (annotates >clip column)")
    p.add_argument("--tail", type=int, default=10, help="max rows to show per table (default: 10)")
    p.add_argument("--full", action="store_true", help="show all rows (overrides --tail)")
    p.add_argument("-ws", "--window-size", type=int, default=250, help="window size in steps for grad_norm/loss tables (default: 250)")
    args = p.parse_args()
    tail = None if args.full else args.tail

    run_dir = resolve_run(args.run, args.run_dir)
    if run_dir is None or not run_dir.exists():
        print(f"run not found: {args.run or args.run_dir or '(latest)'}", file=sys.stderr)
        return 1
    log = run_dir / "train.log"
    if not log.exists():
        print(f"train.log not found in {run_dir}", file=sys.stderr)
        return 1

    rows, meta = parse_log(log)
    print(f"# Run: {run_dir.name}")
    print(f"# Log: {log}")
    if not rows:
        print("(no training step lines yet)")
        return 0

    show_latest(rows, meta)

    # Auto-detect clip threshold from the run's resolved config if available.
    clip = args.clip
    if clip is None:
        cfg_path = run_dir / "config.yaml"
        if cfg_path.exists():
            for line in cfg_path.read_text().splitlines():
                m = re.search(r"^\s*grad_clip_norm:\s*([\d.eE+\-]+)", line)
                if m:
                    clip = float(m.group(1))
                    break

    show_landmark_trajectory(rows, meta, tail=tail)
    show_grad_norm_windows(rows, clip_threshold=clip, tail=tail, window_size=args.window_size)
    show_loss_windows(rows, meta, tail=tail, window_size=args.window_size)
    show_recent_speed(rows)

    if args.vs:
        baseline_dir = resolve_run(args.vs, None)
        if baseline_dir is None or not (baseline_dir / "train.log").exists():
            print(f"\n(baseline run not found: {args.vs})", file=sys.stderr)
            return 0
        b_rows, b_meta = parse_log(baseline_dir / "train.log")
        print(f"\n# Baseline: {baseline_dir.name}")
        show_baseline_comparison(rows, b_rows, b_meta, tail=tail)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
