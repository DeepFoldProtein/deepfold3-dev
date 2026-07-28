#!/usr/bin/env bash
# Refresh the whole af3-vs-r14 FoldBench report with the latest available r14 checkpoint.
# Every plot script auto-detects the latest benched step (the one whose summary.csv exists);
# external-model rows are fixed and cached, so only r14 is recomputed — from its already-benched
# raw CSVs. No OST, no external JSON re-parse: fast and safe to run after each checkpoint's
# eval lands.
#
#   ./update_foldbench.sh              # auto-detect latest r14 step
#   ./update_foldbench.sh --step 19000 # pin a specific step
#
# One-time prerequisite (only if foldbench_results.tar.gz changes):
#   .venv/bin/python export/af3-vs-r14-foldbench/precompute_external.py
set -euo pipefail
cd /home/jovyan/work/deepfold3-dev
D=export/af3-vs-r14-foldbench
if [ ! -f $D/foldbench_ext_cache.json ]; then
  echo "[update] external cache missing — building it once…"
  .venv/bin/python $D/precompute_external.py
fi
# af3_ref head-to-head: overview, confidence calibration, multimer decomposition
.venv/bin/python $D/plot_overview.py "$@"
.venv/bin/python $D/plot_confidence.py "$@"
.venv/bin/python $D/plot_multimer_breakdown.py "$@"
# external models: mean→oracle (8 models) + rank-1→oracle (6 with ranking_score)
.venv/bin/python $D/plot_foldbench_models.py "$@"
.venv/bin/python $D/plot_foldbench_conf.py "$@"
# rank-1 vs oracle twin-bar, from FoldBench's own per-target summary (7 external + r14)
.venv/bin/python $D/plot_foldbench_ranked.py "$@"
# training-cutoff control (targets released after Boltz-2's 2023-06-01 cutoff)
if [ -f $D/foldbench_target_dates.csv ]; then
  .venv/bin/python $D/plot_foldbench_models.py --after 2023-06-01 "$@"
  .venv/bin/python $D/plot_foldbench_ranked.py --after 2023-06-01 "$@"
fi
