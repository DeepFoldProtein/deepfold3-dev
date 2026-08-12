#!/usr/bin/env bash
# Refresh the FoldBench report with the latest available r14 checkpoint.
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
# RETIRED, deliberately not run: plot_overview.py and plot_multimer_breakdown.py.
# Both are head-to-heads against the `af3_ref` sweep, which was folded before the
# bench attached templates and is therefore a handicapped AlphaFold3 (3.8 points
# under the published AF3 on protein-protein, 9.9 on antibody-antigen, same
# targets). Refreshing them would keep publishing that comparison as the report's
# headline figure. The scripts stay in the repo: re-add these two lines once
# af3_ref has been re-folded WITH templates, or once precompute_external.py covers
# the remaining six categories and they can be re-based on FoldBench's own AF3
# predictions (the cache holds only ppi + agab today). The frozen step-23k tables
# they produced are kept in report.md and in overview_data.csv /
# multimer_breakdown_data.csv.
.venv/bin/python $D/plot_confidence.py "$@"
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
