#!/usr/bin/env bash
# Probe + draw the router at each new r17 checkpoint.
#
# Settings match the run's own eval so the router numbers describe the same
# network the eval/* metrics do: EMA shadow weights, and num_recycles = 3
# (= min(max_recycles, 4)). See export/mhr-trimul/README.md section 7.
set -u
REPO=/home/jovyan/work/deepfold3-dev
RUN=$REPO/runs/df3-stage0-r17
LOGS=${LOGS:-$REPO/tmp/mhr_probe}
mkdir -p "$LOGS"
cd "$REPO" || exit 1

for want in "$@"; do
  ckpt=$RUN/checkpoint_$want.pt
  until [ -f "$ckpt" ]; do sleep 180; done
  sleep 90                     # let the write settle
  echo "[$(date -u +%H:%M)] probing step $want"
  CUDA_VISIBLE_DEVICES=${GPU:-0} timeout 3600 python export/mhr-trimul/probe_router.py \
      --checkpoint "$ckpt" --n-examples 20 --snapshot shadow --recycles 3 \
      --out export/mhr-trimul/router_stats_step${want}_shadow.csv \
      > "$LOGS/probe_step$want.log" 2>&1
  CUDA_VISIBLE_DEVICES=${GPU:-0} timeout 3600 python export/mhr-trimul/plot_router_map.py \
      --checkpoint "$ckpt" --snapshot shadow --recycles 3 \
      > "$LOGS/plot_step$want.log" 2>&1
  echo "[$(date -u +%H:%M)] step $want done"
done
