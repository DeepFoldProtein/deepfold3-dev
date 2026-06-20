#!/usr/bin/env bash
# Run the df3v1 FoldBench-small sweeps sequentially across all 4 GPUs:
# monomer_protein, then interface_protein_protein, then interface_antibody_antigen
# (each = full 5k-grid, 17 checkpoints, latest-first). A separate watcher
# (watch_eval.sh) OST-evaluates each step as it completes.
#
#   nohup bash bench/scripts/orchestrate.sh > bench/runs/df3v1/logs/orchestrate.log 2>&1 &
set -u
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
SERIES=df3v1
for CAT in monomer_protein interface_protein_protein interface_antibody_antigen; do
  echo "[orch] === sweeping $CAT (all 17 checkpoints) ==="
  $PY bench/scripts/run_sweep.py --series "$SERIES" --steps all --category "$CAT" \
    --gpus 0,1,2,3 --num-samples 5 --num-recycles 10 --seed 42 --snapshot average
  echo "[orch] === $CAT sweep done ==="
done
echo "[orch] all sweeps complete"
