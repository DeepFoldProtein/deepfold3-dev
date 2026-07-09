#!/usr/bin/env bash
# Resume interface_protein_protein: finish base <=768 (115k..95k) + eval, then
# raise cap to <=1536 and fold the 36 medium targets for all 7 ckpts (125k first)
# + eval. Fully resumable: run_sweep --skip-existing, eval_ost skips done steps.
set -u
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
SERIES=df3v1; CAT=interface_protein_protein
ts(){ date +%F\ %T; }

echo "[resume] $(ts) PHASE A: finish base <=768 (115k..95k)"
$PY bench/scripts/run_sweep.py --series "$SERIES" --category "$CAT" \
  --steps 115000,110000,105000,100000,95000 --gpus 0,1,2,3 \
  --num-samples 5 --num-recycles 10 --seed 42 --snapshot average
for N in 115000 110000 105000 100000 95000; do
  [ -f bench/runs/$SERIES/eval/$CAT/step_$N/summary.csv ] && continue
  echo "[resume] $(ts) eval base step $N"
  $PY bench/scripts/eval_ost.py --series "$SERIES" --step "$N" --category "$CAT" --workers 48
done
$PY bench/scripts/summarize.py --series "$SERIES" --category "$CAT" || true
echo "[resume] $(ts) PHASE A done"

echo "[resume] $(ts) PHASE B: build <=1536 inputs + fold 36 medium"
BK="bench/inputs/${CAT}_manifest.csv.le768.bak"
[ -f "$BK" ] || cp "bench/inputs/${CAT}_manifest.csv" "$BK"
$PY bench/scripts/build_inputs.py --category "$CAT" --max-tokens 1536
echo "[resume] $(ts) in_small now: $(awk -F, 'NR>1 && $5==1' bench/inputs/${CAT}_manifest.csv|wc -l) targets"
$PY bench/scripts/run_sweep.py --series "$SERIES" --category "$CAT" \
  --steps 125000,120000,115000,110000,105000,100000,95000 --gpus 0,1,2,3 \
  --num-samples 5 --num-recycles 10 --seed 42 --snapshot average
for N in 125000 120000 115000 110000 105000 100000 95000; do
  echo "[resume] $(ts) re-eval step $N (<=1536, 228 targets)"
  $PY bench/scripts/eval_ost.py --series "$SERIES" --step "$N" --category "$CAT" --workers 48
done
$PY bench/scripts/summarize.py --series "$SERIES" --category "$CAT" || true
echo "[resume] $(ts) ALL_DONE"
