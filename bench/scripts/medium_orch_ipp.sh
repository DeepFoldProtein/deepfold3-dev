#!/usr/bin/env bash
# After the current <=768 interface_protein_protein sweep (steps 95k-125k) finishes,
# raise the cap to <=1536 and fold the 36 newly-included medium targets (768<t<=1536)
# for the same 7 not-run ckpts (125k first), then OST-eval. Resumable / skip-existing.
#
#   nohup bash bench/scripts/medium_orch_ipp.sh > bench/runs/df3v1/logs/medium_orch_ipp.log 2>&1 &
set -u
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
SERIES=df3v1; CAT=interface_protein_protein
STEPS="125000 120000 115000 110000 105000 100000 95000"

echo "[mediumorch] $(date +%F\ %T) waiting for base <=768 sweep (7 steps) to finish ..."
while true; do
  alldone=1
  for N in $STEPS; do
    pred="bench/runs/$SERIES/pred/$CAT/step_$N"
    [ -d "$pred" ] || { alldone=0; continue; }
    logs=$(ls "$pred"/gpu*.log 2>/dev/null | wc -l)
    fin=$(grep -l "\[batch\] finished" "$pred"/gpu*.log 2>/dev/null | wc -l)
    { [ "$logs" -gt 0 ] && [ "$fin" -eq "$logs" ]; } || alldone=0
  done
  [ "$alldone" -eq 1 ] && break
  sleep 120
done
echo "[mediumorch] $(date +%T) base sweep done."

# 1) back up <=768 manifest, rebuild inputs at <=1536 (adds 36 medium JSONs)
BK="bench/inputs/${CAT}_manifest.csv.le768.bak"
[ -f "$BK" ] || cp "bench/inputs/${CAT}_manifest.csv" "$BK"
echo "[mediumorch] building <=1536 inputs ..."
$PY bench/scripts/build_inputs.py --category "$CAT" --max-tokens 1536
echo "[mediumorch] in_small=1 now: $(awk -F, 'NR>1 && $5==1' bench/inputs/${CAT}_manifest.csv | wc -l) targets"

# 2) sweep <=1536 (228 targets; skip-existing -> only the 36 new run), 125k first
$PY bench/scripts/run_sweep.py --series "$SERIES" --category "$CAT" \
  --steps 125000,120000,115000,110000,105000,100000,95000 \
  --gpus 0,1,2,3 --num-samples 5 --num-recycles 10 --seed 42 --snapshot average

# 3) re-eval each step over the expanded 228-target set, rebuild trajectory
for N in $STEPS; do
  echo "[mediumorch] $(date +%T) OST eval step $N (<=1536) ..."
  $PY bench/scripts/eval_ost.py --series "$SERIES" --step "$N" --category "$CAT" --workers 48
done
$PY bench/scripts/summarize.py --series "$SERIES" --category "$CAT" || true
echo "[mediumorch] $(date +%T) ALL DONE (<=1536, 7 ckpts)"
