#!/usr/bin/env bash
# One-off eval watcher for the 95k-125k steps (watch_eval.sh only covers <=90k).
# OST-evaluates each (category, step) the moment its inference finishes, then
# rebuilds the category trajectory. Resumable (skips steps whose summary.csv exists).
#
#   nohup bash bench/scripts/watch_eval_95k_125k.sh df3v1 interface_protein_protein \
#         > bench/runs/df3v1/logs/watch_ipp_95k_125k.log 2>&1 &
set -u
cd "$(dirname "$0")/../.."
SERIES="${1:?series}"; shift
CATS=("$@"); [ ${#CATS[@]} -eq 0 ] && CATS=(interface_protein_protein)
PY=.venv/bin/python
STEPS="125000 120000 115000 110000 105000 100000 95000"
TOTAL=$(echo "$STEPS" | wc -w)

while true; do
  alldone=1
  for CAT in "${CATS[@]}"; do
    newly=0
    for N in $STEPS; do
      pred="bench/runs/$SERIES/pred/$CAT/step_$N"
      summ="bench/runs/$SERIES/eval/$CAT/step_$N/summary.csv"
      [ -f "$summ" ] && continue
      [ -d "$pred" ] || { alldone=0; continue; }
      logs=$(ls "$pred"/gpu*.log 2>/dev/null | wc -l)
      fin=$(grep -l "\[batch\] finished" "$pred"/gpu*.log 2>/dev/null | wc -l)
      if [ "$logs" -gt 0 ] && [ "$fin" -eq "$logs" ]; then
        echo "[watch] $CAT step $N complete -> OST eval $(date +%H:%M:%S)"
        $PY bench/scripts/eval_ost.py --series "$SERIES" --step "$N" --category "$CAT" --workers 48
        newly=1
      else
        alldone=0
      fi
    done
    if [ "$newly" -eq 1 ]; then
      $PY bench/scripts/summarize.py --series "$SERIES" --category "$CAT" || true
    fi
    d=0; for N in $STEPS; do [ -f "bench/runs/$SERIES/eval/$CAT/step_$N/summary.csv" ] && d=$((d+1)); done
    [ "$d" -lt "$TOTAL" ] && alldone=0
  done
  [ "$alldone" -eq 1 ] && { echo "[watch] all steps evaluated — done $(date +%H:%M:%S)"; break; }
  sleep 120
done
