#!/usr/bin/env bash
# End-to-end K-Fold FoldBench-small run into bench/runs/kfold (see kfold_bench.py).
# Every stage is resumable: prepare skips seeds with done.txt, predict skips (unit, seed)
# with done.txt, OST skips cached detail JSONs. Re-run the script to fill any gap.
#
#   nohup bash bench/scripts/kfold_run.sh > bench/runs/kfold/logs/run.log 2>&1 &
#
# Env:
#   PROCS_PER_GPU  K-Fold predict processes per GPU (default 2). Each is one `kfold
#                  predict --num-gpus N` over a disjoint split of the queries, so GPU r
#                  hosts rank r of every split. Leftovers (e.g. an OOM-killed split) are
#                  retried at one process per GPU.
#   KF             kfold executable (default references/kfold/.venv/bin/kfold)
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
KF="${KF:-references/kfold/.venv/bin/kfold}"
W=bench/runs/kfold/work
LOG=bench/runs/kfold/logs
NGPU=$(nvidia-smi -L | wc -l)
PROCS="${PROCS_PER_GPU:-2}"
CATS="monomer_protein monomer_rna monomer_dna interface_protein_protein interface_antibody_antigen \
interface_protein_peptide interface_protein_dna interface_protein_rna"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
mkdir -p "$LOG"
say() { echo "[kfold] $(date '+%m-%d %H:%M:%S') $*"; }

# 1. AtlasFold apo/prior ensembles (5 seeds x 5 samples per unique protein sequence).
$PY bench/scripts/kfold_bench.py uniq
while pgrep -f "kfold prepare" >/dev/null; do sleep 60; done   # passes launched by hand
say "prepare: gap-filling pass"
$KF prepare --input $W/uniq --out-dir $W/prep --seed 1 2 3 4 5 --num-gpus "$NGPU" > $LOG/prepare_fill.log 2>&1 \
  || { say "prepare failed — see $LOG/prepare_fill.log"; exit 1; }
say "prepare: $(find $W/prep/apo -name done.txt | wc -l) seed dirs done"

# 2. K-Fold queries wired to those PDBs, validated with K-Fold's own parser: `kfold predict`
#    parses with skip_invalid=False, so one bad query takes down every job in its split.
$PY bench/scripts/kfold_bench.py queries
"$(dirname "$KF")/python" - <<'EOF' || { say "invalid K-Fold queries — fix before predicting"; exit 1; }
import sys
from pathlib import Path

from kfold.inference import assets
from kfold.inference.query import parse_single_file

ccd = assets.load_ccd()
bad = []
for p in sorted(Path("bench/runs/kfold/work/queries/all").glob("*.yaml")):
    try:
        parse_single_file(p, ccd)
    except Exception as e:  # noqa: BLE001
        bad.append(f"{p.name}: {e}")
print(f"[validate] {len(bad)} invalid queries" + "".join(f"\n  {b}" for b in bad[:20]), flush=True)
sys.exit(1 if bad else 0)
EOF

# 3. Predict. predict_pass <input dir> <procs per gpu> <seeds...>
predict_pass() {
  local src=$1 procs=$2; shift 2
  local tag split pids=() k i=0 f
  tag="$(basename "$src")_s$(echo "$@" | tr ' ' '-')_p$procs"
  split=$W/split/$tag
  rm -rf "$split"
  for k in $(seq 0 $((procs - 1))); do mkdir -p "$split/$k"; done
  for f in "$src"/*.yaml; do ln -s "$(realpath "$f")" "$split/$((i % procs))/"; i=$((i + 1)); done
  say "predict $tag: $i units x seeds [$*] on $NGPU GPUs x $procs"
  for k in $(seq 0 $((procs - 1))); do
    $KF predict --input "$split/$k" --out-dir $W/out --seed "$@" --num-gpus "$NGPU" > "$LOG/predict_${tag}_$k.log" 2>&1 &
    pids+=($!)
  done
  for k in "${!pids[@]}"; do
    wait "${pids[$k]}" || say "WARN predict $tag split $k exited non-zero — see $LOG/predict_${tag}_$k.log"
  done
}
predict_pass $W/queries/all "$PROCS" 42 43
predict_pass $W/queries/topup "$PROCS" 44 45
predict_pass $W/queries/all 1 42 43          # leftovers only; a no-op when complete
predict_pass $W/queries/topup 1 44 45

# 4. Bench tree + OST.
$PY bench/scripts/kfold_bench.py tree
export PATH="$HOME/.local/fastbin:$PATH"
[ -x "$HOME/.conda/envs/foldbench/bin/ost" ] && export OST_CMD="$HOME/.conda/envs/foldbench/bin/ost compare-structures"
for c in $CATS; do
  $PY bench/scripts/eval_ost.py --series kfold --step 0 --category "$c" --workers 64 2>&1 | grep -E "^\[eval\]"
done

# 5. Per-category comparison against AF3 and DF3 r19 @120k.
$PY bench/scripts/compare_series.py --out-dir export/kfold-foldbench
say "done"
