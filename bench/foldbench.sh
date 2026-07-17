#!/bin/bash
# ============================================================================
# FoldBench end-to-end driver for a df3 checkpoint series (Slurm cluster).
#
#   1. builds inference inputs for each category (idempotent)
#   2. submits the GPU inference sweep as a Slurm job array across ALL cluster
#      GPUs (one inference-batch per GPU), latest step first
#   3. submits a CPU OST-eval + trajectory-plot job that starts automatically
#      when the sweep finishes (Slurm afterany dependency)
#
# ----------------------------------------------------------------------------
# Add a new checkpoint and run it — the whole point of this script:
#   1) drop the exported weights next to the others, named for the series/step:
#        bench/ckpts/<series>/step_<N>.safetensors
#      (a symlink is fine, e.g.
#        ln -s /path/df3v3r10-step-7000.safetensors \
#              bench/ckpts/df3v3/step_7000.safetensors )
#   2) run:  bench/foldbench.sh 7000
#   That's it. Trajectories extend automatically; prior steps are untouched
#   (the sweep and OST are resumable, so re-running only fills gaps).
#
# Usage:
#   bench/foldbench.sh <STEPS>            # e.g. 6000,5000  or  7000
# Env overrides:
#   SERIES=df3v3            checkpoint series (dir under bench/ckpts/)
#   CATEGORIES=all          comma list, or 'all' = the 8 swept categories
#   SEEDS=42,43,44,45       model seeds (FoldBench uses 4)
#   NUM_SAMPLES=5           diffusion samples per seed
#   GPU_TYPE=auto           Slurm GRES gpu type (auto|RTX4090|RTX6000|any)
#   SHARDS=<auto>           array tasks = max concurrent GPUs (default: all)
#   NO_EVAL=0               set 1 to submit only the sweep (no eval job)
# ============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
PY=.venv/bin/python

STEPS="${1:?usage: bench/foldbench.sh STEPS   (e.g. 6000,5000)}"
SERIES="${SERIES:-df3v3}"
CATEGORIES="${CATEGORIES:-all}"
SEEDS="${SEEDS:-42,43,44,45}"
NUM_SAMPLES="${NUM_SAMPLES:-5}"
GPU_TYPE="${GPU_TYPE:-auto}"
NO_EVAL="${NO_EVAL:-0}"

ALL_CATS="monomer_protein monomer_rna monomer_dna interface_protein_protein interface_antibody_antigen interface_protein_dna interface_protein_rna interface_protein_peptide"
if [ "$CATEGORIES" = all ]; then CATS="$ALL_CATS"; else CATS="$(echo "$CATEGORIES" | tr ',' ' ')"; fi
STEPS_SP="$(echo "$STEPS" | tr ',' ' ')"
LOGDIR="bench/runs/$SERIES/logs"; mkdir -p "$LOGDIR"

echo "== FoldBench driver: series=$SERIES steps=$STEPS categories=[$CATS] =="

# 1. Build inputs (only when a manifest is missing; cheap + idempotent otherwise)
for c in $CATS; do
  if [ ! -f "bench/inputs/${c}_manifest.csv" ]; then
    echo "-- build_inputs $c"
    $PY bench/scripts/build_inputs.py --category "$c"
  fi
done

# 2. Submit the inference sweep (slurm_sweep sorts steps latest-first)
SHARDS_ARG=""; [ -n "${SHARDS:-}" ] && SHARDS_ARG="--shards $SHARDS"
$PY bench/scripts/slurm_sweep.py --series "$SERIES" --steps "$STEPS" \
    --categories "$CATEGORIES" --gpu-type "$GPU_TYPE" \
    --seed "$SEEDS" --num-samples "$NUM_SAMPLES" $SHARDS_ARG

if [ "$NO_EVAL" = 1 ]; then
  echo "== NO_EVAL=1: sweep submitted; run eval later with:"
  echo "   sbatch --cpus-per-task=32 --partition=normal --wrap \"bench/scripts/eval_summarize.sh $SERIES '$STEPS_SP' '$CATS'\""
  exit 0
fi

# 3. Submit ONE eval+summarize job per category, each gated (afterany) on just
#    that category's sweep arrays. So each category is OST-scored and its
#    trajectory replotted as soon as its folding finishes — results appear
#    incrementally (monomer_protein first) instead of only after the whole sweep.
JOBIDS="bench/runs/$SERIES/pred/_slurm/jobids.tsv"
if [ ! -s "$JOBIDS" ]; then
  echo "!! no $JOBIDS — nothing submitted? skipping eval." >&2; exit 1
fi
# categories in submission (priority) order
CATS_SUBMITTED=$(awk -F'\t' '!seen[$1]++{print $1}' "$JOBIDS")
for cat in $CATS_SUBMITTED; do
  dep=$(awk -F'\t' -v c="$cat" '$1==c{printf ":%s",$3}' "$JOBIDS")
  csteps=$(awk -F'\t' -v c="$cat" '$1==c{print $2}' "$JOBIDS" | sort -u | tr '\n' ' ')
  evalcats="$cat"; [ "$cat" = all ] && evalcats="$CATS"   # single-array fallback
  EVAL_SBATCH="$LOGDIR/eval_${SERIES}_${cat}.sbatch"
  cat > "$EVAL_SBATCH" <<EOF
#!/bin/bash
#SBATCH --job-name=fbeval_${cat}
#SBATCH --partition=normal
#SBATCH --cpus-per-task=32
#SBATCH --output=${REPO}/${LOGDIR}/eval_${cat}_%j.out
#SBATCH --open-mode=append
set -uo pipefail
bash ${REPO}/bench/scripts/eval_summarize.sh "$SERIES" "$csteps" "$evalcats" 8
EOF
  echo "-- eval[$cat] after$dep"
  sbatch --dependency=afterany${dep} "$EVAL_SBATCH"
done

echo "== submitted. Watch with:  squeue -u \$USER   |  tail -f $LOGDIR/*.out =="
