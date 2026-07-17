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

# 3. Gather sweep array job-ids to build the eval dependency
DEP=""
for step in $STEPS_SP; do
  jf="bench/runs/$SERIES/pred/_slurm/step_${step}/jobid.txt"
  [ -f "$jf" ] && DEP="${DEP}:$(tr -d '[:space:]' < "$jf")"
done

if [ "$NO_EVAL" = 1 ]; then
  echo "== NO_EVAL=1: sweep submitted; run eval later with:"
  echo "   sbatch --cpus-per-task=32 --partition=normal --wrap \"bench/scripts/eval_summarize.sh $SERIES '$STEPS_SP' '$CATS'\""
  exit 0
fi

# 4. Submit the eval + summarize job, gated on the sweep finishing.
EVAL_SBATCH="$LOGDIR/eval_${SERIES}.sbatch"
cat > "$EVAL_SBATCH" <<EOF
#!/bin/bash
#SBATCH --job-name=fb_eval_${SERIES}
#SBATCH --partition=normal
#SBATCH --cpus-per-task=32
#SBATCH --output=${REPO}/${LOGDIR}/eval_summarize_%j.out
#SBATCH --open-mode=append
set -uo pipefail
bash ${REPO}/bench/scripts/eval_summarize.sh "$SERIES" "$STEPS_SP" "$CATS" 32
EOF

DEPFLAG=""; [ -n "$DEP" ] && DEPFLAG="--dependency=afterany${DEP}"
echo "-- submitting eval job ${DEPFLAG:+(after sweep$DEP)}"
sbatch $DEPFLAG "$EVAL_SBATCH"

echo "== submitted. Watch with:  squeue -u \$USER   |  tail -f $LOGDIR/*.out =="
