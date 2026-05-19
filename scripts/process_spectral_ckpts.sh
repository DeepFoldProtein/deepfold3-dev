#!/usr/bin/env bash
# Process new checkpoints for a training arm into spectral_quick.py text dumps,
# then refresh the evolution plot. Idempotent — skips already-processed steps.
#
# Usage:
#   bash scripts/process_spectral_ckpts.sh                   # default: chiral arm
#   bash scripts/process_spectral_ckpts.sh <arm-name> <prefix>
#
# Example:
#   bash scripts/process_spectral_ckpts.sh df3-rsf-25e5-c10-g35 rsf
#
# Outputs:
#   tmp/spectral/<prefix>_step_<N>.txt   — per-ckpt spectral text dump
#   tmp/spectral/<arm>/step_<N>/         — per-ckpt plots / artifacts
#   tmp/spectral/chiral_eff_rank.png    — evolution plot (chiral vs rsf)
#   tmp/spectral/chiral_eff_rank.csv    — eff_rank CSV for current arm

set -euo pipefail

ARM="${1:-df3-chiral-25e5-c10-g35}"
PREFIX="${2:-chiral}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
RUN_DIR="${REPO}/runs/${ARM}"
SPECTRAL_DIR="${REPO}/tmp/spectral"
PER_STEP_DIR="${SPECTRAL_DIR}/${ARM}"

cd "${REPO}"

if [[ ! -d "${RUN_DIR}" ]]; then
    echo "run dir not found: ${RUN_DIR}" >&2
    exit 1
fi

# Find all ckpt steps (excluding _best) and already-processed ones.
mapfile -t ALL_STEPS < <(
    ls "${RUN_DIR}"/checkpoint_*.pt 2>/dev/null \
      | grep -v best \
      | sed 's/.*checkpoint_\([0-9]*\)\.pt/\1/' \
      | sort -n
)
mapfile -t PROCESSED < <(
    ls "${SPECTRAL_DIR}/${PREFIX}_step_"*.txt 2>/dev/null \
      | sed 's/.*step_\([0-9]*\)\.txt/\1/' \
      | sort -n
)

# Set-difference: steps in ALL_STEPS but not in PROCESSED.
declare -A done_map
for s in "${PROCESSED[@]}"; do done_map[$s]=1; done
NEW_STEPS=()
for s in "${ALL_STEPS[@]}"; do
    if [[ -z "${done_map[$s]:-}" ]]; then NEW_STEPS+=("$s"); fi
done

echo "arm: ${ARM}"
echo "prefix: ${PREFIX}"
echo "total ckpts: ${#ALL_STEPS[@]}"
echo "already done: ${#PROCESSED[@]}"
echo "new: ${#NEW_STEPS[@]}"

if [[ ${#NEW_STEPS[@]} -eq 0 ]]; then
    echo "nothing to do; refreshing plot only"
    uv run scripts/plot_spectral_evolution.py
    exit 0
fi

# Detect GPU count (defaults to 4 if nvidia-smi missing).
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
if [[ "${NUM_GPUS}" -lt 1 ]]; then NUM_GPUS=1; fi
echo "using ${NUM_GPUS} GPUs in parallel"

mkdir -p "${PER_STEP_DIR}"

# Process in batches of NUM_GPUS, one step per GPU.
BATCH=()
for s in "${NEW_STEPS[@]}"; do
    BATCH+=("$s")
    if [[ ${#BATCH[@]} -eq ${NUM_GPUS} ]]; then
        gpu=0
        for bs in "${BATCH[@]}"; do
            mkdir -p "${PER_STEP_DIR}/step_${bs}"
            CUDA_VISIBLE_DEVICES=$gpu uv run scripts/spectral_quick.py \
                "${RUN_DIR}/checkpoint_${bs}.pt" --device cuda \
                --out-dir "${PER_STEP_DIR}/step_${bs}" \
                > "${SPECTRAL_DIR}/${PREFIX}_step_${bs}.txt" 2>&1 &
            gpu=$((gpu + 1))
        done
        wait
        echo "  batch done: ${BATCH[*]}"
        BATCH=()
    fi
done
# Tail batch.
if [[ ${#BATCH[@]} -gt 0 ]]; then
    gpu=0
    for bs in "${BATCH[@]}"; do
        mkdir -p "${PER_STEP_DIR}/step_${bs}"
        CUDA_VISIBLE_DEVICES=$gpu uv run scripts/spectral_quick.py \
            "${RUN_DIR}/checkpoint_${bs}.pt" --device cuda \
            --out-dir "${PER_STEP_DIR}/step_${bs}" \
            > "${SPECTRAL_DIR}/${PREFIX}_step_${bs}.txt" 2>&1 &
        gpu=$((gpu + 1))
    done
    wait
    echo "  tail batch done: ${BATCH[*]}"
fi

echo "refreshing plot"
uv run scripts/plot_spectral_evolution.py
