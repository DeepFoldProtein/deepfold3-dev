#!/usr/bin/env bash
# Per-task launcher under `srun`: translate Slurm's task layout into the
# torchrun-style env that deepfold3.training.distributed reads
# (RANK / WORLD_SIZE / LOCAL_RANK + MASTER_ADDR / MASTER_PORT), then exec.
# One task per GPU; every task sees all 4 GPUs of its node and pins
# cuda:LOCAL_RANK itself (srun --gpu-bind=none).
set -euo pipefail
export RANK="${SLURM_PROCID}"
export WORLD_SIZE="${SLURM_NTASKS}"
export LOCAL_RANK="${SLURM_LOCALID}"
export LOCAL_WORLD_SIZE="${SLURM_NTASKS_PER_NODE:-4}"
export MASTER_ADDR="${MASTER_ADDR:-$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)}"
export MASTER_PORT="${MASTER_PORT:-$((20000 + SLURM_JOB_ID % 20000))}"
exec "$@"
