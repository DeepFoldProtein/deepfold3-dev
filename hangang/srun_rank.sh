#!/usr/bin/env bash
# Per-task launcher under `srun`: translate Slurm's task layout into the
# torchrun-style env that deepfold3.training.distributed reads
# (RANK / WORLD_SIZE / LOCAL_RANK + MASTER_ADDR / MASTER_PORT), then exec.
# One task per GPU; every task sees all 4 GPUs of its node and pins
# cuda:LOCAL_RANK itself (srun --gpu-bind=none).
#
# RANK is derived from this node's position in the job's node list, NOT from
# SLURM_PROCID: Slurm does not promise that task 0 lands on the first listed
# node, and init_process_group makes RANK 0 host the TCPStore at MASTER_ADDR
# (the first listed node). On a scattered 64-node allocation task 0 sat
# elsewhere, so nothing listened at MASTER_ADDR and every rank timed out with
# "Connection refused". torchrun's c10d rendezvous (KISTI's example) avoids
# this by electing the host by name; this keeps one task per GPU and pins
# rank 0 to the MASTER_ADDR node instead.
set -euo pipefail
nodes=$(scontrol show hostnames "${SLURM_JOB_NODELIST}")
node_idx=$(grep -nx "$(hostname -s)" <<<"${nodes}" | cut -d: -f1)
[[ -n "${node_idx}" ]] || { echo "srun_rank: $(hostname -s) not in ${SLURM_JOB_NODELIST}" >&2; exit 1; }
export LOCAL_WORLD_SIZE="${SLURM_NTASKS_PER_NODE%%(*}"
export LOCAL_RANK="${SLURM_LOCALID}"
export RANK=$(( (node_idx - 1) * LOCAL_WORLD_SIZE + LOCAL_RANK ))
export WORLD_SIZE="${SLURM_NTASKS}"
export MASTER_ADDR="${MASTER_ADDR:-$(head -n1 <<<"${nodes}")}"
export MASTER_PORT="${MASTER_PORT:-$((20000 + SLURM_JOB_ID % 20000))}"
exec "$@"
