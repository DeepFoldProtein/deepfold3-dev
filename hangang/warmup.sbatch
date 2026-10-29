#!/bin/bash
# Fill the persistent Triton autotune cache for a training config, once, on one
# GH200 node — so the 256 training ranks load the tuned kernel configs instead
# of each benchmarking every (kernel, shape) on its first steps. Run after
# setup_env.sh (and check.sbatch), before submit.sh, and again after changing
# the model widths, max_tokens or batch_size_per_rank:
#
#   cd /scratch/$USER/deepfold3/deepfold3-dev
#   source hangang/env.sh
#   CONFIG_SRC=configs/stage2.yaml sbatch --comment="$(df3_slurm_comment)" hangang/warmup.sbatch
#
# --real-model drives the real forward_training + backward at the config's own
# shapes (max_tokens, batch_size_per_rank, diffusion batch/chunk), with every
# block stack collapsed to depth 1 — identical shape coverage, a fraction of
# the time. Already-cached shapes are skipped, so a re-run is cheap. The cache
# lands in $DEEPFOLD3_TRITON_CACHE_DIR (env.sh).
#
# Its own job rather than a step of submit.sh: the queue allows 5 active jobs
# per user and a full chain already uses all 5.
#
#SBATCH --job-name=df3-warmup
#SBATCH --partition=normal
#SBATCH --comment="field=sciai;appl=pytorch-ddp"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail
source "${SLURM_SUBMIT_DIR}/hangang/env.sh"
cd "${REPO}"

CONFIG_SRC="${CONFIG_SRC:-configs/initial.yaml}"
echo "warming ${CONFIG_SRC} -> ${DEEPFOLD3_TRITON_CACHE_DIR}"
"${REPO}/.venv/bin/python" -c "import torch; print('torch', torch.__version__, torch.cuda.get_device_name(0))"

"${REPO}/.venv/bin/deepfold3" warmup-triton --real-model --config "${CONFIG_SRC}"

echo "cache: $(ls "${DEEPFOLD3_TRITON_CACHE_DIR}" | wc -l) files in ${DEEPFOLD3_TRITON_CACHE_DIR}"
echo "=== warmup done"
