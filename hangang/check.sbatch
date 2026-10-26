#!/bin/bash
# Pre-flight on one GH200 node: GPUs + NCCL, pack integrity, data smoke test.
# Run after transfer.sh verify and setup_env.sh, before the first training job:
#
#   cd /scratch/$USER/deepfold3/deepfold3-dev      # jobs must be submitted from /scratch
#   sbatch --comment="$(source hangang/env.sh; df3_slurm_comment)" hangang/check.sbatch
#   CHECKSUMS=1 sbatch ... hangang/check.sbatch    # also re-hash every shard (slow: reads all data)
#
#SBATCH --job-name=df3-check
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail
source "${SLURM_SUBMIT_DIR}/hangang/env.sh"
cd "${REPO}"
PY="${REPO}/.venv/bin/python"
DF3="${REPO}/.venv/bin/deepfold3"

echo "=== 1. GPUs"
nvidia-smi --query-gpu=name,memory.total --format=csv
"${PY}" -c "import torch; assert torch.cuda.device_count() == 4, torch.cuda.device_count(); print('cuda ok:', torch.cuda.get_device_name(0))"

echo "=== 2. NCCL all-reduce across the node's 4 GPUs (srun + rank wrapper)"
srun --mpi=none --gpu-bind=none --ntasks=4 "${REPO}/hangang/srun_rank.sh" "${PY}" - <<'EOF'
import os, time, torch, torch.distributed as dist
r, w = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
dist.init_process_group("nccl", device_id=torch.device("cuda", int(os.environ["LOCAL_RANK"])))
x = torch.ones(256 << 20, device="cuda")  # 1 GiB fp32
for _ in range(3): dist.all_reduce(x)
torch.cuda.synchronize(); t = time.perf_counter()
for _ in range(10): dist.all_reduce(x)
torch.cuda.synchronize(); dt = (time.perf_counter() - t) / 10
assert float(x[0]) == w ** 13, float(x[0])
if r == 0: print(f"all_reduce 1 GiB x{w}: {dt*1e3:.1f} ms ({2*(w-1)/w*x.numel()*4/dt/1e9:.0f} GB/s bus)")
dist.destroy_process_group()
EOF

echo "=== 3. Packs"
find "${DATA}" -path '*/_pack/meta.json' -printf '%h\n' | sort | while read -r p; do
    extra=()
    [[ "${CHECKSUMS:-0}" == 1 ]] && extra=(--checksums --workers 64)
    "${DF3}" pack-verify --pack "${p}" --sample 500 "${extra[@]}"
done
files=$(find "${DF3_ROOT}" -type f | wc -l)
echo "files under ${DF3_ROOT}: ${files} (group quota 5M)"
lfs quota -h -g "$(id -gn)" /scratch || true

echo "=== 4. Data smoke test (training DataLoader on the packed tree)"
df3_make_config "${REPO}/configs/initial.yaml" "${RUNS}/check-${SLURM_JOB_ID}/config.yaml"
grep -n "/data/" "${RUNS}/check-${SLURM_JOB_ID}/config.yaml" | grep -v "^\s*[0-9]*:\s*#" || true
"${PY}" "${REPO}/hangang/smoke_data.py" --config "${RUNS}/check-${SLURM_JOB_ID}/config.yaml" \
    --ccd "${DATA}/components.cif.gz" --ccd-cache "${DATA}/ccd_cache.json" \
    --mmcif-dir "${DATA}/pdb/divided" --batches 50 --workers 16
echo "=== check passed"
