# Sourced by every hangang/ script and Slurm job. Override any variable in the
# environment before sourcing; the defaults assume this layout on /scratch:
#
#   /scratch/$USER/deepfold3/
#   ├── data/            the packed tree from hangang/transfer.sh (mirrors /data)
#   ├── deepfold3-dev/   this repo (+ .venv, aarch64)
#   ├── runs/            training outputs / checkpoints
#   └── bin/             uv
#
# /scratch is purged by atime after 15 days without access (see keepalive.sh)
# and allows 5M files per *group*: keep the venv and runs lean.

export DF3_ROOT="${DF3_ROOT:-/scratch/${USER}/deepfold3}"
export DATA="${DATA:-${DF3_ROOT}/data}"
export REPO="${REPO:-${DF3_ROOT}/deepfold3-dev}"
export RUNS="${RUNS:-${DF3_ROOT}/runs}"
export PATH="${DF3_ROOT}/bin:${PATH}"

# uv: wheel cache in $HOME (500K-file quota, not purged), venv on /scratch.
# Hard links cannot cross file systems, so copy into the venv.
export UV_CACHE_DIR="${UV_CACHE_DIR:-${HOME}/.cache/uv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-${HOME}/.local/share/uv/python}"
export UV_LINK_MODE=copy

# Unbuffered Python output: under srun stdout is a pipe, so Python block-buffers
# it (~8 KB) and the job's .out file lags far behind what the ranks have done
# (rank 0's progress lines sit unflushed). env.sh is sourced by every job
# script and srun passes the environment to the ranks.
export PYTHONUNBUFFERED=1

# Persistent Triton autotune cache (``deepfold3 warmup-triton`` fills it, every
# training rank reads it at import). $HOME: visible on every node, not purged,
# and the cache is a few dozen JSON files. Keyed by GPU model, so one warmup
# serves every GH200 job; see hangang/warmup.sbatch.
export DEEPFOLD3_TRITON_CACHE_DIR="${DEEPFOLD3_TRITON_CACHE_DIR:-${HOME}/.cache/deepfold3-triton}"

# CUDA build of torch for the venv. The lockfile pins PyPI's torch, which is
# the CUDA 13 build; Hangang's driver supports CUDA 12.8 (driver API 12080),
# so setup_env.sh swaps in the same torch version from PyTorch's cu126 index.
# Set to "" to keep the lockfile's build (e.g. after a driver upgrade).
export DF3_TORCH_CUDA="${DF3_TORCH_CUDA-cu126}"

# Slurm accounting: every job must carry --comment="field=...;appl=..." with
# values from `showappl` on a login node (AI-for-science, PyTorch DDP). The
# same default is baked into the #SBATCH headers of check.sbatch/train.sbatch,
# so keep them in sync if you change it; submit.sh passes this one explicitly.
export DF3_SLURM_FIELD="${DF3_SLURM_FIELD:-sciai}"
export DF3_SLURM_APPL="${DF3_SLURM_APPL:-pytorch-ddp}"

export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

# Multi-node setup over HPE Slingshot, called by train.sbatch when the job has
# more than one node (never on a single node: KISTI notes the OFI plugin can
# fail NCCL init when no Slingshot VNI is allocated). Follows
# docs-ksc.gitbook.io/hangang-user-guide/software/{nccl-slingshot,pytorch}:
#   - aws-ofi-nccl/1.20.0 + libfabric/2.3.1: the CXI net plugin. The venv's
#     NCCL wheel ships none and would fall back to TCP sockets. The module also
#     sets NCCL_SOCKET_IFNAME=hsn0..3 and FI_CXI_* defaults.
#   - NCCL_PROTO=^LL128: LL128 gave wrong multi-node AllReduce results in
#     KISTI's validation.
#   - NCCL_DEBUG=INFO with INIT,NET: the job log must show
#     "NCCL INFO Using network AWS Libfabric" ("Using network Socket" = the
#     plugin did not load).
# Rendezvous: MASTER_ADDR stays the first node's hostname, as KISTI's PyTorch
# example sets it (srun_rank.sh). Hostnames resolve to the management network
# (bond0, *.hostmgmt), not Slingshot. DF3_RDZV_HSN=1 points MASTER_ADDR (the
# TCPStore every rank dials at init) and Gloo at the first node's hsn0 address
# instead: the fallback if init again times out connecting to <host>:<port>.
df3_slingshot_env() {
    set +u
    if ! type module >/dev/null 2>&1; then
        for init in /etc/profile.d/lmod.sh /etc/profile.d/modules.sh /usr/share/lmod/lmod/init/bash; do
            [[ -f "${init}" ]] && source "${init}" && break
        done
    fi
    type module >/dev/null 2>&1 || { echo "df3_slingshot_env: no 'module' command" >&2; return 1; }
    module purge || true  # Lmod may warn about sticky modules
    module load aws-ofi-nccl/1.20.0 libfabric/2.3.1 || { set -u; return 1; }
    set -u
    export NCCL_NET="AWS Libfabric"
    export FI_PROVIDER=cxi
    export NCCL_IB_DISABLE=1
    export NCCL_PROTO=^LL128
    export NCCL_DEBUG="${DF3_NCCL_DEBUG:-INFO}"
    export NCCL_DEBUG_SUBSYS=INIT,NET

    # GPUDirect RDMA. Cray's libfabric dlopens the unversioned "libcudart.so"
    # for CUDA HMEM; the venv's CUDA runtime wheel ships only libcudart.so.12,
    # so HMEM init failed and aws-ofi-nccl logged "Could not disable CUDA API
    # usage for HMEM, disabling GDR" (every transfer staged through host
    # memory). A symlink to the same file torch already loaded keeps a single
    # cudart in the process. GDRCOPY / GDR_LEVEL follow Isambard-AI's GH200 +
    # Slingshot NCCL defaults. Check the log for "GPU Direct RDMA Enabled".
    local cudart
    cudart=$(ls "${REPO}"/.venv/lib/python3*/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12 2>/dev/null | head -n1)
    if [[ -n "${cudart}" ]]; then
        mkdir -p "${DF3_ROOT}/lib/cudart"
        ln -sfn "${cudart}" "${DF3_ROOT}/lib/cudart/libcudart.so"
        export LD_LIBRARY_PATH="${DF3_ROOT}/lib/cudart${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    else
        echo "df3_slingshot_env: no libcudart.so.12 in the venv; GDR stays off" >&2
    fi
    export FI_HMEM_CUDA_USE_GDRCOPY=1
    export NCCL_GDRCOPY_ENABLE=1
    export NCCL_NET_GDR_LEVEL=PHB

    if [[ "${DF3_RDZV_HSN:-0}" != 1 ]]; then
        echo "slingshot: $(module -t list 2>&1 | tr '\n' ' '); MASTER_ADDR = first node's hostname"
        return 0
    fi
    export GLOO_SOCKET_IFNAME=hsn0
    local first addr
    first=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)
    if [[ "$(hostname -s)" == "${first}" ]]; then
        addr=$(ip -4 -o addr show hsn0 | awk '{print $4}' | cut -d/ -f1 | head -n1)
    else
        addr=$(srun --overlap -N1 -n1 -w "${first}" ip -4 -o addr show hsn0 | awk '{print $4}' | cut -d/ -f1 | head -n1)
    fi
    [[ -n "${addr}" ]] || { echo "df3_slingshot_env: no hsn0 address on ${first}" >&2; return 1; }
    export MASTER_ADDR="${addr}"
    echo "slingshot: MASTER_ADDR=${MASTER_ADDR} (${first} hsn0), $(module -t list 2>&1 | tr '\n' ' ')"
}

df3_slurm_comment() { echo "field=${DF3_SLURM_FIELD};appl=${DF3_SLURM_APPL}"; }

# Freeze a config into a self-contained per-run copy: resolve its `base:`
# chain (a relative `base:` would otherwise be looked up next to the copy, in
# the run directory) and rewrite /data/... paths to $DATA (configs are written
# for the trainjob volume /data; a list-valued `datasets:` cannot be patched
# through `base:` inheritance). Written via a temp file so a failure never
# leaves a half-written config for the next chain link to reuse.
df3_make_config() {
    local src=$1 dst=$2
    mkdir -p "$(dirname "${dst}")"
    "${REPO}/.venv/bin/python" - "${src}" "${dst}.tmp" <<'PY' || { rm -f "${dst}.tmp"; return 1; }
import sys, yaml
from deepfold3.config_loader import resolve_config
with open(sys.argv[2], "w") as f:
    f.write(f"# Frozen from {sys.argv[1]} (base chain resolved)\n")
    yaml.safe_dump(resolve_config(sys.argv[1]), f, default_flow_style=False, sort_keys=False)
PY
    sed -i "s#\([[:space:]:]\)/data/#\1${DATA}/#g" "${dst}.tmp"
    mv "${dst}.tmp" "${dst}"
}
