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

# Slurm accounting: every job must carry --comment="field=...;appl=..." with
# values from `showappl` on a login node (AI-for-science, PyTorch DDP). The
# same default is baked into the #SBATCH headers of check.sbatch/train.sbatch,
# so keep them in sync if you change it; submit.sh passes this one explicitly.
export DF3_SLURM_FIELD="${DF3_SLURM_FIELD:-sciai}"
export DF3_SLURM_APPL="${DF3_SLURM_APPL:-pytorch-ddp}"

# Multi-node NCCL over HPE Slingshot needs the libfabric (OFI) net plugin with
# the CXI provider; without it NCCL silently falls back to TCP sockets. The
# venv's NCCL (nvidia-nccl-cu13 wheel) does not ship the plugin, so point
# NCCL at the system one. Find it with
#   /apps/common/kisti-container/bin/kisti-container --doctor
# or `module avail aws-ofi-nccl`, and set DF3_OFI_NCCL_LIB to the directory
# holding libnccl-net*.so. Confirm in a job log: "Selected provider is cxi".
export DF3_OFI_NCCL_LIB="${DF3_OFI_NCCL_LIB:-}"
if [[ -n "${DF3_OFI_NCCL_LIB}" ]]; then
    export LD_LIBRARY_PATH="${DF3_OFI_NCCL_LIB}:${LD_LIBRARY_PATH:-}"
    export NCCL_NET="${NCCL_NET:-AWS Libfabric}"
    export FI_PROVIDER="${FI_PROVIDER:-cxi}"
fi
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

df3_slurm_comment() { echo "field=${DF3_SLURM_FIELD};appl=${DF3_SLURM_APPL}"; }

# Rewrite a config's /data/... paths to $DATA (configs are written for the
# trainjob volume /data; a list-valued `datasets:` cannot be patched through
# `base:` inheritance, so the paths are substituted in a per-run copy).
df3_make_config() {
    local src=$1 dst=$2
    mkdir -p "$(dirname "${dst}")"
    sed "s#\([[:space:]:]\)/data/#\1${DATA}/#g" "${src}" >"${dst}"
}
