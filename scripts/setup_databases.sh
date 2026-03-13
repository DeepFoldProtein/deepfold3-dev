#!/bin/bash -e
#
# Download and prepare MMseqs2 databases for DeepFold3 local MSA search.
#
# Usage:
#   ./scripts/setup_databases.sh /path/to/db_folder          # CPU-only
#   GPU=1 ./scripts/setup_databases.sh /path/to/db_folder    # GPU+CPU compatible
#
# Options (environment variables):
#   GPU=1               Build GPU-compatible databases (requires MMseqs2 >= release 16)
#   MMSEQS_NO_INDEX=1   Skip index creation
#   DOWNLOADS_ONLY=1    Download tarballs only, do not extract or index
#   SKIP_TEMPLATES=1    Skip PDB100 FoldSeek template download
#
# Databases (~500 GB total):
#   uniref30_2302_db             Primary sequence database for MSA search
#   colabfold_envdb_202108_db    Environmental/metagenomic sequences (--use-env)
#   pdb100_230517                PDB sequences for template search (--use-templates)
#   pdb100_foldseek_230517       Structural template profiles
#
# The script is idempotent: each step writes a sentinel file (*_READY) on
# completion so it can be safely re-run after interruption.
#
# Adapted from ColabFold setup_databases.sh
# https://github.com/sokrypton/ColabFold

set -euo pipefail

WORKDIR="${1:?Usage: $0 /path/to/db_folder}"
ARIA_NUM_CONN=8
MMSEQS_NO_INDEX="${MMSEQS_NO_INDEX:-}"
DOWNLOADS_ONLY="${DOWNLOADS_ONLY:-}"
GPU="${GPU:-}"

mkdir -p -- "${WORKDIR}"
cd "${WORKDIR}"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

fail() { echo "Error: $1" >&2; exit 1; }

has() { command -v "$1" >/dev/null 2>&1; }

has mmseqs || fail "mmseqs not found in PATH. Install MMseqs2 (https://mmseqs.com)."

# Pick the best available download tool.
DOWNLOAD_CMD=""
if   has aria2c; then DOWNLOAD_CMD="aria2c"
elif has curl;   then DOWNLOAD_CMD="curl"
elif has wget;   then DOWNLOAD_CMD="wget"
else fail "No download tool found. Install aria2c, curl, or wget."; fi

download() {
    local url="$1" output="$2"
    echo "Downloading ${url} -> ${output}"
    case "${DOWNLOAD_CMD}" in
        aria2c) aria2c --max-connection-per-server="${ARIA_NUM_CONN}" \
                       --allow-overwrite=true \
                       -o "$(basename "${output}")" \
                       -d "$(dirname "${output}")" "${url}" ;;
        curl)   curl -L -o "${output}" "${url}" ;;
        wget)   wget -O "${output}" "${url}" ;;
    esac
}

# ---------------------------------------------------------------------------
# 1. Download
# ---------------------------------------------------------------------------

if [ ! -f DOWNLOADS_READY ]; then
    # Pre-built databases that work with both GPU and CPU.
    download "https://opendata.mmseqs.org/colabfold/uniref30_2302.db.tar.gz" \
             "uniref30_2302.tar.gz"
    download "https://opendata.mmseqs.org/colabfold/colabfold_envdb_202108.db.tar.gz" \
             "colabfold_envdb_202108.tar.gz"
    download "https://opendata.mmseqs.org/colabfold/uniref30_2302_newtaxonomy.tar.gz" \
             "uniref30_2302_newtaxonomy.tar.gz"
    download "https://opendata.mmseqs.org/colabfold/pdb100_230517.fasta.gz" \
             "pdb100_230517.fasta.gz"
    if [ -z "${SKIP_TEMPLATES:-}" ]; then
        download "https://opendata.mmseqs.org/colabfold/pdb100_foldseek_230517.tar.gz" \
                 "pdb100_foldseek_230517.tar.gz"
    fi
    touch DOWNLOADS_READY
fi

[ -n "${DOWNLOADS_ONLY}" ] && { echo "Downloads complete."; exit 0; }

# ---------------------------------------------------------------------------
# 2. Process databases
# ---------------------------------------------------------------------------

# Merge small DB files into single data+index pairs.
export MMSEQS_FORCE_MERGE=1

GPU_INDEX_PAR=""
if [ -n "${GPU}" ]; then
    mmseqs --help 2>&1 | grep -q 'gpuserver' \
        || fail "Installed MMseqs2 has no GPU support. Update to release 16+."
    GPU_INDEX_PAR="--split 1 --index-subset 2"
fi

# --- UniRef30 ---
if [ ! -f UNIREF30_READY ]; then
    echo "Processing UniRef30..."
    tar xzf uniref30_2302.tar.gz

    if [ -z "${MMSEQS_NO_INDEX}" ]; then
        mmseqs createindex uniref30_2302_db tmp1 --remove-tmp-files 1 ${GPU_INDEX_PAR}
    fi

    # Updated taxonomy/pairing files
    if [ -f uniref30_2302_newtaxonomy.tar.gz ]; then
        tar xzf uniref30_2302_newtaxonomy.tar.gz
    fi
    if [ -e uniref30_2302_db_mapping ]; then
        # Convert text mapping to binary mmap format if needed.
        TAXHEADER=$(od -An -N4 -t x4 uniref30_2302_db_mapping | tr -d ' ')
        if [ "${TAXHEADER}" != "0c170013" ]; then
            mmseqs createbintaxmapping uniref30_2302_db_mapping uniref30_2302_db_mapping.bin
            mv -f uniref30_2302_db_mapping.bin uniref30_2302_db_mapping
        fi
        ln -sf uniref30_2302_db_mapping uniref30_2302_db.idx_mapping
    fi
    if [ -e uniref30_2302_db_taxonomy ]; then
        ln -sf uniref30_2302_db_taxonomy uniref30_2302_db.idx_taxonomy
    fi

    touch UNIREF30_READY
    echo "UniRef30 ready."
fi

# --- ColabFold EnvDB ---
if [ ! -f COLABDB_READY ]; then
    echo "Processing ColabFold EnvDB..."
    tar xzf colabfold_envdb_202108.tar.gz

    if [ -z "${MMSEQS_NO_INDEX}" ]; then
        mmseqs createindex colabfold_envdb_202108_db tmp2 --remove-tmp-files 1 ${GPU_INDEX_PAR}
    fi

    touch COLABDB_READY
    echo "ColabFold EnvDB ready."
fi

# --- PDB100 (template search) ---
if [ ! -f PDB_READY ]; then
    echo "Processing PDB100..."
    # GPU and pre-built databases need padded sequence DB.
    if [ -n "${GPU}" ]; then
        mmseqs createdb pdb100_230517.fasta.gz pdb100_230517_tmp
        mmseqs makepaddedseqdb pdb100_230517_tmp pdb100_230517
        mmseqs rmdb pdb100_230517_tmp
    else
        mmseqs createdb pdb100_230517.fasta.gz pdb100_230517
    fi

    if [ -z "${MMSEQS_NO_INDEX}" ]; then
        mmseqs createindex pdb100_230517 tmp3 --remove-tmp-files 1 ${GPU_INDEX_PAR}
    fi

    touch PDB_READY
    echo "PDB100 ready."
fi

# --- PDB100 FoldSeek templates ---
if [ ! -f PDB100_READY ] && [ -z "${SKIP_TEMPLATES:-}" ]; then
    echo "Extracting PDB100 FoldSeek templates..."
    tar xzf pdb100_foldseek_230517.tar.gz pdb100_a3m.ffdata pdb100_a3m.ffindex
    touch PDB100_READY
    echo "PDB100 FoldSeek ready."
fi

echo ""
echo "All databases ready in ${WORKDIR}"
