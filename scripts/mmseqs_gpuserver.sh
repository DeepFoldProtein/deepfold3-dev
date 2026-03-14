#!/bin/bash -e
# ═══════════════════════════════════════════════════════════════════════════════
# GPU server startup — from soedinglab/MMseqs2-App backend/worker.go
# lines 1961–2001 (ColabFold app, gpuserver startup goroutines).
#
# The Go worker starts persistent mmseqs gpuserver processes for each DB
# at server startup. Each gpuserver keeps the DB index in GPU memory and
# serves prefilter requests over a Unix socket, saving ~1.5s GPU init
# per search request.
#
# Usage:
#   # Start all GPU servers (run in background before submitting jobs)
#   ./mmseqs_gpuserver.sh /data/dbs/
#
#   # With per-DB GPU device assignment
#   UNIREF_CUDA_VISIBLE_DEVICES=0 \
#   PDB_CUDA_VISIBLE_DEVICES=1 \
#   ENV_CUDA_VISIBLE_DEVICES=0,1 \
#   ./mmseqs_gpuserver.sh /data/dbs/
# ═══════════════════════════════════════════════════════════════════════════════
MMSEQS="${MMSEQS:-mmseqs}"
DB_DIR="${1:?Usage: $0 <db_dir>}"

# ── UniRef30 gpuserver (--max-seqs 10000) ─────────────────────────────────────
(
  if [ -n "${UNIREF_CUDA_VISIBLE_DEVICES}" ]; then
    export CUDA_VISIBLE_DEVICES="${UNIREF_CUDA_VISIBLE_DEVICES}"
  fi
  "${MMSEQS}" gpuserver "${DB_DIR}/uniref30_2302_db.idx" --db-load-mode 0 --max-seqs 10000
) &

# ── PDB100 gpuserver (--max-seqs 300) ─────────────────────────────────────────
if [ -e "${DB_DIR}/pdb100_230517.idx.dbtype" ] || [ -e "${DB_DIR}/pdb100_230517.dbtype" ]; then
  (
    if [ -n "${PDB_CUDA_VISIBLE_DEVICES}" ]; then
      export CUDA_VISIBLE_DEVICES="${PDB_CUDA_VISIBLE_DEVICES}"
    fi
    "${MMSEQS}" gpuserver "${DB_DIR}/pdb100_230517.idx" --db-load-mode 0 --max-seqs 300
  ) &
fi

# ── Environmental DB gpuserver (--max-seqs 10000) ─────────────────────────────
if [ -e "${DB_DIR}/colabfold_envdb_202108_db.idx.dbtype" ] || [ -e "${DB_DIR}/colabfold_envdb_202108_db.dbtype" ]; then
  (
    if [ -n "${ENV_CUDA_VISIBLE_DEVICES}" ]; then
      export CUDA_VISIBLE_DEVICES="${ENV_CUDA_VISIBLE_DEVICES}"
    fi
    "${MMSEQS}" gpuserver "${DB_DIR}/colabfold_envdb_202108_db.idx" --db-load-mode 0 --max-seqs 10000
  ) &
fi

echo "GPU servers started. PIDs:"
jobs -p
wait
