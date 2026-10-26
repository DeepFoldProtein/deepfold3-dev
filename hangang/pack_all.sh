#!/usr/bin/env bash
# Build the packed staging tree that is copied to Hangang (KISTI 6th system).
#
# Hangang's Lustre /scratch allows 5M files per group; the unpacked training
# data is ~63M files. This writes a tree laid out exactly like the trainjob
# volume /data, in which every bulky subtree is replaced by a `_pack/`
# (tar shards + index, see src/deepfold3/data/pack.py) and only the small
# files the loaders open directly stay plain (manifest.json, ids.txt,
# df3_cache/, the CCD). Copy that tree as-is; nothing else is needed.
#
# Usage (on the host that sees /store and /gpfs):
#   bash hangang/pack_all.sh                       # every step, resumable
#   ONLY="patchr_training afdb_nvda" bash hangang/pack_all.sh
#   SRC=/store/deepfold3/data DEST=/store/deepfold3/packed WORKERS=48 bash hangang/pack_all.sh
#
# Every pack step is resumable (re-run the same command); finished packs are
# skipped. Logs: $DEST/_logs/<step>.log. `_build/` dirs hold resume state and
# are NOT transferred.

set -euo pipefail

SRC="${SRC:-/store/deepfold3/data}"
DEST="${DEST:-/store/deepfold3/packed}"
WORKERS="${WORKERS:-32}"
IO_THREADS="${IO_THREADS:-16}"
ONLY="${ONLY:-}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
LOG="${DEST}/_logs"
DF3=(uv run --project "${REPO}" deepfold3)

mkdir -p "${DEST}" "${LOG}"

want() { [[ -z "${ONLY}" || " ${ONLY} " == *" $1 "* ]]; }

# pack <step> <relative data root> [deepfold3 pack options...]
pack() {
    local step=$1 rel=$2
    shift 2
    want "${step}" || return 0
    mkdir -p "${DEST}/${rel}"
    echo "=== [${step}] pack ${SRC}/${rel} -> ${DEST}/${rel}/_pack" | tee -a "${LOG}/${step}.log"
    "${DF3[@]}" pack --src "${SRC}/${rel}" --out "${DEST}/${rel}/_pack" \
        --workers "${WORKERS}" --io-threads "${IO_THREADS}" "$@" 2>&1 | tee -a "${LOG}/${step}.log"
    "${DF3[@]}" pack-verify --pack "${DEST}/${rel}/_pack" --src "${SRC}/${rel}" --sample 2000 \
        2>&1 | tee -a "${LOG}/${step}.log"
}

# plain <step> <relative dir> <entry>... : copy small files/dirs as real files
plain() {
    local step=$1 rel=$2
    shift 2
    want "${step}" || return 0
    mkdir -p "${DEST}/${rel}"
    for e in "$@"; do
        echo "=== [${step}] copy ${SRC}/${rel}/${e}" | tee -a "${LOG}/${step}.log"
        cp -rL --preserve=timestamps "${SRC}/${rel}/${e}" "${DEST}/${rel}/"
    done
}

# --- top level: CCD and mmCIF metadata -------------------------------------
plain ccd . components.cif.gz ccd_cache.json mmcif_meta.json

# --- PDB: template chain cache (mmCIF + assemblies) and the mmCIF mirror ----
# The cache serves every template chain at train time; the of3 sets parse
# mmCIF directly, so divided/ is packed too. assemblies/ is only reached
# through the cache and is not needed.
pack pdb_cache pdb/cache --shard-size 4G
pack pdb_divided pdb/divided --shard-size 2G

# --- OF3 weighted-PDB set (format: of3pdb) ---------------------------------
pack pdb_training_set pdb_training_set --shard-size 4G \
    --include preprocessed_pdb_data/standard/structure_files \
    --include alignment_arrays \
    --include templates/template_structure_arrays \
    --include templates/train_template_cache \
    --include templates/val_template_cache
plain pdb_training_set pdb_training_set df3_cache df3_cache_val

# --- df3 distillation sets --------------------------------------------------
# afdb_nvda ships a {ab}.tar.gz archive next to every shard dir; skip them.
pack afdb_nvda distillation/afdb_nvda --shard-size 4G \
    --include structures --include msa --include monomer_templates --exclude '*.tar.gz'
plain afdb_nvda distillation/afdb_nvda manifest.json README.md

pack teddymer distillation/teddymer --shard-size 2G \
    --include structures --include msa --include monomer_templates
plain teddymer distillation/teddymer manifest.json

pack patchr_training distillation/patchr_training --shard-size 4G \
    --include structures --include msa --include monomer_templates --include masks
plain patchr_training distillation/patchr_training manifest.json clusters.txt sequences.fasta

# --- OF3 monomer distillation (format: of3, flat id directories) -----------
# ids.txt lives in $SRC/ids/ (the /gpfs source is over quota); it becomes
# preprocessed/ids.txt next to the pack.
for set in short_monomers long_monomers; do
    size=8G
    [[ ${set} == short_monomers ]] && size=2G
    if want "${set}"; then
        pack "${set}" "distillation/${set}/preprocessed" --shard-size "${size}" \
            --ids-file "${SRC}/ids/${set}.ids.txt" \
            --member structure.npz --member alignment.npz --member template.npz
        cp --preserve=timestamps "${SRC}/ids/${set}.ids.txt" "${DEST}/distillation/${set}/preprocessed/ids.txt"
    fi
done

# --- summary: what will be transferred --------------------------------------
cd "${DEST}"
files=$(find . -path ./_logs -prune -o -name _build -prune -o -type f -print | wc -l)
bytes=$(find . -path ./_logs -prune -o -name _build -prune -o -type f -printf '%s\n' | awk '{s+=$1} END {print s+0}')
printf 'Transfer set under %s: %d files, %.2f TiB\n' "${DEST}" "${files}" "$(echo "${bytes} / 1099511627776" | bc -l)" \
    | tee -a "${LOG}/summary.log"
