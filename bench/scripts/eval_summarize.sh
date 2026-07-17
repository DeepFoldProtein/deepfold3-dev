#!/bin/bash
# OST-score a set of df3 steps and (re)build the FoldBench trajectory plots.
#
# Runs on CPU only (OST via the apptainer image; see eval_ost.py). Safe to run
# directly on a node/login shell or as the body of a Slurm job. Resumable:
# eval_ost skips comparisons whose detail JSON already exists.
#
# Usage:
#   bench/scripts/eval_summarize.sh SERIES "STEP...(space list)" "CATEGORY...(space list)" [WORKERS]
# e.g.
#   bench/scripts/eval_summarize.sh df3v3 "6000 5000" "monomer_protein interface_protein_protein" 32
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
PY=.venv/bin/python

SERIES="${1:-df3v3}"
STEPS="${2:-6000 5000}"
CATS="${3:-monomer_protein monomer_rna monomer_dna interface_protein_protein interface_antibody_antigen interface_protein_dna interface_protein_rna interface_protein_peptide}"
WORKERS="${4:-32}"

# OST image (overridable). eval_ost.py already defaults to this; set here so a
# direct run on any node is self-contained.
export OST_SIF="${OST_SIF:-/store/deepfold/casp17_sync/benchmark/foldbench/ost.sif}"

# DSSP (mkdssp v4) for the fold-class breakdown. dssp_classify -> dssp_f1 honors
# $MKDSSP first, else PATH. Point it at the local conda-forge install (shared on
# /store so Slurm nodes see it); harmless if absent (breakdown is skipped).
_MKDSSP="/store/deepfold/users/vv137/foldbench/miniconda3/bin/mkdssp"
[ -x "$_MKDSSP" ] && export MKDSSP="$_MKDSSP"

echo "[eval] series=$SERIES steps='$STEPS' workers=$WORKERS"
echo "[eval] categories: $CATS"

# 1. Score every (step, category) with OpenStructure.
for step in $STEPS; do
  for c in $CATS; do
    echo "=== OST eval  step=$step  category=$c ==="
    $PY bench/scripts/eval_ost.py --series "$SERIES" --step "$step" --category "$c" --workers "$WORKERS"
  done
done

# 2. DSSP fold classes for monomer_protein (optional — needs mkdssp on PATH).
if command -v mkdssp >/dev/null 2>&1; then
  echo "=== DSSP classify monomer_protein ==="
  $PY bench/scripts/dssp_classify.py --category monomer_protein || echo "[eval] dssp_classify failed (non-fatal)"
else
  echo "[eval] mkdssp not found — skipping DSSP fold-class breakdown (trajectories still built)"
fi

# 3. Rebuild per-category trajectory CSVs + plots across ALL scored steps
#    (history steps 1000-4000 + whatever we just scored).
for c in $CATS; do
  echo "=== summarize $c ==="
  $PY bench/scripts/summarize.py --series "$SERIES" --category "$c" || echo "[eval] summarize $c failed (non-fatal)"
done

# 4. Order-parameter analyses over the prediction CIFs (all steps auto-globbed;
#    no OST needed). rg_scaling = Rg~N^b collapse exponent b(step) for monomers;
#    rg_decompose = intra/inter-chain Rg decomposition for the ppi interface.
#
# These read prediction CIFs, which the CSV-only history (steps 1000-4000) does
# not carry, so a plain run would recompute only the steps with CIFs and drop
# the historical rows. run_rg() preserves history: it backs up the existing CSV
# and merges back any step the fresh run couldn't recompute, so the b(step)
# series stays complete in the CSV (the scatter PNG still only draws steps whose
# CIFs exist here — future steps that retain their CIFs fill it in fully).
run_rg() {  # $1=script  $2=category  $3=csv_basename
  local csv="bench/runs/$SERIES/$3" bak
  bak="$(mktemp)"; [ -f "$csv" ] && cp "$csv" "$bak"
  $PY "bench/scripts/$1" --series "$SERIES" --category "$2" || echo "[eval] $1 failed (non-fatal)"
  if [ -s "$bak" ] && [ -f "$csv" ]; then
    $PY - "$bak" "$csv" <<'PYEOF'
import csv, sys
old, new = sys.argv[1], sys.argv[2]
rd = lambda p: list(csv.reader(open(p)))
o, n = rd(old), rd(new)
if not n:
    sys.exit(0)
hdr = n[0]; si = hdr.index("step")
have = {r[si] for r in n[1:]}
rows = n[1:] + [r for r in o[1:] if r and r[si] not in have]
rows.sort(key=lambda r: int(r[si]))
w = csv.writer(open(new, "w", newline="")); w.writerow(hdr); w.writerows(rows)
print(f"[merge] {new}: {len(n)-1} recomputed + {len(rows)-(len(n)-1)} preserved from history")
PYEOF
  fi
  rm -f "$bak"
}

case " $CATS " in *" monomer_protein "*)
  echo "=== rg_scaling monomer_protein ==="
  run_rg rg_scaling.py monomer_protein rg_scaling_monomer_protein.csv
esac
case " $CATS " in *" interface_protein_protein "*)
  echo "=== rg_decompose interface_protein_protein ==="
  run_rg rg_decompose.py interface_protein_protein rg_decompose_interface_protein_protein.csv
esac

echo "[eval] done. Trajectories + PNGs under bench/runs/$SERIES/"
