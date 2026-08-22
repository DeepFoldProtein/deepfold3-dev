"""Set up a single inference benchmark target.

Given a PDB ID, this script:
  1. Downloads the reference mmCIF from RCSB → ``{out_dir}/{pdb}/_ref.cif.gz``
  2. Extracts the protein sequence (chain A by default) from ``_entity_poly``
  3. Generates an MSA + template hits via the ColabFold server
     (``deepfold3 msa --mode server``) → ``{out_dir}/{pdb}/msa_out/``
  4. Writes a ``query.json`` referencing the MSA / template paths

After all targets are set up, inference can be invoked with the standard
``deepfold3 inference --json {out_dir}/{pdb}/query.json --output ...
--params ... --num-samples 30``.

Skips any step whose output already exists, so the script is idempotent.

Usage:
    uv run python scripts/setup_bench_target.py --pdb-id 9GZD --out-dir tmp/bench
    # batch:
    for ID in 9GZD 9L86 29MW 9EDF 9OT9 26LN 9E9P; do
        uv run python scripts/setup_bench_target.py --pdb-id $ID --out-dir tmp/bench
    done
"""

from __future__ import annotations

import argparse
import gzip
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

import gemmi


def download_ref(pdb_id: str, dest: Path) -> None:
    if dest.exists():
        return
    pdb_lower = pdb_id.lower()
    url = f"https://files.rcsb.org/download/{pdb_lower}.cif.gz"
    print(f"[{pdb_id}] downloading {url}")
    urllib.request.urlretrieve(url, dest)


def extract_sequence(cif_gz: Path, chain_id: str = "A") -> tuple[str, list[tuple[int, int]]]:
    """Return (one-letter sequence of chain A entity, list of helix
    ``(beg_label_seq, end_label_seq)`` for sanity).
    """
    with gzip.open(cif_gz, "rt") as f:
        data = f.read()
    doc = gemmi.cif.read_string(data)
    block = doc.sole_block()
    seq_pair = block.find_pair("_entity_poly.pdbx_seq_one_letter_code_can")
    if seq_pair is None:
        raise RuntimeError(f"no _entity_poly sequence in {cif_gz}")
    seq = "".join(seq_pair[1].replace(";", "").split())
    # Helix regions for downstream classification logs
    helices = []
    hx_table = block.find("_struct_conf.", ["beg_label_seq_id", "end_label_seq_id"])
    if hx_table:
        for row in hx_table:
            try:
                helices.append((int(row[0]), int(row[1])))
            except ValueError:
                continue
    return seq, helices


def run_msa(target_dir: Path, sequence: str, pdb_id: str) -> Path:
    """Run ``deepfold3 msa --mode server`` for the sequence; return msa dir."""
    msa_dir = target_dir / "msa_out"
    a3m = msa_dir / "unpaired_0.a3m"
    if a3m.exists():
        print(f"[{pdb_id}] MSA already at {a3m} — skipping search")
        return msa_dir
    msa_dir.mkdir(exist_ok=True, parents=True)
    print(f"[{pdb_id}] running ColabFold-server MSA (≈3-5 min)...")
    proc = subprocess.run(
        ["deepfold3", "msa", "--sequence", sequence, "--output", str(msa_dir), "--mode", "server"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise RuntimeError(f"deepfold3 msa failed for {pdb_id}")
    if not a3m.exists():
        raise RuntimeError(f"MSA produced no {a3m}")
    return msa_dir


def write_query_json(target_dir: Path, sequence: str, pdb_id: str) -> Path:
    qpath = target_dir / "query.json"
    if qpath.exists():
        return qpath
    body = {
        "name": pdb_id.lower(),
        "dialect": "deepfold",
        "version": 3,
        "modelSeeds": [42],
        "sequences": [
            {
                "protein": {
                    "id": "A",
                    "sequence": sequence,
                    "unpairedMsaPath": "./msa_out/unpaired_0.a3m",
                    "templatesPath": "./msa_out/templates/templates_0.json",
                }
            }
        ],
    }
    qpath.write_text(json.dumps(body, indent=2) + "\n")
    return qpath


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdb-id", required=True, help="4-character PDB ID (case insensitive)")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("tmp/bench"),
        help="Benchmark root (one subdir per target). Default: tmp/bench",
    )
    ap.add_argument("--chain", default="A")
    args = ap.parse_args()

    pdb_id = args.pdb_id.upper()
    target_dir = args.out_dir / pdb_id
    target_dir.mkdir(exist_ok=True, parents=True)

    ref = target_dir / "_ref.cif.gz"
    download_ref(pdb_id, ref)
    seq, helices = extract_sequence(ref, args.chain)
    print(f"[{pdb_id}] seq length={len(seq)}, {len(helices)} HELX_P regions")

    run_msa(target_dir, seq, pdb_id)
    qpath = write_query_json(target_dir, seq, pdb_id)
    print(f"[{pdb_id}] ready: {qpath}")


if __name__ == "__main__":
    main()
