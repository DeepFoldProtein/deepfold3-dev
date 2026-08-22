"""Generate FASTA and complexes JSON for MSA pipeline submission.

Reads PDB and PATCHR manifests, checks existing A3M/paired MSA coverage,
and outputs files for use with msa_pipeline.py.

Usage:
    uv run python scripts/generate_msa_inputs.py

Outputs (on shared /data volume):
    /data/msa_search/combined.fasta          — missing + pair-needed protein sequences
    /data/msa_search/complexes.json          — heteromers needing pair search
    /data/msa_search/stats.json              — summary statistics
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from deepfold3.data.const import MOL_PROTEIN


def load_manifest(path: Path) -> list[dict]:
    print(f"Loading {path} ...", end=" ", flush=True)
    with open(path) as f:
        records = json.load(f)
    print(f"{len(records)} records")
    return records


def collect_protein_msa_ids(records: list[dict]) -> set[str]:
    """Collect all valid protein msa_ids from manifest records."""
    ids = set()
    for r in records:
        for c in r["chains"]:
            if c.get("valid", True) and c.get("mol_type", 0) == MOL_PROTEIN and c.get("msa_id"):
                ids.add(c["msa_id"])
    return ids


def collect_heteromers(records: list[dict]) -> dict[str, list[str]]:
    """Collect heteromeric complexes: record_id → [msa_ids] for protein chains.

    Only includes records with 2+ unique protein sequences.
    """
    out: dict[str, list[str]] = {}
    for r in records:
        msa_ids = []
        for c in r["chains"]:
            if c.get("valid", True) and c.get("mol_type", 0) == MOL_PROTEIN and c.get("msa_id"):
                msa_ids.append(c["msa_id"])
        if len(msa_ids) > 1 and len(set(msa_ids)) >= 2:
            out[r["id"]] = msa_ids
    return out


def load_existing_a3m(a3m_dir: str) -> set[str]:
    """Get set of msa_ids that already have A3M files."""
    if not os.path.isdir(a3m_dir):
        return set()
    print(f"Scanning {a3m_dir} ...", end=" ", flush=True)
    ids = {f[:-4] for f in os.listdir(a3m_dir) if f.endswith(".a3m")}
    print(f"{len(ids)} files")
    return ids


def load_existing_paired(paired_dir: str) -> set[str]:
    """Get set of record_ids that already have paired MSA dirs."""
    if not os.path.isdir(paired_dir):
        return set()
    print(f"Scanning {paired_dir} ...", end=" ", flush=True)
    ids = set(os.listdir(paired_dir))
    print(f"{len(ids)} dirs")
    return ids


def load_cached_fasta(path: Path) -> dict[str, str]:
    """Load msa_id → sequence from a cached FASTA file."""
    if not path.exists():
        return {}
    print(f"Loading {path} ...", end=" ", flush=True)
    seqs: dict[str, str] = {}
    name = None
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                name = line[1:]
            elif name is not None:
                seqs[name] = line
                name = None
    print(f"{len(seqs)} sequences")
    return seqs


def main():
    base = Path(__file__).resolve().parent.parent
    pdb_dir = base / "data" / "pdb_training"
    patchr_dir = base / "data" / "patchr_training"
    merged_dir = Path("/data/msa/patchr/merged")
    out_dir = Path("/data/msa_search")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load manifests
    pdb_records = load_manifest(pdb_dir / "manifest.json")
    patchr_records = load_manifest(patchr_dir / "manifest.json")

    # 2. Collect all unique protein msa_ids
    all_msa_ids = collect_protein_msa_ids(pdb_records) | collect_protein_msa_ids(patchr_records)
    print(f"\nTotal unique protein msa_ids: {len(all_msa_ids)}")

    # 3. Collect heteromeric complexes — SEPARATE per dataset.
    #    PDB and PATCHR process the same PDB IDs with different assemblies/chains,
    #    yielding different msa_ids. Merging them would confuse pairaln.
    pdb_het = collect_heteromers(pdb_records)
    patchr_het = collect_heteromers(patchr_records)
    print(f"Heteromers: PDB {len(pdb_het)}, PATCHR {len(patchr_het)}")

    # 4. Check existing coverage
    existing_a3m = load_existing_a3m(str(merged_dir / "a3m"))
    existing_paired = load_existing_paired(str(merged_dir / "paired"))

    # 5. Determine what's missing
    missing_monomer = all_msa_ids - existing_a3m
    missing_pdb_complexes = {rid: chains for rid, chains in pdb_het.items() if rid not in existing_paired}
    missing_patchr_complexes = {rid: chains for rid, chains in patchr_het.items() if rid not in existing_paired}

    # Pair-needed msa_ids (must be in FASTA even if already have A3M)
    pair_needed = set()
    for complexes in [missing_pdb_complexes, missing_patchr_complexes]:
        for chains in complexes.values():
            pair_needed.update(chains)

    fasta_ids = missing_monomer | pair_needed
    re_search = fasta_ids & existing_a3m

    print("\n--- Summary ---")
    print(f"Missing monomer MSA:   {len(missing_monomer)}")
    print(f"Missing paired MSA:    PDB {len(missing_pdb_complexes)}, PATCHR {len(missing_patchr_complexes)} complexes")
    print(f"FASTA entries needed:  {len(fasta_ids)}")
    print(f"  re-search overhead:  {len(re_search)} (already have A3M, needed for pair)")

    # 6. Load cached FASTAs to get sequences
    pdb_seqs = load_cached_fasta(pdb_dir / "msa_sequences.fasta")
    patchr_seqs = load_cached_fasta(patchr_dir / "msa_sequences.fasta")

    # Merge (pdb takes precedence if both have same msa_id)
    all_seqs = {**patchr_seqs, **pdb_seqs}
    print(f"\nCombined cached sequences: {len(all_seqs)}")

    # Check coverage
    fasta_ids & set(all_seqs)
    not_found = fasta_ids - set(all_seqs)
    if not_found:
        print(f"WARNING: {len(not_found)} msa_ids not found in cached FASTAs!")
        print(f"  These will need NPZ extraction. First 5: {list(not_found)[:5]}")
        # Fall back to NPZ extraction for missing sequences
        _extract_missing(not_found, pdb_dir, patchr_dir, pdb_records, patchr_records, all_seqs)
        fasta_ids & set(all_seqs)
        not_found = fasta_ids - set(all_seqs)
        if not_found:
            print(f"WARNING: Still missing {len(not_found)} sequences after NPZ extraction, skipping them")
            fasta_ids -= not_found

    # 7. Write FASTA
    fasta_path = out_dir / "combined.fasta"
    print(f"\nWriting {fasta_path} ({len(fasta_ids)} sequences) ...", end=" ", flush=True)
    with open(fasta_path, "w") as f:
        for mid in sorted(fasta_ids):
            seq = all_seqs[mid]
            f.write(f">{mid}\n{seq}\n")
    print("done")

    # 8. Write complexes JSONs — separate per dataset to avoid record ID collisions
    def write_complexes(complexes: dict[str, list[str]], path: Path, label: str) -> dict[str, list[str]]:
        filtered = {}
        for rid, chains in complexes.items():
            valid_chains = [c for c in chains if c in fasta_ids]
            if len(valid_chains) > 1 and len(set(valid_chains)) >= 2:
                filtered[rid] = valid_chains
        print(f"Writing {path} ({len(filtered)} {label} complexes) ...", end=" ", flush=True)
        with open(path, "w") as f:
            json.dump(filtered, f)
        print("done")
        if len(filtered) < len(complexes):
            print(f"  ({len(complexes) - len(filtered)} dropped due to missing sequences)")
        return filtered

    filtered_pdb = write_complexes(missing_pdb_complexes, out_dir / "pdb_complexes.json", "PDB")
    filtered_patchr = write_complexes(missing_patchr_complexes, out_dir / "patchr_complexes.json", "PATCHR")

    # 9. Write stats
    stats = {
        "total_unique_msa_ids": len(all_msa_ids),
        "existing_a3m": len(existing_a3m),
        "missing_monomer": len(missing_monomer),
        "pdb_heteromers": len(pdb_het),
        "patchr_heteromers": len(patchr_het),
        "existing_paired": len(existing_paired),
        "missing_pdb_complexes": len(filtered_pdb),
        "missing_patchr_complexes": len(filtered_patchr),
        "fasta_entries": len(fasta_ids),
        "re_search_overhead": len(re_search),
    }
    stats_path = out_dir / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nStats written to {stats_path}")

    # 10. Print submission commands
    n_chunks = max(1, len(fasta_ids) // 1000)
    n_pdb_pair = max(1, len(filtered_pdb) // 500)
    n_patchr_pair = max(1, len(filtered_patchr) // 500)

    print("\n--- Submission commands ---")
    print(f"# 1. Monomer search only ({len(fasta_ids)} seqs)")
    print("python examples/deepfold3/msa_pipeline.py \\")
    print("    --query /data/msa_search/combined.fasta \\")
    print("    --run-name msa_search \\")
    print(f"    --chunks {n_chunks} --gpus-per-task 1 --parallelism 120")
    print()
    print(f"# 2. PDB pair search ({len(filtered_pdb)} complexes)")
    print("python examples/deepfold3/msa_pipeline.py \\")
    print("    --query /data/msa_search/combined.fasta \\")
    print("    --run-name msa_search_pdb_pair \\")
    print(f"    --chunks {n_chunks} \\")
    print("    --pair-only --complexes /data/msa_search/pdb_complexes.json \\")
    print(f"    --pair-batches {n_pdb_pair} --pair-parallelism 12")
    print()
    print(f"# 3. PATCHR pair search ({len(filtered_patchr)} complexes)")
    print("python examples/deepfold3/msa_pipeline.py \\")
    print("    --query /data/msa_search/combined.fasta \\")
    print("    --run-name msa_search_patchr_pair \\")
    print(f"    --chunks {n_chunks} \\")
    print("    --pair-only --complexes /data/msa_search/patchr_complexes.json \\")
    print(f"    --pair-batches {n_patchr_pair} --pair-parallelism 12")
    print()
    print("# After completion, import results:")
    print(
        "uv run deepfold3 import-msa --data-dir data/pdb_training --msa-dir /data/msa/msa_search/merged/a3m --workers 32"
    )
    print(
        "uv run deepfold3 import-msa --data-dir data/patchr_training --msa-dir /data/msa/msa_search/merged/a3m --workers 32"
    )
    print(
        "uv run deepfold3 import-paired-msa --data-dir data/pdb_training --paired-dir /data/msa/msa_search_pdb_pair/merged/paired --workers 32"
    )
    print(
        "uv run deepfold3 import-paired-msa --data-dir data/patchr_training --paired-dir /data/msa/msa_search_patchr_pair/merged/paired --workers 32"
    )


def _extract_missing(
    missing_ids: set[str],
    pdb_dir: Path,
    patchr_dir: Path,
    pdb_records: list[dict],
    patchr_records: list[dict],
    out_seqs: dict[str, str],
) -> None:
    """Extract sequences from structure NPZ files for msa_ids not in cached FASTA."""

    from deepfold3.data.paths import structure_path
    from deepfold3.data.prepare.sequences import get_chain_residue_types, residue_types_to_sequence
    from deepfold3.data.types import Structure

    # Build msa_id → (data_dir, record_id, chain_id) mapping
    targets: dict[str, list[tuple[Path, str, int]]] = {}
    for records, data_dir in [(pdb_records, pdb_dir), (patchr_records, patchr_dir)]:
        for r in records:
            for c in r["chains"]:
                mid = c.get("msa_id")
                if mid and mid in missing_ids and mid not in out_seqs:
                    if mid not in targets:
                        targets[mid] = []
                    targets[mid].append((data_dir / "structures", r["id"], c["chain_id"]))

    print(f"  Extracting {len(targets)} sequences from NPZ files ...")
    found = 0
    for mid, locs in targets.items():
        for structures_dir, record_id, chain_id in locs:
            npz_path = structure_path(structures_dir, record_id)
            if not npz_path.exists():
                continue
            try:
                structure = Structure.load(npz_path)
                res_types = get_chain_residue_types(structure, chain_id)
                if len(res_types) > 0:
                    out_seqs[mid] = residue_types_to_sequence(res_types)
                    found += 1
                    break
            except Exception:
                continue
    print(f"  Extracted {found}/{len(targets)} sequences from NPZ")


if __name__ == "__main__":
    main()
