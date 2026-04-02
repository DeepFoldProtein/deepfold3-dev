#!/usr/bin/env python3
"""Sample PDB Atlas YAML entries and extract FASTA for pipeline testing.

Usage:
    # Extract FASTA from all YAMLs (canonical sequences)
    python scripts/atlas_sample.py --yaml-dir data/pdb_atlas_multimer_20260317/yamls \
        --output data/atlas_sample --fasta-only

    # Sample N entries and prepare CIFs + FASTA for pipeline testing
    python scripts/atlas_sample.py --yaml-dir data/pdb_atlas_multimer_20260317/yamls \
        --cif-dir data/pdb_atlas_multimer_20260317/cifs \
        --output data/atlas_sample --limit 50 --seed 42

    # Sample with diversity: at least 1 ligand-containing, 1 multi-chain, etc.
    python scripts/atlas_sample.py --yaml-dir data/pdb_atlas_multimer_20260317/yamls \
        --cif-dir data/pdb_atlas_multimer_20260317/cifs \
        --output data/atlas_sample --limit 50 --diverse
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

import yaml


def parse_yaml(path: Path) -> dict | None:
    """Parse an Atlas YAML and extract sequences + metadata."""
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
    except Exception:
        return None

    if not data or "sequences" not in data:
        return None

    pdb_id = path.stem
    chains = []
    has_ligand = False
    has_nucleic = False
    total_residues = 0

    for entry in data["sequences"]:
        if "protein" in entry:
            p = entry["protein"]
            seq = p.get("sequence", "")
            chains.append({"type": "protein", "id": p.get("id", "?"), "sequence": seq})
            total_residues += len(seq)
        elif "rna" in entry:
            r = entry["rna"]
            seq = r.get("sequence", "")
            chains.append({"type": "rna", "id": r.get("id", "?"), "sequence": seq})
            has_nucleic = True
            total_residues += len(seq)
        elif "dna" in entry:
            d = entry["dna"]
            seq = d.get("sequence", "")
            chains.append({"type": "dna", "id": d.get("id", "?"), "sequence": seq})
            has_nucleic = True
            total_residues += len(seq)
        elif "ligand" in entry:
            lig = entry["ligand"]
            chains.append({
                "type": "ligand",
                "id": lig.get("id", "?"),
                "ccd": lig.get("ccd", "UNK"),
            })
            has_ligand = True

    return {
        "pdb_id": pdb_id,
        "chains": chains,
        "num_chains": len(chains),
        "num_protein": sum(1 for c in chains if c["type"] == "protein"),
        "has_ligand": has_ligand,
        "has_nucleic": has_nucleic,
        "total_residues": total_residues,
    }


def write_fasta(entries: list[dict], output: Path) -> None:
    """Write FASTA file with all polymer sequences from parsed entries."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        for entry in entries:
            pdb_id = entry["pdb_id"]
            for chain in entry["chains"]:
                if chain["type"] in ("protein", "rna", "dna"):
                    header = f">{pdb_id}_{chain['id']} type={chain['type']}"
                    f.write(f"{header}\n{chain['sequence']}\n")


def sample_diverse(entries: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Sample N entries with diversity: ensure ligand, nucleic, multi-chain coverage."""
    selected = []
    remaining = list(entries)

    # Ensure at least one of each category if available
    categories = [
        ("ligand-containing", lambda e: e["has_ligand"]),
        ("nucleic-containing", lambda e: e["has_nucleic"]),
        ("multi-chain (3+)", lambda e: e["num_protein"] >= 3),
        ("homo-dimer", lambda e: e["num_protein"] == 2 and len(set(
            c["sequence"] for c in e["chains"] if c["type"] == "protein"
        )) == 1),
        ("large (>500 res)", lambda e: e["total_residues"] > 500),
        ("small (<100 res)", lambda e: 0 < e["total_residues"] < 100),
    ]

    for name, pred in categories:
        if len(selected) >= n:
            break
        candidates = [e for e in remaining if pred(e)]
        if candidates:
            pick = rng.choice(candidates)
            selected.append(pick)
            remaining.remove(pick)

    # Fill rest randomly
    if len(selected) < n and remaining:
        extra = rng.sample(remaining, min(n - len(selected), len(remaining)))
        selected.extend(extra)

    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample Atlas YAMLs for pipeline testing")
    parser.add_argument("--yaml-dir", required=True, help="Directory of Atlas YAML files")
    parser.add_argument("--cif-dir", default=None, help="Directory of Atlas CIF files (for copying)")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--limit", type=int, default=None, help="Max entries to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--fasta-only", action="store_true", help="Only generate FASTA, skip CIF copy")
    parser.add_argument("--diverse", action="store_true", help="Ensure diverse sampling")
    args = parser.parse_args()

    yaml_dir = Path(args.yaml_dir)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    # Parse all YAMLs
    yaml_files = sorted(yaml_dir.glob("*.yaml"))
    print(f"Found {len(yaml_files)} YAML files")

    entries = []
    for yf in yaml_files:
        e = parse_yaml(yf)
        if e is not None:
            entries.append(e)

    print(f"Parsed {len(entries)} entries")

    # Stats
    n_ligand = sum(1 for e in entries if e["has_ligand"])
    n_nucleic = sum(1 for e in entries if e["has_nucleic"])
    n_multi = sum(1 for e in entries if e["num_protein"] >= 2)
    print(f"  with ligands: {n_ligand}")
    print(f"  with nucleic: {n_nucleic}")
    print(f"  multi-chain protein: {n_multi}")

    # Sample
    rng = random.Random(args.seed)
    if args.limit and args.limit < len(entries):
        if args.diverse:
            sampled = sample_diverse(entries, args.limit, rng)
        else:
            sampled = rng.sample(entries, args.limit)
        print(f"Sampled {len(sampled)} entries")
    else:
        sampled = entries

    # Write FASTA
    fasta_path = output / "sequences.fasta"
    write_fasta(sampled, fasta_path)
    print(f"FASTA written to {fasta_path}")

    # Write sampled IDs
    ids_path = output / "sampled_ids.txt"
    with open(ids_path, "w") as f:
        for e in sampled:
            f.write(f"{e['pdb_id']}\n")
    print(f"IDs written to {ids_path}")

    # Copy CIFs if requested
    if not args.fasta_only and args.cif_dir:
        cif_dir = Path(args.cif_dir)
        cif_out = output / "cifs"
        cif_out.mkdir(exist_ok=True)
        copied = 0
        for e in sampled:
            # Atlas CIF naming: {pdb_id}.cif (no _model_0 in current data)
            src = cif_dir / f"{e['pdb_id']}.cif"
            if not src.exists():
                src = cif_dir / f"{e['pdb_id']}_model_0.cif"
            if src.exists():
                shutil.copy2(src, cif_out / src.name)
                copied += 1
        print(f"Copied {copied}/{len(sampled)} CIF files to {cif_out}")

    # Write summary
    summary = output / "summary.txt"
    with open(summary, "w") as f:
        f.write(f"Total entries: {len(sampled)}\n")
        for e in sampled:
            chains_desc = ", ".join(
                f"{c['type']}:{c['id']}" + (f"({c.get('ccd', '')})" if c["type"] == "ligand" else "")
                for c in e["chains"]
            )
            f.write(f"{e['pdb_id']}: {e['num_chains']} chains ({chains_desc}), {e['total_residues']} res\n")
    print(f"Summary written to {summary}")


if __name__ == "__main__":
    main()
