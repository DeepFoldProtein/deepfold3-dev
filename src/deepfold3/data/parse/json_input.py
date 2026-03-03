"""Protenix-format JSON input parser.

Parses JSON files specifying sequences, MSA paths, and template paths
for structure prediction. Currently supports single-chain protein inputs.

Example JSON format::

    [
        {
            "sequences": [
                {
                    "proteinChain": {
                        "sequence": "MKTII...",
                        "count": 1,
                        "pairedMsaPath": "./path/to/pairing.a3m",
                        "unpairedMsaPath": "./path/to/non_pairing.a3m",
                        "templatesPath": "./path/to/concat.hhr"
                    }
                }
            ],
            "modelSeeds": [1, 2, 3],
            "name": "job_name"
        }
    ]
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path


@dataclasses.dataclass
class ProteinChainInput:
    """A single protein chain specification from JSON input.

    Attributes:
        sequence: 1-letter amino acid sequence.
        count: Number of copies of this chain (must be 1 for now).
        paired_msa_path: Path to paired MSA A3M file (optional).
        unpaired_msa_path: Path to unpaired MSA A3M file (optional).
        templates_path: Path to HHR template file (optional).
    """

    sequence: str
    count: int
    paired_msa_path: Path | None = None
    unpaired_msa_path: Path | None = None
    templates_path: Path | None = None


@dataclasses.dataclass
class JobInput:
    """A single prediction job from JSON input.

    Attributes:
        name: Job name / identifier.
        protein_chains: List of protein chain inputs.
        model_seeds: List of random seeds for diffusion sampling.
    """

    name: str
    protein_chains: list[ProteinChainInput]
    model_seeds: list[int]


def parse_json_input(json_path: str | Path) -> list[JobInput]:
    """Parse a Protenix-format JSON input file.

    Args:
        json_path: Path to the JSON file.

    Returns:
        List of JobInput objects, one per job entry.

    Raises:
        ValueError: If the JSON contains unsupported entity types or invalid
            chain counts.
    """
    json_path = Path(json_path)
    base_dir = json_path.parent

    with open(json_path) as f:
        data = json.load(f)

    if not isinstance(data, list):
        data = [data]

    jobs: list[JobInput] = []
    for entry in data:
        name = entry.get("name", "unnamed")
        model_seeds = entry.get("modelSeeds", [42])

        protein_chains: list[ProteinChainInput] = []
        for seq_item in entry.get("sequences", []):
            if "proteinChain" in seq_item:
                pc = seq_item["proteinChain"]
                count = pc.get("count", 1)
                if count != 1:
                    raise ValueError(f"Only count=1 is supported, got count={count} in job {name!r}")

                chain = ProteinChainInput(
                    sequence=pc["sequence"],
                    count=count,
                    paired_msa_path=_resolve_path(base_dir, pc.get("pairedMsaPath")),
                    unpaired_msa_path=_resolve_path(base_dir, pc.get("unpairedMsaPath")),
                    templates_path=_resolve_path(base_dir, pc.get("templatesPath")),
                )
                protein_chains.append(chain)
            elif "rnaChain" in seq_item or "dnaChain" in seq_item or "ligand" in seq_item:
                entity = next(k for k in seq_item if k != "count")
                raise ValueError(f"Unsupported entity type {entity!r} in job {name!r} — only proteinChain is supported")

        if not protein_chains:
            raise ValueError(f"No proteinChain found in job {name!r}")

        jobs.append(JobInput(name=name, protein_chains=protein_chains, model_seeds=model_seeds))

    return jobs


def _resolve_path(base_dir: Path, raw: str | None) -> Path | None:
    """Resolve a possibly-relative path against the JSON file's directory.

    Tries resolution in order:
    1. Relative to ``base_dir`` (the JSON file's parent directory)
    2. Relative to ancestor directories of ``base_dir`` (walks up)
    3. Relative to CWD
    4. Falls back to the JSON-relative path (for a clear error message)
    """
    if raw is None:
        return None
    p = Path(raw)
    if p.is_absolute():
        return p
    # Try JSON directory and its ancestors
    candidate = base_dir
    for _ in range(6):
        resolved = candidate / p
        if resolved.exists():
            return resolved
        parent = candidate.parent
        if parent == candidate:
            break
        candidate = parent
    # Try CWD
    if p.exists():
        return p.resolve()
    # Return the JSON-relative path (caller will get a clear error)
    return base_dir / p
