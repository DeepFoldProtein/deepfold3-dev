"""JSON input parser for structure prediction jobs.

Supports two JSON dialects:

1. **Protenix format** — ``proteinChain``, ``dnaSequence``/``dnaChain``,
   ``rnaSequence``/``rnaChain`` keys.
2. **AlphaFold 3 format** — ``protein``, ``dna``, ``rna`` keys with
   ``id`` fields for chain naming and optional inline MSA strings.

Format is auto-detected per entry.  Both dialects produce the same
``JobInput`` output.

Protenix example::

    [
        {
            "sequences": [
                {"proteinChain": {"sequence": "MKTII...", "count": 1, ...}},
                {"dnaSequence": {"sequence": "ATGC", "count": 1}},
                {"rnaSequence": {"sequence": "AUGC", "count": 1}}
            ],
            "modelSeeds": [1, 2, 3],
            "name": "job_name"
        }
    ]

AlphaFold 3 example::

    [
        {
            "sequences": [
                {"protein": {"id": "A", "sequence": "MKTII...", "unpairedMsaPath": "..."}},
                {"dna": {"id": ["B", "C"], "sequence": "ATGC"}},
                {"rna": {"id": "D", "sequence": "AUGC"}}
            ],
            "modelSeeds": [1],
            "name": "job_name",
            "dialect": "alphafold3",
            "version": 2
        }
    ]
"""

from __future__ import annotations

import dataclasses
import json
import tempfile
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
class NucleotideChainInput:
    """A single DNA or RNA chain specification from JSON input.

    Attributes:
        sequence: 1-letter nucleotide sequence (ACGT for DNA, ACGU for RNA).
        count: Number of copies of this chain (must be 1 for now).
        paired_msa_path: Path to paired MSA A3M file (optional).
        unpaired_msa_path: Path to unpaired MSA A3M file (optional).
    """

    sequence: str
    count: int
    paired_msa_path: Path | None = None
    unpaired_msa_path: Path | None = None


@dataclasses.dataclass
class JobInput:
    """A single prediction job from JSON input.

    Attributes:
        name: Job name / identifier.
        protein_chains: List of protein chain inputs.
        dna_chains: List of DNA chain inputs.
        rna_chains: List of RNA chain inputs.
        model_seeds: List of random seeds for diffusion sampling.
    """

    name: str
    protein_chains: list[ProteinChainInput]
    dna_chains: list[NucleotideChainInput]
    rna_chains: list[NucleotideChainInput]
    model_seeds: list[int]


_DNA_KEYS = ("dnaSequence", "dnaChain")
_RNA_KEYS = ("rnaSequence", "rnaChain")


def parse_json_input(json_path: str | Path) -> list[JobInput]:
    """Parse a JSON input file (auto-detects Protenix or AF3 format).

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
        if _is_af3_format(entry):
            jobs.append(_parse_af3_entry(entry, base_dir))
        else:
            jobs.append(_parse_protenix_entry(entry, base_dir))

    return jobs


def _is_af3_format(entry: dict) -> bool:
    """Detect whether a job entry uses AlphaFold 3 JSON format.

    AF3 format is identified by:
    - Explicit ``"dialect": "alphafold3"`` field, OR
    - Sequence items using ``protein``/``dna``/``rna`` keys (without
      ``Chain``/``Sequence`` suffix)
    """
    if entry.get("dialect") == "alphafold3":
        return True
    for seq_item in entry.get("sequences", []):
        if any(k in seq_item for k in ("protein", "dna", "rna")):
            return True
    return False


def _parse_protenix_entry(entry: dict, base_dir: Path) -> JobInput:
    """Parse a single Protenix-format job entry."""
    name = entry.get("name", "unnamed")
    model_seeds = entry.get("modelSeeds", [42])

    protein_chains: list[ProteinChainInput] = []
    dna_chains: list[NucleotideChainInput] = []
    rna_chains: list[NucleotideChainInput] = []

    for seq_item in entry.get("sequences", []):
        if "proteinChain" in seq_item:
            pc = seq_item["proteinChain"]
            count = pc.get("count", 1)
            if count != 1:
                raise ValueError(f"Only count=1 is supported, got count={count} in job {name!r}")
            protein_chains.append(
                ProteinChainInput(
                    sequence=pc["sequence"],
                    count=count,
                    paired_msa_path=_resolve_path(base_dir, pc.get("pairedMsaPath")),
                    unpaired_msa_path=_resolve_path(base_dir, pc.get("unpairedMsaPath")),
                    templates_path=_resolve_path(base_dir, pc.get("templatesPath")),
                )
            )
        elif any(k in seq_item for k in _DNA_KEYS):
            key = next(k for k in _DNA_KEYS if k in seq_item)
            dc = seq_item[key]
            count = dc.get("count", 1)
            if count != 1:
                raise ValueError(f"Only count=1 is supported, got count={count} in job {name!r}")
            dna_chains.append(
                NucleotideChainInput(
                    sequence=dc["sequence"],
                    count=count,
                    paired_msa_path=_resolve_path(base_dir, dc.get("pairedMsaPath")),
                    unpaired_msa_path=_resolve_path(base_dir, dc.get("unpairedMsaPath")),
                )
            )
        elif any(k in seq_item for k in _RNA_KEYS):
            key = next(k for k in _RNA_KEYS if k in seq_item)
            rc = seq_item[key]
            count = rc.get("count", 1)
            if count != 1:
                raise ValueError(f"Only count=1 is supported, got count={count} in job {name!r}")
            rna_chains.append(
                NucleotideChainInput(
                    sequence=rc["sequence"],
                    count=count,
                    paired_msa_path=_resolve_path(base_dir, rc.get("pairedMsaPath")),
                    unpaired_msa_path=_resolve_path(base_dir, rc.get("unpairedMsaPath")),
                )
            )
        elif "ligand" in seq_item:
            raise ValueError(f"Unsupported entity type 'ligand' in job {name!r}")

    if not protein_chains and not dna_chains and not rna_chains:
        raise ValueError(f"No chains found in job {name!r}")

    return JobInput(
        name=name,
        protein_chains=protein_chains,
        dna_chains=dna_chains,
        rna_chains=rna_chains,
        model_seeds=model_seeds,
    )


def _parse_af3_entry(entry: dict, base_dir: Path) -> JobInput:
    """Parse a single AlphaFold 3-format job entry.

    AF3 format uses ``protein``, ``dna``, ``rna`` keys in sequence items.
    Each has an ``id`` field (string or list of strings for multi-copy)
    and a ``sequence`` field.  MSA can be provided inline (``unpairedMsa``,
    ``pairedMsa``) or as file paths (``unpairedMsaPath``, ``pairedMsaPath``).
    Templates are specified via ``templates`` list or ``templatesPath``.
    """
    name = entry.get("name", "unnamed")
    model_seeds = entry.get("modelSeeds", [42])

    protein_chains: list[ProteinChainInput] = []
    dna_chains: list[NucleotideChainInput] = []
    rna_chains: list[NucleotideChainInput] = []

    for seq_item in entry.get("sequences", []):
        if "protein" in seq_item:
            p = seq_item["protein"]
            ids = _ensure_id_list(p.get("id", []))
            seq = p["sequence"]
            unpaired = _resolve_af3_msa(base_dir, p.get("unpairedMsaPath"), p.get("unpairedMsa"))
            paired = _resolve_af3_msa(base_dir, p.get("pairedMsaPath"), p.get("pairedMsa"))
            tmpl = _resolve_path(base_dir, p.get("templatesPath"))
            for _ in ids or [None]:
                protein_chains.append(
                    ProteinChainInput(
                        sequence=seq,
                        count=1,
                        paired_msa_path=paired,
                        unpaired_msa_path=unpaired,
                        templates_path=tmpl,
                    )
                )
        elif "dna" in seq_item:
            d = seq_item["dna"]
            ids = _ensure_id_list(d.get("id", []))
            seq = d["sequence"]
            unpaired = _resolve_af3_msa(base_dir, d.get("unpairedMsaPath"), d.get("unpairedMsa"))
            paired = _resolve_af3_msa(base_dir, d.get("pairedMsaPath"), d.get("pairedMsa"))
            for _ in ids or [None]:
                dna_chains.append(
                    NucleotideChainInput(
                        sequence=seq,
                        count=1,
                        paired_msa_path=paired,
                        unpaired_msa_path=unpaired,
                    )
                )
        elif "rna" in seq_item:
            r = seq_item["rna"]
            ids = _ensure_id_list(r.get("id", []))
            seq = r["sequence"]
            unpaired = _resolve_af3_msa(base_dir, r.get("unpairedMsaPath"), r.get("unpairedMsa"))
            paired = _resolve_af3_msa(base_dir, r.get("pairedMsaPath"), r.get("pairedMsa"))
            for _ in ids or [None]:
                rna_chains.append(
                    NucleotideChainInput(
                        sequence=seq,
                        count=1,
                        paired_msa_path=paired,
                        unpaired_msa_path=unpaired,
                    )
                )
        elif "ligand" in seq_item:
            raise ValueError(f"Unsupported entity type 'ligand' in job {name!r}")

    if not protein_chains and not dna_chains and not rna_chains:
        raise ValueError(f"No chains found in job {name!r}")

    return JobInput(
        name=name,
        protein_chains=protein_chains,
        dna_chains=dna_chains,
        rna_chains=rna_chains,
        model_seeds=model_seeds,
    )


def _ensure_id_list(ids: str | list[str]) -> list[str]:
    """Normalize ``id`` field to a list of chain IDs."""
    if isinstance(ids, str):
        return [ids]
    return list(ids)


def _resolve_af3_msa(base_dir: Path, path_str: str | None, inline_str: str | None) -> Path | None:
    """Resolve MSA from either a file path or an inline string.

    If an inline MSA string is provided (and no file path), it is written to
    a temporary file and the path to that file is returned.
    """
    if path_str is not None:
        return _resolve_path(base_dir, path_str)
    if inline_str is not None and inline_str.strip():
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".a3m", delete=False)
        tmp.write(inline_str)
        tmp.close()
        return Path(tmp.name)
    return None


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
