"""JSON input parser for structure prediction jobs.

Supports two JSON dialects:

1. **Protenix format** — ``proteinChain``, ``dnaSequence``/``dnaChain``,
   ``rnaSequence``/``rnaChain`` keys.
2. **AlphaFold 3 format** — ``protein``, ``dna``, ``rna`` keys with
   ``id`` fields for chain naming and optional inline MSA strings.

Format is auto-detected per entry.  Both dialects produce the same
``JobInput`` output.

All file I/O (reading A3M files, downloading template CIFs) happens at
parse time.  The resulting dataclasses hold parsed objects (MSA, template
hits) rather than file paths, so downstream consumers need no I/O.

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
import logging
from dataclasses import field
from pathlib import Path

from deepfold3.data import const
from deepfold3.data.parse.a3m import parse_a3m
from deepfold3.data.parse.templates import TemplateHitInput, fetch_pdb_mmcif
from deepfold3.data.types import MSA

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ProteinChainInput:
    """A single protein chain specification from JSON input.

    Attributes:
        sequence: 1-letter amino acid sequence.
        count: Number of copies of this chain (must be 1 for now).
        paired_msa: Parsed paired MSA object (optional).
        unpaired_msa: Parsed unpaired MSA object (optional).
        template_hits: Parsed template hits ready for feature extraction.
    """

    sequence: str
    count: int
    paired_msa: MSA | None = None
    unpaired_msa: MSA | None = None
    template_hits: list[TemplateHitInput] = field(default_factory=list)


@dataclasses.dataclass
class NucleotideChainInput:
    """A single DNA or RNA chain specification from JSON input.

    Attributes:
        sequence: 1-letter nucleotide sequence (ACGT for DNA, ACGU for RNA).
        count: Number of copies of this chain (must be 1 for now).
        mol_type: const.MOL_DNA or const.MOL_RNA.
        paired_msa: Parsed paired MSA object (optional).
        unpaired_msa: Parsed unpaired MSA object (optional).
    """

    sequence: str
    count: int
    mol_type: int
    paired_msa: MSA | None = None
    unpaired_msa: MSA | None = None


@dataclasses.dataclass
class LigandChainInput:
    """A single ligand chain specification from JSON input.

    Exactly one of ``ccd_ids`` or ``smiles`` must be provided.

    Attributes:
        ccd_ids: List of CCD component codes (e.g. ["ATP"]).
        smiles: SMILES string for a custom molecule.
        count: Number of copies of this ligand.
    """

    ccd_ids: list[str] | None = None
    smiles: str | None = None
    count: int = 1


@dataclasses.dataclass
class JobInput:
    """A single prediction job from JSON input.

    Attributes:
        name: Job name / identifier.
        protein_chains: List of protein chain inputs.
        dna_chains: List of DNA chain inputs.
        rna_chains: List of RNA chain inputs.
        ligand_chains: List of ligand chain inputs.
        model_seeds: List of random seeds for diffusion sampling.
    """

    name: str
    protein_chains: list[ProteinChainInput]
    dna_chains: list[NucleotideChainInput]
    rna_chains: list[NucleotideChainInput]
    ligand_chains: list[LigandChainInput] = field(default_factory=list)
    model_seeds: list[int] = field(default_factory=lambda: [42])


_DNA_KEYS = ("dnaSequence", "dnaChain")
_RNA_KEYS = ("rnaSequence", "rnaChain")


def parse_json_input(
    json_path: str | Path,
    max_templates: int = 4,
    template_cache_dir: str | Path | None = None,
) -> list[JobInput]:
    """Parse a JSON input file (auto-detects Protenix or AF3 format).

    All file I/O (reading A3M files, downloading template CIFs) happens
    here.  The returned ``JobInput`` objects hold fully parsed MSA and
    template hit objects.

    Args:
        json_path: Path to the JSON file.
        max_templates: Maximum number of template hits to load per chain.
        template_cache_dir: Directory to cache downloaded template mmCIF files.

    Returns:
        List of JobInput objects, one per job entry.

    Raises:
        ValueError: If the JSON contains unsupported entity types or invalid
            chain counts.
    """
    json_path = Path(json_path)
    base_dir = json_path.parent

    cache_dir = Path(template_cache_dir) if template_cache_dir is not None else None

    with open(json_path) as f:
        data = json.load(f)

    if not isinstance(data, list):
        data = [data]

    jobs: list[JobInput] = []
    for entry in data:
        if _is_af3_format(entry):
            jobs.append(_parse_af3_entry(entry, base_dir, max_templates, cache_dir))
        else:
            jobs.append(_parse_protenix_entry(entry, base_dir, max_templates, cache_dir))

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
        if any(k in seq_item for k in ("protein", "dna", "rna", "ligand")):
            # AF3 uses bare "ligand" key too, but so does Protenix.
            # Distinguish by checking for AF3-specific subkeys.
            if "ligand" in seq_item and not any(k in seq_item for k in ("protein", "dna", "rna")):
                lig = seq_item["ligand"]
                if isinstance(lig, dict) and ("ccdCodes" in lig or "id" in lig):
                    return True
                continue  # might be Protenix
            return True
    return False


def _parse_protenix_entry(entry: dict, base_dir: Path, max_templates: int, cache_dir: Path | None) -> JobInput:
    """Parse a single Protenix-format job entry."""
    name = entry.get("name", "unnamed")
    model_seeds = entry.get("modelSeeds", [42])

    protein_chains: list[ProteinChainInput] = []
    dna_chains: list[NucleotideChainInput] = []
    rna_chains: list[NucleotideChainInput] = []
    ligand_chains: list[LigandChainInput] = []

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
                    paired_msa=_resolve_msa(base_dir, const.MOL_PROTEIN, pc.get("pairedMsaPath"), None),
                    unpaired_msa=_resolve_msa(base_dir, const.MOL_PROTEIN, pc.get("unpairedMsaPath"), None),
                    template_hits=_resolve_template_hits(base_dir, pc.get("templatesPath"), max_templates, cache_dir),
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
                    mol_type=const.MOL_DNA,
                    paired_msa=_resolve_msa(base_dir, const.MOL_DNA, dc.get("pairedMsaPath"), None),
                    unpaired_msa=_resolve_msa(base_dir, const.MOL_DNA, dc.get("unpairedMsaPath"), None),
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
                    mol_type=const.MOL_RNA,
                    paired_msa=_resolve_msa(base_dir, const.MOL_RNA, rc.get("pairedMsaPath"), None),
                    unpaired_msa=_resolve_msa(base_dir, const.MOL_RNA, rc.get("unpairedMsaPath"), None),
                )
            )
        elif "ligand" in seq_item:
            lig = seq_item["ligand"]
            ligand_str = lig.get("ligand", "")
            count = lig.get("count", 1)
            if isinstance(ligand_str, str) and ligand_str.startswith("CCD_"):
                ccd_code = ligand_str[4:]
                for _ in range(count):
                    ligand_chains.append(LigandChainInput(ccd_ids=[ccd_code], count=1))
            elif isinstance(ligand_str, str) and ligand_str:
                # Assume SMILES if not CCD_ prefixed
                for _ in range(count):
                    ligand_chains.append(LigandChainInput(smiles=ligand_str, count=1))

    if not protein_chains and not dna_chains and not rna_chains and not ligand_chains:
        raise ValueError(f"No chains found in job {name!r}")

    return JobInput(
        name=name,
        protein_chains=protein_chains,
        dna_chains=dna_chains,
        rna_chains=rna_chains,
        ligand_chains=ligand_chains,
        model_seeds=model_seeds,
    )


def _parse_af3_entry(entry: dict, base_dir: Path, max_templates: int, cache_dir: Path | None) -> JobInput:
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
    ligand_chains: list[LigandChainInput] = []

    for seq_item in entry.get("sequences", []):
        if "protein" in seq_item:
            p = seq_item["protein"]
            ids = _ensure_id_list(p.get("id", []))
            seq = p["sequence"]
            unpaired = _resolve_msa(base_dir, const.MOL_PROTEIN, p.get("unpairedMsaPath"), p.get("unpairedMsa"))
            paired = _resolve_msa(base_dir, const.MOL_PROTEIN, p.get("pairedMsaPath"), p.get("pairedMsa"))
            tmpl_hits = _resolve_template_hits(base_dir, p.get("templatesPath"), max_templates, cache_dir)
            for _ in ids or [None]:
                protein_chains.append(
                    ProteinChainInput(
                        sequence=seq,
                        count=1,
                        paired_msa=paired,
                        unpaired_msa=unpaired,
                        template_hits=tmpl_hits,
                    )
                )
        elif "dna" in seq_item:
            d = seq_item["dna"]
            ids = _ensure_id_list(d.get("id", []))
            seq = d["sequence"]
            unpaired = _resolve_msa(base_dir, const.MOL_DNA, d.get("unpairedMsaPath"), d.get("unpairedMsa"))
            paired = _resolve_msa(base_dir, const.MOL_DNA, d.get("pairedMsaPath"), d.get("pairedMsa"))
            for _ in ids or [None]:
                dna_chains.append(
                    NucleotideChainInput(
                        sequence=seq,
                        count=1,
                        mol_type=const.MOL_DNA,
                        paired_msa=paired,
                        unpaired_msa=unpaired,
                    )
                )
        elif "rna" in seq_item:
            r = seq_item["rna"]
            ids = _ensure_id_list(r.get("id", []))
            seq = r["sequence"]
            unpaired = _resolve_msa(base_dir, const.MOL_RNA, r.get("unpairedMsaPath"), r.get("unpairedMsa"))
            paired = _resolve_msa(base_dir, const.MOL_RNA, r.get("pairedMsaPath"), r.get("pairedMsa"))
            for _ in ids or [None]:
                rna_chains.append(
                    NucleotideChainInput(
                        sequence=seq,
                        count=1,
                        mol_type=const.MOL_RNA,
                        paired_msa=paired,
                        unpaired_msa=unpaired,
                    )
                )
        elif "ligand" in seq_item:
            lig = seq_item["ligand"]
            ids = _ensure_id_list(lig.get("id", []))
            ccd_codes = lig.get("ccdCodes")
            smiles_str = lig.get("smiles")
            for _ in ids or [None]:
                if ccd_codes:
                    ligand_chains.append(LigandChainInput(ccd_ids=list(ccd_codes), count=1))
                elif smiles_str:
                    ligand_chains.append(LigandChainInput(smiles=smiles_str, count=1))

    if not protein_chains and not dna_chains and not rna_chains and not ligand_chains:
        raise ValueError(f"No chains found in job {name!r}")

    return JobInput(
        name=name,
        protein_chains=protein_chains,
        dna_chains=dna_chains,
        rna_chains=rna_chains,
        ligand_chains=ligand_chains,
        model_seeds=model_seeds,
    )


def _ensure_id_list(ids: str | list[str]) -> list[str]:
    """Normalize ``id`` field to a list of chain IDs."""
    if isinstance(ids, str):
        return [ids]
    return list(ids)


# ---------------------------------------------------------------------------
# MSA resolution
# ---------------------------------------------------------------------------


def _resolve_msa(base_dir: Path, mol_type: int, path_str: str | None, inline_str: str | None) -> MSA | None:
    """Resolve MSA from a file path or inline string into a parsed MSA object.

    Args:
        base_dir: Base directory for resolving relative paths.
        mol_type: Molecule type (MOL_PROTEIN, MOL_DNA, MOL_RNA).
        path_str: Path to an A3M file (relative or absolute), or None.
        inline_str: Inline A3M text, or None.

    Returns:
        Parsed MSA object, or None if neither path nor inline string is provided.
    """
    if path_str is not None:
        resolved = _resolve_path(base_dir, path_str)
        if resolved is not None:
            try:
                text = resolved.read_text()
                msa = parse_a3m(text, mol_type)
                if msa.sequences.shape[0] > 0:
                    return msa
            except (FileNotFoundError, OSError) as e:
                logger.warning("Failed to read MSA file %s: %s", resolved, e)
        return None

    if inline_str is not None and inline_str.strip():
        msa = parse_a3m(inline_str, mol_type)
        if msa.sequences.shape[0] > 0:
            return msa

    return None


# ---------------------------------------------------------------------------
# Template resolution
# ---------------------------------------------------------------------------


def _resolve_template_hits(
    base_dir: Path, templates_path_str: str | None, max_templates: int, cache_dir: Path | None
) -> list[TemplateHitInput]:
    """Resolve template hits from a file path into parsed TemplateHitInput objects.

    Supports two template path formats:
    - ``.json``: JSON template hits file with pre-computed alignments and local CIF files
    - ``.hhr``: HHsearch output — parses hits and downloads CIFs from RCSB

    Args:
        base_dir: Base directory for resolving relative paths.
        templates_path_str: Path to template file (JSON or HHR), or None.
        max_templates: Maximum number of template hits to load.
        cache_dir: Directory to cache downloaded template mmCIF files.

    Returns:
        List of TemplateHitInput objects (empty if no templates specified).
    """
    if templates_path_str is None:
        return []

    resolved = _resolve_path(base_dir, templates_path_str)
    if resolved is None or not resolved.exists():
        logger.warning("Templates path not found: %s", templates_path_str)
        return []

    suffix = resolved.suffix.lower()
    if suffix == ".json":
        return _load_json_template_hits(resolved, max_templates)
    elif suffix == ".hhr":
        return _load_hhr_template_hits(resolved, max_templates, cache_dir)
    else:
        logger.warning("Unsupported template file type: %s", suffix)
        return []


def _load_json_template_hits(json_path: Path, max_templates: int) -> list[TemplateHitInput]:
    """Load template hits from a JSON file with pre-computed alignments.

    Expected format::

        [
            {
                "name": "template_hit_0.cif",
                "queryIndices": [0, 1, 5, ...],
                "templateIndices": [10, 11, 15, ...]
            },
            ...
        ]

    Returns:
        List of TemplateHitInput with mmCIF text and index pairs.
    """
    json_dir = json_path.parent

    try:
        with open(json_path) as f:
            hits = json.load(f)
    except Exception:
        logger.warning("Failed to read JSON template file %s", json_path, exc_info=True)
        return []

    if not isinstance(hits, list):
        logger.warning("JSON template hits file %s is not an array", json_path)
        return []

    result: list[TemplateHitInput] = []
    for hit in hits[:max_templates]:
        try:
            cif_name = hit["name"]
            query_indices = hit["queryIndices"]
            template_indices = hit["templateIndices"]

            if len(query_indices) != len(template_indices):
                logger.warning("Mismatched index lengths in template %s, skipping", cif_name)
                continue

            cif_path = json_dir / cif_name
            if not cif_path.exists():
                logger.warning("Template CIF not found: %s", cif_path)
                continue

            mmcif_text = cif_path.read_text()
            result.append(
                TemplateHitInput(
                    mmcif_text=mmcif_text,
                    query_indices=query_indices,
                    template_indices=template_indices,
                )
            )
            logger.info("Loaded JSON template hit %s", cif_name)
        except Exception:
            logger.warning("Failed to process JSON template hit", exc_info=True)

    return result


def _load_hhr_template_hits(hhr_path: Path, max_templates: int, cache_dir: Path | None) -> list[TemplateHitInput]:
    """Load template hits from an HHR file, downloading CIFs from RCSB.

    Returns:
        List of TemplateHitInput with mmCIF text and HHR hit objects.
    """
    from deepfold3.data.parse.hhr import parse_hhr

    try:
        hhr_text = hhr_path.read_text()
    except (FileNotFoundError, OSError) as e:
        logger.warning("Failed to read HHR file %s: %s", hhr_path, e)
        return []

    hits = parse_hhr(hhr_text)
    if not hits:
        logger.warning("No hits found in HHR file %s", hhr_path)
        return []

    result: list[TemplateHitInput] = []
    for hit in hits:
        if len(result) >= max_templates:
            break
        try:
            mmcif_text = fetch_pdb_mmcif(hit.pdb_id, cache_dir=cache_dir)
            result.append(
                TemplateHitInput(
                    mmcif_text=mmcif_text,
                    hhr_hit=hit,
                    chain_id=hit.chain_id,
                )
            )
            logger.info(
                "Loaded HHR template %s_%s (e_value=%.1e, aligned_cols=%d)",
                hit.pdb_id,
                hit.chain_id,
                hit.e_value,
                hit.aligned_cols,
            )
        except Exception:
            logger.warning("Failed to download template %s_%s", hit.pdb_id, hit.chain_id, exc_info=True)

    return result


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


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
