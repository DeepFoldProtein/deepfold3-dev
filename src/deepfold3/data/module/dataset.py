"""Dataset and inference input utilities.

TrainingDataset: loads Records, crops, and featurizes for training.
InferenceInput:  builds features directly from a sequence string (no mmCIF needed).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.utils.data

from deepfold3.data import const
from deepfold3.data.crop.cropper import Cropper
from deepfold3.data.feature.featurizer import Featurizer
from deepfold3.data.feature.ref_atoms import get_ref_atoms
from deepfold3.data.tokenize.tokenizer import Tokenizer
from deepfold3.data.types import (
    ATOM_DTYPE,
    BOND_DTYPE,
    CHAIN_DTYPE,
    MSA,
    RESIDUE_DTYPE,
    Input,
    Record,
    Structure,
)

if TYPE_CHECKING:
    from deepfold3.data.parse.templates import TemplateFeatures


class TrainingDataset(torch.utils.data.Dataset):
    """PyTorch dataset for DeepFold3 training.

    Loads Record -> Structure NPZ + MSA NPZ -> tokenize -> crop -> featurize.

    Args:
        records: List of Record objects.
        data_dir: Root directory containing structures/ and msa/ subdirectories.
        max_tokens: Maximum tokens per sample.
        max_atoms: Maximum atoms per sample.
        max_msa_seqs: Maximum MSA sequences.
    """

    def __init__(
        self,
        records: list[Record],
        data_dir: Path,
        max_tokens: int = 384,
        max_atoms: int = 4096,
        max_msa_seqs: int = 2048,
    ):
        self.records = records
        self.data_dir = Path(data_dir)
        self.max_tokens = max_tokens
        self.max_atoms = max_atoms
        self.tokenizer = Tokenizer()
        self.cropper = Cropper()
        self.featurizer = Featurizer(max_msa_seqs=max_msa_seqs)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        record = self.records[idx]

        # Load structure
        structure_path = self.data_dir / "structures" / f"{record.id}.npz"
        structure = Structure.load(structure_path)

        # Load MSAs keyed by chain index (treated as unpaired for disk-loaded MSAs)
        msa: dict[int, MSA] = {}
        for chain_info in record.chains:
            msa_path = self.data_dir / "msa" / f"{chain_info.msa_id}.npz"
            if msa_path.exists():
                msa[chain_info.chain_id] = MSA.load(msa_path)

        inp = Input(structure=structure, paired_msa={}, unpaired_msa=msa)

        # Tokenize -> crop -> featurize
        tokenized = self.tokenizer.tokenize(inp)
        tokenized = self.cropper.crop(tokenized, self.max_tokens, self.max_atoms)
        feats = self.featurizer.process(tokenized, training=True, random_rot=True)

        return feats


def _tokenize_and_featurize(
    inp: Input,
    max_msa_seqs: int = 16384,
    templates: list[TemplateFeatures] | None = None,
    template_feats_per_chain: dict[int, list[TemplateFeatures]] | None = None,
) -> dict[str, torch.Tensor]:
    """Shared tokenize -> featurize pipeline for inference inputs."""
    tokenizer = Tokenizer()
    featurizer = Featurizer(max_msa_seqs=max_msa_seqs)
    tokenized = tokenizer.tokenize(inp)
    return featurizer.process(
        tokenized,
        training=False,
        random_rot=False,
        templates=templates,
        template_feats_per_chain=template_feats_per_chain,
    )


class InferenceInput:
    """Build model-ready features from a raw sequence string, A3M, or JSON.

    Uses CCD ideal coordinates for structure building.  MSA is either
    parsed from an A3M file or synthesized as a single-sequence placeholder.
    """

    @staticmethod
    def from_a3m(
        a3m_text: str,
        mol_type: str = "protein",
        max_msa_seqs: int = 16384,
        templates: list[TemplateFeatures] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Build features from an A3M-format MSA string.

        The query sequence (first record) determines the structure;
        all sequences contribute to the MSA features.

        Args:
            a3m_text: Full A3M text (headers + sequences).
            mol_type: 'protein', 'rna', or 'dna'.
            max_msa_seqs: Maximum MSA sequences to keep.
            templates: Optional list of TemplateFeatures for template embedding.
        """
        from deepfold3.data.parse.a3m import parse_a3m

        mt = const.MOL_TYPE_IDS.get(mol_type.upper(), const.MOL_PROTEIN)

        msa_obj = parse_a3m(a3m_text, mt)
        if msa_obj.sequences.shape[0] == 0:
            raise ValueError("A3M file contains no sequences")

        query_seq = _msa_indices_to_sequence(msa_obj.sequences[0], mt)
        structure = _sequence_to_structure(query_seq, mt)
        inp = Input(structure=structure, paired_msa={}, unpaired_msa={0: msa_obj})
        return _tokenize_and_featurize(inp, max_msa_seqs=max_msa_seqs, templates=templates)

    @staticmethod
    def from_sequence(
        sequence: str,
        mol_type: str = "protein",
        max_msa_seqs: int = 1,
        templates: list[TemplateFeatures] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Build features for a single sequence.

        Args:
            sequence: 1-letter sequence string (protein) or RNA/DNA sequence.
            mol_type: 'protein', 'rna', or 'dna'.
            max_msa_seqs: Maximum MSA sequences (default 1 = single sequence).
            templates: Optional list of TemplateFeatures for template embedding.

        Returns:
            dict[str, torch.Tensor] ready for model.forward().
        """
        mt = const.MOL_TYPE_IDS.get(mol_type.upper(), const.MOL_PROTEIN)
        structure = _sequence_to_structure(sequence, mt)
        msa = _sequence_to_msa(sequence, mt)
        inp = Input(structure=structure, paired_msa={}, unpaired_msa={0: msa})
        return _tokenize_and_featurize(inp, max_msa_seqs=max_msa_seqs, templates=templates)

    @staticmethod
    def from_json(
        json_path: str | Path,
        job_index: int = 0,
        max_msa_seqs: int = 16384,
        max_templates: int = 4,
        template_cache_dir: str | Path | None = None,
    ) -> dict[str, torch.Tensor]:
        """Build features from a Protenix-format or AF3-format JSON input file.

        Supports multi-chain complexes with protein, DNA, and RNA chains.
        Each chain can optionally specify paired/unpaired MSAs and templates.

        All file I/O (reading A3M files, downloading template CIFs) is handled
        by ``parse_json_input()``.  This method only does computation.

        Args:
            json_path: Path to the JSON input file.
            job_index: Which job entry to use (default 0).
            max_msa_seqs: Maximum MSA sequences to keep.
            max_templates: Maximum number of templates to extract.
            template_cache_dir: Directory to cache downloaded template mmCIF files.

        Returns:
            dict[str, torch.Tensor] ready for model.forward().
        """
        from deepfold3.data.parse.json_input import parse_json_input
        from deepfold3.data.parse.templates import TemplateHitInput, template_hit_input_to_features

        json_path = Path(json_path)
        jobs = parse_json_input(json_path, max_templates=max_templates, template_cache_dir=template_cache_dir)

        if job_index >= len(jobs):
            raise IndexError(f"Job index {job_index} out of range (file has {len(jobs)} jobs)")

        job = jobs[job_index]

        # Collect all chains in order: protein, DNA, RNA
        # Each entry: (sequence, mol_type, paired_msa, unpaired_msa, template_hits)
        chain_specs: list[tuple[str, int, MSA | None, MSA | None, list[TemplateHitInput]]] = []
        for pc in job.protein_chains:
            chain_specs.append((pc.sequence, const.MOL_PROTEIN, pc.paired_msa, pc.unpaired_msa, pc.template_hits))
        for dc in job.dna_chains:
            chain_specs.append((dc.sequence, dc.mol_type, dc.paired_msa, dc.unpaired_msa, []))
        for rc in job.rna_chains:
            chain_specs.append((rc.sequence, rc.mol_type, rc.paired_msa, rc.unpaired_msa, []))

        if not chain_specs:
            raise ValueError(f"No chains found in job {job.name!r}")

        # Build multi-chain structure with auto-generated chain names
        chain_tuples = [(seq, mt, chr(ord("A") + i)) for i, (seq, mt, _, _, _) in enumerate(chain_specs)]
        structure = _build_complex_structure(chain_tuples)

        # Assemble MSA dicts from parsed objects
        paired_msa_dict: dict[int, MSA] = {}
        unpaired_msa_dict: dict[int, MSA] = {}
        for asym_id, (seq, mt, paired_msa, unpaired_msa, _) in enumerate(chain_specs):
            if paired_msa is not None:
                paired_msa_dict[asym_id] = paired_msa
            if unpaired_msa is not None:
                unpaired_msa_dict[asym_id] = unpaired_msa
            elif paired_msa is None:
                # No MSA at all — fall back to single-sequence MSA
                unpaired_msa_dict[asym_id] = _sequence_to_msa(seq, mt)
            else:
                # Only paired exists — use single-sequence as unpaired
                unpaired_msa_dict[asym_id] = _sequence_to_msa(seq, mt)

        # Convert template hits to features
        template_feats: list[TemplateFeatures] | None = None
        template_feats_per_chain: dict[int, list[TemplateFeatures]] = {}

        # Check for whole-complex JSON template hits (query_indices populated)
        json_tmpl_found = False
        for _asym_id, (_seq, _mt, _, _, tmpl_hits) in enumerate(chain_specs):
            if tmpl_hits and tmpl_hits[0].query_indices is not None:
                # JSON-mode templates: whole-complex alignment
                total_tokens = sum(len(s) for s, _, _, _, _ in chain_specs)
                template_feats = [template_hit_input_to_features(h, total_tokens) for h in tmpl_hits]
                if not template_feats:
                    template_feats = None
                json_tmpl_found = True
                break

        # Fall back to per-chain HHR templates
        if not json_tmpl_found:
            for asym_id, (seq, mt, _, _, tmpl_hits) in enumerate(chain_specs):
                if mt == const.MOL_PROTEIN and tmpl_hits:
                    chain_templates = [template_hit_input_to_features(h, len(seq)) for h in tmpl_hits]
                    if chain_templates:
                        template_feats_per_chain[asym_id] = chain_templates

            if len(template_feats_per_chain) == 1:
                template_feats = next(iter(template_feats_per_chain.values()))
            elif len(template_feats_per_chain) > 1:
                template_feats = next(iter(template_feats_per_chain.values()))

        inp = Input(structure=structure, paired_msa=paired_msa_dict, unpaired_msa=unpaired_msa_dict)
        return _tokenize_and_featurize(
            inp,
            max_msa_seqs=max_msa_seqs,
            templates=template_feats,
            template_feats_per_chain=template_feats_per_chain if len(template_feats_per_chain) > 1 else None,
        )


def _letter_to_res_name(letter: str, mol_type: int) -> str:
    """Map a single-character residue letter to its canonical residue name."""
    if mol_type == const.MOL_PROTEIN:
        return const.ONE_LETTER_TO_THREE.get(letter, "UNK")
    elif mol_type == const.MOL_RNA:
        return const.RNA_LETTER_TO_RES.get(letter, "N")
    elif mol_type == const.MOL_DNA:
        return const.DNA_LETTER_TO_RES.get(letter, "DN")
    return "UNK"


def _build_chain_atoms_residues(
    sequence: str,
    mol_type: int,
    atom_offset: int,
) -> tuple[list[tuple], list[tuple], int]:
    """Build atom and residue tuples for a single chain.

    Returns:
        (atoms_list, residues_list, total_atoms_in_chain)
    """
    atoms_list: list[tuple] = []
    residues_list: list[tuple] = []

    atom_idx = atom_offset
    for ri, letter in enumerate(sequence):
        res_name_3 = _letter_to_res_name(letter, mol_type)
        res_type = const.res_name_to_type(res_name_3)

        ref = get_ref_atoms(res_name_3, mol_type)
        n_atoms = int(ref.mask.sum())

        for j in range(n_atoms):
            atoms_list.append(
                (
                    float(ref.positions[j, 0]),
                    float(ref.positions[j, 1]),
                    float(ref.positions[j, 2]),
                    int(ref.element[j]),
                    0.0,  # charge
                    _atom_name_from_chars(ref.atom_name_chars[j]).encode("ascii"),
                    True,
                    0.0,  # b_factor
                )
            )

        residues_list.append(
            (
                res_name_3[:5].encode("ascii"),
                res_type,
                mol_type,
                ri + 1,  # res_id (1-based)
                atom_idx,
                n_atoms,
                True,  # is_standard
                True,  # is_present
            )
        )
        atom_idx += n_atoms

    return atoms_list, residues_list, atom_idx - atom_offset


def _build_complex_structure(chains: list[tuple[str, int, str]]) -> Structure:
    """Build a multi-chain Structure with sequence-based entity_id grouping.

    Chains with identical (normalized) sequences share the same entity_id,
    with incrementing sym_id to distinguish copies. This matches AF3 semantics
    where entity_id encodes sequence identity and sym_id encodes copy number.

    Args:
        chains: List of (sequence, mol_type, chain_name) tuples. mol_type is
            one of const.MOL_PROTEIN, const.MOL_DNA, const.MOL_RNA.

    Returns:
        Structure with properly assigned entity_id and sym_id.
    """
    # Map normalized sequences to (entity_id, next_sym_id)
    seq_to_entity: dict[str, list[int]] = {}  # normalized_seq -> [entity_id, next_sym_id]
    next_entity_id = 1

    all_atoms: list[tuple] = []
    all_residues: list[tuple] = []
    all_chains: list[tuple] = []

    atom_offset = 0
    res_offset = 0

    for chain_idx, (sequence, mol_type, chain_name) in enumerate(chains):
        sequence = sequence.strip().upper()
        n_residues = len(sequence)

        # Determine entity_id and sym_id based on sequence identity
        if sequence in seq_to_entity:
            entity_id = seq_to_entity[sequence][0]
            sym_id = seq_to_entity[sequence][1]
            seq_to_entity[sequence][1] += 1
        else:
            entity_id = next_entity_id
            sym_id = 0
            seq_to_entity[sequence] = [entity_id, 1]
            next_entity_id += 1

        # Build atoms and residues for this chain
        chain_atoms, chain_residues, chain_atom_count = _build_chain_atoms_residues(sequence, mol_type, atom_offset)
        all_atoms.extend(chain_atoms)
        all_residues.extend(chain_residues)

        all_chains.append(
            (
                chain_name.encode("ascii"),
                mol_type,
                entity_id,  # sequence-based entity_id (1-based)
                sym_id,  # copy index within entity (0-based)
                chain_idx,  # asym_id (unique chain index)
                res_offset,  # res_idx
                n_residues,  # res_num
                atom_offset,  # atom_idx
                chain_atom_count,  # atom_num
            )
        )

        atom_offset += chain_atom_count
        res_offset += n_residues

    atoms_arr = np.array(all_atoms, dtype=ATOM_DTYPE) if all_atoms else np.empty(0, dtype=ATOM_DTYPE)
    residues_arr = np.array(all_residues, dtype=RESIDUE_DTYPE) if all_residues else np.empty(0, dtype=RESIDUE_DTYPE)
    chains_arr = np.array(all_chains, dtype=CHAIN_DTYPE) if all_chains else np.empty(0, dtype=CHAIN_DTYPE)
    bonds_arr = np.empty(0, dtype=BOND_DTYPE)

    return Structure(
        atoms=atoms_arr,
        bonds=bonds_arr,
        residues=residues_arr,
        chains=chains_arr,
    )


def _sequence_to_structure(sequence: str, mol_type: int) -> Structure:
    """Build a single-chain Structure from a 1-letter sequence (no mmCIF needed)."""
    return _build_complex_structure([(sequence, mol_type, "A")])


def _sequence_to_msa(sequence: str, mol_type: int) -> MSA:
    """Build a single-sequence MSA from a 1-letter sequence."""
    N = len(sequence)
    sequences = np.zeros((1, N), dtype=np.int8)
    mask = np.ones((1, N), dtype=bool)
    deletion_matrix = np.zeros((1, N), dtype=np.int8)

    if mol_type == const.MOL_PROTEIN:
        lookup = const.PROTEIN_MSA_ID
        unk = 20
    elif mol_type == const.MOL_RNA:
        lookup = const.RNA_MSA_ID
        unk = const.MSA_UNK_NUCLEIC
    elif mol_type == const.MOL_DNA:
        lookup = const.DNA_MSA_ID
        unk = const.MSA_UNK_NUCLEIC
    else:
        lookup = const.PROTEIN_MSA_ID
        unk = 20

    for i, ch in enumerate(sequence.upper()):
        sequences[0, i] = lookup.get(ch, unk)

    return MSA(sequences=sequences, mask=mask, deletion_matrix=deletion_matrix)


def _atom_name_from_chars(chars: np.ndarray) -> str:
    """Decode 4-int atom name encoding back to string."""
    return "".join(chr(int(c) + 32) for c in chars).strip()


def _msa_indices_to_sequence(encoded: np.ndarray, mol_type: int) -> str:
    """Convert MSA-index row back to a 1-letter sequence string.

    Gap positions (index 21) are skipped — this reconstructs the query
    match-state sequence from the first row of a parsed A3M.
    """
    if mol_type == const.MOL_PROTEIN:
        rev = {v: k for k, v in const.PROTEIN_MSA_ID.items() if k not in ("-", "B", "J", "O", "U", "Z")}
        rev[20] = "X"
        rev[21] = "-"
    elif mol_type == const.MOL_RNA:
        rev = {v: k for k, v in const.RNA_MSA_ID.items()}
        rev[const.MSA_UNK_NUCLEIC] = "N"
    elif mol_type == const.MOL_DNA:
        rev = {v: k for k, v in const.DNA_MSA_ID.items()}
        rev[const.MSA_UNK_NUCLEIC] = "N"
    else:
        rev = {v: k for k, v in const.PROTEIN_MSA_ID.items() if k not in ("-", "B", "J", "O", "U", "Z")}
        rev[20] = "X"
        rev[21] = "-"

    return "".join(rev.get(int(idx), "X") for idx in encoded if int(idx) != 21)
