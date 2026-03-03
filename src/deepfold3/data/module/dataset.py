"""Dataset and inference input utilities.

TrainingDataset: loads Records, crops, and featurizes for training.
InferenceInput:  builds features directly from a sequence string (no mmCIF needed).
"""

from __future__ import annotations

from pathlib import Path

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


class TrainingDataset(torch.utils.data.Dataset):
    """PyTorch dataset for DeepFold3 training.

    Loads Record → Structure NPZ + MSA NPZ → tokenize → crop → featurize.

    Parameters
    ----------
    records :       List of Record objects
    data_dir :      Root directory containing structures/ and msa/ subdirectories
    max_tokens :    Maximum tokens per sample
    max_atoms :     Maximum atoms per sample
    max_msa_seqs :  Maximum MSA sequences
    """

    def __init__(
        self,
        records: list[Record],
        data_dir: Path,
        max_tokens: int = 384,
        max_atoms: int = 2048,
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

        # Load MSAs keyed by chain index
        msa: dict[int, MSA] = {}
        for chain_info in record.chains:
            msa_path = self.data_dir / "msa" / f"{chain_info.msa_id}.npz"
            if msa_path.exists():
                msa[chain_info.chain_id] = MSA.load(msa_path)

        inp = Input(structure=structure, msa=msa)

        # Tokenize → crop → featurize
        tokenized = self.tokenizer.tokenize(inp)
        tokenized = self.cropper.crop(tokenized, self.max_tokens, self.max_atoms)
        feats = self.featurizer.process(tokenized, training=True, random_rot=True)

        return feats


class InferenceInput:
    """Build model-ready features from a raw sequence string.

    No mmCIF or MSA files required — uses CCD ideal coordinates and a
    single-sequence "MSA" (just the query).
    """

    @staticmethod
    def from_a3m(
        a3m_text: str,
        mol_type: str = "protein",
        max_msa_seqs: int = 2048,
    ) -> dict[str, torch.Tensor]:
        """Build features from an A3M-format MSA string.

        The query sequence (first record) determines the structure;
        all sequences contribute to the MSA features.

        Parameters
        ----------
        a3m_text :    Full A3M text (headers + sequences)
        mol_type :    'protein', 'rna', or 'dna'
        max_msa_seqs: maximum MSA sequences to keep
        """
        from deepfold3.data.parse.a3m import parse_a3m

        mt = const.MOL_TYPE_IDS.get(mol_type.upper(), const.MOL_PROTEIN)

        # Parse MSA — query sequence is the first record
        msa_obj = parse_a3m(a3m_text, mt)
        if msa_obj.sequences.shape[0] == 0:
            raise ValueError("A3M file contains no sequences")

        # Extract query sequence from MSA row 0 (MSA indices → 1-letter)
        query_encoded = msa_obj.sequences[0]  # [N] int8 MSA indices
        query_seq = _msa_indices_to_sequence(query_encoded, mt)

        # Build structure from query sequence
        structure = _sequence_to_structure(query_seq, mt)

        inp = Input(structure=structure, msa={0: msa_obj})

        tokenizer = Tokenizer()
        featurizer = Featurizer(max_msa_seqs=max_msa_seqs)

        tokenized = tokenizer.tokenize(inp)
        feats = featurizer.process(tokenized, training=False, random_rot=False)

        return feats

    @staticmethod
    def from_sequence(
        sequence: str,
        mol_type: str = "protein",
        max_msa_seqs: int = 1,
    ) -> dict[str, torch.Tensor]:
        """Build features for a single sequence.

        Parameters
        ----------
        sequence :    1-letter sequence string (protein) or RNA/DNA sequence
        mol_type :    'protein', 'rna', or 'dna'
        max_msa_seqs: maximum MSA sequences (default 1 = single sequence)

        Returns
        -------
        dict[str, torch.Tensor] ready for model.forward()
        """
        mt = const.MOL_TYPE_IDS.get(mol_type.upper(), const.MOL_PROTEIN)

        # Build Structure from sequence
        structure = _sequence_to_structure(sequence, mt)

        # Build single-sequence MSA
        msa = _sequence_to_msa(sequence, mt)

        inp = Input(structure=structure, msa={0: msa})

        tokenizer = Tokenizer()
        featurizer = Featurizer(max_msa_seqs=max_msa_seqs)

        tokenized = tokenizer.tokenize(inp)
        feats = featurizer.process(tokenized, training=False, random_rot=False)

        return feats


def _sequence_to_structure(sequence: str, mol_type: int) -> Structure:
    """Build a Structure from a 1-letter sequence (no mmCIF needed)."""
    sequence = sequence.strip().upper()
    n = len(sequence)

    atoms_list: list[tuple] = []
    residues_list: list[tuple] = []

    atom_idx = 0
    for ri, letter in enumerate(sequence):
        # Map 1-letter to residue name
        if mol_type == const.MOL_PROTEIN:
            res_name_3 = const.ONE_LETTER_TO_THREE.get(letter, "UNK")
        elif mol_type == const.MOL_RNA:
            rna_map = {"A": "A", "G": "G", "C": "C", "U": "U"}
            res_name_3 = rna_map.get(letter, "N")
        elif mol_type == const.MOL_DNA:
            dna_map = {"A": "DA", "G": "DG", "C": "DC", "T": "DT"}
            res_name_3 = dna_map.get(letter, "DN")
        else:
            res_name_3 = "UNK"

        res_type = const.res_name_to_type(res_name_3)

        # Get reference atoms (ideal CCD positions)
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

    # Single chain
    total_atoms = atom_idx
    chains_list = [
        (
            b"A",  # name
            mol_type,  # mol_type
            1,  # entity_id (1-based, mmCIF convention)
            0,  # sym_id
            0,  # asym_id
            0,  # res_idx
            n,  # res_num
            0,  # atom_idx
            total_atoms,  # atom_num
        )
    ]

    atoms_arr = np.array(atoms_list, dtype=ATOM_DTYPE) if atoms_list else np.empty(0, dtype=ATOM_DTYPE)
    residues_arr = np.array(residues_list, dtype=RESIDUE_DTYPE)
    chains_arr = np.array(chains_list, dtype=CHAIN_DTYPE)
    bonds_arr = np.empty(0, dtype=BOND_DTYPE)

    return Structure(
        atoms=atoms_arr,
        bonds=bonds_arr,
        residues=residues_arr,
        chains=chains_arr,
    )


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
