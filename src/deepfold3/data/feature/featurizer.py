"""Featurizer: Tokenized → dict[str, Tensor].

Produces all features needed by the DeepFold3 model forward pass.
Feature specifications follow AlphaFold3 exactly.
"""

from __future__ import annotations

import numpy as np
import torch

from deepfold3.data import const
from deepfold3.data.feature.ref_atoms import get_ref_atoms
from deepfold3.data.types import Tokenized


def _random_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    """Sample a uniformly random 3×3 rotation matrix using QR decomposition."""
    H = rng.standard_normal((3, 3)).astype(np.float64)
    Q, R = np.linalg.qr(H)
    # Ensure proper rotation (det=+1)
    Q = Q * np.sign(np.diag(R))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q.astype(np.float32)


class Featurizer:
    """Convert Tokenized data into model-ready tensors."""

    def __init__(
        self,
        max_atoms: int = const.MAX_ATOMS_PER_TOKEN,
        max_msa_seqs: int = 2048,
    ):
        self.max_atoms = max_atoms
        self.max_msa_seqs = max_msa_seqs

    def process(
        self,
        data: Tokenized,
        training: bool = True,
        random_rot: bool = True,
        seed: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Process tokenized data into a feature dictionary.

        Returns
        -------
        dict with keys:
            res_type          [N]         int64
            residue_index     [N]         int64
            token_index       [N]         int64
            asym_id           [N]         int64
            entity_id         [N]         int64
            sym_id            [N]         int64
            seq_mask          [N]         bool
            ref_pos           [N, M, 3]   float32
            ref_mask          [N, M]      bool
            ref_element       [N, M]      int64
            ref_charge        [N, M]      float32
            ref_atom_name_chars [N, M, 4] int64
            msa               [S, N]      int64
            msa_mask          [S, N]      bool
            deletion_matrix   [S, N]      int64
            profile           [N, 31]     float32
            deletion_mean     [N]         float32
        """
        rng = np.random.default_rng(seed)
        tokens = data.tokens
        N = len(tokens)

        if N == 0:
            return self._empty_features()

        M = self.max_atoms

        # ------------------------------------------------------------------
        # Basic token features
        # ------------------------------------------------------------------
        res_type = torch.from_numpy(tokens["res_type"].astype(np.int64))
        residue_index = torch.from_numpy(tokens["residue_index"].astype(np.int64))
        token_index = torch.from_numpy(tokens["token_idx"].astype(np.int64))
        asym_id = torch.from_numpy(tokens["asym_id"].astype(np.int64))
        entity_id = torch.from_numpy(tokens["entity_id"].astype(np.int64))
        sym_id = torch.from_numpy(tokens["sym_id"].astype(np.int64))
        seq_mask = torch.ones(N, dtype=torch.bool)

        # ------------------------------------------------------------------
        # Reference atom features
        # ------------------------------------------------------------------
        ref_pos = np.zeros((N, M, 3), dtype=np.float32)
        ref_mask = np.zeros((N, M), dtype=bool)
        ref_element = np.zeros((N, M), dtype=np.int64)
        ref_charge = np.zeros((N, M), dtype=np.float32)
        ref_atom_name_chars = np.zeros((N, M, 4), dtype=np.int64)

        for i, tok in enumerate(tokens):
            mol_type_tok = int(tok["mol_type"])

            # Get residue name
            # For polymer residues, look up in residues array
            # For ligand (nonpolymer), each token = 1 atom
            if mol_type_tok == const.MOL_NONPOLYMER:
                # Single atom token: use actual atom data
                atom_idx = int(tok["atom_idx"])
                if atom_idx < len(data.structure.atoms):
                    atom = data.structure.atoms[atom_idx]
                    atom_name = atom["name"].decode("ascii", errors="replace").strip()
                    elem_int = int(np.clip(atom["element"], 0, 127))
                    chg = float(atom["charge"])
                    ref_atom_data = get_ref_atoms(
                        res_name="LIG",
                        mol_type=mol_type_tok,
                        max_atoms=M,
                        atom_names_override=[atom_name],
                    )
                    ref_atom_data.element[0] = elem_int
                    ref_atom_data.charge[0] = chg
                else:
                    ref_atom_data = get_ref_atoms("UNK", mol_type_tok, M)
            else:
                # Find the residue name from the structure
                # token atom_idx points to first atom of this residue
                atom_idx = int(tok["atom_idx"])
                atom_num = int(tok["atom_num"])
                res_name = _find_res_name(data, atom_idx)
                ref_atom_data = get_ref_atoms(res_name, mol_type_tok, M)

                # Override positions and elements with actual structure data
                # (for training we use ideal CCD positions with random rotation)
                # Copy actual atom info into ref features
                for j in range(min(atom_num, M)):
                    if atom_idx + j < len(data.structure.atoms):
                        atom = data.structure.atoms[atom_idx + j]
                        if bool(atom["is_present"]):
                            ref_atom_data.element[j] = int(np.clip(atom["element"], 0, 127))
                            ref_atom_data.charge[j] = float(atom["charge"])
                            aname = atom["name"].decode("ascii", errors="replace").strip()
                            from deepfold3.data.feature.ref_atoms import _encode_atom_name

                            ref_atom_data.atom_name_chars[j] = _encode_atom_name(aname)

            # Apply random augmentation (center → rotate → translate) per token.
            # Matches AF3's random_augmentation(): center to mean of valid atoms,
            # random SO(3) rotation, then random N(0,1) translation.
            if random_rot and ref_atom_data.mask.any():
                valid = ref_atom_data.mask.astype(bool)
                center = ref_atom_data.positions[valid].mean(axis=0)
                rot = _random_rotation_matrix(rng)  # [3, 3]
                translation = rng.standard_normal(3).astype(np.float32)
                ref_atom_data.positions[:] = (ref_atom_data.positions - center) @ rot.T + translation

            ref_pos[i] = ref_atom_data.positions
            ref_mask[i] = ref_atom_data.mask
            ref_element[i] = ref_atom_data.element
            ref_charge[i] = ref_atom_data.charge
            ref_atom_name_chars[i] = ref_atom_data.atom_name_chars

        # ------------------------------------------------------------------
        # MSA features
        # ------------------------------------------------------------------
        msa_tensor, msa_mask_tensor, del_matrix_tensor = self._build_msa_features(data, tokens, N, rng)

        # Profile: per-position residue frequency [N, 31]
        # Includes gap class (index 21) to encode coverage, matching AF3.
        profile = self._compute_profile(msa_tensor, msa_mask_tensor, N)

        # Deletion mean: raw mean of deletion counts per position [N]
        # AF3 stores the raw mean (arctan transform is applied separately in create_msa_feat).
        deletion_mean = del_matrix_tensor.float().mean(dim=0)  # [N]

        pair_mask = seq_mask.unsqueeze(1) & seq_mask.unsqueeze(0)  # [N, N]

        return {
            "res_type": res_type,
            "residue_index": residue_index,
            "token_index": token_index,
            "asym_id": asym_id,
            "entity_id": entity_id,
            "sym_id": sym_id,
            "seq_mask": seq_mask,
            "pair_mask": pair_mask,
            "ref_pos": torch.from_numpy(ref_pos),
            "ref_mask": torch.from_numpy(ref_mask),
            "ref_element": torch.from_numpy(ref_element),
            "ref_charge": torch.from_numpy(ref_charge),
            "ref_atom_name_chars": torch.from_numpy(ref_atom_name_chars),
            "msa": msa_tensor.long(),
            "msa_mask": msa_mask_tensor,
            "deletion_matrix": del_matrix_tensor.long(),
            "profile": profile,
            "deletion_mean": deletion_mean,
        }

    def _build_msa_features(
        self,
        data: Tokenized,
        tokens: np.ndarray,
        N: int,
        rng: np.random.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build [S, N] MSA tensors.

        Concatenates MSA from all chains; pads to same column count.
        """
        # Collect per-chain MSA aligned to token positions
        all_seqs: list[np.ndarray] = []
        all_masks: list[np.ndarray] = []
        all_dels: list[np.ndarray] = []

        for asym_id, msa in data.msa.items():
            chain_mask = tokens["asym_id"] == asym_id
            n_chain = int(chain_mask.sum())
            if n_chain == 0:
                continue

            chain_pos = np.where(chain_mask)[0]  # positions in [0, N)
            S, N_msa = msa.sequences.shape

            # Pad/truncate MSA columns to match n_chain
            seqs = np.full((S, N), 21, dtype=np.int8)  # gap for cross-chain padding
            masks = np.zeros((S, N), dtype=bool)
            dels = np.zeros((S, N), dtype=np.int8)

            L = min(N_msa, n_chain)
            seqs[:, chain_pos[:L]] = msa.sequences[:, :L]
            masks[:, chain_pos[:L]] = msa.mask[:, :L]
            dels[:, chain_pos[:L]] = msa.deletion_matrix[:, :L]

            all_seqs.append(seqs)
            all_masks.append(masks)
            all_dels.append(dels)

        if not all_seqs:
            # No MSA: single row of zeros (gap)
            msa_arr = np.zeros((1, N), dtype=np.int8)
            mask_arr = np.ones((1, N), dtype=bool)
            del_arr = np.zeros((1, N), dtype=np.int8)
        else:
            # Combine all chains' MSAs (pair/unpair handling simplified here:
            # just stack all query sequences from all chains as separate rows)
            # The first row from each chain's MSA is the query
            combined_seqs_list: list[np.ndarray] = []
            combined_masks_list: list[np.ndarray] = []
            combined_dels_list: list[np.ndarray] = []

            # First add query (row 0) from all chains merged into one row
            query_seq = np.zeros(N, dtype=np.int8)
            query_mask = np.zeros(N, dtype=bool)
            query_del = np.zeros(N, dtype=np.int8)
            for seqs, masks, dels in zip(all_seqs, all_masks, all_dels, strict=False):
                query_seq = np.where(masks[0], seqs[0], query_seq)
                query_mask |= masks[0]
                query_del = np.where(masks[0], dels[0], query_del)
            combined_seqs_list.append(query_seq[None])
            combined_masks_list.append(query_mask[None])
            combined_dels_list.append(query_del[None])

            # Add remaining rows from each chain's MSA
            for seqs, masks, dels in zip(all_seqs, all_masks, all_dels, strict=False):
                if seqs.shape[0] > 1:
                    combined_seqs_list.append(seqs[1:])
                    combined_masks_list.append(masks[1:])
                    combined_dels_list.append(dels[1:])

            msa_arr = np.concatenate(combined_seqs_list, axis=0)
            mask_arr = np.concatenate(combined_masks_list, axis=0)
            del_arr = np.concatenate(combined_dels_list, axis=0)

        # Sample to max_msa_seqs
        S = msa_arr.shape[0]
        if S > self.max_msa_seqs:
            idx = np.concatenate([[0], rng.choice(S - 1, self.max_msa_seqs - 1, replace=False) + 1])
            msa_arr = msa_arr[idx]
            mask_arr = mask_arr[idx]
            del_arr = del_arr[idx]

        return (
            torch.from_numpy(msa_arr.astype(np.int16)),
            torch.from_numpy(mask_arr),
            torch.from_numpy(del_arr.astype(np.int16)),
        )

    def _compute_profile(
        self,
        msa: torch.Tensor,  # [S, N] int
        msa_mask: torch.Tensor,  # [S, N] bool
        N: int,
    ) -> torch.Tensor:
        """Compute per-position residue frequency profile [N, 31]."""
        # One-hot encode only the first 31 classes (0–30)
        msa_clipped = msa.long().clamp(0, const.RES_TYPE_COUNT - 1)  # [S, N]
        one_hot = torch.zeros(msa.shape[0], N, const.RES_TYPE_COUNT)
        one_hot.scatter_(2, msa_clipped.unsqueeze(-1), 1.0)

        # Mask out gap positions and invalid rows
        valid = msa_mask.float().unsqueeze(-1)  # [S, N, 1]
        weighted = one_hot * valid  # [S, N, 31]

        denom = valid.sum(0).clamp(min=1.0)  # [N, 1]
        profile = weighted.sum(0) / denom  # [N, 31]
        return profile

    def _empty_features(self) -> dict[str, torch.Tensor]:
        M = self.max_atoms
        return {
            "res_type": torch.zeros(0, dtype=torch.int64),
            "residue_index": torch.zeros(0, dtype=torch.int64),
            "token_index": torch.zeros(0, dtype=torch.int64),
            "asym_id": torch.zeros(0, dtype=torch.int64),
            "entity_id": torch.zeros(0, dtype=torch.int64),
            "sym_id": torch.zeros(0, dtype=torch.int64),
            "seq_mask": torch.zeros(0, dtype=torch.bool),
            "ref_pos": torch.zeros(0, M, 3),
            "ref_mask": torch.zeros(0, M, dtype=torch.bool),
            "ref_element": torch.zeros(0, M, dtype=torch.int64),
            "ref_charge": torch.zeros(0, M),
            "ref_atom_name_chars": torch.zeros(0, M, 4, dtype=torch.int64),
            "msa": torch.zeros(1, 0, dtype=torch.int64),
            "msa_mask": torch.zeros(1, 0, dtype=torch.bool),
            "deletion_matrix": torch.zeros(1, 0, dtype=torch.int64),
            "profile": torch.zeros(0, const.RES_TYPE_COUNT),
            "deletion_mean": torch.zeros(0),
        }


def _find_res_name(data: Tokenized, atom_idx: int) -> str:
    """Find residue name from atom index by searching residues array."""
    for res in data.structure.residues:
        start = int(res["atom_idx"])
        end = start + int(res["atom_num"])
        if start <= atom_idx < end:
            return res["name"].decode("ascii", errors="replace").strip()
    return "UNK"
