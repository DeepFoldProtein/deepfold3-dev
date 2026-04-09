"""Featurizer: Tokenized → dict[str, Tensor].

Produces all features needed by the DeepFold3 model forward pass.
Feature specifications follow AlphaFold3 exactly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from deepfold3.data import const
from deepfold3.data.feature.ref_atoms import _encode_atom_name, get_ref_atoms
from deepfold3.data.types import Tokenized

if TYPE_CHECKING:
    from deepfold3.data.parse.templates import TemplateFeatures


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
        max_msa_seqs: int = const.MAX_MSA_ROWS,
        max_templates: int = 4,
    ):
        self.max_atoms = max_atoms
        self.max_msa_seqs = max_msa_seqs
        self.max_templates = max_templates

    def process(
        self,
        data: Tokenized,
        training: bool = True,
        random_rot: bool = True,
        seed: int | None = None,
        templates: list[TemplateFeatures] | None = None,
        template_feats_per_chain: dict[int, list[TemplateFeatures]] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Processes tokenized data into a feature dictionary.

        Args:
            data: Tokenized input (tokens, bonds, structure, MSA).
            training: Whether to apply training augmentations.
            random_rot: Whether to apply random rotation to ref_pos.
            seed: Random seed for augmentation reproducibility.
            templates: Flat list of TemplateFeatures (single-chain or whole-complex).
            template_feats_per_chain: Per-chain template features keyed by asym_id.

        Returns:
            Dict of tensors with keys:
                res_type [N] int64, residue_index [N] int64,
                token_index [N] int64, asym_id [N] int64,
                entity_id [N] int64, sym_id [N] int64,
                seq_mask [N] bool, pair_mask [N,N] bool,
                frames_mask [N] bool, token_to_rep_atom [N] int64,
                ref_pos [N,M,3] float32, ref_mask [N,M] bool,
                ref_element [N,M] int64, ref_charge [N,M] float32,
                ref_atom_name_chars [N,M,4] int64,
                ref_space_uid [N,M] int64,
                msa [S,N] int64, msa_mask [S,N] bool,
                deletion_matrix [S,N] int64, profile [N,31] float32,
                deletion_mean [N] float32,
                bond_matrix [N,N,1] float32 (if ligand bonds exist),
                template_* [T,N,...] (if templates provided).
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

        # ref_space_uid: groups atoms sharing the same reference space.
        # For polymer: uid = token_idx. For ligand: uid = first token of the ligand residue.
        ref_space_uid = np.zeros((N, M), dtype=np.int64)

        # Pre-compute ligand residue → CCD entry mapping for ref_pos lookup
        _ligand_ccd_cache: dict[str, tuple[list[str], np.ndarray]] = {}

        for i, tok in enumerate(tokens):
            mol_type_tok = int(tok["mol_type"])

            if mol_type_tok == const.MOL_NONPOLYMER:
                # Single atom token: look up CCD residue for ideal positions
                atom_idx = int(tok["atom_idx"])
                res_name = _find_res_name(data, atom_idx)
                if atom_idx < len(data.structure.atoms):
                    atom = data.structure.atoms[atom_idx]
                    atom_name = atom["name"].decode("ascii", errors="replace").strip()
                    elem_int = int(np.clip(atom["element"], 0, 127))
                    chg = float(atom["charge"])

                    # Look up this atom's ideal position from the CCD entry
                    if res_name not in _ligand_ccd_cache:
                        from deepfold3.data.feature.ref_atoms import _get_ccd_entry

                        entry = _get_ccd_entry(res_name)
                        if entry is not None:
                            _ligand_ccd_cache[res_name] = (entry.atom_names, entry.positions)
                        else:
                            _ligand_ccd_cache[res_name] = ([], np.zeros((0, 3), dtype=np.float32))

                    ccd_names, ccd_pos = _ligand_ccd_cache[res_name]
                    ref_atom_data = get_ref_atoms(
                        res_name=res_name,
                        mol_type=mol_type_tok,
                        max_atoms=M,
                        atom_names_override=[atom_name],
                    )

                    # Set ideal position from CCD if available, else from Structure atoms
                    if atom_name in ccd_names:
                        ccd_idx = ccd_names.index(atom_name)
                        ref_atom_data.positions[0] = ccd_pos[ccd_idx]
                    else:
                        # Fall back to positions stored in Structure (from CcdMolecule or mmCIF)
                        ref_atom_data.positions[0] = [
                            float(atom["x"]),
                            float(atom["y"]),
                            float(atom["z"]),
                        ]

                    ref_atom_data.element[0] = elem_int
                    ref_atom_data.charge[0] = chg
                else:
                    ref_atom_data = get_ref_atoms("UNK", mol_type_tok, M)

                # ref_space_uid for ligand: first token of the same residue
                res_info = _find_residue(data, atom_idx)
                if res_info is not None:
                    first_atom = int(res_info["atom_idx"])
                    # Find token index of first atom in this residue
                    first_tok = _find_token_by_atom_idx(tokens, first_atom)
                    ref_space_uid[i, :] = first_tok
                else:
                    ref_space_uid[i, :] = i
            else:
                atom_idx = int(tok["atom_idx"])
                atom_num = int(tok["atom_num"])
                res_name = _find_res_name(data, atom_idx)
                ref_atom_data = get_ref_atoms(res_name, mol_type_tok, M)

                for j in range(min(atom_num, M)):
                    if atom_idx + j < len(data.structure.atoms):
                        atom = data.structure.atoms[atom_idx + j]
                        if bool(atom["is_present"]):
                            ref_atom_data.element[j] = int(np.clip(atom["element"], 0, 127))
                            ref_atom_data.charge[j] = float(atom["charge"])
                            aname = atom["name"].decode("ascii", errors="replace").strip()
                            ref_atom_data.atom_name_chars[j] = _encode_atom_name(aname)

                # Polymer: ref_space_uid = token index
                ref_space_uid[i, :] = i

            ref_pos[i] = ref_atom_data.positions
            ref_mask[i] = ref_atom_data.mask
            ref_element[i] = ref_atom_data.element
            ref_charge[i] = ref_atom_data.charge
            ref_atom_name_chars[i] = ref_atom_data.atom_name_chars

        # Apply random augmentation grouped by ref_space_uid.
        # For polymer tokens each gets independent augmentation.
        # For ligand tokens all tokens sharing the same uid get the same rotation+translation.
        if random_rot:
            unique_uids = np.unique(ref_space_uid[:, 0])
            for uid in unique_uids:
                uid_mask = ref_space_uid[:, 0] == uid
                uid_indices = np.where(uid_mask)[0]

                # Collect all valid positions across tokens in this group
                all_valid_pos = []
                for idx in uid_indices:
                    valid = ref_mask[idx].astype(bool)
                    if valid.any():
                        all_valid_pos.append(ref_pos[idx][valid])

                if not all_valid_pos:
                    continue

                all_valid = np.concatenate(all_valid_pos, axis=0)
                center = all_valid.mean(axis=0)
                rot = _random_rotation_matrix(rng)
                translation = rng.standard_normal(3).astype(np.float32)

                for idx in uid_indices:
                    ref_pos[idx] = (ref_pos[idx] - center) @ rot.T + translation

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

        # ------------------------------------------------------------------
        # frames_mask: True for polymer tokens (which have backbone frames)
        # Used for pTM/ipTM computation — excludes frameless tokens (ions, ligand atoms)
        # ------------------------------------------------------------------
        mol_types = tokens["mol_type"]
        frames_mask = torch.from_numpy(
            ((mol_types == const.MOL_PROTEIN) | (mol_types == const.MOL_RNA) | (mol_types == const.MOL_DNA)).astype(
                bool
            )
        )

        # ------------------------------------------------------------------
        # token_to_rep_atom: index of representative atom per token
        # CB for protein (CA for GLY), C1' for nucleic acids, 0 for ligand
        # ------------------------------------------------------------------
        token_to_rep_atom = np.zeros(N, dtype=np.int64)
        for i, tok in enumerate(tokens):
            mt = int(tok["mol_type"])
            rt = int(tok["res_type"])
            if mt == const.MOL_PROTEIN:
                if rt == 7:  # GLY
                    token_to_rep_atom[i] = 1  # CA
                else:
                    token_to_rep_atom[i] = 4 if ref_mask[i, 4] else 1  # CB if present, else CA
            elif mt in (const.MOL_RNA, const.MOL_DNA):
                # C1' — find its index in the ref atoms
                for j in range(M):
                    if ref_mask[i, j]:
                        name_ints = ref_atom_name_chars[i, j]
                        name = "".join(chr(int(c) + 32) for c in name_ints).strip()
                        if name == "C1'":
                            token_to_rep_atom[i] = j
                            break
            # Ligand: default 0 (first/only atom)

        # ------------------------------------------------------------------
        # mol_type: molecule type per token (protein/DNA/RNA/non-polymer)
        # Used by output layer to determine entity type correctly (ligands
        # have res_type=20 same as UNK protein, so res_type alone is ambiguous).
        # ------------------------------------------------------------------
        mol_type_tensor = torch.from_numpy(tokens["mol_type"].astype(np.int64))

        # ------------------------------------------------------------------
        # token_res_name: residue name per token as [N, 5] int8
        # Encoded as raw ASCII bytes, padded with 0 (null).
        # Needed by output layer to write correct CCD codes for ligands
        # (res_type=20 maps to "UNK" but actual name may be e.g. "PCG").
        # ------------------------------------------------------------------
        token_res_name = np.zeros((N, 5), dtype=np.int8)
        for i, tok in enumerate(tokens):
            atom_idx = int(tok["atom_idx"])
            name = _find_res_name(data, atom_idx)
            name_bytes = name.encode("ascii", errors="replace")[:5]
            for j, b in enumerate(name_bytes):
                token_res_name[i, j] = b

        result = {
            "res_type": res_type,
            "residue_index": residue_index,
            "token_index": token_index,
            "asym_id": asym_id,
            "entity_id": entity_id,
            "sym_id": sym_id,
            "mol_type": mol_type_tensor,
            "token_res_name": torch.from_numpy(token_res_name),
            "seq_mask": seq_mask,
            "pair_mask": pair_mask,
            "frames_mask": frames_mask,
            "token_to_rep_atom": torch.from_numpy(token_to_rep_atom),
            "ref_pos": torch.from_numpy(ref_pos),
            "ref_mask": torch.from_numpy(ref_mask),
            "ref_element": torch.from_numpy(ref_element),
            "ref_charge": torch.from_numpy(ref_charge),
            "ref_atom_name_chars": torch.from_numpy(ref_atom_name_chars),
            "ref_space_uid": torch.from_numpy(ref_space_uid),
            "msa": msa_tensor.long(),
            "msa_mask": msa_mask_tensor,
            "deletion_matrix": del_matrix_tensor.long(),
            "profile": profile,
            "deletion_mean": deletion_mean,
        }

        # ------------------------------------------------------------------
        # Ground-truth atom positions for training loss
        # Uses actual observed coordinates from the Structure (not CCD ideal).
        # ------------------------------------------------------------------
        if training:
            atom_gt_positions = np.zeros((N, M, 3), dtype=np.float32)
            atom_gt_mask = np.zeros((N, M), dtype=bool)

            for i, tok in enumerate(tokens):
                atom_idx = int(tok["atom_idx"])
                atom_num = int(tok["atom_num"])
                mol_type_tok = int(tok["mol_type"])

                if mol_type_tok == const.MOL_NONPOLYMER:
                    # Ligand: single atom per token
                    if atom_idx < len(data.structure.atoms):
                        atom = data.structure.atoms[atom_idx]
                        if bool(atom["is_present"]):
                            atom_gt_positions[i, 0] = [float(atom["x"]), float(atom["y"]), float(atom["z"])]
                            atom_gt_mask[i, 0] = True
                else:
                    # Polymer: multiple atoms per token
                    for j in range(min(atom_num, M)):
                        a_idx = atom_idx + j
                        if a_idx < len(data.structure.atoms):
                            atom = data.structure.atoms[a_idx]
                            if bool(atom["is_present"]):
                                atom_gt_positions[i, j] = [float(atom["x"]), float(atom["y"]), float(atom["z"])]
                                atom_gt_mask[i, j] = True

            result["atom_gt_positions"] = torch.from_numpy(atom_gt_positions)
            result["atom_gt_mask"] = torch.from_numpy(atom_gt_mask)

            # Bond features for bond violation loss (Protenix-style: observed GT distances)
            bond_feats = self._build_bond_features(tokens, atom_gt_positions, atom_gt_mask, data.bonds, N, M)
            result.update(bond_feats)

        # Bond matrix: only allocated when ligand bonds exist (polymer-only → skip)
        if len(data.bonds) > 0:
            bond_matrix = torch.zeros(N, N, 1)
            t1 = data.bonds["token_1"].astype(np.intp)
            t2 = data.bonds["token_2"].astype(np.intp)
            bond_matrix[t1, t2, 0] = 1.0
            bond_matrix[t2, t1, 0] = 1.0
            result["bond_matrix"] = bond_matrix

        # ------------------------------------------------------------------
        # Template features
        # ------------------------------------------------------------------
        result.update(self._build_template_features(templates, N, tokens, template_feats_per_chain))

        # ------------------------------------------------------------------
        # Token mask channels (for loss control)
        # ------------------------------------------------------------------
        if data.token_mask is not None:
            for name, arr in data.token_mask.channels.items():
                result[f"mask_{name}"] = torch.tensor(arr)

        return result

    def _build_msa_features(
        self,
        data: Tokenized,
        tokens: np.ndarray,
        N: int,
        rng: np.random.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build [S, N] MSA tensors with paired + dense unpaired rows.

        MSA layout (following AF3 ``merging_features.py``):
        1. Paired rows (``_all_seq``): cross-chain co-evolutionary signal —
           row *i* from each chain's paired MSA placed on the same full-width
           row.  First row is the query.
        2. Dense unpaired rows: each chain's unpaired MSA is padded to the
           same number of rows (``max_S``), then all chains are placed on
           the **same** rows (each chain fills its own columns, gaps elsewhere).
           This is NOT block-diagonal — row *i* has every chain's *i*-th
           unpaired sequence simultaneously.

        Cross-chain gap positions have msa_mask=True (AF3 treats gaps as
        valid aligned positions).
        """
        # Build chain position maps
        chain_pos_map: dict[int, np.ndarray] = {}
        for asym_id in set(tokens["asym_id"]):
            chain_mask = tokens["asym_id"] == asym_id
            chain_pos_map[int(asym_id)] = np.where(chain_mask)[0]

        combined_seqs: list[np.ndarray] = []
        combined_masks: list[np.ndarray] = []
        combined_dels: list[np.ndarray] = []

        # --- 1. Paired rows: zip across chains (up to MAX_PAIRED_MSA_ROWS) ---
        # Row 0 is the query (from paired MSA or, if absent, from unpaired).
        n_paired_rows = 0
        if data.paired_msa:
            paired_counts = []
            paired_chain_ids = []
            for asym_id, msa in data.paired_msa.items():
                if int(asym_id) in chain_pos_map and len(chain_pos_map[int(asym_id)]) > 0:
                    paired_counts.append(msa.sequences.shape[0])
                    paired_chain_ids.append(int(asym_id))
            if paired_counts and min(paired_counts) > 0:
                n_paired_rows = min(min(paired_counts), const.MAX_PAIRED_MSA_ROWS + 1)
                for row_i in range(n_paired_rows):
                    row_seq = np.full(N, 21, dtype=np.int8)
                    row_mask = np.ones(N, dtype=bool)
                    row_del = np.zeros(N, dtype=np.int8)
                    for asym_id in paired_chain_ids:
                        msa = data.paired_msa[asym_id]
                        chain_pos = chain_pos_map[asym_id]
                        L = min(msa.sequences.shape[1], len(chain_pos))
                        row_seq[chain_pos[:L]] = msa.sequences[row_i, :L]
                        row_mask[chain_pos[:L]] = msa.mask[row_i, :L]
                        row_del[chain_pos[:L]] = msa.deletion_matrix[row_i, :L]
                    combined_seqs.append(row_seq[None])
                    combined_masks.append(row_mask[None])
                    combined_dels.append(row_del[None])

        # If no paired MSA, build query row from unpaired row 0
        if n_paired_rows == 0:
            query_seq = np.full(N, 21, dtype=np.int8)
            query_mask = np.ones(N, dtype=bool)
            query_del = np.zeros(N, dtype=np.int8)
            for asym_id, msa in data.unpaired_msa.items():
                chain_pos = chain_pos_map.get(int(asym_id))
                if chain_pos is None or len(chain_pos) == 0 or msa.sequences.shape[0] == 0:
                    continue
                L = min(msa.sequences.shape[1], len(chain_pos))
                query_seq[chain_pos[:L]] = msa.sequences[0, :L]
                query_mask[chain_pos[:L]] = msa.mask[0, :L]
                query_del[chain_pos[:L]] = msa.deletion_matrix[0, :L]
            combined_seqs.append(query_seq[None])
            combined_masks.append(query_mask[None])
            combined_dels.append(query_del[None])

        # --- 2. Dense unpaired rows (AF3: pad-to-max then concat along columns) ---
        # For each chain, crop unpaired rows to at most
        # (max_msa_seqs - n_paired_rows) then pad all chains to the same
        # number of rows.  Place each chain's data in its own columns and
        # gap-fill the rest.
        max_unpaired = self.max_msa_seqs - max(n_paired_rows, 1)
        # Collect per-chain unpaired MSA (skip query row = row 0)
        chain_unpaired: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]] = []
        max_S_unpaired = 0
        for asym_id, msa in data.unpaired_msa.items():
            if msa.sequences.shape[0] <= 1:
                continue
            # Rows 1.. (skip query)
            end = min(msa.sequences.shape[0], 1 + max_unpaired)
            chain_unpaired.append(
                (
                    int(asym_id),
                    msa.sequences[1:end],
                    msa.mask[1:end],
                    msa.deletion_matrix[1:end],
                )
            )
            max_S_unpaired = max(max_S_unpaired, end - 1)

        if max_S_unpaired > 0:
            # Build dense [max_S_unpaired, N] arrays — all chains on same rows
            dense_seqs = np.full((max_S_unpaired, N), 21, dtype=np.int8)
            dense_masks = np.ones((max_S_unpaired, N), dtype=bool)
            dense_dels = np.zeros((max_S_unpaired, N), dtype=np.int8)
            for asym_id, seqs, masks, dels in chain_unpaired:
                chain_pos = chain_pos_map.get(asym_id)
                if chain_pos is None or len(chain_pos) == 0:
                    continue
                S_chain = seqs.shape[0]
                L = min(seqs.shape[1], len(chain_pos))
                dense_seqs[:S_chain, chain_pos[:L]] = seqs[:, :L]
                dense_masks[:S_chain, chain_pos[:L]] = masks[:, :L]
                dense_dels[:S_chain, chain_pos[:L]] = dels[:, :L]
            combined_seqs.append(dense_seqs)
            combined_masks.append(dense_masks)
            combined_dels.append(dense_dels)

        if not combined_seqs:
            msa_arr = np.full((1, N), 21, dtype=np.int8)
            mask_arr = np.ones((1, N), dtype=bool)
            del_arr = np.zeros((1, N), dtype=np.int8)
        else:
            msa_arr = np.concatenate(combined_seqs, axis=0)
            mask_arr = np.concatenate(combined_masks, axis=0)
            del_arr = np.concatenate(combined_dels, axis=0)

        # Cap total rows at max_msa_seqs
        S = msa_arr.shape[0]
        if S > self.max_msa_seqs:
            msa_arr = msa_arr[: self.max_msa_seqs]
            mask_arr = mask_arr[: self.max_msa_seqs]
            del_arr = del_arr[: self.max_msa_seqs]

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

    @staticmethod
    def _build_bond_features(
        tokens: np.ndarray,
        atom_gt_positions: np.ndarray,  # [N, M, 3]
        atom_gt_mask: np.ndarray,  # [N, M]
        token_bonds: np.ndarray,  # TOKEN_BOND_DTYPE from Tokenized
        N: int,
        M: int,
    ) -> dict[str, torch.Tensor]:
        """Build bond pair features for bond violation loss.

        Collects three types of bonds:

        1. **Intra-residue backbone** (polymer): N-CA (atoms 0-1),
           CA-C (atoms 1-2) within each polymer token.
        2. **Inter-residue peptide/phosphodiester**: C[i]-N[i+1] for
           protein (atom 2 of token i → atom 0 of token i+1),
           O3'→P for nucleic acids.
        3. **Ligand-ligand bonds**: from ``Tokenized.bonds`` (token-level,
           atom index 0 for each single-atom ligand token).

        Uses **observed GT distances** as targets (Protenix approach), not
        CCD ideal distances. Only includes bonds where both atoms are
        resolved (``atom_gt_mask`` is True for both).

        Returns:
            Dict with keys:
                ``bond_atom_indices`` ``[B, 4]`` — (token_1, atom_1, token_2, atom_2)
                ``bond_gt_distances`` ``[B]`` — observed distances (Angstroms)
                ``bond_loss_mask``    ``[B]`` — True where both atoms are resolved
        """
        bonds: list[tuple[int, int, int, int]] = []  # (t1, a1, t2, a2)

        # --- 1. Intra-residue backbone bonds ---
        # Protein: N(0)-CA(1), CA(1)-C(2)
        # RNA/DNA: handled via CCD atom ordering, but the first 3 backbone
        # atoms (P, OP1, OP2 or similar) vary — skip nucleic intra-residue
        # for now, rely on inter-residue bonds.
        for i in range(N):
            mt = int(tokens[i]["mol_type"])
            if mt == const.MOL_PROTEIN:
                # N-CA bond (atoms 0-1)
                bonds.append((i, 0, i, 1))
                # CA-C bond (atoms 1-2)
                bonds.append((i, 1, i, 2))

        # --- 2. Inter-residue backbone bonds ---
        for i in range(N - 1):
            mt_i = int(tokens[i]["mol_type"])
            mt_j = int(tokens[i + 1]["mol_type"])
            chain_i = int(tokens[i]["asym_id"])
            chain_j = int(tokens[i + 1]["asym_id"])

            # Must be same chain and both polymer
            if chain_i != chain_j:
                continue
            if mt_i == const.MOL_NONPOLYMER or mt_j == const.MOL_NONPOLYMER:
                continue

            if mt_i == const.MOL_PROTEIN and mt_j == const.MOL_PROTEIN:
                # C[i] (atom 2) → N[i+1] (atom 0)
                bonds.append((i, 2, i + 1, 0))
            elif mt_i in (const.MOL_RNA, const.MOL_DNA) and mt_j in (const.MOL_RNA, const.MOL_DNA):
                # O3'[i] → P[i+1]: find O3' in token i, P in token i+1
                # Standard nucleic backbone: O3' is typically atom index ~8-10
                # P is typically atom 0. We use a simple heuristic: scan for
                # the atom names. If not found, skip.
                pass  # TODO: nucleic backbone bonds (need atom name lookup)

        # --- 3. Ligand-ligand bonds from tokenizer ---
        for bond in token_bonds:
            t1, t2 = int(bond["token_1"]), int(bond["token_2"])
            if 0 <= t1 < N and 0 <= t2 < N:
                # Ligand tokens have 1 atom each → atom index 0
                bonds.append((t1, 0, t2, 0))

        if not bonds:
            return {
                "bond_atom_indices": torch.zeros(0, 4, dtype=torch.long),
                "bond_gt_distances": torch.zeros(0, dtype=torch.float32),
                "bond_loss_mask": torch.zeros(0, dtype=torch.bool),
            }

        indices = np.array(bonds, dtype=np.int64)  # [B, 4]
        t1, a1, t2, a2 = indices[:, 0], indices[:, 1], indices[:, 2], indices[:, 3]

        # Compute GT distances and mask
        pos1 = atom_gt_positions[t1, a1]  # [B, 3]
        pos2 = atom_gt_positions[t2, a2]  # [B, 3]
        gt_dist = np.linalg.norm(pos1 - pos2, axis=-1)  # [B]
        mask = atom_gt_mask[t1, a1] & atom_gt_mask[t2, a2]  # [B]

        return {
            "bond_atom_indices": torch.from_numpy(indices),
            "bond_gt_distances": torch.from_numpy(gt_dist.astype(np.float32)),
            "bond_loss_mask": torch.from_numpy(mask),
        }

    def _build_template_features(
        self,
        templates: list[TemplateFeatures] | None,
        N: int,
        tokens: np.ndarray | None = None,
        template_feats_per_chain: dict[int, list[TemplateFeatures]] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Build template feature tensors from TemplateFeatures list.

        For multimer: if template_feats_per_chain is provided, each chain's
        templates populate only that chain's region of the [T, N] tensors.
        Single-chain (or single chain with templates): uses flat list as before.

        Returns dict with keys:
            template_aatype            [T, N]         int64
            template_pseudo_beta       [T, N, 3]      float32
            template_pseudo_beta_mask  [T, N]         bool
            template_backbone_frame    [T, N, 3, 3]   float32
            template_backbone_mask     [T, N]         bool
        """
        if not templates and not template_feats_per_chain:
            return {
                "template_aatype": torch.zeros(0, N, dtype=torch.int64),
                "template_pseudo_beta": torch.zeros(0, N, 3),
                "template_pseudo_beta_mask": torch.zeros(0, N, dtype=torch.bool),
                "template_backbone_frame": torch.zeros(0, N, 3, 3),
                "template_backbone_mask": torch.zeros(0, N, dtype=torch.bool),
            }

        if template_feats_per_chain and tokens is not None:
            return self._build_per_chain_template_features(template_feats_per_chain, tokens, N)

        # Single chain / flat list path (backward compatible)
        return self._build_single_chain_template_features(templates, N)

    def _build_single_chain_template_features(
        self,
        templates: list[TemplateFeatures] | None,
        N: int,
    ) -> dict[str, torch.Tensor]:
        """Build template features from a flat list (single-chain case)."""
        if not templates:
            return {
                "template_aatype": torch.zeros(0, N, dtype=torch.int64),
                "template_pseudo_beta": torch.zeros(0, N, 3),
                "template_pseudo_beta_mask": torch.zeros(0, N, dtype=torch.bool),
                "template_backbone_frame": torch.zeros(0, N, 3, 3),
                "template_backbone_mask": torch.zeros(0, N, dtype=torch.bool),
            }

        T = len(templates)
        t_aatype = np.zeros((T, N), dtype=np.int64)
        t_pb = np.zeros((T, N, 3), dtype=np.float32)
        t_pb_mask = np.zeros((T, N), dtype=bool)
        t_frame = np.zeros((T, N, 3, 3), dtype=np.float32)
        t_frame_mask = np.zeros((T, N), dtype=bool)

        for t, tmpl in enumerate(templates):
            L = min(len(tmpl.aatype), N)
            self._fill_template_slot(t_aatype[t], t_pb[t], t_pb_mask[t], t_frame[t], t_frame_mask[t], tmpl, 0, L)

        return {
            "template_aatype": torch.from_numpy(t_aatype),
            "template_pseudo_beta": torch.from_numpy(t_pb),
            "template_pseudo_beta_mask": torch.from_numpy(t_pb_mask),
            "template_backbone_frame": torch.from_numpy(t_frame),
            "template_backbone_mask": torch.from_numpy(t_frame_mask),
        }

    def _build_per_chain_template_features(
        self,
        template_feats_per_chain: dict[int, list[TemplateFeatures]],
        tokens: np.ndarray,
        N: int,
    ) -> dict[str, torch.Tensor]:
        """Build template features with per-chain template population.

        Each chain's templates fill only that chain's region of the [T, N] tensors.
        """
        T = min(self.max_templates, max((len(v) for v in template_feats_per_chain.values()), default=0))
        if T == 0:
            return {
                "template_aatype": torch.zeros((0, N), dtype=torch.long),
                "template_pseudo_beta": torch.zeros((0, N, 3)),
                "template_pseudo_beta_mask": torch.zeros((0, N), dtype=torch.bool),
                "template_backbone_frame": torch.zeros((0, N, 3, 3)),
                "template_backbone_mask": torch.zeros((0, N), dtype=torch.bool),
            }
        t_aatype = np.zeros((T, N), dtype=np.int64)
        t_pb = np.zeros((T, N, 3), dtype=np.float32)
        t_pb_mask = np.zeros((T, N), dtype=bool)
        t_frame = np.zeros((T, N, 3, 3), dtype=np.float32)
        t_frame_mask = np.zeros((T, N), dtype=bool)

        for asym_id, chain_templates in template_feats_per_chain.items():
            chain_mask = tokens["asym_id"] == asym_id
            chain_pos = np.where(chain_mask)[0]
            if len(chain_pos) == 0:
                continue

            for t, tmpl in enumerate(chain_templates[:T]):
                L = min(len(tmpl.aatype), len(chain_pos))
                start = int(chain_pos[0])
                self._fill_template_slot(
                    t_aatype[t], t_pb[t], t_pb_mask[t], t_frame[t], t_frame_mask[t], tmpl, start, L
                )

        return {
            "template_aatype": torch.from_numpy(t_aatype),
            "template_pseudo_beta": torch.from_numpy(t_pb),
            "template_pseudo_beta_mask": torch.from_numpy(t_pb_mask),
            "template_backbone_frame": torch.from_numpy(t_frame),
            "template_backbone_mask": torch.from_numpy(t_frame_mask),
        }

    @staticmethod
    def _fill_template_slot(
        aatype_row: np.ndarray,
        pb_row: np.ndarray,
        pb_mask_row: np.ndarray,
        frame_row: np.ndarray,
        frame_mask_row: np.ndarray,
        tmpl: TemplateFeatures,
        start: int,
        length: int,
    ) -> None:
        """Fill a slice of template arrays from a single TemplateFeatures."""
        end = start + length
        aatype_row[start:end] = tmpl.aatype[:length]
        pos = tmpl.atom_positions[:length]  # [L, M, 3]
        mask = tmpl.atom_mask[:length]  # [L, M]

        # Pseudo-beta: CB (index 3) for all, CA (index 1) for GLY (type 7)
        is_gly = tmpl.aatype[:length] == 7
        cb_pos = pos[:, 3, :]
        ca_pos = pos[:, 1, :]
        pb_row[start:end] = np.where(is_gly[:, None], ca_pos, cb_pos)
        pb_mask_row[start:end] = np.where(is_gly, mask[:, 1], mask[:, 3])

        # Backbone frames from N(0), CA(1), C(2) via Gram-Schmidt
        n_pos = pos[:, 0, :]
        ca = pos[:, 1, :]
        c_pos = pos[:, 2, :]

        fm = mask[:, 0] & mask[:, 1] & mask[:, 2]
        frame_mask_row[start:end] = fm

        eps = 1e-8
        v1 = c_pos - ca
        v2 = n_pos - ca
        v1_norm = np.linalg.norm(v1, axis=-1, keepdims=True) + eps
        e1 = v1 / v1_norm
        proj = np.sum(v2 * e1, axis=-1, keepdims=True) * e1
        u2 = v2 - proj
        u2_norm = np.linalg.norm(u2, axis=-1, keepdims=True) + eps
        e2 = u2 / u2_norm
        e3 = np.cross(e1, e2)
        frame_row[start:end] = np.stack([e1, e2, e3], axis=1)

    def _empty_features(self) -> dict[str, torch.Tensor]:
        M = self.max_atoms
        return {
            "res_type": torch.zeros(0, dtype=torch.int64),
            "residue_index": torch.zeros(0, dtype=torch.int64),
            "token_index": torch.zeros(0, dtype=torch.int64),
            "asym_id": torch.zeros(0, dtype=torch.int64),
            "entity_id": torch.zeros(0, dtype=torch.int64),
            "sym_id": torch.zeros(0, dtype=torch.int64),
            "mol_type": torch.zeros(0, dtype=torch.int64),
            "token_res_name": torch.zeros(0, 5, dtype=torch.int8),
            "seq_mask": torch.zeros(0, dtype=torch.bool),
            "pair_mask": torch.zeros(0, 0, dtype=torch.bool),
            "frames_mask": torch.zeros(0, dtype=torch.bool),
            "token_to_rep_atom": torch.zeros(0, dtype=torch.int64),
            "ref_pos": torch.zeros(0, M, 3),
            "ref_mask": torch.zeros(0, M, dtype=torch.bool),
            "ref_element": torch.zeros(0, M, dtype=torch.int64),
            "ref_charge": torch.zeros(0, M),
            "ref_atom_name_chars": torch.zeros(0, M, 4, dtype=torch.int64),
            "ref_space_uid": torch.zeros(0, M, dtype=torch.int64),
            "msa": torch.zeros(1, 0, dtype=torch.int64),
            "msa_mask": torch.zeros(1, 0, dtype=torch.bool),
            "deletion_matrix": torch.zeros(1, 0, dtype=torch.int64),
            "profile": torch.zeros(0, const.RES_TYPE_COUNT),
            "deletion_mean": torch.zeros(0),
            "template_aatype": torch.zeros(0, 0, dtype=torch.int64),
            "template_pseudo_beta": torch.zeros(0, 0, 3),
            "template_pseudo_beta_mask": torch.zeros(0, 0, dtype=torch.bool),
            "template_backbone_frame": torch.zeros(0, 0, 3, 3),
            "template_backbone_mask": torch.zeros(0, 0, dtype=torch.bool),
        }


def _find_res_name(data: Tokenized, atom_idx: int) -> str:
    """Find residue name from atom index by searching residues array."""
    for res in data.structure.residues:
        start = int(res["atom_idx"])
        end = start + int(res["atom_num"])
        if start <= atom_idx < end:
            return res["name"].decode("ascii", errors="replace").strip()
    return "UNK"


def _find_residue(data: Tokenized, atom_idx: int) -> np.void | None:
    """Find the residue record containing the given atom index."""
    for res in data.structure.residues:
        start = int(res["atom_idx"])
        end = start + int(res["atom_num"])
        if start <= atom_idx < end:
            return res
    return None


def _find_token_by_atom_idx(tokens: np.ndarray, atom_idx: int) -> int:
    """Find the token index that starts at the given atom index."""
    matches = np.where(tokens["atom_idx"] == atom_idx)[0]
    if len(matches) > 0:
        return int(matches[0])
    return 0
