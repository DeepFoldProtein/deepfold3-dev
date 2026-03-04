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
        max_msa_seqs: int = 16384,
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
            bond_matrix       [N, N, 1]   float32  (only when ligand bonds exist)
            template_aatype            [T, N]         int64    (if templates provided)
            template_pseudo_beta       [T, N, 3]      float32
            template_pseudo_beta_mask  [T, N]         bool
            template_backbone_frame    [T, N, 3, 3]   float32
            template_backbone_mask     [T, N]         bool
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

        result = {
            "res_type": res_type,
            "residue_index": residue_index,
            "token_index": token_index,
            "asym_id": asym_id,
            "entity_id": entity_id,
            "sym_id": sym_id,
            "seq_mask": seq_mask,
            "pair_mask": pair_mask,
            "frames_mask": frames_mask,
            "token_to_rep_atom": torch.from_numpy(token_to_rep_atom),
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

        return result

    def _build_msa_features(
        self,
        data: Tokenized,
        tokens: np.ndarray,
        N: int,
        rng: np.random.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build [S, N] MSA tensors with paired + unpaired rows.

        MSA layout (following AF3):
        1. Query row (row 0): merged from all chains' query sequences
        2. Paired rows: cross-chain co-evolutionary signal — row i from each
           chain's paired MSA placed on the same full-width row
        3. Unpaired block-diagonal rows: each chain's unpaired MSA occupies
           its own block, gaps elsewhere

        Cross-chain gap positions have msa_mask=True (AF3 treats gaps as valid
        aligned positions).
        """
        # Build chain position maps
        chain_pos_map: dict[int, np.ndarray] = {}
        for asym_id in set(tokens["asym_id"]):
            chain_mask = tokens["asym_id"] == asym_id
            chain_pos_map[int(asym_id)] = np.where(chain_mask)[0]

        # Helper: place a chain's MSA into full-width [S, N] arrays
        def _place_chain_msa(msa, asym_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            chain_pos = chain_pos_map.get(asym_id)
            if chain_pos is None or len(chain_pos) == 0:
                return np.empty((0, N), np.int8), np.empty((0, N), bool), np.empty((0, N), np.int8)
            S_msa, N_msa = msa.sequences.shape
            seqs = np.full((S_msa, N), 21, dtype=np.int8)  # gap for cross-chain
            masks = np.ones((S_msa, N), dtype=bool)  # AF3: all positions valid, even cross-chain gaps
            dels = np.zeros((S_msa, N), dtype=np.int8)
            L = min(N_msa, len(chain_pos))
            seqs[:, chain_pos[:L]] = msa.sequences[:, :L]
            masks[:, chain_pos[:L]] = msa.mask[:, :L]
            dels[:, chain_pos[:L]] = msa.deletion_matrix[:, :L]
            return seqs, masks, dels

        combined_seqs: list[np.ndarray] = []
        combined_masks: list[np.ndarray] = []
        combined_dels: list[np.ndarray] = []

        # --- 1. Query row: merge row 0 from all chains' unpaired MSAs ---
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

        # --- 2. Paired rows: zip across chains ---
        if data.paired_msa:
            # Determine number of paired rows (min across chains, excluding query row)
            paired_counts = []
            paired_chain_ids = []
            for asym_id, msa in data.paired_msa.items():
                if int(asym_id) in chain_pos_map and len(chain_pos_map[int(asym_id)]) > 0:
                    # Skip query row (row 0) — paired rows start from row 1
                    n_paired = max(0, msa.sequences.shape[0] - 1)
                    paired_counts.append(n_paired)
                    paired_chain_ids.append(int(asym_id))
            if paired_counts and min(paired_counts) > 0:
                n_paired_rows = min(paired_counts)
                for row_i in range(n_paired_rows):
                    row_seq = np.full(N, 21, dtype=np.int8)
                    row_mask = np.ones(N, dtype=bool)  # cross-chain gaps are valid
                    row_del = np.zeros(N, dtype=np.int8)
                    for asym_id in paired_chain_ids:
                        msa = data.paired_msa[asym_id]
                        chain_pos = chain_pos_map[asym_id]
                        L = min(msa.sequences.shape[1], len(chain_pos))
                        # row_i+1 because row 0 is query
                        row_seq[chain_pos[:L]] = msa.sequences[row_i + 1, :L]
                        row_mask[chain_pos[:L]] = msa.mask[row_i + 1, :L]
                        row_del[chain_pos[:L]] = msa.deletion_matrix[row_i + 1, :L]
                    combined_seqs.append(row_seq[None])
                    combined_masks.append(row_mask[None])
                    combined_dels.append(row_del[None])

        # --- 3. Unpaired block-diagonal rows ---
        for asym_id, msa in data.unpaired_msa.items():
            if msa.sequences.shape[0] <= 1:
                continue
            # Skip query row (already added above)
            seqs, masks, dels = _place_chain_msa(msa, int(asym_id))
            if seqs.shape[0] > 1:
                combined_seqs.append(seqs[1:])
                combined_masks.append(masks[1:])
                combined_dels.append(dels[1:])

        if not combined_seqs:
            msa_arr = np.full((1, N), 21, dtype=np.int8)
            mask_arr = np.ones((1, N), dtype=bool)
            del_arr = np.zeros((1, N), dtype=np.int8)
        else:
            msa_arr = np.concatenate(combined_seqs, axis=0)
            mask_arr = np.concatenate(combined_masks, axis=0)
            del_arr = np.concatenate(combined_dels, axis=0)

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
        T = min(self.max_templates, max(len(v) for v in template_feats_per_chain.values()))
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
            "seq_mask": torch.zeros(0, dtype=torch.bool),
            "pair_mask": torch.zeros(0, 0, dtype=torch.bool),
            "frames_mask": torch.zeros(0, dtype=torch.bool),
            "token_to_rep_atom": torch.zeros(0, dtype=torch.int64),
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
