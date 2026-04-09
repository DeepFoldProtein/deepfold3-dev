"""Tokenizer: maps a parsed Input structure to a Tokenized representation.

Rules (AF3 / Boltz):
  - Polymer chains (protein, RNA, DNA): 1 token per residue
  - Non-polymer chains (ligands): 1 token per heavy atom
"""

from __future__ import annotations

import numpy as np

from deepfold3.data import const
from deepfold3.data.types import (
    TOKEN_BOND_DTYPE,
    TOKEN_DTYPE,
    Input,
    Tokenized,
)


class Tokenizer:
    """Convert an Input (structure + MSA) to a Tokenized object."""

    def tokenize(self, inp: Input) -> Tokenized:
        """Tokenize the input.

        Parameters
        ----------
        inp : Input

        Returns
        -------
        Tokenized
        """
        structure = inp.structure
        tokens_list: list[tuple] = []
        token_bonds_list: list[tuple] = []

        token_idx = 0

        for chain in structure.chains:
            mol_type = int(chain["mol_type"])
            asym_id = int(chain["asym_id"])
            entity_id = int(chain["entity_id"])
            sym_id = int(chain["sym_id"])

            res_start = int(chain["res_idx"])
            res_end = res_start + int(chain["res_num"])

            if mol_type in (const.MOL_PROTEIN, const.MOL_RNA, const.MOL_DNA):
                # One token per residue
                for ri in range(res_start, res_end):
                    res = structure.residues[ri]
                    res_type = int(res["res_type"])
                    res_id = int(res["res_id"])
                    atom_idx = int(res["atom_idx"])
                    atom_num = int(res["atom_num"])

                    # Center coordinates: use first atom or CA/C1' if present
                    cx, cy, cz = _get_center_coords(
                        structure.atoms,
                        atom_idx,
                        atom_num,
                        res["name"].decode("ascii", errors="replace").strip(),
                        mol_type,
                    )

                    tokens_list.append(
                        (
                            token_idx,  # token_idx
                            res_id,  # residue_index (PDB number)
                            res_type,  # res_type
                            mol_type,  # mol_type
                            asym_id,  # asym_id
                            entity_id,  # entity_id
                            sym_id,  # sym_id
                            atom_idx,  # atom_idx
                            atom_num,  # atom_num
                            cx,
                            cy,
                            cz,  # center coords
                        )
                    )

                    # NOTE: backbone bonds are NOT included in the bond matrix.
                    # AF3 restricts the bond matrix to polymer-ligand and
                    # ligand-ligand covalent bonds only.  Backbone connectivity
                    # is already captured by the sequence/relative-position encoding.

                    token_idx += 1

            elif mol_type == const.MOL_NONPOLYMER:
                # One token per heavy atom
                first_token_in_ligand = token_idx
                ligand_atom_start = int(chain["atom_idx"])

                for ri in range(res_start, res_end):
                    res = structure.residues[ri]
                    atom_idx_start = int(res["atom_idx"])
                    atom_num = int(res["atom_num"])

                    for ai in range(atom_num):
                        global_ai = atom_idx_start + ai
                        atom = structure.atoms[global_ai]
                        cx = float(atom["x"])
                        cy = float(atom["y"])
                        cz = float(atom["z"])
                        local_atom_idx = global_ai - ligand_atom_start

                        tokens_list.append(
                            (
                                token_idx,
                                local_atom_idx,  # residue_index = atom index within ligand
                                20,  # res_type = UNK
                                mol_type,
                                asym_id,
                                entity_id,
                                sym_id,
                                global_ai,  # atom_idx = global atom index
                                1,  # atom_num = 1 (each token = 1 atom)
                                cx,
                                cy,
                                cz,
                            )
                        )
                        token_idx += 1

                # Ligand intra-bonds: for each bond where both atoms are in this ligand
                # atom global idx → token idx
                atom_to_token = {
                    int(chain["atom_idx"]) + i: first_token_in_ligand + i for i in range(int(chain["atom_num"]))
                }
                for bond in structure.bonds:
                    a1 = int(bond["atom_1"])
                    a2 = int(bond["atom_2"])
                    if a1 in atom_to_token and a2 in atom_to_token:
                        t1 = atom_to_token[a1]
                        t2 = atom_to_token[a2]
                        if t1 != t2:
                            token_bonds_list.append((t1, t2))

        tokens_arr = np.array(tokens_list, dtype=TOKEN_DTYPE) if tokens_list else np.empty(0, dtype=TOKEN_DTYPE)
        bonds_arr = (
            np.array(token_bonds_list, dtype=TOKEN_BOND_DTYPE)
            if token_bonds_list
            else np.empty(0, dtype=TOKEN_BOND_DTYPE)
        )

        return Tokenized(
            tokens=tokens_arr,
            bonds=bonds_arr,
            structure=structure,
            paired_msa=inp.paired_msa,
            unpaired_msa=inp.unpaired_msa,
            token_mask=inp.token_mask,
        )


def _get_center_coords(
    atoms: np.ndarray,
    atom_idx: int,
    atom_num: int,
    res_name: str,
    mol_type: int,
) -> tuple[float, float, float]:
    """Get center atom coordinates for a residue.

    Falls back to first atom if named center atom is not found.
    """
    if atom_num == 0:
        return 0.0, 0.0, 0.0

    center_atom_name = const.RES_CENTER_ATOM.get(res_name)

    if center_atom_name is not None:
        for i in range(atom_num):
            name = atoms[atom_idx + i]["name"].decode("ascii", errors="replace").strip()
            if name == center_atom_name:
                a = atoms[atom_idx + i]
                return float(a["x"]), float(a["y"]), float(a["z"])

    # Fallback: first atom
    a = atoms[atom_idx]
    return float(a["x"]), float(a["y"]), float(a["z"])
