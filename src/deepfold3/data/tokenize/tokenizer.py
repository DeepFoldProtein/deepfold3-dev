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


def _find_modified_residues(structure) -> set[int]:
    """Residue indices that are covalently modified (→ atomize per-atom).

    Mirrors OF3 ``tokenization.find_modified_residue_atom_ids`` (AF3 §2.6): a
    canonical/standard polymer residue is treated as *modified* — and therefore
    tokenized per-atom instead of as a single residue token — when it has a
    covalent bond to a *different* residue that is NOT the standard polymer
    backbone link (peptide C–N for protein, phosphodiester P / O3' / O5' for
    nucleic acids).

    Restricted to bonds whose partner is a NON-POLYMER (ligand / glycan): a
    standard residue is atomized when covalently attached to a ligand (e.g. an
    N-glycosylated ASN bonded to a NAG), but a polymer–polymer non-backbone link
    (disulfide, isopeptide crosslink) atomizes neither residue. OF3 reaches the
    same result by stripping such polymer–polymer links in ``prefilter_bonds``
    *before* this check (``cleanup_structure_of3``); requiring a non-polymer
    partner here makes the outcome identical whether or not the stored bonds were
    prefiltered — so existing (un-re-prepared) NPZs, whose ``_struct_conn``
    disulfides are still present, are NOT suddenly atomized at load time. The
    peptide / phosphodiester backbone-atom exemption is kept for exact OF3 parity.

    Returns:
        Set of indices into ``structure.residues``.
    """
    residues = structure.residues
    atoms = structure.atoms
    bonds = structure.bonds
    if len(bonds) == 0 or len(residues) == 0:
        return set()

    n_atoms = len(atoms)
    atom_res = np.full(n_atoms, -1, dtype=np.int64)
    for ri, res in enumerate(residues):
        s = int(res["atom_idx"])
        c = int(res["atom_num"])
        atom_res[s : s + c] = ri

    # Per-atom polymer flag (from the owning chain's mol_type) so a modification
    # bond can be required to have exactly one non-polymer (ligand) endpoint.
    atom_is_polymer = np.zeros(n_atoms, dtype=bool)
    for ch in structure.chains:
        s = int(ch["atom_idx"])
        c = int(ch["atom_num"])
        if int(ch["mol_type"]) in (const.MOL_PROTEIN, const.MOL_RNA, const.MOL_DNA):
            atom_is_polymer[s : s + c] = True

    modified: set[int] = set()
    for bond in bonds:
        a1 = int(bond["atom_1"])
        a2 = int(bond["atom_2"])
        if not (0 <= a1 < n_atoms and 0 <= a2 < n_atoms):
            continue
        r1 = int(atom_res[a1])
        r2 = int(atom_res[a2])
        if r1 < 0 or r2 < 0 or r1 == r2:
            continue  # intra-residue (or unmapped) bond → not a modification
        n1 = atoms[a1]["name"].decode("ascii", errors="replace").strip()
        n2 = atoms[a2]["name"].decode("ascii", errors="replace").strip()
        is_peptide = n1 in const.PEPTIDE_BOND_ATOMS and n2 in const.PEPTIDE_BOND_ATOMS
        is_phospho = n1 in const.PHOSPHODIESTER_BOND_ATOMS and n2 in const.PHOSPHODIESTER_BOND_ATOMS
        if is_peptide or is_phospho:
            continue  # standard backbone link → not a modification
        # Atomize only the standard polymer residue attached to a NON-POLYMER
        # partner. Polymer–polymer (disulfide/crosslink) and ligand–ligand bonds
        # do not mark a standard residue as modified.
        p1 = bool(atom_is_polymer[a1])
        p2 = bool(atom_is_polymer[a2])
        if p1 and not p2:
            modified.add(r1)
        elif p2 and not p1:
            modified.add(r2)
    return modified


def _token_row(**fields: object) -> tuple:
    """Build a ``TOKEN_DTYPE`` row from named fields, ordered by the dtype.

    Name-based (matching :mod:`deepfold3.data.tokenize.atomize`) so a positional
    tuple can never silently misplace values when a ``TOKEN_DTYPE`` field is
    added or reordered — a missing field raises ``KeyError`` instead. Every field
    must be supplied.
    """
    return tuple(fields[name] for name in TOKEN_DTYPE.names)


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
        # Per-token MSA column index within the chain.  Standard polymer
        # residues increment sequentially; expanded non-standard residues
        # share the same column.  Ligand tokens get -1.
        msa_col_list: list[int] = []

        token_idx = 0
        # Global atom index → (token_idx, local_atom_within_token)
        atom_to_token: dict[int, tuple[int, int]] = {}

        # AF3 §2.6: standard polymer residues that are covalently modified
        # (non-backbone inter-residue bond, e.g. glycosylation) are atomized
        # per-atom rather than kept as a single residue token.
        modified_res = _find_modified_residues(structure)

        for chain in structure.chains:
            mol_type = int(chain["mol_type"])
            asym_id = int(chain["asym_id"])
            entity_id = int(chain["entity_id"])
            sym_id = int(chain["sym_id"])

            res_start = int(chain["res_idx"])
            res_end = res_start + int(chain["res_num"])

            # Per-chain token ordinal (reset each chain). Symmetric copies of an
            # entity are tokenized identically, so token k of every copy gets the
            # same value — a cross-copy key that survives cropping (see
            # ``sym_token_index`` in TOKEN_DTYPE). Increments wherever token_idx does.
            chain_tok_ord = 0

            if mol_type in (const.MOL_PROTEIN, const.MOL_RNA, const.MOL_DNA):
                chain_msa_col = 0  # per-chain MSA column counter
                for ri in range(res_start, res_end):
                    res = structure.residues[ri]
                    res_type = int(res["res_type"])
                    res_id = int(res["res_id"])
                    atom_idx = int(res["atom_idx"])
                    atom_num = int(res["atom_num"])
                    is_standard = bool(res["is_standard"])
                    # Standard nucleotides (res_type 22-30) are always a single
                    # token, regardless of a stored ``is_standard`` that may be
                    # stale from a prepare run whose ``res_type < 20`` rule wrongly
                    # atomised them — so existing NPZs are fixed without a
                    # re-prepare. (New prepares store the corrected flag.)
                    if 22 <= res_type <= 30:
                        is_standard = True

                    # AF3 §2.6: a covalently-modified standard residue is
                    # atomized (routed to the per-atom path below), even though
                    # its name/res_type is canonical.
                    if ri in modified_res:
                        is_standard = False

                    if is_standard:
                        # Standard residue: one token per residue
                        cx, cy, cz = _get_center_coords(
                            structure.atoms,
                            atom_idx,
                            atom_num,
                            res["name"].decode("ascii", errors="replace").strip(),
                            mol_type,
                        )

                        tokens_list.append(
                            _token_row(
                                token_idx=token_idx,
                                residue_index=res_id,
                                res_type=res_type,
                                mol_type=mol_type,
                                asym_id=asym_id,
                                entity_id=entity_id,
                                sym_id=sym_id,
                                sym_token_index=chain_tok_ord,
                                atom_idx=atom_idx,
                                atom_num=atom_num,
                                center_x=cx,
                                center_y=cy,
                                center_z=cz,
                                atom_in_res=0,  # whole-residue token
                            )
                        )
                        msa_col_list.append(chain_msa_col)

                        for ai in range(atom_num):
                            atom_to_token[atom_idx + ai] = (token_idx, ai)

                        token_idx += 1
                        chain_tok_ord += 1
                        chain_msa_col += 1
                    else:
                        # AF3 §2.6: modified (non-standard) residue — one token per atom.
                        # All expanded atoms share the same MSA column.
                        for ai in range(atom_num):
                            global_ai = atom_idx + ai
                            atom = structure.atoms[global_ai]
                            cx = float(atom["x"])
                            cy = float(atom["y"])
                            cz = float(atom["z"])

                            tokens_list.append(
                                _token_row(
                                    token_idx=token_idx,
                                    residue_index=res_id,
                                    res_type=20,  # UNK for per-atom tokens
                                    mol_type=mol_type,
                                    asym_id=asym_id,
                                    entity_id=entity_id,
                                    sym_id=sym_id,
                                    sym_token_index=chain_tok_ord,
                                    atom_idx=global_ai,
                                    atom_num=1,
                                    center_x=cx,
                                    center_y=cy,
                                    center_z=cz,
                                    atom_in_res=ai,  # local atom index within residue
                                )
                            )
                            msa_col_list.append(chain_msa_col)
                            atom_to_token[global_ai] = (token_idx, 0)
                            token_idx += 1
                            chain_tok_ord += 1
                        chain_msa_col += 1

            elif mol_type == const.MOL_NONPOLYMER:
                # One token per heavy atom
                ligand_atom_start = int(chain["atom_idx"])

                for ri in range(res_start, res_end):
                    res = structure.residues[ri]
                    atom_idx_start = int(res["atom_idx"])
                    atom_num = int(res["atom_num"])
                    res_id = int(res["res_id"])

                    for ai in range(atom_num):
                        global_ai = atom_idx_start + ai
                        atom = structure.atoms[global_ai]
                        cx = float(atom["x"])
                        cy = float(atom["y"])
                        cz = float(atom["z"])
                        local_atom_idx = global_ai - ligand_atom_start

                        tokens_list.append(
                            _token_row(
                                token_idx=token_idx,
                                residue_index=res_id,  # shared across the ligand's atoms
                                res_type=20,  # UNK
                                mol_type=mol_type,
                                asym_id=asym_id,
                                entity_id=entity_id,
                                sym_id=sym_id,
                                sym_token_index=chain_tok_ord,
                                atom_idx=global_ai,  # global atom index
                                atom_num=1,  # each token = 1 atom
                                center_x=cx,
                                center_y=cy,
                                center_z=cz,
                                atom_in_res=local_atom_idx,  # CCD-local atom index (keys ligand symmetry)
                            )
                        )
                        msa_col_list.append(-1)  # ligands have no MSA column
                        atom_to_token[global_ai] = (token_idx, 0)
                        token_idx += 1
                        chain_tok_ord += 1

        # Build token bonds from structure bonds (cross-entity + ligand intra-bonds)
        seen_bonds: set[tuple[int, int]] = set()
        for bond in structure.bonds:
            a1, a2 = int(bond["atom_1"]), int(bond["atom_2"])
            loc1 = atom_to_token.get(a1)
            loc2 = atom_to_token.get(a2)
            if loc1 is None or loc2 is None:
                continue
            t1, la1 = loc1
            t2, la2 = loc2
            if t1 == t2:
                continue  # skip intra-token bonds
            bond_key = (min(t1, t2), max(t1, t2))
            if bond_key in seen_bonds:
                continue  # deduplicate
            seen_bonds.add(bond_key)
            token_bonds_list.append((t1, t2, la1, la2))

        tokens_arr = np.array(tokens_list, dtype=TOKEN_DTYPE) if tokens_list else np.empty(0, dtype=TOKEN_DTYPE)
        bonds_arr = (
            np.array(token_bonds_list, dtype=TOKEN_BOND_DTYPE)
            if token_bonds_list
            else np.empty(0, dtype=TOKEN_BOND_DTYPE)
        )

        msa_col_arr = np.array(msa_col_list, dtype=np.int32) if msa_col_list else np.empty(0, dtype=np.int32)

        return Tokenized(
            tokens=tokens_arr,
            bonds=bonds_arr,
            structure=structure,
            paired_msa=inp.paired_msa,
            unpaired_msa=inp.unpaired_msa,
            token_mask=inp.token_mask,
            msa_col_idx=msa_col_arr,
        )


def _get_center_coords(
    atoms: np.ndarray,
    atom_idx: int,
    atom_num: int,
    res_name: str,
    mol_type: int,
) -> tuple[float, float, float]:
    """Get center atom coordinates for a residue.

    Only *present* (resolved) atoms are considered: a materialised unresolved
    atom (``is_present=False``, coords 0) must never be chosen as the token
    center, or a residue whose CA/C1' is unresolved would get a spurious
    (0, 0, 0) center. Falls back to the first present atom, then to the origin
    for a fully-unresolved residue.
    """
    if atom_num == 0:
        return 0.0, 0.0, 0.0

    center_atom_name = const.RES_CENTER_ATOM.get(res_name)

    if center_atom_name is not None:
        for i in range(atom_num):
            a = atoms[atom_idx + i]
            if not bool(a["is_present"]):
                continue
            name = a["name"].decode("ascii", errors="replace").strip()
            if name == center_atom_name:
                return float(a["x"]), float(a["y"]), float(a["z"])

    # Fallback: first present atom
    for i in range(atom_num):
        a = atoms[atom_idx + i]
        if bool(a["is_present"]):
            return float(a["x"]), float(a["y"]), float(a["z"])

    return 0.0, 0.0, 0.0
