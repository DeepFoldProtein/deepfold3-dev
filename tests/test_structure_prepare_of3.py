"""OF3-parity structure-cleanup steps in the mmCIF→NPZ prepare pipeline.

Covers, on small synthetic Structures (no gemmi / CIF needed):
  * Item 7 — arginine NH1/NH2 canonicalisation (``fix_arginine_naming``)
  * Item 6 — bond prefilter (``prefilter_bonds``)
  * Item 5 — terminal-atom removal (``finalize_standard_residue_atoms``)
  * Item 4a — unresolved intra-residue atom materialisation (same fn)
  * Item 3 — modified-residue atomization (tokenizer ``_find_modified_residues``)
"""

from __future__ import annotations

import numpy as np

from deepfold3.data import const
from deepfold3.data.prepare.structure import (
    finalize_standard_residue_atoms,
    fix_arginine_naming,
    prefilter_bonds,
)
from deepfold3.data.tokenize.tokenizer import Tokenizer, _find_modified_residues
from deepfold3.data.types import (
    ATOM_DTYPE,
    BOND_DTYPE,
    CHAIN_DTYPE,
    RESIDUE_DTYPE,
    Input,
    Structure,
)

# --- synthetic-structure builder -------------------------------------------

# residue spec: (name, res_type, res_id, is_standard, [(atom_name, x, y, z, present), ...])
# chain spec:   (mol_type, entity_id, sym_id, [residue spec, ...])


def _build(chain_specs) -> Structure:
    atoms_rows: list[tuple] = []
    res_rows: list[tuple] = []
    chain_rows: list[tuple] = []

    for asym_id, (mol_type, entity_id, sym_id, residues) in enumerate(chain_specs):
        chain_atom_start = len(atoms_rows)
        chain_res_start = len(res_rows)
        for name, res_type, res_id, is_standard, atom_list in residues:
            res_atom_start = len(atoms_rows)
            n_present = 0
            for aname, x, y, z, present in atom_list:
                atoms_rows.append(
                    (
                        float(x),
                        float(y),
                        float(z),
                        const.element_name_to_int(aname.strip()[0]) if aname.strip() else 0,
                        0.0,
                        aname[:4].ljust(4).encode("ascii"),
                        bool(present),
                        0.0,
                    )
                )
                n_present += int(bool(present))
            res_rows.append(
                (
                    name[:5].encode("ascii"),
                    int(res_type),
                    int(mol_type),
                    int(res_id),
                    res_atom_start,
                    len(atoms_rows) - res_atom_start,
                    bool(is_standard),
                    n_present > 0,
                )
            )
        chain_rows.append(
            (
                f"C{asym_id}"[:4].encode("ascii"),
                int(mol_type),
                int(entity_id),
                int(sym_id),
                int(asym_id),
                chain_res_start,
                len(res_rows) - chain_res_start,
                chain_atom_start,
                len(atoms_rows) - chain_atom_start,
            )
        )

    return Structure(
        atoms=np.array(atoms_rows, dtype=ATOM_DTYPE) if atoms_rows else np.empty(0, dtype=ATOM_DTYPE),
        bonds=np.empty(0, dtype=BOND_DTYPE),
        residues=np.array(res_rows, dtype=RESIDUE_DTYPE) if res_rows else np.empty(0, dtype=RESIDUE_DTYPE),
        chains=np.array(chain_rows, dtype=CHAIN_DTYPE) if chain_rows else np.empty(0, dtype=CHAIN_DTYPE),
    )


def _names(atoms, s, n) -> list[str]:
    return [atoms[i]["name"].decode("ascii").strip() for i in range(s, s + n)]


def _coord(atoms, s, n, name):
    for i in range(s, s + n):
        if atoms[i]["name"].decode("ascii").strip() == name:
            return np.array([atoms[i]["x"], atoms[i]["y"], atoms[i]["z"]])
    raise KeyError(name)


# --- Item 7: arginine NH1/NH2 canonicalisation -----------------------------


def _arg_structure(nh1_xyz, nh2_xyz):
    atoms = [
        ("CD", 0.0, 0.0, 0.0, True),
        ("NE", 1.0, 0.0, 0.0, True),
        ("CZ", 2.0, 0.0, 0.0, True),
        ("NH1", *nh1_xyz, True),
        ("NH2", *nh2_xyz, True),
    ]
    return _build([(const.MOL_PROTEIN, 1, 0, [("ARG", 1, 5, True, atoms)])])


def test_arg_naming_swaps_when_nh2_closer_to_cd():
    # NH1 far (10 Å), NH2 near (3 Å) → must swap so NH1 becomes the near one.
    st = _arg_structure(nh1_xyz=(10.0, 0.0, 0.0), nh2_xyz=(3.0, 0.0, 0.0))
    fix_arginine_naming(st.atoms, st.residues)
    cd = np.array([0.0, 0.0, 0.0])
    d_nh1 = np.linalg.norm(_coord(st.atoms, 0, 5, "NH1") - cd)
    d_nh2 = np.linalg.norm(_coord(st.atoms, 0, 5, "NH2") - cd)
    assert d_nh1 < d_nh2
    assert d_nh1 == 3.0  # the (3,0,0) atom now carries the NH1 label


def test_arg_naming_no_swap_when_already_correct():
    st = _arg_structure(nh1_xyz=(3.0, 0.0, 0.0), nh2_xyz=(10.0, 0.0, 0.0))
    fix_arginine_naming(st.atoms, st.residues)
    assert _coord(st.atoms, 0, 5, "NH1")[0] == 3.0
    assert _coord(st.atoms, 0, 5, "NH2")[0] == 10.0


def test_arg_naming_skips_when_atoms_missing():
    # Only NH1 + CD present → cannot decide, leave untouched.
    atoms = [("CD", 0.0, 0.0, 0.0, True), ("NH1", 9.0, 0.0, 0.0, True)]
    st = _build([(const.MOL_PROTEIN, 1, 0, [("ARG", 1, 5, True, atoms)])])
    fix_arginine_naming(st.atoms, st.residues)
    assert _names(st.atoms, 0, 2) == ["CD", "NH1"]


# --- Item 6: bond prefilter ------------------------------------------------


def _two_chain_bond_structure():
    # chain 0: protein res 1 (CA) + res 3 (CA); chain 1: ligand ZN (single atom).
    st = _build(
        [
            (
                const.MOL_PROTEIN,
                1,
                0,
                [
                    ("CYS", 4, 1, True, [("CA", 0.0, 0.0, 0.0, True), ("SG", 1.0, 0.0, 0.0, True)]),
                    ("CYS", 4, 3, True, [("CA", 5.0, 0.0, 0.0, True), ("SG", 2.0, 0.0, 0.0, True)]),
                ],
            ),
            (const.MOL_NONPOLYMER, 2, 0, [("ZN", 20, 1, False, [("ZN", 1.5, 1.0, 0.0, True)])]),
        ]
    )
    # atom indices: 0=CA(r1) 1=SG(r1) 2=CA(r3) 3=SG(r3) 4=ZN
    return st


def _bonds(pairs):
    return np.array([(a, b, 1) for a, b in pairs], dtype=BOND_DTYPE)


def test_prefilter_removes_long_bond():
    st = _two_chain_bond_structure()
    # SG(r1)@(1,0,0) to CA(r3)@(5,0,0) = 4 Å > 2.4 → removed.
    bonds = _bonds([(1, 2)])
    out = prefilter_bonds(st.atoms, st.residues, st.chains, bonds)
    assert len(out) == 0


def test_prefilter_removes_intra_chain_nonconsecutive_disulfide():
    st = _two_chain_bond_structure()
    # SG(r1)@(1,0,0) to SG(r3)@(2,0,0): 1 Å (short) but res 1↔3 non-consecutive → removed.
    bonds = _bonds([(1, 3)])
    out = prefilter_bonds(st.atoms, st.residues, st.chains, bonds)
    assert len(out) == 0


def test_prefilter_keeps_polymer_ligand_bond():
    st = _two_chain_bond_structure()
    # SG(r1)@(1,0,0) to ZN@(1.5,1,0): ~1.12 Å, protein↔ligand (not poly-poly) → kept.
    bonds = _bonds([(1, 4)])
    out = prefilter_bonds(st.atoms, st.residues, st.chains, bonds)
    assert len(out) == 1


def test_prefilter_removes_inter_chain_dative():
    st = _two_chain_bond_structure()
    bonds = _bonds([(1, 4)])  # short protein↔ligand bond
    is_dative = np.array([True], dtype=bool)
    out = prefilter_bonds(st.atoms, st.residues, st.chains, bonds, is_dative=is_dative)
    assert len(out) == 0  # dative + inter-chain → removed even though short/ligand


def test_prefilter_removes_inter_chain_poly_link():
    # two protein chains, short cross-chain disulfide.
    st = _build(
        [
            (const.MOL_PROTEIN, 1, 0, [("CYS", 4, 1, True, [("SG", 0.0, 0.0, 0.0, True)])]),
            (const.MOL_PROTEIN, 1, 1, [("CYS", 4, 1, True, [("SG", 1.0, 0.0, 0.0, True)])]),
        ]
    )
    out = prefilter_bonds(st.atoms, st.residues, st.chains, _bonds([(0, 1)]))
    assert len(out) == 0


def test_prefilter_keeps_consecutive_intra_chain_bond():
    # short bond between consecutive residues in one chain (|Δres|==1) is kept.
    st = _build(
        [
            (
                const.MOL_PROTEIN,
                1,
                0,
                [
                    ("ALA", 0, 1, True, [("C", 0.0, 0.0, 0.0, True)]),
                    ("ALA", 0, 2, True, [("N", 1.3, 0.0, 0.0, True)]),
                ],
            )
        ]
    )
    out = prefilter_bonds(st.atoms, st.residues, st.chains, _bonds([(0, 1)]))
    assert len(out) == 1


# --- Item 5: terminal-atom removal -----------------------------------------


def test_terminal_oxt_stripped_from_standard_protein():
    atoms = [(nm, 0.0, 0.0, 0.0, True) for nm in ["N", "CA", "C", "O", "CB", "OXT"]]
    st = _build([(const.MOL_PROTEIN, 1, 0, [("ALA", 0, 1, True, atoms)])])
    a, r, c, b = finalize_standard_residue_atoms(st.atoms, st.residues, st.chains, st.bonds)
    names = _names(a, 0, int(r[0]["atom_num"]))
    assert "OXT" not in names
    assert set(names) == set(const.REF_ATOMS["ALA"])  # exactly the standard set


def test_terminal_op3_stripped_from_standard_nucleic():
    # RNA "A": include OP3 (5' leaving oxygen) → stripped.
    ref = const.REF_ATOMS["A"]
    atoms = [(nm, 0.0, 0.0, 0.0, True) for nm in ref] + [("OP3", 0.0, 0.0, 0.0, True)]
    st = _build([(const.MOL_RNA, 1, 0, [("A", 22, 1, True, atoms)])])
    a, r, c, b = finalize_standard_residue_atoms(st.atoms, st.residues, st.chains, st.bonds)
    names = _names(a, 0, int(r[0]["atom_num"]))
    assert "OP3" not in names


def test_terminal_atoms_kept_for_nonstandard_residue():
    # SEP is not a canonical standard name → passed through verbatim (keeps OXT).
    atoms = [(nm, 0.0, 0.0, 0.0, True) for nm in ["N", "CA", "C", "O", "CB", "OG", "P", "OXT"]]
    st = _build([(const.MOL_PROTEIN, 1, 0, [("SEP", 15, 1, True, atoms)])])
    a, r, c, b = finalize_standard_residue_atoms(st.atoms, st.residues, st.chains, st.bonds)
    names = _names(a, 0, int(r[0]["atom_num"]))
    assert "OXT" in names and "P" in names  # untouched


# --- Item 4a: unresolved intra-residue atom materialisation ----------------


def test_missing_atoms_materialised_as_unresolved():
    # ALA with only N, CA, C present → O, CB materialised (is_present=False).
    atoms = [(nm, 1.0, 2.0, 3.0, True) for nm in ["N", "CA", "C"]]
    st = _build([(const.MOL_PROTEIN, 1, 0, [("ALA", 0, 1, True, atoms)])])
    a, r, c, b = finalize_standard_residue_atoms(st.atoms, st.residues, st.chains, st.bonds)
    n = int(r[0]["atom_num"])
    names = _names(a, 0, n)
    assert set(names) == set(const.REF_ATOMS["ALA"])  # full reference set
    present = {a[i]["name"].decode().strip(): bool(a[i]["is_present"]) for i in range(n)}
    assert present["N"] and present["CA"] and present["C"]
    assert not present["O"] and not present["CB"]  # materialised → masked
    # materialised atoms carry sensible element + zero coords
    for i in range(n):
        if not a[i]["is_present"]:
            assert (a[i]["x"], a[i]["y"], a[i]["z"]) == (0.0, 0.0, 0.0)
    assert bool(r[0]["is_present"])  # residue still counts as present


def test_materialisation_remaps_bonds_and_preserves_present_coords():
    # Two residues; a covalent bond between present atoms must survive w/ remap.
    st = _build(
        [
            (
                const.MOL_PROTEIN,
                1,
                0,
                [
                    ("ALA", 0, 1, True, [("N", 0.0, 0.0, 0.0, True), ("CA", 1.0, 0.0, 0.0, True)]),
                    ("CYS", 4, 2, True, [("CA", 2.0, 0.0, 0.0, True), ("SG", 3.0, 0.0, 0.0, True)]),
                ],
            )
        ]
    )
    # bond CA(res1, idx1) — SG(res2, idx3) before materialisation.
    bonds = _bonds([(1, 3)])
    a, r, c, b = finalize_standard_residue_atoms(st.atoms, st.residues, st.chains, bonds)
    assert len(b) == 1
    # resolve remapped endpoints by name and check they still point at CA/SG.
    n1 = a[int(b[0]["atom_1"])]["name"].decode().strip()
    n2 = a[int(b[0]["atom_2"])]["name"].decode().strip()
    assert {n1, n2} == {"CA", "SG"}


# --- Item 3: modified-residue atomization ----------------------------------


def _asn_glycosylation_structure():
    # protein ASN (res 1) + ligand NAG (res, own chain); bond ASN.ND2 — NAG.C1.
    st = _build(
        [
            (
                const.MOL_PROTEIN,
                1,
                0,
                [
                    ("ALA", 0, 1, True, [("N", 0.0, 0, 0, True), ("CA", 1.0, 0, 0, True), ("C", 2.0, 0, 0, True)]),
                    (
                        "ASN",
                        2,
                        2,
                        True,
                        [
                            ("N", 3.0, 0, 0, True),
                            ("CA", 4.0, 0, 0, True),
                            ("C", 5.0, 0, 0, True),
                            ("ND2", 6.0, 0, 0, True),
                        ],
                    ),
                ],
            ),
            (const.MOL_NONPOLYMER, 2, 0, [("NAG", 20, 1, False, [("C1", 6.5, 0, 0, True)])]),
        ]
    )
    return st


def test_find_modified_residue_via_ligand_bond():
    st = _asn_glycosylation_structure()
    # ND2 is global atom idx 6; NAG C1 is idx 7.
    st = Structure(atoms=st.atoms, bonds=_bonds([(6, 7)]), residues=st.residues, chains=st.chains)
    modified = _find_modified_residues(st)
    assert 1 in modified  # ASN residue index (0=ALA, 1=ASN)
    assert 0 not in modified  # ALA untouched


def test_peptide_bond_not_flagged_as_modification():
    st = _asn_glycosylation_structure()
    # C(res0)=idx2 to N(res1)=idx3: both peptide-bond atoms → NOT a modification.
    st = Structure(atoms=st.atoms, bonds=_bonds([(2, 3)]), residues=st.residues, chains=st.chains)
    assert _find_modified_residues(st) == set()


def test_tokenizer_atomizes_modified_residue():
    st = _asn_glycosylation_structure()
    st = Structure(atoms=st.atoms, bonds=_bonds([(6, 7)]), residues=st.residues, chains=st.chains)
    tok = Tokenizer().tokenize(Input(structure=st, paired_msa={}, unpaired_msa={}))
    # ALA → 1 token; ASN atomized → 4 per-atom tokens; NAG ligand → 1 token.
    asym = tok.tokens["asym_id"]
    res_idx = tok.tokens["residue_index"]
    # ASN is res_id 2 on chain 0 → 4 tokens (per-atom).
    asn_tokens = tok.tokens[(asym == 0) & (res_idx == 2)]
    assert len(asn_tokens) == 4
    assert all(t["atom_num"] == 1 for t in asn_tokens)
    # ALA (res_id 1) stays a single residue token.
    ala_tokens = tok.tokens[(asym == 0) & (res_idx == 1)]
    assert len(ala_tokens) == 1


def test_tokenizer_keeps_unmodified_residue_single_token():
    st = _asn_glycosylation_structure()
    # No bonds → ASN not modified → single token.
    st = Structure(atoms=st.atoms, bonds=np.empty(0, dtype=BOND_DTYPE), residues=st.residues, chains=st.chains)
    tok = Tokenizer().tokenize(Input(structure=st, paired_msa={}, unpaired_msa={}))
    asym = tok.tokens["asym_id"]
    res_idx = tok.tokens["residue_index"]
    asn_tokens = tok.tokens[(asym == 0) & (res_idx == 2)]
    assert len(asn_tokens) == 1


def _two_cys_structure():
    # one protein chain, two CYS residues, each N/CA/C/CB/SG.
    return _build(
        [
            (
                const.MOL_PROTEIN,
                1,
                0,
                [
                    (
                        "CYS", 4, 1, True,
                        [("N", 0, 0, 0, True), ("CA", 1, 0, 0, True), ("C", 2, 0, 0, True),
                         ("CB", 3, 0, 0, True), ("SG", 4, 0, 0, True)],
                    ),
                    (
                        "CYS", 4, 2, True,
                        [("N", 0, 5, 0, True), ("CA", 1, 5, 0, True), ("C", 2, 5, 0, True),
                         ("CB", 3, 5, 0, True), ("SG", 4, 5, 0, True)],
                    ),
                ],
            ),
        ]
    )


def test_disulfide_does_not_atomize_cysteines():
    # SG(res0)=idx4, SG(res1)=idx9: a polymer-polymer (disulfide) non-backbone
    # bond must NOT mark either CYS as modified (OF3 strips it in prefilter;
    # DF3 requires a non-polymer partner, so un-re-prepared NPZs still holding
    # _struct_conn disulfides are not suddenly atomized).
    st = _two_cys_structure()
    st = Structure(atoms=st.atoms, bonds=_bonds([(4, 9)]), residues=st.residues, chains=st.chains)
    assert _find_modified_residues(st) == set()
    tok = Tokenizer().tokenize(Input(structure=st, paired_msa={}, unpaired_msa={}))
    # Both CYS stay single residue tokens (not per-atom).
    assert len(tok.tokens) == 2
    assert all(t["atom_num"] > 1 for t in tok.tokens)
