"""mmCIF structure parser.

Uses gemmi to read mmCIF files and converts to DeepFold3 Structure dataclass.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import gemmi
except ImportError as e:
    raise ImportError("gemmi is required for mmCIF parsing: pip install gemmi") from e

from deepfold3.data import const
from deepfold3.data.types import (
    ATOM_DTYPE,
    BOND_DTYPE,
    CHAIN_DTYPE,
    RESIDUE_DTYPE,
    Structure,
)


def _element_to_int(element_name: str) -> int:
    """Convert element symbol string to atomic number, clamped to [0, 127]."""
    sym = element_name.strip().upper()
    num = const.ELEMENT_TO_INT.get(sym, 0)
    return int(np.clip(num, 0, 127))


def _mol_type_from_gemmi(entity: gemmi.Entity) -> int | None:
    """Map gemmi entity type to DeepFold3 mol type. Returns None if should be skipped."""
    if entity.entity_type == gemmi.EntityType.Polymer:
        pt = entity.polymer_type
        if pt in (gemmi.PolymerType.PeptideL, gemmi.PolymerType.PeptideD):
            return const.MOL_PROTEIN
        elif pt in (gemmi.PolymerType.Dna, gemmi.PolymerType.DnaRnaHybrid):
            return const.MOL_DNA
        elif pt in (gemmi.PolymerType.Rna,):
            return const.MOL_RNA
        else:
            # Unknown polymer: treat as protein
            return const.MOL_PROTEIN
    elif entity.entity_type in (gemmi.EntityType.NonPolymer, gemmi.EntityType.Branched):
        return const.MOL_NONPOLYMER
    elif entity.entity_type == gemmi.EntityType.Water:
        return None  # skip water
    return None


def _get_res_type(res_name: str, mol_type: int) -> int:
    """Get residue type index for a given residue name and molecule type."""
    if mol_type == const.MOL_NONPOLYMER:
        return 20  # UNK
    return const.res_name_to_type(res_name)


def parse_mmcif(path: Path) -> Structure:
    """Parse an mmCIF file into a DeepFold3 Structure.

    Parameters
    ----------
    path : Path to .cif or .cif.gz file

    Returns
    -------
    Structure with atoms, bonds, residues, chains arrays
    """
    structure = gemmi.read_structure(str(path))
    # Use first model
    model = structure[0]

    # Build entity → mol_type mapping
    entity_mol_type: dict[str, int] = {}
    for entity in structure.entities:
        mt = _mol_type_from_gemmi(entity)
        if mt is not None:
            entity_mol_type[entity.name] = mt

    # Determine entity_id by sequence deduplication
    # entity name → entity_id (integer)
    entity_name_to_id: dict[str, int] = {}
    entity_id_counter = 1  # 1-based (mmCIF convention)

    # Also track entity sequences for deduplication
    entity_seq_to_id: dict[str, int] = {}

    def get_entity_id(entity_name: str, seq_key: str) -> int:
        nonlocal entity_id_counter
        if seq_key not in entity_seq_to_id:
            entity_seq_to_id[seq_key] = entity_id_counter
            entity_id_counter += 1
        entity_name_to_id[entity_name] = entity_seq_to_id[seq_key]
        return entity_seq_to_id[seq_key]

    # First pass: assign entity IDs by residue-name sequence
    for chain in model:
        subchain = chain.subchains()
        for sc_label, _residues in subchain.items():
            # Get entity for this subchain
            entity_name = structure.find_chain(sc_label).entity_name if False else None
            # Use chain entity name
            pass  # deferred to second pass

    # Collect chains, residues, atoms, bonds
    all_atoms: list[tuple] = []
    all_bonds: list[tuple] = []
    all_residues: list[tuple] = []
    all_chains: list[tuple] = []

    atom_idx_global = 0
    res_idx_global = 0
    asym_id = 0

    # Track entity copy counts for sym_id
    entity_sym_counter: dict[int, int] = {}

    # atom global index map: (chain_idx, gemmi atom address) → global idx
    # We'll use position-based indexing
    atom_global_map: dict[int, int] = {}  # gemmi atom serial → global idx

    for chain in model:
        # Process each subchain (= unique chain instance in mmCIF)
        for sc_label in chain.subchains():
            subchain_residues = chain.get_subchain(sc_label)
            if not subchain_residues:
                continue

            # Find entity
            try:
                entity_name = model.find_chain(sc_label).entity_name
            except Exception:
                entity_name = chain.name

            if entity_name not in entity_mol_type:
                continue  # skip water and unknown

            mol_type = entity_mol_type[entity_name]

            # Build sequence key for entity deduplication
            res_names = []
            for res in subchain_residues:
                if res.entity_type == gemmi.EntityType.Water:
                    continue
                res_names.append(res.name)
            seq_key = entity_name + "|" + ",".join(res_names)
            entity_id = get_entity_id(entity_name, seq_key)

            sym_id = entity_sym_counter.get(entity_id, 0)
            entity_sym_counter[entity_id] = sym_id + 1

            chain_atom_start = atom_idx_global
            chain_res_start = res_idx_global

            for res in subchain_residues:
                if res.entity_type == gemmi.EntityType.Water:
                    continue

                res_name = res.name[:5]  # truncate to 5 chars
                res_type = _get_res_type(res.name, mol_type)
                try:
                    res_id = res.seqid.num
                except Exception:
                    res_id = res_idx_global

                is_standard = mol_type != const.MOL_NONPOLYMER and res_type < 20
                res_atom_start = atom_idx_global
                res_atom_count = 0

                for atom in res:
                    # Skip hydrogens / deuterium
                    if atom.element in (gemmi.Element("H"), gemmi.Element("D")):
                        continue

                    pos = atom.pos
                    elem_name = atom.element.name
                    element = _element_to_int(elem_name)
                    charge = float(atom.charge) if hasattr(atom, "charge") else 0.0
                    atom_name = atom.name[:4].ljust(4)
                    is_present = True  # gemmi only gives resolved atoms

                    all_atoms.append(
                        (
                            float(pos.x),
                            float(pos.y),
                            float(pos.z),
                            element,
                            charge,
                            atom_name.encode("ascii"),
                            is_present,
                        )
                    )
                    atom_global_map[atom_idx_global] = atom_idx_global
                    atom_idx_global += 1
                    res_atom_count += 1

                all_residues.append(
                    (
                        res_name.encode("ascii"),
                        res_type,
                        mol_type,
                        int(res_id),
                        res_atom_start,
                        res_atom_count,
                        is_standard,
                        res_atom_count > 0,  # is_present
                    )
                )
                res_idx_global += 1

            chain_atom_count = atom_idx_global - chain_atom_start
            chain_res_count = res_idx_global - chain_res_start

            if chain_res_count == 0:
                continue

            all_chains.append(
                (
                    sc_label[:4].encode("ascii"),
                    mol_type,
                    entity_id,
                    sym_id,
                    asym_id,
                    chain_res_start,
                    chain_res_count,
                    chain_atom_start,
                    chain_atom_count,
                )
            )
            asym_id += 1

    # Parse inter-residue bonds from structure connections
    for connection in structure.connections:
        try:
            a1 = model.find_cra(connection.partner1)
            a2 = model.find_cra(connection.partner2)
            if a1.atom is None or a2.atom is None:
                continue
            # Map to global indices by position lookup
            # We don't have a reliable gemmi → global map without serial numbers,
            # so skip bond parsing from connections for now (bonds within residues
            # can be inferred from CCD during featurization)
        except Exception:
            continue

    # Convert to numpy arrays
    atoms_arr = np.array(all_atoms, dtype=ATOM_DTYPE) if all_atoms else np.empty(0, dtype=ATOM_DTYPE)
    bonds_arr = np.array(all_bonds, dtype=BOND_DTYPE) if all_bonds else np.empty(0, dtype=BOND_DTYPE)
    residues_arr = np.array(all_residues, dtype=RESIDUE_DTYPE) if all_residues else np.empty(0, dtype=RESIDUE_DTYPE)
    chains_arr = np.array(all_chains, dtype=CHAIN_DTYPE) if all_chains else np.empty(0, dtype=CHAIN_DTYPE)

    return Structure(
        atoms=atoms_arr,
        bonds=bonds_arr,
        residues=residues_arr,
        chains=chains_arr,
    )
