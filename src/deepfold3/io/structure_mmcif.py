"""Convert Structure dataclass to mmCIF format.

Provides the inverse of parse/mmcif.py: Structure → mmCIF string.
Supports both experimental structures and predicted structures (with pLDDT
stored as B-factor).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import gemmi
except ImportError as e:
    raise ImportError("gemmi is required for mmCIF output: pip install gemmi") from e

from deepfold3.data import const
from deepfold3.data.const import THREE_LETTER_TO_ONE
from deepfold3.data.types import Structure
from deepfold3.io.mmcif import (
    _INT_TO_ELEMENT,
    _add_entity_descriptions,
    _build_header,
    _build_plddt_metric_table,
    _enrich_cif_text,
    _fix_seq_one_letter_code,
)

# mol_type -> (gemmi EntityType, gemmi PolymerType)
_MOL_TYPE_TO_GEMMI: dict[int, tuple[gemmi.EntityType, gemmi.PolymerType]] = {
    const.MOL_PROTEIN: (gemmi.EntityType.Polymer, gemmi.PolymerType.PeptideL),
    const.MOL_DNA: (gemmi.EntityType.Polymer, gemmi.PolymerType.Dna),
    const.MOL_RNA: (gemmi.EntityType.Polymer, gemmi.PolymerType.Rna),
    const.MOL_NONPOLYMER: (gemmi.EntityType.NonPolymer, gemmi.PolymerType.Unknown),
}


def structure_to_mmcif(
    structure: Structure,
    entry_id: str | None = None,
    entity_descriptions: dict[str, str] | None = None,
    title: str | None = None,
) -> str:
    """Convert a Structure to an mmCIF string.

    Parameters
    ----------
    structure : Structure with atoms, residues, chains arrays.
    entry_id  : Entry identifier for the mmCIF header. Defaults to structure.name.
    entity_descriptions : Optional mapping from chain name to description
                          string, written as _entity.pdbx_description.
    title     : Optional structure title, written as _struct.title.

    Returns
    -------
    mmCIF-formatted string.
    """
    if entry_id is None:
        entry_id = structure.name if structure.name != "unset" else "deepfold3"

    st = gemmi.Structure()
    st.name = entry_id
    st.cell = gemmi.UnitCell(1.0, 1.0, 1.0, 90.0, 90.0, 90.0)
    st.spacegroup_hm = "P 1"

    model = gemmi.Model("1")

    atoms = structure.atoms
    residues = structure.residues
    chains = structure.chains

    for ci in range(len(chains)):
        chain_rec = chains[ci]
        chain_name = chain_rec["name"].decode("ascii").strip()
        chain = gemmi.Chain(chain_name)

        res_start = int(chain_rec["res_idx"])
        res_end = res_start + int(chain_rec["res_num"])

        for ri in range(res_start, res_end):
            res_rec = residues[ri]
            res_name = res_rec["name"].decode("ascii").strip()
            res_id = int(res_rec["res_id"])

            residue = gemmi.Residue()
            residue.name = res_name
            residue.seqid = gemmi.SeqId(str(res_id))

            mol_type = int(chain_rec["mol_type"])
            entity_type, _ = _MOL_TYPE_TO_GEMMI.get(mol_type, (gemmi.EntityType.NonPolymer, gemmi.PolymerType.Unknown))
            residue.entity_type = entity_type

            atom_start = int(res_rec["atom_idx"])
            atom_end = atom_start + int(res_rec["atom_num"])

            for ai in range(atom_start, atom_end):
                a = atoms[ai]
                if not bool(a["is_present"]):
                    continue

                atom_name = a["name"].decode("ascii").strip()
                if not atom_name:
                    continue

                elem_num = int(a["element"])
                elem_sym = _INT_TO_ELEMENT.get(elem_num, "C")

                atom = gemmi.Atom()
                atom.name = atom_name
                atom.pos = gemmi.Position(float(a["x"]), float(a["y"]), float(a["z"]))
                atom.occ = 1.0
                atom.b_iso = float(a["b_factor"])
                try:
                    atom.element = gemmi.Element(elem_sym)
                except Exception:
                    atom.element = gemmi.Element("C")

                residue.add_atom(atom)

            if len(residue) > 0:
                chain.add_residue(residue)

        if len(chain) > 0:
            model.add_chain(chain)

    st.add_model(model)

    # Setup entities and fix subchains
    st.setup_entities()

    for i, entity in enumerate(st.entities, start=1):
        entity.name = str(i)

    # Set subchains to chain names (single pass)
    subchain_remap: dict[str, str] = {}
    for mdl in st:
        for chain in mdl:
            for res in chain:
                old_sub = res.subchain
                if old_sub and old_sub not in subchain_remap:
                    subchain_remap[old_sub] = chain.name
                res.subchain = chain.name

    for entity in st.entities:
        entity.subchains = [subchain_remap.get(s, s) for s in entity.subchains]

    # Set entity.full_sequence for _entity_poly_seq table
    chain_sequences: dict[str, list[str]] = {}
    for chain in st[0]:
        chain_sequences[chain.name] = [res.name for res in chain]

    for entity in st.entities:
        if entity.entity_type == gemmi.EntityType.Polymer and entity.subchains:
            chain_name = entity.subchains[0]
            seq = chain_sequences.get(chain_name, [])
            entity.full_sequence = seq

    st.assign_label_seq_id()

    # Generate base mmCIF from gemmi
    doc = st.make_mmcif_document()
    base_text = doc.as_string()

    # Fix _chem_comp.type values
    base_text = _enrich_cif_text(base_text)

    # Fix _entity_poly.pdbx_seq_one_letter_code
    entity_sequences: dict[str, str] = {}
    for entity in st.entities:
        if entity.entity_type == gemmi.EntityType.Polymer and entity.subchains:
            chain_name = entity.subchains[0]
            seq = chain_sequences.get(chain_name, [])
            one_letter = "".join(THREE_LETTER_TO_ONE.get(r, "X") for r in seq)
            entity_sequences[entity.name] = one_letter
    base_text = _fix_seq_one_letter_code(base_text, entity_sequences)

    # Add entity descriptions
    if entity_descriptions:
        ent_desc_by_id: dict[str, str] = {}
        for entity in st.entities:
            for sub in entity.subchains:
                if sub in entity_descriptions:
                    ent_desc_by_id[entity.name] = entity_descriptions[sub]
                    break
        base_text = _add_entity_descriptions(base_text, ent_desc_by_id)

    # Insert header
    header = _build_header(entry_id, title=title)
    lines = base_text.split("\n", 1)
    final_text = lines[0] + "\n" + header + lines[1]

    # Append pLDDT metric table if b_factors are non-zero (predicted structure)
    has_plddt = bool(np.any(atoms["b_factor"] > 0)) if len(atoms) > 0 else False
    if has_plddt:
        plddt_per_residue: dict[tuple[str, str, int], float] = {}
        for ci in range(len(chains)):
            chain_rec = chains[ci]
            chain_name = chain_rec["name"].decode("ascii").strip()
            res_start = int(chain_rec["res_idx"])
            res_end = res_start + int(chain_rec["res_num"])
            for ri in range(res_start, res_end):
                res_rec = residues[ri]
                res_name = res_rec["name"].decode("ascii").strip()
                res_id = int(res_rec["res_id"])
                atom_start = int(res_rec["atom_idx"])
                atom_end = atom_start + int(res_rec["atom_num"])
                res_atoms = atoms[atom_start:atom_end]
                present = res_atoms["is_present"]
                if np.any(present):
                    avg_plddt = float(np.mean(res_atoms["b_factor"][present]))
                else:
                    avg_plddt = 0.0
                plddt_per_residue[(chain_name, res_name, res_id)] = avg_plddt
        final_text += _build_plddt_metric_table(plddt_per_residue)

    return final_text


def save_structure_mmcif(
    path: str | Path,
    structure: Structure,
    entry_id: str | None = None,
    entity_descriptions: dict[str, str] | None = None,
    title: str | None = None,
) -> None:
    """Write a Structure to an mmCIF file.

    Parameters
    ----------
    path      : Output file path (.cif).
    structure : Structure to write.
    entry_id  : Optional entry identifier.
    entity_descriptions : Optional mapping from chain name to description.
    title     : Optional structure title, written as _struct.title.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        structure_to_mmcif(structure, entry_id=entry_id, entity_descriptions=entity_descriptions, title=title)
    )
