"""Write predicted structures to mmCIF format.

Each diffusion sample becomes a separate MODEL block, matching PDB
multi-model convention (NMR ensembles, AF3 samples).

pLDDT per atom is stored in the B-factor column and also as
_ma_qa_metric_local per-residue confidence scores.
"""

from __future__ import annotations

from pathlib import Path

import torch

try:
    import gemmi
except ImportError as e:
    raise ImportError("gemmi is required for mmCIF output: pip install gemmi") from e

from deepfold3.data.const import ELEMENT_TO_INT, MOL_NONPOLYMER, RES_TYPE_INDEX_TO_NAME, THREE_LETTER_TO_ONE
from deepfold3.io.confidences import compute_plddt
from deepfold3.model.model import DeepFold3Output

# Reverse element map: atomic number -> symbol
_INT_TO_ELEMENT: dict[int, str] = {v: k for k, v in ELEMENT_TO_INT.items()}
_INT_TO_ELEMENT[0] = "X"  # unknown / padding

# chem_comp type for standard residues
_CHEM_COMP_TYPE: dict[str, str] = {
    "ALA": "L-peptide linking",
    "ARG": "L-peptide linking",
    "ASN": "L-peptide linking",
    "ASP": "L-peptide linking",
    "CYS": "L-peptide linking",
    "GLN": "L-peptide linking",
    "GLU": "L-peptide linking",
    "GLY": "peptide linking",
    "HIS": "L-peptide linking",
    "ILE": "L-peptide linking",
    "LEU": "L-peptide linking",
    "LYS": "L-peptide linking",
    "MET": "L-peptide linking",
    "PHE": "L-peptide linking",
    "PRO": "L-peptide linking",
    "SER": "L-peptide linking",
    "THR": "L-peptide linking",
    "TRP": "L-peptide linking",
    "TYR": "L-peptide linking",
    "VAL": "L-peptide linking",
    "UNK": "L-peptide linking",
    "A": "RNA linking",
    "G": "RNA linking",
    "C": "RNA linking",
    "U": "RNA linking",
    "DA": "DNA linking",
    "DG": "DNA linking",
    "DC": "DNA linking",
    "DT": "DNA linking",
}


def asym_id_to_chain_name(asym_id: int) -> str:
    """Map 0-based asym_id to a PDB chain name (A, B, ..., Z, AA, ...)."""
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    n = len(letters)
    if asym_id < n:
        return letters[asym_id]
    return letters[asym_id // n - 1] + letters[asym_id % n]


from deepfold3.data.feature.ref_atoms import decode_atom_name_chars as _decode_atom_name  # noqa: E402


def _cif_quote(value: str) -> str:
    """Quote a CIF value if it contains spaces or special characters."""
    if " " in value or "'" in value or value.startswith("_") or value.startswith("#"):
        if "'" not in value:
            return f"'{value}'"
        if '"' not in value:
            return f'"{value}"'
        return f"\n;{value}\n;"
    return value


def _build_header(entry_id: str, title: str | None = None) -> str:
    """Build standard mmCIF header categories."""
    title_str = _cif_quote(title) if title else "'DeepFold3 predicted structure'"
    entry_id = _cif_quote(entry_id)
    return f"""\
#
_audit_conform.dict_name       mmcif_pdbx.dic
_audit_conform.dict_version    5.398
_audit_conform.dict_location   http://mmcif.pdb.org/dictionaries/ascii/mmcif_pdbx.dic
#
_struct.entry_id                  {entry_id}
_struct.title                     {title_str}
_struct.pdbx_model_details        ?
_struct.pdbx_model_type_details   ?
#
_software.pdbx_ordinal      1
_software.name              DeepFold3
_software.classification    'model building'
_software.version           ?
#
"""


def _build_plddt_metric_table(
    plddt_per_residue: dict[tuple[str, str, int], float],
) -> str:
    """Build _ma_qa_metric + _ma_qa_metric_local tables for per-residue pLDDT."""
    lines = [
        "#",
        "_ma_qa_metric.id                 1",
        "_ma_qa_metric.name               pLDDT",
        "_ma_qa_metric.description        'Predicted per-residue confidence score (0-100)'",
        "_ma_qa_metric.type               pLDDT",
        "_ma_qa_metric.mode               local",
        "_ma_qa_metric.software_group_id  1",
        "#",
        "loop_",
        "_ma_qa_metric_local.ordinal_id",
        "_ma_qa_metric_local.model_id",
        "_ma_qa_metric_local.label_asym_id",
        "_ma_qa_metric_local.label_comp_id",
        "_ma_qa_metric_local.label_seq_id",
        "_ma_qa_metric_local.metric_id",
        "_ma_qa_metric_local.metric_value",
    ]
    ordinal = 1
    for (chain_name, res_name, seq_id), plddt_val in plddt_per_residue.items():
        lines.append(f"{ordinal} 1 {chain_name} {res_name} {seq_id} 1 {plddt_val:.2f}")
        ordinal += 1
    lines.append("#")
    return "\n".join(lines) + "\n"


def _enrich_cif_text(text: str) -> str:
    """Fix _chem_comp.type values in the mmCIF text."""
    lines = text.split("\n")
    in_chem_comp = False
    result = []
    for line in lines:
        if "_chem_comp.type" in line and "loop_" not in line:
            in_chem_comp = True
            result.append(line)
            continue
        if in_chem_comp and line.strip() and not line.startswith("_") and not line.startswith("loop_"):
            parts = line.split()
            if len(parts) >= 2:
                comp_id = parts[0]
                comp_type = _CHEM_COMP_TYPE.get(comp_id, "non-polymer")
                result.append(f"{comp_id} '{comp_type}'")
                continue
            else:
                in_chem_comp = False
        elif in_chem_comp and (not line.strip() or line.startswith("loop_") or line.startswith("#")):
            in_chem_comp = False
        result.append(line)
    return "\n".join(result)


def _add_entity_descriptions(text: str, entity_descriptions: dict[str, str]) -> str:
    """Insert _entity.pdbx_description into the _entity loop block.

    Args:
        text: mmCIF text.
        entity_descriptions: Mapping from entity id (str) to description text.

    Returns:
        Updated mmCIF text with pdbx_description column added.
    """
    if not entity_descriptions:
        return text

    lines = text.split("\n")
    result: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # Detect start of _entity loop
        if line.strip() == "loop_" and i + 1 < len(lines) and lines[i + 1].strip() == "_entity.id":
            result.append(line)
            i += 1
            # Collect column headers
            col_names: list[str] = []
            while i < len(lines) and lines[i].strip().startswith("_entity."):
                col_names.append(lines[i].strip())
                result.append(lines[i])
                i += 1
            # Add description column header
            result.append("_entity.pdbx_description")
            id_col = next((j for j, c in enumerate(col_names) if c == "_entity.id"), 0)
            # Process data rows until blank line, loop_, or #
            while (
                i < len(lines)
                and lines[i].strip()
                and not lines[i].strip().startswith("loop_")
                and not lines[i].strip().startswith("#")
            ):
                parts = lines[i].split()
                if len(parts) > id_col:
                    entity_id = parts[id_col]
                    desc = entity_descriptions.get(entity_id)
                    if desc:
                        result.append(lines[i] + " " + _cif_quote(desc))
                    else:
                        result.append(lines[i] + " ?")
                else:
                    result.append(lines[i] + " ?")
                i += 1
        else:
            result.append(line)
            i += 1
    return "\n".join(result)


def _fix_seq_one_letter_code(text: str, entity_sequences: dict[str, str]) -> str:
    """Replace ? in pdbx_seq_one_letter_code with actual 1-letter sequence."""
    lines = text.split("\n")
    in_entity_poly = False
    columns: list[str] = []
    seq_col_idx = -1
    entity_col_idx = -1
    result = []

    for line in lines:
        if "_entity_poly." in line and not line.strip().startswith("loop_"):
            in_entity_poly = True
            columns.append(line.strip())
            if "pdbx_seq_one_letter_code" in line:
                seq_col_idx = len(columns) - 1
            if line.strip() == "_entity_poly.entity_id":
                entity_col_idx = len(columns) - 1
            result.append(line)
            continue
        if in_entity_poly and line.strip() and not line.startswith("_"):
            in_entity_poly = False
            if seq_col_idx >= 0 and entity_col_idx >= 0:
                parts = line.split()
                if len(parts) > max(seq_col_idx, entity_col_idx):
                    entity_id = parts[entity_col_idx]
                    seq = entity_sequences.get(entity_id, "?")
                    if len(seq) > 80:
                        parts[seq_col_idx] = f"\n;{seq}\n;"
                    else:
                        parts[seq_col_idx] = seq if seq else "?"
                    result.append(" ".join(parts))
                    continue
        result.append(line)
    return "\n".join(result)


def save_mmcif(
    path: str | Path,
    output: DeepFold3Output,
    features: dict[str, torch.Tensor],
    num_plddt_bins: int = 50,
    entity_descriptions: dict[str, str] | None = None,
    title: str | None = None,
    sample_idx: int | None = None,
) -> None:
    """Write predicted structure(s) to an mmCIF file.

    Converts model output to Structure objects (one per sample), then
    delegates to structure_to_mmcif() for mmCIF generation.

    If ``sample_idx`` is provided, a single-model mmCIF is written for
    that sample index. Otherwise, when ``num_samples == 1`` the single
    sample is written, and when ``num_samples > 1`` the legacy
    multi-MODEL mmCIF writer is used (one MODEL block per sample).

    Parameters
    ----------
    path :           Output file path (.cif)
    output :         DeepFold3Output from model.forward()
    features :       Feature dict from featurizer (must include ref_mask,
                     ref_element, ref_atom_name_chars, res_type,
                     asym_id, residue_index)
    num_plddt_bins : Number of pLDDT bins used by the confidence head (50)
    entity_descriptions : Optional mapping from chain name to description
                          string, written as _entity.pdbx_description.
    title :          Optional structure title, written as _struct.title.
    sample_idx :     If provided, write only this single sample (one MODEL
                     block). Used by the AF3-compatible per-(seed, sample)
                     output layout.
    """
    from deepfold3.io.structure_builder import prediction_to_structure
    from deepfold3.io.structure_mmcif import save_structure_mmcif

    path = Path(path)

    num_samples = output.atom_positions.shape[0]

    # Explicit sample_idx → always write single-model mmCIF via the
    # Structure-based path.
    if sample_idx is not None:
        if sample_idx < 0 or sample_idx >= num_samples:
            raise IndexError(f"sample_idx={sample_idx} out of range for {num_samples} samples")
        structure = prediction_to_structure(
            output, features, sample_idx=sample_idx, name=path.stem, num_plddt_bins=num_plddt_bins
        )
        save_structure_mmcif(path, structure, entry_id=path.stem, entity_descriptions=entity_descriptions, title=title)
        return

    if num_samples == 1:
        structure = prediction_to_structure(
            output, features, sample_idx=0, name=path.stem, num_plddt_bins=num_plddt_bins
        )
        save_structure_mmcif(path, structure, entry_id=path.stem, entity_descriptions=entity_descriptions, title=title)
    else:
        # For backward compatibility: write multi-model mmCIF using legacy path
        # for multi-sample outputs, since Structure only holds one model.
        _save_mmcif_legacy(path, output, features, num_plddt_bins, entity_descriptions=entity_descriptions, title=title)


def _save_mmcif_legacy(
    path: Path,
    output: DeepFold3Output,
    features: dict[str, torch.Tensor],
    num_plddt_bins: int = 50,
    entity_descriptions: dict[str, str] | None = None,
    title: str | None = None,
) -> None:
    """Legacy multi-model mmCIF writer (used when num_samples > 1)."""
    atom_positions = output.atom_positions.cpu()
    ref_mask = features["ref_mask"].cpu()
    ref_element = features["ref_element"].cpu()
    ref_atom_name_chars = features["ref_atom_name_chars"].cpu()
    res_type = features["res_type"].cpu()
    asym_id = features["asym_id"].cpu()
    residue_index = features["residue_index"].cpu()

    # mol_type feature: definitive polymer/non-polymer indicator
    mol_type_feat = features.get("mol_type")
    if mol_type_feat is not None:
        mol_type_feat = mol_type_feat.cpu()

    # token_res_name feature: actual CCD residue name
    token_res_name = features.get("token_res_name")
    if token_res_name is not None:
        token_res_name = token_res_name.cpu()

    num_samples, N, M = atom_positions.shape[:3]

    if output.plddt is not None:
        plddt_cpu = output.plddt.cpu()
        if plddt_cpu.dim() == 2:
            plddt_cpu = plddt_cpu.unsqueeze(0)
        plddt_per_sample, _ = compute_plddt(plddt_cpu, ref_mask.float(), num_plddt_bins)
    else:
        plddt_per_sample = torch.zeros(num_samples, N, M)

    st = gemmi.Structure()
    st.name = path.stem
    st.cell = gemmi.UnitCell(1.0, 1.0, 1.0, 90.0, 90.0, 90.0)
    st.spacegroup_hm = "P 1"

    unique_asym_ids = asym_id.unique().tolist()

    for sample_idx in range(num_samples):
        model = gemmi.Model(str(sample_idx + 1))

        for asym in unique_asym_ids:
            chain_name = asym_id_to_chain_name(int(asym))
            chain = gemmi.Chain(chain_name)

            token_indices = (asym_id == asym).nonzero(as_tuple=True)[0]

            for token_i in token_indices.tolist():
                rt = int(res_type[token_i])

                # Resolve residue name: use actual CCD name from feature if available
                if token_res_name is not None:
                    res_name = bytes(int(b) for b in token_res_name[token_i] if int(b) != 0).decode(
                        "ascii", errors="replace"
                    )
                    if not res_name:
                        res_name = RES_TYPE_INDEX_TO_NAME.get(rt, "UNK")
                else:
                    res_name = RES_TYPE_INDEX_TO_NAME.get(rt, "UNK")

                res_seq_num = int(residue_index[token_i])

                residue = gemmi.Residue()
                residue.name = res_name
                residue.seqid = gemmi.SeqId(str(res_seq_num))

                # Determine entity type from mol_type feature (definitive) or
                # fall back to _CHEM_COMP_TYPE lookup
                if mol_type_feat is not None:
                    mt = int(mol_type_feat[token_i])
                    residue.entity_type = (
                        gemmi.EntityType.NonPolymer if mt == MOL_NONPOLYMER else gemmi.EntityType.Polymer
                    )
                elif res_name not in _CHEM_COMP_TYPE:
                    residue.entity_type = gemmi.EntityType.NonPolymer
                else:
                    residue.entity_type = gemmi.EntityType.Polymer

                for atom_j in range(M):
                    if not bool(ref_mask[token_i, atom_j]):
                        continue
                    atom_name = _decode_atom_name(ref_atom_name_chars[token_i, atom_j])
                    if not atom_name:
                        continue
                    elem_num = int(ref_element[token_i, atom_j])
                    elem_sym = _INT_TO_ELEMENT.get(elem_num, "C")
                    x, y, z = atom_positions[sample_idx, token_i, atom_j].tolist()
                    bfactor = float(plddt_per_sample[sample_idx, token_i, atom_j])

                    atom = gemmi.Atom()
                    atom.name = atom_name
                    atom.pos = gemmi.Position(x, y, z)
                    atom.occ = 1.0
                    atom.b_iso = bfactor
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

    st.setup_entities()

    for i, entity in enumerate(st.entities, start=1):
        entity.name = str(i)

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

    chain_sequences: dict[str, list[str]] = {}
    for chain in st[0]:
        chain_sequences[chain.name] = [res.name for res in chain]

    for entity in st.entities:
        if entity.entity_type == gemmi.EntityType.Polymer and entity.subchains:
            chain_name = entity.subchains[0]
            seq = chain_sequences.get(chain_name, [])
            entity.full_sequence = seq

    st.assign_label_seq_id()

    plddt_per_residue: dict[tuple[str, str, int], float] = {}
    for asym in unique_asym_ids:
        chain_name = asym_id_to_chain_name(int(asym))
        token_indices = (asym_id == asym).nonzero(as_tuple=True)[0]
        for token_i in token_indices.tolist():
            mask_i = ref_mask[token_i]
            if mask_i.any():
                plddt_vals = plddt_per_sample[0, token_i][mask_i]
                avg_plddt = float(plddt_vals.mean())
            else:
                avg_plddt = 0.0
            rt = int(res_type[token_i])
            if token_res_name is not None:
                rn = bytes(int(b) for b in token_res_name[token_i] if int(b) != 0).decode("ascii", errors="replace")
                if not rn:
                    rn = RES_TYPE_INDEX_TO_NAME.get(rt, "UNK")
            else:
                rn = RES_TYPE_INDEX_TO_NAME.get(rt, "UNK")
            seq_id = int(residue_index[token_i])
            plddt_per_residue[(chain_name, rn, seq_id)] = avg_plddt

    doc = st.make_mmcif_document()
    base_text = doc.as_string()
    base_text = _enrich_cif_text(base_text)

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

    header = _build_header(path.stem, title=title)
    lines = base_text.split("\n", 1)
    final_text = lines[0] + "\n" + header + lines[1]
    final_text += _build_plddt_metric_table(plddt_per_residue)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(final_text)
