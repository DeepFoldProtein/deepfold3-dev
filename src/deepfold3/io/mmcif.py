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

from deepfold3.data.const import ELEMENT_TO_INT, RES_TYPE_INDEX_TO_NAME, THREE_LETTER_TO_ONE
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


def _decode_atom_name(chars: torch.Tensor) -> str:
    """Decode 4-int atom name tensor (ord(c)-32 encoding) to string."""
    return "".join(chr(int(c) + 32) for c in chars).strip()


def _cif_quote(value: str) -> str:
    """Quote a CIF value if it contains spaces or special characters."""
    if " " in value or "'" in value or value.startswith("_") or value.startswith("#"):
        if "'" not in value:
            return f"'{value}'"
        if '"' not in value:
            return f'"{value}"'
        return f"\n;{value}\n;"
    return value


def _build_header(entry_id: str) -> str:
    """Build standard mmCIF header categories."""
    return f"""\
#
_audit_conform.dict_name       mmcif_pdbx.dic
_audit_conform.dict_version    5.398
_audit_conform.dict_location   http://mmcif.pdb.org/dictionaries/ascii/mmcif_pdbx.dic
#
_struct.entry_id                  {entry_id}
_struct.title                     'DeepFold3 predicted structure'
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
        "_ma_qa_metric.description         'Predicted per-residue confidence score (0-100)'",
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
                comp_type = _CHEM_COMP_TYPE.get(comp_id, "other")
                result.append(f"{comp_id} '{comp_type}'")
                continue
            else:
                in_chem_comp = False
        elif in_chem_comp and (not line.strip() or line.startswith("loop_") or line.startswith("#")):
            in_chem_comp = False
        result.append(line)
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
) -> None:
    """Write predicted structure(s) to an mmCIF file.

    Converts model output to Structure objects (one per sample), then
    delegates to structure_to_mmcif() for mmCIF generation.

    For multi-sample output, only the first sample is written (each sample
    can be saved separately via prediction_to_structure + save_structure_mmcif).

    Parameters
    ----------
    path :           Output file path (.cif)
    output :         DeepFold3Output from model.forward()
    features :       Feature dict from featurizer (must include ref_mask,
                     ref_element, ref_atom_name_chars, res_type,
                     asym_id, residue_index)
    num_plddt_bins : Number of pLDDT bins used by the confidence head (50)
    """
    from deepfold3.io.structure_builder import prediction_to_structure
    from deepfold3.io.structure_mmcif import save_structure_mmcif

    path = Path(path)

    # Build one Structure per sample and write the first one
    # (multi-model mmCIF with all samples from a single Structure would
    # require merging, which is not yet supported — callers can save
    # individual samples separately)
    num_samples = output.atom_positions.shape[0]

    if num_samples == 1:
        structure = prediction_to_structure(
            output, features, sample_idx=0, name=path.stem, num_plddt_bins=num_plddt_bins
        )
        save_structure_mmcif(path, structure, entry_id=path.stem)
    else:
        # For backward compatibility: write multi-model mmCIF using legacy path
        # for multi-sample outputs, since Structure only holds one model.
        _save_mmcif_legacy(path, output, features, num_plddt_bins)


def _save_mmcif_legacy(
    path: Path,
    output: DeepFold3Output,
    features: dict[str, torch.Tensor],
    num_plddt_bins: int = 50,
) -> None:
    """Legacy multi-model mmCIF writer (used when num_samples > 1)."""
    atom_positions = output.atom_positions.cpu()
    ref_mask = features["ref_mask"].cpu()
    ref_element = features["ref_element"].cpu()
    ref_atom_name_chars = features["ref_atom_name_chars"].cpu()
    res_type = features["res_type"].cpu()
    asym_id = features["asym_id"].cpu()
    residue_index = features["residue_index"].cpu()

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
                res_name = RES_TYPE_INDEX_TO_NAME.get(rt, "UNK")
                res_seq_num = int(residue_index[token_i])

                residue = gemmi.Residue()
                residue.name = res_name
                residue.seqid = gemmi.SeqId(str(res_seq_num))
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
            res_name = RES_TYPE_INDEX_TO_NAME.get(rt, "UNK")
            seq_id = int(residue_index[token_i])
            plddt_per_residue[(chain_name, res_name, seq_id)] = avg_plddt

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

    header = _build_header(path.stem)
    lines = base_text.split("\n", 1)
    final_text = lines[0] + "\n" + header + lines[1]
    final_text += _build_plddt_metric_table(plddt_per_residue)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(final_text)
