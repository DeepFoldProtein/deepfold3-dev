"""Build Structure objects from model prediction outputs.

Converts DeepFold3Output + feature dict → Structure, bridging the model
output to the Structure-centric I/O layer.
"""

from __future__ import annotations

import numpy as np
import torch

from deepfold3.data import const
from deepfold3.data.const import RES_TYPE_INDEX_TO_NAME
from deepfold3.data.types import (
    ATOM_DTYPE,
    BOND_DTYPE,
    CHAIN_DTYPE,
    RESIDUE_DTYPE,
    Structure,
)
from deepfold3.io.confidences import compute_plddt
from deepfold3.io.mmcif import _decode_atom_name, asym_id_to_chain_name
from deepfold3.model.model import DeepFold3Output


def _res_type_to_mol_type(res_type: int) -> int:
    """Infer mol_type from res_type index."""
    if res_type <= 20:
        return const.MOL_PROTEIN
    if 22 <= res_type <= 25:
        return const.MOL_RNA
    if 26 <= res_type <= 29:
        return const.MOL_DNA
    return const.MOL_NONPOLYMER


def prediction_to_structure(
    output: DeepFold3Output,
    features: dict[str, torch.Tensor],
    sample_idx: int = 0,
    name: str = "prediction",
    num_plddt_bins: int = 50,
) -> Structure:
    """Convert a model prediction to a Structure.

    Parameters
    ----------
    output       : DeepFold3Output from model.forward().
    features     : Feature dict from featurizer.
    sample_idx   : Which diffusion sample to use.
    name         : Structure name/identifier.
    num_plddt_bins : Number of pLDDT bins (default 50).

    Returns
    -------
    Structure with atom coordinates, pLDDT as b_factor, and full metadata.
    """
    atom_positions = output.atom_positions.cpu()  # [S, N, M, 3]
    ref_mask = features["ref_mask"].cpu()  # [N, M] bool
    ref_element = features["ref_element"].cpu()  # [N, M] int64
    ref_atom_name_chars = features["ref_atom_name_chars"].cpu()  # [N, M, 4]
    res_type = features["res_type"].cpu()  # [N] int64
    asym_id = features["asym_id"].cpu()  # [N] int64
    residue_index = features["residue_index"].cpu()  # [N] int64

    N, M = ref_mask.shape

    # Compute per-atom pLDDT
    if output.plddt is not None:
        plddt_cpu = output.plddt.cpu()
        if plddt_cpu.dim() == 2:
            plddt_cpu = plddt_cpu.unsqueeze(0)
        plddt_per_sample, _ = compute_plddt(plddt_cpu, ref_mask.float(), num_plddt_bins)
    else:
        plddt_per_sample = torch.zeros(atom_positions.shape[0], N, M)

    # Group tokens by asym_id to build chains
    unique_asym_ids = asym_id.unique(sorted=True).tolist()

    all_atoms: list[tuple] = []
    all_residues: list[tuple] = []
    all_chains: list[tuple] = []

    atom_idx_global = 0
    res_idx_global = 0

    # Track entity_id from sequence deduplication
    seq_to_entity: dict[str, list[int]] = {}
    next_entity_id = 1

    for asym in unique_asym_ids:
        token_indices = (asym_id == asym).nonzero(as_tuple=True)[0].tolist()
        chain_name = asym_id_to_chain_name(int(asym))

        # Determine mol_type from first token's res_type
        first_rt = int(res_type[token_indices[0]])
        mol_type = _res_type_to_mol_type(first_rt)

        # Build sequence key for entity dedup
        seq_key = "|".join(str(int(res_type[ti])) for ti in token_indices)
        if seq_key in seq_to_entity:
            entity_id = seq_to_entity[seq_key][0]
            sym_id = seq_to_entity[seq_key][1]
            seq_to_entity[seq_key][1] += 1
        else:
            entity_id = next_entity_id
            sym_id = 0
            seq_to_entity[seq_key] = [entity_id, 1]
            next_entity_id += 1

        chain_atom_start = atom_idx_global
        chain_res_start = res_idx_global

        for token_i in token_indices:
            rt = int(res_type[token_i])
            res_name = RES_TYPE_INDEX_TO_NAME.get(rt, "UNK")
            res_seq_num = int(residue_index[token_i])

            res_atom_start = atom_idx_global
            res_atom_count = 0

            for atom_j in range(M):
                if not bool(ref_mask[token_i, atom_j]):
                    continue

                atom_name = _decode_atom_name(ref_atom_name_chars[token_i, atom_j])
                if not atom_name:
                    continue

                x, y, z = atom_positions[sample_idx, token_i, atom_j].tolist()
                elem_num = int(ref_element[token_i, atom_j])
                bfactor = float(plddt_per_sample[sample_idx, token_i, atom_j])

                all_atoms.append(
                    (
                        x,
                        y,
                        z,
                        elem_num,
                        0.0,  # charge
                        atom_name.ljust(4)[:4].encode("ascii"),
                        True,  # is_present
                        bfactor,  # b_factor
                    )
                )
                atom_idx_global += 1
                res_atom_count += 1

            is_standard = mol_type != const.MOL_NONPOLYMER and rt < 20

            all_residues.append(
                (
                    res_name[:5].encode("ascii"),
                    rt,
                    mol_type,
                    res_seq_num,
                    res_atom_start,
                    res_atom_count,
                    is_standard,
                    res_atom_count > 0,
                )
            )
            res_idx_global += 1

        chain_atom_count = atom_idx_global - chain_atom_start
        chain_res_count = res_idx_global - chain_res_start

        all_chains.append(
            (
                chain_name[:4].encode("ascii"),
                mol_type,
                entity_id,
                sym_id,
                int(asym),
                chain_res_start,
                chain_res_count,
                chain_atom_start,
                chain_atom_count,
            )
        )

    atoms_arr = np.array(all_atoms, dtype=ATOM_DTYPE) if all_atoms else np.empty(0, dtype=ATOM_DTYPE)
    residues_arr = np.array(all_residues, dtype=RESIDUE_DTYPE) if all_residues else np.empty(0, dtype=RESIDUE_DTYPE)
    chains_arr = np.array(all_chains, dtype=CHAIN_DTYPE) if all_chains else np.empty(0, dtype=CHAIN_DTYPE)
    bonds_arr = np.empty(0, dtype=BOND_DTYPE)

    return Structure(
        atoms=atoms_arr,
        bonds=bonds_arr,
        residues=residues_arr,
        chains=chains_arr,
        name=name,
    )


def prediction_to_structures(
    output: DeepFold3Output,
    features: dict[str, torch.Tensor],
    name: str = "prediction",
    num_plddt_bins: int = 50,
) -> list[Structure]:
    """Convert all diffusion samples to Structure objects.

    Returns one Structure per sample, each with pLDDT as b_factor.
    """
    num_samples = output.atom_positions.shape[0]
    return [
        prediction_to_structure(output, features, sample_idx=s, name=f"{name}_s{s}", num_plddt_bins=num_plddt_bins)
        for s in range(num_samples)
    ]
