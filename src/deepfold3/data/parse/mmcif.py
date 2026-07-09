"""gemmi entity → DeepFold3 mol-type mapping.

The mmCIF → :class:`~deepfold3.data.types.Structure` conversion lives in
``deepfold3.data.prepare.structure.process_mmcif`` (the training-prepare path,
with all the AF3 cleanup steps). This module only exposes the small
``mol_type_from_gemmi`` helper that path depends on.
"""

from __future__ import annotations

try:
    import gemmi
except ImportError as e:
    raise ImportError("gemmi is required for mmCIF parsing: pip install gemmi") from e

from deepfold3.data import const


def mol_type_from_gemmi(entity: gemmi.Entity) -> int | None:
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
