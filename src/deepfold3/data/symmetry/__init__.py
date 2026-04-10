"""Symmetry handling for chain and atom permutation alignment.

Three sub-modules:

* :mod:`atom_perms` — per-residue symmetric atom swaps for the standard
  amino-acid 20 (e.g. ASP OD1↔OD2, PHE CD1↔CD2, ARG NH1↔NH2).
* :mod:`chain_perm` — chain assignment alignment for homo-multimer
  training, using Hungarian matching on per-chain RMSD costs.
* :mod:`ligand_perm` — automorphism-group enumeration for arbitrary
  ligands via RDKit ``GetSubstructMatches``.

The algorithms follow AlphaFold 3 §4.2/§4.3. Implementation is
clean-room — Protenix's permutation package was used as a high-level
reference for module structure but no Protenix code is reproduced.
"""

from deepfold3.data.symmetry.atom_perms import (
    RESIDUE_ATOM_SYMMETRY_GROUPS,
    apply_atom_permutation,
    build_residue_symmetry_pairs,
    compute_atom_symmetry_pairs,
)
from deepfold3.data.symmetry.chain_perm import (
    apply_chain_permutation,
    compute_chain_permutation,
)

__all__ = [
    "RESIDUE_ATOM_SYMMETRY_GROUPS",
    "apply_atom_permutation",
    "apply_chain_permutation",
    "build_residue_symmetry_pairs",
    "compute_atom_symmetry_pairs",
    "compute_chain_permutation",
]
