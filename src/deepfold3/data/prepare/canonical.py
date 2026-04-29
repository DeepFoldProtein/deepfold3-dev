"""Canonical-sequence helpers for mmCIF entities.

Reads the canonical polymer sequence from a gemmi entity's ``full_sequence``
(which is populated from ``_entity_poly.pdbx_seq_one_letter_code_can`` /
``_entity_poly_seq.mon_id`` after ``setup_entities()``). This is what AF3 §2
specifies for MSA queries — disordered residues are included, modified
residues are mapped to their standard parent.

Used by:
- ``prepare/structure.py`` — to compute ``msa_id`` from the canonical
  sequence rather than observed-only residues.
- ``parse/templates.py`` — for SEQRES vs structure realignment in the
  template path (existing usage).
"""

from __future__ import annotations

try:
    import gemmi
except ImportError as e:
    raise ImportError("gemmi is required for canonical sequence helpers: pip install gemmi") from e

from deepfold3.data import const


def _one_letter(info) -> str:
    """Extract one-letter code from a gemmi ResidueInfo, defaulting to 'X'.

    Treats whitespace / placeholder codes as 'X' — ``find_tabulated_residue``
    returns a record with ``one_letter_code = ' '`` for unrecognised names
    (including comma-joined microheterogeneity strings like ``"MSE,MET"``).
    """
    if info is None:
        return "X"
    c = info.one_letter_code
    if not c or c in ("?", "\0") or not c.strip():
        return "X"
    return c


def _seqres_one_letter(item) -> str:
    """One-letter code for a seqres monomer name, handling microheterogeneity.

    Gemmi encodes microheterogeneity as a comma-joined name (e.g.
    ``"MSE,MET"``) in ``entity.full_sequence``. We use the first alternative
    so the seqres length matches ``polymer.first_conformer()`` on the
    structure side.
    """
    primary = str(item).split(",", 1)[0].strip()
    if not primary:
        return "X"
    return _one_letter(gemmi.find_tabulated_residue(primary))


def _seqres_three_letter(item) -> str:
    """3-letter mon_id for a seqres entry, with NONSTANDARD → STANDARD mapping.

    Used for entity dedup and ``msa_id`` hashing — must match the existing
    ``_chain_sequence(chain_res_data)`` behaviour in ``prepare/structure.py``,
    which maps non-standard residues (MSE → MET, etc.) before joining.

    Returns ``"UNK"`` for unmapped / unrecognised residues so the join-and-hash
    path stays well-defined.
    """
    primary = str(item).split(",", 1)[0].strip()
    if not primary:
        return "UNK"
    return const._NONSTANDARD_TO_STANDARD.get(primary, primary)


def get_canonical_three_letter(entity) -> list[str] | None:
    """Return the canonical 3-letter mon_id list for a gemmi polymer entity.

    Modified residues are mapped to their standard parent (matches the
    convention used by ``_chain_sequence`` in ``prepare/structure.py``).
    Microheterogeneity collapses to the first alternative.

    Args:
        entity: ``gemmi.Entity`` — caller must ensure ``setup_entities()``
            was called on the parent ``Structure`` so ``full_sequence`` is
            populated.

    Returns:
        List of 3-letter codes (one per polymer position), or ``None`` if
        the entity has no full_sequence (non-polymer, or pre-``setup_entities``,
        or pathological input).
    """
    if entity is None or not entity.full_sequence:
        return None
    return [_seqres_three_letter(item) for item in entity.full_sequence]


def get_canonical_one_letter(entity) -> str | None:
    """Return the canonical one-letter sequence for a gemmi polymer entity.

    Same source as :func:`get_canonical_three_letter` but emits the AF3
    ``pdbx_seq_one_letter_code_can``-equivalent string suitable for MSA
    search. Modified residues are mapped to their parent standard letter
    via gemmi's tabulated residue table (so MSE → M, SEP → S, etc.).

    Always uppercase: gemmi's ``one_letter_code`` uses lowercase for modified
    residues (e.g. MSE → 'm') to distinguish them visually, but AF3's
    canonical form and MMseqs2/HHblits expect uppercase ``A–Z`` for the
    query, so we uppercase here.
    """
    if entity is None or not entity.full_sequence:
        return None
    return "".join(_seqres_one_letter(item) for item in entity.full_sequence).upper()
