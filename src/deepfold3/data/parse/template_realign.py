"""Realign PDB seqres sequences to actual mmCIF structure sequences.

Search tools (MMseqs2, HHsearch) align against PDB SEQRES sequences.  However,
the resolved mmCIF structure may differ from the SEQRES — missing residues,
sequence corrections, or different 3→1-letter-code mappings.  This module
provides a simple gap-insertion realignment algorithm that fixes query-to-hit
mappings so they point to the correct positions in the resolved structure.

Ported from AlphaFold 3 ``template_realign.py`` (CC BY-NC-SA 4.0).
"""

from __future__ import annotations

from collections.abc import Mapping


class AlignmentError(Exception):
    """Failed alignment between the hit sequence and the actual mmCIF sequence."""


def realign_hit_to_structure(
    *,
    hit_sequence: str,
    hit_start_index: int,
    hit_end_index: int,
    full_length: int,
    structure_sequence: str,
    query_to_hit_mapping: Mapping[int, int],
) -> dict[int, int]:
    """Realign a hit subsequence from PDB seqres to the actual structure sequence.

    When PDB seqres differs from the resolved mmCIF structure (e.g. missing
    residues), the original ``query_to_hit_mapping`` (derived from a search
    alignment against seqres) becomes invalid.  This function realigns the hit
    region to the structure sequence and returns a corrected mapping.

    Example::

        query_sequence : ABCDEFGHIJKL
        hit_sequence   : ---DEFGHIJK-      (seqres subsequence)
        struc_sequence : XDEFGHKL           (actual resolved structure)

    Original mapping ``{3:0, 4:1, 5:2, 6:3, 7:4, 8:5, 9:6, 10:7}`` (relative
    to the hit subsequence) is corrected to ``{3:1, 4:2, 5:3, 6:4, 7:5, 10:6}``
    (positions in the structure sequence).

    Args:
        hit_sequence: The PDB seqres hit subsequence (no gaps), spanning from
            ``hit_start_index`` to ``hit_end_index`` in the full seqres.
        hit_start_index: Inclusive 0-based start in the full PDB seqres.
        hit_end_index: Exclusive 0-based end in the full PDB seqres.
        full_length: Length of the full PDB seqres sequence.
        structure_sequence: Actual sequence extracted from the mmCIF structure.
        query_to_hit_mapping: Mapping from 0-based query positions to 0-based
            positions within ``hit_sequence``.

    Returns:
        Corrected mapping from query positions to 0-based positions in the
        ``structure_sequence``.

    Raises:
        AlignmentError: If the structure sequence is longer than the seqres
            (unexpected) or the hit region dimensions are inconsistent.
    """
    max_num_gaps = full_length - len(structure_sequence)
    if max_num_gaps < 0:
        raise AlignmentError(
            f"The structure sequence ({len(structure_sequence)}) "
            f"must not be longer than the PDB seqres sequence ({full_length}):\n"
            f"Structure sequence : {structure_sequence}\n"
            f"PDB seqres hit    : {hit_sequence}"
        )

    if len(hit_sequence) != hit_end_index - hit_start_index:
        raise AlignmentError(
            f"hit_end_index={hit_end_index} - hit_start_index={hit_start_index} "
            f"does not equal len(hit_sequence)={len(hit_sequence)}"
        )

    best_score = -1
    best_start = 0
    best_query_to_hit_mapping: dict[int, int] = {}
    max_num_gaps_before_subseq = min(hit_start_index, max_num_gaps)

    # Try different gap placements before the hit region and pick the
    # alignment with the highest number of matching residues.
    for num_gaps_before_subseq in range(0, max_num_gaps_before_subseq + 1):
        start = hit_start_index - num_gaps_before_subseq
        end = hit_end_index - num_gaps_before_subseq
        structure_subseq = structure_sequence[start:end]

        new_mapping, score = _remap_to_struc_seq(
            hit_seq=hit_sequence,
            struc_seq=structure_subseq,
            max_num_gaps=max_num_gaps - num_gaps_before_subseq,
            mapping=query_to_hit_mapping,
        )
        if score >= best_score:
            # Prefer matches with larger number of gaps before (>= not >).
            best_score = score
            best_start = start
            best_query_to_hit_mapping = new_mapping

    return {q: h + best_start for q, h in best_query_to_hit_mapping.items()}


def _remap_to_struc_seq(
    *,
    hit_seq: str,
    struc_seq: str,
    max_num_gaps: int,
    mapping: Mapping[int, int],
) -> tuple[dict[int, int], int]:
    """Remap a query→hit mapping to match the actual structure subsequence.

    Aligns ``hit_seq`` (from PDB seqres) to ``struc_seq`` (resolved structure)
    using a simple gap-insertion algorithm, then remaps ``mapping`` values from
    hit-sequence positions to structure-sequence positions.

    Args:
        hit_seq: PDB seqres subsequence (no gaps).
        struc_seq: Actual structure subsequence.
        max_num_gaps: Maximum number of gaps insertable in the structure
            sequence (= ``len(seqres) - len(structure_sequence)``).
        mapping: Query-to-hit mapping to remap.

    Returns:
        ``(remapped_mapping, score)`` where *score* is the number of matching
        aligned residues.
    """
    hit_seq_idx = 0
    struc_seq_idx = 0
    hit_to_struc_seq_mapping: dict[int, int] = {}
    score = 0

    # Guaranteed to terminate: both indices advance by ≥1 each iteration.
    remaining_num_gaps = max_num_gaps
    while hit_seq_idx < len(hit_seq) and struc_seq_idx < len(struc_seq):
        if hit_seq[hit_seq_idx] != struc_seq[struc_seq_idx]:
            # Find the best shift in the hit sequence that aligns the next
            # residue to the structure.
            best_shift = 0
            for shift in range(0, remaining_num_gaps + 1):
                next_hit_res = hit_seq[hit_seq_idx + shift : hit_seq_idx + shift + 1]
                next_struc_res = struc_seq[struc_seq_idx : struc_seq_idx + 1]
                if next_hit_res == next_struc_res:
                    best_shift = shift
                    break
            hit_seq_idx += best_shift
            remaining_num_gaps -= best_shift

        hit_to_struc_seq_mapping[hit_seq_idx] = struc_seq_idx
        score += hit_seq[hit_seq_idx] == struc_seq[struc_seq_idx]
        hit_seq_idx += 1
        struc_seq_idx += 1

    fixed_mapping: dict[int, int] = {}
    for query_idx, original_hit_idx in mapping.items():
        fixed_hit_idx = hit_to_struc_seq_mapping.get(original_hit_idx)
        if fixed_hit_idx is not None:
            fixed_mapping[query_idx] = fixed_hit_idx

    return fixed_mapping, score
