"""Tests for template seqres-to-structure realignment."""

from __future__ import annotations

import pytest

from deepfold3.data.parse.template_realign import (
    AlignmentError,
    _remap_to_struc_seq,
    realign_hit_to_structure,
)

# ---------------------------------------------------------------------------
# _remap_to_struc_seq
# ---------------------------------------------------------------------------


class TestRemapToStrucSeq:
    def test_identical_sequences(self):
        """When hit and structure sequences match, mapping is unchanged."""
        mapping, score = _remap_to_struc_seq(
            hit_seq="ABCDE",
            struc_seq="ABCDE",
            max_num_gaps=0,
            mapping={0: 0, 1: 1, 2: 2, 3: 3, 4: 4},
        )
        assert mapping == {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
        assert score == 5

    def test_one_gap_in_structure(self):
        """Structure is missing one residue relative to seqres."""
        # hit_seq (seqres): "ABCDE"
        # struc_seq:        "ABDE"  (C is missing → 1 gap)
        mapping, score = _remap_to_struc_seq(
            hit_seq="ABCDE",
            struc_seq="ABDE",
            max_num_gaps=1,
            mapping={0: 0, 1: 1, 2: 2, 3: 3, 4: 4},
        )
        # hit positions 0→A, 1→B match; 2→C skipped (gap); 3→D, 4→E match
        assert mapping[0] == 0  # A→A
        assert mapping[1] == 1  # B→B
        assert 2 not in mapping  # C gone
        assert mapping[3] == 2  # D→D (now at struct idx 2)
        assert mapping[4] == 3  # E→E (now at struct idx 3)
        assert score == 4

    def test_multiple_gaps(self):
        """Structure is missing multiple residues."""
        # hit: "ABCDEF", struct: "ADF" (B,C,E missing → 3 gaps)
        mapping, score = _remap_to_struc_seq(
            hit_seq="ABCDEF",
            struc_seq="ADF",
            max_num_gaps=3,
            mapping={0: 0, 2: 2, 5: 5},
        )
        assert mapping[0] == 0  # A→A
        # position 2 (C) → no match in struct
        assert mapping[5] == 2  # F→F (struct idx 2)
        assert score >= 2

    def test_empty_mapping(self):
        mapping, score = _remap_to_struc_seq(
            hit_seq="ABC",
            struc_seq="AC",
            max_num_gaps=1,
            mapping={},
        )
        assert mapping == {}

    def test_partial_mapping(self):
        """Only some query positions are in the mapping."""
        mapping, score = _remap_to_struc_seq(
            hit_seq="ABCDE",
            struc_seq="ABCDE",
            max_num_gaps=0,
            mapping={1: 1, 3: 3},
        )
        assert mapping == {1: 1, 3: 3}
        assert score == 5


# ---------------------------------------------------------------------------
# realign_hit_to_structure
# ---------------------------------------------------------------------------


class TestRealignHitToStructure:
    def test_af3_docstring_example(self):
        """Verify the example from AlphaFold 3 docstring."""
        # query : ABCDEFGHIJKL
        # hit   : ---DEFGHIJK-   (seqres[3:11])
        # struct: XDEFGHKL       (8 chars, full seqres is 12 chars)
        mapping = realign_hit_to_structure(
            hit_sequence="DEFGHIJK",
            hit_start_index=3,
            hit_end_index=11,
            full_length=12,
            structure_sequence="XDEFGHKL",
            query_to_hit_mapping={3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7},
        )
        assert mapping == {3: 1, 4: 2, 5: 3, 6: 4, 7: 5, 10: 6}

    def test_no_realignment_needed(self):
        """When seqres and structure have the same length, offsets are applied."""
        mapping = realign_hit_to_structure(
            hit_sequence="CDE",
            hit_start_index=2,
            hit_end_index=5,
            full_length=7,
            structure_sequence="ABCDEFG",
            query_to_hit_mapping={0: 0, 1: 1, 2: 2},
        )
        # Positions offset by start_index=2 → {0:2, 1:3, 2:4}
        assert mapping == {0: 2, 1: 3, 2: 4}

    def test_single_missing_residue(self):
        """One residue missing from structure."""
        # Full seqres: "ABCDE" (5)
        # Structure:   "ABDE"  (4) — C is missing
        # Hit covers "BCD" (positions 1..4 in seqres)
        mapping = realign_hit_to_structure(
            hit_sequence="BCD",
            hit_start_index=1,
            hit_end_index=4,
            full_length=5,
            structure_sequence="ABDE",
            query_to_hit_mapping={0: 0, 1: 1, 2: 2},
        )
        # B is at struct position 1, D is at struct position 2
        assert mapping[0] == 1  # B→B
        assert mapping[2] == 2  # D→D
        # C (query_to_hit 1→1) should be dropped (not in structure)

    def test_missing_residues_at_start(self):
        """Gaps before the hit region in the structure."""
        # Full seqres: "XYZABC" (6)
        # Structure:   "ZABC"   (4) — X,Y missing (2 gaps)
        # Hit covers "ABC" (seqres[3:6])
        mapping = realign_hit_to_structure(
            hit_sequence="ABC",
            hit_start_index=3,
            hit_end_index=6,
            full_length=6,
            structure_sequence="ZABC",
            query_to_hit_mapping={0: 0, 1: 1, 2: 2},
        )
        # Structure "ZABC": A at 1, B at 2, C at 3
        assert mapping == {0: 1, 1: 2, 2: 3}

    def test_structure_longer_than_seqres_raises(self):
        with pytest.raises(AlignmentError, match="must not be longer"):
            realign_hit_to_structure(
                hit_sequence="ABC",
                hit_start_index=0,
                hit_end_index=3,
                full_length=3,
                structure_sequence="ABCDE",
                query_to_hit_mapping={0: 0, 1: 1, 2: 2},
            )

    def test_inconsistent_hit_region_raises(self):
        with pytest.raises(AlignmentError, match="does not equal"):
            realign_hit_to_structure(
                hit_sequence="AB",
                hit_start_index=0,
                hit_end_index=5,  # 5-0=5 != len("AB")=2
                full_length=10,
                structure_sequence="ABCDE",
                query_to_hit_mapping={0: 0},
            )

    def test_empty_mapping(self):
        mapping = realign_hit_to_structure(
            hit_sequence="ABC",
            hit_start_index=0,
            hit_end_index=3,
            full_length=5,
            structure_sequence="ABC",
            query_to_hit_mapping={},
        )
        assert mapping == {}

    def test_hit_at_start_of_seqres(self):
        """Hit starts at position 0 — no room for gaps before."""
        # Full seqres: "ABCDE" (5)
        # Structure:   "ACDE"  (4) — B missing
        # Hit: "ABC" at positions [0:3]
        mapping = realign_hit_to_structure(
            hit_sequence="ABC",
            hit_start_index=0,
            hit_end_index=3,
            full_length=5,
            structure_sequence="ACDE",
            query_to_hit_mapping={0: 0, 1: 1, 2: 2},
        )
        assert mapping[0] == 0  # A→A
        # B is gone, C maps to struct position 1 (shifted due to gap)

    def test_sparse_mapping(self):
        """Mapping only covers a few positions in the hit region."""
        # Full seqres: "ABCDEFGH" (8)
        # Structure:   "ABDEFGH"  (7) — C missing
        # Hit: "CDEF" at positions [2:6]
        mapping = realign_hit_to_structure(
            hit_sequence="CDEF",
            hit_start_index=2,
            hit_end_index=6,
            full_length=8,
            structure_sequence="ABDEFGH",
            query_to_hit_mapping={5: 1, 10: 3},  # query 5→D, query 10→F
        )
        # D is at struct position 2, F is at struct position 4
        assert mapping[5] == 2
        assert mapping[10] == 4


# ---------------------------------------------------------------------------
# _realign_mapping (integration with templates.py helpers)
# ---------------------------------------------------------------------------


class TestRealignMapping:
    def test_none_seqres_passthrough(self):
        """When seqres is None, mapping passes through unchanged."""
        from deepfold3.data.parse.templates import _realign_mapping

        m = {0: 5, 1: 6, 2: 7}
        result = _realign_mapping(None, "ABCDEFGH", m)
        assert result == m

    def test_equal_length_passthrough(self):
        """When seqres and struct have same length, no realignment."""
        from deepfold3.data.parse.templates import _realign_mapping

        m = {0: 2, 1: 3}
        result = _realign_mapping("ABCDE", "ABCDE", m)
        assert result == m

    def test_length_mismatch_triggers_realignment(self):
        """When lengths differ, realignment is applied."""
        from deepfold3.data.parse.templates import _realign_mapping

        # seqres: "ABCDE" (5), struct: "ABDE" (4) — C missing
        # Mapping: query 0→pos 0 (A), query 1→pos 2 (C), query 2→pos 3 (D)
        m = {0: 0, 1: 2, 2: 3}
        result = _realign_mapping("ABCDE", "ABDE", m)
        assert result[0] == 0  # A→A (struct pos 0)
        assert result[2] == 2  # D→D (struct pos 2)
        # C (seqres pos 2) is missing from structure

    def test_empty_mapping(self):
        from deepfold3.data.parse.templates import _realign_mapping

        result = _realign_mapping("ABCDE", "ABDE", {})
        assert result == {}

    def test_fallback_on_alignment_error(self):
        """When realignment fails, original mapping is returned."""
        from deepfold3.data.parse.templates import _realign_mapping

        # Create a scenario that will fail: structure longer than seqres
        # _realign_mapping should catch AlignmentError and return original
        m = {0: 0, 1: 1}
        result = _realign_mapping("AB", "ABCDE", m)
        # AlignmentError caught → returns original mapping
        assert result == m
