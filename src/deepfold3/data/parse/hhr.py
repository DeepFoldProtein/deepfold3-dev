"""HHsearch result (.hhr) parser.

Parses HHR files produced by HHsearch/HHblits into structured TemplateHit
objects containing alignment information for template feature extraction.

Reference format:
    https://github.com/soedinglab/hh-suite/wiki
"""

from __future__ import annotations

import dataclasses
import re


@dataclasses.dataclass(frozen=True)
class TemplateHit:
    """A single template hit from an HHR file.

    Attributes:
        index: 1-based hit number from the HHR file.
        pdb_id: 4-character PDB ID (uppercased).
        chain_id: Single-character chain identifier.
        aligned_cols: Number of aligned columns.
        sum_probs: Sum of per-column probabilities (quality metric).
        e_value: E-value of the hit.
        query_sequence: Aligned query sequence (with gaps as '-').
        hit_sequence: Aligned hit sequence (with gaps as '-').
        indices_query: 0-based query residue indices for each alignment column
            (-1 for gap positions).
        indices_hit: 0-based hit residue indices for each alignment column
            (-1 for gap positions).
    """

    index: int
    pdb_id: str
    chain_id: str
    aligned_cols: int
    sum_probs: float
    e_value: float
    query_sequence: str
    hit_sequence: str
    indices_query: list[int]
    indices_hit: list[int]

    @property
    def query_to_hit_mapping(self) -> dict[int, int]:
        """Map from 0-based query index to 0-based hit index for aligned (non-gap) positions."""
        return {qi: hi for qi, hi in zip(self.indices_query, self.indices_hit, strict=True) if qi != -1 and hi != -1}


def parse_hhr(hhr_text: str) -> list[TemplateHit]:
    """Parse an HHR file into a list of TemplateHit objects.

    Args:
        hhr_text: Full contents of an .hhr file.

    Returns:
        List of TemplateHit, one per alignment block.
    """
    # Split into blocks starting with "No <number>"
    blocks = re.split(r"^No ", hhr_text, flags=re.MULTILINE)

    hits: list[TemplateHit] = []
    for block in blocks[1:]:  # skip preamble before first "No "
        hit = _parse_block(block)
        if hit is not None:
            hits.append(hit)

    return hits


_HEADER_RE = re.compile(
    r"^(\d+)\s*\n"  # hit index
    r">(\S+)\s",  # name (e.g. "4O5A_A")
    re.MULTILINE,
)

_STATS_RE = re.compile(
    r"Probab=(\S+)\s+"
    r"E-value=(\S+)\s+"
    r"Score=(\S+)\s+"
    r"Aligned_cols=(\d+)\s+"
    r"Identities=\S+\s+"
    r"Similarity=\S+\s+"
    r"Sum_probs=(\S+)"
)

_QUERY_RE = re.compile(r"^Q query\s+(\d+)\s+(\S+)\s+\d+\s+\(\d+\)")
_TEMPL_RE = re.compile(r"^T (?!Consensus|ss_dssp|ss_pred)\S+\s+(\d+)\s+(\S+)\s+\d+\s+\(\d+\)")


def _parse_block(block_text: str) -> TemplateHit | None:
    """Parse a single HHR alignment block."""
    header_match = _HEADER_RE.search(block_text)
    if header_match is None:
        return None

    hit_index = int(header_match.group(1))
    pdb_id, chain_id = _parse_pdb_chain(header_match.group(2))

    stats_match = _STATS_RE.search(block_text)
    if stats_match is None:
        return None

    e_value = float(stats_match.group(2))
    aligned_cols = int(stats_match.group(4))
    sum_probs = float(stats_match.group(5))

    # Extract alignment lines
    query_parts: list[tuple[int, str]] = []
    hit_parts: list[tuple[int, str]] = []

    for line in block_text.splitlines():
        line = line.rstrip()

        q_match = _QUERY_RE.match(line)
        if q_match:
            query_parts.append((int(q_match.group(1)), q_match.group(2)))
            continue

        t_match = _TEMPL_RE.match(line)
        if t_match:
            hit_parts.append((int(t_match.group(1)), t_match.group(2)))
            continue

    if not query_parts or not hit_parts or len(query_parts) != len(hit_parts):
        return None

    # Build aligned sequences and index arrays
    query_seq_parts: list[str] = []
    hit_seq_parts: list[str] = []
    indices_query: list[int] = []
    indices_hit: list[int] = []

    for (q_start, q_seq), (h_start, h_seq) in zip(query_parts, hit_parts, strict=True):
        if not query_seq_parts:
            q_counter = q_start - 1  # 1-based -> 0-based
            h_counter = h_start - 1

        query_seq_parts.append(q_seq)
        hit_seq_parts.append(h_seq)

        for qc, hc in zip(q_seq, h_seq, strict=True):
            qi = -1
            hi = -1
            if qc != "-":
                qi = q_counter
                q_counter += 1
            if hc != "-":
                hi = h_counter
                h_counter += 1
            indices_query.append(qi)
            indices_hit.append(hi)

    return TemplateHit(
        index=hit_index,
        pdb_id=pdb_id,
        chain_id=chain_id,
        aligned_cols=aligned_cols,
        sum_probs=sum_probs,
        e_value=e_value,
        query_sequence="".join(query_seq_parts),
        hit_sequence="".join(hit_seq_parts),
        indices_query=indices_query,
        indices_hit=indices_hit,
    )


def _parse_pdb_chain(name: str) -> tuple[str, str]:
    """Parse 'XXXX_Y' style names into (pdb_id, chain_id).

    Handles formats like '4O5A_A', '1SXG_I', etc.
    """
    parts = name.split("_", 1)
    pdb_id = parts[0].upper()
    chain_id = parts[1] if len(parts) > 1 else "A"
    return pdb_id, chain_id
