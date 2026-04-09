"""Multimer template discovery from per-chain m8 hits.

Reads pre-computed m8 files (one per unique chain sequence, keyed by msa_id)
and discovers multimer templates: PDB entries that are structural homologs of
multiple query chains in a training complex.

Uses bipartite matching (Hungarian algorithm) to optimally pair query chains
to template chains, avoiding conflicts where multiple query chains map to the
same template chain.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from deepfold3.data.msa.mmseqs2 import TemplateM8Hit, parse_m8_single

logger = logging.getLogger(__name__)


@dataclass
class ChainAlignment:
    """Alignment of one query chain to one template chain."""

    query_chain_idx: int  # index in record.chains
    template_chain_id: str  # chain ID in template PDB
    identity: float
    evalue: float
    qstart: int  # 1-based
    qend: int  # 1-based
    tstart: int  # 1-based
    tend: int  # 1-based
    cigar: str


@dataclass
class MultimerTemplateHit:
    """A multimer template: one PDB entry covering multiple query chains."""

    template_pdb_id: str
    chain_alignments: list[ChainAlignment] = field(default_factory=list)
    score: float = 0.0

    @property
    def num_chains_covered(self) -> int:
        return len(self.chain_alignments)


def find_multimer_templates(
    record_pdb_id: str,
    chain_msa_ids: list[str],
    m8_dir: Path,
    *,
    max_templates: int = 4,
    max_hits_per_chain: int = 200,
) -> list[MultimerTemplateHit]:
    """Discover multimer templates from per-chain m8 files.

    Args:
        record_pdb_id: PDB ID of the training complex (excluded from results).
        chain_msa_ids: List of msa_id hashes, one per chain in the complex.
        m8_dir: Directory containing ``{msa_id}.m8`` files.
        max_templates: Maximum number of templates to return.
        max_hits_per_chain: Maximum m8 hits to consider per chain.

    Returns:
        List of MultimerTemplateHit, sorted by score descending.
    """
    from scipy.optimize import linear_sum_assignment

    m8_dir = Path(m8_dir)
    num_chains = len(chain_msa_ids)

    m8_cache: dict[str, list[TemplateM8Hit]] = {}
    for msa_id in set(chain_msa_ids):
        m8_path = m8_dir / f"{msa_id}.m8"
        if not m8_path.exists():
            m8_cache[msa_id] = []
            continue
        m8_cache[msa_id] = parse_m8_single(m8_path.read_text())

    hits_by_pdb: dict[str, list[tuple[int, TemplateM8Hit]]] = defaultdict(list)
    for chain_idx, msa_id in enumerate(chain_msa_ids):
        for hit in m8_cache.get(msa_id, [])[:max_hits_per_chain]:
            hits_by_pdb[hit.pdb_id].append((chain_idx, hit))

    hits_by_pdb.pop(record_pdb_id.upper(), None)

    results: list[MultimerTemplateHit] = []

    for tmpl_pdb_id, chain_hits in hits_by_pdb.items():
        best: dict[tuple[int, str], TemplateM8Hit] = {}
        for chain_idx, hit in chain_hits:
            key = (chain_idx, hit.chain_id)
            if key not in best or hit.identity > best[key].identity:
                best[key] = hit

        query_indices = sorted({k[0] for k in best})
        template_chains = sorted({k[1] for k in best})

        if not query_indices or not template_chains:
            continue

        n_q = len(query_indices)
        n_t = len(template_chains)
        cost = np.full((n_q, n_t), 1e6, dtype=np.float64)

        qi_map = {qi: i for i, qi in enumerate(query_indices)}
        tc_map = {tc: j for j, tc in enumerate(template_chains)}

        for (qi, tc), hit in best.items():
            cost[qi_map[qi], tc_map[tc]] = -hit.identity

        row_ind, col_ind = linear_sum_assignment(cost)

        alignments: list[ChainAlignment] = []
        for r, c in zip(row_ind, col_ind, strict=True):
            if cost[r, c] >= 1e6:
                continue
            qi = query_indices[r]
            tc = template_chains[c]
            hit = best[(qi, tc)]
            alignments.append(
                ChainAlignment(
                    query_chain_idx=qi,
                    template_chain_id=tc,
                    identity=hit.identity,
                    evalue=hit.evalue,
                    qstart=hit.qstart,
                    qend=hit.qend,
                    tstart=hit.tstart,
                    tend=hit.tend,
                    cigar=hit.cigar,
                )
            )

        if not alignments:
            continue

        n_covered = len(alignments)
        mean_identity = sum(a.identity for a in alignments) / n_covered
        # Score: chain coverage fraction + mean sequence identity
        score = 0.5 * (n_covered / max(num_chains, 1)) + 0.5 * mean_identity

        results.append(
            MultimerTemplateHit(
                template_pdb_id=tmpl_pdb_id,
                chain_alignments=alignments,
                score=score,
            )
        )

    results.sort(key=lambda h: (-h.num_chains_covered, -h.score))
    return results[:max_templates]


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def save_template_hits(hits: list[MultimerTemplateHit], path: Path) -> None:
    """Save multimer template hits to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = []
    for h in hits:
        d = asdict(h)
        d["num_chains_covered"] = h.num_chains_covered
        data.append(d)
    with open(path, "w") as f:
        json.dump(data, f, separators=(",", ":"))


def load_template_hits(path: Path) -> list[MultimerTemplateHit]:
    """Load multimer template hits from a JSON file."""
    with open(path) as f:
        data = json.load(f)
    return [
        MultimerTemplateHit(
            template_pdb_id=d["template_pdb_id"],
            chain_alignments=[ChainAlignment(**a) for a in d["chain_alignments"]],
            score=d["score"],
        )
        for d in data
    ]
