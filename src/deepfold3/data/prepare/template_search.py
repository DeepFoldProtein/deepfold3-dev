"""Template discovery from m8 hits.

Two parallel pipelines, both fed by per-sequence m8 files (one per unique
chain ``msa_id``):

* **Multimer templates** (``find_multimer_templates``) — one PDB entry that
  covers multiple query chains, with cross-chain pairing via the Hungarian
  algorithm. Stored at ``templates/{ab}/{record_id}.json`` (per record,
  using ``data.paths.template_path``). The release-date cutoff for a
  multimer hit is the **maximum** release date across its constituent
  chains: if any chain was released after the cutoff, the whole template
  is rejected.
* **Monomer templates** (``find_monomer_templates``) — single-chain hits
  per unique sequence, deduplicated by ``msa_id``. Stored at
  ``monomer_templates/{ab}/{msa_id}.json`` (per sequence, using
  ``data.paths.monomer_template_path``). Per-hit filters are applied at
  prepare time; the release-date cutoff is **deferred to training time**
  so the same precomputed JSON serves every record that uses this
  sequence.

Per-hit filters in both pipelines (AlphaFold 3 §2.4):

* ``min_hit_length``: ``tend - tstart + 1 >= 10``
* ``max_subsequence_ratio``: drop near-exact subsequences of the query
  (length_ratio > ratio AND identity ≥ 0.99 — approximated since the m8
  doesn't carry the template sequence letters needed for AF3's exact
  substring check).
* ``min_align_ratio``: aligned residues / **query length** ≥ 0.1
  (matches AF3's ``align_ratio``).
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


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ChainAlignment:
    """Alignment of one query chain to one template chain (multimer pipeline)."""

    query_chain_idx: int  # index in record.chains
    template_chain_id: str  # chain ID in template PDB
    identity: float  # 0..1 fraction
    evalue: float
    qstart: int  # 1-based
    qend: int  # 1-based
    tstart: int  # 1-based
    tend: int  # 1-based
    cigar: str


@dataclass
class MultimerTemplateHit:
    """A multimer template: one PDB entry covering one or more query chains.

    The ``release_date`` is the **maximum** release date across all chains
    used in the alignment (the multimer extension of AF3's per-chain
    cutoff: if any constituent chain was released after the cutoff, the
    whole template is rejected). ``min_evalue`` is the best e-value across
    the constituent chain alignments and is the primary sort key.
    """

    template_pdb_id: str
    chain_alignments: list[ChainAlignment] = field(default_factory=list)
    score: float = 0.0
    min_evalue: float = float("inf")
    release_date: str | None = None

    @property
    def num_chains_covered(self) -> int:
        return len(self.chain_alignments)


@dataclass
class MonomerTemplateHit:
    """A single-chain template hit (one PDB chain → one query chain).

    ``release_date`` is the initial release date of the *template* PDB entry,
    looked up from a precomputed mmCIF metadata cache at prepare time so the
    training-time date-cutoff filter is a cheap string compare.
    """

    template_pdb_id: str
    template_chain_id: str
    identity: float  # 0..1 fraction
    evalue: float
    qstart: int  # 1-based
    qend: int  # 1-based
    tstart: int  # 1-based
    tend: int  # 1-based
    cigar: str
    release_date: str | None = None


# ---------------------------------------------------------------------------
# Per-hit filters (AF3 §2.4)
# ---------------------------------------------------------------------------


def _date_minus_days(date_str: str, days: int) -> str:
    """Subtract ``days`` from an ISO ``YYYY-MM-DD`` date string."""
    import datetime

    d = datetime.date.fromisoformat(date_str)
    return (d - datetime.timedelta(days=days)).isoformat()


def _passes_hit_filters(
    hit: TemplateM8Hit,
    query_length: int,
    *,
    max_subsequence_ratio: float,
    min_hit_length: int,
    min_align_ratio: float,
) -> bool:
    """Per-hit AlphaFold 3 §2.4 ``Hit.keep`` filters.

    * ``min_hit_length``: ``tend - tstart + 1 >= min_hit_length``.
    * ``max_subsequence_ratio``: drop near-exact subsequences of the query
      (leakage guard). AF3 checks ``len(template) / len(query) > ratio``
      AND template is a substring of query — we approximate the substring
      check with ``identity >= 0.99`` since m8 hits don't carry template
      residue letters.
    * ``min_align_ratio``: ``aligned residues / query length >= ratio``,
      matching AF3's ``align_ratio`` (relative to query length).
    """
    hit_length = hit.tend - hit.tstart + 1
    if hit_length < min_hit_length:
        return False

    aligned = hit.qend - hit.qstart + 1
    length_ratio = hit_length / max(query_length, 1)
    if length_ratio > max_subsequence_ratio and hit.identity >= 0.99:
        return False

    align_ratio = aligned / max(query_length, 1)
    if align_ratio < min_align_ratio:
        return False

    return True


def _read_filtered_m8(
    msa_id: str,
    query_length: int,
    m8_dir: Path,
    *,
    max_hits: int,
    max_subsequence_ratio: float,
    min_hit_length: int,
    min_align_ratio: float,
) -> list[TemplateM8Hit]:
    """Read one m8 file and apply per-hit filters, capping at ``max_hits``.

    Shared helper for the multimer and monomer pipelines so the parsing,
    filtering, and per-chain hit cap logic stay in one place.
    """
    m8_path = Path(m8_dir) / f"{msa_id}.m8"
    if not m8_path.exists():
        return []
    out: list[TemplateM8Hit] = []
    for hit in parse_m8_single(m8_path.read_text()):
        if not _passes_hit_filters(
            hit,
            query_length,
            max_subsequence_ratio=max_subsequence_ratio,
            min_hit_length=min_hit_length,
            min_align_ratio=min_align_ratio,
        ):
            continue
        out.append(hit)
        if len(out) >= max_hits:
            break
    return out


# ---------------------------------------------------------------------------
# Multimer template discovery
# ---------------------------------------------------------------------------


def find_multimer_templates(
    record_pdb_id: str,
    chain_msa_ids: list[str],
    m8_dir: Path,
    *,
    chain_lengths: list[int] | None = None,
    release_dates: dict[str, str] | None = None,
    query_release_date: str | None = None,
    max_template_date: str | None = None,
    query_cutoff_days: int = 60,
    max_templates: int = 20,
    max_hits_per_chain: int = 200,
    max_subsequence_ratio: float = 0.95,
    min_hit_length: int = 10,
    min_align_ratio: float = 0.1,
) -> list[MultimerTemplateHit]:
    """Discover multimer templates from per-chain m8 hits, AF3 §2.4 filtered.

    Args:
        record_pdb_id: PDB ID of the training complex (excluded from results).
        chain_msa_ids: ``msa_id`` for each chain in the query (positional).
        m8_dir: Directory containing ``{msa_id}.m8`` files.
        chain_lengths: Per-chain query length (residue count). Used by the
            per-hit filters; defaults to ``[1] * num_chains`` which makes the
            length/align/subsequence filters effectively trivial.
        release_dates: Mapping ``{template_pdb_id_uppercase → ISO date}``
            (typically the JSON written by ``deepfold3 scan-mmcif``). When
            provided alongside any of the date-cutoff knobs, templates whose
            **maximum** release date across the matched chains exceeds the
            effective cutoff are dropped.
        query_release_date: Initial release date of the query (training)
            example, ISO ``YYYY-MM-DD``. Templates released within
            ``query_cutoff_days`` of this date are dropped.
        max_template_date: Global hard cap on template release dates.
        query_cutoff_days: Days a template must precede the query release
            (AF3 default 60).
        max_templates: Maximum number of templates returned (AF3 default 20).
        max_hits_per_chain: m8 hits considered per chain after filtering.
        max_subsequence_ratio, min_hit_length, min_align_ratio: AF3 §2.4
            per-hit filter parameters.

    Returns:
        Up to ``max_templates`` ``MultimerTemplateHit`` objects, sorted by
        ``min_evalue`` ascending then chain coverage descending.
    """
    from scipy.optimize import linear_sum_assignment

    m8_dir = Path(m8_dir)
    num_chains = len(chain_msa_ids)
    if chain_lengths is None:
        chain_lengths = [1] * num_chains

    # Effective release-date cutoff: min(max_template_date, query - N days)
    cutoff: str | None = max_template_date
    if query_release_date is not None:
        query_minus = _date_minus_days(query_release_date, query_cutoff_days)
        cutoff = query_minus if cutoff is None else min(cutoff, query_minus)

    # Read + filter m8 hits per chain (cached so identical sequences are
    # only parsed once even if they occur as multiple chains).
    filtered_cache: dict[tuple[str, int], list[TemplateM8Hit]] = {}
    hits_by_pdb: dict[str, list[tuple[int, TemplateM8Hit]]] = defaultdict(list)
    for chain_idx, msa_id in enumerate(chain_msa_ids):
        query_length = chain_lengths[chain_idx]
        cache_key = (msa_id, query_length)
        if cache_key not in filtered_cache:
            filtered_cache[cache_key] = _read_filtered_m8(
                msa_id,
                query_length,
                m8_dir,
                max_hits=max_hits_per_chain,
                max_subsequence_ratio=max_subsequence_ratio,
                min_hit_length=min_hit_length,
                min_align_ratio=min_align_ratio,
            )
        for hit in filtered_cache[cache_key]:
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

        # Multimer date cutoff: take the MAX release date across the matched
        # chains, then compare to the effective cutoff.
        release_date: str | None = None
        if release_dates is not None:
            chain_dates = []
            for _ in alignments:
                d = release_dates.get(tmpl_pdb_id) or release_dates.get(tmpl_pdb_id.lower())
                if d is not None:
                    chain_dates.append(d)
            release_date = max(chain_dates) if chain_dates else None
            if cutoff is not None and (release_date is None or release_date > cutoff):
                continue

        n_covered = len(alignments)
        mean_identity = sum(a.identity for a in alignments) / n_covered
        score = 0.5 * (n_covered / max(num_chains, 1)) + 0.5 * mean_identity
        min_evalue = min(a.evalue for a in alignments)

        results.append(
            MultimerTemplateHit(
                template_pdb_id=tmpl_pdb_id,
                chain_alignments=alignments,
                score=score,
                min_evalue=min_evalue,
                release_date=release_date,
            )
        )

    results.sort(key=lambda h: (h.min_evalue, -h.num_chains_covered, -h.score))
    return results[:max_templates]


# ---------------------------------------------------------------------------
# Monomer template discovery
# ---------------------------------------------------------------------------


def find_monomer_templates(
    msa_id: str,
    query_length: int,
    m8_dir: Path,
    *,
    release_dates: dict[str, str] | None = None,
    self_pdb_ids: set[str] | None = None,
    max_templates: int = 20,
    max_hits_per_chain: int = 200,
    max_subsequence_ratio: float = 0.95,
    min_hit_length: int = 10,
    min_align_ratio: float = 0.1,
) -> list[MonomerTemplateHit]:
    """Discover single-chain templates for one query sequence.

    The result is independent of any specific record — date-cutoff filtering
    is intentionally **deferred to training time** so the same precomputed
    list can serve every record that uses this sequence.

    Args:
        msa_id: SHA256 hash of the chain sequence (the m8 filename stem).
        query_length: Number of residues in the query chain.
        m8_dir: Directory containing ``{msa_id}.m8`` files.
        release_dates: Optional mapping ``{template_pdb_id → ISO date}`` used
            to annotate each hit with its release date so the dataloader can
            apply the cutoff at training time without re-reading the cache.
        self_pdb_ids: Optional set of uppercase PDB IDs to exclude (e.g. the
            query's own complex). When unset, no self-exclusion is applied
            since the same sequence can legitimately appear in many records.
        max_templates: Maximum templates returned (AF3 default 20).
        max_hits_per_chain: m8 hits considered after per-hit filtering.
        max_subsequence_ratio, min_hit_length, min_align_ratio: AF3 §2.4
            per-hit filter parameters.

    Returns:
        Up to ``max_templates`` ``MonomerTemplateHit`` objects, sorted by
        e-value ascending. ``release_date`` is populated when
        ``release_dates`` is provided.
    """
    raw_hits = _read_filtered_m8(
        msa_id,
        query_length,
        m8_dir,
        max_hits=max_hits_per_chain,
        max_subsequence_ratio=max_subsequence_ratio,
        min_hit_length=min_hit_length,
        min_align_ratio=min_align_ratio,
    )
    out: list[MonomerTemplateHit] = []
    for hit in raw_hits:
        if self_pdb_ids is not None and hit.pdb_id.upper() in self_pdb_ids:
            continue
        rd: str | None = None
        if release_dates is not None:
            rd = release_dates.get(hit.pdb_id) or release_dates.get(hit.pdb_id.lower())
        out.append(
            MonomerTemplateHit(
                template_pdb_id=hit.pdb_id,
                template_chain_id=hit.chain_id,
                identity=hit.identity,
                evalue=hit.evalue,
                qstart=hit.qstart,
                qend=hit.qend,
                tstart=hit.tstart,
                tend=hit.tend,
                cigar=hit.cigar,
                release_date=rd,
            )
        )
    out.sort(key=lambda h: h.evalue)
    return out[:max_templates]


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
            score=d.get("score", 0.0),
            min_evalue=d.get("min_evalue", float("inf")),
            release_date=d.get("release_date"),
        )
        for d in data
    ]


def save_monomer_template_hits(hits: list[MonomerTemplateHit], path: Path) -> None:
    """Save per-sequence monomer template hits to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump([asdict(h) for h in hits], f, separators=(",", ":"))


def load_monomer_template_hits(path: Path) -> list[MonomerTemplateHit]:
    """Load per-sequence monomer template hits from a JSON file."""
    with open(path) as f:
        data = json.load(f)
    return [MonomerTemplateHit(**d) for d in data]
