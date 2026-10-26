"""Per-chain template cache (memory LRU + hierarchical sharded disk).

Training-time template extraction used to re-parse the *whole* template
mmCIF with gemmi on every hit (``parse_mmcif_structure`` →
``setup_entities`` over all chains/atoms), then slice out a single chain.
That cost is independent of the query size — a small query that hits a
large template (ribosome, capsid) paid a multi-hundred-ms full parse — and
was a dominant source of periodic dataloader stalls (step-time spikes under
DDP, where the slowest sample in the global batch sets the pace).

This module caches the **query-independent** per-chain intermediate that
template extraction needs::

    ChainTemplate(seqres, struct_seq, lookup)

where ``lookup`` maps each 0-based structure residue index to its
``(aatype, atom_positions[A,3], atom_mask[A])`` features. Only the cheap
query-specific steps (``_realign_mapping`` + ``_fill_template_features``)
run per hit; the gemmi parse happens at most once per template PDB.

Two cache tiers:

1. **Memory LRU** (per worker process): hot chains served without touching
   disk. Keyed by ``(pdb_id, chain_id)``.
2. **Hierarchical sharded disk** (shared across workers / epochs / runs)::

       {root}/{shard(pdb)}/{pdb}/{chain}.npz
       {root}/{shard(pdb)}/{pdb}/_done        # per-PDB "fully parsed" sentinel

   The sentinel makes the disk cache authoritative per PDB: a missing chain
   file with the sentinel present is a *known* miss (chain not in template),
   so we never re-parse. Pre-populate the whole cache with
   ``deepfold3 build-pdb-cache``.

On a cold chain we parse the template once and write **all** its polymer
chains to disk (slice the PDB per chain), so multimer hits and repeated
template PDBs across samples are served from a single parse.
"""

from __future__ import annotations

import logging
import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from deepfold3.data import const, pack
from deepfold3.data.paths import shard

logger = logging.getLogger(__name__)

# Sentinel marking a *known* miss (the chain genuinely does not exist in the
# template, established after a full parse) so we negative-cache instead of
# re-parsing forever. Distinct from ``None`` ("unknown — needs a parse").
_NEG: object = object()

LookupT = dict[int, tuple[int, np.ndarray, np.ndarray]]


@dataclass
class ChainTemplate:
    """Query-independent per-chain template intermediate.

    Attributes:
        seqres: Full entity (SEQRES) one-letter sequence, or ``None`` if the
            entity record is unavailable.
        struct_seq: One-letter sequence of resolved polymer residues
            (``first_conformer`` collapsed); ``len == len(lookup)``.
        lookup: ``{i: (aatype, atom_positions[A,3], atom_mask[A])}`` for each
            0-based structure residue index ``i`` in ``0..len(struct_seq)-1``.
    """

    seqres: str | None
    struct_seq: str
    lookup: LookupT


# --------------------------------------------------------------------------
# Disk (de)serialisation
# --------------------------------------------------------------------------
def chain_cache_path(root: Path, pdb_id: str, chain_id: str) -> Path:
    """Sharded path to one chain's cached npz."""
    pdb = pdb_id.lower()
    return Path(root) / shard(pdb) / pdb / f"{_safe(chain_id)}.npz"


def pdb_done_path(root: Path, pdb_id: str) -> Path:
    """Per-PDB sentinel marking a completed parse (all chains written)."""
    pdb = pdb_id.lower()
    return Path(root) / shard(pdb) / pdb / "_done"


def _safe(chain_id: str) -> str:
    """Make a chain id safe as a path component."""
    return chain_id.replace("/", "_").replace(os.sep, "_") or "_"


def _atomic_write_npz(path: Path, **arrays: np.ndarray) -> None:
    """Write an npz atomically (tmp file + replace) — safe under parallel writers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    try:
        with open(tmp, "wb") as f:
            np.savez(f, **arrays)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def _serialize(ct: ChainTemplate) -> dict[str, np.ndarray]:
    """Pack a ChainTemplate into flat arrays for npz storage.

    ``lookup`` keys are guaranteed contiguous ``0..L-1`` (built via
    ``enumerate`` in :func:`_build_polymer_lookup`), so it serialises to dense
    ``aatype``/``positions``/``mask`` arrays indexed by structure position.
    """
    L = len(ct.struct_seq)
    A = const.MAX_ATOMS_PER_TOKEN
    aatype = np.full(L, 20, dtype=np.int16)
    positions = np.zeros((L, A, 3), dtype=np.float32)
    mask = np.zeros((L, A), dtype=bool)
    for i in range(L):
        entry = ct.lookup.get(i)
        if entry is None:
            continue
        aa, pos, msk = entry
        aatype[i] = aa
        positions[i] = pos
        mask[i] = msk
    return {
        "has_seqres": np.array([1 if ct.seqres is not None else 0], dtype=np.int8),
        "seqres": np.frombuffer((ct.seqres or "").encode("ascii"), dtype=np.uint8).copy(),
        "struct_seq": np.frombuffer(ct.struct_seq.encode("ascii"), dtype=np.uint8).copy(),
        "aatype": aatype,
        "positions": positions,
        "mask": mask,
    }


def _deserialize(data) -> ChainTemplate:
    """Reconstruct a ChainTemplate from a loaded npz mapping."""
    seqres = bytes(data["seqres"].tobytes()).decode("ascii") if int(data["has_seqres"][0]) else None
    struct_seq = bytes(data["struct_seq"].tobytes()).decode("ascii")
    aatype = data["aatype"]
    positions = data["positions"]
    mask = data["mask"]
    lookup: LookupT = {i: (int(aatype[i]), positions[i], mask[i]) for i in range(len(aatype))}
    return ChainTemplate(seqres=seqres, struct_seq=struct_seq, lookup=lookup)


# --------------------------------------------------------------------------
# Parse (slice a template PDB into all its polymer chains)
# --------------------------------------------------------------------------
def extract_chains_from_structure(st, max_atoms: int = const.MAX_ATOMS_PER_TOKEN) -> dict[str, ChainTemplate]:
    """Slice a parsed gemmi Structure into per-chain ChainTemplates.

    Keyed by ``chain.name`` (auth_asym_id) to match
    :func:`deepfold3.data.parse.templates.select_template_chain`. Non-polymer
    and empty chains are skipped.
    """
    from deepfold3.data.parse.templates import (
        _build_polymer_lookup,
        _get_chain_sequences,
    )

    out: dict[str, ChainTemplate] = {}
    model = st[0]
    for ch in model:
        if ch.name in out:
            continue  # first chain with this name wins (matches select_template_chain)
        polymer = ch.get_polymer()
        if not polymer:
            continue
        seqres, struct_seq = _get_chain_sequences(st, ch)
        if not struct_seq:
            continue
        lookup = _build_polymer_lookup(ch, max_atoms)
        out[ch.name] = ChainTemplate(seqres=seqres, struct_seq=struct_seq, lookup=lookup)
    return out


def template_features_from_chain(
    ct: ChainTemplate,
    query_to_seqres: dict[int, int],
    query_length: int,
    max_atoms: int = const.MAX_ATOMS_PER_TOKEN,
    indices_are_struct: bool = False,
):
    """Extract query-aligned ``TemplateFeatures`` from a cached ChainTemplate.

    This is the only per-hit work once the chain is cached: ``_realign_mapping``
    (seqres→structure) + ``_fill_template_features`` (pure numpy). No gemmi.

    Args:
        indices_are_struct: If True, the template indices in ``query_to_seqres``
            are already 0-based **resolved-residue** (structure) positions —
            they index ``ct.struct_seq`` / ``ct.lookup`` directly — so the
            seqres→structure realignment is skipped. This is the case for
            structure-based search hits (e.g. Foldseek), whose alignment is
            computed against the observed-residue sequence rather than SEQRES;
            applying ``_realign_mapping`` to such indices would double-shift and
            corrupt the alignment. Default False keeps the SEQRES-space behaviour
            used by MMseqs2 / HHsearch hits.
    """
    from deepfold3.data.parse.templates import _fill_template_features, _realign_mapping

    if indices_are_struct:
        # Already structure-space: keep only in-range positions (lookup keys are
        # 0..len(struct_seq)-1) and use directly, no realignment.
        n = len(ct.struct_seq)
        mapping = {q: s for q, s in query_to_seqres.items() if 0 <= s < n}
    else:
        mapping = _realign_mapping(ct.seqres, ct.struct_seq, query_to_seqres)
    feats, _ = _fill_template_features(ct.lookup, mapping, query_length, max_atoms)
    return feats


def build_pdb_cache_entry(
    pdb_id: str,
    mmcif_dir: Path,
    cache_root: Path,
    max_atoms: int = const.MAX_ATOMS_PER_TOKEN,
    overwrite: bool = False,
) -> int:
    """Parse one template PDB and write all polymer chains to the disk cache.

    Used by both the offline ``build-pdb-cache`` command and the lazy
    training-time fallback. Idempotent: a present ``_done`` sentinel is
    skipped unless ``overwrite``.

    Returns:
        Number of chains written (0 if the PDB is missing / unparseable /
        already cached).
    """
    from deepfold3.data.parse.templates import parse_mmcif_structure, read_pdb_mmcif_local

    done = pdb_done_path(cache_root, pdb_id)
    if done.exists() and not overwrite:
        return 0

    text = read_pdb_mmcif_local(pdb_id, Path(mmcif_dir))
    if text is None:
        logger.debug("build_pdb_cache_entry: %s not found in mirror", pdb_id)
        return 0
    st = parse_mmcif_structure(text, label=pdb_id)
    if st is None:
        return 0

    chains = extract_chains_from_structure(st, max_atoms)
    for cname, ct in chains.items():
        _atomic_write_npz(chain_cache_path(cache_root, pdb_id, cname), **_serialize(ct))
    # Sentinel last: presence ⇒ every chain that exists has been written.
    done.parent.mkdir(parents=True, exist_ok=True)
    done.touch()
    return len(chains)


# --------------------------------------------------------------------------
# Two-tier cache
# --------------------------------------------------------------------------
class PdbChainCache:
    """Memory-LRU + hierarchical-disk cache of per-chain template intermediates.

    One instance per dataset (and therefore per DataLoader worker process).
    The memory LRU is per-process; the disk cache is shared.

    Args:
        cache_dir: Disk cache root (sharded layout). ``None`` disables the
            disk tier (memory-only, parse-on-miss).
        mmcif_dir: PDB mmCIF mirror for cold-parse fallback. ``None`` disables
            parsing (disk-only — useful when the cache is fully pre-built).
        mem_capacity: Max number of chains held in the memory LRU.
        max_atoms: Atoms per token (feature width).
    """

    def __init__(
        self,
        cache_dir: Path | None,
        mmcif_dir: Path | None,
        mem_capacity: int = 512,
        max_atoms: int = const.MAX_ATOMS_PER_TOKEN,
    ):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.mmcif_dir = Path(mmcif_dir) if mmcif_dir else None
        # A packed cache is read-only: cold parses still fill the memory LRU
        # but are not written back (the pack cannot grow, and writing loose
        # files beside it is exactly the file count packing removes).
        self._disk_writable = self.cache_dir is not None and not pack.register(self.cache_dir)
        pack.register(self.mmcif_dir)
        self.mem_capacity = max(1, int(mem_capacity))
        self.max_atoms = max_atoms
        # (pdb_lower, chain_id) -> ChainTemplate | _NEG
        self._mem: OrderedDict[tuple[str, str], object] = OrderedDict()

    def get(self, pdb_id: str, chain_id: str) -> ChainTemplate | None:
        """Return the ChainTemplate for ``(pdb_id, chain_id)`` or ``None``.

        Resolution order: memory LRU → disk → cold parse (writes disk + mem).
        Known misses are negative-cached so repeated lookups stay cheap.
        """
        key = (pdb_id.lower(), chain_id)
        hit = self._mem.get(key)
        if hit is not None:
            self._mem.move_to_end(key)
            return None if hit is _NEG else hit  # type: ignore[return-value]

        # Disk tier.
        if self.cache_dir is not None:
            status = self._load_disk(pdb_id, chain_id)
            if status is not None:  # HIT or known-MISS
                self._mem_put(key, status)
                return None if status is _NEG else status  # type: ignore[return-value]

        # Cold: parse the whole PDB once, cache every chain.
        chains = self._parse_and_cache(pdb_id)
        if chains is None:
            self._mem_put(key, _NEG)
            return None
        for cname, ct in chains.items():
            self._mem_put((pdb_id.lower(), cname), ct)
        result = chains.get(chain_id)
        if result is None:
            self._mem_put(key, _NEG)
        return result

    def _mem_put(self, key: tuple[str, str], value: object) -> None:
        self._mem[key] = value
        self._mem.move_to_end(key)
        while len(self._mem) > self.mem_capacity:
            self._mem.popitem(last=False)

    def _load_disk(self, pdb_id: str, chain_id: str) -> object | None:
        """Disk lookup. Returns ChainTemplate (hit), ``_NEG`` (known miss via
        sentinel), or ``None`` (unknown — needs a parse)."""
        path = chain_cache_path(self.cache_dir, pdb_id, chain_id)
        if pack.exists(path):
            try:
                # Not ``read_npz``: this tier is re-read constantly (~99.8% hit
                # rate), so its pages must stay cached.
                with np.load(pack.open_binary(path), allow_pickle=False) as data:
                    return _deserialize(data)
            except (OSError, ValueError, EOFError, KeyError) as e:
                logger.debug("Corrupt chain cache %s: %s — reparsing", path, e)
                return None
        if pack.exists(pdb_done_path(self.cache_dir, pdb_id)):
            return _NEG  # PDB parsed, this chain absent → known miss
        return None

    def _parse_and_cache(self, pdb_id: str) -> dict[str, ChainTemplate] | None:
        """Cold-parse a PDB, write all chains to disk, return them.

        ``None`` if the PDB cannot be parsed (missing mirror entry, no mmcif
        dir, or unparseable)."""
        if self.mmcif_dir is None:
            return None
        from deepfold3.data.parse.templates import parse_mmcif_structure, read_pdb_mmcif_local

        text = read_pdb_mmcif_local(pdb_id, self.mmcif_dir)
        if text is None:
            return None
        st = parse_mmcif_structure(text, label=pdb_id)
        if st is None:
            return None
        chains = extract_chains_from_structure(st, self.max_atoms)
        if self._disk_writable:
            for cname, ct in chains.items():
                try:
                    _atomic_write_npz(chain_cache_path(self.cache_dir, pdb_id, cname), **_serialize(ct))
                except OSError as e:
                    logger.debug("Failed to write chain cache for %s/%s: %s", pdb_id, cname, e)
            try:
                done = pdb_done_path(self.cache_dir, pdb_id)
                done.parent.mkdir(parents=True, exist_ok=True)
                done.touch()
            except OSError:
                pass
        return chains
