"""Weighted-PDB training dataset backed by OpenFold 3's ``pdb_training_set``.

Reads the upstream artefacts **in place** — no bulk re-preprocessing — and feeds
them through DeepFold3's own tokenize → crop → featurize pipeline:

===================================================  =========================
``preprocessed_pdb_data/standard/structure_files/``  biotite ``AtomArray`` NPZ
``alignment_arrays/{rep}.npz``                       per-source MSA blocks
``templates/train_template_cache/{rep}.npz``         per-query template hits
``templates/template_structure_arrays/{pdb}/``       template chain geometry
``dataset_caches/*.json`` (converted)                clusters / dates / links
===================================================  =========================

The dataset cache is **not** read as JSON here: it is converted once, offline,
into the memory-mapped columnar cache of :mod:`deepfold3.data.prepare.of3_pdb_cache`
(``deepfold3 build-of3-cache``). That is what keeps a 1.7 GB metadata blob out of
every rank and worker without pulling in LMDB.

Sampling is AF3 §2.5.1, computed directly on the cache's columns
(:func:`of3_anchor_weights`): the sampler draws a *structure* by the cluster
marginal and this dataset draws that structure's chain / interface crop **anchor**
at load time — the same two-stage decomposition ``data/sample/weighted.py``
applies to ``format: df3`` records.

Only the *conversion logic* is original — no OpenFold 3 code is copied. The file
layout and field names are treated as a data format specification.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.utils.data

from deepfold3.data import const
from deepfold3.data.crop.cropper import Cropper, compute_chain_msa_cols, remap_msa_col_idx_to_slice
from deepfold3.data.feature.featurizer import Featurizer
from deepfold3.data.io_cache import read_npz
from deepfold3.data.module.dataset import _count_drop, _per_sample_rng, _retry_getitem
from deepfold3.data.prepare.of3_pdb_cache import OF3PdbCache
from deepfold3.data.prepare.openfold3 import convert_of3_atom_array
from deepfold3.data.tokenize.atomize import maybe_atomize
from deepfold3.data.tokenize.tokenizer import Tokenizer
from deepfold3.data.types import MSA, Input

if TYPE_CHECKING:
    from deepfold3.data.parse.templates import TemplateFeatures

logger = logging.getLogger(__name__)

# Vertical concatenation order of the per-database MSA blocks, and the block used
# for cross-chain pairing. Mirrors OF3 ``MSASettings.aln_order`` /
# ``msas_to_pair``; keys absent from a given alignment file are skipped.
ALN_ORDER: tuple[str, ...] = (
    "uniref90_hits",
    "bfd_uniclust_hits",
    "bfd_uniref_hits",
    "cfdb_uniref30",
    "cfdb_hits",
    "mgnify_hits",
    "rfam_hits",
    "rnacentral_hits",
    "nt_hits",
    "nucleotide_collection_hits",
    "concat_cfdb_uniref100_filtered",
    "mmseqs_colabfold",
    "colabfold_main",
)
PAIRING_KEYS: tuple[str, ...] = ("uniprot_hits", "uniprot")

# AF3 §2.5.1 / OF3 ``sample_weights`` defaults, kept here so the weights this
# module computes match ``data/sample/weighted.py``'s Record-based path exactly.
ALPHA_PROT, ALPHA_NUCL, ALPHA_LIG = 3.0, 3.0, 1.0
BETA_CHAIN, BETA_INTERFACE = 0.5, 1.0


def _char_table(mol_type: int) -> np.ndarray:
    """ASCII byte → MSA class index lookup for one molecule type."""
    table = np.full(256, const.MSA_UNK_NUCLEIC if mol_type != const.MOL_PROTEIN else 20, dtype=np.int8)
    mapping = {
        const.MOL_PROTEIN: const.PROTEIN_MSA_ID,
        const.MOL_RNA: const.RNA_MSA_ID,
        const.MOL_DNA: const.DNA_MSA_ID,
    }.get(mol_type, const.PROTEIN_MSA_ID)
    for ch, idx in mapping.items():
        table[ord(ch)] = idx
        if ch.isalpha():
            table[ord(ch.lower())] = idx
    table[ord("-")] = const.MSA_GAP
    table[ord(".")] = const.MSA_GAP
    return table


_CHAR_TABLES: dict[int, np.ndarray] = {m: _char_table(m) for m in (const.MOL_PROTEIN, const.MOL_RNA, const.MOL_DNA)}


def _encode_msa(chars: np.ndarray, mol_type: int) -> np.ndarray:
    """``<U1`` [S, N] alignment characters → int8 MSA class indices."""
    table = _CHAR_TABLES.get(mol_type, _CHAR_TABLES[const.MOL_PROTEIN])
    byte_view = chars.astype("S1").view(np.uint8).reshape(chars.shape)
    return table[byte_view]


# ---------------------------------------------------------------------------
# AF3 §2.5.1 weights, straight off the columnar cache
# ---------------------------------------------------------------------------


def of3_anchor_weights(
    cache: OF3PdbCache,
    *,
    keep: np.ndarray | None = None,
    alpha_prot: float = ALPHA_PROT,
    alpha_nucl: float = ALPHA_NUCL,
    alpha_lig: float = ALPHA_LIG,
    beta_chain: float = BETA_CHAIN,
    beta_interface: float = BETA_INTERFACE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-anchor and per-structure AF3 §2.5.1 sampling weights.

    Vectorised over the cache's columns — no ``Record`` objects are built, so a
    200k-structure set costs a few array passes instead of minutes of Python
    object construction on every rank.

    ``N_clust`` is recomputed **here** rather than read from the cache's upstream
    ``cluster_size``: when ``keep`` excludes structures (a release-date cutoff),
    the surviving cluster occupancy is what AF3 balances against.

    Args:
        cache: Opened columnar cache.
        keep: Optional bool [S] mask of structures to sample from. Excluded
            structures get weight 0 and are left out of the cluster counts.
        alpha_prot, alpha_nucl, alpha_lig: Molecule-type multipliers.
        beta_chain, beta_interface: Chain vs interface balance.

    Returns:
        ``(w_chain [C], w_iface [I], w_struct [S])``. ``w_struct`` is normalised
        (the sampler's record marginal); the two anchor arrays are unnormalised
        ``w_a`` in the same units, so a per-structure slice of them is the anchor
        conditional.
    """
    n_struct = len(cache)
    chain_off = np.asarray(cache.struct_chain_off)
    iface_off = np.asarray(cache.struct_iface_off)
    chain_counts = np.diff(chain_off)
    iface_counts = np.diff(iface_off)

    if keep is None:
        keep = np.ones(n_struct, dtype=bool)
    keep = np.asarray(keep, dtype=bool)
    chain_keep = np.repeat(keep, chain_counts)
    iface_keep = np.repeat(keep, iface_counts)

    mol = np.asarray(cache.chain_moltype)
    alpha = np.full(mol.shape, alpha_lig, dtype=np.float64)
    alpha[mol == const.MOL_PROTEIN] = alpha_prot
    alpha[(mol == const.MOL_RNA) | (mol == const.MOL_DNA)] = alpha_nucl

    # Chain clusters: N_clust = number of *kept* chains in the same cluster.
    cluster = np.asarray(cache.chain_cluster).astype(np.int64)
    _, cluster_idx = np.unique(cluster, return_inverse=True)
    cluster_size = np.bincount(cluster_idx[chain_keep], minlength=cluster_idx.max() + 1 if len(cluster_idx) else 1)
    n_clust_chain = np.maximum(cluster_size[cluster_idx], 1)
    w_chain = np.where(chain_keep, beta_chain * alpha / n_clust_chain, 0.0)

    # Interface clusters: AF3 §2.5.3 — the unordered pair of chain clusters.
    if len(iface_off) > 1 and iface_off[-1] > 0:
        struct_of_iface = np.repeat(np.arange(n_struct, dtype=np.int64), iface_counts)
        base = chain_off[struct_of_iface]
        row_a = base + np.asarray(cache.iface_a)
        row_b = base + np.asarray(cache.iface_b)
        icid = cache.interface_cluster_codes()
        _, icid_idx = np.unique(icid, return_inverse=True)
        icid_size = np.bincount(icid_idx[iface_keep], minlength=icid_idx.max() + 1)
        n_clust_iface = np.maximum(icid_size[icid_idx], 1)
        alpha_sum = alpha[row_a] + alpha[row_b]
        w_iface = np.where(iface_keep, beta_interface * alpha_sum / n_clust_iface, 0.0)
    else:
        w_iface = np.zeros(0, dtype=np.float64)

    w_struct = np.zeros(n_struct, dtype=np.float64)
    if len(w_chain):
        np.add.at(w_struct, np.repeat(np.arange(n_struct, dtype=np.int64), chain_counts), w_chain)
    if len(w_iface):
        np.add.at(w_struct, np.repeat(np.arange(n_struct, dtype=np.int64), iface_counts), w_iface)
    total = w_struct.sum()
    w_struct = w_struct / total if total > 0 else np.full(n_struct, 1.0 / max(n_struct, 1))
    return w_chain, w_iface, w_struct


# ---------------------------------------------------------------------------
# Alignment arrays
# ---------------------------------------------------------------------------


def read_of3_alignment(npz_path: Path, keys: frozenset[str] | None = None) -> dict[str, dict]:
    """Load one ``alignment_arrays/{rep}.npz`` into ``{source: {msa, del, meta}}``.

    Each key is one search database. **Every** block repeats the query as its
    row 0, so the blocks cannot simply be stacked — see :func:`build_chain_msa`.

    ``keys`` restricts which blocks are decompressed. The members are DEFLATE'd
    inside the NPZ and ``np.load`` decodes lazily, so skipping the 50k-row
    ``uniprot_hits`` block on a monomer (which never pairs) is most of the read
    cost avoided, not a marginal saving.
    """
    data = read_npz(npz_path, allow_pickle=True)
    out: dict[str, dict] = {}
    for key in data.keys():
        if keys is not None and str(key) not in keys:
            continue
        item = data[key].item()
        msa = item.get("msa")
        if msa is None or msa.size == 0:
            continue
        out[str(key)] = {
            "msa": msa,
            "deletion_matrix": item.get("deletion_matrix"),
            "metadata": item.get("metadata"),
        }
    return out


_MAIN_KEYS = frozenset(ALN_ORDER)
_MAIN_AND_PAIRING_KEYS = _MAIN_KEYS | frozenset(PAIRING_KEYS)


def build_chain_msa(
    blocks: dict[str, dict],
    mol_type: int,
    max_seqs: int,
    cols: np.ndarray | None = None,
) -> MSA | None:
    """Assemble one chain's unpaired MSA from its per-database blocks.

    Follows OF3 ``create_main``: concatenate the blocks in :data:`ALN_ORDER`,
    drop duplicate rows (order-stable), and keep the query as row 0 exactly once
    — every block carries its own copy of it. ``cols`` slices the alignment to
    the crop's surviving chain-local columns before any of the row work, so a
    50k-row UniProt block never gets materialised at full width.
    """
    ordered = [blocks[k] for k in ALN_ORDER if k in blocks]
    if not ordered:
        ordered = [blocks[k] for k in sorted(blocks) if k not in PAIRING_KEYS]
    if not ordered:
        return None

    def _slice(arr: np.ndarray) -> np.ndarray:
        return arr if cols is None else arr[:, cols]

    seqs = [_encode_msa(_slice(b["msa"]), mol_type) for b in ordered]
    dels = [
        np.zeros(s.shape, dtype=np.int8)
        if b["deletion_matrix"] is None
        else np.clip(_slice(b["deletion_matrix"]), 0, 127).astype(np.int8)
        for s, b in zip(seqs, ordered, strict=True)
    ]

    query = seqs[0][:1]
    query_del = dels[0][:1]
    body = np.concatenate([s[1:] for s in seqs], axis=0) if len(seqs) > 1 or seqs[0].shape[0] > 1 else seqs[0][:0]
    body_del = np.concatenate([d[1:] for d in dels], axis=0) if len(dels) > 1 or dels[0].shape[0] > 1 else dels[0][:0]

    if body.shape[0] > 1:
        view = np.ascontiguousarray(body).view(np.dtype((np.void, body.dtype.itemsize * body.shape[1])))
        _, uniq = np.unique(view, return_index=True)
        uniq.sort()
        body, body_del = body[uniq], body_del[uniq]

    if max_seqs > 0 and body.shape[0] > max_seqs - 1:
        body, body_del = body[: max_seqs - 1], body_del[: max_seqs - 1]

    return MSA(
        sequences=np.concatenate([query, body], axis=0),
        mask=np.ones((1 + body.shape[0], query.shape[1]), dtype=bool),
        deletion_matrix=np.concatenate([query_del, body_del], axis=0),
    )


def _species_ids(metadata: np.ndarray | None) -> np.ndarray | None:
    """Species token of each UniProt header (``tr|ID|ID_SPECIES/start-end``).

    Row 0 is the query and has no species; it comes back as ``""``. Mirrors OF3
    ``process_msa_pairing_metadata``, which splits on ``[|_/:-]`` and takes the
    4th field.
    """
    if metadata is None or metadata.size == 0:
        return None
    import re

    pattern = re.compile(r"[|_/:\-]")
    out = np.empty(metadata.shape[0], dtype=object)
    for i, raw in enumerate(metadata.astype(str)):
        parts = pattern.split(raw)
        out[i] = parts[3] if len(parts) > 3 else ""
    out[0] = ""
    return out


def pair_uniprot_msas(
    per_rep: dict[str, dict],
    mol_types: dict[str, int],
    cols_per_rep: dict[str, np.ndarray | None],
    max_rows: int = const.MAX_PAIRED_MSA_ROWS,
    max_per_species: int = 600,
) -> dict[str, MSA]:
    """Species-pair the UniProt blocks of a complex's alignment representatives.

    AF2-Multimer style, as AF3 §2.3 keeps it: within each species, the *r*-th hit
    of one chain pairs with the *r*-th hit of every other chain (the blocks are
    already in search-rank order). Species seen by only one representative are
    dropped (``min_chains_paired_partial=2``), and rows covering more chains sort
    first so the depth budget goes to the fullest pairs.

    Returns row-aligned per-representative :class:`MSA` blocks — the featurizer
    zips them by row index, so every returned block has the same depth. ``{}``
    when fewer than two representatives have a pairable block.
    """
    reps = [r for r, blocks in per_rep.items() if any(k in blocks for k in PAIRING_KEYS)]
    if len(reps) < 2:
        return {}

    # species → {rep: [row indices]}
    by_species: dict[str, dict[str, list[int]]] = {}
    raw: dict[str, dict] = {}
    for rep in reps:
        block = next(per_rep[rep][k] for k in PAIRING_KEYS if k in per_rep[rep])
        species = _species_ids(block.get("metadata"))
        if species is None:
            continue
        raw[rep] = block
        for row in range(1, len(species)):
            sp = species[row]
            if not sp:
                continue
            slot = by_species.setdefault(sp, {}).setdefault(rep, [])
            if len(slot) < max_per_species:
                slot.append(row)
    if len(raw) < 2:
        return {}

    # (n_chains_covered, -rank) ordered pair rows: each entry maps rep → row or -1.
    candidates: list[tuple[int, int, dict[str, int]]] = []
    for _sp, per in by_species.items():
        present = {r: rows for r, rows in per.items() if r in raw}
        if len(present) < 2:
            continue
        depth = max(len(rows) for rows in present.values())
        for rank in range(depth):
            assign = {r: rows[rank] for r, rows in present.items() if rank < len(rows)}
            if len(assign) < 2:
                continue
            candidates.append((len(assign), rank, assign))
    if not candidates:
        return {}

    candidates.sort(key=lambda t: (-t[0], t[1]))
    candidates = candidates[:max_rows]

    out: dict[str, MSA] = {}
    n_rows = len(candidates)
    for rep, block in raw.items():
        cols = cols_per_rep.get(rep)
        mol_type = mol_types.get(rep, const.MOL_PROTEIN)
        src = block["msa"] if cols is None else block["msa"][:, cols]
        src_del = block["deletion_matrix"]
        if src_del is not None and cols is not None:
            src_del = src_del[:, cols]
        rows = np.array([a.get(rep, -1) for _, _, a in candidates], dtype=np.int64)
        width = src.shape[1]
        seqs = np.full((n_rows, width), const.MSA_GAP, dtype=np.int8)
        dels = np.zeros((n_rows, width), dtype=np.int8)
        hit = rows >= 0
        if hit.any():
            seqs[hit] = _encode_msa(src[rows[hit]], mol_type)
            if src_del is not None:
                dels[hit] = np.clip(src_del[rows[hit]], 0, 127).astype(np.int8)
        out[rep] = MSA(sequences=seqs, mask=np.ones((n_rows, width), dtype=bool), deletion_matrix=dels)
    return out


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------


def load_of3_template_hits(npz_path: Path) -> dict[str, dict]:
    """Load one ``train_template_cache/{rep}.npz``.

    Keys are ``{pdb_id}_{chain_id}``; each value carries ``release_date`` (str),
    ``index`` (search rank) and ``idx_map`` [K, 2] int — **1-based query position**
    paired with the template chain's ``res_id``.
    """
    data = read_npz(npz_path, allow_pickle=True)
    out: dict[str, dict] = {}
    for key in data.keys():
        item = data[key].item()
        if item.get("idx_map") is None or len(item["idx_map"]) == 0:
            continue
        out[str(key)] = item
    return out


def template_residue_lookup(npz_path: Path, max_atoms: int = const.MAX_ATOMS_PER_TOKEN) -> dict[int, tuple]:
    """``res_id`` → ``(aatype, positions [max_atoms, 3], mask [max_atoms])``.

    Reads a ``template_structure_arrays/{pdb}/{pdb}_{chain}.npz`` — the template
    chain rendered over its **full SEQRES**, unresolved residues carrying NaN
    coordinates and zero occupancy. Because the array is already SEQRES-indexed
    by ``res_id``, the hit's ``idx_map`` addresses it directly: no local mmCIF
    mirror and no seqres→structure realignment (the two things the ``format: df3``
    template path needs, and the usual source of silently-misaligned templates).
    """
    from deepfold3.data.parse.templates import _get_canonical_atoms

    data = read_npz(npz_path)
    coord = np.asarray(data["coord"], dtype=np.float32)
    res_id = np.asarray(data["res_id"]).astype(np.int64)
    res_name = np.char.strip(data["res_name"].astype(str))
    atom_name = np.char.strip(data["atom_name"].astype(str))
    occ = np.asarray(data["occupancy"], dtype=np.float64) if "occupancy" in data else np.ones(len(res_id))
    resolved = (occ > 0.0) & np.isfinite(coord).all(axis=1)

    starts = np.flatnonzero(np.r_[True, res_id[1:] != res_id[:-1]]) if len(res_id) else np.zeros(0, dtype=np.int64)
    ends = np.append(starts[1:], len(res_id))

    lookup: dict[int, tuple] = {}
    for s, e in zip(starts, ends, strict=True):
        if not resolved[s:e].any():
            continue
        name = str(res_name[s])
        slots = {a: i for i, a in enumerate(_get_canonical_atoms(name)[:max_atoms])}
        pos = np.zeros((max_atoms, 3), dtype=np.float32)
        mask = np.zeros(max_atoms, dtype=bool)
        for a in range(s, e):
            if not resolved[a]:
                continue
            slot = slots.get(str(atom_name[a]))
            if slot is not None:
                pos[slot] = coord[a]
                mask[slot] = True
        if mask.any():
            lookup[int(res_id[s])] = (const.res_name_to_type(name), pos, mask)
    return lookup


def _msa_cols_from_res_id(tokenized):
    """Index each token's MSA column by ``res_id - 1``, as OpenFold 3 does::

        msa_column_positions = msa_token_mapper.res_id - 1
        msa_array = msa_array_vstack.msa[:, msa_column_positions]

    The tokenizer's own ``msa_col_idx`` is a running per-chain counter. That is
    correct for ``format: df3`` (residue order defines the columns) and, for an
    OF3 chain whose ``res_id`` runs 1..N without duplicates, produces *exactly
    the same* indices — so this is a no-op on 991 of 996 chains sampled.

    It differs at **microheterogeneity**: two residues modelled at one SEQRES
    position. 3l4j res_id 362 carries PTR and TYR, 4gfh res_id 782 the same,
    6gpv has two such sites in one chain. Those are one alignment column, and
    OF3 treats them so (it emits PTR atomized plus TYR standard, all sharing the
    res_id). The running counter instead advances twice and walks off the end of
    the alignment — ``index 757 is out of bounds for axis 1 with size 757`` — which
    escaped a dataloader worker and killed a 112-rank job.

    Measured over 996 chains: ``max(res_id) <= width`` holds 100% of the time,
    ``n_residues <= width`` only 99.50%.

    Applied straight after tokenization so the crop's ``query_col_idx`` — which
    places template features, and which OF3's template ``idx_map`` also indexes
    by SEQRES position — inherits the same convention.
    """
    from dataclasses import replace

    col = tokenized.msa_col_idx
    if col is None:
        return tokenized
    # Ligand / column-less tokens keep -1; polymer tokens take their SEQRES
    # position. res_id is 1-based in OF3's preprocessed structures.
    rebased = np.where(col >= 0, tokenized.tokens["residue_index"].astype(np.int64) - 1, -1)
    return replace(tokenized, msa_col_idx=rebased.astype(col.dtype))


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class OF3PdbDataset(torch.utils.data.Dataset):
    """AF3 weighted-PDB dataset over an OpenFold 3 ``pdb_training_set`` tree.

    Args:
        cache: Columnar dataset cache (:class:`OF3PdbCache`).
        root: ``pdb_training_set`` root. Every sub-path below defaults to the
            upstream layout under it and can be overridden individually.
        indices: Optional subset of structure indices into ``cache`` (a
            release-date cutoff filter). ``None`` = all.
        anchor_sampling: Draw the crop anchor from the AF3 §2.5.1 conditional at
            load time (weighted datasets). ``False`` lets the cropper self-seed.
        weights: Precomputed ``(w_chain, w_iface)``. Reuses the arrays the
            sampler already built instead of recomputing them per dataset.
        max_templates: Per-sample template cap (AF3 §2.4).
        max_template_date: Hard ISO cap on template release dates. Fail-closed:
            ``None`` drops **all** templates rather than risk leaking a future
            one, matching the ``format: df3`` path.
        template_cutoff_days: Extra per-query tightening — a hit must also be at
            least this many days older than the query's own release (AF3 §2.4
            uses 60).
    """

    def __init__(
        self,
        cache: OF3PdbCache,
        root: Path,
        *,
        indices: np.ndarray | None = None,
        structures_dir: Path | None = None,
        alignment_dir: Path | None = None,
        template_cache_dir: Path | None = None,
        template_array_dir: Path | None = None,
        max_tokens: int = 384,
        max_atoms: int = 4096,
        max_msa_seqs: int = const.MAX_MSA_ROWS,
        max_chains: int = 0,
        max_neighborhood: int = 40,
        contiguous_crop_weight: float = 0.2,
        spatial_crop_weight: float = 0.4,
        interface_crop_weight: float = 0.4,
        interface_distance: float = 15.0,
        max_templates: int = 4,
        max_template_date: str | None = None,
        template_cutoff_days: int = 60,
        use_paired_msa: bool = True,
        is_distillation: bool = False,
        name: str = "",
        exclude_loss_mol_types: tuple[int, ...] | list[int] = (),
        atomize_prob: float = 0.0,
        atomize_source: str = "random",
        atomize_max_residues: int = 0,
        msa_crop_size: int = 1024,
        n_msa_crops: int = 0,
        anchor_sampling: bool = False,
        pad_templates: bool = False,
        weights: tuple[np.ndarray, np.ndarray] | None = None,
        curator_weights: Path | str | None = None,
    ):
        self.cache = cache
        self.root = Path(root)
        self.indices = None if indices is None else np.asarray(indices, dtype=np.int64)
        self.structures_dir = Path(
            structures_dir or self.root / "preprocessed_pdb_data" / "standard" / "structure_files"
        )
        self.alignment_dir = Path(alignment_dir or self.root / "alignment_arrays")
        # Upstream splits the per-query template hits by subset. Fall back to the
        # validation cache when the training one is absent so an eval dataset
        # pointed at the same root still gets templates instead of silently none.
        if template_cache_dir is None:
            template_cache_dir = self.root / "templates" / "train_template_cache"
            if not template_cache_dir.is_dir():
                alt = self.root / "templates" / "val_template_cache"
                if alt.is_dir():
                    template_cache_dir = alt
        self.template_cache_dir = Path(template_cache_dir)
        self.template_array_dir = Path(template_array_dir or self.root / "templates" / "template_structure_arrays")

        self.name = name
        self.max_tokens = max_tokens
        self.max_atoms = max_atoms
        self.max_msa_seqs = max_msa_seqs
        self.max_chains = max_chains
        self.max_templates = max_templates
        self.max_template_date = max_template_date or None
        self.template_cutoff_days = int(template_cutoff_days)
        self.use_paired_msa = use_paired_msa
        self.is_distillation = is_distillation
        self.exclude_loss_mol_types = tuple(int(m) for m in (exclude_loss_mol_types or ()))
        self.atomize_prob = float(atomize_prob)
        self.atomize_source = atomize_source
        self.atomize_max_residues = int(atomize_max_residues)

        self._anchor_sampling = anchor_sampling
        w_chain, w_iface, w_struct = of3_anchor_weights(cache, keep=self._keep_mask())
        # Per-item draw distribution over *this dataset's* items, handed to
        # ``MultiDatasetSampler`` via its ``weights`` key. Renormalised over the
        # kept subset so it sums to 1 after a cutoff filter drops structures.
        sel = w_struct if self.indices is None else w_struct[self.indices]
        if curator_weights:
            sel = sel * self._curator_multiplier(curator_weights, name)
        total = float(sel.sum())
        self.sampling_weights = sel / total if total > 0 else np.full(len(self), 1.0 / max(len(self), 1))
        if anchor_sampling:
            self._w_chain, self._w_iface = (w_chain, w_iface) if weights is None else weights
        else:
            self._w_chain = self._w_iface = None

        self.tokenizer = Tokenizer()
        self.cropper = Cropper(
            max_neighborhood=max_neighborhood,
            contiguous_weight=contiguous_crop_weight,
            spatial_weight=spatial_crop_weight,
            interface_weight=interface_crop_weight,
            interface_distance=interface_distance,
        )
        self.featurizer = Featurizer(
            max_msa_seqs=max_msa_seqs,
            max_templates=max_templates,
            msa_crop_size=msa_crop_size,
            n_msa_crops=n_msa_crops,
            pad_templates=pad_templates,
        )
        # Templates need BOTH halves: the per-query hit lists and the template
        # chain geometry they point at. Checking only the hit lists made a
        # half-synced mirror (a very live failure mode — the tree is ~TB and
        # `aws s3 sync` lands directories in whatever order) look exactly like a
        # deliberately template-free run: every hit resolved to a missing array
        # file, got skipped at debug level, and training quietly proceeded at
        # T=0 with nothing in the log to say so. Fail loud once, up front.
        self._has_templates = cache.has_templates and self.template_cache_dir.is_dir()
        if self._has_templates and not self.template_array_dir.is_dir():
            logger.warning(
                "%s: template hits are present (%s) but the template geometry directory "
                "%s does not exist — TEMPLATES ARE DISABLED for this dataset. Sync "
                "templates/template_structure_arrays/ (or pass template_array_dir) to "
                "enable them.",
                name or "of3pdb",
                self.template_cache_dir,
                self.template_array_dir,
            )
            self._has_templates = False
        elif not self._has_templates and cache.has_templates:
            logger.warning(
                "%s: dataset cache lists template hits but %s does not exist — "
                "templates are disabled for this dataset.",
                name or "of3pdb",
                self.template_cache_dir,
            )

    def _curator_multiplier(self, path: Path | str, name: str) -> np.ndarray:
        """Per-item multiplier on the AF3 cluster marginal, keyed by ``pdb_id``.

        The curated sampler is exactly ``w_i^AF3 · m_i``: the cluster balancing
        AF3 §2.5.1 prescribes is left untouched and the model-aware term rides
        on top, so the ``m ≡ 1`` arm IS the baseline (no separate code path to
        drift). Entries the file does not mention keep ``m = 1`` — a partial
        scoring pass degrades toward baseline rather than silently zeroing the
        unscored majority.
        """
        z = np.load(Path(path), allow_pickle=False)
        lut = dict(zip((str(x) for x in z["pdb_id"]), z["weight"].astype(np.float64), strict=True))
        idx = self.indices if self.indices is not None else np.arange(len(self.cache))
        mult = np.ones(len(idx), dtype=np.float64)
        hit = 0
        for j, si in enumerate(idx):
            w = lut.get(self.cache.pdb_id(int(si)))
            if w is not None:
                mult[j] = w
                hit += 1
        logger.info(
            "%s: curator weights from %s — %d/%d items covered (%.1f%%), multiplier range [%.3f, %.3f], mean %.3f",
            name or "of3pdb",
            path,
            hit,
            len(idx),
            100.0 * hit / max(len(idx), 1),
            mult.min(),
            mult.max(),
            mult.mean(),
        )
        return mult

    # -- indexing -----------------------------------------------------------

    def _keep_mask(self) -> np.ndarray:
        keep = np.zeros(len(self.cache), dtype=bool)
        if self.indices is None:
            keep[:] = True
        else:
            keep[self.indices] = True
        return keep

    def __len__(self) -> int:
        return len(self.cache) if self.indices is None else int(self.indices.shape[0])

    def _struct_index(self, idx: int) -> int:
        return int(idx) if self.indices is None else int(self.indices[idx])

    def _sample_id(self, idx: int) -> str:
        return self.cache.pdb_id(self._struct_index(idx))

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return _retry_getitem(self, idx, self._load_sample, self._sample_id)

    # -- anchor -------------------------------------------------------------

    def _draw_anchor(self, idx: int, si: int, key_to_asym: dict[str, int]) -> tuple[str, int, int] | None:
        """AF3 §2.5.1 conditional: one of this structure's chain / interface anchors."""
        if self._w_chain is None:
            return None
        c_lo, c_hi = self.cache.chain_slice(si)
        i_lo, i_hi = self.cache.iface_slice(si)
        w = np.concatenate([self._w_chain[c_lo:c_hi], self._w_iface[i_lo:i_hi]])
        total = w.sum()
        if total <= 0:
            return None
        rng = _per_sample_rng(idx, 0x85EBCA6B)
        pick = int(np.searchsorted(np.cumsum(w), rng.random() * total))
        pick = min(pick, len(w) - 1)
        n_chain = c_hi - c_lo
        keys = self.cache.chain_key[c_lo:c_hi]
        if pick < n_chain:
            aid = key_to_asym.get(keys[pick].decode("ascii"))
            return None if aid is None else ("chain", aid, -1)
        j = i_lo + (pick - n_chain)
        a = key_to_asym.get(keys[int(self.cache.iface_a[j])].decode("ascii"))
        b = key_to_asym.get(keys[int(self.cache.iface_b[j])].decode("ascii"))
        if a is None or b is None:
            return None
        return ("interface", a, b)

    # -- load ---------------------------------------------------------------

    def _load_sample(self, idx: int) -> dict[str, torch.Tensor]:
        si = self._struct_index(idx)
        pdb_id = self.cache.pdb_id(si)

        struct_npz = self.structures_dir / pdb_id / f"{pdb_id}.npz"
        if not struct_npz.exists():
            raise FileNotFoundError(f"OF3 structure not found: {struct_npz}")
        # build_record=False: every field a Record would carry is already in the
        # columnar cache (chain metadata, interfaces), and deriving it costs
        # O(n_chains²) KDTree interface queries. OF3 splits large assemblies very
        # finely — 4v88 is 1779 chains — where that measured 28.3 s of a 28.9 s
        # conversion, on the dataloader hot path, before the max_chains pre-crop
        # even runs. It was the whole of a 29.8 s data_wait in the r19b run.
        structure, _record, chain_keys = convert_of3_atom_array(struct_npz, pdb_id=pdb_id, build_record=False)

        # The cache indexes chains by OF3's chain key; the structure gives the
        # same keys in file order, which is what asym_id counts. Chains the cache
        # dropped (or that the NPZ lacks) simply have no metadata and get no MSA.
        key_to_asym = {k: i for i, k in enumerate(chain_keys)}
        c_lo, _ = self.cache.chain_slice(si)
        meta_by_asym: dict[int, object] = {}
        chain_row_of_asym: dict[int, int] = {}
        for off, meta in enumerate(self.cache.chains(si)):
            aid = key_to_asym.get(meta.key)
            if aid is None:
                continue
            meta_by_asym[aid] = meta
            chain_row_of_asym[aid] = c_lo + off

        crop_anchor = self._draw_anchor(idx, si, key_to_asym)

        # Large-assembly chain pre-crop (AF3 §2.5.4), anchored so the sampled
        # chain / interface is never the one dropped.
        if self.max_chains > 0 and len(structure.chains) > self.max_chains:
            from deepfold3.data.prepare.filters import select_interface_chains

            kept = select_interface_chains(
                structure,
                anchor=crop_anchor,
                max_chains=self.max_chains,
                interface_distance=self.cropper.interface_distance,
                ligand_inclusion_distance=5.0,
                rng=_per_sample_rng(idx, 0x9E3779B1),
            )
            mask = np.isin(structure.chains["asym_id"], list(kept))
            structure = type(structure)(
                atoms=structure.atoms,
                residues=structure.residues,
                chains=structure.chains[mask],
                bonds=structure.bonds,
                name=structure.name,
            )

        # Crop first, load MSAs / templates second: tokenization never reads MSA
        # content, so a 40-chain assembly cropped to max_tokens reads the few
        # surviving chains' alignments instead of all of them.
        tokenized = self.tokenizer.tokenize(Input(structure=structure, paired_msa={}, unpaired_msa={}))
        tokenized = _msa_cols_from_res_id(tokenized)
        if self.exclude_loss_mol_types:
            from deepfold3.data.masking import MASK_LOSS_WEIGHT, TokenMask

            arr = TokenMask.loss_weight_excluding_mol_types(tokenized.tokens, self.exclude_loss_mol_types)
            tokenized = tokenized.with_mask(TokenMask({MASK_LOSS_WEIGHT: arr}))

        tokenized = maybe_atomize(
            tokenized,
            idx,
            prob=self.atomize_prob,
            source=self.atomize_source,
            max_residues=self.atomize_max_residues,
            interface_distance=self.cropper.interface_distance,
        )
        tokenized = self.cropper.crop(tokenized, self.max_tokens, self.max_atoms, anchor=crop_anchor)

        surviving = {int(a) for a in tokenized.tokens["asym_id"]}
        chain_msa_cols = compute_chain_msa_cols(tokenized.tokens["asym_id"], tokenized.msa_col_idx)

        paired_msa, unpaired_msa = self._load_msas(tokenized, meta_by_asym, surviving, chain_msa_cols)

        template_col_idx = tokenized.query_col_idx
        remapped = remap_msa_col_idx_to_slice(tokenized.tokens["asym_id"], tokenized.msa_col_idx, chain_msa_cols)
        from dataclasses import replace

        tokenized = replace(tokenized, paired_msa=paired_msa, unpaired_msa=unpaired_msa, msa_col_idx=remapped)

        template_feats = None
        if self._has_templates:
            template_feats = self._load_templates(si, meta_by_asym, chain_row_of_asym, surviving, structure)

        feats = self.featurizer.process(
            tokenized,
            training=True,
            random_rot=True,
            template_feats_per_chain=template_feats,
            template_col_idx=template_col_idx,
        )
        feats["is_distillation"] = torch.tensor(self.is_distillation, dtype=torch.bool)
        res = self.cache.resolution(si)
        feats["resolution"] = torch.tensor(const.RESOLUTION_UNKNOWN if res is None else res, dtype=torch.float32)
        feats["_record_id"] = f"{self.name}:{pdb_id}" if self.name else pdb_id
        return feats

    # -- MSA ----------------------------------------------------------------

    def _load_msas(
        self,
        tokenized,
        meta_by_asym: dict,
        surviving: set[int],
        chain_msa_cols: dict[int, np.ndarray],
    ) -> tuple[dict[int, MSA], dict[int, MSA]]:
        """Per-chain unpaired MSAs (+ species-paired blocks) for surviving chains.

        Chains that share an ``alignment_representative_id`` (every copy of a
        homomeric entity) read and decode the file **once**; the columns each copy
        keeps can still differ after cropping, so the per-chain slice is applied
        per chain.

        A polymer chain with no alignment falls back to a single-sequence MSA read
        off its own tokens. That is not a rare corner here: OF3 searches proteins
        and RNA only, so **every DNA chain** has ``alignment_representative_id:
        null``. Without the fallback such a chain is simply absent from
        ``unpaired_msa``, and the featurizer leaves its columns at ``MSA_GAP`` with
        ``msa_mask=True`` — the model never sees the chain's own sequence and its
        profile / deletion mean are all zero, which inference (where the query row
        is always present) never does.
        """
        wanted: dict[int, str] = {}
        mol_of_rep: dict[str, int] = {}
        for aid, meta in meta_by_asym.items():
            if aid not in surviving or chain_msa_cols.get(aid) is None:
                continue
            if meta.aln_rep_id:
                wanted[aid] = meta.aln_rep_id
                mol_of_rep.setdefault(meta.aln_rep_id, meta.mol_type)

        # Pairing needs two distinct representatives, so a monomer (or a pure
        # homomer, whose paired rows the featurizer synthesises from the unpaired
        # MSA anyway) can skip decompressing the UniProt block entirely.
        pairing = self.use_paired_msa and len(set(wanted.values())) > 1
        keys = _MAIN_AND_PAIRING_KEYS if pairing else _MAIN_KEYS

        blocks_by_rep: dict[str, dict] = {}
        rep_of_asym: dict[int, str] = {}
        for aid, rep in wanted.items():
            if rep not in blocks_by_rep:
                path = self.alignment_dir / f"{rep}.npz"
                if not path.exists():
                    continue
                try:
                    blocks_by_rep[rep] = read_of3_alignment(path, keys)
                except Exception:
                    logger.debug("OF3 alignment unreadable: %s", path, exc_info=True)
                    continue
            rep_of_asym[aid] = rep

        unpaired: dict[int, MSA] = {}
        for aid, rep in rep_of_asym.items():
            msa = build_chain_msa(blocks_by_rep[rep], mol_of_rep[rep], self.max_msa_seqs, chain_msa_cols[aid])
            if msa is not None and not msa.is_empty:
                unpaired[aid] = msa

        from deepfold3.data.module.dataset import TrainingDataset

        for aid, cols in chain_msa_cols.items():
            if aid in unpaired or aid not in surviving:
                continue
            unpaired[aid] = TrainingDataset._self_msa_from_tokens(tokenized, aid, cols)

        paired: dict[int, MSA] = {}
        if pairing:
            # One representative may back several chains with different crops; the
            # paired block is built per representative, so pair on a single
            # column set per representative (the first chain that uses it) and
            # scatter the result back to every chain of that representative.
            cols_per_rep: dict[str, np.ndarray | None] = {}
            for aid, rep in rep_of_asym.items():
                cols_per_rep.setdefault(rep, chain_msa_cols[aid])
            by_rep = pair_uniprot_msas(
                {r: blocks_by_rep[r] for r in cols_per_rep},
                mol_of_rep,
                cols_per_rep,
                max_rows=const.MAX_PAIRED_MSA_ROWS,
            )
            for aid, rep in rep_of_asym.items():
                block = by_rep.get(rep)
                # Only chains whose crop matches the representative's paired
                # column set can reuse it; the rest keep unpaired rows only.
                if block is not None and np.array_equal(cols_per_rep[rep], chain_msa_cols[aid]):
                    paired[aid] = block

        return paired, unpaired

    # -- templates ----------------------------------------------------------

    def _effective_cutoff(self, si: int) -> str | None:
        """Tightest applicable template date cap for this query, or ``None``.

        ``min(max_template_date, query_release - cutoff_days)`` (AF3 §2.4). Returns
        ``None`` when no cap can be established, and the caller then drops every
        template — an unknown cutoff must never mean "unfiltered".
        """
        caps = []
        if self.max_template_date:
            caps.append(self.max_template_date)
        released = self.cache.release_date(si)
        if released and self.template_cutoff_days > 0:
            import datetime

            try:
                d = datetime.date.fromisoformat(released) - datetime.timedelta(days=self.template_cutoff_days)
                caps.append(d.isoformat())
            except ValueError:
                pass
        return min(caps) if caps else None

    def _load_templates(
        self,
        si: int,
        meta_by_asym: dict,
        chain_row_of_asym: dict[int, int],
        surviving: set[int],
        structure,
    ) -> dict[int, list[TemplateFeatures]] | None:
        from deepfold3.data.parse.templates import _fill_template_features

        cutoff = self._effective_cutoff(si)
        if cutoff is None:
            return None
        pdb_id = self.cache.pdb_id(si)
        res_num = {int(c["asym_id"]): int(c["res_num"]) for c in structure.chains}

        out: dict[int, list[TemplateFeatures]] = {}
        for aid, meta in meta_by_asym.items():
            if aid not in surviving or not meta.aln_rep_id:
                continue
            allowed = set(self.cache.template_ids(chain_row_of_asym[aid])) if self.cache.has_templates else None
            if allowed is not None and not allowed:
                # The cache lists no templates for this chain, so OF3 never wrote
                # a hit file for its representative and there is nothing to look
                # up. Its template search is protein-only (every RNA / DNA /
                # ligand chain lands here) and ~7% of protein chains have no hits
                # either — 12,210 of the training set's representatives in total.
                # Without this the lookup is a stat() per such chain per sample
                # that can only ever miss, and it reports each miss as a data
                # gap, burying any real one.
                continue
            path = self.template_cache_dir / f"{meta.aln_rep_id}.npz"
            if not path.exists():
                _count_drop("OF3 template hit list missing", f"{pdb_id} chain {aid}: {path}")
                continue
            try:
                hits = load_of3_template_hits(path)
            except Exception:
                _count_drop("OF3 template hit list unreadable", f"{pdb_id} chain {aid}: {path}", exc_info=True)
                continue

            usable = [
                (tid, hit)
                for tid, hit in hits.items()
                if tid.split("_")[0].lower() != pdb_id
                and (allowed is None or tid in allowed)
                and hit.get("release_date")
                and str(hit["release_date"]) <= cutoff
            ]
            if not usable:
                continue
            # AF3 §2.4: k ~ min(Uniform[0, n], max_templates) — k = 0 is a real
            # draw, so template-free inference stays in-distribution.
            k = min(random.randint(0, len(usable)), self.max_templates)
            if k == 0:
                continue
            chosen = random.sample(usable, k)

            query_len = res_num.get(aid, 0)
            if query_len == 0:
                continue
            feats: list[TemplateFeatures] = []
            for tid, hit in chosen:
                arr_path = self.template_array_dir / tid.split("_")[0] / f"{tid}.npz"
                if not arr_path.exists():
                    # A *partially* synced geometry tree passes the directory
                    # check in __init__ and then loses templates one file at a
                    # time. The k draw above already happened, so each miss
                    # silently shrinks this sample's template count — count it,
                    # or the run just trains with thinner templates than the
                    # config asked for and nothing ever says so.
                    _count_drop("OF3 template geometry missing", f"{pdb_id} chain {aid}: {arr_path}")
                    continue
                try:
                    lookup = template_residue_lookup(arr_path)
                except Exception:
                    _count_drop("OF3 template geometry unreadable", f"{pdb_id} chain {aid}: {arr_path}", exc_info=True)
                    continue
                idx_map = np.asarray(hit["idx_map"])
                # idx_map[:, 0] is 1-based within the query chain; idx_map[:, 1]
                # is the template's res_id, which keys `lookup` directly.
                mapping = {int(q) - 1: int(t) for q, t in idx_map}
                tf, matched = _fill_template_features(lookup, mapping, query_len, const.MAX_ATOMS_PER_TOKEN)
                if matched > 0:
                    feats.append(tf)
            if feats:
                out[aid] = feats
        return out or None
