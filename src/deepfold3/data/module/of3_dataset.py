"""On-the-fly OpenFold 3 dataset — loads OF3 preprocessed NPZs without bulk conversion.

Converts OF3 structure.npz, alignment.npz, and template.npz to DeepFold3 format
at training time, avoiding the storage and time cost of ``deepfold3 import-of3``
for large datasets.

Only the *conversion logic* is original — no OpenFold 3 code is copied.
The NPZ field names are treated as a data format specification.
"""

from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.utils.data

from deepfold3.data import const
from deepfold3.data.crop.cropper import Cropper
from deepfold3.data.feature.featurizer import Featurizer
from deepfold3.data.io_cache import read_npz
from deepfold3.data.module.dataset import _build_record_anchors, _draw_crop_anchor, _retry_getitem
from deepfold3.data.prepare.openfold3 import convert_of3_structure
from deepfold3.data.tokenize.tokenizer import Tokenizer
from deepfold3.data.types import (
    MSA,
    ChainInfo,
    Input,
    Record,
)

if TYPE_CHECKING:
    from deepfold3.data.parse.templates import TemplateFeatures

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Vectorised char → int8 MSA encoding table
# ---------------------------------------------------------------------------

_CHAR_TO_MSA: np.ndarray = np.full(256, 20, dtype=np.int8)  # default = UNK
for _ch, _idx in const.PROTEIN_MSA_ID.items():
    _CHAR_TO_MSA[ord(_ch)] = _idx
    # Also handle lowercase (insertions are stripped, but just in case)
    if _ch.isalpha():
        _CHAR_TO_MSA[ord(_ch.lower())] = _idx


# ---------------------------------------------------------------------------
# Alignment (MSA) conversion
# ---------------------------------------------------------------------------


def convert_of3_alignment(npz_path: Path, max_seqs: int = 0) -> MSA:
    """Convert an OF3 alignment.npz to a DeepFold3 MSA.

    The OF3 alignment.npz stores MSA data as pickled object arrays.
    Each key maps to a dict with ``msa`` (<U1 char array [S, N]),
    ``deletion_matrix`` (int64 [S, N]), and ``metadata`` (ignored).

    Args:
        npz_path: Path to OF3 alignment.npz.
        max_seqs: If > 0, cap the total number of MSA rows (query + hits).

    Returns:
        DeepFold3 MSA with int8-encoded sequences.
    """
    data = read_npz(npz_path, allow_pickle=True)

    msa_parts: list[np.ndarray] = []
    del_parts: list[np.ndarray] = []

    for key in data.keys():
        item = data[key].item()
        msa_chars = item["msa"]  # <U1 [S, N]
        del_matrix = item["deletion_matrix"]  # int64 [S, N]
        msa_parts.append(msa_chars)
        del_parts.append(del_matrix)

    if not msa_parts:
        return MSA(
            sequences=np.zeros((0, 0), dtype=np.int8),
            mask=np.zeros((0, 0), dtype=bool),
            deletion_matrix=np.zeros((0, 0), dtype=np.int8),
        )

    # Cap rows *before* concatenation to avoid allocating a huge intermediate.
    if max_seqs > 0:
        kept_msa: list[np.ndarray] = []
        kept_del: list[np.ndarray] = []
        total = 0
        for m, d in zip(msa_parts, del_parts, strict=True):
            remaining = max_seqs - total
            if remaining <= 0:
                break
            kept_msa.append(m[:remaining])
            kept_del.append(d[:remaining])
            total += kept_msa[-1].shape[0]
        msa_parts, del_parts = kept_msa, kept_del

    all_msa = np.concatenate(msa_parts, axis=0) if len(msa_parts) > 1 else msa_parts[0]
    all_del = np.concatenate(del_parts, axis=0) if len(del_parts) > 1 else del_parts[0]

    # Char <U1 → ASCII bytes → uint8 → lookup table → int8
    ascii_bytes = all_msa.astype("S1")  # <U1 → |S1 (single ASCII byte)
    byte_view = ascii_bytes.view(np.uint8).reshape(all_msa.shape)
    encoded = _CHAR_TO_MSA[byte_view]  # int8 [S, N]

    # Mask: OF3 MSAs are pre-aligned (no padding), so all positions are valid
    mask = np.ones(encoded.shape, dtype=bool)

    # Deletion matrix: clip to int8 range
    deletion_matrix = np.clip(all_del, 0, 127).astype(np.int8)

    return MSA(sequences=encoded, mask=mask, deletion_matrix=deletion_matrix)


# ---------------------------------------------------------------------------
# Template conversion
# ---------------------------------------------------------------------------


@dataclass
class OF3TemplateHit:
    """A single template hit parsed from OF3 template.npz.

    Attributes:
        template_pdb_id: 4-character PDB ID of the template.
        template_chain_id: Chain name within the template PDB.
        release_date: PDB release date (YYYY-MM-DD) or None.
        idx_map: [K, 2] int array — (query_pos_1based, template_seqres_pos_1based).
    """

    template_pdb_id: str
    template_chain_id: str
    release_date: str | None
    idx_map: np.ndarray  # [K, 2] int64


def load_of3_templates(npz_path: Path) -> list[OF3TemplateHit]:
    """Load template hits from an OF3 template.npz.

    Each key in the NPZ is a ``{pdb_id}_{chain_id}`` string mapping to
    a pickled dict with ``release_date`` (str), ``index`` (int), and
    ``idx_map`` ([K, 2] int64 — query-to-template SEQRES position pairs,
    both 1-based).

    Args:
        npz_path: Path to OF3 template.npz.

    Returns:
        List of OF3TemplateHit objects.
    """
    data = read_npz(npz_path, allow_pickle=True)
    hits: list[OF3TemplateHit] = []
    for key in data.keys():
        item = data[key].item()
        # Key format: "{pdb_id}_{chain_id}" e.g. "6s18_A"
        parts = str(key).rsplit("_", 1)
        if len(parts) == 2:
            pdb_id, chain_id = parts
        else:
            pdb_id, chain_id = str(key), "A"
        hits.append(
            OF3TemplateHit(
                template_pdb_id=pdb_id,
                template_chain_id=chain_id,
                release_date=item.get("release_date"),
                idx_map=item["idx_map"],
            )
        )
    return hits


def _extract_of3_template_features(
    hit: OF3TemplateHit,
    query_length: int,
    mmcif_dir: Path,
) -> TemplateFeatures | None:
    """Extract TemplateFeatures from an OF3 template hit.

    Reads the template PDB from a local mmCIF mirror, then uses the OF3
    ``idx_map`` (1-based query→template SEQRES mapping) to align template
    residue features onto the query sequence.

    The pipeline is identical to ``_load_monomer_templates`` in
    ``TrainingDataset`` except the alignment comes from ``idx_map``
    instead of a CIGAR string.

    Args:
        hit: OF3 template hit with idx_map.
        query_length: Number of residues in the query chain.
        mmcif_dir: Path to PDB mmCIF mirror.

    Returns:
        TemplateFeatures or None on failure.
    """
    from deepfold3.data.parse.templates import (
        _build_polymer_lookup,
        _fill_template_features,
        _get_chain_sequences,
        _realign_mapping,
        parse_mmcif_structure,
        read_pdb_mmcif_local,
        select_template_chain_by_label,
    )

    mmcif_text = read_pdb_mmcif_local(hit.template_pdb_id, mmcif_dir)
    if mmcif_text is None:
        return None
    st = parse_mmcif_structure(mmcif_text, label=hit.template_pdb_id)
    if st is None:
        return None

    # OF3 template hits use label_asym_id (not auth_asym_id)
    chain = select_template_chain_by_label(st, hit.template_chain_id, label=hit.template_pdb_id)
    if chain is None:
        return None

    # Convert OF3 idx_map (1-based) to 0-based query→seqres dict
    idx_map = hit.idx_map
    if idx_map.shape[0] == 0:
        return None
    query_to_seqres = {int(idx_map[i, 0]) - 1: int(idx_map[i, 1]) - 1 for i in range(idx_map.shape[0])}

    seqres, struct_seq = _get_chain_sequences(st, chain)
    mapping = _realign_mapping(seqres, struct_seq, query_to_seqres)
    lookup = _build_polymer_lookup(chain, const.MAX_ATOMS_PER_TOKEN)
    feats, _ = _fill_template_features(lookup, mapping, query_length, const.MAX_ATOMS_PER_TOKEN)
    return feats


# ---------------------------------------------------------------------------
# Record discovery (lightweight — no NPZ reading)
# ---------------------------------------------------------------------------


def discover_of3_records(
    of3_dir: Path,
    id_list: Path | None = None,
) -> list[Record]:
    """Build minimal Record objects from an OF3 preprocessed directory.

    If ``id_list`` is provided (or ``ids.txt`` exists in ``of3_dir``),
    reads structure IDs from that file (one per line, ``#`` comments).
    Otherwise scans ``of3_dir`` for subdirectories containing ``structure.npz``.

    No NPZ files are opened — Records are built from directory names alone.

    Args:
        of3_dir: Root OF3 preprocessed directory.
        id_list: Optional path to a text file listing structure IDs.

    Returns:
        List of minimal Record objects (single chain, no interfaces).
    """
    ids: list[str] = []

    # Try id_list argument, then ids.txt in of3_dir
    list_path = id_list or (of3_dir / "ids.txt")
    if list_path is not None and Path(list_path).exists():
        with open(list_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    ids.append(line)
        logger.info("Loaded %d OF3 structure IDs from %s", len(ids), list_path)
    else:
        # Fast directory scan — no stat beyond is_dir()
        with os.scandir(of3_dir) as it:
            for entry in it:
                if entry.is_dir(follow_symlinks=False):
                    ids.append(entry.name)
        ids.sort()
        logger.info("Scanned %d OF3 structures from %s", len(ids), of3_dir)

        # Cache to ids.txt so subsequent runs (and other DDP ranks) skip the scan.
        # Atomic write via temp file + rename to avoid DDP race conditions.
        cache_path = of3_dir / "ids.txt"
        try:
            tmp = cache_path.with_suffix(f".tmp.{os.getpid()}")
            tmp.write_text("\n".join(ids) + "\n")
            tmp.replace(cache_path)
            logger.info("Cached %d OF3 IDs to %s", len(ids), cache_path)
        except OSError:
            pass  # read-only filesystem — harmless

    records: list[Record] = []
    for struct_id in ids:
        chain_info = ChainInfo(
            chain_id=0,
            chain_name="A",
            mol_type=const.MOL_PROTEIN,
            cluster_id=struct_id,
            msa_id=struct_id,
            num_residues=0,
            entity_id=0,
        )
        records.append(Record(id=struct_id, chains=[chain_info], interfaces=[]))

    return records


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class OF3Dataset(torch.utils.data.Dataset):
    """PyTorch dataset that loads OF3 preprocessed data on-the-fly.

    Converts OF3 structure.npz, alignment.npz, and optionally template.npz
    to DeepFold3 format at ``__getitem__`` time, feeding through the standard
    tokenize → crop → featurize pipeline.

    Args:
        records: List of Record objects (from ``discover_of3_records``).
        of3_dir: Root OF3 preprocessed directory (``{id}/structure.npz``).
        max_tokens: Maximum tokens per sample after cropping.
        max_atoms: Maximum atoms per sample after cropping.
        max_msa_seqs: Maximum MSA sequences to keep.
        is_distillation: Mark samples as distillation (disables confidence loss;
            diffusion + distogram still trained).
        mmcif_dir: PDB mmCIF mirror for template extraction (optional).
        max_templates: Maximum templates per sample (AF3 §2.4).
        max_template_date: Hard cutoff on template release date (ISO YYYY-MM-DD).
            Hits with ``release_date > max_template_date`` are dropped. OF3
            ``template.npz`` carries ``release_date`` per hit; undated hits are
            dropped. Fail-closed: ``None`` drops **all** templates (no cutoff →
            cannot prove any hit is in-range) rather than leaking future ones.
    """

    def __init__(
        self,
        records: list[Record],
        of3_dir: Path,
        max_tokens: int = 384,
        max_atoms: int = 4096,
        max_msa_seqs: int = const.MAX_MSA_ROWS,
        is_distillation: bool = False,
        max_neighborhood: int = 40,
        contiguous_crop_weight: float = 0.2,
        spatial_crop_weight: float = 0.4,
        interface_crop_weight: float = 0.4,
        interface_distance: float = 15.0,
        mmcif_dir: Path | None = None,
        max_templates: int = 4,
        max_template_date: str | None = None,
        name: str = "",
        exclude_loss_mol_types: tuple[int, ...] | list[int] = (),
        msa_crop_size: int = 1024,
        n_msa_crops: int = 0,
        anchor_sampling: bool = False,
        pad_templates: bool = False,
    ):
        self.records = records
        # Dataset label used to disambiguate record ids in NaN-skip logs.
        # See ``TrainingDataset.__init__`` for the full rationale.
        self.name = name
        self.of3_dir = Path(of3_dir)
        self.max_tokens = max_tokens
        self.max_atoms = max_atoms
        self.max_msa_seqs = max_msa_seqs
        self.is_distillation = is_distillation
        self.mmcif_dir = Path(mmcif_dir) if mmcif_dir else None
        self.max_templates = max_templates
        self.max_template_date = max_template_date
        # See ``TrainingDataset.exclude_loss_mol_types`` for semantics.
        self.exclude_loss_mol_types = tuple(int(m) for m in (exclude_loss_mol_types or ()))
        # Two-stage AF3 §2.5.1 anchor sampling (weighted datasets only). OF3
        # sets are monomeric so the anchor is the single chain, but the branch
        # is kept for parity with ``TrainingDataset``. See ``_draw_crop_anchor``.
        self._record_anchors = _build_record_anchors(records, anchor_sampling)

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
            # Must track ``TrainingDataset``: of3 and df3 samples share one
            # DataLoader, so padding only one of them leaves T varying across
            # ranks and the straggler this is meant to remove survives.
            # (Correctness does not depend on the two agreeing — the
            # ``template_mask`` is emitted either way.)
            pad_templates=pad_templates,
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return _retry_getitem(self, idx, self._load_sample, self._sample_id)

    def _sample_id(self, idx: int) -> str:
        return self.records[idx].id

    def _load_sample(self, idx: int) -> dict[str, torch.Tensor]:
        record = self.records[idx]
        struct_id = record.id
        sample_dir = self.of3_dir / struct_id

        # 1. Load and convert structure
        struct_npz = sample_dir / "structure.npz"
        if not struct_npz.exists():
            raise FileNotFoundError(f"OF3 structure not found: {struct_npz}")
        structure, live_record = convert_of3_structure(struct_npz, pdb_id=struct_id)

        # 2. Load and convert MSA (optional)
        unpaired_msa: dict[int, MSA] = {}
        aln_npz = sample_dir / "alignment.npz"
        if aln_npz.exists():
            try:
                msa = convert_of3_alignment(aln_npz, max_seqs=self.max_msa_seqs)
                if msa.sequences.size > 0:
                    unpaired_msa[0] = msa
            except Exception:
                logger.debug("Failed to load OF3 alignment for %s", struct_id, exc_info=True)

        # 3. Load templates (requires mmcif_dir)
        template_feats_per_chain: dict[int, list[TemplateFeatures]] | None = None
        if self.mmcif_dir is not None:
            tpl_npz = sample_dir / "template.npz"
            if tpl_npz.exists():
                template_feats_per_chain = self._load_templates(tpl_npz, live_record)

        # 4. Build Input (monomer: no paired MSA)
        inp = Input.from_msa(structure, unpaired_msa)

        # 5. Tokenize -> attach mask -> crop -> featurize
        tokenized = self.tokenizer.tokenize(inp)
        if self.exclude_loss_mol_types:
            from deepfold3.data.masking import MASK_LOSS_WEIGHT, TokenMask

            arr = TokenMask.loss_weight_excluding_mol_types(tokenized.tokens, self.exclude_loss_mol_types)
            tokenized = tokenized.with_mask(TokenMask({MASK_LOSS_WEIGHT: arr}))
        crop_anchor = _draw_crop_anchor(self._record_anchors, idx)
        tokenized = self.cropper.crop(tokenized, self.max_tokens, self.max_atoms, anchor=crop_anchor)
        feats = self.featurizer.process(
            tokenized, training=True, random_rot=True, template_feats_per_chain=template_feats_per_chain
        )

        # 6. Distillation flag
        feats["is_distillation"] = torch.tensor(self.is_distillation, dtype=torch.bool)

        # OF3 sets carry no resolution metadata, but the key is part of the
        # batch-uniform contract (``dataset._REQUIRED_KEYS``), so emit the
        # sentinel rather than omitting it. See ``const.RESOLUTION_UNKNOWN``.
        feats["resolution"] = torch.tensor(const.RESOLUTION_UNKNOWN, dtype=torch.float32)

        # NaN-skip diagnostic (same protocol as TrainingDataset): prefix
        # with dataset name when set so extras don't collide on raw PDB ids.
        feats["_record_id"] = f"{self.name}:{record.id}" if self.name else record.id

        return feats

    def _load_templates(self, tpl_npz: Path, live_record: Record) -> dict[int, list[TemplateFeatures]] | None:
        """Load and extract template features from an OF3 template.npz.

        Follows AF3 §2.4: randomly sample k = min(Uniform[0, n], max_templates)
        templates from available hits (excluding self-hits).
        """
        try:
            hits = load_of3_templates(tpl_npz)
        except Exception:
            logger.debug("Failed to load OF3 template index for %s", live_record.id, exc_info=True)
            return None

        if not hits:
            return None

        # AF3 §2.4: drop self-hits + enforce hard release-date cutoff. Distillation
        # queries have no meaningful ``released`` date, so the per-query
        # ``release - cutoff_days`` tightening used by the PATCHR path doesn't apply.
        # Fail-closed: a hit is kept only when the cutoff is set and the hit's own
        # release date is known and within it — an unset cutoff or an undated hit is
        # dropped, never leaked (mirrors the PATCHR path in dataset.py).
        record_id_upper = live_record.id.upper()
        cutoff = self.max_template_date
        filtered = [
            h
            for h in hits
            if h.template_pdb_id.upper() != record_id_upper
            and cutoff is not None
            and h.release_date is not None
            and h.release_date <= cutoff
        ]
        if not filtered:
            return None

        # AF3: k ~ min(Uniform[0, n], max_templates)
        n_avail = len(filtered)
        k = min(random.randint(0, n_avail), self.max_templates)
        if k == 0:
            return None
        sampled = random.sample(filtered, k)

        # Get query length from the live record (first chain)
        query_length = live_record.chains[0].num_residues if live_record.chains else 0
        if query_length == 0:
            return None

        chain_feats: list[TemplateFeatures] = []
        for hit in sampled:
            try:
                feats = _extract_of3_template_features(hit, query_length, self.mmcif_dir)
                if feats is not None:
                    chain_feats.append(feats)
            except Exception:
                logger.debug(
                    "Failed to extract OF3 template %s_%s for %s",
                    hit.template_pdb_id,
                    hit.template_chain_id,
                    live_record.id,
                )

        if not chain_feats:
            return None

        # Single chain (monomer): chain_id=0
        return {0: chain_feats}
