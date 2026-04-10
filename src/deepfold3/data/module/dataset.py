"""Dataset and inference input utilities.

TrainingDataset: loads Records, crops, and featurizes for training.
InferenceInput:  builds features directly from a sequence string (no mmCIF needed).
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
from deepfold3.data.crop.cropper import Cropper
from deepfold3.data.feature.featurizer import Featurizer
from deepfold3.data.feature.ref_atoms import get_ref_atoms
from deepfold3.data.masking import TokenMask
from deepfold3.data.msa.paths import paired_msa_path, unpaired_msa_path
from deepfold3.data.paths import mask_path, monomer_template_path, structure_path, template_path
from deepfold3.data.tokenize.tokenizer import Tokenizer
from deepfold3.data.types import (
    ATOM_DTYPE,
    BOND_DTYPE,
    CHAIN_DTYPE,
    MSA,
    RESIDUE_DTYPE,
    Input,
    Record,
    Structure,
)

if TYPE_CHECKING:
    from deepfold3.data.ccd import CcdMolecule
    from deepfold3.data.parse.templates import TemplateFeatures

logger = logging.getLogger(__name__)


class TrainingDataset(torch.utils.data.Dataset):
    """PyTorch dataset for DeepFold3 training.

    Loads Record -> Structure NPZ + MSA NPZ -> tokenize -> crop -> featurize.

    Args:
        records: List of Record objects.
        data_dir: Root directory containing structures/ and msa/ subdirectories.
        max_tokens: Maximum tokens per sample.
        max_atoms: Maximum atoms per sample.
        max_msa_seqs: Maximum MSA sequences.
    """

    def __init__(
        self,
        records: list[Record],
        data_dir: Path,
        max_tokens: int = 384,
        max_atoms: int = 4096,
        max_msa_seqs: int = const.MAX_MSA_ROWS,
        is_distillation: bool = False,
        mmcif_dir: Path | None = None,
        max_templates: int = 4,
        template_cutoff_days: int = 60,
        max_template_date: str | None = None,
        prefer_monomer_templates: bool = True,
        max_neighborhood: int = 40,
        contiguous_crop_weight: float = 0.2,
        spatial_crop_weight: float = 0.4,
        interface_crop_weight: float = 0.4,
        interface_distance: float = 15.0,
        sampling_rows: list | None = None,
    ):
        self.records = records
        # Optional flat sampling-row table from ``WeightedClusterSampler``.
        # When set, ``__getitem__`` interprets the index as a row id and
        # routes the row's anchor (chain or interface) into the cropper.
        # When unset (DistributedSampler / no sampler) the index is a
        # plain record index.
        self.sampling_rows = sampling_rows
        self.data_dir = Path(data_dir)
        self.max_tokens = max_tokens
        self.max_atoms = max_atoms
        self.is_distillation = is_distillation
        self.tokenizer = Tokenizer()
        self.cropper = Cropper(
            max_neighborhood=max_neighborhood,
            contiguous_weight=contiguous_crop_weight,
            spatial_weight=spatial_crop_weight,
            interface_weight=interface_crop_weight,
            interface_distance=interface_distance,
        )
        # Optional ligand symmetry cache (built once per data dir via
        # ``deepfold3 build-ligand-cache``). Loaded lazily into memory and
        # passed to the featurizer so the loss can apply ligand atom
        # permutations during training.
        ligand_cache_path = self.data_dir / "ligand_symmetry.json"
        ligand_perm_cache: dict | None = None
        if ligand_cache_path.exists():
            import json as _json

            try:
                ligand_perm_cache = _json.loads(ligand_cache_path.read_text())
                logger.info("Loaded ligand symmetry cache: %d CCDs", len(ligand_perm_cache))
            except Exception:
                logger.warning("Failed to load %s; ligand symmetry disabled", ligand_cache_path)
        self.featurizer = Featurizer(max_msa_seqs=max_msa_seqs, ligand_perm_cache=ligand_perm_cache)
        # Mask/template availability is checked lazily per record at __getitem__
        # time. Pre-scanning would walk hundreds of thousands of files per worker
        # under DDP and is unnecessary — kernel dentry cache makes the per-record
        # exists() effectively free after the first hit.
        self._has_masks = (self.data_dir / "masks").is_dir()
        self.mmcif_dir = Path(mmcif_dir) if mmcif_dir else None
        self.max_templates = max_templates
        self.template_cutoff_days = template_cutoff_days
        self.max_template_date = max_template_date
        self.prefer_monomer_templates = prefer_monomer_templates
        self._has_templates = (self.data_dir / "templates").is_dir()
        self._has_monomer_templates = (self.data_dir / "monomer_templates").is_dir()

    def __len__(self) -> int:
        if self.sampling_rows is not None:
            return len(self.sampling_rows)
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # Resolve idx → record + optional crop anchor
        crop_anchor: tuple[str, int, int] | None = None
        if self.sampling_rows is not None:
            row = self.sampling_rows[idx]
            record = self.records[row.record_idx]
            if row.anchor_a >= 0:
                crop_anchor = (row.anchor_kind, int(row.anchor_a), int(row.anchor_b))
        else:
            record = self.records[idx]

        # Load structure
        struct_path = structure_path(self.data_dir / "structures", record.id)
        if not struct_path.exists():
            raise FileNotFoundError(f"Structure file not found: {struct_path}")
        structure = Structure.load(struct_path)

        # Load per-chain MSAs (unpaired + optional paired)
        msa_dir = self.data_dir / "msa"
        unpaired_msa: dict[int, MSA] = {}
        paired_msa: dict[int, MSA] = {}
        for chain_info in record.chains:
            # Unpaired MSA
            msa_path = unpaired_msa_path(msa_dir, str(chain_info.msa_id))
            if msa_path.exists():
                unpaired_msa[chain_info.chain_id] = MSA.load(msa_path)

            # Paired MSA (cross-chain co-evolutionary signal)
            if chain_info.paired_msa_id is not None:
                paired_path = paired_msa_path(msa_dir, str(chain_info.paired_msa_id))
                if paired_path.exists():
                    paired_msa[chain_info.chain_id] = MSA.load(paired_path)

        inp = Input(structure=structure, paired_msa=paired_msa, unpaired_msa=unpaired_msa)

        # Load templates: prefer per-sequence monomer (AF3 §2.4) when
        # available, fall back to multimer otherwise.
        template_feats_per_chain: dict[int, list[TemplateFeatures]] | None = None
        if self.mmcif_dir:
            if self.prefer_monomer_templates and self._has_monomer_templates:
                template_feats_per_chain = self._load_monomer_templates(record)
            elif self._has_templates:
                tpl_index = template_path(self.data_dir / "templates", record.id)
                if tpl_index.exists():
                    n_templates = random.randint(0, self.max_templates)
                    if n_templates > 0:
                        template_feats_per_chain = self._load_multimer_templates(record, n_templates)

        # Tokenize -> attach mask -> crop -> featurize
        tokenized = self.tokenizer.tokenize(inp)

        if self._has_masks:
            mp = mask_path(self.data_dir / "masks", record.id)
            if mp.exists():
                tokenized = tokenized.with_mask(TokenMask.load(str(mp)))

        tokenized = self.cropper.crop(tokenized, self.max_tokens, self.max_atoms, anchor=crop_anchor)
        feats = self.featurizer.process(
            tokenized, training=True, random_rot=True, template_feats_per_chain=template_feats_per_chain
        )

        # Add resolution from manifest metadata (enables confidence loss gating)
        if record.structure_info and record.structure_info.resolution is not None:
            feats["resolution"] = torch.tensor(record.structure_info.resolution, dtype=torch.float32)

        # Distillation flag: disables confidence/distogram losses
        feats["is_distillation"] = torch.tensor(self.is_distillation, dtype=torch.bool)

        return feats

    def _load_monomer_templates(self, record: Record) -> dict[int, list[TemplateFeatures]] | None:
        """Load per-chain monomer template features (AlphaFold 3 §2.4).

        For each protein chain in the record:

        1. Load the precomputed ``monomer_templates/{ab}/{msa_id}.json`` list
           of single-chain hits (already filtered by AF3 per-hit criteria at
           prepare time, sorted by e-value, capped at 20).
        2. Apply the **training-time** release-date cutoff:
           ``min(max_template_date, record.released - cutoff_days)``. The
           cutoff is per-record because the same sequence can appear in
           records with different release dates.
        3. Drop self-hits whose ``template_pdb_id`` matches the record id.
        4. Sample ``k = min(Uniform[0, n], max_templates)`` random hits, then
           extract template features from the local PDB mmCIF mirror.
        """
        import datetime

        from deepfold3.data.parse.templates import (
            _build_polymer_lookup,
            _fill_template_features,
            _get_chain_sequences,
            _realign_mapping,
            cigar_to_index_pairs,
            parse_mmcif_structure,
            read_pdb_mmcif_local,
            select_template_chain,
        )
        from deepfold3.data.prepare.template_search import load_monomer_template_hits

        templates_root = self.data_dir / "monomer_templates"
        record_id_upper = record.id.upper()

        # Effective cutoff = min(max_template_date, record.released - N days)
        cutoff: str | None = self.max_template_date
        query_release = record.structure_info.released if record.structure_info else None
        if query_release is not None:
            try:
                d = datetime.date.fromisoformat(query_release) - datetime.timedelta(days=self.template_cutoff_days)
                query_minus = d.isoformat()
                cutoff = query_minus if cutoff is None else min(cutoff, query_minus)
            except ValueError:
                pass

        result: dict[int, list[TemplateFeatures]] = {}
        for chain_info in record.chains:
            if not chain_info.valid or chain_info.mol_type != const.MOL_PROTEIN:
                continue
            tpl_path = monomer_template_path(templates_root, str(chain_info.msa_id))
            if not tpl_path.exists():
                continue
            try:
                hits = load_monomer_template_hits(tpl_path)
            except Exception:
                logger.debug("Failed to load monomer template index for %s", chain_info.msa_id)
                continue

            # Date cutoff (per-record) + drop self-hits
            filtered = [
                h
                for h in hits
                if h.template_pdb_id.upper() != record_id_upper
                and (cutoff is None or (h.release_date is not None and h.release_date <= cutoff))
            ]
            if not filtered:
                continue

            # AF3: k ~ min(Uniform[0, n], max_templates)
            n_avail = len(filtered)
            k = min(random.randint(0, n_avail), self.max_templates)
            if k == 0:
                continue
            sampled = random.sample(filtered, k)

            chain_feats: list[TemplateFeatures] = []
            for hit in sampled:
                mmcif_text = read_pdb_mmcif_local(hit.template_pdb_id, self.mmcif_dir)
                if mmcif_text is None:
                    continue
                st = parse_mmcif_structure(mmcif_text, label=hit.template_pdb_id)
                if st is None:
                    continue
                try:
                    query_indices, seqres_indices = cigar_to_index_pairs(hit.cigar, hit.qstart, hit.tstart)
                    if not query_indices:
                        continue
                    chain = select_template_chain(st, hit.template_chain_id, label=hit.template_pdb_id)
                    if chain is None:
                        continue
                    query_to_seqres = dict(zip(query_indices, seqres_indices, strict=True))
                    seqres, struct_seq = _get_chain_sequences(st, chain)
                    mapping = _realign_mapping(seqres, struct_seq, query_to_seqres)
                    lookup = _build_polymer_lookup(chain, const.MAX_ATOMS_PER_TOKEN)
                    feats, _ = _fill_template_features(
                        lookup, mapping, chain_info.num_residues, const.MAX_ATOMS_PER_TOKEN
                    )
                except Exception:
                    logger.debug(
                        "Failed to extract monomer template %s chain %s for %s/%s",
                        hit.template_pdb_id,
                        hit.template_chain_id,
                        record.id,
                        chain_info.chain_name,
                    )
                    continue
                chain_feats.append(feats)

            if chain_feats:
                result[chain_info.chain_id] = chain_feats

        return result if result else None

    def _load_multimer_templates(self, record: Record, n_templates: int) -> dict[int, list[TemplateFeatures]] | None:
        """Load multimer template features from pre-indexed JSON + PDB mmCIF.

        Parses each template PDB once and reuses the gemmi Structure across
        chain alignments to avoid redundant mmCIF parsing.
        """
        from deepfold3.data.parse.templates import (
            _build_polymer_lookup,
            _fill_template_features,
            _get_chain_sequences,
            _realign_mapping,
            cigar_to_index_pairs,
            parse_mmcif_structure,
            read_pdb_mmcif_local,
            select_template_chain,
        )
        from deepfold3.data.prepare.template_search import load_template_hits

        tpl_path = template_path(self.data_dir / "templates", record.id)
        try:
            hits = load_template_hits(tpl_path)
        except Exception:
            logger.debug("Failed to load template index for %s", record.id)
            return None

        if len(hits) > n_templates:
            hits = random.sample(hits, n_templates)

        result: dict[int, list[TemplateFeatures]] = {}
        for hit in hits:
            mmcif_text = read_pdb_mmcif_local(hit.template_pdb_id, self.mmcif_dir)
            if mmcif_text is None:
                continue

            # Parse mmCIF once per template PDB
            st = parse_mmcif_structure(mmcif_text, label=hit.template_pdb_id)
            if st is None:
                continue

            # Per-chain dropout: randomly drop chain alignments so the model
            # sees partial multimer context.  Always keep at least one chain.
            alignments = hit.chain_alignments
            if len(alignments) > 1:
                alignments = [a for a in alignments if random.random() < 0.5]
                if not alignments:
                    alignments = [random.choice(hit.chain_alignments)]

            for aln in alignments:
                chain_info = record.chains[aln.query_chain_idx]
                query_indices, seqres_indices = cigar_to_index_pairs(aln.cigar, aln.qstart, aln.tstart)
                if not query_indices:
                    continue
                try:
                    chain = select_template_chain(st, aln.template_chain_id, label=hit.template_pdb_id)
                    if chain is None:
                        continue
                    query_to_seqres = dict(zip(query_indices, seqres_indices, strict=True))
                    seqres, struct_seq = _get_chain_sequences(st, chain)
                    mapping = _realign_mapping(seqres, struct_seq, query_to_seqres)
                    lookup = _build_polymer_lookup(chain, const.MAX_ATOMS_PER_TOKEN)
                    feats, _ = _fill_template_features(
                        lookup, mapping, chain_info.num_residues, const.MAX_ATOMS_PER_TOKEN
                    )
                except Exception:
                    logger.debug(
                        "Failed to extract template %s chain %s for %s",
                        hit.template_pdb_id,
                        aln.template_chain_id,
                        record.id,
                    )
                    continue
                result.setdefault(chain_info.chain_id, []).append(feats)

        return result if result else None


def _tokenize_and_featurize(
    inp: Input,
    max_msa_seqs: int = const.MAX_MSA_ROWS,
    templates: list[TemplateFeatures] | None = None,
    template_feats_per_chain: dict[int, list[TemplateFeatures]] | None = None,
) -> dict[str, torch.Tensor]:
    """Shared tokenize -> featurize pipeline for inference inputs."""
    tokenizer = Tokenizer()
    featurizer = Featurizer(max_msa_seqs=max_msa_seqs)
    tokenized = tokenizer.tokenize(inp)
    return featurizer.process(
        tokenized,
        training=False,
        random_rot=False,
        templates=templates,
        template_feats_per_chain=template_feats_per_chain,
    )


class InferenceInput:
    """Build model-ready features from a raw sequence string, A3M, or JSON.

    Uses CCD ideal coordinates for structure building.  MSA is either
    parsed from an A3M file or synthesized as a single-sequence placeholder.
    """

    @staticmethod
    def from_a3m(
        a3m_text: str,
        mol_type: str = "protein",
        max_msa_seqs: int = const.MAX_MSA_ROWS,
        templates: list[TemplateFeatures] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Build features from an A3M-format MSA string.

        The query sequence (first record) determines the structure;
        all sequences contribute to the MSA features.

        Args:
            a3m_text: Full A3M text (headers + sequences).
            mol_type: 'protein', 'rna', or 'dna'.
            max_msa_seqs: Maximum MSA sequences to keep.
            templates: Optional list of TemplateFeatures for template embedding.
        """
        from deepfold3.data.parse.a3m import parse_a3m

        mt = const.MOL_TYPE_IDS.get(mol_type.upper(), const.MOL_PROTEIN)

        msa_obj = parse_a3m(a3m_text, mt)
        if msa_obj.sequences.shape[0] == 0:
            raise ValueError("A3M file contains no sequences")

        query_seq = _msa_indices_to_sequence(msa_obj.sequences[0], mt)
        structure = _sequence_to_structure(query_seq, mt)
        inp = Input(structure=structure, paired_msa={}, unpaired_msa={0: msa_obj})
        return _tokenize_and_featurize(inp, max_msa_seqs=max_msa_seqs, templates=templates)

    @staticmethod
    def from_sequence(
        sequence: str,
        mol_type: str = "protein",
        max_msa_seqs: int = 1,
        templates: list[TemplateFeatures] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Build features for a single sequence.

        Args:
            sequence: 1-letter sequence string (protein) or RNA/DNA sequence.
            mol_type: 'protein', 'rna', or 'dna'.
            max_msa_seqs: Maximum MSA sequences (default 1 = single sequence).
            templates: Optional list of TemplateFeatures for template embedding.

        Returns:
            dict[str, torch.Tensor] ready for model.forward().
        """
        mt = const.MOL_TYPE_IDS.get(mol_type.upper(), const.MOL_PROTEIN)
        structure = _sequence_to_structure(sequence, mt)
        msa = _sequence_to_msa(sequence, mt)
        inp = Input(structure=structure, paired_msa={}, unpaired_msa={0: msa})
        return _tokenize_and_featurize(inp, max_msa_seqs=max_msa_seqs, templates=templates)

    @staticmethod
    def from_json(
        json_path: str | Path,
        job_index: int = 0,
        max_msa_seqs: int = const.MAX_MSA_ROWS,
        max_templates: int = 4,
        template_cache_dir: str | Path | None = None,
        pdb_mmcif_dir: str | Path | None = None,
    ) -> dict[str, torch.Tensor]:
        """Build features from a Protenix-format or AF3-format JSON input file.

        Supports multi-chain complexes with protein, DNA, and RNA chains.
        Each chain can optionally specify paired/unpaired MSAs and templates.

        All file I/O (reading A3M files, downloading template CIFs) is handled
        by ``parse_json_input()``.  This method only does computation.

        Args:
            json_path: Path to the JSON input file.
            job_index: Which job entry to use (default 0).
            max_msa_seqs: Maximum MSA sequences to keep.
            max_templates: Maximum number of templates to extract.
            template_cache_dir: Directory to cache downloaded template mmCIF files.
            pdb_mmcif_dir: Optional local PDB mmCIF mirror directory
                (layout: ``{dir}/{mid2}/{id}.cif.gz``).

        Returns:
            dict[str, torch.Tensor] ready for model.forward().
        """
        from deepfold3.data.parse.json_input import parse_json_input

        json_path = Path(json_path)
        jobs = parse_json_input(
            json_path, max_templates=max_templates, template_cache_dir=template_cache_dir, pdb_mmcif_dir=pdb_mmcif_dir
        )

        if job_index >= len(jobs):
            raise IndexError(f"Job index {job_index} out of range (file has {len(jobs)} jobs)")

        return InferenceInput.from_job(jobs[job_index], max_msa_seqs=max_msa_seqs)

    @staticmethod
    def from_job(
        job,
        max_msa_seqs: int = const.MAX_MSA_ROWS,
    ) -> dict[str, torch.Tensor]:
        """Build features from a pre-parsed JobInput object.

        Use this to avoid double-parsing when you already have a JobInput
        (e.g., from ``parse_json_input()``).

        Args:
            job: A ``JobInput`` from ``parse_json_input()``.
            max_msa_seqs: Maximum MSA sequences to keep.

        Returns:
            dict[str, torch.Tensor] ready for model.forward().
        """
        from deepfold3.data.parse.templates import TemplateHitInput, template_hit_input_to_features

        # Collect all polymer chains in order: protein, DNA, RNA
        # Each entry: (sequence, mol_type, paired_msa, unpaired_msa, template_hits)
        chain_specs: list[tuple[str, int, MSA | None, MSA | None, list[TemplateHitInput]]] = []
        for pc in job.protein_chains:
            chain_specs.append((pc.sequence, const.MOL_PROTEIN, pc.paired_msa, pc.unpaired_msa, pc.template_hits))
        for dc in job.dna_chains:
            chain_specs.append((dc.sequence, dc.mol_type, dc.paired_msa, dc.unpaired_msa, []))
        for rc in job.rna_chains:
            chain_specs.append((rc.sequence, rc.mol_type, rc.paired_msa, rc.unpaired_msa, []))

        if not chain_specs and not job.ligand_chains:
            raise ValueError(f"No chains found in job {job.name!r}")

        # Build chain names
        next_chain_idx = 0
        chain_tuples = []
        for seq, mt, _, _, _ in chain_specs:
            chain_tuples.append((seq, mt, chr(ord("A") + next_chain_idx)))
            next_chain_idx += 1

        # TODO: structure builder must consume chain.modifications (per-residue
        # CCD substitutions for PTMs / modified bases), job.bonded_atom_pairs
        # (explicit covalent inter-entity bonds), and job.user_ccd (inline mmCIF
        # block registering custom CCD components).  Currently parsed by
        # parse_json_input() but silently ignored during structure assembly.

        # Build ligand molecules
        ligand_mol_chains: list[tuple[CcdMolecule, str]] = []
        if job.ligand_chains:
            from deepfold3.data.ccd import load_ccd_molecule, smiles_to_molecule

            for lig in job.ligand_chains:
                if lig.ccd_ids:
                    for ccd_code in lig.ccd_ids:
                        mol = load_ccd_molecule(ccd_code)
                        chain_name = chr(ord("A") + next_chain_idx)
                        ligand_mol_chains.append((mol, chain_name))
                        next_chain_idx += 1
                elif lig.smiles:
                    mol = smiles_to_molecule(lig.smiles)
                    chain_name = chr(ord("A") + next_chain_idx)
                    ligand_mol_chains.append((mol, chain_name))
                    next_chain_idx += 1

        # Build structure (with ligands if present)
        if ligand_mol_chains:
            structure = _build_complex_structure_with_ligands(chain_tuples, ligand_mol_chains)
        elif chain_tuples:
            structure = _build_complex_structure(chain_tuples)
        else:
            raise ValueError(f"No valid chains in job {job.name!r}")

        # Assemble MSA dicts from parsed objects (polymer chains only)
        paired_msa_dict: dict[int, MSA] = {}
        unpaired_msa_dict: dict[int, MSA] = {}
        for asym_id, (seq, mt, paired_msa, unpaired_msa, _) in enumerate(chain_specs):
            if paired_msa is not None:
                paired_msa_dict[asym_id] = paired_msa
            if unpaired_msa is not None:
                unpaired_msa_dict[asym_id] = unpaired_msa
            elif paired_msa is None:
                # No MSA at all — fall back to single-sequence MSA
                unpaired_msa_dict[asym_id] = _sequence_to_msa(seq, mt)
            else:
                # Only paired exists — use single-sequence as unpaired
                unpaired_msa_dict[asym_id] = _sequence_to_msa(seq, mt)
        # Ligand chains don't have MSA — skip

        # Convert template hits to features
        template_feats: list[TemplateFeatures] | None = None
        template_feats_per_chain: dict[int, list[TemplateFeatures]] = {}

        # Check for whole-complex JSON template hits (query_indices populated)
        json_tmpl_found = False
        for _asym_id, (_seq, _mt, _, _, tmpl_hits) in enumerate(chain_specs):
            if tmpl_hits and tmpl_hits[0].query_indices is not None:
                # JSON-mode templates: whole-complex alignment
                total_tokens = sum(len(s) for s, _, _, _, _ in chain_specs)
                template_feats = [template_hit_input_to_features(h, total_tokens) for h in tmpl_hits]
                if not template_feats:
                    template_feats = None
                json_tmpl_found = True
                break

        # Fall back to per-chain HHR templates
        if not json_tmpl_found:
            for asym_id, (seq, mt, _, _, tmpl_hits) in enumerate(chain_specs):
                if mt == const.MOL_PROTEIN and tmpl_hits:
                    chain_templates = [template_hit_input_to_features(h, len(seq)) for h in tmpl_hits]
                    if chain_templates:
                        template_feats_per_chain[asym_id] = chain_templates

            if len(template_feats_per_chain) == 1:
                template_feats = next(iter(template_feats_per_chain.values()))
            elif len(template_feats_per_chain) > 1:
                template_feats = next(iter(template_feats_per_chain.values()))

        inp = Input(structure=structure, paired_msa=paired_msa_dict, unpaired_msa=unpaired_msa_dict)
        return _tokenize_and_featurize(
            inp,
            max_msa_seqs=max_msa_seqs,
            templates=template_feats,
            template_feats_per_chain=template_feats_per_chain if len(template_feats_per_chain) > 1 else None,
        )


def _letter_to_res_name(letter: str, mol_type: int) -> str:
    """Map a single-character residue letter to its canonical residue name."""
    if mol_type == const.MOL_PROTEIN:
        return const.ONE_LETTER_TO_THREE.get(letter, "UNK")
    elif mol_type == const.MOL_RNA:
        return const.RNA_LETTER_TO_RES.get(letter, "N")
    elif mol_type == const.MOL_DNA:
        return const.DNA_LETTER_TO_RES.get(letter, "DN")
    return "UNK"


def _build_chain_atoms_residues(
    sequence: str,
    mol_type: int,
    atom_offset: int,
) -> tuple[list[tuple], list[tuple], int]:
    """Builds atom and residue tuples for a single polymer chain.

    Args:
        sequence: 1-letter sequence string.
        mol_type: Molecule type (MOL_PROTEIN, MOL_DNA, MOL_RNA).
        atom_offset: Global atom index offset for this chain.

    Returns:
        A tuple of (atoms_list, residues_list, total_atoms_in_chain).
    """
    atoms_list: list[tuple] = []
    residues_list: list[tuple] = []

    atom_idx = atom_offset
    for ri, letter in enumerate(sequence):
        res_name_3 = _letter_to_res_name(letter, mol_type)
        res_type = const.res_name_to_type(res_name_3)

        ref = get_ref_atoms(res_name_3, mol_type)
        n_atoms = int(ref.mask.sum())

        for j in range(n_atoms):
            atoms_list.append(
                (
                    float(ref.positions[j, 0]),
                    float(ref.positions[j, 1]),
                    float(ref.positions[j, 2]),
                    int(ref.element[j]),
                    0.0,  # charge
                    _atom_name_from_chars(ref.atom_name_chars[j]).encode("ascii"),
                    True,
                    0.0,  # b_factor
                )
            )

        residues_list.append(
            (
                res_name_3[:5].encode("ascii"),
                res_type,
                mol_type,
                ri + 1,  # res_id (1-based)
                atom_idx,
                n_atoms,
                True,  # is_standard
                True,  # is_present
            )
        )
        atom_idx += n_atoms

    return atoms_list, residues_list, atom_idx - atom_offset


def _build_ligand_chain_atoms_residues(
    mol: CcdMolecule,
    atom_offset: int,
) -> tuple[list[tuple], list[tuple], list[tuple], int]:
    """Builds atom, bond, and residue tuples for a ligand chain.

    Each CCD component becomes one residue with all its heavy atoms.

    Args:
        mol: CcdMolecule with atom data and bonds.
        atom_offset: Global atom index offset for this chain.

    Returns:
        A tuple of (atoms_list, bonds_list, residues_list, total_atoms).
    """
    atoms_list: list[tuple] = []
    bonds_list: list[tuple] = []

    for j, name in enumerate(mol.atom_names):
        elem_int = const.ELEMENT_TO_INT.get(mol.elements[j].upper(), 6)
        atoms_list.append(
            (
                float(mol.positions[j, 0]),
                float(mol.positions[j, 1]),
                float(mol.positions[j, 2]),
                int(np.clip(elem_int, 0, 127)),
                float(mol.charges[j]),
                name[:4].encode("ascii"),
                True,
                0.0,  # b_factor
            )
        )

    for a1, a2, order in mol.bonds:
        bonds_list.append((atom_offset + a1, atom_offset + a2, order))

    n_atoms = len(mol.atom_names)
    res_type = 20  # UNK — ligand tokens use res_type=20 in the model
    residues_list = [
        (
            mol.res_name[:5].encode("ascii"),
            res_type,
            const.MOL_NONPOLYMER,
            1,  # res_id
            atom_offset,
            n_atoms,
            False,  # is_standard
            True,  # is_present
        )
    ]

    return atoms_list, bonds_list, residues_list, n_atoms


def _build_complex_structure(chains: list[tuple[str, int, str]]) -> Structure:
    """Builds a multi-chain Structure with sequence-based entity_id grouping.

    Chains with identical (normalized) sequences share the same entity_id,
    with incrementing sym_id to distinguish copies.

    Args:
        chains: List of (sequence, mol_type, chain_name) tuples.

    Returns:
        Structure with properly assigned entity_id and sym_id.
    """
    # Map normalized sequences to (entity_id, next_sym_id)
    seq_to_entity: dict[str, list[int]] = {}  # normalized_seq -> [entity_id, next_sym_id]
    next_entity_id = 1

    all_atoms: list[tuple] = []
    all_residues: list[tuple] = []
    all_chains: list[tuple] = []

    atom_offset = 0
    res_offset = 0

    for chain_idx, (sequence, mol_type, chain_name) in enumerate(chains):
        sequence = sequence.strip().upper()
        n_residues = len(sequence)

        # Determine entity_id and sym_id based on sequence identity
        if sequence in seq_to_entity:
            entity_id = seq_to_entity[sequence][0]
            sym_id = seq_to_entity[sequence][1]
            seq_to_entity[sequence][1] += 1
        else:
            entity_id = next_entity_id
            sym_id = 0
            seq_to_entity[sequence] = [entity_id, 1]
            next_entity_id += 1

        # Build atoms and residues for this chain
        chain_atoms, chain_residues, chain_atom_count = _build_chain_atoms_residues(sequence, mol_type, atom_offset)
        all_atoms.extend(chain_atoms)
        all_residues.extend(chain_residues)

        all_chains.append(
            (
                chain_name.encode("ascii"),
                mol_type,
                entity_id,  # sequence-based entity_id (1-based)
                sym_id,  # copy index within entity (0-based)
                chain_idx,  # asym_id (unique chain index)
                res_offset,  # res_idx
                n_residues,  # res_num
                atom_offset,  # atom_idx
                chain_atom_count,  # atom_num
            )
        )

        atom_offset += chain_atom_count
        res_offset += n_residues

    atoms_arr = np.array(all_atoms, dtype=ATOM_DTYPE) if all_atoms else np.empty(0, dtype=ATOM_DTYPE)
    residues_arr = np.array(all_residues, dtype=RESIDUE_DTYPE) if all_residues else np.empty(0, dtype=RESIDUE_DTYPE)
    chains_arr = np.array(all_chains, dtype=CHAIN_DTYPE) if all_chains else np.empty(0, dtype=CHAIN_DTYPE)
    bonds_arr = np.empty(0, dtype=BOND_DTYPE)

    return Structure(
        atoms=atoms_arr,
        bonds=bonds_arr,
        residues=residues_arr,
        chains=chains_arr,
    )


def _build_complex_structure_with_ligands(
    chains: list[tuple[str, int, str]],
    ligand_chains: list[tuple[CcdMolecule, str]],
) -> Structure:
    """Builds a multi-chain Structure with both polymer and ligand chains.

    Args:
        chains: List of (sequence, mol_type, chain_name) tuples for polymer chains.
        ligand_chains: List of (CcdMolecule, chain_name) tuples for ligand chains.

    Returns:
        Structure with entity_id, sym_id, and intra-ligand bonds.
    """
    # Start with polymer chains
    structure = _build_complex_structure(chains)

    if not ligand_chains:
        return structure

    # Extend with ligand chains
    all_atoms = list(structure.atoms)
    all_bonds = list(structure.bonds)
    all_residues = list(structure.residues)
    all_chains = list(structure.chains)

    atom_offset = len(structure.atoms)
    res_offset = len(structure.residues)
    chain_idx = len(chains)

    # Entity tracking: ligand sequences grouped by res_name for entity_id
    seq_to_entity: dict[str, list[int]] = {}
    next_entity_id = max((int(c["entity_id"]) for c in structure.chains), default=0) + 1

    for mol, chain_name in ligand_chains:
        lig_atoms, lig_bonds, lig_residues, n_atoms = _build_ligand_chain_atoms_residues(mol, atom_offset)
        all_atoms.extend(lig_atoms)
        all_residues.extend(lig_residues)

        # Add bonds as BOND_DTYPE
        for a1, a2, order in lig_bonds:
            all_bonds.append((a1, a2, order))

        # Entity/sym assignment by ligand res_name
        seq_key = mol.res_name
        if seq_key in seq_to_entity:
            entity_id = seq_to_entity[seq_key][0]
            sym_id = seq_to_entity[seq_key][1]
            seq_to_entity[seq_key][1] += 1
        else:
            entity_id = next_entity_id
            sym_id = 0
            seq_to_entity[seq_key] = [entity_id, 1]
            next_entity_id += 1

        all_chains.append(
            (
                chain_name[:4].encode("ascii"),
                const.MOL_NONPOLYMER,
                entity_id,
                sym_id,
                chain_idx,
                res_offset,
                1,  # res_num (one residue per ligand)
                atom_offset,
                n_atoms,
            )
        )

        atom_offset += n_atoms
        res_offset += 1
        chain_idx += 1

    atoms_arr = np.array(all_atoms, dtype=ATOM_DTYPE) if all_atoms else np.empty(0, dtype=ATOM_DTYPE)
    bonds_arr = np.array(all_bonds, dtype=BOND_DTYPE) if all_bonds else np.empty(0, dtype=BOND_DTYPE)
    residues_arr = np.array(all_residues, dtype=RESIDUE_DTYPE) if all_residues else np.empty(0, dtype=RESIDUE_DTYPE)
    chains_arr = np.array(all_chains, dtype=CHAIN_DTYPE) if all_chains else np.empty(0, dtype=CHAIN_DTYPE)

    return Structure(
        atoms=atoms_arr,
        bonds=bonds_arr,
        residues=residues_arr,
        chains=chains_arr,
    )


def _sequence_to_structure(sequence: str, mol_type: int) -> Structure:
    """Build a single-chain Structure from a 1-letter sequence (no mmCIF needed)."""
    return _build_complex_structure([(sequence, mol_type, "A")])


def _sequence_to_msa(sequence: str, mol_type: int) -> MSA:
    """Build a single-sequence MSA from a 1-letter sequence."""
    N = len(sequence)
    sequences = np.zeros((1, N), dtype=np.int8)
    mask = np.ones((1, N), dtype=bool)
    deletion_matrix = np.zeros((1, N), dtype=np.int8)

    if mol_type == const.MOL_PROTEIN:
        lookup = const.PROTEIN_MSA_ID
        unk = 20
    elif mol_type == const.MOL_RNA:
        lookup = const.RNA_MSA_ID
        unk = const.MSA_UNK_NUCLEIC
    elif mol_type == const.MOL_DNA:
        lookup = const.DNA_MSA_ID
        unk = const.MSA_UNK_NUCLEIC
    else:
        lookup = const.PROTEIN_MSA_ID
        unk = 20

    for i, ch in enumerate(sequence.upper()):
        sequences[0, i] = lookup.get(ch, unk)

    return MSA(sequences=sequences, mask=mask, deletion_matrix=deletion_matrix)


from deepfold3.data.feature.ref_atoms import decode_atom_name_chars as _atom_name_from_chars  # noqa: E402, F401


def _msa_indices_to_sequence(encoded: np.ndarray, mol_type: int) -> str:
    """Convert MSA-index row back to a 1-letter sequence string.

    Gap positions (index 21) are skipped — this reconstructs the query
    match-state sequence from the first row of a parsed A3M.
    """
    if mol_type == const.MOL_PROTEIN:
        rev = {v: k for k, v in const.PROTEIN_MSA_ID.items() if k not in ("-", "B", "J", "O", "U", "Z")}
        rev[20] = "X"
        rev[21] = "-"
    elif mol_type == const.MOL_RNA:
        rev = {v: k for k, v in const.RNA_MSA_ID.items()}
        rev[const.MSA_UNK_NUCLEIC] = "N"
    elif mol_type == const.MOL_DNA:
        rev = {v: k for k, v in const.DNA_MSA_ID.items()}
        rev[const.MSA_UNK_NUCLEIC] = "N"
    else:
        rev = {v: k for k, v in const.PROTEIN_MSA_ID.items() if k not in ("-", "B", "J", "O", "U", "Z")}
        rev[20] = "X"
        rev[21] = "-"

    return "".join(rev.get(int(idx), "X") for idx in encoded if int(idx) != 21)
