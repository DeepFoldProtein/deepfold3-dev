# AF3 Paper Reference — Data Pipeline

Organized reference from the AlphaFold 3 supplementary material (Abramson et al., Nature 2024). Section numbers refer to the supplementary information.

> **Source**: `resources/41586_2024_7487_MOESM1_ESM.txt`
>
> **Caution**: The paper contains known typos (e.g. the diffusion loss EDM weighting formula). Cross-reference with reference implementations (Protenix, OpenFold 3) before trusting any equation.

## §2 Data Pipeline

### §2.1 Parsing

- Input: mmCIF files
- Cleanup: resolve alt-locs (largest occupancy), MSE→MET, remove waters, fix arginine NH1/NH2 naming
- First bioassembly expanded
- Inference: dummy mmCIF with zeroed atom coordinates (from CCD codes or SMILES)

### §2.2 Genetic Search

**Protein databases (5):**

| Database | Search tool | Max sequences |
|----------|-------------|---------------|
| UniRef90 | jackhmmer | 10,000 |
| UniProt | jackhmmer | 50,000 |
| Uniclust30 + BFD | HHBlits v3 | None |
| Reduced BFD | jackhmmer | 5,000 |
| MGnify | jackhmmer | 5,000 |

**RNA databases (3):**

| Database | Clustering tool | Search tool | Max sequences |
|----------|----------------|-------------|---------------|
| Rfam | mmseqs easy-cluster | nhmmer | 10,000 |
| RNACentral | mmseqs easy-linclust | nhmmer | 10,000 |
| Nucleotide collection | mmseqs easy-cluster | nhmmer | 10,000 |

UniProt results kept separate for cross-chain pairing. All others stacked + deduplicated. During training, MSA subsampled: k = Uniform[1, n].

### §2.3 MSA Processing

- Up to N_msa = 16,384 rows
- Row 0: query sequence
- Next rows (up to 8,191): paired UniProt MSA (species-based, AlphaFold-Multimer method)
- Remaining rows: dense unpaired (first rows from original MSA per chain)

### §2.4 Template Search

- **Single chain only** — no multi-chain templates
- Input: deduplicated UniRef90 MSA (cropped to 300 sequences for training)
- hmmbuild → hmmsearch against PDB SEQRES fasta
- Date filter: template release ≥ 60 days before training example release (distillation: max 2018-04-30, inference: max 2021-09-30)
- Remove: exact query matches (>95% coverage), short templates (<10 res or <10% query coverage)
- Sort by e-value, max 20 returned, model uses max 4 (N_templ ≤ 4)
- Training: k random templates from n available, k ~ min(Uniform[0, n], 4)

### §2.5 Training Data

**5 datasets** (Table 3):

| Name | Strategy | Weight |
|------|----------|--------|
| Weighted PDB | weighted | 0.5 |
| Disordered protein PDB distillation | weighted | 0.02 |
| Protein monomer distillation | uniform | 0.495 |
| Short protein monomer distillation | uniform | 0.005 |
| RNA distillation | uniform | 0.05 |

(Transcription factor sets only during fine-tuning: negatives 0.011, positives 0.021)

### §2.5.1 Weighted PDB Sampling

Sample chains AND interfaces (chain pairs with min heavy atom separation < 5 Å).

**Weight formula:**

```
w ∝ (β_r / N_clust) × (α_prot × n_prot + α_nuc × n_nuc + α_ligand × n_ligand)
```

| Parameter | Value |
|-----------|-------|
| β_chain | 0.5 |
| β_interface | 1.0 |
| α_prot | 3.0 |
| α_nuc | 3.0 |
| α_ligand | 1.0 |

The selected chain/interface biases cropping to select crops near it.

### §2.5.3 Clustering

- **Protein**: 40% sequence homology
- **Nucleic acids**: 100% homology
- **Peptides** (<10 residues): 100% homology
- **Small molecules**: CCD identity (only identical molecules share a cluster)
- **Modified residues**: mapped to standard residues via SCOP convention before clustering
- **Interface clustering**: join on chain cluster IDs — interfaces share a cluster only if their constituent chain pairs share the same chain cluster pairs

### §2.5.4 Filtering

**Target-level:**
- Released before 2021-09-30
- Resolution ≤ 9 Å
- Max 300 polymer chains (training), 1000 (evaluation)
- Min 4 resolved residues per polymer chain

**Bioassembly-level:**
- Remove hydrogens
- Remove all-unknown-residue polymer chains
- Remove clashing chains (>30% of atoms within 1.7 Å of another chain; tie-break: higher clash %, then fewer atoms, then larger chain_id)
- Remove atoms outside CCD-defined atom names
- Remove leaving atoms for covalent ligands
- Filter protein chains with consecutive Cα > 10 Å apart
- If >20 chains: select random interface token (<15 Å to another chain), keep closest 20 chains

**Crystallization aids** (Table 9): removed if method indicates crystallography.

### §2.6 Tokenization

- Standard amino acid → 1 token
- Standard nucleotide → 1 token
- Modified amino acid/nucleotide → per-atom (N atoms → N tokens)
- Ligands → per-atom

**Token centre atoms:**
- Cα for standard amino acids
- C1' for standard nucleotides
- First (only) atom for per-atom tokens

### §2.7 Cropping

Three strategies with dataset-specific weights:

| Dataset | Contiguous | Spatial | Spatial interface |
|---------|-----------|---------|-------------------|
| Weighted PDB | 0.20 | 0.40 | 0.40 |
| Disordered PDB | 0.20 | 0.40 | 0.40 |
| Monomer distillation | 0.25 | 0.75 | 0.00 |

- **Contiguous**: select contiguous sequences per chain (AlphaFold-Multimer Algorithm 1)
- **Spatial**: random token centre atom as seed, expand by 3D proximity (AlphaFold-Multimer Algorithm 2)
- **Spatial interface**: seed from interface tokens (centre atom < 15 Å to another chain's centre atom)

For Weighted PDB / Disordered PDB: seed atom chosen from the selected chain/interface.

### §2.8 Features (Table 5)

Key features:

| Feature | Shape | Description |
|---------|-------|-------------|
| `token_bonds` | [N, N] | Polymer-ligand and ligand-ligand bonds; **bonds < 2.4 Å during training** |
| `ref_pos` | [N_atom, 3] | Reference conformer with random rotation + translation |
| `ref_element` | [N_atom, 128] | One-hot element up to atomic number 128 |
| `ref_charge` | [N_atom] | Atom charge |
| `ref_atom_name_chars` | [N_atom, 4, 64] | One-hot atom name (4 chars × 64 classes) |
| `ref_space_uid` | [N_atom] | (chain_id, residue_index) → integer |
| `msa` | [N_msa, N_token, 32] | One-hot MSA (32 classes) |
| `profile` | [N_token, 32] | Residue type distribution in MSA |
| `template_distogram` | [N_templ, N, N, 39] | Cβ distance histogram (38 bins, 3.25–50.75 Å) |
| `template_unit_vector` | [N_templ, N, N, 3] | Cα displacement in local frame |

## §3.7.1 Diffusion Training

- **Batch**: 48 diffusion samples per trunk forward
- **Augmentation**: CentreRandomAugmentation (centre, random rotation, random translation with σ=1 Å)
- **Noise sampling**: σ_data × exp(−1.2 + 1.5 × N(0,1))
- **Noise schedule** (inference): t = σ_data × (s_max^(1/p) + t × (s_min^(1/p) − s_max^(1/p)))^p, s_max=160, s_min=4e-4, p=7, 200 steps

**Loss (Eq. 6):**
```
L_diffusion = (t² + σ_data²) / (t + σ_data)² × (L_MSE + α_bond × L_bond) + L_smooth_lddt
```

- **L_MSE** (Eq. 3): weighted aligned MSE with mol-type reweight (α_dna = α_rna = 5, α_ligand = 10)
- **L_bond** (Eq. 5): MSE on bonded atom distances vs GT distances (polymer-ligand bonds only)
- **L_smooth_lddt** (Algo 27): sigmoid-scored distance-difference at cutoffs {0.5, 1.0, 2.0, 4.0} Å; nucleotide inclusion radius 30 Å vs 15 Å for proteins

## §5 Training and Inference

### §5.2 Training Stages

| | Stage 1 | Fine-tune 1 | Fine-tune 2 | Fine-tune 3 |
|---|---|---|---|---|
| Crop size | 384 | 640 | 768 | 768 |
| Init from | Random | Stage 1 | Fine-tune 1 | Fine-tune 2 |
| Diffusion batch | 48 | 32 | 32 | 32 |
| Bond loss weight | 0 | 1 | 1 | 1 |
| Smooth LDDT | On | **Off** | On | N/A |
| Train structure | Yes | Yes | Yes | No |
| Train PAE | No | No | No | Yes |
| Disorder PDB weight | 0.02 | 0.01 | 0.02 | 0.02 |
| TF distillation | No | No | Yes | Yes |
| Max chains | 20 | 20 | 20 | 50 |
| Samples (M) | ~20 | ~1.5 | ~1.5 | ~1.8 |
| Days (256 A100s) | ~10 | ~3 | ~5 | ~2 |

### §5.3 Loss

```
L = α_conf × (L_plddt + L_pde + L_resolved + α_pae × L_pae) + α_diff × L_diffusion + α_disto × L_distogram
```

| Weight | Value |
|--------|-------|
| α_confidence | 1e-4 |
| α_diffusion | 4.0 |
| α_distogram | 3e-2 |
| α_pae | 0 (stage 3: 1.0) |

Confidence losses gated on resolution ∈ [0.1, 4.0] Å.

### §5.4 Optimization

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| β₁, β₂ | 0.9, 0.95 |
| ε | 1e-8 |
| Learning rate | 1.8e-3 |
| Warmup | Linear, 1000 steps |
| LR decay | ×0.95 every 50,000 steps |
| Batch size | 256 |
| Gradient clipping | L2 norm > 10 |

### §5.5 Dropout

Applied in MsaModule, Pairformer stack, and template embedder. See respective algorithms for rates (25% for triangle ops, etc.).

### §5.6 Inference

- EMA of trained parameters, decay 0.999
- No cropping during inference
