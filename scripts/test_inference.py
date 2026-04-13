#!/usr/bin/env python3
"""Test inference on training examples.

Loads prepared structures from the training data directory, runs the model
forward pass, writes predicted mmCIFs, and reports quality metrics (CA-CA
clashes, consecutive CA distances, pLDDT).

Usage::

    # Single structure
    python scripts/test_inference.py --data-dir data/patchr_training \
        --params resources/alphafold3.safetensors --pdb-ids 4ttp,2n3k

    # Multiple structures, save to directory
    python scripts/test_inference.py --data-dir data/patchr_training \
        --params resources/alphafold3.safetensors --pdb-ids 4ttp,2n3k,8gbq \
        --output data/inference_test/results --num-samples 1

    # Use CCD cache for better ref_pos
    python scripts/test_inference.py --data-dir data/patchr_training \
        --params resources/alphafold3.safetensors --pdb-ids 4ttp \
        --ccd-cache data/ccd_cache.json
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def compute_ca_metrics(structure) -> dict:
    """Compute CA-CA clash and consecutive CA distance metrics."""
    from deepfold3.data import const

    atoms = structure.atoms
    chains = structure.chains
    residues = structure.residues

    all_ca_pos = []
    all_ca_chain = []
    consec_dists = []

    for chain in chains:
        mol_type = int(chain["mol_type"])
        if mol_type != const.MOL_PROTEIN:
            continue
        asym_id = int(chain["asym_id"])
        res_start = int(chain["res_idx"])
        res_end = res_start + int(chain["res_num"])

        chain_ca_pos = []
        for ri in range(res_start, res_end):
            res = residues[ri]
            atom_start = int(res["atom_idx"])
            atom_end = atom_start + int(res["atom_num"])
            for ai in range(atom_start, atom_end):
                a = atoms[ai]
                if a["name"].decode("ascii").strip() == "CA" and a["is_present"]:
                    pos = np.array([a["x"], a["y"], a["z"]], dtype=np.float32)
                    chain_ca_pos.append(pos)
                    all_ca_pos.append(pos)
                    all_ca_chain.append(asym_id)
                    break

        # Consecutive CA distances
        for i in range(len(chain_ca_pos) - 1):
            d = np.linalg.norm(chain_ca_pos[i + 1] - chain_ca_pos[i])
            consec_dists.append(d)

    metrics = {}
    if not all_ca_pos:
        return {"n_ca": 0}

    ca_pos = np.stack(all_ca_pos)
    ca_chain = np.array(all_ca_chain)
    n_ca = len(ca_pos)
    metrics["n_ca"] = n_ca

    # Consecutive CA stats
    if consec_dists:
        consec = np.array(consec_dists)
        metrics["consec_ca_mean"] = float(consec.mean())
        metrics["consec_ca_max"] = float(consec.max())
        metrics["consec_ca_gt10"] = int((consec > 10.0).sum())

    # All-vs-all CA clashes (< 1.7 Å between different chains)
    from scipy.spatial import KDTree

    tree = KDTree(ca_pos)
    pairs = tree.query_pairs(r=1.7)
    inter_clashes = sum(1 for i, j in pairs if ca_chain[i] != ca_chain[j])
    intra_clashes = sum(1 for i, j in pairs if ca_chain[i] == ca_chain[j] and abs(i - j) > 1)
    metrics["inter_chain_clashes"] = inter_clashes
    metrics["intra_chain_clashes"] = intra_clashes

    # Steric clashes (< 3.0 Å, all CA pairs excluding bonded neighbors)
    pairs_30 = tree.query_pairs(r=3.0)
    steric = sum(1 for i, j in pairs_30 if abs(i - j) > 1)
    metrics["steric_clashes_3A"] = steric

    return metrics


def run_inference(
    data_dir: Path,
    params_path: Path,
    pdb_ids: list[str],
    output_dir: Path,
    num_recycles: int = 10,
    num_samples: int = 1,
    ccd_cache: str | None = None,
    device: str = "cuda",
):
    from deepfold3.data import const
    from deepfold3.data.crop.cropper import Cropper
    from deepfold3.data.feature.featurizer import Featurizer
    from deepfold3.data.msa.paths import unpaired_msa_path
    from deepfold3.data.paths import structure_path
    from deepfold3.data.prepare.manifest import load_manifest
    from deepfold3.data.tokenize.tokenizer import Tokenizer
    from deepfold3.data.types import Input, MSA, Structure
    from deepfold3.io.mmcif import save_mmcif
    from deepfold3.io.structure_builder import prediction_to_structure
    from deepfold3.io.structure_mmcif import save_structure_mmcif
    from deepfold3.model.config import ModelConfig
    from deepfold3.model.model import DeepFold3

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load CCD cache if provided
    if ccd_cache:
        from deepfold3.data.ccd import load_ccd_cache, set_ccd_components_path
        from deepfold3.data.feature.ref_atoms import set_ref_pos_cache

        _, ref_pos, _ = load_ccd_cache(ccd_cache)
        set_ref_pos_cache(ref_pos)
        log.info("Loaded CCD cache: %d ref_pos entries", len(ref_pos))

    # Load manifest
    records = load_manifest(data_dir / "manifest.json")
    record_map = {r.id: r for r in records}
    log.info("Loaded %d records from manifest", len(records))

    # Build model
    model_cfg = ModelConfig()
    model = DeepFold3(model_cfg)
    if params_path.suffix == ".safetensors":
        from safetensors.torch import load_file
        state = load_file(str(params_path), device=device)
    else:
        ckpt = torch.load(str(params_path), map_location=device, weights_only=False)
        state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model = model.to(device).eval()
    log.info("Model loaded from %s (%d keys)", params_path, len(state))

    tokenizer = Tokenizer()
    featurizer = Featurizer()

    for pdb_id in pdb_ids:
        record = record_map.get(pdb_id)
        if record is None:
            log.warning("Record %s not found in manifest, skipping", pdb_id)
            continue

        sp = structure_path(data_dir / "structures", record.id)
        if not sp.exists():
            log.warning("Structure %s not found, skipping", sp)
            continue

        structure = Structure.load(sp)
        n_chains = sum(1 for c in record.chains if c.valid)
        n_residues = sum(c.num_residues for c in record.chains if c.valid)

        # Load MSAs
        msa_dir = data_dir / "msa"
        unpaired_msa = {}
        for ci in record.chains:
            if not ci.valid:
                continue
            mp = unpaired_msa_path(msa_dir, str(ci.msa_id))
            if mp.exists():
                unpaired_msa[ci.chain_id] = MSA.load(mp)

        msa_depth = sum(m.sequences.shape[0] for m in unpaired_msa.values()) if unpaired_msa else 0
        log.info("=== %s: %d chains, %d residues, MSA depth %d ===", pdb_id, n_chains, n_residues, msa_depth)

        # Tokenize + featurize
        inp = Input(structure=structure, paired_msa={}, unpaired_msa=unpaired_msa)
        tokenized = tokenizer.tokenize(inp)
        features = featurizer.process(tokenized, training=False, random_rot=False)

        # Move to device + add batch dim
        batched = {k: v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v for k, v in features.items()}

        t0 = time.time()
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(
                batched,
                num_recycles=num_recycles,
                num_samples=num_samples,
            )
        dt = time.time() - t0
        log.info("  Forward pass: %.1fs", dt)

        del batched

        # Squeeze batch dim from output (model returns [B, S, ...], we need [S, ...])
        import dataclasses as _dc

        output = _dc.replace(
            output,
            **{f.name: getattr(output, f.name).squeeze(0) for f in _dc.fields(output) if isinstance(getattr(output, f.name), torch.Tensor)},
        )

        # Save mmCIF per sample and compute metrics
        for s in range(num_samples):
            pred_structure = prediction_to_structure(
                output, features, sample_idx=s, name=f"{pdb_id}_s{s}"
            )

            cif_path = output_dir / f"{pdb_id}_s{s}.cif"
            save_structure_mmcif(cif_path, pred_structure, entry_id=pdb_id)

            # Compute metrics
            metrics = compute_ca_metrics(pred_structure)

            # pLDDT: convert logits → [0, 100] via compute_plddt
            if hasattr(output, "plddt") and output.plddt is not None:
                from deepfold3.io.confidences import compute_plddt
                plddt_per_atom, _ = compute_plddt(output.plddt.cpu().float(), features["ref_mask"].float())
                atom_mask = features["ref_mask"].bool()
                mean_plddt = plddt_per_atom[s][atom_mask].mean().item()
            else:
                mean_plddt = float("nan")

            log.info(
                "  sample %d: pLDDT=%.1f | CA=%d | consec_ca: mean=%.2f max=%.2f gt10=%d | "
                "clashes: inter=%d intra=%d steric_3A=%d | → %s",
                s,
                mean_plddt,
                metrics.get("n_ca", 0),
                metrics.get("consec_ca_mean", 0),
                metrics.get("consec_ca_max", 0),
                metrics.get("consec_ca_gt10", 0),
                metrics.get("inter_chain_clashes", 0),
                metrics.get("intra_chain_clashes", 0),
                metrics.get("steric_clashes_3A", 0),
                cif_path,
            )

        del output
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Test inference on training examples")
    parser.add_argument("--data-dir", required=True, type=Path, help="Training data directory")
    parser.add_argument("--params", required=True, type=Path, help="Model weights (.safetensors)")
    parser.add_argument("--pdb-ids", required=True, help="Comma-separated PDB IDs")
    parser.add_argument("--output", default=Path("data/inference_test/results"), type=Path, help="Output directory")
    parser.add_argument("--num-recycles", default=10, type=int)
    parser.add_argument("--num-samples", default=1, type=int)
    parser.add_argument("--ccd-cache", default=None, help="CCD cache JSON for ref_pos")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    pdb_ids = [x.strip() for x in args.pdb_ids.split(",")]
    run_inference(
        data_dir=args.data_dir,
        params_path=args.params,
        pdb_ids=pdb_ids,
        output_dir=args.output,
        num_recycles=args.num_recycles,
        num_samples=args.num_samples,
        ccd_cache=args.ccd_cache,
        device=args.device,
    )


if __name__ == "__main__":
    main()
