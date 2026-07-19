"""Per-token masking abstraction for training-time loss control.

Supports multiple named mask channels (e.g., "fixed" for distillation datasets,
"region" for CDR annotation) propagated through the data pipeline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from deepfold3.data.io_cache import read_npz

# Reserved channel names — use these constants to avoid stringly-typed coupling.
MASK_FIXED = "fixed"
MASK_REGION = "region"
MASK_LOSS_WEIGHT = "loss_weight"


@dataclass
class TokenMask:
    """Per-token mask channels, propagated through the data pipeline.

    Each channel is a named ``np.ndarray`` of shape ``[N_tokens]``.

    Reserved channel names:
        ``"fixed"`` (float32): Per-token confidence that the atom is from
            experimental data. 1.0 = original experimental, 0.0 = model-generated
            (e.g. PATCHR ATLAS distillation). Intermediate values allowed.
        ``"region"`` (int16): Semantic region label ID
            (0 = unset, 1+ = application-defined, e.g. CDR-H1/H2/H3).
        ``"loss_weight"`` (float32): Per-token multiplicative loss weight.
            1.0 = full supervision, 0.0 = excluded from all per-atom and
            per-pair losses. Used to drop unreliable supervision (e.g.
            PATCHR-inpainted ligand atoms) while keeping the affected
            tokens in the model input. Typically synthesised by the
            dataset from ``DatasetConfig.exclude_loss_mol_types`` instead
            of being precomputed.
    """

    channels: dict[str, np.ndarray]

    def select(self, indices: np.ndarray) -> TokenMask:
        """Return a new TokenMask with only the selected token indices."""
        return TokenMask({k: v[indices] for k, v in self.channels.items()})

    def get(self, name: str) -> np.ndarray | None:
        """Get a channel by name, or ``None`` if absent."""
        return self.channels.get(name)

    @classmethod
    def load(cls, path: str | np.lib.npyio.NpzFile) -> TokenMask:
        """Load from an NPZ file. Each array in the file becomes a channel."""
        if isinstance(path, np.lib.npyio.NpzFile):
            npz = path
        else:
            npz = read_npz(path)
        return cls({k: npz[k] for k in npz.files})

    @staticmethod
    def fixed_from_patchr(
        json_path: str | Path,
        tokens: np.ndarray,
        chains: np.ndarray,
        chain_infos: list | None = None,
    ) -> np.ndarray:
        """Build a ``"fixed"`` channel from a PATCHR ATLAS inpainting metadata JSON.

        Args:
            json_path: Path to ``{pdb}_{chains}_inpainting_metadata.json``.
            tokens: TOKEN_DTYPE array ``[N_tokens]`` (needs ``asym_id``,
                ``residue_index``, ``mol_type``).
            chains: CHAIN_DTYPE array from Structure (needs ``name``, ``asym_id``).
            chain_infos: Optional list of ChainInfo from manifest. When provided,
                ``auth_chain_name`` is used to match JSON chain names to Structure
                chains (handles assembly chain renaming).

        Returns:
            float32 array ``[N_tokens]``: 1.0 = fully fixed (experimental),
            0.0 = fully inpainted, fraction for partially fixed.
        """
        with open(json_path) as f:
            meta = json.load(f)

        # Build auth_chain_name → asym_id mapping.
        # Prefer auth_chain_name from ChainInfo (original CIF chain name before
        # assembly renaming); fall back to Structure chain name.
        name_to_asym: dict[str, int] = {}
        if chain_infos is not None:
            for ci in chain_infos:
                key = ci.auth_chain_name if ci.auth_chain_name is not None else ci.chain_name
                name_to_asym[key] = ci.chain_id
        else:
            for c in chains:
                name_to_asym[c["name"].decode("ascii").strip()] = int(c["asym_id"])

        # Build per-residue lookup: (asym_id, res_id) → fixed_value
        # and per-chain aggregate for ligands: asym_id → fixed_value
        from deepfold3.data import const

        fixed_lookup: dict[tuple[int, int], float] = {}
        chain_aggregate: dict[int, float] = {}  # for non-polymer chains
        for chain_name, info in meta["chains"].items():
            asym_id = name_to_asym.get(chain_name)
            if asym_id is None:
                continue
            n_fixed = len(info["fully_fixed_residues"])
            n_partial = len(info["partially_fixed_residues"])
            n_inpainted = len(info["fully_inpainted_residues"])
            for res_id in info["fully_fixed_residues"]:
                fixed_lookup[(asym_id, res_id)] = 1.0
            for entry in info["partially_fixed_residues"]:
                total = entry["total_atoms"]
                frac = entry["fixed_atoms"] / total if total > 0 else 0.0
                fixed_lookup[(asym_id, entry["residue"])] = frac
            for res_id in info["fully_inpainted_residues"]:
                fixed_lookup[(asym_id, res_id)] = 0.0
            # Aggregate for ligand chains (1 residue → many atom-tokens)
            n_total = n_fixed + n_partial + n_inpainted
            if n_total > 0:
                partial_sum = sum(
                    e["fixed_atoms"] / e["total_atoms"] if e["total_atoms"] > 0 else 0.0
                    for e in info["partially_fixed_residues"]
                )
                chain_aggregate[asym_id] = (n_fixed + partial_sum) / n_total

        # Map to token-level
        N = len(tokens)
        fixed = np.ones(N, dtype=np.float32)  # default: fully fixed
        for i in range(N):
            asym_id = int(tokens[i]["asym_id"])
            mol_type = int(tokens[i]["mol_type"])
            if mol_type == const.MOL_NONPOLYMER:
                # Ligand: all atom-tokens get the chain-level aggregate
                if asym_id in chain_aggregate:
                    fixed[i] = chain_aggregate[asym_id]
            else:
                key = (asym_id, int(tokens[i]["residue_index"]))
                if key in fixed_lookup:
                    fixed[i] = fixed_lookup[key]

        return fixed

    @staticmethod
    def loss_weight_excluding_mol_types(
        tokens: np.ndarray,
        exclude: tuple[int, ...] | list[int],
    ) -> np.ndarray:
        """Build a ``"loss_weight"`` channel that zeros tokens whose ``mol_type`` is in *exclude*.

        Returns a float32 array of shape ``[N_tokens]`` with 0.0 on every
        token whose ``mol_type`` matches any value in *exclude* and 1.0
        elsewhere. ``exclude=()`` returns an all-ones array.

        Used by datasets that mark some mol_types as "do not supervise"
        (e.g. PATCHR ligands when their poses are inpainted).
        """
        N = len(tokens)
        out = np.ones(N, dtype=np.float32)
        if not exclude:
            return out
        exclude_set = {int(m) for m in exclude}
        for i in range(N):
            if int(tokens[i]["mol_type"]) in exclude_set:
                out[i] = 0.0
        return out
