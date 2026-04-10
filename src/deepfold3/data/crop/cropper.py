"""Cropper: selects a subset of tokens from a Tokenized structure.

Two strategies (randomly weighted):
  1. Contiguous: pick a chain, then a start position, take consecutive tokens.
  2. Spatial:    pick a seed token, expand by 3D proximity (KD-tree).
"""

from __future__ import annotations

import random

import numpy as np

from deepfold3.data.types import ATOM_DTYPE, BOND_DTYPE, TOKEN_BOND_DTYPE, TOKEN_DTYPE, Structure, Tokenized
from deepfold3.data.types import MSA as MSAType


class Cropper:
    """Crop a Tokenized input to at most max_tokens / max_atoms.

    Three crop strategies (Protenix-style weights):
      1. Contiguous (20%): pick a chain, walk sequentially.
      2. Spatial (40%): pick a random seed token, expand by 3D proximity.
      3. Interface-seeded spatial (40%): seed from interface tokens (within
         ``interface_distance`` of another chain), then expand spatially.

    When no interface tokens exist, interface crop falls back to spatial.
    """

    def __init__(
        self,
        max_neighborhood: int = 40,
        contiguous_weight: float = 0.2,
        spatial_weight: float = 0.4,
        interface_weight: float = 0.4,
        interface_distance: float = 15.0,
    ):
        self.max_neighborhood = max_neighborhood
        self.contiguous_weight = contiguous_weight
        self.spatial_weight = spatial_weight
        self.interface_weight = interface_weight
        self.interface_distance = interface_distance

    def crop(
        self,
        data: Tokenized,
        max_tokens: int,
        max_atoms: int,
        chain_id: int | None = None,
        anchor: tuple[str, int, int] | None = None,
        seed: int | None = None,
    ) -> Tokenized:
        """Crop tokens to ≤ max_tokens and ≤ max_atoms.

        Parameters
        ----------
        data :       Tokenized input
        max_tokens : Maximum number of tokens
        max_atoms :  Maximum number of atoms (across all tokens)
        chain_id :   Legacy: if given, force contiguous crop from this chain.
        anchor :     ``(kind, anchor_a, anchor_b)`` from the weighted sampler.
            ``kind="chain"`` → contiguous crop on ``anchor_a``;
            ``kind="interface"`` → interface-seeded spatial crop using
            tokens at the interface between ``anchor_a`` and ``anchor_b``;
            ``None`` → fall back to weighted random strategy choice. AF3
            §2.5.5 + Protenix's design route the sampler's chosen
            chain/interface directly to the cropper so the per-anchor
            weighting actually drives crop coverage.
        seed :       Random seed

        Returns
        -------
        Tokenized with a subset of tokens (and updated indices)
        """
        tokens = data.tokens
        N = len(tokens)

        if N == 0:
            return data

        # Quick check: if already within limits, return as-is
        total_atoms = int(tokens["atom_num"].sum())
        if N <= max_tokens and total_atoms <= max_atoms:
            return data

        rng = random.Random(seed)

        if chain_id is not None:
            selected = self._contiguous_crop(tokens, max_tokens, max_atoms, chain_id, rng)
        elif anchor is not None:
            kind, a, b = anchor
            if kind == "chain":
                selected = self._contiguous_crop(tokens, max_tokens, max_atoms, a, rng)
            elif kind == "interface":
                selected = self._interface_spatial_crop(
                    tokens, max_tokens, max_atoms, rng, anchor_chains=(a, b)
                )
            else:
                selected = self._spatial_crop(tokens, max_tokens, max_atoms, rng)
        else:
            # Weighted random choice among three strategies (no anchor)
            r = rng.random()
            if r < self.contiguous_weight:
                selected = self._contiguous_crop(tokens, max_tokens, max_atoms, None, rng)
            elif r < self.contiguous_weight + self.interface_weight:
                selected = self._interface_spatial_crop(tokens, max_tokens, max_atoms, rng)
            else:
                selected = self._spatial_crop(tokens, max_tokens, max_atoms, rng)

        return self._reindex(data, selected)

    # ------------------------------------------------------------------
    # Contiguous cropping
    # ------------------------------------------------------------------

    def _contiguous_crop(
        self,
        tokens: np.ndarray,
        max_tokens: int,
        max_atoms: int,
        chain_id: int | None,
        rng: random.Random,
    ) -> np.ndarray:
        """Return selected token indices (sorted) for contiguous cropping."""
        if chain_id is not None:
            chain_mask = tokens["asym_id"] == chain_id
        else:
            # Pick a random chain
            chains = np.unique(tokens["asym_id"])
            chosen_chain = rng.choice(chains.tolist())
            chain_mask = tokens["asym_id"] == chosen_chain

        chain_indices = np.where(chain_mask)[0]
        if len(chain_indices) == 0:
            chain_indices = np.arange(len(tokens))

        max_start = max(0, len(chain_indices) - 1)
        start_in_chain = rng.randint(0, max_start)
        start_global = int(chain_indices[start_in_chain])

        # Expand from start_global, collecting tokens sequentially
        selected: list[int] = []
        atom_count = 0
        for i in range(start_global, len(tokens)):
            an = int(tokens[i]["atom_num"])
            if len(selected) >= max_tokens or atom_count + an > max_atoms:
                break
            selected.append(i)
            atom_count += an

        return np.array(selected, dtype=np.int64)

    # ------------------------------------------------------------------
    # Spatial cropping (KD-tree based)
    # ------------------------------------------------------------------

    def _spatial_crop(
        self,
        tokens: np.ndarray,
        max_tokens: int,
        max_atoms: int,
        rng: random.Random,
    ) -> np.ndarray:
        """Return selected token indices (sorted) for spatial cropping."""
        try:
            from scipy.spatial import KDTree
        except ImportError:
            # Fall back to contiguous
            return self._contiguous_crop(tokens, max_tokens, max_atoms, None, rng)

        coords = np.stack(
            [
                tokens["center_x"].astype(np.float64),
                tokens["center_y"].astype(np.float64),
                tokens["center_z"].astype(np.float64),
            ],
            axis=1,
        )

        # Replace zeros (missing coords) with centroid
        valid = ~np.all(coords == 0.0, axis=1)
        if valid.sum() == 0:
            return self._contiguous_crop(tokens, max_tokens, max_atoms, None, rng)

        centroid = coords[valid].mean(axis=0)
        coords[~valid] = centroid

        # Pick a seed token
        seed_idx = rng.randint(0, len(tokens) - 1)
        seed_coord = coords[seed_idx : seed_idx + 1]

        tree = KDTree(coords)
        # Query nearest neighbors progressively
        k = min(max_tokens * 2, len(tokens))
        dists, indices = tree.query(seed_coord, k=k)
        indices = np.atleast_1d(indices[0])  # flatten; atleast_1d handles k=1

        selected: list[int] = []
        atom_count = 0
        for idx in indices:
            an = int(tokens[idx]["atom_num"])
            if len(selected) >= max_tokens or atom_count + an > max_atoms:
                break
            selected.append(int(idx))
            atom_count += an

        return np.array(sorted(selected), dtype=np.int64)

    # ------------------------------------------------------------------
    # Interface-seeded spatial cropping
    # ------------------------------------------------------------------

    def _interface_spatial_crop(
        self,
        tokens: np.ndarray,
        max_tokens: int,
        max_atoms: int,
        rng: random.Random,
        anchor_chains: tuple[int, int] | None = None,
    ) -> np.ndarray:
        """Spatial crop seeded from interface tokens (near another chain).

        Finds tokens whose representative atom is within ``interface_distance``
        of a token from a different chain. If ``anchor_chains`` is given,
        the candidate seeds are restricted to interface tokens **between
        those two specific chains** so the sampler's interface choice
        actually drives the crop. Without an anchor, all interfaces are
        eligible. Falls back to standard spatial crop if no qualifying
        interface tokens exist.
        """
        try:
            from scipy.spatial import KDTree
        except ImportError:
            return self._spatial_crop(tokens, max_tokens, max_atoms, rng)

        coords = np.stack(
            [
                tokens["center_x"].astype(np.float64),
                tokens["center_y"].astype(np.float64),
                tokens["center_z"].astype(np.float64),
            ],
            axis=1,
        )

        valid = ~np.all(coords == 0.0, axis=1)
        if valid.sum() < 2:
            return self._spatial_crop(tokens, max_tokens, max_atoms, rng)

        centroid = coords[valid].mean(axis=0)
        coords[~valid] = centroid

        # Find interface tokens: tokens within interface_distance of another chain.
        # When ``anchor_chains`` is given, only count tokens of those two
        # specific chains as eligible seeds (the sampler's interface
        # selection drives which interface gets cropped).
        asym_ids = tokens["asym_id"]
        tree = KDTree(coords)
        interface_indices: list[int] = []

        if anchor_chains is not None:
            a, b = anchor_chains
            for i in range(len(tokens)):
                if not valid[i]:
                    continue
                if asym_ids[i] not in (a, b):
                    continue
                neighbors = tree.query_ball_point(coords[i], self.interface_distance)
                other = b if asym_ids[i] == a else a
                if any(asym_ids[j] == other for j in neighbors):
                    interface_indices.append(i)
        else:
            for i in range(len(tokens)):
                if not valid[i]:
                    continue
                neighbors = tree.query_ball_point(coords[i], self.interface_distance)
                for j in neighbors:
                    if asym_ids[j] != asym_ids[i]:
                        interface_indices.append(i)
                        break

        if not interface_indices:
            return self._spatial_crop(tokens, max_tokens, max_atoms, rng)

        # Pick seed from interface tokens, then expand spatially
        seed_idx = rng.choice(interface_indices)
        seed_coord = coords[seed_idx : seed_idx + 1]

        k = min(max_tokens * 2, len(tokens))
        dists, indices = tree.query(seed_coord, k=k)
        indices = np.atleast_1d(indices[0])

        selected: list[int] = []
        atom_count = 0
        for idx in indices:
            an = int(tokens[idx]["atom_num"])
            if len(selected) >= max_tokens or atom_count + an > max_atoms:
                break
            selected.append(int(idx))
            atom_count += an

        return np.array(sorted(selected), dtype=np.int64)

    # ------------------------------------------------------------------
    # Re-index after crop
    # ------------------------------------------------------------------

    def _reindex(self, data: Tokenized, selected: np.ndarray) -> Tokenized:
        """Build a new Tokenized with only the selected token indices."""
        if len(selected) == 0:
            empty_tokens = np.empty(0, dtype=TOKEN_DTYPE)
            empty_bonds = np.empty(0, dtype=TOKEN_BOND_DTYPE)
            return Tokenized(
                tokens=empty_tokens,
                bonds=empty_bonds,
                structure=data.structure,
                paired_msa=data.paired_msa,
                unpaired_msa=data.unpaired_msa,
            )

        old_tokens = data.tokens[selected]

        # Build a mapping old token_idx → new token_idx
        old_to_new: dict[int, int] = {int(old_tokens[i]["token_idx"]): i for i in range(len(old_tokens))}

        # Create new tokens array with updated token_idx
        new_tokens = old_tokens.copy()
        for i in range(len(new_tokens)):
            new_tokens[i]["token_idx"] = i

        # Filter bonds: keep only bonds where both endpoints are in selected set
        valid_old_idxs = set(old_to_new.keys())
        new_bonds_list: list[tuple] = []
        for bond in data.bonds:
            t1, t2 = int(bond["token_1"]), int(bond["token_2"])
            if t1 in valid_old_idxs and t2 in valid_old_idxs:
                new_bonds_list.append((old_to_new[t1], old_to_new[t2]))

        new_bonds = (
            np.array(new_bonds_list, dtype=TOKEN_BOND_DTYPE) if new_bonds_list else np.empty(0, dtype=TOKEN_BOND_DTYPE)
        )

        # Crop the atom arrays to only atoms referenced by selected tokens
        # Collect atom ranges
        atom_indices: list[int] = []
        for tok in new_tokens:
            a_start = int(tok["atom_idx"])
            a_num = int(tok["atom_num"])
            atom_indices.extend(range(a_start, a_start + a_num))

        if atom_indices:
            atom_sel = np.array(atom_indices, dtype=np.int64)
            new_atoms = data.structure.atoms[atom_sel]
            # Remap atom_idx in tokens
            old_to_new_atom = {old: new for new, old in enumerate(atom_indices)}
            for i in range(len(new_tokens)):
                old_start = int(old_tokens[i]["atom_idx"])
                new_tokens[i]["atom_idx"] = old_to_new_atom.get(old_start, 0)
        else:
            new_atoms = np.empty(0, dtype=ATOM_DTYPE)

        # Build cropped structure (bonds not remapped for now)
        new_structure = Structure(
            atoms=new_atoms,
            bonds=np.empty(0, dtype=BOND_DTYPE),
            residues=data.structure.residues,
            chains=data.structure.chains,
        )

        # Crop paired and unpaired MSA to match selected tokens (by asym_id).
        # For each chain, find which positions within the original chain were
        # selected, then index the corresponding MSA columns.
        # Build per-chain column index mapping: for each asym_id, the indices
        # within the original chain ordering that were selected.
        chain_col_indices: dict[int, np.ndarray] = {}
        for asym_id in set(int(t["asym_id"]) for t in new_tokens):
            orig_chain_mask = data.tokens["asym_id"] == asym_id
            orig_chain_positions = np.where(orig_chain_mask)[0]
            keep = np.isin(orig_chain_positions, selected)
            chain_col_indices[asym_id] = np.where(keep)[0].astype(np.int64)

        def _crop_msa_dict(msa_dict: dict[int, MSAType]) -> dict[int, MSAType]:
            cropped: dict[int, MSAType] = {}
            for asym_id, msa in msa_dict.items():
                if asym_id not in chain_col_indices:
                    continue
                cols = chain_col_indices[asym_id]
                if len(cols) == 0:
                    continue
                cropped[asym_id] = MSAType(
                    sequences=msa.sequences[:, cols],
                    mask=msa.mask[:, cols],
                    deletion_matrix=msa.deletion_matrix[:, cols],
                )
            return cropped

        return Tokenized(
            tokens=new_tokens,
            bonds=new_bonds,
            structure=new_structure,
            paired_msa=_crop_msa_dict(data.paired_msa),
            unpaired_msa=_crop_msa_dict(data.unpaired_msa),
            token_mask=data.token_mask.select(selected) if data.token_mask is not None else None,
        )
