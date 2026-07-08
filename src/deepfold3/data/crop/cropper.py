"""Cropper: selects a subset of tokens from a Tokenized structure.

Two strategies (randomly weighted):
  1. Contiguous: pick a chain, then a start position, take consecutive tokens.
  2. Spatial:    pick a seed token, expand by 3D proximity (KD-tree).
"""

from __future__ import annotations

import random

import numpy as np

from deepfold3.data.types import (
    ATOM_DTYPE,
    BOND_DTYPE,
    CHAIN_DTYPE,
    RESIDUE_DTYPE,
    TOKEN_BOND_DTYPE,
    TOKEN_DTYPE,
    Structure,
    Tokenized,
)
from deepfold3.data.types import MSA as MSAType


def compute_chain_msa_cols(asym_ids: np.ndarray, msa_col_idx: np.ndarray | None) -> dict[int, np.ndarray]:
    """Per-chain MSA column indices referenced by a set of tokens.

    Maps each asym_id present in ``asym_ids`` to the deduplicated, order-stable
    chain-local MSA column indices its tokens reference (via ``msa_col_idx``).
    Ligand / column-less tokens (``msa_col_idx == -1``) are dropped; chains
    with no MSA columns are omitted. Returns ``{}`` if ``msa_col_idx`` is None.

    Shared by :class:`Cropper` (to crop loaded MSA) and the training dataset's
    crop-then-load path (to load+slice only the MSA of surviving chains).
    """
    chain_msa_cols: dict[int, np.ndarray] = {}
    if msa_col_idx is None:
        return chain_msa_cols
    asym_ids = np.asarray(asym_ids)
    for asym_id in {int(a) for a in asym_ids}:
        mask = asym_ids == asym_id
        cols = msa_col_idx[mask]
        cols = cols[cols >= 0]
        if len(cols) > 0:
            # Deduplicate (expanded non-standard tokens share a column) while
            # preserving order via a stable unique.
            _, idx = np.unique(cols, return_index=True)
            chain_msa_cols[int(asym_id)] = cols[np.sort(idx)]
    return chain_msa_cols


def remap_msa_col_idx_to_slice(
    asym_ids: np.ndarray,
    msa_col_idx: np.ndarray | None,
    chain_msa_cols: dict[int, np.ndarray],
) -> np.ndarray | None:
    """Rebase ``msa_col_idx`` to positions within each chain's column-sliced MSA.

    A crop slices every chain's MSA to ``chain_msa_cols[aid]`` and re-indexes the
    result to ``0..k-1`` (column *j* of the slice = original column
    ``chain_msa_cols[aid][j]``). Token references in ``msa_col_idx``, however,
    still hold the *original* chain-local columns, so a later gather
    (``msa_data[:, msa_col_idx]``) would index the slice with stale values and
    silently drop / mis-align columns. This remaps each valid entry to its rank
    within its chain's ``cols`` so the gather lands on the correct sliced column.

    Ligand / column-less tokens (``-1``) are preserved. Call **exactly once**, at
    the point the MSA is sliced (the cropper's internal slice, or the training
    dataset's crop-then-load slice) — it is not idempotent.
    """
    if msa_col_idx is None:
        return None
    asym_ids = np.asarray(asym_ids)
    remapped = np.full_like(msa_col_idx, -1)
    for aid, cols in chain_msa_cols.items():
        rank = {int(c): j for j, c in enumerate(cols)}
        sel = (asym_ids == aid) & (msa_col_idx >= 0)
        if not sel.any():
            continue
        remapped[sel] = np.array(
            [rank.get(int(v), -1) for v in msa_col_idx[sel]], dtype=msa_col_idx.dtype
        )
    return remapped


class Cropper:
    """Crop a Tokenized input to at most max_tokens / max_atoms.

    The crop *strategy* is sampled from the per-dataset crop weights
    (AF3 §2.7), independently of the sampler anchor:
      1. Contiguous: pick a chain, walk sequentially.
      2. Spatial: pick a seed token, expand by 3D proximity.
      3. Interface-seeded spatial: seed from interface tokens (within
         ``interface_distance`` of another chain), then expand spatially.

    The sampler anchor (a weighted chain or interface) biases *seeding* only
    — a chain anchor becomes the contiguous chain / spatial seed pool, an
    interface anchor restricts interface seeds — but never dictates the
    strategy. When no interface tokens exist, interface crop falls back to
    spatial.
    """

    def __init__(
        self,
        max_neighborhood: int = 40,
        contiguous_weight: float = 0.2,
        spatial_weight: float = 0.4,
        interface_weight: float = 0.4,
        interface_distance: float = 15.0,
        min_resolved: int = 4,
        max_attempts: int = 8,
    ):
        self.max_neighborhood = max_neighborhood
        self.contiguous_weight = contiguous_weight
        self.spatial_weight = spatial_weight
        self.interface_weight = interface_weight
        self.interface_distance = interface_distance
        # IntFold §5.4.3: crops must contain ≥min_resolved residues so Kabsch
        # alignment in the loss is well-posed. Retry up to max_attempts times
        # with fresh RNG streams if a candidate crop falls short.
        self.min_resolved = min_resolved
        self.max_attempts = max_attempts

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

        # Per-token resolved = any atom [atom_idx, atom_idx+atom_num) has
        # is_present=True. Computed via prefix-sum on the atom mask so the
        # per-attempt check is a cheap indexed diff.
        atom_present = data.structure.atoms["is_present"].astype(np.int32)
        present_prefix = np.concatenate([[0], np.cumsum(atom_present)])
        tok_start = tokens["atom_idx"].astype(np.int64)
        tok_end = tok_start + tokens["atom_num"].astype(np.int64)
        token_resolved = (present_prefix[tok_end] - present_prefix[tok_start]) > 0
        total_resolved = int(token_resolved.sum())
        attempts = self.max_attempts if total_resolved >= self.min_resolved else 1

        # Derive seed hints from the sampler anchor. AF3 §2.7 samples the crop
        # *strategy* independently from the per-dataset crop weights and uses
        # the weighted chain/interface only to *seed* a spatial/interface crop
        # — it never dictates the strategy. So the anchor biases seeding only.
        # A legacy explicit ``chain_id`` still forces a contiguous crop.
        anchor_chain: int | None = None
        anchor_iface: tuple[int, int] | None = None
        force_contiguous_chain: int | None = None
        if chain_id is not None:
            force_contiguous_chain = int(chain_id)
            anchor_chain = int(chain_id)
        elif anchor is not None:
            kind, a, b = anchor
            if kind == "chain":
                anchor_chain = int(a)
            elif kind == "interface":
                anchor_iface = (int(a), int(b))
                anchor_chain = int(a)

        cand = None
        for attempt in range(attempts):
            attempt_rng = random.Random(rng.random()) if attempt > 0 else rng

            strategy = "contiguous" if force_contiguous_chain is not None else self._sample_strategy(attempt_rng)

            if strategy == "contiguous":
                cand = self._contiguous_crop(tokens, max_tokens, max_atoms, anchor_chain, attempt_rng)
            elif strategy == "interface":
                cand = self._interface_spatial_crop(
                    tokens,
                    max_tokens,
                    max_atoms,
                    attempt_rng,
                    token_resolved=token_resolved,
                    anchor_chains=anchor_iface,
                )
            else:  # spatial
                cand = self._spatial_crop(
                    tokens,
                    max_tokens,
                    max_atoms,
                    attempt_rng,
                    token_resolved=token_resolved,
                    anchor_chain=anchor_chain,
                )

            if int(token_resolved[cand].sum()) >= self.min_resolved:
                break

        # ``cand`` holds either a crop that cleared ``min_resolved`` or the
        # last-attempt fallback (used when ``total_resolved < min_resolved``,
        # or when no attempt cleared the bar).
        return self._reindex(data, cand)

    # ------------------------------------------------------------------
    # Strategy / seed selection
    # ------------------------------------------------------------------

    def _sample_strategy(self, rng: random.Random) -> str:
        """Sample a crop strategy from the (unnormalized) crop weights.

        AF3 §2.7 draws the crop method independently of the seed
        chain/interface. The cumulative order (contiguous, interface,
        spatial) matches the legacy anchor-free branch.
        """
        total = self.contiguous_weight + self.spatial_weight + self.interface_weight
        if total <= 0:
            return "contiguous"
        r = rng.random() * total
        if r < self.contiguous_weight:
            return "contiguous"
        if r < self.contiguous_weight + self.interface_weight:
            return "interface"
        return "spatial"

    @staticmethod
    def _seed_pool(
        tokens: np.ndarray,
        valid: np.ndarray,
        token_resolved: np.ndarray | None,
        anchor_chain: int | None,
    ) -> list[int]:
        """Candidate seed token indices for a spatial crop.

        Narrows to the tightest non-empty pool in priority order:
        resolved ∧ anchor-chain → resolved → valid-center → all tokens.
        Seeding on a resolved token in the sampler's preferred chain matches
        AF3 §2.7 (crop seeded on the weighted chain) and avoids seeding on a
        missing-coordinate token that was collapsed onto the centroid.
        """
        base = valid.copy()
        if token_resolved is not None:
            r = base & token_resolved
            if r.any():
                base = r
        if anchor_chain is not None:
            a = base & (tokens["asym_id"] == anchor_chain)
            if a.any():
                base = a
        idx = np.where(base)[0]
        if len(idx) == 0:
            idx = np.arange(len(tokens))
        return idx.tolist()

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
        token_resolved: np.ndarray | None = None,
        anchor_chain: int | None = None,
    ) -> np.ndarray:
        """Return selected token indices (sorted) for spatial cropping.

        The seed token is drawn from ``_seed_pool`` (resolved tokens in the
        ``anchor_chain`` when available), so the crop is centred on the
        sampler's preferred chain rather than a uniformly random token.
        """
        try:
            from scipy.spatial import KDTree
        except ImportError:
            # Fall back to contiguous
            return self._contiguous_crop(tokens, max_tokens, max_atoms, anchor_chain, rng)

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
            return self._contiguous_crop(tokens, max_tokens, max_atoms, anchor_chain, rng)

        centroid = coords[valid].mean(axis=0)
        coords[~valid] = centroid

        # Pick a seed token from the preferred (resolved, anchor-chain) pool
        seed_idx = rng.choice(self._seed_pool(tokens, valid, token_resolved, anchor_chain))
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
        token_resolved: np.ndarray | None = None,
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
            return self._spatial_crop(tokens, max_tokens, max_atoms, rng, token_resolved=token_resolved)

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
            return self._spatial_crop(tokens, max_tokens, max_atoms, rng, token_resolved=token_resolved)

        centroid = coords[valid].mean(axis=0)
        coords[~valid] = centroid

        # Only resolved tokens are eligible interface seeds (an unresolved
        # token collapsed onto the centroid must never seed a crop, and two
        # such tokens from different chains sit at distance 0 → a false
        # interface). ``valid`` already excludes centre-less tokens; intersect
        # with ``token_resolved`` when available for the stricter guarantee.
        seed_eligible = valid.copy()
        if token_resolved is not None:
            r = seed_eligible & token_resolved
            if r.any():
                seed_eligible = r

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
                if not seed_eligible[i]:
                    continue
                if asym_ids[i] not in (a, b):
                    continue
                neighbors = tree.query_ball_point(coords[i], self.interface_distance)
                other = b if asym_ids[i] == a else a
                # ``valid[j]`` guards against a centroid-collapsed unresolved
                # neighbour faking a zero-distance cross-chain contact.
                if any(asym_ids[j] == other and valid[j] for j in neighbors):
                    interface_indices.append(i)
        else:
            for i in range(len(tokens)):
                if not seed_eligible[i]:
                    continue
                neighbors = tree.query_ball_point(coords[i], self.interface_distance)
                for j in neighbors:
                    if asym_ids[j] != asym_ids[i] and valid[j]:
                        interface_indices.append(i)
                        break

        if not interface_indices:
            return self._spatial_crop(tokens, max_tokens, max_atoms, rng, token_resolved=token_resolved)

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

    @staticmethod
    def _reindex_hierarchy(
        residues: np.ndarray,
        chains: np.ndarray,
        old_to_new_atom: dict[int, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Re-index the residue / chain hierarchy to the cropped atom array.

        Cropping slices ``atoms`` and remaps each token's ``atom_idx``, but the
        residue / chain records also carry ``atom_idx`` / ``res_idx`` offsets
        into the *original* arrays. Left stale, any post-crop lookup that
        resolves a residue by atom index (``_find_res_name`` / ``_find_residue``
        for the ``ref_pos`` reference conformer, and the ligand-swap
        chain→residue CCD-code lookup) returns the WRONG residue. This rebuilds
        both levels so the structure is self-consistent with the cropped atoms.

        A residue's kept atoms are contiguous in the cropped array (tokens of a
        residue are emitted consecutively and the selection is sorted), so an
        ``(atom_idx, atom_num)`` range is exact. Residues / chains with no kept
        atoms are dropped; ``res_idx`` in chains is remapped to the new residue
        positions.
        """
        # Residues: keep those with ≥1 kept atom; remap atom range.
        kept_res_order: list[int] = []
        new_res_atom_idx: list[int] = []
        new_res_atom_num: list[int] = []
        old_to_new_res: dict[int, int] = {}
        for ridx in range(len(residues)):
            r_start = int(residues[ridx]["atom_idx"])
            r_num = int(residues[ridx]["atom_num"])
            kept = [old_to_new_atom[a] for a in range(r_start, r_start + r_num) if a in old_to_new_atom]
            if not kept:
                continue
            old_to_new_res[ridx] = len(kept_res_order)
            kept_res_order.append(ridx)
            new_res_atom_idx.append(min(kept))
            new_res_atom_num.append(len(kept))

        if kept_res_order:
            new_residues = residues[kept_res_order].copy()
            new_residues["atom_idx"] = np.array(new_res_atom_idx, dtype=new_residues["atom_idx"].dtype)
            new_residues["atom_num"] = np.array(new_res_atom_num, dtype=new_residues["atom_num"].dtype)
        else:
            new_residues = np.empty(0, dtype=RESIDUE_DTYPE)

        # Chains: keep those with ≥1 kept residue; remap res range + atom range.
        kept_chain_order: list[int] = []
        new_chain_res_idx: list[int] = []
        new_chain_res_num: list[int] = []
        new_chain_atom_idx: list[int] = []
        new_chain_atom_num: list[int] = []
        for cidx in range(len(chains)):
            c_start = int(chains[cidx]["res_idx"])
            c_num = int(chains[cidx]["res_num"])
            kept_r = [old_to_new_res[r] for r in range(c_start, c_start + c_num) if r in old_to_new_res]
            if not kept_r:
                continue
            kept_chain_order.append(cidx)
            first_new_res = min(kept_r)
            new_chain_res_idx.append(first_new_res)
            new_chain_res_num.append(len(kept_r))
            new_chain_atom_idx.append(int(new_residues[first_new_res]["atom_idx"]))
            new_chain_atom_num.append(int(sum(int(new_residues[r]["atom_num"]) for r in kept_r)))

        if kept_chain_order:
            new_chains = chains[kept_chain_order].copy()
            new_chains["res_idx"] = np.array(new_chain_res_idx, dtype=new_chains["res_idx"].dtype)
            new_chains["res_num"] = np.array(new_chain_res_num, dtype=new_chains["res_num"].dtype)
            new_chains["atom_idx"] = np.array(new_chain_atom_idx, dtype=new_chains["atom_idx"].dtype)
            new_chains["atom_num"] = np.array(new_chain_atom_num, dtype=new_chains["atom_num"].dtype)
        else:
            new_chains = np.empty(0, dtype=CHAIN_DTYPE)

        return new_residues, new_chains

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
                new_bonds_list.append((old_to_new[t1], old_to_new[t2], int(bond["atom_1"]), int(bond["atom_2"])))

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
            old_to_new_atom = {}
            new_atoms = np.empty(0, dtype=ATOM_DTYPE)

        new_residues, new_chains = self._reindex_hierarchy(
            data.structure.residues, data.structure.chains, old_to_new_atom
        )

        # Build cropped structure (bonds not remapped for now). Residues and
        # chains are re-indexed to stay consistent with the cropped atom array
        # (see ``_reindex_hierarchy``).
        new_structure = Structure(
            atoms=new_atoms,
            bonds=np.empty(0, dtype=BOND_DTYPE),
            residues=new_residues,
            chains=new_chains,
        )

        # Crop paired and unpaired MSA to match selected tokens.
        # Use msa_col_idx (AF3 standard_token_idxs) when available — it
        # correctly maps expanded non-standard residue tokens to their
        # shared MSA column.  Fall back to positional indexing for legacy
        # Tokenized objects without the field.
        chain_msa_cols: dict[int, np.ndarray] = {}
        if data.msa_col_idx is not None:
            # Gather per-chain MSA column indices for the selected tokens.
            chain_msa_cols = compute_chain_msa_cols(new_tokens["asym_id"], data.msa_col_idx[selected])
        else:
            # Legacy fallback: assume 1 token = 1 MSA column (breaks for
            # chains with expanded non-standard residues).
            for asym_id in set(int(t["asym_id"]) for t in new_tokens):
                orig_chain_mask = data.tokens["asym_id"] == asym_id
                orig_chain_positions = np.where(orig_chain_mask)[0]
                keep = np.isin(orig_chain_positions, selected)
                chain_msa_cols[asym_id] = np.where(keep)[0].astype(np.int64)

        def _crop_msa_dict(msa_dict: dict[int, MSAType]) -> dict[int, MSAType]:
            cropped: dict[int, MSAType] = {}
            for asym_id, msa in msa_dict.items():
                if asym_id not in chain_msa_cols:
                    continue
                cols = chain_msa_cols[asym_id]
                if len(cols) == 0:
                    continue
                cropped[asym_id] = MSAType(
                    sequences=msa.sequences[:, cols],
                    mask=msa.mask[:, cols],
                    deletion_matrix=msa.deletion_matrix[:, cols],
                )
            return cropped

        # Crop msa_col_idx to the selected tokens. Rebase it to the column-sliced
        # MSAs *only* when this cropper actually sliced an MSA here (the sliced
        # MSA is re-indexed 0..k-1 while the raw columns are not). In the training
        # crop-then-load path the MSA is empty at crop time and gets sliced later
        # by the dataset, which rebases then — rebasing here would destroy the
        # original columns the dataset needs to slice from disk (double rebase).
        new_msa_col_idx = data.msa_col_idx[selected] if data.msa_col_idx is not None else None
        sliced_msa_here = bool(data.paired_msa) or bool(data.unpaired_msa)
        if new_msa_col_idx is not None and sliced_msa_here:
            new_msa_col_idx = remap_msa_col_idx_to_slice(new_tokens["asym_id"], new_msa_col_idx, chain_msa_cols)

        return Tokenized(
            tokens=new_tokens,
            bonds=new_bonds,
            structure=new_structure,
            paired_msa=_crop_msa_dict(data.paired_msa),
            unpaired_msa=_crop_msa_dict(data.unpaired_msa),
            token_mask=data.token_mask.select(selected) if data.token_mask is not None else None,
            msa_col_idx=new_msa_col_idx,
        )
