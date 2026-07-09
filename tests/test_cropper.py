"""Tests for the Cropper (contiguous / spatial / interface-seeded)."""

from __future__ import annotations

import numpy as np
import pytest

from deepfold3.data.crop.cropper import (
    Cropper,
    compute_chain_msa_cols,
    remap_msa_col_idx_to_slice,
)
from deepfold3.data.types import (
    ATOM_DTYPE,
    BOND_DTYPE,
    CHAIN_DTYPE,
    MSA,
    RESIDUE_DTYPE,
    TOKEN_BOND_DTYPE,
    TOKEN_DTYPE,
    Structure,
    Tokenized,
)

# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def make_tokens(
    n: int,
    *,
    n_chains: int = 1,
    atoms_per_token: int = 5,
    coords: np.ndarray | None = None,
    unresolved: set[int] | None = None,
) -> tuple[np.ndarray, Structure]:
    """Build a synthetic TOKEN_DTYPE array + matching minimal Structure.

    Tokens are distributed across ``n_chains`` (roughly equal sizes). Each
    token claims ``atoms_per_token`` consecutive atoms. If ``coords`` is
    provided (shape [n, 3]), it sets (center_x, center_y, center_z) and also
    the corresponding atom (x, y, z). Otherwise tokens are placed on a
    1D line with spacing 10Å.

    ``unresolved`` is a set of token indices whose atoms are marked
    ``is_present=False`` (so the token is *unresolved*) while KEEPING a non-zero
    center coord (so it is still geometrically ``valid``). This isolates the
    item-10 invariant: exclusion must be driven by the resolved mask, not by the
    zero-coord ``valid`` mask.
    """
    unresolved = unresolved or set()
    if coords is None:
        coords = np.zeros((n, 3), dtype=np.float32)
        coords[:, 0] = np.arange(n) * 10.0

    tokens = np.zeros(n, dtype=TOKEN_DTYPE)

    # Split into chains as evenly as possible.
    chain_assign = np.zeros(n, dtype=np.int32)
    if n_chains > 1:
        boundaries = np.linspace(0, n, n_chains + 1, dtype=int)
        for c in range(n_chains):
            chain_assign[boundaries[c] : boundaries[c + 1]] = c

    atom_cursor = 0
    for i in range(n):
        tokens[i]["token_idx"] = i
        tokens[i]["residue_index"] = i
        tokens[i]["res_type"] = 0
        tokens[i]["mol_type"] = 0
        tokens[i]["asym_id"] = int(chain_assign[i])
        tokens[i]["entity_id"] = int(chain_assign[i])
        tokens[i]["sym_id"] = 0
        tokens[i]["atom_idx"] = atom_cursor
        tokens[i]["atom_num"] = atoms_per_token
        tokens[i]["center_x"] = float(coords[i, 0])
        tokens[i]["center_y"] = float(coords[i, 1])
        tokens[i]["center_z"] = float(coords[i, 2])
        atom_cursor += atoms_per_token

    total_atoms = atom_cursor
    atoms = np.zeros(total_atoms, dtype=ATOM_DTYPE)
    for i in range(n):
        a0 = int(tokens[i]["atom_idx"])
        an = int(tokens[i]["atom_num"])
        present = i not in unresolved
        atoms[a0 : a0 + an]["x"] = float(coords[i, 0])
        atoms[a0 : a0 + an]["y"] = float(coords[i, 1])
        atoms[a0 : a0 + an]["z"] = float(coords[i, 2])
        atoms[a0 : a0 + an]["is_present"] = present
        atoms[a0 : a0 + an]["element"] = 6

    residues = np.zeros(n, dtype=RESIDUE_DTYPE)
    for i in range(n):
        residues[i]["res_id"] = i
        residues[i]["atom_idx"] = int(tokens[i]["atom_idx"])
        residues[i]["atom_num"] = atoms_per_token
        residues[i]["is_present"] = i not in unresolved

    chains = np.zeros(n_chains, dtype=CHAIN_DTYPE)
    for c in range(n_chains):
        chain_mask = chain_assign == c
        first_tok = int(np.where(chain_mask)[0][0])
        last_tok = int(np.where(chain_mask)[0][-1])
        chains[c]["name"] = f"C{c}".encode()
        chains[c]["mol_type"] = 0
        chains[c]["entity_id"] = c
        chains[c]["sym_id"] = 0
        chains[c]["asym_id"] = c
        chains[c]["res_idx"] = first_tok
        chains[c]["res_num"] = last_tok - first_tok + 1
        chains[c]["atom_idx"] = int(tokens[first_tok]["atom_idx"])
        chains[c]["atom_num"] = (last_tok - first_tok + 1) * atoms_per_token

    structure = Structure(
        atoms=atoms,
        bonds=np.empty(0, dtype=BOND_DTYPE),
        residues=residues,
        chains=chains,
    )
    return tokens, structure


def make_tokenized(
    n: int,
    *,
    n_chains: int = 1,
    atoms_per_token: int = 5,
    coords: np.ndarray | None = None,
    with_msa: bool = False,
    with_bonds: bool = False,
    unresolved: set[int] | None = None,
) -> Tokenized:
    tokens, structure = make_tokens(
        n,
        n_chains=n_chains,
        atoms_per_token=atoms_per_token,
        coords=coords,
        unresolved=unresolved,
    )

    paired_msa: dict[int, MSA] = {}
    unpaired_msa: dict[int, MSA] = {}
    if with_msa:
        # One MSA per chain, width matches that chain's token count.
        for c in range(n_chains):
            chain_mask = tokens["asym_id"] == c
            width = int(chain_mask.sum())
            seqs = np.zeros((3, width), dtype=np.int8)
            seqs[0] = np.arange(width, dtype=np.int8) % 20
            mask = np.ones((3, width), dtype=bool)
            delmat = np.zeros((3, width), dtype=np.int8)
            unpaired_msa[c] = MSA(sequences=seqs, mask=mask, deletion_matrix=delmat)
            paired_msa[c] = MSA(
                sequences=seqs.copy(),
                mask=mask.copy(),
                deletion_matrix=delmat.copy(),
            )

    bonds = np.empty(0, dtype=TOKEN_BOND_DTYPE)
    if with_bonds and n >= 4:
        bond_list = [(0, 1, 0, 0), (1, 2, 0, 0), (2, 3, 0, 0), (0, n - 1, 0, 0)]
        bonds = np.array(bond_list, dtype=TOKEN_BOND_DTYPE)

    return Tokenized(
        tokens=tokens,
        bonds=bonds,
        structure=structure,
        paired_msa=paired_msa,
        unpaired_msa=unpaired_msa,
    )


# ---------------------------------------------------------------------------
# 1. Contiguous crop — single chain
# ---------------------------------------------------------------------------


def test_contiguous_crop_single_chain():
    data = make_tokenized(100, n_chains=1)
    cropper = Cropper(contiguous_weight=1.0, spatial_weight=0.0, interface_weight=0.0)
    import random

    # Use chain_id=0 so the start position is deterministic with seed=0:
    # Random(0).randint(0, 99) == 49, leaving 51 tokens ahead → 20 fit.
    selected = cropper._contiguous_crop(data.tokens, max_tokens=20, max_atoms=10_000, chain_id=0, rng=random.Random(0))

    assert len(selected) == 20
    # Sorted (monotonically non-decreasing)
    assert np.all(np.diff(selected) >= 1)
    # All from single chain
    assert set(data.tokens[selected]["asym_id"].tolist()) == {0}
    # Contiguous: [i, i+1, ...]
    assert np.array_equal(selected, np.arange(selected[0], selected[0] + 20))


# ---------------------------------------------------------------------------
# 2. Contiguous crop — atom budget
# ---------------------------------------------------------------------------


def test_contiguous_crop_atom_budget():
    # 50 tokens, 100 atoms each → atom budget of 350 should cap at 3 tokens.
    data = make_tokenized(50, atoms_per_token=100)
    cropper = Cropper()
    import random

    selected = cropper._contiguous_crop(data.tokens, max_tokens=20, max_atoms=350, chain_id=0, rng=random.Random(0))

    # 3 tokens * 100 atoms = 300 ≤ 350, adding a 4th would be 400 > 350.
    assert len(selected) == 3
    # Never exceed the atom budget.
    total_atoms = int(data.tokens[selected]["atom_num"].sum())
    assert total_atoms <= 350


# ---------------------------------------------------------------------------
# 3. Spatial crop — neighborhood
# ---------------------------------------------------------------------------


def test_spatial_crop_neighborhood():
    pytest.importorskip("scipy")
    rng_np = np.random.default_rng(7)
    n = 200

    coords = np.zeros((n, 3), dtype=np.float32)
    # First 30: clustered near origin in a ball of radius 2
    cluster_size = 30
    cluster_offsets = rng_np.uniform(-2.0, 2.0, size=(cluster_size, 3))
    coords[:cluster_size] = cluster_offsets
    # Rest: placed far away (> 1000 Å) in a distant line
    coords[cluster_size:, 0] = np.arange(n - cluster_size) * 50.0 + 10_000.0

    data = make_tokenized(n, coords=coords)
    cropper = Cropper()
    import random

    # Force seed_idx = 0 via a Random whose first randint(0, n-1) lands in the
    # cluster. seed=0 → Random(0) — verify explicitly.
    rng = random.Random(0)
    selected = cropper._spatial_crop(data.tokens, max_tokens=cluster_size, max_atoms=10_000_000, rng=rng)

    assert len(selected) == cluster_size
    # The seed was chosen via randint(0, n-1); we don't know which seed landed
    # without re-running rng. Instead verify the returned tokens are mutually
    # close (spatial coherence) by comparing against *some* seed. Since only
    # the 30 cluster tokens are within 2Å of any cluster token, and the others
    # are > 9000Å away, the crop must be either (a) entirely within the
    # cluster (if seed was in cluster) or (b) a contiguous far-away run.
    sel_coords = coords[selected]
    in_cluster = np.all(np.abs(sel_coords) < 10.0, axis=1)
    if in_cluster.all():
        # Seed was in the cluster → selection is exactly the 30-point cluster.
        assert set(selected.tolist()) == set(range(cluster_size))
    else:
        # Seed was far-away; at least verify all selected tokens are far from
        # the cluster (spatial coherence held) and none came from the cluster.
        assert not in_cluster.any()


def test_spatial_crop_neighborhood_forced_seed():
    """Version that forces the seed into the cluster via chain layout."""
    pytest.importorskip("scipy")

    n = 200
    cluster_size = 30
    coords = np.zeros((n, 3), dtype=np.float32)
    # Place entire cluster at fixed small offsets.
    coords[:cluster_size, 0] = np.linspace(-1.0, 1.0, cluster_size)
    # Far-away tokens
    coords[cluster_size:, 0] = np.arange(n - cluster_size) * 50.0 + 10_000.0

    data = make_tokenized(n, coords=coords)
    cropper = Cropper()

    # Try multiple seeds to find one that lands in the cluster.
    import random

    for seed in range(50):
        rng = random.Random(seed)
        # Simulate the internal seed_idx pick:
        probe = random.Random(seed).randint(0, n - 1)
        if probe < cluster_size:
            selected = cropper._spatial_crop(data.tokens, max_tokens=cluster_size, max_atoms=10_000_000, rng=rng)
            assert set(selected.tolist()) == set(range(cluster_size))
            return
    pytest.fail("No seed landed in cluster in 50 tries")


# ---------------------------------------------------------------------------
# 4. Interface crop — finds interface
# ---------------------------------------------------------------------------


def test_interface_crop_finds_interface():
    pytest.importorskip("scipy")
    n = 40
    n_a = 20
    coords = np.zeros((n, 3), dtype=np.float32)
    # Chain A on a line x=[0..19] * 30 (far apart within A → no self-neighbors
    # via interface_distance, but that's fine — interface test is cross-chain)
    # Actually keep A spread out so only the last 5 tokens are near B.
    coords[:n_a, 0] = np.arange(n_a) * 30.0

    # Chain B clustered near A's last 5 tokens (indices 15..19, x=450..570).
    # Place B near A[15] = 450.
    coords[n_a:, 0] = 455.0 + np.arange(n - n_a) * 3.0  # 455..515
    coords[n_a:, 1] = 5.0  # small y offset, all within 15Å of A[15..19]

    data = make_tokenized(n, n_chains=2, coords=coords)

    # Verify which A tokens are interface tokens (within 15Å of any B).
    import random

    cropper = Cropper(interface_distance=15.0)

    # Find interface tokens on A manually.
    # A token i is an interface token iff min |a_i - b_j| <= 15.
    a_coords = coords[:n_a]
    b_coords = coords[n_a:]
    dists = np.linalg.norm(a_coords[:, None, :] - b_coords[None, :, :], axis=-1)
    min_d_to_b = dists.min(axis=1)
    interface_a = np.where(min_d_to_b <= 15.0)[0]
    assert len(interface_a) >= 1, "Test setup: no interface tokens on A"

    # Run the cropper.
    selected = cropper._interface_spatial_crop(data.tokens, max_tokens=15, max_atoms=10_000_000, rng=random.Random(3))

    assert len(selected) > 0
    # The crop must span both chains (the seed is near B, so nearest neighbors
    # will include some B tokens too).
    asyms = set(data.tokens[selected]["asym_id"].tolist())
    assert 0 in asyms and 1 in asyms, f"Expected both chains, got {asyms}"


# ---------------------------------------------------------------------------
# 5. Interface crop — falls back on monomer
# ---------------------------------------------------------------------------


def test_interface_crop_falls_back_for_monomer():
    pytest.importorskip("scipy")
    n = 50
    data = make_tokenized(n, n_chains=1)
    cropper = Cropper(interface_distance=15.0)

    import random

    sel_iface = cropper._interface_spatial_crop(data.tokens, max_tokens=20, max_atoms=10_000_000, rng=random.Random(11))
    sel_spatial = cropper._spatial_crop(data.tokens, max_tokens=20, max_atoms=10_000_000, rng=random.Random(11))

    assert np.array_equal(sel_iface, sel_spatial)


# ---------------------------------------------------------------------------
# 6. Dispatch weights
# ---------------------------------------------------------------------------


def test_crop_dispatch_no_crop_needed():
    data = make_tokenized(10, atoms_per_token=5)
    cropper = Cropper()
    out = cropper.crop(data, max_tokens=100, max_atoms=100_000)
    # Must return the exact same object unmodified.
    assert out is data


def test_crop_dispatch_forced_contiguous():
    pytest.importorskip("scipy")
    data = make_tokenized(100, n_chains=1)
    cropper = Cropper(contiguous_weight=1.0, spatial_weight=0.0, interface_weight=0.0)
    out = cropper.crop(data, max_tokens=20, max_atoms=10_000, seed=0)
    assert len(out.tokens) == 20
    # Contiguous crop → tokens should be consecutive in the original indexing.
    # We check by confirming residue_index is strictly ascending by 1.
    ridx = out.tokens["residue_index"]
    assert np.all(np.diff(ridx) == 1)


def test_crop_dispatch_forced_spatial():
    pytest.importorskip("scipy")
    data = make_tokenized(100, n_chains=2)
    cropper = Cropper(contiguous_weight=0.0, spatial_weight=1.0, interface_weight=0.0)
    out = cropper.crop(data, max_tokens=20, max_atoms=10_000, seed=0)
    assert len(out.tokens) == 20


def test_crop_dispatch_forced_interface():
    pytest.importorskip("scipy")
    n = 60
    n_a = 30
    coords = np.zeros((n, 3), dtype=np.float32)
    coords[:n_a, 0] = np.arange(n_a) * 2.0
    coords[n_a:, 0] = np.arange(n - n_a) * 2.0
    coords[n_a:, 1] = 5.0  # chain B is 5Å offset → interfaces everywhere
    data = make_tokenized(n, n_chains=2, coords=coords)

    cropper = Cropper(
        contiguous_weight=0.0,
        spatial_weight=0.0,
        interface_weight=1.0,
        interface_distance=15.0,
    )
    out = cropper.crop(data, max_tokens=20, max_atoms=10_000, seed=0)
    assert len(out.tokens) == 20
    # Interface selection should produce a crop spanning both chains.
    asyms = set(out.tokens["asym_id"].tolist())
    assert len(asyms) == 2


# ---------------------------------------------------------------------------
# 7. Reindex invariants
# ---------------------------------------------------------------------------


def test_reindex_invariants():
    pytest.importorskip("scipy")
    n = 60
    data = make_tokenized(
        n,
        n_chains=2,
        atoms_per_token=4,
        with_msa=True,
        with_bonds=True,
    )
    cropper = Cropper(contiguous_weight=1.0, spatial_weight=0.0, interface_weight=0.0)
    out = cropper.crop(data, max_tokens=15, max_atoms=10_000, seed=0)

    # 1. token_idx re-numbered [0..len-1]
    assert np.array_equal(out.tokens["token_idx"], np.arange(len(out.tokens)))

    # 2. All kept bond endpoints are in range.
    for b in out.bonds:
        assert 0 <= int(b["token_1"]) < len(out.tokens)
        assert 0 <= int(b["token_2"]) < len(out.tokens)

    # 3. New structure's atoms length equals sum(atom_num) over tokens.
    expected_atoms = int(out.tokens["atom_num"].sum())
    assert len(out.structure.atoms) == expected_atoms

    # 4. MSA columns sliced consistently with kept tokens, per chain.
    for asym_id in set(out.tokens["asym_id"].tolist()):
        kept_in_chain = int((out.tokens["asym_id"] == asym_id).sum())
        if asym_id in out.unpaired_msa:
            assert out.unpaired_msa[asym_id].sequences.shape[1] == kept_in_chain
            assert out.unpaired_msa[asym_id].mask.shape[1] == kept_in_chain
            assert out.unpaired_msa[asym_id].deletion_matrix.shape[1] == kept_in_chain
        if asym_id in out.paired_msa:
            assert out.paired_msa[asym_id].sequences.shape[1] == kept_in_chain

    # 5. atom_idx in new tokens points to valid positions in new atoms array.
    for t in out.tokens:
        a0 = int(t["atom_idx"])
        an = int(t["atom_num"])
        assert 0 <= a0 <= len(out.structure.atoms)
        assert a0 + an <= len(out.structure.atoms)


# ---------------------------------------------------------------------------
# MSA remap CONTENT invariant (BUG-A regression)
#
# test_reindex_invariants above only checks MSA *shapes*. The bug that shipped
# and was fixed (commit 6e75dbe) was a content mis-alignment: after a crop
# slices each chain's MSA to its surviving columns and re-indexes them 0..k-1,
# the token->column map (``msa_col_idx``) still held the ORIGINAL columns, so
# the featurizer gathered stale / dropped columns. These tests pin the value
# invariant a shape check can never catch.
# ---------------------------------------------------------------------------


def test_remap_msa_col_idx_gather_invariant():
    """Sliced-MSA gather at the remapped index == full-MSA gather at the
    original column, for every token — the exact invariant BUG A violated.

    Uses non-contiguous, non-zero-start columns per chain (the spatial /
    interface crop case): impossible to get right by positional slicing.
    """
    asym = np.array([0, 0, 0, 1, 1])
    # Surviving tokens reference these ORIGINAL chain-local columns (out of
    # order, not starting at 0). Chain 0 full width 10, chain 1 full width 6.
    msa_col_idx = np.array([5, 2, 8, 3, 0])
    # Distinct value per (chain, column) so any mis-gather is detectable.
    full = {
        0: np.array([[100 + c for c in range(10)]]),
        1: np.array([[200 + c for c in range(6)]]),
    }

    chain_cols = compute_chain_msa_cols(asym, msa_col_idx)
    # What _load_msa_sliced / the cropper's _crop_msa_dict do: slice each
    # chain's MSA to its surviving columns, implicitly re-indexed 0..k-1.
    sliced = {aid: full[aid][:, cols] for aid, cols in chain_cols.items()}
    remapped = remap_msa_col_idx_to_slice(asym, msa_col_idx, chain_cols)

    assert (remapped >= 0).all()  # no valid token collapsed to a gap
    for tok in range(len(asym)):
        aid = int(asym[tok])
        got = sliced[aid][0, remapped[tok]]
        want = full[aid][0, msa_col_idx[tok]]
        assert got == want, f"token {tok}: sliced gather {got} != full gather {want}"


def test_remap_msa_col_idx_preserves_ligand_gaps():
    """Ligand / column-less tokens (-1) stay -1 through the remap."""
    asym = np.array([0, 0, 0])
    msa_col_idx = np.array([7, -1, 3])
    chain_cols = compute_chain_msa_cols(asym, msa_col_idx)
    remapped = remap_msa_col_idx_to_slice(asym, msa_col_idx, chain_cols)
    assert int(remapped[1]) == -1
    # 7 and 3 rebase to their rank within the sorted-by-first-occurrence cols.
    assert int(remapped[0]) == 0 and int(remapped[2]) == 1


# ===========================================================================
# ITEM 8 — Contiguous crop = AF-Multimer 2.7.2 Algorithm 1 (OF3 parity)
#
# The rewritten _contiguous_crop permutes ALL chains and splits the token
# budget across them; each chain contributes a random contiguous sub-segment.
# ===========================================================================


def _segment_is_contiguous_within_chain(selected: np.ndarray, chain_global: np.ndarray) -> bool:
    """A chain's contribution must be a contiguous run of that chain's tokens."""
    picked = np.intersect1d(selected, chain_global)
    if len(picked) <= 1:
        return True
    pos = np.searchsorted(chain_global, picked)
    return bool(np.all(np.diff(pos) == 1))


def test_contiguous_afm_splits_budget_across_chains():
    """3 chains of 30 tokens, budget 45: the crop must span >=2 chains, fill the
    budget exactly (no atom cap), and each chain's slice is contiguous."""
    import random

    tokens, _ = make_tokens(90, n_chains=3)  # 30 tokens per chain
    cropper = Cropper()
    for seed in range(10):
        sel = cropper._contiguous_crop(
            tokens, max_tokens=45, max_atoms=10_000_000, chain_id=None, rng=random.Random(seed)
        )
        # Budget is fully used (total capacity 90 >= 45) and never exceeded.
        assert len(sel) == 45
        # Sorted (required by _reindex_hierarchy).
        assert np.all(np.diff(sel) >= 1)
        # No single 30-token chain can fill 45 → must span >=2 chains.
        assert len({int(a) for a in tokens["asym_id"][sel]}) >= 2
        # Per-chain contribution is a contiguous sub-segment.
        for c in np.unique(tokens["asym_id"]):
            chain_global = np.where(tokens["asym_id"] == c)[0]
            assert _segment_is_contiguous_within_chain(sel, chain_global)


def test_contiguous_afm_respects_atom_budget_multichain():
    """The atom cap is preserved: with 100 atoms/token and a 250-atom budget the
    multi-chain contiguous crop keeps at most 2 tokens."""
    import random

    tokens, _ = make_tokens(60, n_chains=3, atoms_per_token=100)
    cropper = Cropper()
    sel = cropper._contiguous_crop(tokens, max_tokens=30, max_atoms=250, chain_id=None, rng=random.Random(0))
    assert 0 < len(sel) <= 2
    assert int(tokens["atom_num"][sel].sum()) <= 250


def test_contiguous_afm_single_chain_still_contiguous():
    """Single chain reduces to a single contiguous segment (legacy behaviour)."""
    import random

    tokens, _ = make_tokens(80, n_chains=1)
    cropper = Cropper()
    sel = cropper._contiguous_crop(tokens, max_tokens=25, max_atoms=10_000_000, chain_id=None, rng=random.Random(4))
    assert len(sel) == 25
    assert np.array_equal(sel, np.arange(sel[0], sel[0] + 25))
    assert {int(a) for a in tokens["asym_id"][sel]} == {0}


# ===========================================================================
# ITEM 10 — Spatial / interface crop candidate pool = resolved tokens only
#
# Unresolved token-center atoms are never crop candidates (OF3
# get_resolved_token_center_atoms). Here the unresolved tokens keep NON-ZERO
# coords, so only the resolved mask (not the zero-coord `valid` mask) can
# exclude them.
# ===========================================================================


def test_spatial_crop_excludes_unresolved_tokens():
    pytest.importorskip("scipy")
    n = 60
    coords = np.zeros((n, 3), dtype=np.float32)
    coords[:, 0] = np.arange(n) * 1.0 + 1.0  # tight cluster, all non-zero centers
    unresolved = set(range(0, n, 2))  # every other token unresolved (but valid coords)
    data = make_tokenized(n, coords=coords, unresolved=unresolved)

    cropper = Cropper(contiguous_weight=0.0, spatial_weight=1.0, interface_weight=0.0)
    out = cropper.crop(data, max_tokens=20, max_atoms=10_000_000, seed=1)

    # residue_index preserves the ORIGINAL token index through the crop.
    sel_orig = {int(r) for r in out.tokens["residue_index"]}
    assert sel_orig.isdisjoint(unresolved), f"selected unresolved tokens: {sel_orig & unresolved}"
    # Every selected token's atoms are resolved.
    assert bool(out.structure.atoms["is_present"].all())
    assert len(out.tokens) == 20


def test_interface_crop_excludes_unresolved_tokens():
    pytest.importorskip("scipy")
    n = 40
    n_a = 20
    coords = np.zeros((n, 3), dtype=np.float32)
    coords[:n_a, 0] = np.arange(n_a) * 3.0
    coords[n_a:, 0] = np.arange(n - n_a) * 3.0
    coords[n_a:, 1] = 4.0  # chain B offset 4Å → dense cross-chain interface
    unresolved = {5, 6, n_a, n_a + 1}  # unresolved tokens on both chains near the interface
    data = make_tokenized(n, n_chains=2, coords=coords, unresolved=unresolved)

    cropper = Cropper(contiguous_weight=0.0, spatial_weight=0.0, interface_weight=1.0, interface_distance=15.0)
    out = cropper.crop(data, max_tokens=20, max_atoms=10_000_000, seed=2)

    sel_orig = {int(r) for r in out.tokens["residue_index"]}
    assert sel_orig.isdisjoint(unresolved)
    assert bool(out.structure.atoms["is_present"].all())
    # The interface crop still spans both chains.
    assert len({int(a) for a in out.tokens["asym_id"]}) == 2


# ===========================================================================
# ITEM 9 — Large-assembly chain "pre-crop" = OF3 crop_chainwise
#
# select_interface_chains now takes the Structure + sampled anchor and keeps
# the spatially-closest N chains, re-includes proximal ligands, and ALWAYS
# keeps the preferred chain/interface.
# ===========================================================================


def _make_structure(chain_specs: list[dict]) -> Structure:
    """Build a minimal Structure from per-chain specs.

    Each spec: ``{"mol_type": int, "coords": [[x,y,z], ...], "present": [bool]|None}``.
    Chains occupy contiguous atom ranges (as the real pipeline guarantees).
    """
    atoms_parts: list[np.ndarray] = []
    residues_parts: list[np.ndarray] = []
    chains = np.zeros(len(chain_specs), dtype=CHAIN_DTYPE)
    atom_cursor = 0
    res_cursor = 0
    for aid, spec in enumerate(chain_specs):
        pts = np.asarray(spec["coords"], dtype=np.float64)
        k = len(pts)
        present = spec.get("present")
        present = np.ones(k, dtype=bool) if present is None else np.asarray(present, dtype=bool)

        a = np.zeros(k, dtype=ATOM_DTYPE)
        a["x"], a["y"], a["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
        a["is_present"] = present
        a["element"] = 6
        atoms_parts.append(a)

        r = np.zeros(1, dtype=RESIDUE_DTYPE)
        r[0]["atom_idx"] = atom_cursor
        r[0]["atom_num"] = k
        r[0]["is_present"] = bool(present.any())
        residues_parts.append(r)

        chains[aid]["name"] = f"C{aid}".encode()
        chains[aid]["mol_type"] = spec["mol_type"]
        chains[aid]["asym_id"] = aid
        chains[aid]["res_idx"] = res_cursor
        chains[aid]["res_num"] = 1
        chains[aid]["atom_idx"] = atom_cursor
        chains[aid]["atom_num"] = k
        atom_cursor += k
        res_cursor += 1

    atoms = np.concatenate(atoms_parts) if atoms_parts else np.empty(0, dtype=ATOM_DTYPE)
    residues = np.concatenate(residues_parts) if residues_parts else np.empty(0, dtype=RESIDUE_DTYPE)
    return Structure(atoms=atoms, bonds=np.empty(0, dtype=BOND_DTYPE), residues=residues, chains=chains)


def test_select_interface_chains_below_limit_keeps_all():
    from deepfold3.data import const
    from deepfold3.data.prepare.filters import select_interface_chains

    specs = [{"mol_type": const.MOL_PROTEIN, "coords": [[x * 10.0, 0.0, 0.0]]} for x in range(3)]
    struct = _make_structure(specs)
    kept = select_interface_chains(struct, anchor=None, max_chains=20)
    assert kept == {0, 1, 2}


def test_select_interface_chains_closest_n():
    pytest.importorskip("scipy")
    import random

    from deepfold3.data import const
    from deepfold3.data.prepare.filters import select_interface_chains

    # 5 protein chains on a line at x = 0,10,20,30,40; anchor chain 0.
    specs = [{"mol_type": const.MOL_PROTEIN, "coords": [[x * 10.0, 0.0, 0.0]]} for x in range(5)]
    struct = _make_structure(specs)
    kept = select_interface_chains(struct, anchor=("chain", 0, -1), max_chains=3, rng=random.Random(0))
    assert kept == {0, 1, 2}  # 3 spatially closest to chain 0 (at x=0)


def test_select_interface_chains_anchor_always_kept():
    """The prior greedy (anchor-independent) pre-crop could drop the sampled
    chain. The OF3 approach force-keeps it — even when it is far away AND has
    no resolved atoms (so it is not even a ranking candidate)."""
    pytest.importorskip("scipy")
    import random

    from deepfold3.data import const
    from deepfold3.data.prepare.filters import select_interface_chains

    specs = [{"mol_type": const.MOL_PROTEIN, "coords": [[x * 10.0, 0.0, 0.0]]} for x in range(4)]
    # Chain 4: far away and entirely unresolved.
    specs.append({"mol_type": const.MOL_PROTEIN, "coords": [[1000.0, 0.0, 0.0]], "present": [False]})
    struct = _make_structure(specs)
    kept = select_interface_chains(struct, anchor=("chain", 4, -1), max_chains=3, rng=random.Random(0))
    assert 4 in kept


def test_select_interface_chains_ligand_reinclusion():
    pytest.importorskip("scipy")
    import random

    from deepfold3.data import const
    from deepfold3.data.prepare.filters import select_interface_chains

    specs = [{"mol_type": const.MOL_PROTEIN, "coords": [[x * 10.0, 0.0, 0.0]]} for x in range(4)]
    specs.append({"mol_type": const.MOL_NONPOLYMER, "coords": [[1.0, 0.0, 0.0]]})  # chain 4: near chain 0
    specs.append({"mol_type": const.MOL_NONPOLYMER, "coords": [[1000.0, 0.0, 0.0]]})  # chain 5: far
    struct = _make_structure(specs)

    kept = select_interface_chains(
        struct, anchor=("chain", 0, -1), max_chains=3, ligand_inclusion_distance=5.0, rng=random.Random(0)
    )
    # 4 protein chains (>3 effective) → pre-crop triggers; ligands don't count.
    assert {0, 1, 2}.issubset(kept)
    assert 3 not in kept  # 4th protein chain is beyond the closest 3
    assert 4 in kept  # ligand within 5Å of a kept chain → re-included
    assert 5 not in kept  # distant ligand dropped


def test_select_interface_chains_interface_anchor_keeps_both():
    pytest.importorskip("scipy")
    import random

    from deepfold3.data import const
    from deepfold3.data.prepare.filters import select_interface_chains

    # 6 protein chains; interface anchor (0, 5) at opposite ends (no direct
    # contact) — both must survive regardless of the closest-N ranking.
    specs = [{"mol_type": const.MOL_PROTEIN, "coords": [[x * 10.0, 0.0, 0.0]]} for x in range(6)]
    struct = _make_structure(specs)
    kept = select_interface_chains(struct, anchor=("interface", 0, 5), max_chains=3, rng=random.Random(1))
    assert 0 in kept and 5 in kept
