"""Tests for the Cropper (contiguous / spatial / interface-seeded)."""

from __future__ import annotations

import numpy as np
import pytest

from deepfold3.data.crop.cropper import Cropper
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
) -> tuple[np.ndarray, Structure]:
    """Build a synthetic TOKEN_DTYPE array + matching minimal Structure.

    Tokens are distributed across ``n_chains`` (roughly equal sizes). Each
    token claims ``atoms_per_token`` consecutive atoms. If ``coords`` is
    provided (shape [n, 3]), it sets (center_x, center_y, center_z) and also
    the corresponding atom (x, y, z). Otherwise tokens are placed on a
    1D line with spacing 10Å.
    """
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
        atoms[a0 : a0 + an]["x"] = float(coords[i, 0])
        atoms[a0 : a0 + an]["y"] = float(coords[i, 1])
        atoms[a0 : a0 + an]["z"] = float(coords[i, 2])
        atoms[a0 : a0 + an]["is_present"] = True
        atoms[a0 : a0 + an]["element"] = 6

    residues = np.zeros(n, dtype=RESIDUE_DTYPE)
    for i in range(n):
        residues[i]["res_id"] = i
        residues[i]["atom_idx"] = int(tokens[i]["atom_idx"])
        residues[i]["atom_num"] = atoms_per_token
        residues[i]["is_present"] = True

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
) -> Tokenized:
    tokens, structure = make_tokens(
        n,
        n_chains=n_chains,
        atoms_per_token=atoms_per_token,
        coords=coords,
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
        bond_list = [(0, 1), (1, 2), (2, 3), (0, n - 1)]
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
