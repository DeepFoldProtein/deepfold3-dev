"""Analytic and invariance tests for the interface-burial targets."""

import numpy as np
import pytest
from scipy.spatial import cKDTree

from deepfold3.data.sasa import (
    PROBE_RADIUS,
    VDW_BY_Z,
    SasaTargets,
    compute_sasa_targets,
    shrake_rupley,
    sphere_points,
)

Z_C = 6
R_EXT = VDW_BY_Z[Z_C] + PROBE_RADIUS


def _mono(coords: list[list[tuple[float, float, float]]], chains: list[int]) -> dict:
    """Build featurizer-shaped arrays from a list of per-token atom coordinate lists."""
    n = len(coords)
    m = max(len(c) for c in coords)
    pos = np.zeros((n, m, 3), dtype=np.float32)
    mask = np.zeros((n, m), dtype=bool)
    elem = np.zeros((n, m), dtype=np.int64)
    for i, atoms in enumerate(coords):
        for j, xyz in enumerate(atoms):
            pos[i, j] = xyz
            mask[i, j] = True
            elem[i, j] = Z_C
    return dict(
        atom_pos=pos,
        atom_mask=mask,
        elements=elem,
        asym_id=np.asarray(chains, dtype=np.int64),
        seq_mask=np.ones(n, dtype=bool),
    )


def test_sphere_points_are_unit_and_balanced():
    p = sphere_points(512)
    assert np.allclose(np.linalg.norm(p, axis=1), 1.0)
    assert np.abs(p.mean(0)).max() < 0.05


def test_isolated_atom_matches_full_sphere():
    xyz = np.zeros((1, 3))
    ext = np.array([R_EXT])
    sasa = shrake_rupley(np.arange(1), xyz, ext, cKDTree(xyz), n_points=2048)
    assert sasa[0] == pytest.approx(4 * np.pi * R_EXT**2, rel=1e-9)


def test_two_overlapping_spheres_match_analytic_caps():
    """Each sphere loses exactly one spherical cap of height R - d/2 to the other."""
    d = 2.0
    xyz = np.array([[0.0, 0.0, 0.0], [d, 0.0, 0.0]])
    ext = np.full(2, R_EXT)
    sasa = shrake_rupley(np.arange(2), xyz, ext, cKDTree(xyz), n_points=20000)
    h = R_EXT - d / 2.0
    expect = 4 * np.pi * R_EXT**2 - 2 * np.pi * R_EXT * h
    assert sasa[0] == pytest.approx(expect, rel=2e-3)
    assert sasa[1] == pytest.approx(expect, rel=2e-3)


def test_separated_chains_bury_nothing():
    args = _mono([[(0.0, 0.0, 0.0)], [(50.0, 0.0, 0.0)]], [0, 1])
    t = compute_sasa_targets(**args, n_points=128)
    assert t.dsasa.max() == 0.0
    assert t.n_contact.max() == 0.0
    # with nothing buried, chain-alone and complex SASA agree
    assert np.allclose(t.sasa_chain, t.sasa_cplx)


def test_contacting_chains_bury_the_analytic_cap():
    d = 3.0
    args = _mono([[(0.0, 0.0, 0.0)], [(d, 0.0, 0.0)]], [0, 1])
    t = compute_sasa_targets(**args, n_points=4096)
    h = R_EXT - d / 2.0
    cap = 2 * np.pi * R_EXT * h
    # token 0 is buried by chain 1 == partner slot 1, and vice versa
    assert t.dsasa[0, 1] == pytest.approx(cap, rel=5e-3)
    assert t.dsasa[1, 0] == pytest.approx(cap, rel=5e-3)
    # a chain never buries itself
    assert t.dsasa[0, 0] == 0.0 and not t.loss_mask[0, 0]


def test_bsa_conservation_identity():
    """sum_i dsasa[i<-B] + sum_j dsasa[j<-A] == 2 * BSA(A, B)."""
    rng = np.random.default_rng(0)
    a = rng.normal(scale=4.0, size=(12, 3))
    b = rng.normal(scale=4.0, size=(12, 3)) + np.array([7.0, 0.0, 0.0])
    args = _mono([[tuple(x)] for x in np.vstack([a, b])], [0] * 12 + [1] * 12)
    t = compute_sasa_targets(**args, n_points=1024)
    side_a = t.dsasa[t.own_asym == 0, 1].sum()
    side_b = t.dsasa[t.own_asym == 1, 0].sum()
    assert side_a > 0 and side_b > 0
    assert t.bsa()[(0, 1)] == pytest.approx((side_a + side_b) / 2.0, rel=1e-6)


def test_rigid_motion_invariance_up_to_sampling_noise():
    """Exactly invariant it is not: the sphere points are fixed in the lab frame, so a
    rotation resamples the surface. This pins the residual to the sampling error."""
    rng = np.random.default_rng(1)
    a = rng.normal(scale=4.0, size=(10, 3))
    b = rng.normal(scale=4.0, size=(10, 3)) + np.array([6.0, 1.0, 0.0])
    args = _mono([[tuple(x)] for x in np.vstack([a, b])], [0] * 10 + [1] * 10)
    base = compute_sasa_targets(**args, n_points=512)

    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    q *= np.sign(np.linalg.det(q))
    moved = dict(args)
    moved["atom_pos"] = (args["atom_pos"] @ q.T + np.array([13.0, -4.0, 2.5])).astype(np.float32)
    rot = compute_sasa_targets(**moved, n_points=512)
    pos = base.dsasa > 2.0
    assert np.abs(rot.dsasa[pos] - base.dsasa[pos]).max() < 0.06 * base.dsasa[pos].max()
    assert np.abs(rot.sasa_chain - base.sasa_chain).max() < 0.06 * base.sasa_chain.max()
    # the neighbour-density targets are counts and ARE exactly invariant
    assert np.array_equal(base.nv, rot.nv)
    assert np.array_equal(base.nv_cross, rot.nv_cross)


def test_masked_atoms_are_invisible():
    """An unresolved atom must occlude nothing and receive nothing."""
    args = _mono([[(0.0, 0.0, 0.0), (1.2, 0.0, 0.0)], [(3.0, 0.0, 0.0)]], [0, 1])
    full = compute_sasa_targets(**args, n_points=1024)
    args2 = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in args.items()}
    args2["atom_mask"][0, 1] = False
    dropped = compute_sasa_targets(**args2, n_points=1024)
    assert dropped.n_atoms[0] == 1 and full.n_atoms[0] == 2
    # one fewer occluder in chain 0 -> chain 1's token loses less area to it
    assert dropped.dsasa[1, 0] < full.dsasa[1, 0]


def test_crop_consistency():
    """The target on a crop equals the target recomputed from the cropped structure."""
    rng = np.random.default_rng(2)
    a = rng.normal(scale=5.0, size=(14, 3))
    b = rng.normal(scale=5.0, size=(14, 3)) + np.array([8.0, 0.0, 0.0])
    coords = [[tuple(x)] for x in np.vstack([a, b])]
    chains = [0] * 14 + [1] * 14
    keep = np.array([i for i in range(28) if i % 7 != 3])

    args = _mono(coords, chains)
    args["seq_mask"] = np.zeros(28, dtype=bool)
    args["seq_mask"][keep] = True
    masked = compute_sasa_targets(**args, n_points=512)

    sub = _mono([coords[i] for i in keep], [chains[i] for i in keep])
    cropped = compute_sasa_targets(**sub, n_points=512)
    assert np.allclose(masked.dsasa[keep], cropped.dsasa, atol=1e-4)


def test_dsasa_is_zero_inflated_but_positives_are_real():
    """The documented weakness: most residues are exactly 0. Assert it, don't hide it."""
    rng = np.random.default_rng(3)
    a = rng.normal(scale=9.0, size=(60, 3))
    b = rng.normal(scale=9.0, size=(60, 3)) + np.array([22.0, 0.0, 0.0])
    args = _mono([[tuple(x)] for x in np.vstack([a, b])], [0] * 60 + [1] * 60)
    t = compute_sasa_targets(**args, n_points=256)
    pos = t.dsasa[t.loss_mask] > 2.0
    assert 0.0 < pos.mean() < 0.5
    assert isinstance(t, SasaTargets)


def test_sampling_noise_floor_is_documented():
    """DEFAULT_N_POINTS is a target-quality decision, not a speed knob. If someone
    lowers it, this fails and points at the table in the module docstring."""
    from deepfold3.data.sasa import DEFAULT_N_POINTS

    rng = np.random.default_rng(5)
    a = rng.normal(scale=7.0, size=(40, 3))
    b = rng.normal(scale=7.0, size=(40, 3)) + np.array([13.0, 0.0, 0.0])
    args = _mono([[tuple(x)] for x in np.vstack([a, b])], [0] * 40 + [1] * 40)
    ref = compute_sasa_targets(**args, n_points=4096)
    pos = ref.dsasa[:, 1] > 2.0

    vals = []
    for s in range(6):
        g = np.random.default_rng(300 + s)
        q, _ = np.linalg.qr(g.normal(size=(3, 3)))
        q *= np.sign(np.linalg.det(q))
        moved = dict(args)
        moved["atom_pos"] = (args["atom_pos"] @ q.T).astype(np.float32)
        vals.append(compute_sasa_targets(**moved, n_points=DEFAULT_N_POINTS).dsasa[:, 1])
    cv = (np.stack(vals)[:, pos].std(0) / np.maximum(np.stack(vals)[:, pos].mean(0), 1e-6)).mean()
    assert cv < 0.15, f"per-residue rotation CV {cv:.3f} at n_points={DEFAULT_N_POINTS}"
