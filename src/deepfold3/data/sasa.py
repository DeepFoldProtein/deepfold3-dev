"""Solvent-accessible surface area targets for the interface-burial auxiliary head.

Everything here is computed **on the cropped complex**, never precomputed: the target
must be the burial of the structure the model actually sees, or it is being asked to
predict occlusion by atoms outside its own crop (``export/dsasa-aux/README.md`` §3.1).
A per-crop recompute also cannot go stale the way the disk token masks did
(``dataset.py:_DISK_TOKEN_MASKS_ENABLED``).

The partner-specific quantity is

    dsasa[i, c] = SASA_i(chain(i) alone) - SASA_i(chain(i) + chain c)

so the baseline already contains every burial caused by residue *i*'s own chain; only
area lost to partner *c* is counted.  Intra-chain burial is a **different** observable
and is returned separately as ``sasa_free - sasa_chain``.

Shrake-Rupley, golden-spiral sphere points.

**Choose ``n_points`` on the per-residue quantity, not the total.**  The sphere points
live in the lab frame, so a rigid rotation of the input -- which the training
augmentation applies on every example -- resamples the surface and perturbs the answer.
Total BSA converges almost immediately because the per-atom errors cancel, but the
per-residue target this head is supervised on does not.  Measured on 7r3z (207 residues,
8 rigid motions, ``export/dsasa-aux/2026-08-08-frozen-probe.md``):

    n_points   BSA CV    per-residue CV    per-residue MAE    occupancy flips
          32    1.1%             22.9%          3.8 A^2                2.42%
          64    0.5%             12.0%          2.1 A^2                1.17%
         128    0.4%              7.1%          1.4 A^2                0.78%
         256    0.1%              4.4%          0.8 A^2                0.16%

against a mean positive |dSASA| of 34 A^2.  ``n_points=32`` is fine for a BSA-level
target and adds an 11% noise floor to a per-residue one; 128 is the cheapest setting
where the target noise is small against the signal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

PROBE_RADIUS = 1.4
DEFAULT_N_POINTS = 128
DEFAULT_SHELL = 8.0  # atoms further than this from the partner have dSASA == 0 exactly
DEFAULT_CONTACT_DIST = 4.5
DEFAULT_NV_RADIUS = 10.0

# Bondi van der Waals radii, keyed by atomic number.  Same table as
# ``export/ppi-failure-profile/scripts/gt_features.py`` so BSA computed here is
# directly comparable to the BSA the PPI contact-density law was fitted on.
_VDW_BY_SYMBOL = {
    "H": 1.20,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "F": 1.47,
    "NA": 2.27,
    "MG": 1.73,
    "SI": 2.10,
    "P": 1.80,
    "S": 1.80,
    "CL": 1.75,
    "K": 2.75,
    "CA": 2.31,
    "MN": 1.61,
    "FE": 1.56,
    "CO": 1.52,
    "NI": 1.63,
    "CU": 1.40,
    "ZN": 1.39,
    "SE": 1.90,
    "BR": 1.85,
    "I": 1.98,
}
_Z_BY_SYMBOL = {
    "H": 1,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "NA": 11,
    "MG": 12,
    "SI": 14,
    "P": 15,
    "S": 16,
    "CL": 17,
    "K": 19,
    "CA": 20,
    "MN": 25,
    "FE": 26,
    "CO": 27,
    "NI": 28,
    "CU": 29,
    "ZN": 30,
    "SE": 34,
    "BR": 35,
    "I": 53,
}
DEFAULT_VDW = 1.70

VDW_BY_Z = np.full(128, DEFAULT_VDW, dtype=np.float64)
for _sym, _r in _VDW_BY_SYMBOL.items():
    VDW_BY_Z[_Z_BY_SYMBOL[_sym]] = _r


def sphere_points(n_points: int) -> np.ndarray:
    """Golden-spiral points on the unit sphere, shape [n_points, 3]."""
    i = np.arange(n_points) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n_points)
    theta = np.pi * (1.0 + 5.0**0.5) * i
    return np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1)


def shrake_rupley(
    query: np.ndarray,
    xyz: np.ndarray,
    ext: np.ndarray,
    tree: cKDTree,
    n_points: int = DEFAULT_N_POINTS,
    block: int = 256,
) -> np.ndarray:
    """SASA of the atoms indexed by ``query``, occluded by every atom in ``xyz``.

    Args:
        query: [Q] indices into ``xyz`` of the atoms whose SASA is wanted.
        xyz: [A, 3] coordinates of all occluding atoms (query atoms included).
        ext: [A] extended radii (vdW + probe).
        tree: ``cKDTree(xyz)``, passed in so callers can reuse it across states.
        n_points: sphere sampling density.
        block: atoms per vectorised chunk; bounds the probe-point KD-tree size.

    Returns:
        [Q] float64 SASA in A^2, aligned with ``query``.
    """
    sphere = sphere_points(n_points)
    out = np.zeros(len(query), dtype=np.float64)
    if len(query) == 0:
        return out
    rmax = float(ext.max())
    for s in range(0, len(query), block):
        idx = query[s : s + block]
        pts = (xyz[idx][:, None, :] + ext[idx][:, None, None] * sphere[None]).reshape(-1, 3)
        owner = idx[np.repeat(np.arange(len(idx)), n_points)]
        pr = cKDTree(pts).sparse_distance_matrix(tree, rmax, output_type="ndarray")
        buried = np.zeros(len(pts), dtype=bool)
        if len(pr):
            keep = (pr["v"] < ext[pr["j"]]) & (owner[pr["i"]] != pr["j"])
            buried[pr["i"][keep]] = True
        out[s : s + block] = (~buried).reshape(len(idx), n_points).mean(1) * 4 * np.pi * ext[idx] ** 2
    return out


@dataclass
class SasaTargets:
    """Per-token SASA-derived targets for one cropped complex.

    ``K`` is the number of partner chain slots; ``partner_asym[c]`` gives the
    ``asym_id`` that slot ``c`` refers to.

    Attributes:
        dsasa: [N, K] partner-specific buried area, A^2. Zero-inflated by construction.
        n_contact: [N, K] partner heavy atoms within ``contact_dist`` (arm-C target).
        loss_mask: [N, K] supervision valid: token real, partner slot real, partner
            is not the token's own chain.
        partner_asym: [K] asym_id per slot.
        sasa_chain: [N] SASA in the token's own chain alone (the dSASA baseline).
        sasa_cplx: [N] SASA in the full cropped complex (all chains).
        sasa_free: [N] SASA of the token's atoms occluded only by each other, i.e. the
            residue removed from its chain. ``sasa_free - sasa_chain`` is intra-chain
            burial, a *different* observable from dsasa; ``sasa_chain / sasa_free`` is a
            self-consistent relative accessibility.
        nv: [N] heavy atoms within ``nv_radius`` of any of the token's atoms,
            excluding the token's own atoms (dense neighbour-density target).
        nv_cross: [N] same, restricted to atoms of other chains.
        completeness: [N] fraction of expected atom slots actually present.
        n_atoms: [N] resolved heavy atoms per token.
        own_asym: [N] the token's own chain id, so ``bsa()`` can group both sides.
    """

    dsasa: np.ndarray
    n_contact: np.ndarray
    loss_mask: np.ndarray
    partner_asym: np.ndarray
    sasa_chain: np.ndarray
    sasa_cplx: np.ndarray
    sasa_free: np.ndarray
    nv: np.ndarray
    nv_cross: np.ndarray
    completeness: np.ndarray
    n_atoms: np.ndarray
    own_asym: np.ndarray

    def bsa(self) -> dict[tuple[int, int], float]:
        """Buried surface area per unordered chain pair, A^2 (half the two-sided sum)."""
        out: dict[tuple[int, int], float] = {}
        for c, pc in enumerate(self.partner_asym):
            if pc < 0:
                continue
            for a in np.unique(self.own_asym[self.loss_mask[:, c]]):
                key = (int(min(a, pc)), int(max(a, pc)))
                out[key] = (
                    out.get(key, 0.0) + float(self.dsasa[self.loss_mask[:, c] & (self.own_asym == a), c].sum()) / 2.0
                )
        return out


def compute_sasa_targets(
    atom_pos: np.ndarray,
    atom_mask: np.ndarray,
    elements: np.ndarray,
    asym_id: np.ndarray,
    seq_mask: np.ndarray,
    *,
    ref_mask: np.ndarray | None = None,
    partner_asym: np.ndarray | None = None,
    probe: float = PROBE_RADIUS,
    n_points: int = DEFAULT_N_POINTS,
    shell: float = DEFAULT_SHELL,
    contact_dist: float = DEFAULT_CONTACT_DIST,
    nv_radius: float = DEFAULT_NV_RADIUS,
    dense_targets: bool = True,
) -> SasaTargets:
    """Build every SASA-derived target for one cropped complex.

    Args:
        atom_pos: [N, M, 3] ground-truth coordinates in the dense token-atom layout.
        atom_mask: [N, M] resolved-atom mask.
        elements: [N, M] atomic number per atom slot (``ref_element`` with
            ``ref_element_zero_index=False``).
        asym_id: [N] chain id per token.
        seq_mask: [N] real-token mask.
        ref_mask: [N, M] expected atom slots, for ``completeness``. Defaults to
            ``atom_mask`` (completeness then degenerates to 1).
        partner_asym: [K] explicit partner slot order. Defaults to every chain present,
            sorted.
        probe: solvent probe radius.
        n_points: Shrake-Rupley sphere sampling density.
        shell: only atoms within this distance of a partner chain can change SASA on
            complexation; the rest contribute exactly zero and are skipped.
        contact_dist: heavy-atom cutoff for ``n_contact``.
        nv_radius: cutoff for the neighbour-density targets.
        dense_targets: also compute ``sasa_cplx`` / ``sasa_free`` / ``nv`` / ``nv_cross``.
            The auxiliary head only needs ``dsasa`` and ``n_contact``; the dense
            observables are for the frozen-trunk probe, which compares candidate targets.
            Turning them off roughly halves the cost.

    Returns:
        ``SasaTargets`` with all arrays in token order.
    """
    atom_pos = np.asarray(atom_pos, dtype=np.float64)
    n_tok, n_slot = atom_mask.shape
    valid = np.asarray(atom_mask, dtype=bool) & np.asarray(seq_mask, dtype=bool)[:, None]

    tok_of_atom, slot_of_atom = np.nonzero(valid)
    xyz = atom_pos[tok_of_atom, slot_of_atom]
    z = np.clip(np.asarray(elements)[tok_of_atom, slot_of_atom].astype(np.int64), 0, 127)
    rad = VDW_BY_Z[z]
    ext = rad + probe
    chain_of_atom = np.asarray(asym_id)[tok_of_atom]
    n_atom = len(xyz)

    if partner_asym is None:
        partner_asym = np.unique(chain_of_atom) if n_atom else np.zeros(0, dtype=np.int64)
    partner_asym = np.asarray(partner_asym, dtype=np.int64)
    n_part = len(partner_asym)

    dsasa = np.zeros((n_tok, n_part), dtype=np.float32)
    n_contact = np.zeros((n_tok, n_part), dtype=np.float32)
    loss_mask = np.zeros((n_tok, n_part), dtype=bool)
    sasa_chain = np.zeros(n_tok, dtype=np.float32)
    sasa_cplx = np.zeros(n_tok, dtype=np.float32)
    sasa_free = np.zeros(n_tok, dtype=np.float32)
    nv = np.zeros(n_tok, dtype=np.float32)
    nv_cross = np.zeros(n_tok, dtype=np.float32)
    n_atoms_tok = valid.sum(1).astype(np.float32)
    expected = np.asarray(ref_mask, dtype=bool).sum(1) if ref_mask is not None else valid.sum(1)
    completeness = np.where(expected > 0, valid.sum(1) / np.maximum(expected, 1), 0.0).astype(np.float32)

    targets = SasaTargets(
        dsasa=dsasa,
        n_contact=n_contact,
        loss_mask=loss_mask,
        partner_asym=partner_asym,
        sasa_chain=sasa_chain,
        sasa_cplx=sasa_cplx,
        sasa_free=sasa_free,
        nv=nv,
        nv_cross=nv_cross,
        completeness=completeness,
        n_atoms=n_atoms_tok,
        own_asym=np.asarray(asym_id).copy(),
    )
    if n_atom == 0:
        return targets

    def per_token(atom_vals: np.ndarray, atom_idx: np.ndarray) -> np.ndarray:
        out = np.zeros(n_tok, dtype=np.float64)
        np.add.at(out, tok_of_atom[atom_idx], atom_vals)
        return out

    tree_all = cKDTree(xyz)
    all_idx = np.arange(n_atom)

    # --- dense states -----------------------------------------------------------
    if dense_targets:
        sasa_cplx[:] = per_token(shrake_rupley(all_idx, xyz, ext, tree_all, n_points), all_idx)

    if dense_targets:
        # Residue in isolation: occluders are the token's own atoms only. Done in blocks
        # against one shared tree with a same-token filter -- a KD-tree per token would be
        # N tiny tree builds and dominated the whole function.
        sphere = sphere_points(n_points)
        rmax_all = float(ext.max())
        for s in range(0, n_atom, 256):
            loc = np.arange(s, min(s + 256, n_atom))
            pts = (xyz[loc][:, None, :] + ext[loc][:, None, None] * sphere[None]).reshape(-1, 3)
            owner = loc[np.repeat(np.arange(len(loc)), n_points)]
            pr = cKDTree(pts).sparse_distance_matrix(tree_all, rmax_all, output_type="ndarray")
            buried = np.zeros(len(pts), dtype=bool)
            if len(pr):
                keep = (
                    (pr["v"] < ext[pr["j"]])
                    & (owner[pr["i"]] != pr["j"])
                    & (tok_of_atom[owner[pr["i"]]] == tok_of_atom[pr["j"]])
                )
                buried[pr["i"][keep]] = True
            free_atom = (~buried).reshape(len(loc), n_points).mean(1) * 4 * np.pi * ext[loc] ** 2
            np.add.at(sasa_free, tok_of_atom[loc], free_atom)

        # neighbour density
        pairs_nv = tree_all.sparse_distance_matrix(tree_all, nv_radius, output_type="ndarray")
        if len(pairs_nv):
            i_nv, j_nv = pairs_nv["i"], pairs_nv["j"]
            keep = tok_of_atom[i_nv] != tok_of_atom[j_nv]
            np.add.at(nv, tok_of_atom[i_nv[keep]], 1.0)
            keep_x = keep & (chain_of_atom[i_nv] != chain_of_atom[j_nv])
            np.add.at(nv_cross, tok_of_atom[i_nv[keep_x]], 1.0)

    # --- per-chain isolated state, reused across every partner ------------------
    chain_atoms = {int(a): np.nonzero(chain_of_atom == a)[0] for a in np.unique(chain_of_atom)}
    chain_tree = {a: cKDTree(xyz[idx]) for a, idx in chain_atoms.items()}
    chain_sasa: dict[int, np.ndarray] = {}
    for a, idx in chain_atoms.items():
        s = shrake_rupley(np.arange(len(idx)), xyz[idx], ext[idx], chain_tree[a], n_points)
        chain_sasa[a] = s
        sasa_chain[:] += per_token(s, idx)

    # --- partner-specific burial -------------------------------------------------
    for c, pa in enumerate(partner_asym):
        pa = int(pa)
        if pa not in chain_atoms:
            continue
        idx_p = chain_atoms[pa]
        for a, idx_a in chain_atoms.items():
            if a == pa:
                continue
            tok_a = np.unique(tok_of_atom[idx_a])
            loss_mask[tok_a, c] = True

            pr = chain_tree[a].sparse_distance_matrix(chain_tree[pa], shell, output_type="ndarray")
            if not len(pr):
                continue
            close = pr["v"] < contact_dist
            if close.any():
                np.add.at(n_contact[:, c], tok_of_atom[idx_a[pr["i"][close]]], 1.0)

            near_local = np.unique(pr["i"]).astype(int)
            if not len(near_local):
                continue
            # bound state = chain a + chain pa only (pairwise, matching the BSA
            # convention in gt_features.py; 3-way interfaces double-count).
            sub = np.concatenate([idx_a, idx_p])
            xyz_sub, ext_sub = xyz[sub], ext[sub]
            tree_sub = cKDTree(xyz_sub)
            bound = shrake_rupley(near_local, xyz_sub, ext_sub, tree_sub, n_points)
            delta = chain_sasa[a][near_local] - bound
            np.add.at(dsasa[:, c], tok_of_atom[idx_a[near_local]], delta.astype(np.float32))

    np.maximum(dsasa, 0.0, out=dsasa)
    return targets
