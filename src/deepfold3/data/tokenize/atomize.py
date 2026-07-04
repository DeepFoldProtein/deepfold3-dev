"""Stochastic atomization of standard polymer residues (training-only).

Standard protein / RNA / DNA residues are tokenized 1-token-per-residue. This
module expands a *selected subset* of those residues into 1-token-per-heavy-atom
— the exact layout non-standard (modified) residues and ligands already use — so
the model sees atom-granularity tokens for them.

Used purely as a training-time regularizer (see ``DataConfig.atomize_prob``):
inference never atomizes standard residues. Its value is augmenting the
atomized-token code path (exercised at inference for modified residues, bonded
ligands and glycans) and mildly regularizing token boundaries.

The transform runs on a :class:`~deepfold3.data.types.Tokenized` object AFTER
tokenization but BEFORE cropping, so the crop's token/atom budget accounts for
the expansion. It leaves ``structure`` untouched — only the token→atom grouping
changes — and re-derives ``bonds`` (existing token bonds remapped to the split
atom tokens + intra-residue connectivity synthesised from the CCD).

Decisions (see project memory ``project_stochastic_atomization_design``):
  * ``res_type`` keeps the real amino-acid / nucleotide identity (NOT UNK), so
    MSA / relative-position / symmetry all key off the true residue.
  * atomized tokens are detected downstream by ``atom_num == 1 & polymer`` — the
    featurizer routes them through its single-atom path.
  * ``residue_index`` is shared across a residue's atom tokens (drives the
    within-residue ``rel_token`` encoding).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from deepfold3.data import const
from deepfold3.data.masking import MASK_REGION, TokenMask
from deepfold3.data.types import TOKEN_BOND_DTYPE, TOKEN_DTYPE, Tokenized

_POLYMER = (const.MOL_PROTEIN, const.MOL_RNA, const.MOL_DNA)


@lru_cache(maxsize=512)
def _ccd_bond_names(res_name: str) -> tuple[tuple[str, str], ...]:
    """CCD heavy-atom connectivity for a residue as ``(name_a, name_b)`` pairs.

    Cached per residue name (standard residues are few, so this warms fast).
    Returns an empty tuple when the CCD entry is unavailable — intra-residue
    bond synthesis is an enhancement, not required for the token split.
    """
    try:
        from deepfold3.data.ccd import load_ccd_molecule

        mol = load_ccd_molecule(res_name)
    except Exception:
        return ()
    names = mol.atom_names
    out: list[tuple[str, str]] = []
    for a, b, _order in mol.bonds:
        if 0 <= a < len(names) and 0 <= b < len(names):
            out.append((names[a], names[b]))
    return tuple(out)


def _eligible_residue_tokens(tokenized: Tokenized, source: str, interface_distance: float) -> np.ndarray:
    """Indices of residue tokens eligible for atomization under ``source``.

    Eligible = a standard polymer residue token (``polymer`` mol_type with
    ``atom_num > 1``; modified residues and ligands are already atom tokens).
    """
    tokens = tokenized.tokens
    is_residue = np.isin(tokens["mol_type"], _POLYMER) & (tokens["atom_num"] > 1)
    eligible = np.where(is_residue)[0]
    if len(eligible) == 0:
        return eligible

    if source == "random":
        return eligible

    if source == "region":
        region = tokenized.token_mask.get(MASK_REGION) if tokenized.token_mask is not None else None
        if region is None:
            return np.empty(0, dtype=np.int64)
        keep = np.asarray(region)[eligible] >= 1
        return eligible[keep]

    if source == "interface":
        asym = tokens["asym_id"]
        if len(np.unique(asym)) < 2:
            return np.empty(0, dtype=np.int64)
        from scipy.spatial import cKDTree

        centers = np.stack([tokens["center_x"], tokens["center_y"], tokens["center_z"]], axis=1).astype(np.float64)
        tree = cKDTree(centers)
        out: list[int] = []
        for i in eligible:
            neigh = tree.query_ball_point(centers[i], interface_distance)
            if any(asym[j] != asym[i] for j in neigh):
                out.append(int(i))
        return np.asarray(out, dtype=np.int64)

    raise ValueError(f"unknown atomize_source: {source!r}")


def select_residues_to_atomize(
    tokenized: Tokenized,
    rng: np.random.Generator,
    prob: float,
    source: str = "random",
    max_residues: int = 0,
    interface_distance: float = 15.0,
) -> np.ndarray:
    """Bernoulli-sample eligible residue tokens; cap at ``max_residues``.

    Returns a sorted int array of token indices (into ``tokenized.tokens``) to
    atomize. Empty when nothing is selected.
    """
    if prob <= 0.0:
        return np.empty(0, dtype=np.int64)
    eligible = _eligible_residue_tokens(tokenized, source, interface_distance=interface_distance)
    if len(eligible) == 0:
        return eligible
    picked = eligible[rng.random(len(eligible)) < prob]
    if max_residues > 0 and len(picked) > max_residues:
        picked = rng.choice(picked, size=max_residues, replace=False)
    picked.sort()
    return picked


def atomize_tokens(
    tokenized: Tokenized,
    to_atomize: np.ndarray,
) -> Tokenized:
    """Expand the given residue tokens into per-heavy-atom tokens.

    ``to_atomize`` are token indices (into ``tokenized.tokens``) of standard
    polymer residues to split. Returns a new :class:`Tokenized`; ``structure``
    is shared unchanged (only token→atom grouping changes). No-op passthrough
    when ``to_atomize`` is empty.
    """
    if len(to_atomize) == 0:
        return tokenized

    tokens = tokenized.tokens
    atoms = tokenized.structure.atoms
    atomize_set = set(int(i) for i in to_atomize)

    new_rows: list[tuple] = []
    source_map: list[int] = []  # new token → source old token index (for mask/msa expand)
    old_to_new_single: dict[int, int] = {}  # non-atomized old token → new index
    old_atom_to_new: dict[tuple[int, int], int] = {}  # (atomized old token, local atom) → new index

    for i, tok in enumerate(tokens):
        if i not in atomize_set:
            new_idx = len(new_rows)
            row = tuple(tok[name] for name in TOKEN_DTYPE.names)
            new_rows.append(row)
            source_map.append(i)
            old_to_new_single[i] = new_idx
            continue

        # Split this residue into one token per heavy atom.
        a_start = int(tok["atom_idx"])
        a_num = int(tok["atom_num"])
        res_id = int(tok["residue_index"])
        res_type = int(tok["res_type"])  # KEEP real identity
        mol_type = int(tok["mol_type"])
        asym_id = int(tok["asym_id"])
        entity_id = int(tok["entity_id"])
        sym_id = int(tok["sym_id"])
        for j in range(a_num):
            gi = a_start + j
            atom = atoms[gi]
            new_idx = len(new_rows)
            new_rows.append(
                (
                    new_idx,  # token_idx (final, sequential)
                    res_id,  # residue_index — SHARED across the residue's atoms
                    res_type,
                    mol_type,
                    asym_id,
                    entity_id,
                    sym_id,
                    gi,  # atom_idx (global)
                    1,  # atom_num
                    float(atom["x"]),
                    float(atom["y"]),
                    float(atom["z"]),
                )
            )
            source_map.append(i)
            old_atom_to_new[(i, j)] = new_idx

    new_tokens = np.array(new_rows, dtype=TOKEN_DTYPE)
    # token_idx is authoritative-sequential; overwrite (non-atomized rows kept
    # their old value in the tuple copy above).
    new_tokens["token_idx"] = np.arange(len(new_tokens), dtype=np.int32)
    source_map_arr = np.asarray(source_map, dtype=np.int64)

    # ------------------------------------------------------------------
    # Bonds: remap existing token bonds to the split atom tokens, then add
    # intra-residue CCD connectivity for atomized residues. Deduplicated by
    # unordered token pair.
    # ------------------------------------------------------------------
    def _remap(old_tok: int, local_atom: int) -> tuple[int, int] | None:
        if old_tok in atomize_set:
            nt = old_atom_to_new.get((old_tok, local_atom))
            return None if nt is None else (nt, 0)
        nt = old_to_new_single.get(old_tok)
        return None if nt is None else (nt, local_atom)

    bond_rows: list[tuple] = []
    seen: set[tuple[int, int]] = set()

    def _add_bond(t1: int, a1: int, t2: int, a2: int) -> None:
        if t1 == t2:
            return
        key = (t1, t2) if t1 < t2 else (t2, t1)
        if key in seen:
            return
        seen.add(key)
        bond_rows.append((t1, t2, a1, a2))

    for bond in tokenized.bonds:
        r1 = _remap(int(bond["token_1"]), int(bond["atom_1"]))
        r2 = _remap(int(bond["token_2"]), int(bond["atom_2"]))
        if r1 is None or r2 is None:
            continue
        _add_bond(r1[0], r1[1], r2[0], r2[1])

    # Intra-residue backbone / side-chain bonds from the CCD (these are implicit
    # for un-atomized residues; once split they must be re-expressed as token
    # bonds so the pair track sees the residue's connectivity).
    for i in atomize_set:
        tok = tokens[i]
        res_name = tok["res_type"]
        res_name = const.RES_TYPE_INDEX_TO_NAME.get(int(res_name))
        if res_name is None:
            continue
        pairs = _ccd_bond_names(res_name)
        if not pairs:
            continue
        a_start = int(tok["atom_idx"])
        a_num = int(tok["atom_num"])
        name_to_new: dict[str, int] = {}
        for j in range(a_num):
            nm = atoms[a_start + j]["name"].decode("ascii", errors="replace").strip()
            name_to_new[nm] = old_atom_to_new[(i, j)]
        for na, nb in pairs:
            ta = name_to_new.get(na)
            tb = name_to_new.get(nb)
            if ta is not None and tb is not None:
                _add_bond(ta, 0, tb, 0)

    new_bonds = np.array(bond_rows, dtype=TOKEN_BOND_DTYPE) if bond_rows else np.empty(0, dtype=TOKEN_BOND_DTYPE)

    # msa_col_idx and token_mask expand by gathering along the source map
    # (atomized atoms inherit the residue's MSA column / mask value).
    new_msa_col = tokenized.msa_col_idx[source_map_arr] if tokenized.msa_col_idx is not None else None
    new_mask: TokenMask | None = (
        tokenized.token_mask.select(source_map_arr) if tokenized.token_mask is not None else None
    )

    from dataclasses import replace

    return replace(
        tokenized,
        tokens=new_tokens,
        bonds=new_bonds,
        msa_col_idx=new_msa_col,
        token_mask=new_mask,
    )
