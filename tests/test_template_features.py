"""Template feature construction — placement for both builders.

Regression guard for the ``_fill_template_slot(start, length)`` ->
``(dst_pos, src_idx)`` signature change (ab9369f): the single-chain (flat-list)
path was left passing ``(0, L)``, so ``src_idx=L`` read one past the template
array end -> ``IndexError: index L is out of bounds for axis 0 with size L``,
crashing all inference that uses flat-list templates. Training uses the
per-chain builder (also covered here) and was unaffected.
"""

import numpy as np
import pytest

from deepfold3.data import const
from deepfold3.data.feature.featurizer import Featurizer
from deepfold3.data.parse.templates import TemplateFeatures
from deepfold3.data.types import TOKEN_DTYPE


def _make_template(length: int) -> TemplateFeatures:
    # aatype[k] = k so placement is directly checkable.
    return TemplateFeatures(
        aatype=np.arange(length, dtype=np.int64),
        atom_positions=np.zeros((length, const.MAX_ATOMS_PER_TOKEN, 3), np.float32),
        atom_mask=np.ones((length, const.MAX_ATOMS_PER_TOKEN), bool),
    )


def _make_tokens(residues: list[int], asym: int = 0) -> np.ndarray:
    toks = np.zeros(len(residues), dtype=TOKEN_DTYPE)
    for i, r in enumerate(residues):
        toks[i]["token_idx"] = i
        toks[i]["asym_id"] = asym
        toks[i]["residue_index"] = r + 1  # residue_index is 1-based
        toks[i]["mol_type"] = const.MOL_PROTEIN
    return toks


# --- single-chain (flat-list) path — the inference path that regressed ---


@pytest.mark.parametrize("n", [1, 12, 146])
def test_single_chain_no_index_error_and_placement(n):
    """Row k lands at query position k for a full-length (L == N) template."""
    out = Featurizer()._build_single_chain_template_features([_make_template(n)], n)
    aatype = out["template_aatype"]
    assert aatype.shape == (1, n)
    assert aatype[0].tolist() == list(range(n))


def test_single_chain_truncates_to_n():
    """A template longer than N is truncated to its first N rows (no overflow)."""
    out = Featurizer()._build_single_chain_template_features([_make_template(200)], 146)
    assert out["template_aatype"][0].tolist() == list(range(146))


# --- per-chain (crop-aware) path — the training path ---


def test_per_chain_contiguous_crop():
    kept = [4, 5, 6, 7]
    out = Featurizer()._build_per_chain_template_features(
        {0: [_make_template(12)]}, _make_tokens(kept), len(kept), template_col_idx=np.array(kept)
    )
    assert out["template_aatype"][0].tolist() == kept


def test_per_chain_scattered_crop():
    kept = [2, 9, 4, 11]  # non-contiguous (spatial/interface crop order)
    out = Featurizer()._build_per_chain_template_features(
        {0: [_make_template(12)]}, _make_tokens(kept), len(kept), template_col_idx=np.array(kept)
    )
    assert out["template_aatype"][0].tolist() == kept


def test_per_chain_no_crop_positional_fallback():
    out = Featurizer()._build_per_chain_template_features(
        {0: [_make_template(12)]}, _make_tokens(list(range(12))), 12, template_col_idx=None
    )
    assert out["template_aatype"][0].tolist() == list(range(12))


def test_per_chain_out_of_range_dropped():
    kept = [10, 11, 99]  # 99 is beyond the template length -> dropped, masked
    out = Featurizer()._build_per_chain_template_features(
        {0: [_make_template(12)]}, _make_tokens(kept), len(kept), template_col_idx=np.array(kept)
    )
    assert out["template_aatype"][0].tolist()[:2] == [10, 11]
    assert out["template_pseudo_beta_mask"][0].tolist() == [True, True, False]
