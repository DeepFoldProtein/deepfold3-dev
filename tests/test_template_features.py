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


def test_pad_templates_fixes_axis_and_marks_real_slots():
    """``pad_templates`` yields a constant T with a real-vs-pad mask."""
    kept = list(range(12))
    for n_templates in (0, 1, 3):
        per_chain = {0: [_make_template(12) for _ in range(n_templates)]}
        out = Featurizer(max_templates=4, pad_templates=True)._build_template_features(
            None, 12, _make_tokens(kept), per_chain, template_col_idx=np.array(kept)
        )
        for key, val in out.items():
            if key != "template_mask":
                assert val.shape[0] == 4, f"{key} not padded for n={n_templates}"
        assert out["template_mask"].tolist() == [1.0] * n_templates + [0.0] * (4 - n_templates)
        # Pad slots must read as empty, not as spurious structure.
        assert not out["template_pseudo_beta_mask"][n_templates:].any()
        assert not out["template_backbone_mask"][n_templates:].any()


def test_pad_templates_matches_unpadded_output():
    """Padding must not change the template embedding — bit-exact."""
    import torch

    from deepfold3.model.config import OptimConfig, TemplateEmbeddingConfig
    from deepfold3.model.template_embedding import TemplateEmbedding

    torch.manual_seed(0)
    cfg = TemplateEmbeddingConfig()
    cfg.num_blocks = 1
    mod = TemplateEmbedding(cfg, optim=OptimConfig()).eval().double()

    n_tokens, n_real, t_max = 12, 2, 4
    gen = torch.Generator().manual_seed(1)

    def slots(t):
        return {
            "template_aatype": torch.randint(0, 21, (t, n_tokens), generator=gen),
            "template_pseudo_beta": torch.randn(t, n_tokens, 3, generator=gen, dtype=torch.float64),
            "template_pseudo_beta_mask": torch.ones(t, n_tokens, dtype=torch.float64),
            "template_backbone_frame": torch.eye(3, dtype=torch.float64).expand(t, n_tokens, 3, 3).contiguous(),
            "template_backbone_mask": torch.ones(t, n_tokens, dtype=torch.float64),
            "template_ca": torch.randn(t, n_tokens, 3, generator=gen, dtype=torch.float64),
        }

    real = slots(n_real)
    c_pair = mod.single_template_emb.query_norm.normalized_shape[0]
    query_pair = torch.randn(n_tokens, n_tokens, c_pair, dtype=torch.float64)
    pair_mask = torch.ones(n_tokens, n_tokens, dtype=torch.float64)

    # Pad slots carry deliberate garbage: only ``template_mask`` may suppress them.
    pad = t_max - n_real
    junk = slots(pad)
    junk["template_pseudo_beta_mask"] = torch.zeros(pad, n_tokens, dtype=torch.float64)
    junk["template_backbone_mask"] = torch.zeros(pad, n_tokens, dtype=torch.float64)
    padded = {k: torch.cat([real[k], junk[k]]) for k in real}

    with torch.no_grad():
        unpadded_out = mod(query_pair=query_pair, pair_mask=pair_mask, **real)
        padded_out = mod(
            query_pair=query_pair,
            pair_mask=pair_mask,
            template_mask=torch.tensor([1.0] * n_real + [0.0] * pad, dtype=torch.float64),
            **padded,
        )
        # A fully padded sample must equal the legacy T=0 path.
        empty_out = mod(query_pair=query_pair, pair_mask=pair_mask, **slots(0))
        allpad_out = mod(
            query_pair=query_pair,
            pair_mask=pair_mask,
            template_mask=torch.zeros(t_max, dtype=torch.float64),
            **padded,
        )

    assert torch.equal(unpadded_out, padded_out)
    assert torch.equal(empty_out, allpad_out)


def test_pad_templates_reaches_every_dataset_class():
    """A half-padded batch loses templates in collate — all classes must pad.

    ``collate_fn`` zero-fills a key missing from some samples, so if one
    dataset class emits ``template_mask`` and another does not, the unpadded
    samples get an all-zero mask and their real templates are masked away.
    """
    import inspect

    from deepfold3.data.module.dataset import TrainingDataset
    from deepfold3.data.module.of3_dataset import OF3Dataset
    from deepfold3.training.data import build_dataloaders  # noqa: F401 — import guard

    for cls in (TrainingDataset, OF3Dataset):
        assert "pad_templates" in inspect.signature(cls.__init__).parameters, cls.__name__

    # OF3Dataset is fed through an explicit allow-list, not **kwargs.
    src = inspect.getsource(inspect.getmodule(build_dataloaders))
    assert '"pad_templates",' in src, "pad_templates missing from of3_kwarg_names"
