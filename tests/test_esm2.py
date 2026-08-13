"""ESM2 port + PLM packing frontend.

The weight-dependent checks are skipped unless a local ESM2 snapshot is present
(``DF3_ESM2_DIR``, default ``/data/plm/esm2_t33_650M_UR50D``); the structural
ones run everywhere on a tiny randomly-initialised model.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from deepfold3.data import const
from deepfold3.model.esm2 import ESM2Config, ESM2Encoder, esm2_tokenizer
from deepfold3.model.lm_pair import LMPairShim
from deepfold3.model.plm import FrozenPLM

ESM2_DIR = Path(os.environ.get("DF3_ESM2_DIR", "/data/plm/esm2_t33_650M_UR50D"))
_has_weights = (ESM2_DIR / "model.safetensors").exists()
requires_esm2 = pytest.mark.skipif(not _has_weights, reason=f"no ESM2 snapshot at {ESM2_DIR}")

TOKENS = "LAGVSERTIDPKQNFYMHWCXBUZO"
_TOK = esm2_tokenizer()
_CLS, _EOS = _TOK.cls_id, _TOK.eos_id


def _tiny() -> ESM2Encoder:
    torch.manual_seed(0)
    cfg = ESM2Config(hidden_size=64, num_hidden_layers=2, num_attention_heads=4, intermediate_size=128)
    return ESM2Encoder(cfg).eval()


def _encode(seq: str) -> list[int]:
    return [4 + TOKENS.index(c) for c in seq]


def test_packed_with_sequence_id_matches_isolated_chains():
    """The whole per-chain/joined A/B rests on this: packing plus a
    ``sequence_id`` must be numerically identical to folding each chain alone,
    so any measured difference comes from the join and not from packing."""
    model = _tiny()
    a, b = _encode("MQIFVKTL"), _encode("GKQLEDGRTLSD")
    packed = torch.tensor([[_CLS, *a, _EOS, _CLS, *b, _EOS]])
    seq_id = (packed == _CLS).cumsum(1) - 1

    with torch.no_grad():
        joined = model(packed, sequence_id=seq_id)
        alone_a = model(torch.tensor([[_CLS, *a, _EOS]]))
        alone_b = model(torch.tensor([[_CLS, *b, _EOS]]))

    assert torch.allclose(joined[0, 1 : 1 + len(a)], alone_a[0, 1 : 1 + len(a)], atol=1e-5)
    start = len(a) + 3
    assert torch.allclose(joined[0, start : start + len(b)], alone_b[0, 1 : 1 + len(b)], atol=1e-5)


def test_omitting_sequence_id_changes_the_embedding():
    """Guards against the A/B silently comparing two identical arms."""
    model = _tiny()
    a, b = _encode("MQIFVKTL"), _encode("GKQLEDGRTLSD")
    packed = torch.tensor([[_CLS, *a, _EOS, _CLS, *b, _EOS]])
    seq_id = (packed == _CLS).cumsum(1) - 1
    with torch.no_grad():
        per_chain = model(packed, sequence_id=seq_id)
        cross = model(packed)
    assert (per_chain - cross).abs().max() > 1e-3


def test_padding_does_not_leak():
    model = _tiny()
    a = _encode("MQIFVKTL")
    short = torch.tensor([[_CLS, *a, _EOS]])
    padded = torch.tensor([[_CLS, *a, _EOS] + [1] * 7])
    with torch.no_grad():
        assert torch.allclose(model(short)[0, : len(a) + 2], model(padded)[0, : len(a) + 2], atol=1e-5)


def test_res_type_lookup_covers_standard_residues():
    table = _TOK.res_type_to_id
    assert table[const.RES_TYPE_ORDER["ALA"]] == 4 + TOKENS.index("A")
    assert table[const.RES_TYPE_ORDER["TRP"]] == 4 + TOKENS.index("W")
    assert table[const.RES_TYPE_ORDER["UNK"]] == 4 + TOKENS.index("X")


def test_shim_shapes_and_layer_mixture():
    shim = LMPairShim(d_pair=16, d_model=32, num_layers=3)
    hidden = torch.randn(2, 7, 4, 32)
    out = shim(hidden)
    assert out.shape == (2, 7, 7, 16)
    # Zero-init means the mixture starts uniform over all hidden states.
    assert torch.allclose(shim.layer_combine.softmax(0), torch.full((4,), 0.25))
    with pytest.raises(ValueError, match="hidden states"):
        shim(torch.randn(2, 7, 5, 32))


def _two_chain_tokens(lens=(6, 9)):
    n = sum(lens)
    res_type = torch.randint(0, 20, (n,))
    asym_id = torch.cat([torch.full((n_res,), i) for i, n_res in enumerate(lens)])
    residue_index = torch.cat([torch.arange(n_res) for n_res in lens])
    mol_type = torch.full((n,), const.MOL_PROTEIN)
    return res_type, asym_id, residue_index, mol_type, torch.ones(n)


@requires_esm2
@pytest.mark.parametrize("join", ["per_chain", "joined"])
def test_frozen_plm_end_to_end(join):
    plm = FrozenPLM(ESM2_DIR, join=join)
    plm.load(device="cpu", dtype=torch.float32)
    res_type, asym_id, residue_index, mol_type, mask = _two_chain_tokens()
    out = plm(res_type, asym_id, residue_index, mol_type, mask)
    assert out.shape == (15, plm.num_layers + 1, plm.d_model)
    assert torch.isfinite(out).all()
    assert not out.requires_grad
    assert not any(p.requires_grad for p in plm.parameters())


@requires_esm2
def test_non_protein_tokens_get_zero_embedding():
    plm = FrozenPLM(ESM2_DIR)
    plm.load(device="cpu", dtype=torch.float32)
    res_type, asym_id, residue_index, mol_type, mask = _two_chain_tokens()
    mol_type[-4:] = const.MOL_DNA
    out = plm(res_type, asym_id, residue_index, mol_type, mask)
    assert (out[-4:] == 0).all()
    assert (out[:-4] != 0).any()


@requires_esm2
def test_frozen_backbone_stays_out_of_state_dict():
    plm = FrozenPLM(ESM2_DIR)
    plm.load(device="cpu", dtype=torch.float32)
    assert not any("_backbone" in k for k in plm.state_dict())


@requires_esm2
def test_join_choice_only_affects_cross_chain_information():
    """``per_chain`` must be invariant to what the *other* chain is; ``joined``
    must not be. This is the property the experiment measures, checked on the
    real weights rather than on a random init."""
    plm_pc = FrozenPLM(ESM2_DIR, join="per_chain")
    plm_pc.load(device="cpu", dtype=torch.float32)
    plm_jo = FrozenPLM(ESM2_DIR, join="joined")
    plm_jo._backbone, plm_jo._tok = plm_pc._backbone, plm_pc._tok

    torch.manual_seed(0)
    first = torch.randint(0, 20, (6,))
    partner_a, partner_b = torch.randint(0, 20, (9,)), torch.randint(0, 20, (9,))

    def run(plm, partner):
        res_type = torch.cat([first, partner])
        asym_id = torch.cat([torch.zeros(6, dtype=torch.long), torch.ones(9, dtype=torch.long)])
        residue_index = torch.cat([torch.arange(6), torch.arange(9)])
        mol_type = torch.full((15,), const.MOL_PROTEIN)
        return plm(res_type, asym_id, residue_index, mol_type, torch.ones(15))[:6]

    assert torch.allclose(run(plm_pc, partner_a), run(plm_pc, partner_b), atol=1e-4)
    assert (run(plm_jo, partner_a) - run(plm_jo, partner_b)).abs().max() > 1e-3


def test_sentinel_free_tokenizer_packs_without_delimiters():
    """A backend with no BOS/EOS still gets correct per-chain segmentation:
    ``sequence_id`` comes from chain membership, not from token values."""
    from dataclasses import replace

    plm = FrozenPLM("/nonexistent", join="per_chain")
    plm._tok = replace(_TOK, cls_id=None, eos_id=None)
    ids = torch.tensor([10, 11, 12, 20, 21])
    asym = torch.tensor([0, 0, 0, 1, 1])
    packed, seq_id, pos = plm._pack(ids, asym)

    assert torch.equal(packed, ids)  # no delimiters inserted
    assert torch.equal(seq_id, torch.tensor([0, 0, 0, 1, 1]))
    assert torch.equal(pos, torch.arange(5))


def test_sentinel_packing_assigns_delimiters_to_their_chain():
    plm = FrozenPLM("/nonexistent", join="per_chain")
    plm._tok = _TOK
    ids = torch.tensor([10, 11, 20])
    asym = torch.tensor([0, 0, 1])
    packed, seq_id, pos = plm._pack(ids, asym)

    assert torch.equal(packed, torch.tensor([_CLS, 10, 11, _EOS, _CLS, 20, _EOS]))
    assert torch.equal(seq_id, torch.tensor([0, 0, 0, 0, 1, 1, 1]))
    assert torch.equal(packed[pos], ids)
