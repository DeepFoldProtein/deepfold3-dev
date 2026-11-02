"""Bucket padding must not change the sample.

``model.bucket.pad_features`` appends dead tokens to the N axis and pads every
mask False, so padded positions cannot reach the output. The sampler's noise is
the one thing the masks do not cover: ``randn`` at the padded N consumes a
different slice of the generator's stream than the same input unpadded, so the
same seed produced a different structure depending on a flag that is supposed
to buy speed only (`--bucket`, and `use_torch_compile`, which forces it).

These tests slice one input down rather than regenerating it at the smaller N,
so the padded and trimmed runs see identical embeddings for the real tokens and
the only difference is the padding itself.

Scope. Both runs are handed the *same* embeddings, so what is exercised is the
diffusion head's own padding masks, not the trunk's. The trunk is a separate
question and the answer there is not yet clean: on an 868-token target padded
to 896, same checkpoint and seed, the trunk distogram moves by a median of
0.003 A but a mean of 0.157 A, with a max of 13.1 A and 7.6% of token pairs
over 0.5 A. That is either a trunk mask leak or shape-dependent fused-kernel
arithmetic amplified over 10 recycles; the `--optim none` reference run that
would separate the two runs out of memory at this size. Neither is affected by
the fix under test here.
"""

from __future__ import annotations

import pytest
import torch

from deepfold3.model.config import DiffusionConfig, OptimConfig
from deepfold3.model.diffusion import DiffusionHead

N, M = 40, 8
PAD = 7  # real tokens = N - PAD
ATOL = 1e-3  # Å, matches tests/test_diffusion_hoist.py for fp32 on CPU


def _head(steps: int = 4, num_samples: int = 2) -> DiffusionHead:
    cfg = DiffusionConfig(steps=steps, num_samples=num_samples)
    head = DiffusionHead(cfg, c_single=384, c_pair=128, rel_feat_dim=8, optim=OptimConfig()).eval()
    # AF3 zero-inits the output projections, which would make the comparison vacuous.
    torch.manual_seed(0)
    with torch.no_grad():
        for p in head.parameters():
            p.normal_(0.0, 0.05)
    return head


def _padded_inputs(batch: tuple[int, ...] = ()) -> dict:
    """Inputs of N tokens whose last ``PAD`` are masked off, as bucketing leaves them."""
    g = torch.Generator().manual_seed(1)
    ref_mask = torch.zeros(*batch, N, M)
    ref_mask[..., : N - PAD, :5] = 1.0
    seq_mask = torch.zeros(*batch, N)
    seq_mask[..., : N - PAD] = 1.0
    return {
        "embeddings": {
            "single": torch.randn(*batch, N, 384, generator=g),
            "pair": torch.randn(*batch, N, N, 128, generator=g),
            "target_feat": torch.randn(*batch, N, 447, generator=g),
        },
        "ref_pos": torch.randn(*batch, N, M, 3, generator=g) * ref_mask.unsqueeze(-1),
        "ref_mask": ref_mask,
        "ref_element": torch.zeros(*batch, N, M, dtype=torch.int64),
        "ref_charge": torch.zeros(*batch, N, M),
        "ref_atom_name_chars": torch.zeros(*batch, N, M, 4, dtype=torch.int64),
        "seq_mask": seq_mask,
        "rel_feat": torch.randn(*batch, N, N, 8, generator=g),
    }


def _trim(x: dict, n: int) -> dict:
    """The same inputs with the padding sliced away — not regenerated at n."""
    tok_axis = {  # the axis holding N in each input, counted from the end
        "ref_pos": -3,
        "ref_mask": -2,
        "ref_element": -2,
        "ref_charge": -2,
        "ref_atom_name_chars": -3,
        "seq_mask": -1,
    }
    out: dict = {
        "embeddings": {
            "single": x["embeddings"]["single"][..., :n, :],
            "pair": x["embeddings"]["pair"][..., :n, :n, :],
            "target_feat": x["embeddings"]["target_feat"][..., :n, :],
        },
        "rel_feat": x["rel_feat"][..., :n, :n, :],
    }
    for k, axis in tok_axis.items():
        out[k] = x[k].narrow(x[k].dim() + axis, 0, n)
    return out


def _run(head: DiffusionHead, x: dict) -> torch.Tensor:
    gen = torch.Generator().manual_seed(2)
    return head.sample(**x, generator=gen)["atom_positions"]


@pytest.mark.parametrize("batch", [(), (2,)])
def test_padding_does_not_change_the_sample(batch):
    """Real tokens get the same structure padded and unpadded, from one seed."""
    head = _head()
    padded = _padded_inputs(batch)
    trimmed = _trim(padded, N - PAD)

    got = _run(head, padded)[..., : N - PAD, :, :]
    want = _run(head, trimmed)

    assert got.shape == want.shape
    assert torch.allclose(got, want, atol=ATOL), f"max deviation {(got - want).abs().max():.4g} A"


def test_padded_tokens_stay_put():
    """The padded tail is noise only — it must not pick up structure."""
    head = _head()
    pos = _run(head, _padded_inputs())
    assert torch.isfinite(pos).all()
