"""Protein-language-model conditioning of the pair track.

A frozen PLM embeds the (optionally chain-joined) protein sequence; a small
trainable shim projects its hidden states into a pair representation that is
added to the Evoformer's initial ``pair``. Ported from ESMFold2's
``LanguageModelShim`` (``references/biohub-transformers`` →
``models/esmfold2/modeling_esmfold2_common.py``), with the backend left
abstract so ESM2 / ESMC / a future complex-trained PLM are interchangeable.

Why the pair track rather than ``single``: the deficit this targets is
cross-chain information in ``z`` — the pose is committed at the step-0 denoise
and tracks how much cross-chain signal ``z`` carries. Protenix adds its ESM
embedding to ``s_inputs`` instead, which reaches ``z`` only through the outer
product and carries no cross-chain term at all.

Two knobs decide whether any cross-chain signal exists at all:

* ``per_chain`` runs the PLM once per chain. The off-diagonal blocks of
  ``lm_z`` are then an outer product of two independently embedded chains —
  no co-evolutionary or contact information crosses the boundary. This is the
  control arm.
* ``eos_bos`` concatenates chains as ``[BOS] c1 [EOS BOS] c2 ... [EOS]`` and
  runs the PLM once, so its self-attention spans chain boundaries. This is
  what ESMFold2 does. Note it is out-of-distribution for ESM2, which never saw
  a mid-sequence separator during pre-training; whether the resulting
  cross-chain attention is signal or noise is exactly what the A/B measures.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleToPair(nn.Module):
    """Lift a per-token embedding to a pair embedding.

    Concatenates the outer product and the outer difference before the MLP.
    The difference term is what distinguishes this from AF3's
    ``OuterProductMean``: it is antisymmetric, so ``(i, j)`` and ``(j, i)``
    are distinguishable and directional information survives.
    """

    def __init__(self, input_dim: int, downproject_dim: int, output_dim: int) -> None:
        super().__init__()
        self.downproject = nn.Linear(input_dim, downproject_dim, bias=False)
        self.output_mlp = nn.Sequential(
            nn.Linear(2 * downproject_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[..., N, d_in] -> [..., N, N, d_out]."""
        x = self.downproject(x)
        x = torch.cat(
            [x.unsqueeze(-2) * x.unsqueeze(-3), x.unsqueeze(-2) - x.unsqueeze(-3)],
            dim=-1,
        )
        return self.output_mlp(x)


class LMPairShim(nn.Module):
    """Project frozen PLM hidden states into a pair representation.

    The layer mixture is learned: ``layer_combine`` is zero-initialised, so the
    softmax starts uniform over all ``num_layers + 1`` hidden states (embedding
    output included) and the model discovers which depths carry usable
    structure signal. Keeping every layer is the reason the backend must run
    with ``output_hidden_states=True``.

    Args:
        d_pair: Width of the trunk pair representation to match.
        d_model: Backend hidden width (ESM2-650M 1280, ESMC-6B 2560).
        num_layers: Backend transformer depth; ``num_layers + 1`` states are
            mixed (ESM2-650M 33, ESMC-6B 80).
    """

    def __init__(self, d_pair: int, d_model: int, num_layers: int) -> None:
        super().__init__()
        self.d_pair = d_pair
        self.num_states = num_layers + 1
        self.to_pair_channels = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_pair, bias=False),
        )
        self.layer_combine = nn.Parameter(torch.zeros(self.num_states))
        self.to_pair = nn.Sequential(SingleToPair(d_pair, d_pair, d_pair), nn.LayerNorm(d_pair))

    def forward(self, hidden_states: torch.Tensor, *, dropout: float = 0.0) -> torch.Tensor:
        """Args:
            hidden_states: ``[..., N, num_layers + 1, d_model]``, already
                scattered onto the trunk's token layout and **detached** — the
                backend is frozen and must not receive gradient.
            dropout: Applied to the pair output. ESMFold2 forces this on at
                inference too (``training=True``), resampling per recycle, so
                the trunk cannot become dependent on the PLM path.

        Returns:
            ``[..., N, N, d_pair]``.
        """
        if hidden_states.shape[-2] != self.num_states:
            raise ValueError(
                f"expected {self.num_states} hidden states, got {hidden_states.shape[-2]} — "
                "backend depth does not match the configured num_layers"
            )
        z = self.to_pair_channels(hidden_states)  # [..., N, num_states, d_pair]
        weights = self.layer_combine.softmax(0)
        z = torch.einsum("s,...nsd->...nd", weights, z)
        z = self.to_pair(z)  # [..., N, N, d_pair]
        if dropout > 0:
            z = F.dropout(z, p=dropout, training=True)
        return z
