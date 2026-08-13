"""The packed batch — one flat token axis, two nestings of it.

Every example in a micro-batch is concatenated into a single 1-D token axis
with no padding. Two cumulative-length vectors describe the same tensor at two
granularities::

    tokens      [A1 A2 ... A100 | B1 ... B200 || C1 ... C50 | D1 ... D80 | E1 ... E70]
    chain_cu    0             100            300          350          430         500
    complex_cu  0                            300                                   500

A chain-scoped layer attends inside ``chain_cu``; a complex-scoped layer
attends inside ``complex_cu``. Nothing about the hidden tensor changes between
them — the layer kind changes which cumulative-length vector goes to the
kernel, and that is the whole implementation of "different positional
treatment for intra- and inter-chain interaction".

Two position axes are carried because the baseline needs one the main model
does not: ``positions`` restarts at 0 for every chain (what RoPE sees under
layer kind ``c``), while ``global_positions`` runs across the whole complex
(what the global-RoPE baseline, kind ``r``, sees). Four bytes a token buys the
architecture ablation.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch

IGNORE_INDEX = -100


@dataclass
class PackedBatch:
    """A micro-batch of complexes packed into one token axis.

    Attributes:
        tokens: ``[T]`` input token ids, already corrupted by the masker.
        labels: ``[T]`` targets, :data:`IGNORE_INDEX` where the position is not
            scored (i.e. everywhere the masker did not select).
        positions: ``[T]`` residue index within its chain, restarting at 0.
        global_positions: ``[T]`` residue index within its complex.
        chain_cu: ``[n_chains + 1]`` chain boundaries, ``int32``.
        complex_cu: ``[n_complexes + 1]`` complex boundaries, ``int32``.
        chain_index: ``[T]`` chain slot id, globally unique in the batch.
        complex_index: ``[T]`` complex id, globally unique in the batch.
        entity_index: ``[T]`` entity id, globally unique in the batch; equal
            values mean identical sequence, which is what homomer-paired
            masking and the homo/hetero split read.
        iface: ``[T]`` bool, true at interface residues.
        iface_known: ``[T]`` bool, false where the source has no interface
            labels at all. Without it a sequence-only source's all-zero flags
            read as "every residue is non-interface", and the
            interface-versus-non-interface comparison silently becomes a
            comparison between sources.
        source: ``[n_complexes]`` source id, for per-source evaluation.
    """

    tokens: torch.Tensor
    labels: torch.Tensor
    positions: torch.Tensor
    global_positions: torch.Tensor
    chain_cu: torch.Tensor
    complex_cu: torch.Tensor
    chain_index: torch.Tensor
    complex_index: torch.Tensor
    entity_index: torch.Tensor
    iface: torch.Tensor
    iface_known: torch.Tensor
    source: torch.Tensor
    max_chain_len: int
    max_complex_len: int

    @property
    def n_tokens(self) -> int:
        return int(self.tokens.numel())

    @property
    def n_chains(self) -> int:
        return int(self.chain_cu.numel() - 1)

    @property
    def n_complexes(self) -> int:
        return int(self.complex_cu.numel() - 1)

    def to(self, device: torch.device | str, non_blocking: bool = False) -> PackedBatch:
        moved = {}
        for f in fields(self):
            value = getattr(self, f.name)
            moved[f.name] = value.to(device, non_blocking=non_blocking) if torch.is_tensor(value) else value
        return PackedBatch(**moved)

    def pin_memory(self) -> PackedBatch:
        moved = {}
        for f in fields(self):
            value = getattr(self, f.name)
            moved[f.name] = value.pin_memory() if torch.is_tensor(value) else value
        return PackedBatch(**moved)

    def __repr__(self) -> str:
        return (
            f"PackedBatch(T={self.n_tokens}, complexes={self.n_complexes}, chains={self.n_chains}, "
            f"masked={int((self.labels != IGNORE_INDEX).sum())})"
        )
