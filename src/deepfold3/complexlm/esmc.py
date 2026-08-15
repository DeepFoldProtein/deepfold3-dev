"""Load ESMC's released weights into :class:`~deepfold3.complexlm.model.ComplexLM`.

The two architectures were not made compatible after the fact — this model's
blocks were written to ESMC's design in the first place, so the correspondence
is one tensor to one tensor:

===============================================  =========================
ESMC                                             here
===============================================  =========================
``embed.weight``                                 ``embed.weight``
``transformer.blocks.i.attn.layernorm_qkv.0``    ``blocks.i.attn.norm``
``transformer.blocks.i.attn.layernorm_qkv.1``    ``blocks.i.attn.qkv``
``transformer.blocks.i.attn.q_ln`` / ``k_ln``    ``blocks.i.attn.q_norm`` / ``k_norm``
``transformer.blocks.i.attn.out_proj``           ``blocks.i.attn.out_proj``
``transformer.blocks.i.ffn.0``                   ``blocks.i.ffn.norm``
``transformer.blocks.i.ffn.1`` / ``.3``          ``blocks.i.ffn.w_in`` / ``w_out``
``transformer.norm``                             ``norm``
``sequence_head.0`` / ``.2`` / ``.3``            ``head.dense`` / ``head.norm`` / ``head.out``
===============================================  =========================

Two things make it work without any remapping of values. ESM3's
``SEQUENCE_VOCAB`` opens with the same 33 tokens as ESM2 in the same order, and
this model uses ESM2's ids, so the embedding and the readout are **row slices**
of ESMC's 64-row tables rather than a permutation. And ``swiglu_hidden`` returns
exactly ESMC's FFN width (3072 at ``d_model`` 1152), because it is ESMC's rule.

What does **not** transfer is positional: ESMC trained every layer with rotary
embeddings, and half the layers here run complex-scoped without any. Those
blocks keep ESMC's projections and FFN but their attention pattern is
effectively re-learned, so expect an early loss spike concentrated there. That
is the intended trade — the alternative is training the whole trunk from
scratch on a budget two orders of magnitude below ESMC's.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

#: ESMC releases and the shape each one requires.
ESMC_SHAPES = {
    "esmc-300m": {"d_model": 960, "n_layers": 30, "n_heads": 15},
    "esmc-600m": {"d_model": 1152, "n_layers": 36, "n_heads": 18},
}


def _block_map(i: int) -> dict[str, str]:
    src, dst = f"transformer.blocks.{i}", f"blocks.{i}"
    return {
        f"{src}.attn.layernorm_qkv.0": f"{dst}.attn.norm",
        f"{src}.attn.layernorm_qkv.1": f"{dst}.attn.qkv",
        f"{src}.attn.q_ln": f"{dst}.attn.q_norm",
        f"{src}.attn.k_ln": f"{dst}.attn.k_norm",
        f"{src}.attn.out_proj": f"{dst}.attn.out_proj",
        f"{src}.ffn.0": f"{dst}.ffn.norm",
        f"{src}.ffn.1": f"{dst}.ffn.w_in",
        f"{src}.ffn.3": f"{dst}.ffn.w_out",
    }


@torch.no_grad()
def load_esmc(model, path: str | Path, strict: bool = True) -> dict:
    """Copy ESMC weights into ``model``; returns a report of what moved.

    Args:
        model: A :class:`ComplexLM` whose config has ``esmc_compat`` set and
            whose shape matches an entry of :data:`ESMC_SHAPES`.
        strict: Raise if any target parameter is left untouched. The point of
            a warm start is that nothing is silently random, so this defaults
            on; the caller can relax it deliberately.
    """
    blob = torch.load(path, map_location="cpu", weights_only=False)
    src = blob.get("state_dict", blob) if isinstance(blob, dict) else blob
    cfg = model.cfg
    if not cfg.esmc_compat:
        raise ValueError("model.cfg.esmc_compat must be true; ESMC's norm biases do not otherwise exist")

    mapping = {"embed": "embed", "transformer.norm": "norm"}
    mapping |= {"sequence_head.0": "head.dense", "sequence_head.2": "head.norm", "sequence_head.3": "head.out"}
    for i in range(cfg.n_layers):
        mapping |= _block_map(i)

    target = dict(model.named_parameters())
    copied, skipped = [], []
    for src_prefix, dst_prefix in mapping.items():
        for suffix in ("weight", "bias"):
            s_key, d_key = f"{src_prefix}.{suffix}", f"{dst_prefix}.{suffix}"
            if s_key not in src or d_key not in target:
                if s_key in src:
                    skipped.append(s_key)
                continue
            s_val, d_val = src[s_key], target[d_key]
            if s_val.shape != d_val.shape:
                # Only the vocabulary axis may differ: ESM3's table is padded to
                # 64 rows and its first 33 are ESM2's, in order.
                if s_val.shape[1:] == d_val.shape[1:] and s_val.shape[0] > d_val.shape[0]:
                    s_val = s_val[: d_val.shape[0]]
                else:
                    raise ValueError(f"{s_key} {tuple(s_val.shape)} does not fit {d_key} {tuple(d_val.shape)}")
            d_val.copy_(s_val)
            copied.append(d_key)

    missing = sorted(set(target) - set(copied))
    report = {
        "checkpoint": str(path),
        "copied": len(copied),
        "left_random": missing,
        "unused_source": sorted(skipped),
    }
    if missing and strict:
        raise ValueError(f"{len(missing)} parameters were not initialised from ESMC: {missing[:8]}")
    logger.info("ESMC warm start: %d tensors copied, %d left random", len(copied), len(missing))
    return report
