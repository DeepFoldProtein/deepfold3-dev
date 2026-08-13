"""Residue alphabet for the complex language model.

Token ids are ESM2's, deliberately. Nothing here needs them to be — the model
is trained from scratch — but keeping them identical means a warm start from
``esm2_t33_650M_UR50D`` is a plain embedding copy, the frozen-PLM probes in
``export/plm-interface-probe`` can score this model without a second
tokenizer, and PPLM (which uses the same ESM-1b alphabet) lines up id for id.

Two ids that ESM2 spends and this model does not: ``cls`` and ``eos``. The v1
model has no BOS/EOS — a complex is a *set* of chains, and a sentinel at the
front of a serialisation would make the first chain special, which is exactly
the asymmetry the architecture exists to remove. The ids stay reserved so the
embedding table stays copy-compatible.
"""

from __future__ import annotations

import numpy as np

#: ESM2's standard tokens, in ESM2's order; ids are ``4 + index``.
ESM2_TOKENS = "LAGVSERTIDPKQNFYMHWCXBUZO"

CLS_ID = 0
PAD_ID = 1
EOS_ID = 2
UNK_ID = 3
MASK_ID = 32
VOCAB_SIZE = 33

#: The twenty standard amino acids, as token ids — the only classes whose
#: prediction accuracy is meaningful (``X``/``B``/``U``/``Z``/``O`` are
#: ambiguity or rarity codes and are reported separately, if at all).
STANDARD_TOKENS = "ACDEFGHIKLMNPQRSTVWY"


def ascii_to_id_table() -> np.ndarray:
    """``uint8[256]`` mapping a one-letter residue code to its token id.

    Unmapped bytes — including lowercase (an a3m insertion) and ``-`` (a gap) —
    fall to ``UNK_ID`` rather than raising, because these tables are applied to
    whole FASTA files at a time and one stray character should not lose a shard.
    """
    table = np.full(256, UNK_ID, dtype=np.uint8)
    for i, ch in enumerate(ESM2_TOKENS):
        table[ord(ch)] = 4 + i
    return table


def res_type_to_id_table() -> np.ndarray:
    """``uint8[n_res_type]`` mapping ``deepfold3.data.const`` res types to ids.

    Non-protein res types (nucleotides, ``-``) map to ``UNK_ID``; they are
    filtered out upstream, so the value is only a safe default.
    """
    from deepfold3.data import const

    three_to_one = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
        "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
        "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
        "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
        "UNK": "X",
    }  # fmt: skip
    table = np.full(len(const.RES_TYPE_ORDER), UNK_ID, dtype=np.uint8)
    ascii_table = ascii_to_id_table()
    for name, rt in const.RES_TYPE_ORDER.items():
        one = three_to_one.get(name)
        if one is not None:
            table[rt] = ascii_table[ord(one)]
    return table


def id_to_char_table() -> np.ndarray:
    """``|S1[VOCAB_SIZE]`` inverse of :func:`ascii_to_id_table`, for logging."""
    table = np.full(VOCAB_SIZE, b"?", dtype="S1")
    for i, ch in enumerate(ESM2_TOKENS):
        table[4 + i] = ch.encode()
    table[CLS_ID], table[PAD_ID], table[EOS_ID] = b"^", b".", b"$"
    table[UNK_ID], table[MASK_ID] = b"X", b"#"
    return table


def standard_token_ids() -> np.ndarray:
    """Token ids of the twenty standard amino acids, ascending."""
    table = ascii_to_id_table()
    return np.sort(np.array([table[ord(c)] for c in STANDARD_TOKENS], dtype=np.int64))


def decode(ids: np.ndarray) -> str:
    """Token ids -> one-letter string. Debugging aid; not on any hot path."""
    return id_to_char_table()[np.asarray(ids, dtype=np.int64)].tobytes().decode()
