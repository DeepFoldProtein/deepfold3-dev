"""Per-chain-pair ipTM (``compute_chain_pair_iptm``).

The global ipTM pools every cross-chain token pair into one number, so it cannot
rank a prediction whose interfaces differ in quality (an Fab's H-Ag vs L-Ag).
These tests pin the two properties that make the per-pair version usable for
that: it agrees exactly with the global score when there is only one interface,
and it separates a confident interface from a bad one when there is more.
"""

from __future__ import annotations

import torch

from deepfold3.io.confidences import _DEFAULT_CFG, compute_chain_pair_iptm, compute_iptm

NUM_BINS = _DEFAULT_CFG.num_distogram_bins
MAX_ERR = _DEFAULT_CFG.max_error_bin


def _logits(asym: torch.Tensor, pair_error: dict[tuple[int, int], int], default_bin: int = 5) -> torch.Tensor:
    """One-hot-ish PAE logits: ``pair_error`` sets the peak bin per chain pair.

    A low bin is a confident (small predicted error) pair, a high bin a bad one.
    """
    n = len(asym)
    logits = torch.zeros(n, n, NUM_BINS)
    for i in range(n):
        for j in range(n):
            key = (int(asym[i]), int(asym[j]))
            b = pair_error.get(key, pair_error.get((key[1], key[0]), default_bin))
            logits[i, j, b] = 20.0
    return logits


def _masks(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    seq_mask = torch.ones(n)
    return seq_mask, seq_mask.unsqueeze(-1) * seq_mask.unsqueeze(-2)


def test_two_chains_match_global_iptm():
    """One interface -> the pair's value IS the global ipTM, bit for bit."""
    asym = torch.tensor([0] * 12 + [1] * 12)
    seq_mask, pair_mask = _masks(len(asym))
    logits = _logits(asym, {(0, 1): 3})

    glob = compute_iptm(logits, seq_mask, pair_mask, asym, MAX_ERR, NUM_BINS)
    per_pair = compute_chain_pair_iptm(logits, seq_mask, pair_mask, asym, MAX_ERR, NUM_BINS)

    assert list(per_pair) == [(0, 1)]
    assert per_pair[(0, 1)] == glob


def test_separates_a_good_interface_from_a_bad_one():
    """The number the global score cannot give: which interface is trusted."""
    asym = torch.tensor([0] * 10 + [1] * 10 + [2] * 10)
    seq_mask, pair_mask = _masks(len(asym))
    # A-B confidently aligned (low error bin), A-C badly (high error bin).
    logits = _logits(asym, {(0, 1): 1, (0, 2): NUM_BINS - 1, (1, 2): NUM_BINS - 1})

    per_pair = compute_chain_pair_iptm(logits, seq_mask, pair_mask, asym, MAX_ERR, NUM_BINS)
    glob = compute_iptm(logits, seq_mask, pair_mask, asym, MAX_ERR, NUM_BINS)

    assert set(per_pair) == {(0, 1), (0, 2), (1, 2)}
    # Absolute values stay small here because the interface d0 is sized on the
    # two chains (20 tokens -> d0 = 0.32 A), so only the ordering is meaningful.
    assert per_pair[(0, 1)] > 5 * per_pair[(0, 2)]
    assert per_pair[(0, 1)] > 5 * per_pair[(1, 2)]
    # The global score is dominated by the best-aligned row, so it tracks the
    # good pair and says nothing about the bad one — the reason for this split.
    assert glob >= per_pair[(0, 2)]


def test_single_chain_has_no_pairs():
    asym = torch.zeros(8, dtype=torch.long)
    seq_mask, pair_mask = _masks(len(asym))
    logits = _logits(asym, {})
    assert compute_chain_pair_iptm(logits, seq_mask, pair_mask, asym, MAX_ERR, NUM_BINS) == {}


def test_fully_masked_pair_is_omitted():
    """A pair with no unmasked token pair yields no key (not a 0.0 score)."""
    asym = torch.tensor([0] * 6 + [1] * 6 + [2] * 6)
    seq_mask, pair_mask = _masks(len(asym))
    # Mask out every A-C / C-A entry.
    in_a, in_c = asym == 0, asym == 2
    ac = (in_a.unsqueeze(-1) & in_c.unsqueeze(-2)) | (in_c.unsqueeze(-1) & in_a.unsqueeze(-2))
    pair_mask = pair_mask * (~ac).float()
    logits = _logits(asym, {})

    per_pair = compute_chain_pair_iptm(logits, seq_mask, pair_mask, asym, MAX_ERR, NUM_BINS)
    assert (0, 2) not in per_pair
    assert {(0, 1), (1, 2)} <= set(per_pair)
