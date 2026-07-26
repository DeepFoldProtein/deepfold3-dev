"""AF3 Algorithm 5 line 3: the atom-pair geometry gate is ``ref_space_uid``.

The gate decides which atom pairs contribute reference-conformer offsets and
distances to the atom encoder's pair representation. Token identity — the
previous implementation — is equivalent only for multi-atom polymer residue
tokens. For one-atom tokens (ligand, modified residue, atomized residue) it
collapses to the self-pair and silently drops the conformer geometry.
"""

import torch

from deepfold3.model.atom_cross_attn import (
    EvoformerConditioningEncoder,
    _offsets_valid,
    build_atom_subsets,
)


def _subsets(n_tokens: int, atoms_per_token: int):
    ref_mask = torch.ones(n_tokens, atoms_per_token, dtype=torch.bool)
    return build_atom_subsets(ref_mask, Q=32, K=128), ref_mask


class TestOffsetsValid:
    def test_falls_back_to_token_identity_when_uid_absent(self):
        subsets, _ = _subsets(4, 3)
        got = _offsets_valid(subsets, None)
        expected = subsets.token_for_query.unsqueeze(-1) == subsets.token_for_key.unsqueeze(-2)
        assert torch.equal(got, expected)

    def test_multi_atom_polymer_token_agrees_with_token_identity(self):
        """One residue = one token = one reference space → the two gates match."""
        n, m = 4, 3
        subsets, _ = _subsets(n, m)
        # uid == token index, which is what the featurizer produces when every
        # token is a whole residue.
        uid = torch.arange(n).unsqueeze(1).expand(n, m).contiguous()
        by_uid = _offsets_valid(subsets, uid)
        by_token = _offsets_valid(subsets, None)
        assert torch.equal(by_uid, by_token)

    def test_one_atom_tokens_of_one_residue_are_grouped(self):
        """The case token identity gets wrong: 4 atom tokens, one residue.

        Token identity admits only the 4 self-pairs; the ref_space_uid gate
        admits all 16 pairs, so the conformer geometry actually reaches the
        pair representation.
        """
        n, m = 4, 1
        subsets, _ = _subsets(n, m)
        uid = torch.zeros(n, m, dtype=torch.long)  # all four atoms, one residue
        by_uid = _offsets_valid(subsets, uid)
        by_token = _offsets_valid(subsets, None)

        real = subsets.ta_to_q_mask.unsqueeze(-1) & subsets.q_to_k_mask.unsqueeze(-2)
        assert by_uid[real].all(), "all atom pairs of one reference conformer must be valid"
        # Token identity admits only each atom paired with itself, so it lets
        # through strictly fewer pairs than the conformer grouping.
        assert by_token[real].sum() < by_uid[real].sum()

    def test_separate_residues_stay_separate(self):
        n, m = 4, 1
        subsets, _ = _subsets(n, m)
        uid = torch.tensor([[0], [0], [1], [1]])  # two residues of two atoms
        got = _offsets_valid(subsets, uid)
        q_tok = subsets.token_for_query
        k_tok = subsets.token_for_key
        real = subsets.ta_to_q_mask.unsqueeze(-1) & subsets.q_to_k_mask.unsqueeze(-2)
        same_res = (q_tok.unsqueeze(-1) // 2) == (k_tok.unsqueeze(-2) // 2)
        assert torch.equal(got[real], same_res[real])


class TestEncoderUsesTheGate:
    def test_conditioning_encoder_output_changes_for_atom_tokens(self):
        """End-to-end: grouping one-atom tokens changes the encoder output."""
        torch.manual_seed(0)
        n, m = 6, 1
        enc = EvoformerConditioningEncoder(
            per_atom_channels=32,
            per_token_channels=48,
            per_atom_pair_channels=8,
            num_blocks=1,
        )
        enc.eval()
        kwargs = dict(
            ref_pos=torch.randn(n, m, 3),
            ref_mask=torch.ones(n, m),
            ref_element=torch.full((n, m), 6, dtype=torch.long),
            ref_charge=torch.zeros(n, m),
            ref_atom_name_chars=torch.zeros(n, m, 4, dtype=torch.long),
        )
        with torch.no_grad():
            ungated = enc(**kwargs)  # token identity fallback
            grouped = enc(**kwargs, ref_space_uid=torch.zeros(n, m, dtype=torch.long))
        assert ungated.shape == grouped.shape
        assert not torch.allclose(ungated, grouped), "ref_space_uid grouping must reach the pair track"

    def test_whole_residue_tokens_are_unaffected(self):
        """Regression guard: multi-atom tokens keep their previous output."""
        torch.manual_seed(0)
        n, m = 3, 4
        enc = EvoformerConditioningEncoder(
            per_atom_channels=32,
            per_token_channels=48,
            per_atom_pair_channels=8,
            num_blocks=1,
        )
        enc.eval()
        kwargs = dict(
            ref_pos=torch.randn(n, m, 3),
            ref_mask=torch.ones(n, m),
            ref_element=torch.full((n, m), 6, dtype=torch.long),
            ref_charge=torch.zeros(n, m),
            ref_atom_name_chars=torch.zeros(n, m, 4, dtype=torch.long),
        )
        uid = torch.arange(n).unsqueeze(1).expand(n, m).contiguous()
        with torch.no_grad():
            before = enc(**kwargs)
            after = enc(**kwargs, ref_space_uid=uid)
        assert torch.allclose(before, after, atol=1e-6)
