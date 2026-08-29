"""Tests for the shape-complementarity loss (OpenDDE arXiv:2607.03787 §A.2.1).

The fixtures are two facing slabs: chain A fills ``z ∈ [-3, 0]`` with its
representative atoms on the ``z = 0`` face, chain B is the mirror image at
``z = gap``. The same-chain density gradient then points ``+z`` for A and
``-z`` for B, so the pair term sees a textbook complementary interface and the
geometry is checkable by hand.
"""

import pytest
import torch

from deepfold3.data import const
from deepfold3.training.loss import shape_complementarity_loss

GRID = 5
SPACING = 3.8
DEPTH = (0.0, -1.5, -3.0)  # per-residue atom offsets along the burial direction
N_SLOT = 4


def _slab(gap: float, grid: int = GRID) -> tuple[torch.Tensor, ...]:
    """Two facing slabs separated by ``gap`` Angstroms along z."""
    pos = torch.zeros(2 * grid * grid, N_SLOT, 3, dtype=torch.float64)
    mask = torch.zeros(2 * grid * grid, N_SLOT, dtype=torch.bool)
    asym = torch.zeros(2 * grid * grid, dtype=torch.long)
    t = 0
    for chain in (0, 1):
        sign = 1.0 if chain == 0 else -1.0
        base_z = 0.0 if chain == 0 else gap
        for i in range(grid):
            for j in range(grid):
                for slot, dz in enumerate(DEPTH):
                    pos[t, slot] = torch.tensor([i * SPACING, j * SPACING, base_z + sign * dz])
                    mask[t, slot] = True
                asym[t] = chain
                t += 1
    n = pos.shape[0]
    return (
        pos,
        mask,
        torch.zeros(n, dtype=torch.long),  # token_to_rep_atom: slot 0 is the surface atom
        asym,
        torch.full((n,), const.MOL_PROTEIN, dtype=torch.long),
        torch.ones(n, dtype=torch.bool),
    )


def _loss(pred: torch.Tensor, gt_args: tuple[torch.Tensor, ...], **kw):
    pos, mask, rep, asym, mol, seq = gt_args
    return shape_complementarity_loss(pred, pos, mask, rep, asym, mol, seq, **kw)


def _shift_chain_b(gt_args: tuple[torch.Tensor, ...], dz: float) -> torch.Tensor:
    pos, _, _, asym, _, _ = gt_args
    moved = pos.clone()
    moved[asym == 1, :, 2] += dz
    return moved.unsqueeze(0)


def test_zero_at_ground_truth():
    """The predicted field equals the GT field exactly when the structure does."""
    args = _slab(gap=6.0)
    loss, comp = _loss(args[0].unsqueeze(0).expand(3, -1, -1, -1), args)
    assert float(loss) == pytest.approx(0.0, abs=1e-12)
    assert float(comp["shape_comp_coverage"]) == 1.0
    assert float(comp["shape_comp_glob_gt"]) > 0.1  # the fixture really has an interface


def test_se3_invariance():
    """A rigid motion of the whole system leaves the loss at zero."""
    args = _slab(gap=6.0)
    torch.manual_seed(0)
    q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    rot = q * torch.sign(torch.det(q))
    moved = args[0] @ rot.T + torch.tensor([13.0, -7.0, 41.0], dtype=torch.float64)
    loss, _ = _loss(moved.unsqueeze(0), args)
    assert float(loss) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("direction", [1.0, -1.0], ids=["separate", "compress"])
def test_monotone_in_interface_error(direction):
    """The loss rises monotonically as the interface is pulled apart or squeezed."""
    args = _slab(gap=6.0)
    losses = [float(_loss(_shift_chain_b(args, direction * d), args)[0]) for d in (0.0, 0.5, 1.0, 2.0, 3.0)]
    assert losses[0] == pytest.approx(0.0, abs=1e-12)
    assert all(b > a for a, b in zip(losses, losses[1:], strict=False)), losses


def test_compression_regime_is_not_silent():
    """GT-only masking still sees an interface being crushed.

    The pair set comes from the ground truth, so a *new* contact the prediction
    invents is invisible; a GT contact driven below ``clash_distance`` is not.
    That is the regime df3 actually fails in (collapsed blob), so it must carry
    signal.

    Note this asserts only that compression is penalised and monotone. On real
    complexes compression is penalised *harder* than separation
    (``export/shape-complementarity/data/probe6_chosen_config.txt``: push 8 A =
    2.06e-3 vs pull 8 A = 1.19e-3), but that asymmetry comes from rough surfaces
    and a spread of native gaps — an idealised flat slab sitting exactly at the
    ``gap_mean`` optimum does not reproduce it, and this fixture is not evidence
    either way.
    """
    args = _slab(gap=6.0)
    crushed = [float(_loss(_shift_chain_b(args, -d), args)[0]) for d in (1.0, 2.0, 3.0, 4.0)]
    assert all(b > a for a, b in zip(crushed, crushed[1:], strict=False)), crushed
    assert crushed[-1] > 0.5, "a 4 A crush must be a large error, not a rounding one"


def test_smooth_l1_not_huber_scaling():
    """``F.huber_loss(delta=beta)`` would shrink the term by ``beta`` (=100x here).

    Above the knee ``smooth_l1`` has slope 1, so for residuals well above beta
    the token term must sit within beta/2 of the raw L1.
    """
    args = _slab(gap=6.0)
    _, comp = _loss(_shift_chain_b(args, 3.0), args, huber_delta=0.01)
    l1 = float(comp["shape_comp_l1_token"])
    huber = float(comp["shape_comp_token"])
    assert l1 > 0.05, "fixture must produce residuals well above beta"
    assert huber == pytest.approx(l1 - 0.005, rel=0.05)


def test_empty_interface_returns_zero_with_graph():
    """A single-chain crop contributes nothing but keeps the autograd graph shape."""
    pos, mask, rep, asym, mol, seq = _slab(gap=6.0)
    mono = asym == 0
    args = (pos[mono], mask[mono], rep[mono], asym[mono], mol[mono], seq[mono])
    pred = args[0].unsqueeze(0).clone().requires_grad_(True)
    loss, comp = _loss(pred, args)
    assert float(loss.detach()) == 0.0
    assert loss.grad_fn is not None
    assert float(comp["shape_comp_coverage"]) == 0.0
    loss.backward()
    assert torch.all(pred.grad == 0)


def test_non_protein_tokens_are_skipped():
    """The construction is protein-only: a DNA/DNA interface scores nothing."""
    pos, mask, rep, asym, _, seq = _slab(gap=6.0)
    mol = torch.full_like(asym, const.MOL_DNA)
    loss, comp = shape_complementarity_loss(pos.unsqueeze(0), pos, mask, rep, asym, mol, seq)
    assert float(loss) == 0.0
    assert float(comp["shape_comp_coverage"]) == 0.0


def test_gradient_is_finite_at_close_contact():
    args = _slab(gap=6.0)
    for gap in (2.5, 1.0, 0.2):
        pred = _shift_chain_b(args, gap - 6.0).requires_grad_(True)
        loss, _ = _loss(pred, args)
        loss.backward()
        assert torch.isfinite(pred.grad).all(), f"non-finite gradient at gap={gap}"
        assert float(pred.grad.abs().max()) > 0.0


def test_chunk_size_invariance():
    args = _slab(gap=6.0)
    pred = _shift_chain_b(args, 2.0)
    ref = float(_loss(pred, args, chunk_size=0)[0])
    for chunk in (7, 16, 128):
        assert float(_loss(pred, args, chunk_size=chunk)[0]) == pytest.approx(ref, rel=1e-9)


def test_sample_axis_is_averaged():
    """Two samples, one perfect and one displaced, give half the displaced loss."""
    args = _slab(gap=6.0)
    bad = _shift_chain_b(args, 2.0)
    both = torch.cat([args[0].unsqueeze(0), bad], dim=0)
    solo = float(_loss(bad, args)[0])
    pair = float(_loss(both, args)[0])
    assert pair == pytest.approx(solo / 2, rel=1e-9)


# ---------------------------------------------------------------------------
# Integration with the loss entry points
# ---------------------------------------------------------------------------


def _slab_features(gap: float = 6.0) -> tuple[dict, dict]:
    """A ``(pred, features)`` pair carrying the two-slab interface."""
    from deepfold3.data import const as _const

    pos, mask, rep, asym, mol, seq = _slab(gap=gap, grid=3)
    pos, mask = pos.float(), mask
    n, m = mask.shape
    s = 2
    displaced = pos.unsqueeze(0).repeat(s, 1, 1, 1)
    displaced[:, asym == 1, :, 2] += 2.0
    pred = {
        "denoised": displaced,
        "gt_augmented": pos.unsqueeze(0).repeat(s, 1, 1, 1),
        "noise_levels": torch.tensor([0.5, 40.0]),
        "distogram": torch.zeros(n, n, 64),
    }
    features = {
        "atom_gt_positions": pos,
        "atom_gt_mask": mask,
        "seq_mask": seq.float(),
        "token_to_rep_atom": rep,
        "asym_id": asym,
        "mol_type": torch.full((n,), _const.MOL_PROTEIN, dtype=torch.long),
    }
    assert m == N_SLOT
    return pred, features


def test_training_loss_gates_on_alpha_and_features():
    """Off by default; on when ``alpha_shape_comp > 0`` and the features exist."""
    from deepfold3.model.config import TrainingConfig
    from deepfold3.training.loss import training_loss

    pred, features = _slab_features()

    off = TrainingConfig(alpha_confidence=0.0)
    _, losses_off = training_loss(pred, features, off)
    assert "shape_comp_token" not in losses_off

    on = TrainingConfig(alpha_confidence=0.0, alpha_shape_comp=3.0e-2)
    total, losses = training_loss(pred, features, on)
    assert total.isfinite()
    assert float(losses["shape_comp_coverage"]) == 1.0
    assert float(losses["shape_comp_token"]) > 0.0

    # Missing ``asym_id`` must skip the term rather than raise.
    stripped = {k: v for k, v in features.items() if k != "asym_id"}
    _, losses_stripped = training_loss(pred, stripped, on)
    assert "shape_comp_token" not in losses_stripped


def test_sigma_gate_drops_high_noise_samples():
    """``shape_comp_sigma_max`` weights samples without changing graph topology."""
    from deepfold3.model.config import TrainingConfig
    from deepfold3.training.loss import training_loss

    pred, features = _slab_features()
    ungated = TrainingConfig(alpha_confidence=0.0, alpha_shape_comp=3.0e-2)
    gated = TrainingConfig(alpha_confidence=0.0, alpha_shape_comp=3.0e-2, shape_comp_sigma_max=1.0)
    _, l_un = training_loss(pred, features, ungated)
    _, l_ga = training_loss(pred, features, gated)
    # Both samples carry the same displacement, so gating one out changes
    # nothing numerically -- what it must not do is error or NaN.
    assert float(l_ga["shape_comp_token"]) == pytest.approx(float(l_un["shape_comp_token"]), rel=1e-6)

    all_gated = TrainingConfig(alpha_confidence=0.0, alpha_shape_comp=3.0e-2, shape_comp_sigma_max=1e-6)
    total, l_all = training_loss(pred, features, all_gated)
    assert float(l_all["shape_comp_token"]) == 0.0
    assert total.isfinite()


def test_low_precision_inputs_are_upcast():
    """bf16 in must not be computed in bf16.

    The field is ``exp(-d²/2σ²)`` valued in [0, 1] with residuals at the 1e-2
    scale — bf16's 8-bit mantissa would spend itself on the exponent. OpenDDE
    disables autocast around the same computation.
    """
    args = _slab(gap=6.0, grid=3)
    fp32_args = tuple(a.float() if a.is_floating_point() else a for a in args)
    pred32 = _shift_chain_b(fp32_args, 2.0)
    ref = float(_loss(pred32, fp32_args)[0])

    bf16_args = (args[0].bfloat16(),) + args[1:]
    loss, _ = shape_complementarity_loss(pred32.bfloat16(), *bf16_args)
    assert loss.dtype == torch.float32
    assert float(loss) == pytest.approx(ref, rel=5e-2)
