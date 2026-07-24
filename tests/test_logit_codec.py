"""Tests for the logit codec + its inference-side integration.

Covers:
  * round-trip of ``save_logits`` / ``load_logits`` (schemes + self-describing
    ``attrs``), and softmax-recovery accuracy of ``int8_softmax``.
  * ``save_sample_logits`` / ``save_distogram_logits`` → ``load_logit_maps``
    reconstructs the same expected-value / contact maps as the direct torch
    ``compute_*`` helpers.
  * ``discover_logit_npz`` locating the codec files under both output layouts.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from deepfold3.io import confidences as conf
from deepfold3.io.logit_codec import DEFAULT_CLIP, load_logits, measure_recovery, save_logits, softmax
from deepfold3.model.config import ConfidenceConfig, DistogramConfig
from deepfold3.model.model import DeepFold3Output


def _peaked_logits(shape, seed=0):
    """Random logits with a per-position peak (well-defined expectation)."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(shape).astype(np.float32) * 2.0
    # Push a random bin up so softmax is peaked (realistic for a head output).
    peak = rng.integers(0, shape[-1], size=shape[:-1])
    np.put_along_axis(x, peak[..., None], 8.0, axis=-1)
    return x


# --------------------------------------------------------------------------- #
# Low-level codec
# --------------------------------------------------------------------------- #


def test_round_trip_all_schemes_preserve_shape_and_attrs(tmp_path):
    dg = _peaked_logits((12, 12, 64), seed=1)
    pae = _peaked_logits((12, 12, 64), seed=2)
    for scheme in ("float32", "float16", "int8", "int8_softmax", "int4_softmax"):
        p = tmp_path / f"{scheme}.npz"
        save_logits(p, distogram=dg, pae=pae, scheme=scheme, attrs={"scheme": scheme, "seed": 7})
        dec = load_logits(p)
        assert set(dec.heads) == {"distogram", "pae"}
        assert dec.heads["distogram"].shape == dg.shape
        assert dec.heads["pae"].shape == pae.shape
        assert dec.attrs["scheme"] == scheme
        assert dec.attrs["seed"] == 7
        # __getitem__ / __contains__ convenience
        assert "pae" in dec
        assert dec["pae"].shape == pae.shape


def test_int8_softmax_recovers_distribution():
    dg = _peaked_logits((8, 8, 64), seed=3)
    m = measure_recovery(dg, "int8_softmax", clip=DEFAULT_CLIP)
    assert m["kl"] < 1e-3
    # int8_softmax step ≈ clip/255 ≈ 0.063 in logit space → per-prob error ~1e-2.
    assert m["max_abs_prob"] < 1e-2
    assert m["bytes_per_value"] <= 1.01  # ~1 byte/value


def test_float32_scheme_is_lossless(tmp_path):
    pae = _peaked_logits((6, 6, 64), seed=4)
    p = tmp_path / "f32.npz"
    save_logits(p, pae=pae, scheme="float32")
    dec = load_logits(p)
    # Softmax must match to float precision (absolute logits preserved too).
    np.testing.assert_allclose(softmax(dec.heads["pae"]), softmax(pae), atol=1e-6)


def test_extra_arrays_pass_through(tmp_path):
    pae = _peaked_logits((4, 4, 64), seed=5)
    p = tmp_path / "extra.npz"
    save_logits(p, pae=pae, extra={"token_res_ids": np.arange(4)})
    dec = load_logits(p)
    assert "token_res_ids" in dec.extra
    np.testing.assert_array_equal(dec.extra["token_res_ids"], np.arange(4))


# --------------------------------------------------------------------------- #
# Inference integration
# --------------------------------------------------------------------------- #


def _fake_output(n=10, s=2, seed=11):
    cfg = ConfidenceConfig()
    b = cfg.num_distogram_bins
    pae = torch.from_numpy(_peaked_logits((s, n, n, b), seed=seed))
    pde = torch.from_numpy(_peaked_logits((s, n, n, b), seed=seed + 1))
    dg = torch.from_numpy(_peaked_logits((n, n, DistogramConfig.num_bins), seed=seed + 2))
    return DeepFold3Output(
        atom_positions=torch.zeros(s, n, 1, 3),
        atom_mask=torch.ones(n, 1),
        plddt=None,
        pae=pae,
        pde=pde,
        exp_resolved=None,
        distogram=dg,
    )


def test_sample_and_distogram_maps_match_direct_compute(tmp_path):
    out = _fake_output(n=10, s=2)
    logits_npz = tmp_path / "logits.npz"
    dg_npz = tmp_path / "seed-11_distogram.npz"

    assert conf.save_sample_logits(logits_npz, out, sample_idx=1, seed=11)
    assert conf.save_distogram_logits(dg_npz, out, seed=11)

    maps = conf.load_logit_maps(logits_npz=logits_npz, distogram_npz=dg_npz)
    assert set(maps) == {"pae", "pde", "distogram", "contact_probs"}

    pair_mask = torch.ones(10, 10)
    ref_pae = conf.compute_pae(out.pae, pair_mask)[1].numpy()
    ref_pde = conf.compute_pde(out.pde, pair_mask)[0][1].numpy()
    ref_dg = conf.compute_distogram_expected(out.distogram).numpy()
    ref_cp = conf.compute_contact_probs(out.distogram).numpy()

    # int8_softmax → tight but not exact on the expectation.
    np.testing.assert_allclose(maps["pae"], ref_pae, atol=0.15)
    np.testing.assert_allclose(maps["pde"], ref_pde, atol=0.15)
    np.testing.assert_allclose(maps["distogram"], ref_dg, atol=0.15)
    np.testing.assert_allclose(maps["contact_probs"], ref_cp, atol=1e-2)


def test_distogram_head_absent_returns_false(tmp_path):
    out = _fake_output()
    out.distogram = None
    out.pae = None
    out.pde = None
    assert conf.save_distogram_logits(tmp_path / "d.npz", out) is False
    assert conf.save_sample_logits(tmp_path / "l.npz", out) is False


def test_discover_logit_npz_json_layout(tmp_path):
    sample_dir = tmp_path / "seed-42_sample-0"
    sample_dir.mkdir()
    (sample_dir / "confidences.json").write_text("{}")
    (sample_dir / "logits.npz").write_bytes(b"x")
    (tmp_path / "seed-42_distogram.npz").write_bytes(b"x")

    logits_npz, dg_npz = conf.discover_logit_npz(sample_dir / "confidences.json")
    assert logits_npz == sample_dir / "logits.npz"
    assert dg_npz == tmp_path / "seed-42_distogram.npz"


def test_discover_logit_npz_msa_layout(tmp_path):
    (tmp_path / "pred_sample0_confidences.json").write_text("{}")
    (tmp_path / "pred_sample0_logits.npz").write_bytes(b"x")
    (tmp_path / "pred_distogram.npz").write_bytes(b"x")

    logits_npz, dg_npz = conf.discover_logit_npz(tmp_path / "pred_sample0_confidences.json")
    assert logits_npz == tmp_path / "pred_sample0_logits.npz"
    assert dg_npz == tmp_path / "pred_distogram.npz"


def test_resolve_sources_from_logits_npz(tmp_path):
    """A per-sample logits npz resolves its own confidences.json + distogram."""
    sample_dir = tmp_path / "seed-42_sample-0"
    sample_dir.mkdir()
    (sample_dir / "confidences.json").write_text("{}")
    (sample_dir / "logits.npz").write_bytes(b"x")
    (tmp_path / "seed-42_distogram.npz").write_bytes(b"x")

    src = conf.resolve_logit_sources(sample_dir / "logits.npz")
    assert src.conf_json == sample_dir / "confidences.json"
    assert src.logits_npz == sample_dir / "logits.npz"
    assert src.distogram_npz == tmp_path / "seed-42_distogram.npz"

    # The MSA layout resolves the same way from its npz.
    (tmp_path / "pred_sample0_logits.npz").write_bytes(b"x")
    (tmp_path / "pred_sample0_confidences.json").write_text("{}")
    (tmp_path / "pred_distogram.npz").write_bytes(b"x")
    src = conf.resolve_logit_sources(tmp_path / "pred_sample0_logits.npz")
    assert src.conf_json == tmp_path / "pred_sample0_confidences.json"
    assert src.distogram_npz == tmp_path / "pred_distogram.npz"


def test_resolve_sources_from_distogram_npz_and_dir(tmp_path):
    """A distogram npz stands alone; a sample directory resolves everything."""
    sample_dir = tmp_path / "seed-7_sample-1"
    sample_dir.mkdir()
    (sample_dir / "confidences.json").write_text("{}")
    (sample_dir / "logits.npz").write_bytes(b"x")
    (tmp_path / "seed-7_distogram.npz").write_bytes(b"x")

    src = conf.resolve_logit_sources(tmp_path / "seed-7_distogram.npz")
    assert (src.conf_json, src.logits_npz) == (None, None)
    assert src.distogram_npz == tmp_path / "seed-7_distogram.npz"

    src = conf.resolve_logit_sources(sample_dir)
    assert src.conf_json == sample_dir / "confidences.json"
    assert src.logits_npz == sample_dir / "logits.npz"
    assert src.distogram_npz == tmp_path / "seed-7_distogram.npz"


def test_plot_from_logits_npz_only(tmp_path):
    """`deepfold3 plot` renders straight from the npz, without any JSON."""
    pytest.importorskip("matplotlib")
    from deepfold3.scripts.plot_confidences import plot_confidences_file

    out = _fake_output(n=8, s=1, seed=31)
    conf.save_sample_logits(tmp_path / "pred_logits.npz", out, sample_idx=0, seed=31)
    conf.save_distogram_logits(tmp_path / "pred_distogram.npz", out, seed=31)

    png = tmp_path / "maps.png"
    drawn = plot_confidences_file(tmp_path / "pred_logits.npz", png)
    assert drawn == ["pae", "pde", "distogram"]
    assert png.exists()

    # Explicit panel selection, in the requested order.
    drawn = plot_confidences_file(tmp_path / "pred_logits.npz", png, panels="contact,pae")
    assert drawn == ["contact", "pae"]

    # Distogram alone → only the distogram-derived panel.
    assert plot_confidences_file(tmp_path / "pred_distogram.npz", png) == ["distogram"]

    # Nothing plottable → falsy return.
    empty = tmp_path / "empty_dir"
    empty.mkdir()
    assert plot_confidences_file(empty, png) == []


def test_load_logit_maps_from_json_roundtrip(tmp_path):
    out = _fake_output(n=8, s=1, seed=20)
    sample_dir = tmp_path / "seed-20_sample-0"
    sample_dir.mkdir()
    (sample_dir / "confidences.json").write_text("{}")
    conf.save_sample_logits(sample_dir / "logits.npz", out, sample_idx=0, seed=20)
    conf.save_distogram_logits(tmp_path / "seed-20_distogram.npz", out, seed=20)

    maps = conf.load_logit_maps_from_json(sample_dir / "confidences.json")
    assert set(maps) == {"pae", "pde", "distogram", "contact_probs"}
