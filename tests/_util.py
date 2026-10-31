"""Shared test helpers."""

from __future__ import annotations

import torch


def wake_zero_init(module, std: float = 0.02, seed: int = 0):
    """Give a module's zero-initialized weights small values, in place.

    Every AF3-style module zero-initializes its output projection, so a freshly
    constructed one emits exactly zero. A test of the form

        assert_close(module(x), hand_written_reference(x))

    then compares zero with zero and passes whatever the module did in between:
    the reference body — interleaved channel split, einsum direction, LayerNorm
    placement — is never exercised. A missing ``head_norm == "channel"`` branch
    in MHR-TriMul survived a test named ``..._is_exactly_af3`` exactly this way.

    Call this after constructing a module whose output is about to be compared
    against a reference. Run ``pytest --check-vacuous`` to find comparisons that
    still need it. Returns the module, for chaining.
    """
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in module.parameters():
            if p.dim() >= 1 and not bool(p.any()):
                p.copy_(torch.empty_like(p).normal_(0.0, std, generator=gen))
    return module


def run_tiny_inference(tiny_job, output, *extra, params=None):
    """Run ``deepfold3 inference`` on CPU on the ``tiny_job`` fixture; returns the CliRunner result."""
    from click.testing import CliRunner

    from deepfold3.scripts.inference import inference

    in_json, tiny_params = tiny_job
    # fmt: off
    args = [
        "--json", str(in_json), "--params", str(params or tiny_params), "--output", str(output), "--device", "cpu",
        "--optim", "none", "--num-recycles", "1", "--num-samples", "2", "--msa-crop-size", "16",
    ]
    # fmt: on
    return CliRunner().invoke(inference, [*args, *extra])
