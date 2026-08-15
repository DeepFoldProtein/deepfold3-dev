"""Shared test fixtures for DeepFold3 validation.

Usage:
    uv run pytest                          # CPU (default)
    uv run pytest --device cuda            # GPU
    uv run pytest --device cuda:1          # specific GPU
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

AF3_PARAMS_PATH = Path(__file__).parent.parent / "resources" / "af3.bin.zst"


def pytest_addoption(parser):
    parser.addoption(
        "--check-vacuous",
        action="store_true",
        help="report tensor comparisons where both sides are all zero (see wake_zero_init)",
    )
    parser.addoption(
        "--device",
        default="cpu",
        help="Torch device for tests (cpu, cuda, cuda:0, etc.)",
    )


@pytest.fixture(scope="session")
def device(request):
    """Torch device from --device CLI option. Skips if device unavailable."""
    name = request.config.getoption("--device")
    if name.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device(name)


def _features_to_device(feats: dict, dev: torch.device) -> dict:
    """Move all tensors in a feature dict to the given device."""
    return {k: v.to(dev) if isinstance(v, torch.Tensor) else v for k, v in feats.items()}


@pytest.fixture(scope="session")
def af3_raw_params():
    """Load raw AF3 Haiku parameters as nested dict[scope][name] -> ndarray."""
    if not AF3_PARAMS_PATH.exists():
        pytest.skip(f"AF3 params not found: {AF3_PARAMS_PATH}")
    from deepfold3.params import read_af3_params

    return read_af3_params(str(AF3_PARAMS_PATH))


@pytest.fixture(scope="session")
def af3_state_dict(af3_raw_params):
    """Converted PyTorch state_dict from AF3 params."""
    from deepfold3.convert.af3 import haiku_to_torch

    return dict(haiku_to_torch(af3_raw_params))


@pytest.fixture(scope="session")
def model(af3_state_dict, device):
    """DeepFold3 model loaded with AF3 weights, eval mode, on --device."""
    from deepfold3.model.model import DeepFold3

    m = DeepFold3()
    missing, unexpected = m.load_state_dict(af3_state_dict, strict=False)
    assert len(unexpected) == 0, f"Unexpected keys: {unexpected[:10]}"
    m.eval()
    m.to(device)
    return m


@pytest.fixture
def small_features(device):
    """Feature dict for a small 8-residue single-chain protein."""
    N, M, S = 8, 24, 4
    torch.manual_seed(42)

    ref_mask = torch.zeros(N, M, dtype=torch.bool)
    ref_mask[:, :5] = True  # 5 heavy atoms per residue

    feats = {
        "res_type": torch.randint(0, 20, (N,)),
        "residue_index": torch.arange(N),
        "token_index": torch.arange(N),
        "asym_id": torch.zeros(N, dtype=torch.long),
        "entity_id": torch.ones(N, dtype=torch.long),
        "sym_id": torch.ones(N, dtype=torch.long),
        "seq_mask": torch.ones(N, dtype=torch.bool),
        "pair_mask": torch.ones(N, N, dtype=torch.bool),
        "ref_pos": torch.randn(N, M, 3) * ref_mask.unsqueeze(-1).float(),
        "ref_mask": ref_mask,
        "ref_element": (torch.randint(1, 8, (N, M)) * ref_mask.long()),
        "ref_charge": torch.zeros(N, M),
        "ref_atom_name_chars": torch.zeros(N, M, 4, dtype=torch.long),
        "msa": torch.randint(0, 22, (S, N)),
        "msa_mask": torch.ones(S, N, dtype=torch.bool),
        "deletion_matrix": torch.zeros(S, N, dtype=torch.long),
        "profile": torch.zeros(N, 31),
        "deletion_mean": torch.zeros(N),
    }
    return _features_to_device(feats, device)


@pytest.fixture
def medium_features(device):
    """Feature dict for a 32-residue protein (heavier tests)."""
    N, M, S = 32, 24, 16
    torch.manual_seed(123)

    ref_mask = torch.zeros(N, M, dtype=torch.bool)
    ref_mask[:, :5] = True

    # Make a more realistic profile from the MSA
    msa = torch.randint(0, 22, (S, N))
    profile = torch.zeros(N, 31)
    for i in range(S):
        for j in range(N):
            profile[j, msa[i, j].item()] += 1.0
    profile = profile / S

    feats = {
        "res_type": torch.randint(0, 20, (N,)),
        "residue_index": torch.arange(N),
        "token_index": torch.arange(N),
        "asym_id": torch.zeros(N, dtype=torch.long),
        "entity_id": torch.ones(N, dtype=torch.long),
        "sym_id": torch.ones(N, dtype=torch.long),
        "seq_mask": torch.ones(N, dtype=torch.bool),
        "pair_mask": torch.ones(N, N, dtype=torch.bool),
        "ref_pos": torch.randn(N, M, 3) * ref_mask.unsqueeze(-1).float(),
        "ref_mask": ref_mask,
        "ref_element": (torch.randint(1, 8, (N, M)) * ref_mask.long()),
        "ref_charge": torch.zeros(N, M),
        "ref_atom_name_chars": torch.zeros(N, M, 4, dtype=torch.long),
        "msa": msa,
        "msa_mask": torch.ones(S, N, dtype=torch.bool),
        "deletion_matrix": torch.zeros(S, N, dtype=torch.long),
        "profile": profile,
        "deletion_mean": torch.zeros(N),
    }
    return _features_to_device(feats, device)


@pytest.fixture
def rng_generator():
    """Deterministic PyTorch Generator."""
    return torch.Generator().manual_seed(42)


@pytest.fixture
def np_rng():
    """Deterministic numpy random generator."""
    return np.random.default_rng(42)


# ── Zero-init output paths make comparison tests vacuous ────────────────────
#
# Every AF3-style module zero-initializes its output projection, so a freshly
# constructed one emits exactly zero. A test of the form
#
#     assert_close(module(x), hand_written_reference(x))
#
# then compares zero with zero and passes whatever the module did in between —
# the reference body (interleaved channel split, einsum direction, LayerNorm
# placement) is never exercised. A missing ``head_norm == "channel"`` branch in
# MHR-TriMul survived a test named ``..._is_exactly_af3`` exactly this way.


@pytest.fixture
def wake():
    """Fixture form of :func:`tests._util.wake_zero_init`."""
    from tests._util import wake_zero_init

    return wake_zero_init


# ── Opt-in scan for vacuous comparisons: pytest --check-vacuous ─────────────
_VACUOUS = {"hits": None, "total": None, "id": "?"}


def _install_vacuous_scan():
    import collections

    import torch as _torch

    _VACUOUS["hits"] = collections.Counter()
    _VACUOUS["total"] = collections.Counter()
    originals = {
        "equal": _torch.equal,
        "allclose": _torch.allclose,
        "assert_close": getattr(_torch.testing, "assert_close", None),
    }

    def zeroish(x):
        return _torch.is_tensor(x) and x.numel() > 1 and not bool(x.any())

    def wrap(name, fn):
        def inner(a, b, *args, **kwargs):
            _VACUOUS["total"][name] += 1
            if zeroish(a) and zeroish(b):
                _VACUOUS["hits"][f"{_VACUOUS['id']}  [{name}]  shape={tuple(a.shape)}"] += 1
            return fn(a, b, *args, **kwargs)

        return inner

    _torch.equal = wrap("equal", originals["equal"])
    _torch.allclose = wrap("allclose", originals["allclose"])
    if originals["assert_close"] is not None:
        _torch.testing.assert_close = wrap("assert_close", originals["assert_close"])


def pytest_configure(config):
    if config.getoption("--check-vacuous"):
        _install_vacuous_scan()


def pytest_runtest_setup(item):
    if _VACUOUS["hits"] is not None:
        _VACUOUS["id"] = item.nodeid


def pytest_sessionfinish(session, exitstatus):
    hits = _VACUOUS["hits"]
    if hits is None:
        return
    report = [f"comparisons intercepted: {dict(_VACUOUS['total'])}", f"{len(hits)} vacuous (both sides all zero):"]
    report += [f"  x{n:<4d} {k}" for k, n in hits.most_common()]
    text = "\n".join(report)
    out = Path(session.config.rootpath) / "tmp" / "vacuous_hits.txt"
    out.parent.mkdir(exist_ok=True)
    out.write_text(text + "\n")
    print("\n" + text)
