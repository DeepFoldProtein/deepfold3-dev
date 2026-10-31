"""Run manifest, the missing-key check and ``--det`` on the inference CLI."""

from __future__ import annotations

import json
from dataclasses import asdict

import click
import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from deepfold3.io.run_manifest import git_state, sha256_file, write_manifest
from deepfold3.model import guidance
from deepfold3.model.config import OptimConfig
from deepfold3.scripts import _common
from deepfold3.scripts._common import check_loaded_keys
from tests._util import run_tiny_inference


def test_check_loaded_keys_rejects_missing():
    with pytest.raises(click.ClickException, match="1 model keys are missing"):
        check_loaded_keys(["trunk.w"], [], allow_missing=False)


def test_check_loaded_keys_opt_out_returns_the_keys():
    assert check_loaded_keys(["trunk.w"], [], allow_missing=True) == ["trunk.w"]


def test_check_loaded_keys_ignores_legacy_markers_and_unexpected():
    legacy = f"block.{OptimConfig.FUSED_STATE_DICT_MARKERS[0]}weight"
    assert check_loaded_keys([legacy], ["extra"], allow_missing=False) == []


def test_write_manifest_replaces_and_leaves_no_temp(tmp_path):
    path = tmp_path / "run_manifest.json"
    write_manifest(path, {"status": "running"})
    write_manifest(path, {"status": "complete"})
    assert json.loads(path.read_text()) == {"status": "complete"}
    assert [p.name for p in tmp_path.iterdir()] == ["run_manifest.json"]


def test_git_state_outside_a_checkout(monkeypatch):
    monkeypatch.setattr("deepfold3.io.run_manifest._run", lambda *a: None)
    assert git_state() == {"commit": None, "dirty": None}


@pytest.mark.slow
def test_manifest_records_the_run(tiny_job, tmp_path):
    out = tmp_path / "out"
    assert run_tiny_inference(tiny_job, out, "--guidance", "1.5@A:B").exit_code == 0
    m = json.loads((out / "run_manifest.json").read_text())
    params = tiny_job[1]
    assert m["status"] == "complete"
    assert m["seeds"] == [1, 2]
    assert m["msa_crop_size"] == 16
    assert m["optim"] == "none"
    assert m["bucket"]["enabled"] is True
    assert m["bucket"]["num_tokens"] == 20
    assert m["guidance"][0]["scale"] == 1.5
    assert m["guidance"][0]["pairs"] == [["A", "B"]]
    assert m["checkpoint"]["size_bytes"] == params.stat().st_size
    assert m["checkpoint"]["sha256"] == sha256_file(params)
    assert m["gpu"] is None
    assert m["versions"]["torch"] == torch.__version__
    assert set(m["git"]) == {"commit", "dirty"}


@pytest.mark.slow
def test_manifest_marks_a_failed_run(tiny_job, tmp_path, monkeypatch):
    def broken(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("deepfold3.io.confidences.save_sample_logits", broken)
    out = tmp_path / "out"
    assert run_tiny_inference(tiny_job, out).exit_code != 0
    m = json.loads((out / "run_manifest.json").read_text())
    assert m["status"] == "failed"
    assert "disk full" in m["error"]


@pytest.fixture
def partial_params(tiny_job, tmp_path):
    """The tiny weights with one parameter removed."""
    state = load_file(tiny_job[1])
    with safe_open(tiny_job[1], framework="pt") as f:
        metadata = f.metadata()
    dropped = next(iter(state))
    del state[dropped]
    path = tmp_path / "partial.safetensors"
    save_file(state, path, metadata=metadata)
    return path, dropped


@pytest.mark.slow
def test_missing_key_stops_inference_unless_allowed(tiny_job, tmp_path, partial_params):
    path, dropped = partial_params
    refused = run_tiny_inference(tiny_job, tmp_path / "no", params=path)
    assert refused.exit_code != 0
    assert dropped in refused.output
    assert not (tmp_path / "no").exists()

    out = tmp_path / "yes"
    allowed = run_tiny_inference(tiny_job, out, "--allow-missing-keys", params=path)
    assert allowed.exit_code == 0
    assert json.loads((out / "run_manifest.json").read_text())["missing_keys"] == [dropped]


@pytest.fixture
def restore_det_state():
    """``--det`` sets process-wide torch flags; put them back."""
    from deepfold3.model.triton import cache

    algorithms = torch.are_deterministic_algorithms_enabled()
    cudnn = torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark
    yield
    torch.use_deterministic_algorithms(algorithms)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = cudnn
    cache.set_fixed_config(False)


@pytest.mark.slow
def test_det_gives_the_same_bytes_and_says_so(tiny_job, tmp_path, restore_det_state, monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    for name in ("a", "b"):
        assert run_tiny_inference(tiny_job, tmp_path / name, "--det").exit_code == 0
    assert _common.os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert torch.are_deterministic_algorithms_enabled()
    assert json.loads((tmp_path / "a" / "run_manifest.json").read_text())["det"] is True
    files = sorted(p.relative_to(tmp_path / "a") for p in (tmp_path / "a").rglob("*") if p.is_file())
    for f in files:
        if f.name != "run_manifest.json":
            assert (tmp_path / "a" / f).read_bytes() == (tmp_path / "b" / f).read_bytes(), f


def test_det_refuses_a_disabled_triton_cache(monkeypatch):
    monkeypatch.setenv("DEEPFOLD3_TRITON_NO_CACHE", "1")
    with pytest.raises(click.UsageError, match="DEEPFOLD3_TRITON_NO_CACHE"):
        _common.apply_det_recipe()


def test_guidance_spec_is_json_serialisable():
    json.dumps(asdict(guidance.parse_guidance_spec("2.0@A:B@1..160")))


def test_fixed_config_skips_tuning_and_persistence(tmp_path, monkeypatch):
    triton = pytest.importorskip("triton")
    from deepfold3.model.triton import cache

    class Tuned:
        """The part of ``triton.autotune``'s result that ``persistent_autotune`` touches."""

        arg_names = ("M",)
        keys = ("M",)
        best_config = None

        def __init__(self):
            self.cache = {}
            self.benchmarks = 0

        def run(self, *args, **kwargs):
            key = next(iter(self.cache), None)
            if key is None:
                self.benchmarks += 1
                return None
            return self.cache[key]

    tuned = Tuned()
    monkeypatch.setattr(triton, "autotune", lambda configs, key: lambda fn: tuned)
    mgr = cache.CacheManager.get()
    monkeypatch.setattr(mgr, "get_config", lambda *a: pytest.fail("--det must not read the cache"))
    monkeypatch.setattr(mgr, "set_config", lambda *a, **k: pytest.fail("--det must not write the cache"))
    cache.set_fixed_config(True)
    try:
        wrapped = cache.persistent_autotune(configs=["first", "second"], key=["M"], kernel_name="k")(lambda M: None)
        assert wrapped.run(8) == "first"
        assert wrapped.run(8) == "first"
    finally:
        cache.set_fixed_config(False)
    assert tuned.benchmarks == 0
