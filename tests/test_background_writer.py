"""BackgroundWriter order and failure contract, and how ``inference`` uses it."""

from __future__ import annotations

import filecmp
import threading
import time

import pytest
import torch

from deepfold3.io.background_writer import BackgroundWriter, logits_to_host
from deepfold3.model.model import DeepFold3Output
from tests._util import run_tiny_inference


def _fail():
    raise OSError("disk full")


def test_writes_run_in_submission_order_and_block_exit():
    done = []

    def write(i):
        time.sleep(0.01)
        done.append(i)

    with BackgroundWriter(max_pending=2) as writer:
        for i in range(6):
            writer.submit(str(i), write, i)
    assert done == list(range(6))


def test_submit_blocks_at_max_pending():
    release = threading.Event()
    writer = BackgroundWriter(max_pending=1)
    writer.submit("slow", release.wait)
    second = threading.Thread(target=writer.submit, args=("next", lambda: None))
    second.start()
    second.join(0.1)
    assert second.is_alive()
    release.set()
    second.join(5)
    assert not second.is_alive()
    writer.drain()


def test_failed_write_raises_on_exit_and_stays_raised():
    writer = BackgroundWriter(max_pending=2)
    with pytest.raises(RuntimeError, match="a.npz: disk full"), writer:
        writer.submit("a.npz", _fail)
    with pytest.raises(RuntimeError, match="disk full"):
        writer.drain()


def test_failed_write_stops_next_submit_but_later_writes_still_run():
    ran = []
    with pytest.raises(RuntimeError, match="disk full"), BackgroundWriter(max_pending=4) as writer:
        writer.submit("a", _fail)
        writer.submit("b", ran.append, 1)  # may or may not see the failure yet
        writer.drain()
        writer.submit("c", ran.append, 2)
    assert 2 not in ran


def test_body_exception_wins_and_still_waits_for_writes():
    done = []

    def write():
        time.sleep(0.05)
        done.append(1)

    with pytest.raises(ValueError, match="oom"), BackgroundWriter(max_pending=2) as writer:
        writer.submit("a", write)
        writer.submit("b", _fail)
        raise ValueError("oom")
    assert done == [1]


def test_logits_to_host_keeps_values():
    n = 3
    out = DeepFold3Output(
        atom_positions=torch.zeros(1, n, 24, 3),
        atom_mask=torch.ones(n, 24),
        plddt=None,
        pae=torch.randn(1, n, n, 64),
        pde=None,
        exp_resolved=None,
        distogram=torch.randn(n, n, 64),
    )
    host = logits_to_host(out)
    assert host.pde is None
    assert torch.equal(host.pae, out.pae)
    assert torch.equal(host.distogram, out.distogram)
    assert host.atom_positions is out.atom_positions


@pytest.mark.slow
def test_inference_fails_when_a_logit_write_fails(tiny_job, tmp_path, monkeypatch):
    def broken(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("deepfold3.io.confidences.save_sample_logits", broken)
    result = run_tiny_inference(tiny_job, tmp_path / "out")
    assert result.exit_code != 0
    assert "disk full" in str(result.exception)
    assert not (tmp_path / "out" / "ranking_scores.csv").exists()


@pytest.mark.slow
def test_inference_outputs_match_a_synchronous_run(tiny_job, tmp_path, monkeypatch):
    """Same bytes as running each write inline, on the caller's thread."""
    assert run_tiny_inference(tiny_job, tmp_path / "bg").exit_code == 0

    def inline(self, label, fn, /, *args, **kwargs):
        fn(*args, **kwargs)

    monkeypatch.setattr(BackgroundWriter, "submit", inline)
    assert run_tiny_inference(tiny_job, tmp_path / "sync").exit_code == 0

    sync = tmp_path / "sync"
    files = sorted(p.relative_to(sync) for p in sync.rglob("*") if p.is_file() and p.name != "run_manifest.json")
    assert len(files) == 2 * (1 + 2 * 3) + 2
    for f in files:
        assert filecmp.cmp(tmp_path / "bg" / f, tmp_path / "sync" / f, shallow=False), f
