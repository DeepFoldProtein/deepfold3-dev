"""``inference-batch`` resume: a skipped seed still reaches the job's aggregates."""

from __future__ import annotations

import json

import click
import pytest

from deepfold3.scripts.inference_batch import _existing_summaries, _seed_outputs


def _write_seed(tmp_path, seed: int, num_samples: int, *, distogram: bool = False, drop: str | None = None):
    if distogram:
        (tmp_path / f"seed-{seed}_distogram.npz").write_bytes(b"")
    for s in range(num_samples):
        d = tmp_path / f"seed-{seed}_sample-{s}"
        d.mkdir(parents=True, exist_ok=True)
        if drop != "model":
            (d / "model.cif").write_text("data_x\n")
        if drop != "confidences":
            (d / "confidences.json").write_text(json.dumps({"summary": {"ranking_score": 0.5 + s, "ptm": 0.4}}))


def test_seed_outputs_lists_confidences_and_the_distogram(tmp_path):
    names = [p.name for p in _seed_outputs(tmp_path, 42, 2, True)]
    assert names == [
        "seed-42_distogram.npz",
        "model.cif",
        "confidences.json",
        "model.cif",
        "confidences.json",
    ]
    assert all(p.name != "seed-42_distogram.npz" for p in _seed_outputs(tmp_path, 42, 2, False))


@pytest.mark.parametrize("drop", ["confidences", "model"])
def test_a_seed_missing_one_file_is_not_complete(tmp_path, drop):
    """Only ``model.cif`` was checked before, so a run killed between the
    structure and its confidences counted as done with the rest lost."""
    _write_seed(tmp_path, 42, 2, drop=drop)
    assert not all(p.exists() for p in _seed_outputs(tmp_path, 42, 2, False))


def test_a_complete_seed_is_complete(tmp_path):
    _write_seed(tmp_path, 42, 2, distogram=True)
    assert all(p.exists() for p in _seed_outputs(tmp_path, 42, 2, True))


def test_skipped_seeds_are_read_back_for_the_aggregates(tmp_path):
    """Without this the resumed run's ranking_scores.csv holds only its own
    seeds, so a 4-seed sweep resumed at the last seed publishes 5 rows of 20."""
    _write_seed(tmp_path, 42, 3)
    rows = _existing_summaries(tmp_path, 42, 3)
    assert [r["sample"] for r in rows] == [0, 1, 2]
    assert {r["seed"] for r in rows} == {42}
    assert [r["ranking_score"] for r in rows] == [0.5, 1.5, 2.5]


def test_an_unusable_confidences_file_is_an_error(tmp_path):
    _write_seed(tmp_path, 42, 1)
    (tmp_path / "seed-42_sample-0" / "confidences.json").write_text("{}")
    with pytest.raises(click.ClickException, match="unusable"):
        _existing_summaries(tmp_path, 42, 1)
