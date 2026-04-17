"""Tests for `serialize_run_config`: round-trip through `load_config`."""

from __future__ import annotations

import dataclasses
import io

import yaml

from deepfold3.config_loader import load_config
from deepfold3.model.config import DatasetConfig, ModelConfig, TrainingConfig
from deepfold3.scripts.train import _apply_overrides
from deepfold3.training.trainer import _yaml_safe, serialize_run_config


def _reload(dumped_yaml: str, tmp_path) -> dict:
    """Write the dumped YAML to a temp file and reload via the canonical loader."""
    path = tmp_path / "config.yaml"
    path.write_text(dumped_yaml)
    return load_config(path)


def test_round_trip_full(tmp_path):
    train_cfg = TrainingConfig(
        lr=3.14e-4,
        max_steps=12345,
        max_tokens=512,
        dtype="fp32",
        region_loss_weights={3: 5.0},
        extra_datasets=[
            DatasetConfig(
                name="patchr",
                data_dir="/data/x",
                weight=0.5,
                strategy="weighted",
                is_distillation=False,
                format="df3",
                cutoff_date="2021-09-30",
            )
        ],
        eval_datasets=[
            DatasetConfig(name="patchr_eval", data_dir="/data/x", format="df3"),
        ],
    )
    model_cfg = ModelConfig()
    model_cfg.evoformer.num_layer = 7
    model_cfg.precision.fp32_distogram_head = True
    model_cfg.data.max_templates = 2

    out = serialize_run_config(train_cfg, model_cfg)

    assert "training" in out and "model" in out
    assert "datasets" in out and len(out["datasets"]) == 1
    assert "eval_datasets" in out and len(out["eval_datasets"]) == 1
    # Datasets must be promoted out of the training block.
    assert "extra_datasets" not in out["training"]
    assert "eval_datasets" not in out["training"]

    dumped = yaml.safe_dump(out, default_flow_style=False, sort_keys=False)
    # safe_dump must not emit Python-specific tags (e.g. !!python/tuple).
    assert "!!python" not in dumped
    # safe_load round-trip must succeed.
    yaml.safe_load(io.StringIO(dumped))

    reloaded = _reload(dumped, tmp_path)

    new_train = TrainingConfig()
    new_model = ModelConfig()
    for key, val in reloaded.get("training", {}).items():
        if key == "extra_datasets":
            new_train.extra_datasets = [DatasetConfig(**d) for d in val]
        elif key == "eval_datasets":
            new_train.eval_datasets = [DatasetConfig(**d) for d in val]
        elif hasattr(new_train, key):
            setattr(new_train, key, val)
    _apply_overrides(new_model, reloaded.get("model", {}))

    # Tuples become lists through YAML; normalise both sides via asdict→yaml-safe.
    assert _yaml_safe(dataclasses.asdict(new_train)) == _yaml_safe(dataclasses.asdict(train_cfg))
    assert _yaml_safe(dataclasses.asdict(new_model)) == _yaml_safe(dataclasses.asdict(model_cfg))


def test_empty_datasets_not_emitted():
    train_cfg = TrainingConfig()  # no extras / evals
    model_cfg = ModelConfig()
    out = serialize_run_config(train_cfg, model_cfg)
    assert "datasets" not in out
    assert "eval_datasets" not in out
    # And the now-empty fields must not leak inside training either.
    assert "extra_datasets" not in out["training"]
    assert "eval_datasets" not in out["training"]


def test_safe_dump_compatible():
    """The dumped YAML must contain no Python-specific tags."""
    train_cfg = TrainingConfig()
    model_cfg = ModelConfig()
    out = serialize_run_config(train_cfg, model_cfg)
    dumped = yaml.safe_dump(out, default_flow_style=False, sort_keys=False)
    assert "!!python" not in dumped
    yaml.safe_load(io.StringIO(dumped))  # must parse
