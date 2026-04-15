"""Tests for config_loader: YAML/JSON loading, base resolution, deep merge, datasets."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from deepfold3.config_loader import deep_merge, load_config, normalize_datasets, resolve_config
from deepfold3.model.config import DatasetConfig

# ---------------------------------------------------------------------------
# deep_merge
# ---------------------------------------------------------------------------


class TestDeepMerge:
    def test_scalar_override(self):
        assert deep_merge({"a": 1}, {"a": 2}) == {"a": 2}

    def test_new_key(self):
        assert deep_merge({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}

    def test_nested_dict_merge(self):
        base = {"training": {"lr": 0.001, "seed": 42}}
        over = {"training": {"lr": 0.01}}
        result = deep_merge(base, over)
        assert result == {"training": {"lr": 0.01, "seed": 42}}

    def test_deeply_nested(self):
        base = {"model": {"optim": {"use_fused_evo_attention": True, "use_torch_compile": False}}}
        over = {"model": {"optim": {"use_torch_compile": True}}}
        result = deep_merge(base, over)
        assert result["model"]["optim"]["use_fused_evo_attention"] is True
        assert result["model"]["optim"]["use_torch_compile"] is True

    def test_list_replaces(self):
        assert deep_merge({"a": [1, 2, 3]}, {"a": [4, 5]}) == {"a": [4, 5]}

    def test_null_overrides(self):
        assert deep_merge({"a": 1.0}, {"a": None}) == {"a": None}

    def test_non_mutating(self):
        base = {"a": {"b": 1}}
        over = {"a": {"c": 2}}
        result = deep_merge(base, over)
        assert "c" not in base["a"]
        assert result["a"] == {"b": 1, "c": 2}


# ---------------------------------------------------------------------------
# resolve_config
# ---------------------------------------------------------------------------


class TestResolveConfig:
    def test_load_json(self, tmp_path: Path):
        cfg = {"training": {"lr": 0.01}}
        (tmp_path / "cfg.json").write_text(json.dumps(cfg))
        result = resolve_config(tmp_path / "cfg.json")
        assert result == cfg

    def test_load_yaml(self, tmp_path: Path):
        cfg = {"training": {"lr": 0.01}}
        (tmp_path / "cfg.yaml").write_text(yaml.dump(cfg))
        result = resolve_config(tmp_path / "cfg.yaml")
        assert result == cfg

    def test_load_yml(self, tmp_path: Path):
        cfg = {"training": {"lr": 0.01}}
        (tmp_path / "cfg.yml").write_text(yaml.dump(cfg))
        result = resolve_config(tmp_path / "cfg.yml")
        assert result == cfg

    def test_base_resolution_single(self, tmp_path: Path):
        base = {"training": {"lr": 0.001, "seed": 42}}
        (tmp_path / "base.yaml").write_text(yaml.dump(base))

        over = {"base": "base.yaml", "training": {"lr": 0.01}}
        (tmp_path / "child.yaml").write_text(yaml.dump(over))

        result = resolve_config(tmp_path / "child.yaml")
        assert result == {"training": {"lr": 0.01, "seed": 42}}

    def test_base_resolution_chain(self, tmp_path: Path):
        (tmp_path / "a.yaml").write_text(yaml.dump({"training": {"lr": 0.001, "seed": 1}}))
        (tmp_path / "b.yaml").write_text(yaml.dump({"base": "a.yaml", "training": {"lr": 0.01}}))
        (tmp_path / "c.yaml").write_text(yaml.dump({"base": "b.yaml", "training": {"seed": 99}}))

        result = resolve_config(tmp_path / "c.yaml")
        assert result["training"]["lr"] == 0.01  # from b
        assert result["training"]["seed"] == 99  # from c

    def test_cross_format(self, tmp_path: Path):
        base = {"training": {"lr": 0.001}}
        (tmp_path / "base.json").write_text(json.dumps(base))

        over = {"base": "base.json", "training": {"seed": 42}}
        (tmp_path / "child.yaml").write_text(yaml.dump(over))

        result = resolve_config(tmp_path / "child.yaml")
        assert result == {"training": {"lr": 0.001, "seed": 42}}

    def test_circular_reference(self, tmp_path: Path):
        (tmp_path / "a.yaml").write_text(yaml.dump({"base": "b.yaml"}))
        (tmp_path / "b.yaml").write_text(yaml.dump({"base": "a.yaml"}))

        with pytest.raises(ValueError, match="Circular"):
            resolve_config(tmp_path / "a.yaml")

    def test_missing_base(self, tmp_path: Path):
        (tmp_path / "child.yaml").write_text(yaml.dump({"base": "nonexistent.yaml"}))
        with pytest.raises(FileNotFoundError):
            resolve_config(tmp_path / "child.yaml")

    def test_missing_config(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            resolve_config(tmp_path / "nonexistent.yaml")

    def test_unsupported_extension(self, tmp_path: Path):
        (tmp_path / "cfg.toml").write_text("")
        with pytest.raises(ValueError, match="Unsupported"):
            resolve_config(tmp_path / "cfg.toml")

    def test_empty_yaml(self, tmp_path: Path):
        (tmp_path / "empty.yaml").write_text("")
        result = resolve_config(tmp_path / "empty.yaml")
        assert result == {}

    def test_base_key_stripped(self, tmp_path: Path):
        (tmp_path / "base.yaml").write_text(yaml.dump({"a": 1}))
        (tmp_path / "child.yaml").write_text(yaml.dump({"base": "base.yaml", "b": 2}))
        result = resolve_config(tmp_path / "child.yaml")
        assert "base" not in result

    def test_relative_subdirectory(self, tmp_path: Path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (tmp_path / "base.yaml").write_text(yaml.dump({"x": 1}))
        (sub / "child.yaml").write_text(yaml.dump({"base": "../base.yaml", "y": 2}))
        result = resolve_config(sub / "child.yaml")
        assert result == {"x": 1, "y": 2}


# ---------------------------------------------------------------------------
# normalize_datasets
# ---------------------------------------------------------------------------


class TestNormalizeDatasets:
    def test_full_entry(self):
        raw = [
            {
                "name": "afdb",
                "data_dir": "/data/afdb",
                "weight": 0.5,
                "strategy": "uniform",
                "is_distillation": True,
                "format": "of3",
            }
        ]
        result = normalize_datasets(raw)
        assert result == [
            {
                "name": "afdb",
                "data_dir": "/data/afdb",
                "weight": 0.5,
                "strategy": "uniform",
                "is_distillation": True,
                "format": "of3",
                "cutoff_date": "",
            }
        ]

    def test_defaults(self):
        raw = [{"name": "pdb2", "data_dir": "/data/pdb2"}]
        result = normalize_datasets(raw)
        assert result[0]["weight"] == 1.0
        assert result[0]["strategy"] == "uniform"
        assert result[0]["is_distillation"] is False
        assert result[0]["format"] == "df3"
        assert result[0]["cutoff_date"] == ""

    def test_missing_name(self):
        with pytest.raises(ValueError, match="name"):
            normalize_datasets([{"data_dir": "/data"}])

    def test_missing_data_dir(self):
        with pytest.raises(ValueError, match="data_dir"):
            normalize_datasets([{"name": "x"}])

    def test_not_a_dict(self):
        with pytest.raises(ValueError, match="mapping"):
            normalize_datasets(["not_a_dict"])

    def test_empty_list(self):
        assert normalize_datasets([]) == []


# ---------------------------------------------------------------------------
# load_config (integration)
# ---------------------------------------------------------------------------


class TestLoadConfig:
    def test_datasets_moved_to_training(self, tmp_path: Path):
        cfg = {
            "training": {"lr": 0.01},
            "datasets": [{"name": "afdb", "data_dir": "/data/afdb", "weight": 0.5}],
        }
        (tmp_path / "cfg.yaml").write_text(yaml.dump(cfg))
        result = load_config(tmp_path / "cfg.yaml")
        assert "datasets" not in result
        assert result["training"]["extra_datasets"] == [
            {
                "name": "afdb",
                "data_dir": "/data/afdb",
                "weight": 0.5,
                "strategy": "uniform",
                "is_distillation": False,
                "format": "df3",
                "cutoff_date": "",
            }
        ]

    def test_datasets_inherited_from_base(self, tmp_path: Path):
        base = {
            "training": {"lr": 0.001},
            "datasets": [{"name": "afdb", "data_dir": "/data/afdb"}],
        }
        (tmp_path / "base.yaml").write_text(yaml.dump(base))

        child = {"base": "base.yaml", "training": {"lr": 0.01}}
        (tmp_path / "child.yaml").write_text(yaml.dump(child))

        result = load_config(tmp_path / "child.yaml")
        assert result["training"]["lr"] == 0.01
        # datasets inherited from base
        assert len(result["training"]["extra_datasets"]) == 1
        assert result["training"]["extra_datasets"][0]["name"] == "afdb"

    def test_datasets_overridden_by_child(self, tmp_path: Path):
        base = {"datasets": [{"name": "afdb", "data_dir": "/data/afdb"}]}
        (tmp_path / "base.yaml").write_text(yaml.dump(base))

        child = {"base": "base.yaml", "datasets": [{"name": "pdb2", "data_dir": "/data/pdb2"}]}
        (tmp_path / "child.yaml").write_text(yaml.dump(child))

        result = load_config(tmp_path / "child.yaml")
        # child's datasets list replaces base's (list replacement, not merge)
        assert len(result["training"]["extra_datasets"]) == 1
        assert result["training"]["extra_datasets"][0]["name"] == "pdb2"

    def test_no_datasets(self, tmp_path: Path):
        cfg = {"training": {"lr": 0.01}}
        (tmp_path / "cfg.yaml").write_text(yaml.dump(cfg))
        result = load_config(tmp_path / "cfg.yaml")
        assert "extra_datasets" not in result.get("training", {})

    def test_existing_yaml_configs(self):
        """Verify existing YAML configs load without error."""
        configs_dir = Path(__file__).parent.parent / "configs"
        for yaml_file in configs_dir.glob("*.yaml"):
            result = load_config(yaml_file)
            assert isinstance(result, dict)
            assert "training" in result or "model" in result


# ---------------------------------------------------------------------------
# DatasetConfig
# ---------------------------------------------------------------------------


class TestDatasetConfig:
    def test_from_dict(self):
        d = {
            "name": "afdb",
            "data_dir": "/data/afdb",
            "weight": 0.5,
            "strategy": "uniform",
            "is_distillation": True,
            "format": "of3",
        }
        cfg = DatasetConfig(**d)
        assert cfg.name == "afdb"
        assert cfg.data_dir == "/data/afdb"
        assert cfg.weight == 0.5
        assert cfg.strategy == "uniform"
        assert cfg.is_distillation is True
        assert cfg.format == "of3"

    def test_defaults(self):
        cfg = DatasetConfig(name="test", data_dir="/data")
        assert cfg.weight == 1.0
        assert cfg.strategy == "uniform"
        assert cfg.is_distillation is False
        assert cfg.format == "df3"


# ---------------------------------------------------------------------------
# End-to-end: YAML with base + datasets
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_yaml_base_with_datasets(self, tmp_path: Path):
        base_content = """\
training:
  lr: 0.001
  accumulate_grad_batches: 32
  wandb_enabled: true
model:
  optim:
    use_fused_evo_attention: true
"""
        child_content = """\
base: base.yaml
training:
  lr: 0.01
  max_tokens: 640
datasets:
  - name: afdb
    data_dir: /data/afdb
    weight: 0.5
    is_distillation: true
"""
        (tmp_path / "base.yaml").write_text(base_content)
        (tmp_path / "child.yaml").write_text(child_content)

        result = load_config(tmp_path / "child.yaml")

        # Inherited from base
        assert result["training"]["accumulate_grad_batches"] == 32
        assert result["training"]["wandb_enabled"] is True
        assert result["model"]["optim"]["use_fused_evo_attention"] is True

        # Overridden by child
        assert result["training"]["lr"] == 0.01
        assert result["training"]["max_tokens"] == 640

        # Datasets normalised
        ds = result["training"]["extra_datasets"]
        assert len(ds) == 1
        assert ds[0]["name"] == "afdb"
        assert ds[0]["weight"] == 0.5
        assert ds[0]["is_distillation"] is True
