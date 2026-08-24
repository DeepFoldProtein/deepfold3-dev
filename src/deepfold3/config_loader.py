"""Config file loading with YAML/JSON support and base-config inheritance.

Supports:
- JSON (``.json``) and YAML (``.yaml`` / ``.yml``) config files
- ``base:`` key for config inheritance (recursive, with cycle detection)
- Deep-merge semantics: dicts merge recursively, scalars/lists replace
- Top-level ``datasets:`` key normalised into ``training.extra_datasets``
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base*, returning a new dict.

    - Dicts: merged recursively (override keys win on conflict).
    - Lists / scalars / ``None``: override replaces base entirely.
    """
    result = base.copy()
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def _load_file(path: Path) -> dict:
    """Load a JSON or YAML file, returning a dict (empty dict for empty files)."""
    suffix = path.suffix.lower()
    with open(path) as f:
        if suffix == ".json":
            return json.load(f)
        elif suffix in (".yaml", ".yml"):
            return yaml.safe_load(f) or {}
        else:
            raise ValueError(f"Unsupported config file extension: {suffix} (expected .json, .yaml, or .yml)")


def resolve_config(config_path: str | Path, _seen: set[Path] | None = None) -> dict:
    """Load a config file and recursively resolve ``base:`` inheritance.

    Args:
        config_path: Path to a JSON or YAML config file.
        _seen: Internal set of already-visited paths for cycle detection.

    Returns:
        Fully-merged config dict with ``base`` key stripped.

    Raises:
        FileNotFoundError: If the config file or a referenced base does not exist.
        ValueError: If a circular base reference is detected.
    """
    path = Path(config_path).resolve()
    if _seen is None:
        _seen = set()
    if path in _seen:
        chain = " -> ".join(str(p) for p in _seen) + f" -> {path}"
        raise ValueError(f"Circular config base reference: {chain}")
    _seen.add(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    data = _load_file(path)

    base_ref = data.pop("base", None)
    if base_ref is not None:
        base_path = Path(base_ref)
        if not base_path.is_absolute():
            base_path = path.parent / base_path
        base_data = resolve_config(base_path, _seen)
        data = deep_merge(base_data, data)

    return data


def normalize_datasets(raw: list[dict]) -> list[dict]:
    """Validate and fill defaults for dataset entries.

    Args:
        raw: List of dataset dicts from YAML/JSON config.

    Returns:
        Normalised list of dicts ready for ``DatasetConfig`` construction.

    Raises:
        ValueError: If a required field (``name``, ``data_dir``) is missing.
    """
    valid_strategies = {"uniform", "weighted"}
    valid_formats = {"df3", "of3", "of3pdb"}
    result = []
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ValueError(f"datasets[{i}]: expected a mapping, got {type(entry).__name__}")
        for required in ("name", "data_dir"):
            if required not in entry:
                raise ValueError(f"datasets[{i}]: missing required field '{required}'")
        strategy = entry.get("strategy", "uniform")
        fmt = entry.get("format", "df3")
        if strategy not in valid_strategies:
            raise ValueError(f"datasets[{i}]: invalid strategy '{strategy}', must be one of {valid_strategies}")
        if fmt not in valid_formats:
            raise ValueError(f"datasets[{i}]: invalid format '{fmt}', must be one of {valid_formats}")
        excl_raw = entry.get("exclude_loss_mol_types", []) or []
        if not isinstance(excl_raw, list):
            raise ValueError(f"datasets[{i}]: exclude_loss_mol_types must be a list of integers")
        try:
            excl = [int(x) for x in excl_raw]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"datasets[{i}]: exclude_loss_mol_types must contain integers ({exc})") from exc
        cw_raw = entry.get("crop_weights")
        if cw_raw is None:
            crop_weights = None
        elif not isinstance(cw_raw, dict):
            raise ValueError(f"datasets[{i}]: crop_weights must be a mapping, got {type(cw_raw).__name__}")
        else:
            valid_cw_keys = {"contiguous", "spatial", "interface"}
            unknown = set(cw_raw) - valid_cw_keys
            if unknown:
                raise ValueError(f"datasets[{i}]: unknown crop_weights keys {sorted(unknown)} (use {valid_cw_keys})")
            try:
                crop_weights = {k: float(v) for k, v in cw_raw.items()}
            except (TypeError, ValueError) as exc:
                raise ValueError(f"datasets[{i}]: crop_weights values must be numbers ({exc})") from exc
        result.append(
            {
                "name": entry["name"],
                "data_dir": entry["data_dir"],
                "weight": float(entry.get("weight", 1.0)),
                "strategy": strategy,
                "is_distillation": bool(entry.get("is_distillation", False)),
                "format": fmt,
                "cutoff_date": str(entry.get("cutoff_date", "")),
                "exclude_loss_mol_types": excl,
                "crop_weights": crop_weights,
                # ``monomer_template_struct_space`` is the deprecated spelling; it applied
                # only to the monomer path, but the flag is a property of the index files
                # and both paths need it.
                "template_struct_space": bool(
                    entry.get("template_struct_space", entry.get("monomer_template_struct_space", False))
                ),
                "max_template_date": str(entry.get("max_template_date", "")),
                # ``format: of3pdb`` only: where `deepfold3 build-of3-cache` wrote
                # the columnar dataset cache. Empty = ``{data_dir}/df3_cache``.
                "of3_cache_dir": str(entry.get("of3_cache_dir", "")),
                # ``format: of3pdb`` only: per-entry multiplier on the AF3 cluster
                # marginal (see DatasetConfig.curator_weights). This dict is rebuilt
                # field-by-field, so a key omitted here is silently dropped — the same
                # failure mode ``crop_weights`` had (c0e5e55).
                "curator_weights": str(entry.get("curator_weights", "")),
            }
        )
    return result


def load_config(config_path: str | Path) -> dict:
    """Load a config file (JSON or YAML) with base resolution and dataset normalisation.

    This is the main entry point for config loading.  It:

    1. Resolves ``base:`` inheritance chains via :func:`resolve_config`.
    2. Moves a top-level ``datasets:`` key into ``training.extra_datasets``.
    3. Moves a top-level ``eval_datasets:`` key into ``training.eval_datasets``.
    4. Returns a clean dict ready for application to ``TrainingConfig`` /
       ``ModelConfig`` via ``_apply_overrides``.
    """
    data = resolve_config(config_path)

    datasets_raw = data.pop("datasets", None)
    if datasets_raw is not None:
        normalised = normalize_datasets(datasets_raw)
        training = data.setdefault("training", {})
        # Config-defined datasets: set (not append) — base-inherited datasets
        # are already merged by deep_merge, so the final list is authoritative.
        training["extra_datasets"] = normalised

    eval_datasets_raw = data.pop("eval_datasets", None)
    if eval_datasets_raw is not None:
        normalised = normalize_datasets(eval_datasets_raw)
        training = data.setdefault("training", {})
        training["eval_datasets"] = normalised

    return data
