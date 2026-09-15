from pathlib import Path

import pytest
import yaml

from vqi.config import ConfigError, deep_merge, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT = REPO_ROOT / "configs" / "default.yaml"


def test_default_config_loads_and_is_typed():
    cfg = load_config(DEFAULT)
    assert cfg.reproducibility.seed == 42
    assert isinstance(cfg.paths.raw_dir, Path)
    assert isinstance(cfg.evaluation.report_recall_targets, tuple)
    assert cfg.split.duplicates.max_hamming_distance is None or isinstance(
        cfg.split.duplicates.max_hamming_distance, int
    )
    assert len(cfg.hash) == 12


def test_unknown_key_is_rejected(tmp_path: Path):
    override = tmp_path / "o.yaml"
    override.write_text(yaml.safe_dump({"training": {"batch_sizee": 8}}), encoding="utf-8")
    with pytest.raises(ConfigError, match="unknown keys"):
        load_config(DEFAULT, overrides=[override])


def test_wrong_type_is_rejected(tmp_path: Path):
    override = tmp_path / "o.yaml"
    override.write_text(yaml.safe_dump({"training": {"epochs": "twenty"}}), encoding="utf-8")
    with pytest.raises(ConfigError, match="expected an integer"):
        load_config(DEFAULT, overrides=[override])


def test_missing_section_is_rejected(tmp_path: Path):
    partial = tmp_path / "p.yaml"
    partial.write_text(yaml.safe_dump({"project": {"name": "x"}}), encoding="utf-8")
    with pytest.raises(ConfigError, match="missing"):
        load_config(partial)


def test_override_changes_value_and_hash(tmp_path: Path):
    override = tmp_path / "o.yaml"
    override.write_text(yaml.safe_dump({"training": {"batch_size": 8}}), encoding="utf-8")
    base, over = load_config(DEFAULT), load_config(DEFAULT, overrides=[override])
    assert over.training.batch_size == 8
    assert base.training.epochs == over.training.epochs
    assert base.hash != over.hash


def test_deep_merge_is_recursive_and_non_destructive():
    base = {"a": {"x": 1, "y": 2}, "b": 3}
    merged = deep_merge(base, {"a": {"y": 20}})
    assert merged == {"a": {"x": 1, "y": 20}, "b": 3}
    assert base["a"]["y"] == 2
