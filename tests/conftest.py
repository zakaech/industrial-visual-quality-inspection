"""Shared fixtures: a tiny synthetic dataset laid out like KolektorSDD2, plus a config pointing
at it. No real data is needed to run the test suite."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from vqi.config import Config, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


def _write_image(
    path: Path, rng: np.random.Generator, h: int = 64, w: int = 24, base: np.ndarray | None = None
) -> np.ndarray:
    img = base if base is not None else rng.integers(30, 60, size=(h, w, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)
    return img


def _write_mask(path: Path, h: int, w: int, defect: bool) -> int:
    mask = np.zeros((h, w), dtype=np.uint8)
    if defect:
        mask[10:14, 5:9] = 255
    cv2.imwrite(str(path), mask)
    return int((mask > 0).sum())


@pytest.fixture
def synthetic_dataset(tmp_path: Path) -> dict:
    """A miniature raw tree: 40 train images (8 defective) + 20 test images (4 defective),
    one exact duplicate inside train, one train image identical to a test image, one
    nonconforming '(copy)' file and its stray mask."""
    rng = np.random.default_rng(0)
    raw = tmp_path / "data" / "raw" / "kolektor_sdd2"
    train, test = raw / "train", raw / "test"
    train.mkdir(parents=True)
    test.mkdir(parents=True)
    h, w = 64, 24
    info = {"train_ids": [], "test_ids": [], "defective": set()}

    for i in range(40):
        img_id = f"{10000 + i}"
        img = _write_image(train / f"{img_id}.png", rng, h, w)
        defect = i % 5 == 0
        _write_mask(train / f"{img_id}_GT.png", h, w, defect)
        info["train_ids"].append(img_id)
        if defect:
            info["defective"].add(img_id)
        if i == 3:
            first_img = img
    for i in range(20):
        img_id = f"{20000 + i}"
        img = _write_image(test / f"{img_id}.png", rng, h, w)
        defect = i % 5 == 0
        _write_mask(test / f"{img_id}_GT.png", h, w, defect)
        info["test_ids"].append(img_id)
        if defect:
            info["defective"].add(img_id)
        if i == 7:
            test_img = img

    # exact duplicate within train: 10039 becomes a copy of 10003
    _write_image(train / "10039.png", rng, h, w, base=first_img)
    # train image identical to a test image: 10038 := copy of 20007
    _write_image(train / "10038.png", rng, h, w, base=test_img)
    # nonconforming stray copy
    _write_image(train / "10003 (copy).png", rng, h, w, base=first_img)
    cv2.imwrite(str(train / "10003_GT (copy).png"), np.zeros((h, w), dtype=np.uint8))

    info["raw"] = raw
    info["root"] = tmp_path
    return info


@pytest.fixture
def synthetic_config(synthetic_dataset: dict, tmp_path: Path) -> Config:
    """The default config, rebased on the temporary directory and with counts matching the
    synthetic dataset. Everything else is inherited from configs/default.yaml."""
    base = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    override = {
        "paths": {k: str(tmp_path / v) for k, v in base["paths"].items()},
        "dataset": {
            "archive_sha256": "TBD",
            "expected_counts": {
                "train_total": 40,
                "train_defective": 8,
                "test_total": 20,
                "test_defective": 4,
            },
        },
        "logging": {"file": None, "format": "text"},
        "eda": {"contact_sheet": {"n_per_class": 3}, "n_closest_pairs": 4},
        "evaluation": {"bootstrap": {"n_resamples": 50}},
        "training": {
            "num_workers": 0,
            "batch_size": 8,
            "stages": {
                "A": {
                    "trainable": ["head"],
                    "lr": {"head": 1.0e-3},
                    "max_epochs": 3,
                    "patience": 2,
                    "augment": False,
                    "cached_features": True,
                },
                "B": {
                    "trainable": ["layer4", "head"],
                    "lr": {"layer4": 1.0e-4, "head": 1.0e-3},
                    "max_epochs": 1,
                    "patience": 1,
                    "augment": True,
                    "cached_features": False,
                },
            },
            "cache": {"dir": str(tmp_path / "canvases")},
            "dry_run": {
                "batch_sizes": [4],
                "num_workers_options": [0],
                "n_loader_batches": 2,
                "warmup_steps": 1,
                "n_steps": 1,
                "n_cache_check": 3,
                "n_saturation_check": 4,
                "determinism_steps": 2,
            },
        },
        "phase3": {
            "reference_experiment": "T1_svm",
            "paired_bootstrap": {"n_resamples": 30},
            "experiments": [
                {"id": "P1_probe", "stages": ["A"], "pos_weight": 1.0, "sampler": "none"},
                {
                    "id": "P2_B",
                    "stages": ["A", "B"],
                    "pos_weight": "sqrt_ratio",
                    "sampler": "weighted_random",
                },
            ],
        },
        "baseline": {
            "features_cache_dir": str(tmp_path / "cache"),
            "svm": {"c_grid": [1.0], "gamma_grid": ["scale"], "cv_folds": 2},
            "random_forest": {"n_estimators": 20},
            "representative_images": {"n_healthy": 1},
            "cnr": {"variants": ["none", "clahe", "denoise_light"]},
            "experiments": [
                {
                    "id": "T0_contrast",
                    "features": "contrast_only",
                    "preprocessing": "none",
                    "classifier": "none",
                    "class_weight": "none",
                },
                {
                    "id": "T1_svm",
                    "features": "combined",
                    "preprocessing": "none",
                    "classifier": "svm",
                    "class_weight": "balanced",
                },
                {
                    "id": "T2_rf",
                    "features": "local_stats",
                    "preprocessing": "clahe",
                    "classifier": "rf",
                    "class_weight": "balanced",
                },
            ],
        },
    }
    override_path = tmp_path / "override.yaml"
    override_path.write_text(yaml.safe_dump(override), encoding="utf-8")
    return load_config(DEFAULT_CONFIG, overrides=[override_path])
