"""Classical baseline on the synthetic dataset: leakage guards, caching, reproducibility."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from vqi.baseline.classifier import make_svm, select_svm_by_cv
from vqi.baseline.experiments import (
    ForbiddenSplitError,
    experiment_dir,
    extract_features,
    load_split_rows,
    run_all,
    run_experiment,
)
from vqi.baseline.features import FeatureExtractor
from vqi.config import Config
from vqi.data.download import raw_layout
from vqi.data.duplicates import analyse_duplicates, write_duplicates_report
from vqi.data.manifest import (
    build_manifest,
    check_expected_counts,
    read_manifest_meta,
    write_manifest,
)
from vqi.data.splits import make_splits, write_splits
from vqi.preprocessing import Preprocessor
from vqi.preprocessing.compare import run_comparison


@pytest.fixture
def prepared(synthetic_config: Config, synthetic_dataset: dict):
    """Manifest + duplicates + splits written to disk (what Phase 1 leaves behind)."""
    cfg = synthetic_config
    root = synthetic_dataset["root"]
    manifest, strays = build_manifest(cfg, raw_layout(cfg), root)
    manifest, report, nn = analyse_duplicates(manifest, cfg)
    write_manifest(manifest, cfg, check_expected_counts(manifest, cfg), strays=strays)
    write_duplicates_report(report, nn, cfg)
    splits = make_splits(manifest, cfg)
    write_splits(splits, manifest, cfg, manifest_sha256=read_manifest_meta(cfg)["manifest_sha256"])
    return cfg, root


def test_test_split_is_refused(prepared):
    cfg, _ = prepared
    with pytest.raises(ForbiddenSplitError):
        load_split_rows(cfg, "test")
    assert len(load_split_rows(cfg, "train")) > 0


def test_feature_cache_roundtrip(prepared):
    cfg, root = prepared
    pre = Preprocessor.from_config(cfg)
    fe = FeatureExtractor.from_config(cfg)
    first = extract_features(cfg, pre, fe, "val", root)
    second = extract_features(cfg, pre, fe, "val", root)
    assert not first.cached and second.cached
    assert np.array_equal(first.x, second.x) and first.names == second.names
    assert np.isfinite(first.x).all()
    assert list(first.image_ids) == load_split_rows(cfg, "val")["image_id"].tolist()


def test_scaler_is_fitted_on_train_only(prepared):
    cfg, root = prepared
    pre = Preprocessor.from_config(cfg)
    fe = FeatureExtractor.from_config(cfg)
    train = extract_features(cfg, pre, fe, "train", root)
    val = extract_features(cfg, pre, fe, "val", root)
    model = make_svm(1.0, "scale", "balanced").fit(train.x, train.labels)
    scaler = model.named_steps["scaler"]
    assert np.allclose(scaler.mean_, train.x.mean(axis=0))
    assert not np.allclose(scaler.mean_, np.vstack([train.x, val.x]).mean(axis=0))


def test_cv_selection_sees_only_train(prepared):
    cfg, root = prepared
    pre = Preprocessor.from_config(cfg)
    fe = FeatureExtractor.from_config(cfg)
    train = extract_features(cfg, pre, fe, "train", root)
    sel = select_svm_by_cv(train.x, train.labels, train.groups, cfg, "balanced", seed=42)
    assert sel.best_params == {"C": 1.0, "gamma": "scale", "kernel": "rbf"}
    assert len(sel.table) == 1 and "cv_average_precision_mean" in sel.table.columns


def test_run_all_writes_provenance_and_is_reproducible(prepared):
    cfg, root = prepared
    summary = run_all(cfg, root)
    assert list(summary["id"]) == ["T0_contrast", "T1_svm", "T2_rf"]
    assert summary["val_ap"].between(0, 1).all() and summary["val_roc_auc"].between(0, 1).all()

    result = json.loads((experiment_dir(cfg) / "T1_svm.json").read_text(encoding="utf-8"))
    assert result["data"]["evaluated_on"] == "val" and result["cv"]["on_split"] == "train"
    assert result["provenance"]["seed"] == 42 and result["provenance"]["config_hash"] == cfg.hash
    assert result["spec"]["n_features"] == len(fe_names(cfg, "combined"))

    val_ids = set(
        pd.read_csv(experiment_dir(cfg) / "T1_svm_val_scores.csv", dtype={"image_id": str})[
            "image_id"
        ]
    )
    assert val_ids == set(load_split_rows(cfg, "val")["image_id"])
    assert not val_ids & set(f"{20000 + i}" for i in range(20))  # no official-test image scored

    again = run_experiment(cfg.baseline.experiments[1], cfg, root)
    assert again["metrics"]["average_precision"] == result["metrics"]["average_precision"]


def test_compare_preprocessing_runs_on_train_only(prepared):
    cfg, root = prepared
    meta = run_comparison(cfg, root)
    train_ids = set(load_split_rows(cfg, "train")["image_id"])
    assert {r["image_id"] for r in meta["representatives"]} <= train_ids
    out = cfg.paths.reports_dir / cfg.baseline.report_subdir
    cnr = pd.read_csv(out / "cnr_per_image.csv", dtype={"image_id": str})
    assert set(cnr["image_id"]) <= train_ids
    assert set(cnr["variant"]) == {"none", "clahe", "denoise_light"}
    assert (out / "figures" / "contrast_zoom.png").exists()


def fe_names(cfg: Config, group: str) -> list[str]:
    fe = FeatureExtractor.from_config(cfg)
    return fe.names(cfg.baseline.feature_groups[group])
