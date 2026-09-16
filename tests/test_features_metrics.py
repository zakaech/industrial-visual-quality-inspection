"""Feature extraction (shapes, finiteness, determinism, LBP properties) and evaluation metrics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from vqi.baseline.features import FeatureExtractor, lbp_histogram, local_stats, uniform_lut
from vqi.config import Config, load_config
from vqi.evaluation.metrics import (
    bootstrap_ci,
    confusion_at,
    evaluate_scores,
    ranking_metrics,
    recall_target_table,
    threshold_for_recall,
    threshold_sweep,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cfg() -> Config:
    return load_config(REPO_ROOT / "configs" / "default.yaml")


def _canvas(seed: int = 0, h: int = 128, w: int = 64) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(30, 60, size=(h, w, 3), dtype=np.uint8)


# ------------------------------------------------------------------------------ features


def test_feature_vector_matches_declared_names_and_is_finite(cfg: Config):
    fe = FeatureExtractor.from_config(cfg)
    names = fe.names(fe.all_extractors())
    vec = fe.extract(_canvas(), fe.all_extractors())
    assert vec.shape == (len(names),)
    assert np.isfinite(vec).all()
    for group, extractors in cfg.baseline.feature_groups.items():
        sub = fe.extract(_canvas(), extractors)
        assert sub.shape == (len(fe.names(extractors)),), group


def test_features_are_deterministic(cfg: Config):
    fe = FeatureExtractor.from_config(cfg)
    a = fe.extract(_canvas(3), fe.all_extractors())
    b = fe.extract(_canvas(3), fe.all_extractors())
    assert np.array_equal(a, b)


def test_uniform_lut_has_p_plus_2_bins():
    for p in (8, 16):
        lut = uniform_lut(p)
        assert lut.shape == (2**p,)
        assert set(np.unique(lut)) == set(range(p + 2))
        assert lut[0] == 0 and lut[2**p - 1] == p  # all-zero / all-one codes are uniform


def test_lbp_histogram_properties():
    const = np.full((64, 64), 100, dtype=np.uint8)
    h = lbp_histogram(const, 8, 1)
    assert abs(h.sum() - 1.0) < 1e-9 and h[8] == pytest.approx(1.0)  # single bin
    rng = np.random.default_rng(1)
    g = rng.integers(20, 80, size=(96, 48)).astype(np.uint8)
    shifted = np.clip(g.astype(int) + 40, 0, 255).astype(np.uint8)
    assert np.allclose(lbp_histogram(g, 8, 1), lbp_histogram(shifted, 8, 1))  # illumination
    assert lbp_histogram(g, 16, 2).shape == (18,)


def test_local_stats_react_to_a_local_anomaly():
    rng = np.random.default_rng(2)
    plain = rng.integers(40, 50, size=(256, 128)).astype(np.uint8)
    spotted = plain.copy()
    spotted[100:108, 60:68] = 255  # tiny bright defect
    a, b = local_stats(plain, 32), local_stats(spotted, 32)
    assert b[0] > a[0] * 2  # patch_std_max
    assert b[4] > a[4] * 2  # patch_lapvar_max
    assert abs(b[3] - a[3]) < abs(b[0] - a[0])  # the mean moves far less than the max


# ------------------------------------------------------------------------------ metrics


def _toy():
    y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 0])
    s = np.array([0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.8, 0.9, 0.95])
    return y, s


def test_ranking_metrics_and_confusion():
    y, s = _toy()
    m = ranking_metrics(y, s)
    assert m["average_precision"] == pytest.approx(average_precision_score(y, s))
    assert 0.0 < m["roc_auc"] < 1.0 and m["n_pos"] == 3
    c = confusion_at(y, s, 0.5)
    assert (c["tp"], c["fp"], c["fn"], c["tn"]) == (3, 1, 0, 6)
    assert c["recall"] == 1.0 and c["precision"] == pytest.approx(0.75)


def test_threshold_for_recall_is_the_least_permissive_reaching_target():
    y, s = _toy()
    assert threshold_for_recall(y, s, 1.0) == pytest.approx(0.5)
    assert threshold_for_recall(y, s, 0.66) == pytest.approx(0.8)
    tbl = recall_target_table(y, s, [0.5, 1.0])
    assert (tbl["recall"] >= tbl["recall_target"]).all()


def test_sweep_and_bootstrap_are_well_formed():
    y, s = _toy()
    sweep = threshold_sweep(y, s)
    assert len(sweep) == len(np.unique(s))
    assert sweep["recall"].is_monotonic_increasing  # thresholds descend → recall grows
    ci = bootstrap_ci(y, s, average_precision_score, n_resamples=200, seed=1)
    assert 0.0 <= ci["low"] <= ci["high"] <= 1.0 and ci["n_effective"] > 0
    assert bootstrap_ci(y, s, average_precision_score, 50, seed=1) == bootstrap_ci(
        y, s, average_precision_score, 50, seed=1
    )


def test_evaluate_scores_is_json_like_and_rejects_bad_input():
    y, s = _toy()
    out = evaluate_scores(y, s, [0.9, 1.0], 0.5, 100, 0.95, 42)
    assert set(out) >= {"average_precision", "roc_auc", "at_recall_targets", "curves"}
    with pytest.raises(ValueError):
        evaluate_scores(y, np.append(s[:-1], np.nan), [0.9], 0.5, 10, 0.95, 42)
