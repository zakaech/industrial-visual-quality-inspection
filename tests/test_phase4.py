"""Phase 4: calibration diagnostics, threshold policy, Grad-CAM, frozen bundle, single test pass."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from vqi.baseline.experiments import run_all as run_baseline
from vqi.config import Config, load_config
from vqi.data.download import raw_layout
from vqi.data.duplicates import analyse_duplicates, write_duplicates_report
from vqi.data.manifest import (
    build_manifest,
    check_expected_counts,
    read_manifest_meta,
    write_manifest,
)
from vqi.data.splits import make_splits, write_splits
from vqi.evaluation.calibration import (
    TemperatureScaling,
    calibration_metrics,
    cross_validated_calibration,
    expected_calibration_error,
)
from vqi.evaluation.gradcam import GradCAM, cam_mask_overlap
from vqi.evaluation.threshold import bootstrap_at_fixed_threshold, candidate_table, select_threshold
from vqi.inference.predictor import BundleError, Predictor
from vqi.models.factory import build_model
from vqi.training.experiments import run_experiment

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cfg() -> Config:
    return load_config(REPO_ROOT / "configs" / "default.yaml")


# ------------------------------------------------------------------------------ calibration


def test_calibration_metrics_on_perfect_and_biased_scores():
    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.3).astype(int)
    perfect = np.where(y == 1, 0.9, 0.1) + rng.normal(0, 0.01, 2000)
    perfect = np.clip(perfect, 0, 1)
    m = calibration_metrics(y, perfect, 10)
    assert m["brier"] < 0.02
    biased = np.clip(perfect * 0.5, 0, 1)  # systematically under-confident
    assert expected_calibration_error(y, biased, 10, adaptive=True) > expected_calibration_error(
        y, perfect, 10, adaptive=True
    )


def test_temperature_scaling_recovers_a_known_scale():
    rng = np.random.default_rng(1)
    z = rng.normal(0, 2, 5000)
    y = (rng.random(5000) < 1 / (1 + np.exp(-z))).astype(int)
    p_over = 1 / (1 + np.exp(-3 * z))  # over-confident by a factor 3
    cal = TemperatureScaling().fit(y, p_over)
    assert 2.5 < cal.temperature < 3.5
    cv = cross_validated_calibration(y, p_over, "temperature", 10, seed=0)
    assert cv["out_of_fold"]["nll"] < calibration_metrics(y, p_over, 10)["nll"]


# ------------------------------------------------------------------------------ threshold


def test_threshold_policy_and_margin():
    y = np.array([1] * 10 + [0] * 90)
    p = np.concatenate([np.linspace(0.9, 0.2, 10), np.linspace(0.19, 0.01, 90)])
    p[9] = 0.05  # the weakest positive sits among negatives
    t = select_threshold(y, p, 0.9, "none")
    assert t["threshold"] == pytest.approx(np.sort(p[y == 1])[::-1][8])  # 9 of 10 kept
    assert t["at_threshold"]["recall"] >= 0.9
    tm = select_threshold(y, p, 0.9, "midpoint_logit")
    assert tm["next_lower_positive_score"] == pytest.approx(0.05)
    assert 0.05 < tm["threshold"] < tm["threshold_min_recall"]
    assert tm["at_threshold"]["recall"] == tm["at_threshold_min_recall"]["recall"]
    assert tm["margin_cost_extra_fp"] >= 0
    table = candidate_table(y, p, (5.0, 10.0))
    assert len(table) == 10 and "expected_cost_ratio_10" in table.columns
    ci = bootstrap_at_fixed_threshold(y, p, tm["threshold"], 100, 0.95, 1)
    assert ci["recall"]["low"] <= tm["at_threshold"]["recall"] <= ci["recall"]["high"]


# ------------------------------------------------------------------------------ grad-cam


def test_gradcam_shape_range_and_overlap(cfg: Config):
    model = build_model(cfg, 1, 1)
    torch.manual_seed(0)
    torch.nn.init.normal_(model.head[1].weight, std=0.05)  # zero head weights give a null CAM
    cam = GradCAM(model, "layer4")
    x = torch.rand(2, 3, 640, 224)
    maps, probs = cam(x)
    cam.close()
    assert maps.shape == (2, 640, 224) and probs.shape == (2,)
    assert maps.min() >= 0.0 and maps.max() <= 1.0 + 1e-6
    mask = np.zeros((640, 224), dtype=np.uint8)
    mask[100:132, 50:82] = 255
    m = cam_mask_overlap(maps[0], mask)
    assert 0.0 <= m["cam_mass_in_mask"] <= 1.0 and m["mask_area_share"] == pytest.approx(
        1024 / (640 * 224)
    )
    # a map concentrated on the mask gives a share of 1 and a peak inside
    focused = np.zeros((640, 224), dtype=np.float32)
    focused[110:120, 60:70] = 1.0
    assert cam_mask_overlap(focused, mask)["cam_mass_in_mask"] == pytest.approx(1.0)
    assert cam_mask_overlap(focused, mask)["peak_in_mask"] is True


# ------------------------------------------------------------------------------ end to end


@pytest.fixture
def phase3_done(synthetic_config: Config, synthetic_dataset: dict, tmp_path: Path):
    """Phase 1-3 artefacts on the synthetic dataset, then a Phase 4 config pointing at P2_B."""
    cfg = synthetic_config
    root = synthetic_dataset["root"]
    manifest, strays = build_manifest(cfg, raw_layout(cfg), root)
    manifest, report, nn = analyse_duplicates(manifest, cfg)
    write_manifest(manifest, cfg, check_expected_counts(manifest, cfg), strays=strays)
    write_duplicates_report(report, nn, cfg)
    splits = make_splits(manifest, cfg)
    write_splits(splits, manifest, cfg, manifest_sha256=read_manifest_meta(cfg)["manifest_sha256"])
    run_baseline(cfg, root, only=["T1_svm"])
    run_experiment(cfg.phase3.experiments[0], cfg, root)  # P1_probe (used as "X4" stand-in)
    res = run_experiment(cfg.phase3.experiments[1], cfg, root)  # P2_B
    override = tmp_path / "phase4.yaml"
    override.write_text(
        yaml.safe_dump(
            {
                "inference": {"bundle_dir": str(tmp_path / "bundle")},
                "phase4": {
                    "candidate": {
                        "experiment_id": "P2_B",
                        "checkpoint": res["run"]["best_checkpoint"],
                    },
                    "test": {"frozen_baselines": ["E0_contrast_only"]},
                    "error_analysis": {
                        "n_fp_examples": 2,
                        "n_fn_examples": 2,
                        "n_tp_examples": 2,
                        "n_tn_examples": 2,
                    },
                    "gradcam": {"n_per_category": 2},
                },
                "phase3": {"reference_experiment": "T1_svm"},
            }
        ),
        encoding="utf-8",
    )
    cfg4 = load_config(cfg.source_files[0], overrides=[*cfg.source_files[1:], override])
    return cfg4, root


def test_phase4_chain_on_synthetic_data(phase3_done):
    from vqi.evaluation.errors import error_analysis
    from vqi.evaluation.final import (
        TestEvaluationError,
        analyse_validation,
        evaluate_test_once,
        freeze_bundle,
        verify_checkpoint,
    )

    cfg, root = phase3_done
    v = verify_checkpoint(cfg, root)
    assert v["validation"]["deterministic_two_passes"] and v["test_used"] is False
    assert (
        v["validation"]["max_abs_score_diff_vs_recorded"] <= cfg.phase4.verification.score_tolerance
    )

    a = analyse_validation(cfg)
    sel = a["threshold_selection"]
    assert 0.0 < sel["threshold"] <= sel["threshold_min_recall"]
    assert a["calibration"]["test_used"] is False and a["test_used"] is False

    # test refused before the freeze
    with pytest.raises((TestEvaluationError, BundleError)):
        evaluate_test_once(cfg, root)

    frozen = freeze_bundle(cfg)
    assert frozen["model_sha256"] == v["checkpoint_sha256"]
    bundle = Path(frozen["bundle_dir"])
    thr = json.loads((bundle / "threshold.json").read_text(encoding="utf-8"))
    assert thr["frozen"] and thr["test_used"] is False
    with pytest.raises(Exception, match="already exists"):
        freeze_bundle(cfg)

    # the bundle reproduces the checkpoint's validation scores through the Predictor API
    pred = Predictor.from_bundle(cfg)
    ids, y, p = _val_scores(cfg)
    manifest = _manifest(cfg)
    p_bundle = pred.score([root / manifest.loc[i, "image_path"] for i in ids[:5]])
    assert np.allclose(p_bundle, p[:5], atol=1e-6)
    assert pred.predict(root / manifest.loc[ids[0], "image_path"]).decision in ("ACCEPT", "REJECT")

    # the one and only test evaluation
    result = evaluate_test_once(cfg, root)
    assert result["test_used"] and result["single_evaluation"]
    assert result["n"] == 20 and result["n_pos"] == 4
    f = result["final"]["at_threshold"]
    assert f["tp"] + f["fp"] + f["fn"] + f["tn"] == 20
    assert f["threshold"] == pytest.approx(sel["threshold"])
    assert "E0_contrast_only" in result["frozen_baselines"]
    assert (bundle / "final_test_metrics.json").exists()
    with pytest.raises(TestEvaluationError, match="already been evaluated"):
        evaluate_test_once(cfg, root)

    errors = error_analysis(cfg, root)
    assert errors["test_used"] and sum(errors["counts"].values()) == 20
    assert "limits" in errors["gradcam"]


def _val_scores(cfg: Config):
    import pandas as pd

    df = pd.read_csv(
        cfg.paths.reports_dir / cfg.phase4.report_subdir / "val_scores.csv", dtype={"image_id": str}
    )
    return df["image_id"].to_numpy(dtype=str), df["label"].to_numpy(), df["score"].to_numpy()


def _manifest(cfg: Config):
    from vqi.data.manifest import read_manifest

    return read_manifest(cfg).set_index("image_id")


def test_prior_logit_matches_bundle_docs():
    assert math.log(197 / 1668) == pytest.approx(-2.1362, abs=1e-4)
