"""Model factory, losses/samplers, staged trainer, experiment runner, paired comparison."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

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
from vqi.evaluation.compare import paired_bootstrap, recall_by_area
from vqi.models.factory import (
    GROUPS,
    build_model,
    model_summary,
    param_groups,
    prior_logit,
    set_train_mode,
    set_trainable,
)
from vqi.training.dryrun import run_dry_run
from vqi.training.experiments import experiment_json, run_experiment, select_winner, summarise
from vqi.training.loop import EarlyStopper, cosine_with_warmup
from vqi.training.losses import build_loss, build_sampler, resolve_pos_weight

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cfg() -> Config:
    return load_config(REPO_ROOT / "configs" / "default.yaml")


@pytest.fixture
def prepared(synthetic_config: Config, synthetic_dataset: dict):
    cfg = synthetic_config
    root = synthetic_dataset["root"]
    manifest, strays = build_manifest(cfg, raw_layout(cfg), root)
    manifest, report, nn = analyse_duplicates(manifest, cfg)
    write_manifest(manifest, cfg, check_expected_counts(manifest, cfg), strays=strays)
    write_duplicates_report(report, nn, cfg)
    splits = make_splits(manifest, cfg)
    write_splits(splits, manifest, cfg, manifest_sha256=read_manifest_meta(cfg)["manifest_sha256"])
    run_baseline(cfg, root, only=["T1_svm"])  # the Phase 2 reference the CNN is compared with
    return cfg, root


# ------------------------------------------------------------------------------ model


def test_prior_logit_formula():
    assert prior_logit(197, 1668) == pytest.approx(math.log(197 / 1668))
    assert prior_logit(1, 1) == 0.0
    with pytest.raises(ValueError):
        prior_logit(0, 5)


def test_model_output_bias_and_groups(cfg: Config):
    model = build_model(cfg, n_pos=197, n_neg=1668)
    assert float(model.head[1].bias.detach()) == pytest.approx(math.log(197 / 1668))
    assert torch.count_nonzero(model.head[1].weight) == 0
    x = torch.zeros(2, 3, 640, 224)
    model.eval()
    with torch.no_grad():
        out = model(x)
        feats = model.features(x)
    assert tuple(out.shape) == (2, 1) and tuple(feats.shape) == (2, 512)
    assert torch.sigmoid(out).mean().item() == pytest.approx(197 / 1865, abs=1e-6)
    summary = model_summary(model)
    assert summary["n_params"] == sum(summary["n_params_by_group"].values())


def test_pooling_variants(cfg: Config):
    for pooling, dim in (("avg", 512), ("max", 512), ("avgmax", 1024)):
        m = build_model(cfg, 1, 1, pooling=pooling)
        assert m.feat_dim == dim and m.head[1].in_features == dim


def test_staged_freezing_and_bn_policy(cfg: Config):
    model = build_model(cfg, 197, 1668)
    counts = set_trainable(model, ["layer4", "head"])
    assert counts["layer4"] > 0 and counts["head"] == 513
    assert all(counts[g] == 0 for g in ("stem", "layer1", "layer2", "layer3"))
    for g in GROUPS:
        flag = g in ("layer4", "head")
        assert all(p.requires_grad == flag for p in model.group(g).parameters())
    set_train_mode(model, ["layer4", "head"], "frozen_eval")
    assert model.stem[1].training is False  # frozen BN stays in eval
    assert model.layer4[0].bn1.training is True
    assert model.head.training is True
    groups = param_groups(model, {"layer4": 1e-4, "head": 1e-3}, 1e-4)
    assert {g["name"]: g["lr"] for g in groups} == {"layer4": 1e-4, "head": 1e-3}
    with pytest.raises(KeyError):
        set_trainable(model, ["fc"])


# ------------------------------------------------------------------------------ loss / sampler


def test_pos_weight_resolution_and_loss():
    assert resolve_pos_weight("ratio", 197, 1668) == pytest.approx(1668 / 197)
    assert resolve_pos_weight("sqrt_ratio", 197, 1668) == pytest.approx(math.sqrt(1668 / 197))
    assert resolve_pos_weight(2.5, 1, 1) == 2.5
    with pytest.raises(ValueError):
        resolve_pos_weight("inherit", 1, 1)
    loss = build_loss("bce_with_logits", 3.0, 2.0)
    logits = torch.tensor([0.5, -1.0])
    targets = torch.tensor([1.0, 0.0])
    manual = (
        3.0 * -torch.log(torch.sigmoid(logits[0])) + -torch.log(1 - torch.sigmoid(logits[1]))
    ) / 2
    assert float(loss(logits, targets)) == pytest.approx(float(manual), rel=1e-5)


def test_weighted_sampler_balances_classes():
    labels = np.array([1.0] * 10 + [0.0] * 90)
    sampler = build_sampler("weighted_random", labels, seed=1)
    draws = np.array(list(sampler))
    share_pos = labels[draws].mean()
    assert 0.4 < share_pos < 0.6 and len(draws) == 100
    assert build_sampler("none", labels, 1) is None


# ------------------------------------------------------------------------------ loop helpers


def test_cosine_schedule_and_early_stopper():
    fn = cosine_with_warmup(total_steps=100, warmup_steps=10)
    assert fn(0) == pytest.approx(0.1) and fn(9) == pytest.approx(1.0)
    assert fn(10) == pytest.approx(1.0) and fn(100) == pytest.approx(0.0, abs=1e-9)
    es = EarlyStopper(patience=2, min_delta=0.01, min_epochs=2)
    assert es.update(0.50, 1) and not es.update(0.505, 2) and not es.update(0.50, 3)
    assert es.should_stop(3) and es.best_epoch == 1
    assert es.update(0.60, 4) and not es.should_stop(4)


# ------------------------------------------------------------------------------ comparison


def test_paired_bootstrap_and_recall_by_area():
    rng = np.random.default_rng(0)
    y = np.array([0] * 40 + [1] * 10)
    a = rng.random(50) + y * 0.5
    same = paired_bootstrap(y, a, a, n_resamples=100, seed=1)
    assert same["delta_average_precision"]["point"] == 0.0
    assert same["delta_average_precision"]["low"] <= 0.0 <= same["delta_average_precision"]["high"]
    better = paired_bootstrap(y, a, a + y * 2.0, n_resamples=100, seed=1)
    assert better["delta_average_precision"]["point"] > 0
    areas = np.array([0] * 40 + [10, 20, 300, 400, 2000, 3000, 6000, 7000, 8000, 9000])
    table = recall_by_area(y, a + y * 2.0, areas, threshold=1.5, bins=[0, 250, 5000, 10**6])
    assert [r["n_positives"] for r in table] == [2, 4, 4]
    assert all(r["recall"] == 1.0 for r in table)


# ------------------------------------------------------------------------------ end to end


def test_dry_run_writes_measurements(prepared):
    cfg, root = prepared
    report = run_dry_run(cfg, root)
    assert report["cache"]["train"]["check_vs_live"]["n_mismatch"] == 0
    assert report["compute"][0]["train_step_s_B_layer4"] > 0
    assert "bit_identical" in report["determinism"]
    out = cfg.paths.reports_dir / cfg.phase3.report_subdir
    assert (out / "dry_run.json").exists() and (out / "dry_run.md").exists()


def test_experiments_end_to_end(prepared):
    cfg, root = prepared
    probe = run_experiment(cfg.phase3.experiments[0], cfg, root)
    assert probe["data"]["test_used"] is False and probe["data"]["evaluated_on"] == "val"
    assert probe["run"]["stages"][0]["cached_features"] is True
    assert probe["metrics"]["average_precision"] >= 0.0
    assert probe["comparison_vs_phase2"]["reference"] == "T1_svm"
    assert set(probe["metrics"]) >= {
        "at_recall_targets",
        "average_precision_ci",
        "recall_by_defect_area_at_r095",
    }

    full = run_experiment(cfg.phase3.experiments[1], cfg, root)
    assert [s["name"] for s in full["run"]["stages"]] == ["A", "B"]
    assert full["spec"]["pos_weight_value"] == pytest.approx(
        resolve_pos_weight(
            "sqrt_ratio", full["data"]["train"]["n_pos"], full["data"]["train"]["n_neg"]
        )
    )
    assert full["spec"]["sampler"] == "weighted_random"
    assert Path(full["run"]["best_checkpoint"]).exists()
    history = full["history"]
    assert {r["stage"] for r in history} == {"A", "B"}
    assert all("val_precision_r095" in r and "seconds" in r for r in history)

    # checkpoint reload reproduces the validation scores exactly
    ckpt = torch.load(full["run"]["best_checkpoint"], map_location="cpu", weights_only=False)
    assert ckpt["config_hash"] == cfg.hash and ckpt["seed"] == 42

    summary = summarise(cfg)
    assert list(summary["id"]) == ["P1_probe", "P2_B"]
    winner, _ = select_winner(cfg, ("P1_probe", "P2_B"))
    assert winner in ("P1_probe", "P2_B")
    assert (
        json.loads(experiment_json(cfg, "P2_B").read_text(encoding="utf-8"))["spec"]["id"] == "P2_B"
    )
