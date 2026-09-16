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
    batchnorm_modules,
    build_model,
    model_summary,
    param_groups,
    prior_logit,
    set_train_mode,
    set_trainable,
)
from vqi.training.dryrun import run_dry_run
from vqi.training.experiments import experiment_json, run_experiment, select_winner, summarise
from vqi.training.loop import EarlyStopper, StageTransitionError, Trainer, cosine_with_warmup
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


def test_all_eval_policy_keeps_every_batchnorm_in_eval(cfg: Config):
    """Real behaviour, not a config value: after model.train(), all_eval leaves every BatchNorm in
    eval mode (running stats frozen, outputs identical to eval), keeps dropout active in the head
    and keeps γ/β of unfrozen layers trainable."""
    torch.manual_seed(0)
    model = build_model(cfg, 197, 1668)
    set_trainable(model, ["layer4", "head"])
    model.train()
    assert any(m.training for m in batchnorm_modules(model))  # plain train() would train BN
    set_train_mode(model, ["layer4", "head"], "all_eval")
    assert all(m.training is False for m in batchnorm_modules(model))
    assert model.head[0].training is True  # dropout stays active
    assert all(p.requires_grad for m in batchnorm_modules(model.layer4) for p in m.parameters())
    x = torch.rand(4, 3, 640, 224)
    before = [m.running_mean.clone() for m in batchnorm_modules(model)]
    with torch.no_grad():
        feats_train_mode = model.features(x)
        after = [m.running_mean for m in batchnorm_modules(model)]
        model.eval()
        feats_eval = model.features(x)
    assert all(torch.equal(a, b) for a, b in zip(before, after, strict=True))  # no drift
    assert torch.equal(feats_train_mode, feats_eval)
    # the rejected policy really does change the features (what the X2 incident showed)
    set_train_mode(model, ["layer4", "head"], "frozen_eval")
    assert any(m.training for m in batchnorm_modules(model.layer4))
    with torch.no_grad():
        feats_batch_stats = model.features(x)
    assert not torch.allclose(feats_batch_stats, feats_eval)


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
    check = full["run"]["stages"][1]["transition_check"]
    assert check["passed"] and check["max_abs_logit_diff"] <= check["tolerance"]
    assert check["bn_running_mean_drift"] == 0.0 and check["bn_policy"] == "all_eval"
    assert any(r.get("epoch") == 0 and "transition_check" in r for r in full["history"])
    assert full["spec"]["pos_weight_value"] == pytest.approx(
        resolve_pos_weight(
            "sqrt_ratio", full["data"]["train"]["n_pos"], full["data"]["train"]["n_neg"]
        )
    )
    assert full["spec"]["sampler"] == "weighted_random"
    assert Path(full["run"]["best_checkpoint"]).exists()
    history = full["history"]
    assert {r["stage"] for r in history} == {"A", "B"}
    epochs = [r for r in history if r["epoch"] > 0]  # epoch 0 rows carry the transition check
    assert all("val_precision_r095" in r and "seconds" in r for r in epochs)

    # checkpoint reload reproduces the validation scores exactly
    ckpt = torch.load(full["run"]["best_checkpoint"], map_location="cpu", weights_only=False)
    assert ckpt["config_hash"] == cfg.hash and ckpt["seed"] == 42

    summary = summarise(cfg)
    assert list(summary["id"]) == ["P1_probe", "P2_B"]
    assert full["model"]["head_bias_init"] == pytest.approx(
        math.log(full["data"]["train"]["n_pos"] / full["data"]["train"]["n_neg"])
    )
    from vqi.training.report import build_report

    report = build_report(cfg)
    text = report.read_text(encoding="utf-8")
    assert "Comparaison avec la référence Phase 2" in text and "P2_B" in text
    assert (report.parent / "figures" / "pr_curves.png").exists()
    winner, _ = select_winner(cfg, ("P1_probe", "P2_B"))
    assert winner in ("P1_probe", "P2_B")
    assert (
        json.loads(experiment_json(cfg, "P2_B").read_text(encoding="utf-8"))["spec"]["id"] == "P2_B"
    )


def test_transition_check_detects_batch_statistics_policy(prepared, tmp_path):
    """Under the rejected frozen_eval policy the check must fail before any optimizer step."""
    cfg, root = prepared
    override = tmp_path / "bn.yaml"
    override.write_text("training:\n  bn_policy: frozen_eval\n", encoding="utf-8")
    bad_cfg = load_config(cfg.source_files[0], overrides=[*cfg.source_files[1:], override])
    from vqi.preprocessing import Preprocessor
    from vqi.training.losses import build_loss as _build_loss

    model = build_model(bad_cfg, 8, 30)
    trainer = Trainer(
        bad_cfg,
        model,
        Preprocessor.from_config(bad_cfg),
        root,
        tmp_path / "run",
        _build_loss("bce_with_logits", 1.0, 2.0),
        "none",
        1.0,
    )
    set_trainable(model, ["layer4", "head"])
    with pytest.raises(StageTransitionError):
        trainer.transition_check(["layer4", "head"])
    good_cfg = cfg  # all_eval
    trainer_ok = Trainer(
        good_cfg,
        model,
        Preprocessor.from_config(good_cfg),
        root,
        tmp_path / "run_ok",
        _build_loss("bce_with_logits", 1.0, 2.0),
        "none",
        1.0,
    )
    assert trainer_ok.transition_check(["layer4", "head"])["passed"]
