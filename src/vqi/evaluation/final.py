"""Phase 4 — checkpoint verification, validation analysis, threshold freeze, single test pass.

Every function writes its artefacts under ``reports/phase4/`` and returns the numbers it wrote.
Order enforced by the scripts: verify → analyse/calibrate → threshold → freeze → test (once).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from vqi.config import Config
from vqi.data.dataset import build_split, make_loader
from vqi.data.hashing import sha256_file
from vqi.evaluation.metrics import confusion_at, ranking_metrics, threshold_for_recall
from vqi.inference.predictor import Predictor
from vqi.logging_utils import get_logger, log_event
from vqi.models.factory import model_summary
from vqi.preprocessing.pipeline import Preprocessor
from vqi.provenance import provenance
from vqi.training.experiments import experiment_json

logger = get_logger(__name__)


class VerificationError(RuntimeError):
    """The checkpoint does not reproduce the recorded validation results — stop."""


def report_dir(cfg: Config) -> Path:
    d = cfg.paths.reports_dir / cfg.phase4.report_subdir
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=float),
        encoding="utf-8",
        newline="\n",
    )
    return path


# --------------------------------------------------------------------------------------------- #
# Step 2 — checkpoint verification
# --------------------------------------------------------------------------------------------- #


@torch.no_grad()
def score_split(
    cfg: Config, predictor: Predictor, split: str, project_root: Path, use_cache: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic scores of one split through the standard dataset path (no augmentation,
    fixed order, eval mode). Returns (image_ids, labels, probabilities)."""
    ds, _, _ = build_split(cfg, split, predictor.pre, project_root, None, use_cache=use_cache)
    loader = make_loader(ds, cfg.inference.batch_size, False, 0, cfg.reproducibility.seed)
    ids, labels, probs = [], [], []
    for x, y, image_ids in loader:
        logits = predictor.model(x.to(predictor.device)).squeeze(1)
        probs.append(torch.sigmoid(logits).cpu().numpy().astype(np.float64))
        labels.append(y.numpy().astype(int))
        ids.extend(image_ids)
    return np.asarray(ids, dtype=str), np.concatenate(labels), np.concatenate(probs)


def verify_checkpoint(cfg: Config, project_root: Path) -> dict[str, Any]:
    """Load the candidate checkpoint, cross-check its metadata against the Phase 3 result and
    reproduce the validation scores deterministically (twice)."""
    cand = cfg.phase4.candidate
    result = json.loads(experiment_json(cfg, cand.experiment_id).read_text(encoding="utf-8"))
    ckpt_path = Path(cand.checkpoint)
    if not ckpt_path.exists():
        raise VerificationError(f"checkpoint not found: {ckpt_path}")
    if Path(result["run"]["best_checkpoint"]).resolve() != ckpt_path.resolve():
        raise VerificationError("the configured checkpoint is not the run's best checkpoint")

    predictor = Predictor.from_checkpoint(cfg, ckpt_path)
    meta = predictor.metadata
    checks = {
        "stage": (meta["stage"], result["run"]["stages"][-1]["name"]),
        "epoch": (meta["epoch"], result["run"]["stages"][-1]["best_epoch"]),
        "metric_val_ap": (
            round(meta["metric"], 6),
            round(result["run"]["stages"][-1]["best_metric"], 6),
        ),
        "pos_weight": (round(meta["pos_weight"], 6), round(result["spec"]["pos_weight_value"], 6)),
        "sampler": (meta["sampler"], result["spec"]["sampler"]),
        "config_hash": (meta["config_hash"], result["provenance"]["config_hash"]),
        "git_commit": (meta["git_commit"], result["provenance"]["git_commit"]),
        "seed": (meta["seed"], result["provenance"]["seed"]),
        "pooling": (meta["pooling"], result["spec"]["pooling"]),
        "preprocessor": (meta["preprocessor"], result["preprocessor"]),
        "bn_policy_in_run": (result["spec"]["bn_policy"], cfg.training.bn_policy),
    }
    mismatches = {k: v for k, v in checks.items() if v[0] != v[1]}
    if mismatches:
        raise VerificationError(f"checkpoint metadata differs from the run result: {mismatches}")

    summary = model_summary(predictor.model)
    if summary["n_params"] != result["model"]["n_params"]:
        raise VerificationError("parameter count differs from the run result")

    # deterministic inference on validation, twice
    t0 = time.perf_counter()
    ids1, y1, p1 = score_split(cfg, predictor, "val", project_root)
    seconds = time.perf_counter() - t0
    ids2, y2, p2 = score_split(cfg, predictor, "val", project_root)
    if not (np.array_equal(ids1, ids2) and np.array_equal(p1, p2)):
        raise VerificationError("two validation passes differ: inference is not deterministic")

    recorded = pd.read_csv(
        report_dir(cfg).parent
        / cfg.phase3.report_subdir
        / "experiments"
        / f"{cand.experiment_id}_val_scores.csv",
        dtype={"image_id": str},
    ).set_index("image_id")
    rec = recorded.loc[list(ids1), "score"].to_numpy()
    max_abs = float(np.abs(rec - p1).max())
    tol = cfg.phase4.verification.score_tolerance
    m = ranking_metrics(y1, p1)
    thr95 = threshold_for_recall(y1, p1, 0.95)
    at95 = confusion_at(y1, p1, thr95)
    rec95 = {t["recall_target"]: t for t in result["metrics"]["at_recall_targets"]}[0.95]
    reproduced = {
        "max_abs_score_diff_vs_recorded": max_abs,
        "tolerance": tol,
        "average_precision": (m["average_precision"], result["metrics"]["average_precision"]),
        "roc_auc": (m["roc_auc"], result["metrics"]["roc_auc"]),
        "recall_0.95_threshold": (thr95, rec95["threshold"]),
        "recall_0.95_confusion": (
            {k: at95[k] for k in ("tp", "fp", "fn", "tn")},
            {k: rec95[k] for k in ("tp", "fp", "fn", "tn")},
        ),
    }
    if max_abs > tol:
        raise VerificationError(f"validation scores differ from the recorded ones by {max_abs}")
    for key in ("average_precision", "roc_auc"):
        if abs(reproduced[key][0] - reproduced[key][1]) > 1e-6:
            raise VerificationError(f"{key} differs from the run result: {reproduced[key]}")
    if reproduced["recall_0.95_confusion"][0] != reproduced["recall_0.95_confusion"][1]:
        raise VerificationError("confusion at recall>=0.95 differs from the run result")

    out = {
        "experiment_id": cand.experiment_id,
        "checkpoint": str(ckpt_path),
        "checkpoint_sha256": meta["checkpoint_sha256"],
        "checkpoint_metadata": {k: v for k, v in meta.items() if k not in ("preprocessor",)},
        "model": summary,
        "metadata_checks": {k: v[0] for k, v in checks.items() if k != "preprocessor"},
        "validation": {
            "n": int(len(ids1)),
            "n_pos": int(y1.sum()),
            "seconds_per_pass": seconds,
            "deterministic_two_passes": True,
            **reproduced,
        },
        "test_used": False,
        "provenance": provenance(cfg),
    }
    _write_json(report_dir(cfg) / "checkpoint_verification.json", out)
    pd.DataFrame({"image_id": ids1, "label": y1, "score": p1}).to_csv(
        report_dir(cfg) / "val_scores.csv", index=False, lineterminator="\n"
    )
    log_event(
        logger,
        "verify.ok",
        checkpoint=str(ckpt_path),
        max_abs_diff=max_abs,
        val_ap=round(m["average_precision"], 4),
        seconds=round(seconds, 1),
    )
    return out


# --------------------------------------------------------------------------------------------- #
# Steps 3-5 — validation analysis, calibration, threshold selection
# --------------------------------------------------------------------------------------------- #


def read_val_scores(cfg: Config) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(report_dir(cfg) / "val_scores.csv", dtype={"image_id": str})
    return (
        df["image_id"].to_numpy(dtype=str),
        df["label"].to_numpy().astype(int),
        df["score"].to_numpy(dtype=np.float64),
    )


def analyse_validation(cfg: Config) -> dict[str, Any]:
    """Score analysis, calibration diagnostics and threshold selection — validation only."""
    from vqi.evaluation import figures_phase4 as figs
    from vqi.evaluation.calibration import (
        calibration_metrics,
        cross_validated_calibration,
        reliability_table,
    )
    from vqi.evaluation.metrics import evaluate_scores
    from vqi.evaluation.threshold import (
        bootstrap_at_fixed_threshold,
        candidate_table,
        select_threshold,
    )

    out_dir = report_dir(cfg)
    fig_dir = out_dir / "figures"
    ids, y, p = read_val_scores(cfg)
    ev, p4 = cfg.evaluation, cfg.phase4
    seed = cfg.reproducibility.seed

    metrics = evaluate_scores(
        y,
        p,
        p4.threshold.report_recall_targets,
        0.5,
        ev.bootstrap.n_resamples,
        ev.bootstrap.confidence_level,
        seed,
    )

    # calibration: raw diagnostics + cross-validated calibrators (validation only)
    n_bins = p4.calibration.n_bins
    raw = calibration_metrics(y, p, n_bins)
    tables = {"scores bruts": reliability_table(y, p, n_bins, adaptive=True)}
    calibrators: dict[str, Any] = {}
    for method in p4.calibration.methods:
        cv = cross_validated_calibration(y, p, method, n_bins, seed=seed)
        oof = cv.pop("oof_probabilities")
        tables[f"{method} (hors-pli)"] = reliability_table(y, oof, n_bins, adaptive=True)
        calibrators[method] = cv
    keep_raw = raw["ece_equal_mass"] < p4.calibration.keep_raw_if_ece_below
    calibration = {
        "fitted_on": "validation (cross-validated 5-fold for the out-of-fold numbers)",
        "test_used": False,
        "raw": raw,
        "raw_reliability_equal_mass": tables["scores bruts"],
        "raw_reliability_equal_width": reliability_table(y, p, n_bins, adaptive=False),
        "calibrators": calibrators,
        "rule": (
            f"keep raw scores if raw ECE (equal-mass bins) < {p4.calibration.keep_raw_if_ece_below}"
        ),
        "decision": "raw" if keep_raw else "to_decide",
    }

    # threshold selection
    candidates = candidate_table(y, p, p4.threshold.cost_ratios)
    selection = select_threshold(y, p, p4.threshold.target_recall, p4.threshold.margin)
    selection["bootstrap_at_threshold"] = bootstrap_at_fixed_threshold(
        y, p, selection["threshold"], ev.bootstrap.n_resamples, ev.bootstrap.confidence_level, seed
    )
    selection["n_val"], selection["n_val_pos"] = int(len(y)), int(y.sum())

    # figures
    t = selection["threshold"]
    at = selection["at_threshold"]
    figures = {
        "score_distribution": figs.score_distribution(
            y,
            p,
            t,
            fig_dir / "val_score_distribution.png",
            "Validation — distribution des probabilités prédites (X6)",
        ),
        "pr_roc": figs.pr_roc_curves(
            {"X6 (validation)": (y, p)},
            {"seuil final": (at["recall"], at["precision"])},
            fig_dir / "val_pr_roc.png",
            "Validation — courbes PR et ROC (X6)",
            float(y.mean()),
        ),
        "metrics_vs_threshold": figs.metrics_vs_threshold(
            y,
            p,
            t,
            fig_dir / "val_metrics_vs_threshold.png",
            "Validation — rappel, précision et FPR en fonction du seuil",
        ),
        "reliability": figs.reliability_diagram(
            tables,
            fig_dir / "val_reliability.png",
            "Validation — diagramme de fiabilité (bins à effectifs égaux)",
        ),
    }
    payload = {
        "split": "val",
        "n": int(len(y)),
        "n_pos": int(y.sum()),
        "metrics": metrics,
        "calibration": calibration,
        "threshold_selection": selection,
        "figures": {k: v.as_posix() for k, v in figures.items()},
        "test_used": False,
        "provenance": provenance(cfg),
    }
    _write_json(out_dir / "validation_analysis.json", payload)
    candidates.to_csv(out_dir / "threshold_candidates.csv", index=False, lineterminator="\n")
    log_event(
        logger,
        "validation.analysed",
        ap=round(metrics["average_precision"], 4),
        raw_ece=round(raw["ece_equal_mass"], 4),
        threshold=round(t, 5),
        calibration=calibration["decision"],
    )
    return payload


# --------------------------------------------------------------------------------------------- #
# Step 5 — freeze
# --------------------------------------------------------------------------------------------- #


class FreezeError(RuntimeError):
    pass


def freeze_bundle(cfg: Config, calibration_decision: str = "raw") -> dict[str, Any]:
    """Write the frozen bundle: model.pt (byte copy of the verified checkpoint),
    preprocessor.json, threshold.json, model_card.json. Refuses to overwrite a frozen bundle."""
    import shutil

    out_dir = report_dir(cfg)
    bundle = Path(cfg.inference.bundle_dir)
    if (bundle / "threshold.json").exists():
        raise FreezeError(f"a frozen bundle already exists in {bundle}; delete it deliberately")
    verification = json.loads(
        (out_dir / "checkpoint_verification.json").read_text(encoding="utf-8")
    )
    analysis = json.loads((out_dir / "validation_analysis.json").read_text(encoding="utf-8"))
    sel = analysis["threshold_selection"]
    if calibration_decision != "raw":
        raise FreezeError("only raw scores are supported by the bundle in v1")

    bundle.mkdir(parents=True, exist_ok=True)
    src = Path(verification["checkpoint"])
    shutil.copyfile(src, bundle / "model.pt")
    model_sha = sha256_file(bundle / "model.pt")
    if model_sha != verification["checkpoint_sha256"]:
        raise FreezeError("copied model.pt differs from the verified checkpoint")

    pre = Preprocessor.from_config(cfg)
    (bundle / "preprocessor.json").write_text(
        json.dumps(pre.describe(), indent=2), encoding="utf-8", newline="\n"
    )
    threshold = {
        "frozen": True,
        "frozen_at": provenance(cfg)["created_at"],
        "threshold": sel["threshold"],
        "score": "raw sigmoid output of the model (no post-hoc calibration)",
        "policy": sel["policy"],
        "target_recall": sel["target_recall"],
        "margin": sel["margin"],
        "threshold_min_recall": sel["threshold_min_recall"],
        "next_lower_positive_score": sel["next_lower_positive_score"],
        "validation_at_threshold": sel["at_threshold"],
        "validation_bootstrap_at_threshold": sel["bootstrap_at_threshold"],
        "selected_on": "val (466 images, 49 defective)",
        "calibration_decision": calibration_decision,
        "test_used": False,
    }
    _write_json(bundle / "threshold.json", threshold)
    meta = verification["checkpoint_metadata"]
    val_metrics = {k: v for k, v in analysis["metrics"].items() if k != "curves"}
    card = {
        "model": "X6_C_winner — ResNet-18 ImageNet, layer3+layer4+head fine-tuned, avg pooling",
        "experiment_id": verification["experiment_id"],
        "checkpoint_source": verification["checkpoint"],
        "model_sha256": model_sha,
        "checkpoint_metadata": meta,
        "architecture": verification["model"],
        "preprocessor": pre.describe(),
        "training_recipe": {
            "stages": "A (linear probe) -> B (layer4+head, X4) -> C (layer3+layer4+head)",
            "pos_weight": meta["pos_weight"],
            "sampler": meta["sampler"],
            "bn_policy": cfg.training.bn_policy,
            "optimizer": "AdamW, weight_decay 1e-4",
            "lr_stage_C": cfg.training.stages["C"]["lr"],
            "batch_size": cfg.training.batch_size,
            "augmentation": "hflip 0.5, vflip 0.5, brightness/contrast x[0.9,1.1] p 0.3 (train)",
            "seed": cfg.reproducibility.seed,
        },
        "data": {
            "dataset": cfg.dataset.name,
            "splits_file": str(cfg.paths.splits_dir / f"{cfg.split.manifest_version}.csv"),
            "train": "1865 images (197 defective)",
            "val": "466 images (49 defective)",
            "test": "1004 images (110 defective) — not used before the freeze",
        },
        "validation_metrics": val_metrics,
        "threshold": {
            k: v for k, v in threshold.items() if k != "validation_bootstrap_at_threshold"
        },
        "frozen_at": threshold["frozen_at"],
        "provenance": provenance(cfg),
    }
    _write_json(bundle / "model_card.json", card)
    log_event(
        logger,
        "bundle.frozen",
        dir=str(bundle),
        threshold=sel["threshold"],
        model_sha256=model_sha[:16],
    )
    return {"bundle_dir": str(bundle), "model_sha256": model_sha, "threshold": threshold}


# --------------------------------------------------------------------------------------------- #
# Step 6 — the single official test evaluation
# --------------------------------------------------------------------------------------------- #


class TestEvaluationError(RuntimeError):
    """A guard of the single test evaluation was violated."""


TEST_RESULT_FILENAME = "test_evaluation.json"


def _frozen_baselines(
    cfg: Config, project_root: Path, canvases: list[np.ndarray]
) -> dict[str, Any]:
    """Scores of the frozen baselines on the same canvases (same single pass, no decision):
    E0 = contrast only, E5 = Phase 2 SVM refitted deterministically on the cached train features
    with its selected hyper-parameters, X4 = stage-B checkpoint."""
    from vqi.baseline.classifier import make_svm
    from vqi.baseline.features import FeatureExtractor

    out: dict[str, Any] = {}
    wanted = set(cfg.phase4.test.frozen_baselines)
    fe = FeatureExtractor.from_config(cfg)
    all_names = fe.names(fe.all_extractors())
    feats = np.stack([fe.extract(c, fe.all_extractors()) for c in canvases])

    if "E0_contrast_only" in wanted:
        out["E0_contrast_only"] = {
            "scores": feats[:, all_names.index("contrast_std")],
            "description": "std of the luminance (trivial reference)",
        }
    if "E5_grid_edge_check" in wanted:
        e5 = json.loads(
            (
                cfg.paths.reports_dir
                / cfg.baseline.report_subdir
                / "experiments"
                / "E5_grid_edge_check.json"
            ).read_text(encoding="utf-8")
        )
        cache = sorted((cfg.baseline.features_cache_dir).glob("train__none__pad-median__*.npz"))
        if not cache:
            raise TestEvaluationError("Phase 2 train feature cache not found; cannot refit E5")
        z = np.load(cache[0], allow_pickle=False)
        names = list(z["names"])
        cols = [names.index(n) for n in fe.names(e5["spec"]["extractors"])]
        model = make_svm(e5["hyperparameters"]["C"], e5["hyperparameters"]["gamma"], "balanced")
        model.fit(z["x"][:, cols], z["labels"].astype(int))
        test_cols = [all_names.index(n) for n in fe.names(e5["spec"]["extractors"])]
        out["E5_grid_edge_check"] = {
            "scores": model.decision_function(feats[:, test_cols]),
            "description": (
                "82 classical features + RBF SVM (C="
                f"{e5['hyperparameters']['C']}, gamma={e5['hyperparameters']['gamma']}), "
                "refitted on train"
            ),
        }
    if "X4_B_pw_ratio" in wanted:
        x4 = json.loads(experiment_json(cfg, "X4_B_pw_ratio").read_text(encoding="utf-8"))
        pred = Predictor.from_checkpoint(cfg, x4["run"]["best_checkpoint"])
        scores = np.concatenate(
            [pred.score_canvases(canvases[i : i + 16]) for i in range(0, len(canvases), 16)]
        )
        out["X4_B_pw_ratio"] = {"scores": scores, "description": "stage-B checkpoint (X4)"}
    return out


def evaluate_test_once(cfg: Config, project_root: Path) -> dict[str, Any]:
    """Load the official test split for the first and only time, score it with the frozen bundle
    and compute every final metric. Guards: frozen bundle required; refuses if a result exists."""
    from vqi.data.manifest import read_manifest
    from vqi.data.splits import load_split_rows
    from vqi.evaluation import figures_phase4 as figs
    from vqi.evaluation.compare import paired_bootstrap, recall_by_area
    from vqi.evaluation.metrics import evaluate_scores, threshold_sweep
    from vqi.evaluation.threshold import bootstrap_at_fixed_threshold

    out_dir = report_dir(cfg)
    result_path = out_dir / TEST_RESULT_FILENAME
    if cfg.phase4.test.single_evaluation and result_path.exists():
        raise TestEvaluationError(
            f"{result_path} exists: the official test set has already been evaluated once"
        )
    predictor = Predictor.from_bundle(cfg)  # raises if not frozen
    threshold = float(predictor.threshold)
    t_start = time.perf_counter()

    rows = load_split_rows(cfg, "test", cfg.phase4.test.allowed_splits)
    ids = rows["image_id"].to_numpy(dtype=str)
    y = rows["label"].to_numpy().astype(int)
    log_event(
        logger,
        "test.loaded",
        n=int(len(rows)),
        n_pos=int(y.sum()),
        note="first and only load of the official test split",
    )

    # preprocess once (live, no cache), score X6 and the frozen baselines on the same canvases
    canvases = [predictor.pre(project_root / p) for p in rows["image_path"]]
    p = np.concatenate(
        [predictor.score_canvases(canvases[i : i + 16]) for i in range(0, len(canvases), 16)]
    )
    seconds_x6 = time.perf_counter() - t_start
    baselines = _frozen_baselines(cfg, project_root, canvases)

    ev = cfg.evaluation
    seed = cfg.reproducibility.seed
    metrics = evaluate_scores(
        y,
        p,
        cfg.phase4.threshold.report_recall_targets,
        threshold,
        ev.bootstrap.n_resamples,
        ev.bootstrap.confidence_level,
        seed,
    )
    at_thr = metrics["at_default_threshold"]  # = the frozen threshold
    ci_thr = bootstrap_at_fixed_threshold(
        y, p, threshold, ev.bootstrap.n_resamples, ev.bootstrap.confidence_level, seed
    )
    areas = rows["defect_area_px"].to_numpy()
    by_area = recall_by_area(y, p, areas, threshold, cfg.phase3.defect_area_bins)

    decisions = np.where(p >= threshold, predictor.positive_label, predictor.negative_label)
    outcome = np.where(
        (y == 1) & (p >= threshold),
        "TP",
        np.where(
            (y == 1) & (p < threshold), "FN", np.where((y == 0) & (p >= threshold), "FP", "TN")
        ),
    )
    manifest = read_manifest(cfg).set_index("image_id")
    stats_cols = [
        "mean_luminance",
        "std_luminance",
        "channel_spread",
        "defect_area_px",
        "defect_bbox_w",
        "defect_bbox_h",
        "n_defect_components",
        "height",
        "width",
    ]
    scores_df = pd.DataFrame(
        {"image_id": ids, "label": y, "score": p, "decision": decisions, "outcome": outcome}
    )
    for k, b in baselines.items():
        scores_df[f"score_{k}"] = b["scores"]
    scores_df = scores_df.join(manifest.loc[ids, stats_cols].reset_index(drop=True))
    scores_df.to_csv(out_dir / "test_scores.csv", index=False, lineterminator="\n")
    threshold_sweep(y, p).to_csv(
        out_dir / "test_threshold_sweep.csv", index=False, lineterminator="\n"
    )

    baseline_results = {}
    for k, b in baselines.items():
        s = np.asarray(b["scores"], dtype=np.float64)
        m = evaluate_scores(
            y,
            s,
            cfg.phase4.threshold.report_recall_targets,
            float(np.median(s)),
            ev.bootstrap.n_resamples,
            ev.bootstrap.confidence_level,
            seed,
        )
        m.pop("at_default_threshold")
        baseline_results[k] = {
            "description": b["description"],
            "average_precision": m["average_precision"],
            "average_precision_ci": m["average_precision_ci"],
            "roc_auc": m["roc_auc"],
            "at_recall_targets": m["at_recall_targets"],
            "curves": m["curves"],
            "paired_vs_x6": paired_bootstrap(
                y,
                s,
                p,
                cfg.phase3.paired_bootstrap.n_resamples,
                cfg.phase3.paired_bootstrap.confidence_level,
                seed,
            ),
        }

    fig_dir = out_dir / "figures"
    curves = {"X6 (final, test)": (y, p)}
    for k, b in baselines.items():
        curves[f"{k} (test)"] = (y, np.asarray(b["scores"], dtype=np.float64))
    figures = {
        "score_distribution": figs.score_distribution(
            y,
            p,
            threshold,
            fig_dir / "test_score_distribution.png",
            "Test officiel — distribution des probabilités prédites (X6, seuil gelé)",
        ),
        "pr_roc": figs.pr_roc_curves(
            curves,
            {"seuil gelé": (at_thr["recall"], at_thr["precision"])},
            fig_dir / "test_pr_roc.png",
            "Test officiel — PR et ROC (modèle final et références gelées)",
            float(y.mean()),
        ),
        "recall_by_area": figs.recall_by_area_bars(
            by_area,
            fig_dir / "test_recall_by_area.png",
            "Test officiel — rappel par taille de défaut au seuil gelé (X6)",
        ),
    }

    result = {
        "test_used": True,
        "single_evaluation": True,
        "evaluated_at": provenance(cfg)["created_at"],
        "split": "test",
        "n": int(len(y)),
        "n_pos": int(y.sum()),
        "n_neg": int((y == 0).sum()),
        "bundle": {
            "dir": predictor.metadata["bundle_dir"],
            "model_sha256": predictor.metadata["model_sha256"],
            "threshold": predictor.metadata["threshold"],
            "model_card_frozen_at": predictor.metadata["model_card"]["frozen_at"],
        },
        "threshold": threshold,
        "final": {
            "average_precision": metrics["average_precision"],
            "average_precision_ci": metrics["average_precision_ci"],
            "roc_auc": metrics["roc_auc"],
            "roc_auc_ci": metrics["roc_auc_ci"],
            "at_threshold": at_thr,
            "at_threshold_ci": ci_thr,
            "at_recall_targets_information_only": metrics["at_recall_targets"],
            "recall_by_defect_area": by_area,
            "curves": metrics["curves"],
            "outcome_counts": {k: int((outcome == k).sum()) for k in ("TP", "FP", "FN", "TN")},
        },
        "false_negatives": scores_df[scores_df["outcome"] == "FN"]
        .sort_values("score", ascending=False)
        .to_dict(orient="records"),
        "false_positives": scores_df[scores_df["outcome"] == "FP"]
        .sort_values("score", ascending=False)
        .to_dict(orient="records"),
        "frozen_baselines": baseline_results,
        "timing": {
            "x6_preprocess_and_score_seconds": seconds_x6,
            "total_seconds": time.perf_counter() - t_start,
        },
        "figures": {k: v.as_posix() for k, v in figures.items()},
        "provenance": provenance(cfg),
    }
    _write_json(result_path, result)
    _write_json(
        Path(cfg.inference.bundle_dir) / "final_test_metrics.json",
        {
            "test_used": True,
            "evaluated_at": result["evaluated_at"],
            "n": result["n"],
            "n_pos": result["n_pos"],
            "threshold": threshold,
            "average_precision": metrics["average_precision"],
            "average_precision_ci": metrics["average_precision_ci"],
            "roc_auc": metrics["roc_auc"],
            "at_threshold": at_thr,
            "at_threshold_ci": ci_thr,
        },
    )
    log_event(
        logger,
        "test.evaluated_once",
        n=int(len(y)),
        ap=round(metrics["average_precision"], 4),
        recall=round(at_thr["recall"], 4),
        precision=round(at_thr["precision"], 4),
        fpr=round(at_thr["fpr"], 4),
        fn=at_thr["fn"],
        fp=at_thr["fp"],
    )
    return result
