"""Experiment runner for the classical baseline.

One experiment = (feature group, preprocessing variant, padding, classifier, class weight),
declared in ``baseline.experiments`` of the config. For each: features are extracted (and cached
per preprocessing variant) for the train and validation splits, hyper-parameters are selected by
CV inside train, the model is fitted on train and scored on validation, and everything —
config hash, seed, features, timings, metrics, threshold sweep — is written to
``reports/phase2/experiments/<id>.json`` (+ CSVs). ``reports/phase2/experiments.csv`` summarises.

The official test split is refused by :func:`load_split_rows` (``baseline.allowed_splits``).
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vqi.baseline.classifier import (
    default_threshold,
    make_random_forest,
    make_svm,
    resolve_class_weight,
    scores_of,
    select_svm_by_cv,
)
from vqi.baseline.features import FeatureExtractor
from vqi.config import Config, ExperimentSpec
from vqi.data.splits import ForbiddenSplitError, read_splits_meta  # noqa: F401 (re-export)
from vqi.data.splits import load_split_rows as _load_split_rows
from vqi.evaluation.metrics import evaluate_scores, threshold_sweep
from vqi.logging_utils import get_logger, log_event
from vqi.preprocessing import Preprocessor
from vqi.provenance import provenance

logger = get_logger(__name__)


def load_split_rows(cfg: Config, split: str) -> pd.DataFrame:
    """Baseline view of the shared guard: only ``baseline.allowed_splits`` may be loaded."""
    return _load_split_rows(cfg, split, cfg.baseline.allowed_splits)


# --------------------------------------------------------------------------------------------- #
# Feature cache
# --------------------------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FeatureMatrix:
    image_ids: np.ndarray
    labels: np.ndarray
    groups: np.ndarray
    x: np.ndarray
    names: list[str]
    extract_seconds: float
    cached: bool

    def select(self, names: Sequence[str]) -> np.ndarray:
        idx = [self.names.index(n) for n in names]
        return self.x[:, idx]


def _cache_key(cfg: Config, pre: Preprocessor, extractor: FeatureExtractor, split: str) -> str:
    payload = json.dumps(
        {
            "preprocessor": pre.describe(),
            "extractor": {
                "histogram_bins": extractor.histogram_bins,
                "lbp_scales": extractor.lbp_scales,
                "patch_size": extractor.patch_size,
            },
            "splits_sha256": read_splits_meta(cfg)["splits_sha256"],
            "split": split,
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def extract_features(
    cfg: Config,
    pre: Preprocessor,
    extractor: FeatureExtractor,
    split: str,
    project_root: Path,
    use_cache: bool = True,
) -> FeatureMatrix:
    """All extractors for one split under one preprocessing; cached as .npz keyed by content."""
    rows = load_split_rows(cfg, split)
    key = _cache_key(cfg, pre, extractor, split)
    cache_dir = cfg.baseline.features_cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{split}__{pre.variant}__pad-{pre.pad_value}__{key}.npz"
    names = extractor.names(extractor.all_extractors())

    if use_cache and path.exists():
        z = np.load(path, allow_pickle=False)
        if list(z["names"]) == names and list(z["image_ids"]) == rows["image_id"].tolist():
            return FeatureMatrix(
                z["image_ids"], z["labels"], z["groups"], z["x"], names, float(z["seconds"]), True
            )

    t0 = time.perf_counter()
    x = np.stack(
        [
            extractor.extract(pre(project_root / p), extractor.all_extractors())
            for p in rows["image_path"]
        ]
    )
    seconds = time.perf_counter() - t0
    labels = rows["label"].to_numpy().astype(int)
    groups = rows["group_id"].to_numpy().astype(str)
    ids = rows["image_id"].to_numpy().astype(str)
    np.savez(
        path,
        image_ids=ids,
        labels=labels,
        groups=groups,
        x=x,
        names=np.asarray(names),
        seconds=seconds,
    )
    log_event(
        logger,
        "features.extracted",
        split=split,
        variant=pre.variant,
        pad=pre.pad_value,
        n=len(ids),
        n_features=x.shape[1],
        seconds=round(seconds, 1),
    )
    return FeatureMatrix(ids, labels, groups, x, names, seconds, False)


# --------------------------------------------------------------------------------------------- #
# One experiment
# --------------------------------------------------------------------------------------------- #


def experiment_dir(cfg: Config) -> Path:
    return cfg.paths.reports_dir / cfg.baseline.report_subdir / "experiments"


def run_experiment(
    spec: ExperimentSpec, cfg: Config, project_root: Path, use_cache: bool = True
) -> dict[str, Any]:
    seed = cfg.reproducibility.seed
    pre = Preprocessor.from_config(cfg, variant=spec.preprocessing, pad_value=spec.pad_value)
    extractor = FeatureExtractor.from_config(cfg)
    group_extractors = cfg.baseline.feature_groups[spec.features]
    feature_names = extractor.names(group_extractors)

    train = extract_features(cfg, pre, extractor, "train", project_root, use_cache)
    val = extract_features(cfg, pre, extractor, "val", project_root, use_cache)
    x_tr, x_va = train.select(feature_names), val.select(feature_names)

    class_weight = resolve_class_weight(cfg, spec.class_weight)
    t0 = time.perf_counter()
    cv_table: pd.DataFrame | None = None
    best_params: dict[str, Any] = {}
    if spec.classifier == "none":
        # The feature itself is the score (E0): no fitting at all.
        if x_tr.shape[1] != 1:
            raise ValueError("classifier 'none' requires a single feature")
        val_scores = x_va[:, 0]
        thr = float(np.median(x_tr[:, 0]))  # arbitrary reference point, reported as such
    elif spec.classifier == "svm":
        selection = select_svm_by_cv(
            x_tr,
            train.labels,
            train.groups,
            cfg,
            class_weight,
            seed,
            c_grid=spec.c_grid,
            gamma_grid=spec.gamma_grid,
            kernel=spec.kernel,
        )
        best_params, cv_table = selection.best_params, selection.table
        model = make_svm(
            best_params["C"], best_params["gamma"], class_weight, kernel=best_params["kernel"]
        )
        model.fit(x_tr, train.labels)
        val_scores = scores_of(model, x_va)
        thr = default_threshold(model)
    elif spec.classifier == "rf":
        model = make_random_forest(cfg, class_weight, seed)
        model.fit(x_tr, train.labels)
        best_params = {"n_estimators": cfg.baseline.random_forest.n_estimators}
        val_scores = scores_of(model, x_va)
        thr = default_threshold(model)
    else:
        raise ValueError(f"unknown classifier: {spec.classifier}")
    fit_seconds = time.perf_counter() - t0

    ev = cfg.evaluation
    metrics = evaluate_scores(
        val.labels,
        val_scores,
        recall_targets=ev.report_recall_targets,
        default_threshold=thr,
        n_resamples=ev.bootstrap.n_resamples,
        confidence_level=ev.bootstrap.confidence_level,
        seed=seed,
    )
    sweep = threshold_sweep(val.labels, val_scores)

    result = {
        "spec": {
            "id": spec.id,
            "features": spec.features,
            "extractors": list(group_extractors),
            "n_features": len(feature_names),
            "preprocessing": spec.preprocessing,
            "pad_value": pre.pad_value,
            "classifier": spec.classifier,
            "class_weight": spec.class_weight,
            "class_weight_value": class_weight,
        },
        "preprocessor": pre.describe(),
        "hyperparameters": best_params,
        "cv": None
        if cv_table is None
        else {
            "folds": cfg.baseline.svm.cv_folds,
            "scoring": cfg.baseline.svm.cv_scoring,
            "on_split": "train",
            "c_grid": list(spec.c_grid or cfg.baseline.svm.c_grid),
            "gamma_grid": list(spec.gamma_grid or cfg.baseline.svm.gamma_grid),
            "table": cv_table.to_dict(orient="records"),
        },
        "data": {
            "train": {"n": int(len(train.labels)), "n_pos": int(train.labels.sum())},
            "val": {"n": int(len(val.labels)), "n_pos": int(val.labels.sum())},
            "evaluated_on": "val",
        },
        "timing": {
            "extract_train_seconds": round(train.extract_seconds, 2),
            "extract_val_seconds": round(val.extract_seconds, 2),
            "features_from_cache": bool(train.cached and val.cached),
            "fit_and_select_seconds": round(fit_seconds, 2),
        },
        "metrics": metrics,
        "provenance": provenance(cfg),
    }

    out_dir = experiment_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{spec.id}.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
        newline="\n",
    )
    sweep.to_csv(out_dir / f"{spec.id}_threshold_sweep.csv", index=False, lineterminator="\n")
    pd.DataFrame({"image_id": val.image_ids, "label": val.labels, "score": val_scores}).to_csv(
        out_dir / f"{spec.id}_val_scores.csv", index=False, lineterminator="\n"
    )
    log_event(
        logger,
        "experiment.done",
        id=spec.id,
        val_ap=round(metrics["average_precision"], 4),
        val_roc_auc=round(metrics["roc_auc"], 4),
        best=best_params,
        seconds=round(fit_seconds, 1),
    )
    return result


def summarise(results: Sequence[dict[str, Any]], recall_targets: Sequence[float]) -> pd.DataFrame:
    rows = []
    for r in results:
        m = r["metrics"]
        by_target = {t["recall_target"]: t for t in m["at_recall_targets"]}
        row = {
            "id": r["spec"]["id"],
            "features": r["spec"]["features"],
            "n_features": r["spec"]["n_features"],
            "preprocessing": r["spec"]["preprocessing"],
            "pad_value": r["spec"]["pad_value"],
            "classifier": r["spec"]["classifier"],
            "class_weight": r["spec"]["class_weight"],
            "best_C": r["hyperparameters"].get("C"),
            "best_gamma": r["hyperparameters"].get("gamma"),
            "cv_ap_mean": None
            if r["cv"] is None
            else max(t["cv_average_precision_mean"] for t in r["cv"]["table"]),
            "val_ap": m["average_precision"],
            "val_ap_ci_low": m["average_precision_ci"]["low"],
            "val_ap_ci_high": m["average_precision_ci"]["high"],
            "val_roc_auc": m["roc_auc"],
            "val_roc_auc_ci_low": m["roc_auc_ci"]["low"],
            "val_roc_auc_ci_high": m["roc_auc_ci"]["high"],
            "default_thr_precision": m["at_default_threshold"]["precision"],
            "default_thr_recall": m["at_default_threshold"]["recall"],
            "default_thr_f1": m["at_default_threshold"]["f1"],
        }
        for t in recall_targets:
            pt = by_target[float(t)]
            row[f"precision@r{t:.2f}"] = pt["precision"]
            row[f"fpr@r{t:.2f}"] = pt["fpr"]
            row[f"fp@r{t:.2f}"] = pt["fp"]
        row["extract_seconds"] = (
            r["timing"]["extract_train_seconds"] + r["timing"]["extract_val_seconds"]
        )
        row["fit_seconds"] = r["timing"]["fit_and_select_seconds"]
        rows.append(row)
    return pd.DataFrame(rows)


def run_all(
    cfg: Config, project_root: Path, only: Sequence[str] | None = None, use_cache: bool = True
) -> pd.DataFrame:
    specs = [s for s in cfg.baseline.experiments if not only or s.id in only]
    results = [run_experiment(s, cfg, project_root, use_cache) for s in specs]
    summary = summarise(results, cfg.evaluation.report_recall_targets)
    out = cfg.paths.reports_dir / cfg.baseline.report_subdir / "experiments.csv"
    if only and out.exists():  # merge partial reruns into the existing summary
        previous = pd.read_csv(out)
        previous = previous[~previous["id"].isin(summary["id"])]
        summary = pd.concat([previous, summary], ignore_index=True)
        order = {s.id: i for i, s in enumerate(cfg.baseline.experiments)}
        summary = summary.sort_values("id", key=lambda s: s.map(order)).reset_index(drop=True)
    summary.to_csv(out, index=False, lineterminator="\n")
    (out.with_suffix(".meta.json")).write_text(
        json.dumps(provenance(cfg, n_experiments=len(summary)), indent=2, ensure_ascii=False),
        encoding="utf-8",
        newline="\n",
    )
    return summary
