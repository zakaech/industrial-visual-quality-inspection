"""scikit-learn pipelines for the classical baseline and their hyper-parameter selection.

Leakage rules enforced here:
* the scaler lives inside the ``Pipeline`` → fitted on the training fold only;
* SVM hyper-parameters are chosen by stratified, group-aware cross-validation **inside the
  training split**; the validation split is never seen during selection.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from vqi.config import Config


def resolve_class_weight(cfg: Config, name: str) -> Any:
    """``none`` → None, ``balanced`` → "balanced", ``manual`` → the dict from the config."""
    options = cfg.baseline.class_weights
    if name not in options:
        raise KeyError(f"unknown class_weight option '{name}' (known: {sorted(options)})")
    value = options[name]
    if isinstance(value, dict):
        return {int(k): float(v) for k, v in value.items()}
    return value


def make_svm(c: float, gamma: Any, class_weight: Any, kernel: str = "rbf") -> Pipeline:
    svm = SVC(kernel=kernel, C=float(c), gamma=gamma, class_weight=class_weight)
    return Pipeline([("scaler", StandardScaler()), ("svm", svm)])


def make_random_forest(cfg: Config, class_weight: Any, seed: int) -> Pipeline:
    rf = cfg.baseline.random_forest
    return Pipeline(
        [
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=rf.n_estimators,
                    max_features=rf.max_features,
                    class_weight=class_weight,
                    random_state=seed,
                    n_jobs=-1,
                ),
            )
        ]
    )


def scores_of(model: Pipeline, x: np.ndarray) -> np.ndarray:
    """A higher score = more defective. SVM → signed distance to the margin; RF → P(defective)."""
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(x), dtype=np.float64)
    return np.asarray(model.predict_proba(x)[:, 1], dtype=np.float64)


def default_threshold(model: Pipeline) -> float:
    """The classifier's own decision rule (0 on a margin, 0.5 on a probability) — reported for
    reference only; operating points are chosen at recall targets."""
    return 0.0 if hasattr(model, "decision_function") else 0.5


@dataclass(frozen=True)
class CvSelection:
    best_params: dict[str, Any]
    table: pd.DataFrame  # one row per (C, gamma): mean/std of the CV metric


def _cv_metric(name: str):
    if name == "average_precision":
        return average_precision_score
    if name == "roc_auc":
        return roc_auc_score
    raise ValueError(f"unsupported cv_scoring: {name}")


def select_svm_by_cv(
    x_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    cfg: Config,
    class_weight: Any,
    seed: int,
    c_grid: Sequence[float] | None = None,
    gamma_grid: Sequence[Any] | None = None,
    kernel: str | None = None,
) -> CvSelection:
    """Grid over (C, gamma) scored by group-aware stratified k-fold CV on the training split.
    A linear kernel ignores gamma, so its grid collapses to C only."""
    svm_cfg = cfg.baseline.svm
    kernel = kernel or svm_cfg.kernel
    metric = _cv_metric(svm_cfg.cv_scoring)
    splitter = StratifiedGroupKFold(n_splits=svm_cfg.cv_folds, shuffle=True, random_state=seed)
    folds = list(splitter.split(x_train, y_train, groups_train))
    gammas = ["scale"] if kernel == "linear" else list(gamma_grid or svm_cfg.gamma_grid)
    rows = []
    for c in c_grid or svm_cfg.c_grid:
        for gamma in gammas:
            template = make_svm(c, gamma, class_weight, kernel=kernel)
            fold_scores = []
            for tr, va in folds:
                model = clone(template).fit(x_train[tr], y_train[tr])
                fold_scores.append(metric(y_train[va], scores_of(model, x_train[va])))
            rows.append(
                {
                    "C": float(c),
                    "gamma": gamma,
                    f"cv_{svm_cfg.cv_scoring}_mean": float(np.mean(fold_scores)),
                    f"cv_{svm_cfg.cv_scoring}_std": float(np.std(fold_scores)),
                }
            )
    table = pd.DataFrame(rows)
    mean_col = f"cv_{svm_cfg.cv_scoring}_mean"
    # ties → the simplest model (smallest C) wins, deterministically
    best = table.sort_values([mean_col, "C"], ascending=[False, True]).iloc[0]
    return CvSelection(
        best_params={"C": float(best["C"]), "gamma": best["gamma"], "kernel": kernel}, table=table
    )
