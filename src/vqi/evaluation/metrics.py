"""Evaluation of a scorer on a labelled set, independent of the model that produced the scores.

Conventions: positive class = defective (label 1); a higher score means "more defective".
Scores may be SVM decision values, random-forest probabilities or CNN sigmoid outputs — the
threshold analysis works on the score scale and makes no calibration claim.

Business context: a false negative (missed defect) costs more than a false positive
(re-inspection), so the primary threshold-free metric is Average Precision and the operating
points are reported *at recall targets*, not at an arbitrary 0.5.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def _as_arrays(y_true: Sequence[int], scores: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true).astype(int).ravel()
    s = np.asarray(scores, dtype=np.float64).ravel()
    if y.shape != s.shape:
        raise ValueError("y_true and scores must have the same length")
    if not np.isfinite(s).all():
        raise ValueError("scores contain NaN/Inf")
    if set(np.unique(y)) - {0, 1}:
        raise ValueError("y_true must be binary (0/1)")
    return y, s


def ranking_metrics(y_true: Sequence[int], scores: Sequence[float]) -> dict[str, float]:
    y, s = _as_arrays(y_true, scores)
    return {
        "average_precision": float(average_precision_score(y, s)),
        "roc_auc": float(roc_auc_score(y, s)),
        "n_pos": int(y.sum()),
        "n_neg": int((1 - y).sum()),
        "prevalence": float(y.mean()),
    }


def confusion_at(
    y_true: Sequence[int], scores: Sequence[float], threshold: float
) -> dict[str, Any]:
    """Counts and rates when predicting positive iff ``score >= threshold``."""
    y, s = _as_arrays(y_true, scores)
    pred = s >= threshold
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    tn = int((~pred & (y == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    f2 = 5 * precision * recall / (4 * precision + recall) if 4 * precision + recall else 0.0
    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
        "specificity": 1.0 - fpr,
        "f1": f1,
        "f2": f2,
        "accuracy": (tp + tn) / len(y),
    }


def threshold_for_recall(y_true: Sequence[int], scores: Sequence[float], target: float) -> float:
    """Highest threshold whose recall is >= ``target`` (the least permissive operating point that
    still catches the requested share of defects). Returns ``-inf`` if unreachable."""
    y, s = _as_arrays(y_true, scores)
    pos = np.sort(s[y == 1])[::-1]  # descending positive scores
    if len(pos) == 0:
        return float("-inf")
    k = int(np.ceil(target * len(pos)))  # need at least k positives above threshold
    k = min(max(k, 1), len(pos))
    return float(pos[k - 1])


def threshold_sweep(y_true: Sequence[int], scores: Sequence[float]) -> pd.DataFrame:
    """Precision / recall / FPR / counts at every distinct score threshold (descending)."""
    y, s = _as_arrays(y_true, scores)
    thresholds = np.unique(s)[::-1]
    rows = [confusion_at(y, s, t) for t in thresholds]
    return pd.DataFrame(rows)


def recall_target_table(
    y_true: Sequence[int], scores: Sequence[float], targets: Sequence[float]
) -> pd.DataFrame:
    """For each recall target: the operating point that reaches it and its cost (precision/FPR)."""
    rows = []
    for target in targets:
        t = threshold_for_recall(y_true, scores, target)
        c = confusion_at(y_true, scores, t)
        rows.append({"recall_target": float(target), **c})
    return pd.DataFrame(rows)


def bootstrap_ci(
    y_true: Sequence[int],
    scores: Sequence[float],
    metric: Callable[[np.ndarray, np.ndarray], float],
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """Percentile bootstrap CI of ``metric`` over resampled (y, score) pairs.

    Resamples that lack one of the classes are skipped (the metric is undefined for them).
    """
    y, s = _as_arrays(y_true, scores)
    rng = np.random.default_rng(seed)
    values = []
    n = len(y)
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        values.append(float(metric(yb, s[idx])))
    alpha = (1.0 - confidence_level) / 2.0
    arr = np.asarray(values)
    return {
        "low": float(np.quantile(arr, alpha)),
        "high": float(np.quantile(arr, 1 - alpha)),
        "n_effective": int(len(arr)),
    }


def curves(y_true: Sequence[int], scores: Sequence[float]) -> dict[str, list[float]]:
    y, s = _as_arrays(y_true, scores)
    precision, recall, _ = precision_recall_curve(y, s)
    fpr, tpr, _ = roc_curve(y, s)
    return {
        "pr_precision": precision.tolist(),
        "pr_recall": recall.tolist(),
        "roc_fpr": fpr.tolist(),
        "roc_tpr": tpr.tolist(),
    }


def evaluate_scores(
    y_true: Sequence[int],
    scores: Sequence[float],
    recall_targets: Sequence[float],
    default_threshold: float,
    n_resamples: int,
    confidence_level: float,
    seed: int,
) -> dict[str, Any]:
    """Everything a report needs, JSON-serialisable. Used by the classical baseline (Phase 2)
    and by the CNN evaluation (Phase 4) alike."""
    y, s = _as_arrays(y_true, scores)
    ranking = ranking_metrics(y, s)
    ap_ci = bootstrap_ci(y, s, average_precision_score, n_resamples, confidence_level, seed)
    auc_ci = bootstrap_ci(y, s, roc_auc_score, n_resamples, confidence_level, seed)
    targets = recall_target_table(y, s, recall_targets)
    return {
        **ranking,
        "average_precision_ci": ap_ci,
        "roc_auc_ci": auc_ci,
        "at_default_threshold": confusion_at(y, s, default_threshold),
        "at_recall_targets": targets.to_dict(orient="records"),
        "curves": curves(y, s),
    }
