"""Paired comparison of two scorers evaluated on the *same* images.

Marginal confidence intervals of two models overlap easily with 49 positives; resampling the same
indices for both models (paired bootstrap) estimates the distribution of the *difference* directly
and is the right tool to say whether model B beats model A on this validation set.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from vqi.evaluation.metrics import confusion_at, threshold_for_recall


def _point_metrics(y: np.ndarray, s: np.ndarray, recall_target: float) -> dict[str, float]:
    thr = threshold_for_recall(y, s, recall_target)
    c = confusion_at(y, s, thr)
    return {
        "average_precision": float(average_precision_score(y, s)),
        f"precision_at_recall_{recall_target:.2f}": c["precision"],
        f"fpr_at_recall_{recall_target:.2f}": c["fpr"],
    }


def paired_bootstrap(
    y_true: Sequence[int],
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
    recall_target: float = 0.95,
) -> dict[str, Any]:
    """Δ = metric(B) − metric(A) with percentile CIs from paired resampling."""
    y = np.asarray(y_true).astype(int)
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    if not (len(y) == len(a) == len(b)):
        raise ValueError("y_true, scores_a and scores_b must be aligned")
    point_a, point_b = _point_metrics(y, a, recall_target), _point_metrics(y, b, recall_target)
    rng = np.random.default_rng(seed)
    deltas: dict[str, list[float]] = {k: [] for k in point_a}
    n = len(y)
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        ma, mb = (
            _point_metrics(yb, a[idx], recall_target),
            _point_metrics(yb, b[idx], recall_target),
        )
        for k in deltas:
            deltas[k].append(mb[k] - ma[k])
    alpha = (1 - confidence_level) / 2
    out: dict[str, Any] = {
        "n_resamples": n_resamples,
        "confidence_level": confidence_level,
        "n_effective": len(next(iter(deltas.values()))),
        "a": point_a,
        "b": point_b,
    }
    for k, values in deltas.items():
        arr = np.asarray(values)
        out[f"delta_{k}"] = {
            "point": point_b[k] - point_a[k],
            "low": float(np.quantile(arr, alpha)),
            "high": float(np.quantile(arr, 1 - alpha)),
            "share_positive": float((arr > 0).mean()),
        }
    return out


def align_scores(
    ids_a: Sequence[str], scores_a: Sequence[float], ids_b: Sequence[str], scores_b: Sequence[float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inner-join two (image_id → score) tables; returns (ids, a, b) in a common order."""
    da = pd.DataFrame({"image_id": list(ids_a), "a": list(scores_a)})
    db = pd.DataFrame({"image_id": list(ids_b), "b": list(scores_b)})
    m = da.merge(db, on="image_id", how="inner").sort_values("image_id")
    if len(m) != len(da) or len(m) != len(db):
        raise ValueError("the two score tables do not cover the same images")
    return m["image_id"].to_numpy(dtype=str), m["a"].to_numpy(), m["b"].to_numpy()


def recall_by_area(
    y_true: Sequence[int],
    scores: Sequence[float],
    areas: Sequence[int],
    threshold: float,
    bins: Sequence[int],
) -> list[dict[str, Any]]:
    """Recall of the positives per defect-area stratum at a fixed score threshold."""
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=np.float64)
    area = np.asarray(areas)
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:], strict=True):
        sel = (y == 1) & (area >= lo) & (area < hi)
        n = int(sel.sum())
        tp = int((s[sel] >= threshold).sum())
        rows.append(
            {
                "area_bin": f"[{lo}, {hi})",
                "n_positives": n,
                "recall": (tp / n) if n else None,
                "missed": n - tp,
            }
        )
    return rows
