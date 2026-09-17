"""Operating-point selection on the validation split (decision D5, Phase 4 step 5).

Policy ``min_recall``: the least permissive threshold whose validation recall reaches the target
— it sits exactly on the score of the last positive kept, so a small score drift on new parts
would drop that defect. ``margin: midpoint_logit`` therefore moves the threshold half-way (in
logit space) towards the next lower positive: same validation recall, a safety margin, and a
measured cost in extra false positives. Expected costs for several FN/FP cost ratios are listed
for information only: the real ratio is not known.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from vqi.evaluation.metrics import confusion_at, threshold_for_recall

EPS = 1e-9


def _logit(p: float) -> float:
    p = min(max(p, EPS), 1 - EPS)
    return float(np.log(p / (1 - p)))


def _sigmoid(z: float) -> float:
    return float(1.0 / (1.0 + np.exp(-z)))


def candidate_table(y: np.ndarray, p: np.ndarray, cost_ratios: tuple[float, ...]) -> pd.DataFrame:
    """One row per candidate threshold = each positive's score (descending): what predicting
    positive at ``score >= threshold`` costs and catches. Expected cost per image =
    (FN · ratio + FP) / N for each FN/FP cost ratio."""
    pos_scores = np.sort(np.unique(p[y == 1]))[::-1]
    rows = []
    n = len(y)
    for k, t in enumerate(pos_scores, start=1):
        c = confusion_at(y, p, float(t))
        row = {
            "rank_of_positive": k,
            "threshold": float(t),
            **{m: c[m] for m in ("tp", "fp", "fn", "tn", "recall", "precision", "fpr", "f1", "f2")},
        }
        for r in cost_ratios:
            row[f"expected_cost_ratio_{r:g}"] = (c["fn"] * r + c["fp"]) / n
        rows.append(row)
    return pd.DataFrame(rows)


def select_threshold(
    y: np.ndarray, p: np.ndarray, target_recall: float, margin: str
) -> dict[str, Any]:
    """Apply the policy; returns the chosen threshold and every intermediate quantity."""
    t_min = threshold_for_recall(y, p, target_recall)
    pos_sorted = np.sort(p[y == 1])[::-1]
    lower = pos_sorted[pos_sorted < t_min]
    next_positive = float(lower[0]) if len(lower) else None
    if margin == "none" or next_positive is None:
        t_final = float(t_min)
    elif margin == "midpoint_logit":
        t_final = _sigmoid((_logit(float(t_min)) + _logit(next_positive)) / 2.0)
    else:
        raise ValueError(f"unknown margin policy: {margin}")
    at_min, at_final = confusion_at(y, p, float(t_min)), confusion_at(y, p, t_final)
    return {
        "policy": "min_recall",
        "target_recall": target_recall,
        "margin": margin,
        "threshold_min_recall": float(t_min),
        "next_lower_positive_score": next_positive,
        "threshold": t_final,
        "at_threshold_min_recall": at_min,
        "at_threshold": at_final,
        "margin_cost_extra_fp": int(at_final["fp"] - at_min["fp"]),
        "margin_logit_gap": (
            None if next_positive is None else _logit(float(t_min)) - _logit(next_positive)
        ),
    }


def bootstrap_at_fixed_threshold(
    y: np.ndarray,
    p: np.ndarray,
    threshold: float,
    n_resamples: int,
    confidence_level: float,
    seed: int,
) -> dict[str, dict[str, float]]:
    """CIs of recall / precision / FPR at a *fixed* threshold (the threshold is not re-chosen in
    the resamples, which is the honest view once it is frozen)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    vals: dict[str, list[float]] = {
        "recall": [],
        "precision": [],
        "fpr": [],
        "average_precision": [],
    }
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        yb, pb = y[idx], p[idx]
        if yb.min() == yb.max():
            continue
        c = confusion_at(yb, pb, threshold)
        for k in ("recall", "precision", "fpr"):
            vals[k].append(c[k])
        vals["average_precision"].append(float(average_precision_score(yb, pb)))
    alpha = (1 - confidence_level) / 2
    out = {}
    for k, v in vals.items():
        arr = np.asarray(v)
        out[k] = {
            "low": float(np.quantile(arr, alpha)),
            "high": float(np.quantile(arr, 1 - alpha)),
            "n_effective": int(len(arr)),
        }
    return out
