"""Probability calibration diagnostics and (optional) post-hoc calibrators.

The model was trained with ``pos_weight`` = 8.47, so its sigmoid outputs are *not* guaranteed to be
calibrated probabilities. Everything here is fitted and measured on the **validation** split
only; the test split never enters. Calibrators are monotone, so they change the probability
scale but not the ranking — the operating point chosen at a recall target is unaffected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

EPS = 1e-7


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


# --------------------------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------------------------- #


def reliability_table(y: np.ndarray, p: np.ndarray, n_bins: int, adaptive: bool) -> list[dict]:
    """Per-bin mean predicted probability, observed frequency and count.
    ``adaptive=True`` uses equal-mass bins (quantiles), better suited to scores concentrated
    near 0 as here (prevalence ≈ 0.1)."""
    if adaptive:
        edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    else:
        edges = np.linspace(0, 1, n_bins + 1)
    edges[-1] = max(edges[-1], 1.0)
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        sel = idx == b
        n = int(sel.sum())
        rows.append(
            {
                "bin": b,
                "lower": float(edges[b]),
                "upper": float(edges[b + 1]),
                "n": n,
                "mean_predicted": float(p[sel].mean()) if n else None,
                "observed_frequency": float(y[sel].mean()) if n else None,
            }
        )
    return rows


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int, adaptive: bool) -> float:
    rows = reliability_table(y, p, n_bins, adaptive)
    total = sum(r["n"] for r in rows)
    return float(
        sum(
            r["n"] / total * abs(r["mean_predicted"] - r["observed_frequency"])
            for r in rows
            if r["n"]
        )
    )


def brier_score(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def negative_log_likelihood(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def calibration_metrics(y: np.ndarray, p: np.ndarray, n_bins: int) -> dict[str, float]:
    return {
        "ece_equal_width": expected_calibration_error(y, p, n_bins, adaptive=False),
        "ece_equal_mass": expected_calibration_error(y, p, n_bins, adaptive=True),
        "brier": brier_score(y, p),
        "nll": negative_log_likelihood(y, p),
        "mean_predicted": float(p.mean()),
        "prevalence": float(y.mean()),
    }


# --------------------------------------------------------------------------------------------- #
# Calibrators (monotone)
# --------------------------------------------------------------------------------------------- #


@dataclass
class TemperatureScaling:
    """p' = sigmoid(logit(p) / T); T fitted by minimising the NLL (1 parameter)."""

    temperature: float = 1.0

    def fit(self, y: np.ndarray, p: np.ndarray) -> TemperatureScaling:
        z = torch.tensor(_logit(p), dtype=torch.float64)
        t = torch.tensor(y, dtype=torch.float64)
        log_temp = torch.zeros(1, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([log_temp], lr=0.1, max_iter=200, line_search_fn="strong_wolfe")

        def closure() -> torch.Tensor:
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(z / log_temp.exp(), t)
            loss.backward()
            return loss

        opt.step(closure)
        self.temperature = float(log_temp.detach().exp())
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        return _sigmoid(_logit(p) / self.temperature)

    def params(self) -> dict[str, float]:
        return {"temperature": self.temperature}


@dataclass
class PlattScaling:
    """p' = sigmoid(a · logit(p) + b) — logistic regression on the logit (2 parameters)."""

    a: float = 1.0
    b: float = 0.0

    def fit(self, y: np.ndarray, p: np.ndarray) -> PlattScaling:
        lr = LogisticRegression(C=1e6, solver="lbfgs").fit(_logit(p).reshape(-1, 1), y)
        self.a, self.b = float(lr.coef_[0, 0]), float(lr.intercept_[0])
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        return _sigmoid(self.a * _logit(p) + self.b)

    def params(self) -> dict[str, float]:
        return {"a": self.a, "b": self.b}


@dataclass
class Isotonic:
    """Non-parametric monotone map (PAV); flexible, needs more data than we have."""

    model: Any = None

    def fit(self, y: np.ndarray, p: np.ndarray) -> Isotonic:
        self.model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p, y)
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(p), dtype=np.float64)

    def params(self) -> dict[str, Any]:
        return {"n_knots": int(len(self.model.X_thresholds_))}


CALIBRATORS = {"temperature": TemperatureScaling, "platt": PlattScaling, "isotonic": Isotonic}


def cross_validated_calibration(
    y: np.ndarray, p: np.ndarray, method: str, n_bins: int, n_folds: int = 5, seed: int = 42
) -> dict[str, Any]:
    """Out-of-fold calibrated probabilities on the validation split itself (stratified k-fold),
    so that the benefit of a calibrator is measured on samples it was not fitted on."""
    oof = np.zeros_like(p)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for tr, va in skf.split(p, y):
        cal = CALIBRATORS[method]().fit(y[tr], p[tr])
        oof[va] = cal.transform(p[va])
    full = CALIBRATORS[method]().fit(y, p)
    return {
        "method": method,
        "out_of_fold": calibration_metrics(y, oof, n_bins),
        "in_sample": calibration_metrics(y, full.transform(p), n_bins),
        "params_full_fit": full.params(),
        "oof_probabilities": oof,
    }
