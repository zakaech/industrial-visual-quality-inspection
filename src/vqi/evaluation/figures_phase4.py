"""Figures of the final evaluation (validation analysis and test pass). Pure plotting."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from vqi.evaluation.metrics import confusion_at  # noqa: E402


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def score_distribution(
    y: np.ndarray, p: np.ndarray, threshold: float | None, path: Path, title: str
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    bins = np.linspace(0, 1, 41)
    for ax, log in zip(axes, (False, True), strict=True):
        ax.hist(
            p[y == 0],
            bins=bins,
            alpha=0.6,
            label=f"saines (n={int((y == 0).sum())})",
            color="#4c72b0",
        )
        ax.hist(
            p[y == 1],
            bins=bins,
            alpha=0.7,
            label=f"défectueuses (n={int(y.sum())})",
            color="#dd8452",
        )
        if threshold is not None:
            ax.axvline(
                threshold,
                color="black",
                linestyle="--",
                linewidth=1,
                label=f"seuil {threshold:.4f}",
            )
        if log:
            ax.set_yscale("log")
            ax.set_title("échelle log", fontsize=9)
        ax.set_xlabel("probabilité prédite")
        ax.legend(fontsize=7)
    fig.suptitle(title, fontsize=10)
    return _save(fig, path)


def pr_roc_curves(
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    threshold_points: dict[str, tuple[float, float]],
    path: Path,
    title: str,
    prevalence: float,
) -> Path:
    """``curves``: name → (y, p). ``threshold_points``: name → (recall, precision) marker."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for name, (y, p) in curves.items():
        pr_p, pr_r, _ = precision_recall_curve(y, p)
        fpr, tpr, _ = roc_curve(y, p)
        axes[0].plot(pr_r, pr_p, linewidth=1.3, label=name)
        axes[1].plot(fpr, tpr, linewidth=1.3, label=name)
    for name, (r, pr) in threshold_points.items():
        axes[0].scatter([r], [pr], marker="x", s=60, color="black", zorder=5, label=name)
    axes[0].axhline(prevalence, color="gray", linestyle="--", linewidth=0.8)
    axes[0].set_xlabel("rappel")
    axes[0].set_ylabel("précision")
    axes[0].set_title("précision–rappel", fontsize=9)
    axes[1].plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=0.8)
    axes[1].set_xlabel("FPR")
    axes[1].set_ylabel("rappel (TPR)")
    axes[1].set_title("ROC", fontsize=9)
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.legend(fontsize=7)
    fig.suptitle(title, fontsize=10)
    return _save(fig, path)


def metrics_vs_threshold(
    y: np.ndarray, p: np.ndarray, threshold: float | None, path: Path, title: str
) -> Path:
    ts = np.unique(np.concatenate([np.logspace(-4, 0, 400), p]))
    rows = [confusion_at(y, p, float(t)) for t in ts]
    rec = np.array([r["recall"] for r in rows])
    prec = np.array([r["precision"] for r in rows])
    fpr = np.array([r["fpr"] for r in rows])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(ts, rec, label="rappel", color="#dd8452")
    axes[0].plot(ts, prec, label="précision", color="#4c72b0")
    axes[0].set_ylabel("rappel / précision")
    axes[1].plot(ts, rec, label="rappel", color="#dd8452")
    axes[1].plot(ts, fpr, label="FPR", color="#55a868")
    axes[1].set_ylabel("rappel / FPR")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("seuil (échelle log)")
        ax.axhline(0.95, color="gray", linestyle=":", linewidth=0.8)
        if threshold is not None:
            ax.axvline(
                threshold,
                color="black",
                linestyle="--",
                linewidth=1,
                label=f"seuil {threshold:.4f}",
            )
        ax.legend(fontsize=8)
    fig.suptitle(title, fontsize=10)
    return _save(fig, path)


def reliability_diagram(tables: dict[str, list[dict[str, Any]]], path: Path, title: str) -> Path:
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.plot(
        [0, 1], [0, 1], color="gray", linestyle="--", linewidth=0.8, label="parfaitement calibré"
    )
    for name, rows in tables.items():
        xs = [r["mean_predicted"] for r in rows if r["n"]]
        ys = [r["observed_frequency"] for r in rows if r["n"]]
        ns = [r["n"] for r in rows if r["n"]]
        ax.plot(xs, ys, marker="o", linewidth=1, markersize=4, label=name)
        for x, yv, n in zip(xs, ys, ns, strict=True):
            ax.annotate(str(n), (x, yv), fontsize=6, textcoords="offset points", xytext=(3, 3))
    ax.set_xlabel("probabilité prédite (moyenne du bin)")
    ax.set_ylabel("fréquence observée de défauts")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7)
    return _save(fig, path)


def recall_by_area_bars(rows: Sequence[dict[str, Any]], path: Path, title: str) -> Path:
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    labels = [r["area_bin"] for r in rows]
    recalls = [r["recall"] if r["recall"] is not None else 0 for r in rows]
    ax.bar(range(len(rows)), recalls, color="#dd8452")
    for i, r in enumerate(rows):
        ax.text(
            i,
            (r["recall"] or 0) + 0.02,
            f"{r['n_positives'] - r['missed']}/{r['n_positives']}",
            ha="center",
            fontsize=8,
        )
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=15, fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("rappel au seuil final")
    ax.set_title(title, fontsize=10)
    return _save(fig, path)
