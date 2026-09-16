"""Loss functions and class-imbalance helpers.

``pos_weight`` rebalances the *gradient* between classes; it is not the business cost ratio
(which enters at the decision threshold, Phase 4). Symbolic values are resolved from the real
training counts: ``ratio`` = n_neg / n_pos, ``sqrt_ratio`` = sqrt(n_neg / n_pos).
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.utils.data import WeightedRandomSampler

from vqi.seed import make_generator


def resolve_pos_weight(value: Any, n_pos: int, n_neg: int) -> float:
    if isinstance(value, str):
        if value == "ratio":
            return n_neg / n_pos
        if value == "sqrt_ratio":
            return math.sqrt(n_neg / n_pos)
        raise ValueError(f"unknown symbolic pos_weight: {value}")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"pos_weight must be a number or ratio/sqrt_ratio, got {value!r}")
    return float(value)


class FocalLoss(nn.Module):
    """Binary focal loss on logits (kept for the config option; not part of the v1 matrix)."""

    def __init__(self, gamma: float, pos_weight: float) -> None:
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        w = self.pos_weight * targets + (1 - targets)
        return (w * (1 - p_t) ** self.gamma * bce).mean()


def build_loss(name: str, pos_weight: float, focal_gamma: float) -> nn.Module:
    if name == "bce_with_logits":
        return nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32))
    if name == "focal":
        return FocalLoss(gamma=focal_gamma, pos_weight=pos_weight)
    raise ValueError(f"unknown loss: {name}")


def build_sampler(name: str, labels: list[float] | Any, seed: int) -> WeightedRandomSampler | None:
    """``weighted_random``: each sample drawn with probability ∝ 1 / n_class → batches ≈ 50/50,
    the epoch keeps its nominal length (draws with replacement)."""
    if name == "none":
        return None
    if name != "weighted_random":
        raise ValueError(f"unknown sampler: {name}")
    y = torch.as_tensor(labels, dtype=torch.float32)
    n_pos = float((y == 1).sum())
    n_neg = float(len(y) - n_pos)
    weights = torch.where(y == 1, 1.0 / n_pos, 1.0 / n_neg)
    return WeightedRandomSampler(
        weights, num_samples=len(y), replacement=True, generator=make_generator(seed)
    )
