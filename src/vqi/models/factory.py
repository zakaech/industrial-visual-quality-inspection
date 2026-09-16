"""ResNet-18 (ImageNet) with a single-logit head, staged freezing and a BatchNorm policy.

Parameter groups (in network order): ``stem`` (conv1 + bn1), ``layer1`` … ``layer4``, ``head``.
A training stage names the groups that are trainable; everything else has ``requires_grad=False``
and, under the ``frozen_eval`` policy, its BatchNorm layers stay in eval mode (running statistics
frozen) so that small batches cannot drift them.

Head bias initialisation (``model.head_bias_init: prior``): the output bias is set to the prior
log-odds of the *training* split,

    logit(p) = log(p / (1 - p)) = log(n_pos / n_neg),

so that the untrained network already predicts the observed defect rate instead of 0.5; this
stabilises the first updates on an imbalanced set and keeps early probabilities meaningful.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any

import torch
from torch import nn
from torchvision import models

from vqi.config import Config

GROUPS: tuple[str, ...] = ("stem", "layer1", "layer2", "layer3", "layer4", "head")


class DefectClassifier(nn.Module):
    """Backbone → global pooling → dropout → linear(1)."""

    def __init__(self, backbone: nn.Module, feat_dim: int, pooling: str, dropout: float) -> None:
        super().__init__()
        if pooling not in ("avg", "max", "avgmax"):
            raise ValueError(f"unknown pooling: {pooling}")
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1, self.layer2 = backbone.layer1, backbone.layer2
        self.layer3, self.layer4 = backbone.layer3, backbone.layer4
        self.pooling = pooling
        self.feat_dim = feat_dim * (2 if pooling == "avgmax" else 1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.feat_dim, 1))

    def pool(self, fmap: torch.Tensor) -> torch.Tensor:
        avg = fmap.mean(dim=(2, 3))
        if self.pooling == "avg":
            return avg
        mx = fmap.amax(dim=(2, 3))
        return mx if self.pooling == "max" else torch.cat([avg, mx], dim=1)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Pooled features ``[B, feat_dim]`` (what the linear probe trains on)."""
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.pool(x)

    def head_forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.head(feats)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head_forward(self.features(x))

    def group(self, name: str) -> nn.Module:
        if name not in GROUPS:
            raise KeyError(f"unknown parameter group '{name}' (known: {GROUPS})")
        return getattr(self, name)


def prior_logit(n_pos: int, n_neg: int) -> float:
    """logit(p) = log(p / (1 − p)) = log(n_pos / n_neg)."""
    if n_pos <= 0 or n_neg <= 0:
        raise ValueError("both classes must be present to compute the prior logit")
    return math.log(n_pos / n_neg)


def build_model(
    cfg: Config, n_pos: int | None = None, n_neg: int | None = None, pooling: str | None = None
) -> DefectClassifier:
    m = cfg.model
    if m.architecture != "resnet18":
        raise NotImplementedError(f"architecture {m.architecture} is not implemented in v1")
    weights = getattr(models.ResNet18_Weights, m.weights) if m.pretrained else None
    backbone = models.resnet18(weights=weights)
    model = DefectClassifier(
        backbone, feat_dim=512, pooling=pooling or m.pooling, dropout=m.dropout
    )
    linear: nn.Linear = model.head[1]
    nn.init.zeros_(linear.weight)  # the probe starts from the prior, not from a random direction
    if m.head_bias_init == "prior":
        if n_pos is None or n_neg is None:
            raise ValueError("head_bias_init='prior' needs the training class counts")
        nn.init.constant_(linear.bias, prior_logit(n_pos, n_neg))
    elif m.head_bias_init == "zero":
        nn.init.zeros_(linear.bias)
    else:
        raise ValueError(f"unknown head_bias_init: {m.head_bias_init}")
    return model


# --------------------------------------------------------------------------------------------- #
# Freezing / BN policy / optimiser groups
# --------------------------------------------------------------------------------------------- #


def set_trainable(model: DefectClassifier, trainable: Sequence[str]) -> dict[str, int]:
    """Enable gradients on ``trainable`` groups only; returns trainable parameter counts."""
    unknown = set(trainable) - set(GROUPS)
    if unknown:
        raise KeyError(f"unknown groups {sorted(unknown)}")
    counts = {}
    for name in GROUPS:
        flag = name in trainable
        n = 0
        for p in model.group(name).parameters():
            p.requires_grad_(flag)
            n += p.numel() if flag else 0
        counts[name] = n
    counts["total_trainable"] = sum(counts[g] for g in GROUPS)
    return counts


def set_train_mode(model: DefectClassifier, trainable: Sequence[str], bn_policy: str) -> None:
    """``model.train()`` for trainable groups; frozen groups in eval mode under ``frozen_eval``
    (BatchNorm running stats untouched, dropout inactive)."""
    model.train()
    if bn_policy == "frozen_eval":
        for name in GROUPS:
            if name not in trainable:
                model.group(name).eval()
    elif bn_policy != "all_train":
        raise ValueError(f"unknown bn_policy: {bn_policy}")


def param_groups(
    model: DefectClassifier, lr_by_group: dict[str, float], weight_decay: float
) -> list[dict[str, Any]]:
    """One optimiser group per trainable network group, with its own learning rate."""
    groups = []
    for name, lr in lr_by_group.items():
        params = [p for p in model.group(name).parameters() if p.requires_grad]
        if params:
            groups.append(
                {"params": params, "lr": float(lr), "weight_decay": weight_decay, "name": name}
            )
    if not groups:
        raise ValueError("no trainable parameters for the given learning rates")
    return groups


def count_parameters(params: Iterable[torch.nn.Parameter]) -> int:
    return sum(p.numel() for p in params)


def model_summary(model: DefectClassifier) -> dict[str, Any]:
    return {
        "architecture": "resnet18",
        "pooling": model.pooling,
        "feat_dim": model.feat_dim,
        "n_params": count_parameters(model.parameters()),
        "n_params_by_group": {g: count_parameters(model.group(g).parameters()) for g in GROUPS},
        "head_bias": float(model.head[1].bias.detach()[0]),
    }
