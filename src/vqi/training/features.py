"""Frozen-backbone features for the linear-probe stage (stage A).

With the backbone frozen and no augmentation, the pooled 512-d features of every image are a
deterministic function of the image, so they are computed once per (preprocessor, weights,
pooling, split) and cached. The probe then trains in seconds.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from vqi.config import Config
from vqi.data.splits import read_splits_meta
from vqi.logging_utils import get_logger, log_event
from vqi.models.factory import DefectClassifier
from vqi.preprocessing.pipeline import Preprocessor

logger = get_logger(__name__)


@dataclass(frozen=True)
class FrozenFeatures:
    image_ids: np.ndarray
    x: np.ndarray  # float32 [N, feat_dim]
    labels: np.ndarray  # float32 [N]
    seconds: float
    cached: bool


def features_path(cfg: Config, pre: Preprocessor, model: DefectClassifier, split: str) -> Path:
    key = hashlib.sha256(
        json.dumps(
            {
                "preprocessor": pre.describe(),
                "weights": cfg.model.weights,
                "architecture": cfg.model.architecture,
                "pooling": model.pooling,
                "splits_sha256": read_splits_meta(cfg)["splits_sha256"],
                "split": split,
            },
            sort_keys=True,
            default=str,
        ).encode()
    ).hexdigest()[:16]
    d = cfg.paths.processed_dir / "features_cnn"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{split}__{cfg.model.architecture}__{model.pooling}__{key}.npz"


@torch.no_grad()
def extract_frozen_features(
    model: DefectClassifier, loader: DataLoader, device: str = "cpu"
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    model.eval()
    t0 = time.perf_counter()
    ids, feats, labels = [], [], []
    for x, y, image_ids in loader:
        feats.append(model.features(x.to(device)).cpu().numpy().astype(np.float32))
        labels.append(y.numpy().astype(np.float32))
        ids.extend(image_ids)
    return (
        np.asarray(ids, dtype=str),
        np.concatenate(feats),
        np.concatenate(labels),
        time.perf_counter() - t0,
    )


def frozen_features(
    cfg: Config,
    pre: Preprocessor,
    model: DefectClassifier,
    split: str,
    loader: DataLoader,
    expected_ids: list[str],
    use_cache: bool = True,
) -> FrozenFeatures:
    path = features_path(cfg, pre, model, split)
    if use_cache and path.exists():
        z = np.load(path, allow_pickle=False)
        if list(z["image_ids"]) == expected_ids:
            return FrozenFeatures(z["image_ids"], z["x"], z["labels"], float(z["seconds"]), True)
    ids, x, labels, seconds = extract_frozen_features(model, loader, cfg.training.device)
    if list(ids) != expected_ids:
        raise RuntimeError("feature order does not match the split rows (loader must not shuffle)")
    np.savez(path, image_ids=ids, x=x, labels=labels, seconds=seconds)
    log_event(
        logger,
        "features.frozen",
        split=split,
        n=len(ids),
        dim=x.shape[1],
        seconds=round(seconds, 1),
    )
    return FrozenFeatures(ids, x, labels, seconds, False)
