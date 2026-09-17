"""The frozen model as a callable: image (path, bytes or array) → probability + decision.

A *bundle* directory (``inference.bundle_dir``) holds everything the decision depends on:
``model.pt`` (state_dict + architecture metadata), ``preprocessor.json`` (exact preprocessing
description), ``threshold.json`` (frozen operating point and how it was chosen) and
``model_card.json`` (provenance). The same :class:`Predictor` serves the final test evaluation
and the API, so there is no second inference path.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vqi.config import Config
from vqi.data.hashing import sha256_file
from vqi.models.factory import DefectClassifier, build_model
from vqi.preprocessing.pipeline import Preprocessor


class BundleError(RuntimeError):
    """The bundle is incomplete or inconsistent with the running configuration."""


@dataclass(frozen=True)
class Prediction:
    image_id: str
    probability: float
    decision: str
    threshold: float


class Predictor:
    def __init__(
        self,
        model: DefectClassifier,
        pre: Preprocessor,
        threshold: float | None,
        labels: tuple[str, str] = ("REJECT", "ACCEPT"),
        device: str = "cpu",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.model = model.to(device).eval()
        self.pre = pre
        self.threshold = threshold
        self.positive_label, self.negative_label = labels
        self.device = device
        self.metadata = metadata or {}

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_checkpoint(
        cls, cfg: Config, checkpoint: str | Path, threshold: float | None = None
    ) -> Predictor:
        ckpt = torch.load(checkpoint, map_location=cfg.training.device, weights_only=False)
        pre = Preprocessor.from_config(cfg)
        if ckpt.get("preprocessor") != pre.describe():
            raise BundleError("checkpoint preprocessing differs from the current configuration")
        # skeleton only (pretrained=False): every weight comes from the checkpoint, strict
        model = build_model(cfg, 1, 1, pooling=ckpt.get("pooling"), pretrained=False)
        model.load_state_dict(ckpt["model_state"], strict=True)
        labels = (cfg.api.decision_labels.positive, cfg.api.decision_labels.negative)
        meta = {k: v for k, v in ckpt.items() if k != "model_state"}
        meta["checkpoint_sha256"] = sha256_file(checkpoint)
        return cls(model, pre, threshold, labels, cfg.training.device, meta)

    @classmethod
    def from_bundle(cls, cfg: Config, bundle_dir: str | Path | None = None) -> Predictor:
        d = Path(bundle_dir or cfg.inference.bundle_dir)
        for name in ("model.pt", "preprocessor.json", "threshold.json", "model_card.json"):
            if not (d / name).exists():
                raise BundleError(f"bundle file missing: {d / name}")
        pre = Preprocessor.from_config(cfg)
        stored = json.loads((d / "preprocessor.json").read_text(encoding="utf-8"))
        if stored != pre.describe():
            raise BundleError("bundle preprocessing differs from the current configuration")
        thr = json.loads((d / "threshold.json").read_text(encoding="utf-8"))
        if not thr.get("frozen"):
            raise BundleError("threshold.json is not frozen")
        card = json.loads((d / "model_card.json").read_text(encoding="utf-8"))
        ckpt = torch.load(d / "model.pt", map_location=cfg.training.device, weights_only=False)
        # skeleton only (pretrained=False): every weight comes from model.pt, strict
        model = build_model(cfg, 1, 1, pooling=ckpt["pooling"], pretrained=False)
        model.load_state_dict(ckpt["model_state"], strict=True)
        labels = (cfg.api.decision_labels.positive, cfg.api.decision_labels.negative)
        meta = {
            "model_card": card,
            "threshold": thr,
            "bundle_dir": str(d),
            "model_sha256": sha256_file(d / "model.pt"),
        }
        return cls(model, pre, float(thr["threshold"]), labels, cfg.training.device, meta)

    # ------------------------------------------------------------------ scoring
    @torch.no_grad()
    def score_canvases(self, canvases: Sequence[np.ndarray]) -> np.ndarray:
        x = torch.stack([self.pre.to_tensor(c) for c in canvases]).to(self.device)
        return torch.sigmoid(self.model(x).squeeze(1)).cpu().numpy().astype(np.float64)

    def score(self, sources: Iterable[Any], batch_size: int = 16) -> np.ndarray:
        """Probabilities for images given as paths / bytes / arrays, in order."""
        out: list[np.ndarray] = []
        batch: list[np.ndarray] = []
        for src in sources:
            batch.append(self.pre(src))
            if len(batch) == batch_size:
                out.append(self.score_canvases(batch))
                batch = []
        if batch:
            out.append(self.score_canvases(batch))
        return np.concatenate(out) if out else np.zeros(0)

    def decide(self, probability: float) -> str:
        if self.threshold is None:
            raise BundleError("no threshold: the predictor is not frozen")
        return self.positive_label if probability >= self.threshold else self.negative_label

    def predict(self, source: Any, image_id: str = "") -> Prediction:
        p = float(self.score([source], batch_size=1)[0])
        return Prediction(image_id, p, self.decide(p), float(self.threshold))
