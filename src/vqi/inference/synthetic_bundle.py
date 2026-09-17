"""A *synthetic* model bundle for tests and CI: untrained weights, threshold 0.5.

It has the exact layout ``Predictor.from_bundle`` expects (``model.pt``, ``preprocessor.json``,
``threshold.json``, ``model_card.json``) so the whole API path can be exercised without the
real frozen model, the dataset, or any download (``pretrained=False``: no ImageNet weights).
Its ``model_version`` starts with ``SYN-`` and its card says so; it must never be served as a
real model.

    python -m vqi.inference.synthetic_bundle --out DIR [--config configs/default.yaml]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from vqi.config import Config, load_config
from vqi.models.factory import build_model
from vqi.preprocessing import Preprocessor

SYNTHETIC_THRESHOLD = 0.5
SYNTHETIC_FROZEN_AT = "2026-01-01T00:00:00+00:00"


def make_synthetic_bundle(cfg: Config, out_dir: str | Path, seed: int = 0) -> Path:
    """Write the four bundle files plus ``sample.png`` (a synthetic image for smoke tests)."""
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    model = build_model(cfg, 197, 1668, pretrained=False)
    torch.nn.init.normal_(model.head[1].weight, std=0.05)  # non-degenerate scores
    torch.save(
        {"model_state": model.state_dict(), "pooling": model.pooling, "stage": "synthetic"},
        d / "model.pt",
    )
    pre = Preprocessor.from_config(cfg)
    (d / "preprocessor.json").write_text(json.dumps(pre.describe(), indent=2), encoding="utf-8")
    (d / "threshold.json").write_text(
        json.dumps(
            {
                "frozen": True,
                "threshold": SYNTHETIC_THRESHOLD,
                "policy": "min_recall",
                "target_recall": 0.95,
                "margin": "midpoint_logit",
                "selected_on": "synthetic",
                "calibration_decision": "raw",
                "frozen_at": SYNTHETIC_FROZEN_AT,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (d / "model_card.json").write_text(
        json.dumps(
            {
                "model": "synthetic (untrained weights) — tests and CI only, never a real model",
                "experiment_id": "SYN",
                "frozen_at": SYNTHETIC_FROZEN_AT,
                "architecture": {
                    "architecture": cfg.model.architecture,
                    "pooling": model.pooling,
                    "n_params": sum(p.numel() for p in model.parameters()),
                },
                "validation_metrics": {
                    "average_precision": None,
                    "roc_auc": None,
                    "at_recall_targets": [],
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    rng = np.random.default_rng(seed)
    img = rng.integers(30, 60, size=(637, 229, 3), dtype=np.uint8)  # KSDD2-like shape
    cv2.imwrite(str(d / "sample.png"), img)
    return d


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    d = make_synthetic_bundle(load_config(args.config), args.out, seed=args.seed)
    print(f"synthetic bundle written to {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
