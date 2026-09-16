"""The single preprocessing entry point used by training, evaluation and the API.

``Preprocessor.from_config(cfg)`` builds the pipeline the model will be trained *and* served
with (variant ``preprocessing.enhancement.active``); ``variant=`` / ``pad_value=`` overrides exist
only for the Phase 2 experiments.

Pipeline: decode → colour mode → geometry (resize + pad) → enhancement steps → ``uint8`` RGB
canvas. :meth:`Preprocessor.to_tensor` then produces the normalised ``float32`` CHW tensor
expected by a torchvision backbone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from vqi.config import Config
from vqi.preprocessing.enhance import apply_steps, validate_steps
from vqi.preprocessing.geometry import (
    INTERPOLATIONS,
    Geometry,
    apply_geometry,
    apply_geometry_to_mask,
    compute_geometry,
)
from vqi.preprocessing.io import decode_image, to_color_mode


@dataclass(frozen=True)
class Preprocessor:
    height: int
    width: int
    resize_mode: str
    pad_value: str
    interpolation: str
    color_mode: str
    variant: str
    steps: tuple[dict[str, Any], ...] = field(default=())
    mean: tuple[float, ...] = (0.485, 0.456, 0.406)
    std: tuple[float, ...] = (0.229, 0.224, 0.225)

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_config(
        cls, cfg: Config, variant: str | None = None, pad_value: str | None = None
    ) -> Preprocessor:
        pp = cfg.preprocessing
        name = variant or pp.enhancement.active
        if name not in pp.enhancement.variants:
            raise KeyError(
                f"unknown enhancement variant '{name}' (known: {sorted(pp.enhancement.variants)})"
            )
        if pp.interpolation not in INTERPOLATIONS:
            raise ValueError(f"unknown interpolation: {pp.interpolation}")
        return cls(
            height=pp.input_size.height,
            width=pp.input_size.width,
            resize_mode=pp.resize_mode,
            pad_value=pad_value or pp.pad_value,
            interpolation=pp.interpolation,
            color_mode=pp.color_mode,
            variant=name,
            steps=validate_steps(pp.enhancement.variants[name] or []),
            mean=tuple(pp.normalization.mean),
            std=tuple(pp.normalization.std),
        )

    # ------------------------------------------------------------------ image path
    def geometry_for(self, src_h: int, src_w: int) -> Geometry:
        return compute_geometry(src_h, src_w, self.height, self.width, self.resize_mode)

    def normalize_format(self, source: str | Path | bytes | np.ndarray) -> np.ndarray:
        """Decode + colour mode + geometry, **without** enhancement (A.1)."""
        img = source if isinstance(source, np.ndarray) else decode_image(source)
        img = to_color_mode(img, self.color_mode)
        geom = self.geometry_for(img.shape[0], img.shape[1])
        return apply_geometry(img, geom, INTERPOLATIONS[self.interpolation], self.pad_value)

    def enhance(self, canvas_rgb: np.ndarray) -> np.ndarray:
        """Apply this variant's enhancement steps to an already-normalised canvas."""
        return apply_steps(canvas_rgb, self.steps)

    def __call__(self, source: str | Path | bytes | np.ndarray) -> np.ndarray:
        """Full pipeline → ``uint8`` RGB ``(height, width, 3)``."""
        return self.enhance(self.normalize_format(source))

    def transform_mask(self, mask: np.ndarray, src_h: int, src_w: int) -> np.ndarray:
        """Bring a ground-truth mask onto the same canvas as its image (analysis only)."""
        return apply_geometry_to_mask(mask, self.geometry_for(src_h, src_w))

    # ------------------------------------------------------------------ model input
    def to_tensor(self, canvas_rgb: np.ndarray) -> Any:
        """``uint8`` HWC RGB → normalised ``float32`` CHW torch tensor (ImageNet statistics)."""
        import torch  # local import: the classical baseline must not pay for torch

        arr = canvas_rgb.astype(np.float32) / 255.0
        arr = (arr - np.asarray(self.mean, dtype=np.float32)) / np.asarray(
            self.std, dtype=np.float32
        )
        return torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1)))

    # ------------------------------------------------------------------ provenance
    def describe(self) -> dict[str, Any]:
        return {
            "input_size": [self.height, self.width],
            "resize_mode": self.resize_mode,
            "pad_value": self.pad_value,
            "interpolation": self.interpolation,
            "color_mode": self.color_mode,
            "variant": self.variant,
            "steps": list(self.steps),
            "normalization": {"mean": list(self.mean), "std": list(self.std)},
        }
