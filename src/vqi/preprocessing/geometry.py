"""Geometric normalisation to the model canvas.

``keep_aspect_pad`` (decision D7): the image is scaled by a single factor so that it fits inside
the target canvas, then centred and padded. Small defects are never anisotropically squashed; with
KSDD2 (≈637×229 → 640×224) the scale is ≈0.978, i.e. essentially native resolution.

The same :class:`Geometry` is applied to ground-truth masks (nearest-neighbour, zero padding) so
that analysis code can align a mask with a preprocessed image.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

INTERPOLATIONS = {
    "area": cv2.INTER_AREA,
    "linear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC,
    "nearest": cv2.INTER_NEAREST,
}


@dataclass(frozen=True)
class Geometry:
    """How a source image of ``(src_h, src_w)`` lands on the ``(canvas_h, canvas_w)`` canvas."""

    src_h: int
    src_w: int
    canvas_h: int
    canvas_w: int
    scale: float
    resized_h: int
    resized_w: int
    top: int
    left: int

    @property
    def content_box(self) -> tuple[int, int, int, int]:
        """(top, left, bottom, right) of the image content inside the canvas."""
        return self.top, self.left, self.top + self.resized_h, self.left + self.resized_w


def compute_geometry(src_h: int, src_w: int, canvas_h: int, canvas_w: int, mode: str) -> Geometry:
    if mode == "keep_aspect_pad":
        scale = min(canvas_h / src_h, canvas_w / src_w)
        resized_h = max(1, int(round(src_h * scale)))
        resized_w = max(1, int(round(src_w * scale)))
        top = (canvas_h - resized_h) // 2
        left = (canvas_w - resized_w) // 2
        return Geometry(src_h, src_w, canvas_h, canvas_w, scale, resized_h, resized_w, top, left)
    if mode == "stretch":
        return Geometry(src_h, src_w, canvas_h, canvas_w, float("nan"), canvas_h, canvas_w, 0, 0)
    raise ValueError(f"unknown resize_mode: {mode}")


def _pad_fill(img: np.ndarray, pad_value: str) -> int | tuple[int, ...]:
    if pad_value == "zero":
        return 0 if img.ndim == 2 else (0,) * img.shape[2]
    if pad_value == "median":
        if img.ndim == 2:
            return int(np.median(img))
        return tuple(int(np.median(img[:, :, c])) for c in range(img.shape[2]))
    raise ValueError(f"unknown pad_value: {pad_value}")


def apply_geometry(
    img: np.ndarray, geom: Geometry, interpolation: int, pad_value: str
) -> np.ndarray:
    """Resize with ``interpolation`` and place on the canvas, padding with ``pad_value``."""
    resized = cv2.resize(img, (geom.resized_w, geom.resized_h), interpolation=interpolation)
    if (geom.resized_h, geom.resized_w) == (geom.canvas_h, geom.canvas_w):
        return np.ascontiguousarray(resized)
    fill = _pad_fill(resized, pad_value)
    bottom = geom.canvas_h - geom.resized_h - geom.top
    right = geom.canvas_w - geom.resized_w - geom.left
    return cv2.copyMakeBorder(
        resized, geom.top, bottom, geom.left, right, cv2.BORDER_CONSTANT, value=fill
    )


def apply_geometry_to_mask(mask: np.ndarray, geom: Geometry) -> np.ndarray:
    """Masks follow the image with nearest-neighbour interpolation and zero padding."""
    return apply_geometry(mask, geom, cv2.INTER_NEAREST, "zero")
