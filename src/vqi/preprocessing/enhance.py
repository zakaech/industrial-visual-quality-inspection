"""Enhancement operators (OpenCV), each a pure function ``RGB uint8 -> RGB uint8``.

Contrast operators (equalisation, CLAHE, unsharp, Laplacian sharpening) act on the luminance
channel of the LAB representation so that the small colour information of the images is kept.
Smoothing operators (Gaussian, median) act on all three channels.

Operators are registered by name in :data:`OPERATORS`; a preprocessing variant in the config is an
ordered list of ``{"op": name, **params}`` steps applied by :func:`apply_steps`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import cv2
import numpy as np


def _on_luminance(img_rgb: np.ndarray, fn: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    lum = fn(lab[:, :, 0])
    lab[:, :, 0] = np.clip(lum, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def equalize_hist(img_rgb: np.ndarray) -> np.ndarray:
    """Global histogram equalisation of the luminance."""
    return _on_luminance(img_rgb, cv2.equalizeHist)


def clahe(
    img_rgb: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Sequence[int] = (8, 8)
) -> np.ndarray:
    """Contrast-limited adaptive histogram equalisation of the luminance."""
    op = cv2.createCLAHE(
        clipLimit=float(clip_limit), tileGridSize=tuple(int(t) for t in tile_grid_size)
    )
    return _on_luminance(img_rgb, op.apply)


def gaussian_blur(img_rgb: np.ndarray, ksize: int = 3, sigma: float = 0.8) -> np.ndarray:
    """Light linear smoothing (all channels)."""
    k = int(ksize)
    if k % 2 == 0:
        raise ValueError("ksize must be odd")
    return cv2.GaussianBlur(img_rgb, (k, k), sigmaX=float(sigma), sigmaY=float(sigma))


def median_blur(img_rgb: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Non-linear smoothing that removes isolated pixels — and may erase pixel-sized defects."""
    k = int(ksize)
    if k % 2 == 0:
        raise ValueError("ksize must be odd")
    return cv2.medianBlur(img_rgb, k)


def unsharp_mask(img_rgb: np.ndarray, sigma: float = 1.5, amount: float = 1.0) -> np.ndarray:
    """Luminance + amount × (luminance − Gaussian(luminance)): boosts local contrast."""

    def fn(lum: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(lum, (0, 0), sigmaX=float(sigma))
        return lum.astype(np.float32) + float(amount) * (
            lum.astype(np.float32) - blurred.astype(np.float32)
        )

    return _on_luminance(img_rgb, fn)


def laplacian_sharpen(img_rgb: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """Luminance − strength × Laplacian(luminance): emphasises edges, amplifies noise."""

    def fn(lum: np.ndarray) -> np.ndarray:
        lap = cv2.Laplacian(lum, cv2.CV_32F, ksize=3)
        return lum.astype(np.float32) - float(strength) * lap

    return _on_luminance(img_rgb, fn)


OPERATORS: dict[str, Callable[..., np.ndarray]] = {
    "equalize_hist": equalize_hist,
    "clahe": clahe,
    "gaussian_blur": gaussian_blur,
    "median_blur": median_blur,
    "unsharp_mask": unsharp_mask,
    "laplacian_sharpen": laplacian_sharpen,
}


def validate_steps(steps: Sequence[dict[str, Any]]) -> tuple[dict[str, Any], ...]:
    """Check that every step names a known operator; returns an immutable copy."""
    out = []
    for i, step in enumerate(steps):
        if not isinstance(step, dict) or "op" not in step:
            raise ValueError(f"step {i} must be a mapping with an 'op' key, got {step!r}")
        if step["op"] not in OPERATORS:
            raise ValueError(
                f"step {i}: unknown operator '{step['op']}' (known: {sorted(OPERATORS)})"
            )
        out.append(dict(step))
    return tuple(out)


def apply_steps(img_rgb: np.ndarray, steps: Sequence[dict[str, Any]]) -> np.ndarray:
    for step in steps:
        params = {k: v for k, v in step.items() if k != "op"}
        img_rgb = OPERATORS[step["op"]](img_rgb, **params)
    return img_rgb
