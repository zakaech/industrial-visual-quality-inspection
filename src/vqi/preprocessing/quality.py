"""Objective effect of an enhancement on defect visibility (analysis only, never training).

Contrast-to-noise ratio of a defect, from its ground-truth mask:

    CNR = |mean(L[defect]) − mean(L[ring])| / std(L[ring])

where ``L`` is the luminance of the preprocessed image and ``ring`` is a band of ``ring_width``
pixels around the defect. A transformation that makes images "look better" but lowers the CNR of
small defects is rejected on this number, not on taste. Also reported: the fraction of saturated
pixels (0 or 255), which flags clipping artefacts.
"""

from __future__ import annotations

import cv2
import numpy as np


def defect_ring(mask_bool: np.ndarray, ring_width: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ring_width + 1, 2 * ring_width + 1))
    dilated = cv2.dilate(mask_bool.astype(np.uint8), kernel) > 0
    return dilated & ~mask_bool


def defect_cnr(gray: np.ndarray, mask: np.ndarray, ring_width: int = 6) -> float:
    """CNR of the defect region against its surrounding ring; ``nan`` if either set is empty."""
    defect = mask > 0
    if not defect.any():
        return float("nan")
    ring = defect_ring(defect, ring_width)
    if ring.sum() < 8:
        return float("nan")
    g = gray.astype(np.float32)
    inside, around = g[defect], g[ring]
    noise = float(around.std())
    if noise < 1e-6:
        return float("nan")
    return float(abs(inside.mean() - around.mean()) / noise)


def saturated_fraction(gray: np.ndarray) -> float:
    return float(((gray == 0) | (gray == 255)).mean())


def image_stats(gray: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(gray.mean()),
        "std": float(gray.std()),
        "saturated_fraction": saturated_fraction(gray),
    }
