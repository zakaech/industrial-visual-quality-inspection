"""Hand-crafted image descriptors, computed image by image (no dataset-level statistics, so no
leakage is possible at this stage; the only fitted object downstream is the scaler).

Groups (each with its own justification, see docs/02_phase2_preprocessing_baseline.md):

* ``contrast_only`` — the standard deviation of the luminance alone: the trivial baseline the EDA
  exposed (ROC-AUC ≈ 0.72 on the official train set). Everything else must beat it.
* ``global_stats`` — moments, percentiles and channel spread of the whole image.
* ``histogram`` — normalised luminance histogram.
* ``lbp`` — rotation-invariant uniform Local Binary Patterns at one or more (P, R) scales:
  micro-texture, invariant to monotonic illumination changes. Implemented with OpenCV/NumPy
  (bilinear sampling on a circle, uniform-pattern lookup table); no scikit-image dependency.
* ``local_stats`` — a grid of patches; per-patch contrast and Laplacian energy aggregated by max
  and high percentiles. Defects are small and local (median 1.6 % of the image), so a global
  histogram dilutes them; "is there *somewhere* an abnormal patch?" is the right question.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np

from vqi.config import Config
from vqi.preprocessing.io import to_gray

# --------------------------------------------------------------------------------------------- #
# Uniform, rotation-invariant LBP
# --------------------------------------------------------------------------------------------- #


@lru_cache(maxsize=8)
def uniform_lut(points: int) -> np.ndarray:
    """Map every ``points``-bit code to a rotation-invariant uniform bin.

    Uniform patterns (at most two 0/1 transitions on the circle) are binned by their number of
    ones (0..P); all non-uniform patterns share bin P+1. Hence ``P + 2`` bins.
    """
    codes = np.arange(2**points, dtype=np.int64)
    rotated = ((codes << 1) | (codes >> (points - 1))) & ((1 << points) - 1)
    transitions = np.zeros_like(codes)
    diff = codes ^ rotated
    for b in range(points):
        transitions += (diff >> b) & 1
    ones = np.zeros_like(codes)
    for b in range(points):
        ones += (codes >> b) & 1
    lut = np.where(transitions <= 2, ones, points + 1)
    return lut.astype(np.int32)


def lbp_codes(gray: np.ndarray, points: int, radius: int) -> np.ndarray:
    """Raw LBP code image (int32) with circular neighbours sampled by bilinear interpolation.
    The ``radius``-wide border is undefined and must be cropped by the caller."""
    g = gray.astype(np.float32)
    h, w = g.shape
    code = np.zeros((h, w), dtype=np.int32)
    for p in range(points):
        theta = 2.0 * np.pi * p / points
        dx, dy = radius * np.cos(theta), -radius * np.sin(theta)
        shift = np.float32([[1, 0, dx], [0, 1, dy]])
        neighbour = cv2.warpAffine(
            g,
            shift,
            (w, h),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REPLICATE,
        )
        code |= (neighbour >= g).astype(np.int32) << p
    return code


def lbp_histogram(gray: np.ndarray, points: int, radius: int) -> np.ndarray:
    """Normalised histogram (``P + 2`` bins) of uniform rotation-invariant LBP codes."""
    code = lbp_codes(gray, points, radius)
    r = int(radius)
    code = code[r:-r, r:-r] if r > 0 else code
    bins = uniform_lut(points)[code]
    hist = np.bincount(bins.ravel(), minlength=points + 2).astype(np.float64)
    return hist / max(hist.sum(), 1.0)


# --------------------------------------------------------------------------------------------- #
# Other groups
# --------------------------------------------------------------------------------------------- #

PERCENTILES = (1, 5, 25, 50, 75, 95, 99)


def contrast_only(gray: np.ndarray) -> np.ndarray:
    return np.array([gray.std()], dtype=np.float64)


def global_stats(img_rgb: np.ndarray, gray: np.ndarray) -> np.ndarray:
    g = gray.astype(np.float64)
    mean, std = g.mean(), g.std()
    centred = g - mean
    skew = float((centred**3).mean() / (std**3 + 1e-12))
    kurt = float((centred**4).mean() / (std**4 + 1e-12) - 3.0)
    ch = img_rgb.astype(np.int16)
    spread = (
        np.abs(ch[:, :, 0] - ch[:, :, 1]).mean()
        + np.abs(ch[:, :, 1] - ch[:, :, 2]).mean()
        + np.abs(ch[:, :, 0] - ch[:, :, 2]).mean()
    ) / 3.0
    pct = np.percentile(g, PERCENTILES)
    return np.concatenate([[mean, std, g.min(), g.max()], pct, [skew, kurt, float(spread)]]).astype(
        np.float64
    )


def global_stats_names() -> list[str]:
    return (
        ["mean", "std", "min", "max"]
        + [f"p{p}" for p in PERCENTILES]
        + ["skewness", "kurtosis", "channel_spread"]
    )


def histogram(gray: np.ndarray, bins: int) -> np.ndarray:
    hist = cv2.calcHist([gray], [0], None, [bins], [0, 256]).ravel().astype(np.float64)
    return hist / max(hist.sum(), 1.0)


def local_stats(gray: np.ndarray, patch_size: int) -> np.ndarray:
    """Per-patch std and Laplacian variance on a non-overlapping grid, aggregated over patches."""
    g = gray.astype(np.float32)
    lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
    h, w = g.shape
    ph, pw = h // patch_size, w // patch_size
    g = g[: ph * patch_size, : pw * patch_size].reshape(ph, patch_size, pw, patch_size)
    lap = lap[: ph * patch_size, : pw * patch_size].reshape(ph, patch_size, pw, patch_size)
    patch_std = g.std(axis=(1, 3)).ravel()
    patch_lapvar = lap.var(axis=(1, 3)).ravel()
    out = []
    for values in (patch_std, patch_lapvar):
        out += [values.max(), np.percentile(values, 99), values.std(), values.mean()]
    return np.asarray(out, dtype=np.float64)


def local_stats_names() -> list[str]:
    return [
        f"patch_{stat}_{agg}" for stat in ("std", "lapvar") for agg in ("max", "p99", "std", "mean")
    ]


# --------------------------------------------------------------------------------------------- #
# Extractor
# --------------------------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FeatureExtractor:
    histogram_bins: int
    lbp_scales: tuple[tuple[int, int], ...]
    patch_size: int

    @classmethod
    def from_config(cls, cfg: Config) -> FeatureExtractor:
        b = cfg.baseline
        return cls(
            histogram_bins=b.histogram_bins,
            lbp_scales=tuple((s.points, s.radius) for s in b.lbp),
            patch_size=b.local_stats.patch_size,
        )

    # every extractor name -> (feature names, function(img_rgb, gray) -> vector)
    def extractor_names(self) -> dict[str, list[str]]:
        return {
            "contrast_only": ["contrast_std"],
            "global_stats": global_stats_names(),
            "histogram": [f"hist_{i}" for i in range(self.histogram_bins)],
            "lbp": [f"lbp_p{p}_r{r}_{b}" for p, r in self.lbp_scales for b in range(p + 2)],
            "local_stats": local_stats_names(),
        }

    def extract_one(self, name: str, img_rgb: np.ndarray, gray: np.ndarray) -> np.ndarray:
        if name == "contrast_only":
            return contrast_only(gray)
        if name == "global_stats":
            return global_stats(img_rgb, gray)
        if name == "histogram":
            return histogram(gray, self.histogram_bins)
        if name == "lbp":
            return np.concatenate([lbp_histogram(gray, p, r) for p, r in self.lbp_scales])
        if name == "local_stats":
            return local_stats(gray, self.patch_size)
        raise KeyError(f"unknown extractor: {name}")

    def extract(self, img_rgb: np.ndarray, extractors: Sequence[str]) -> np.ndarray:
        """Concatenate the requested extractors' features for one preprocessed RGB canvas."""
        gray = to_gray(img_rgb)
        parts = [self.extract_one(name, img_rgb, gray) for name in extractors]
        vec = np.concatenate(parts).astype(np.float64)
        if not np.isfinite(vec).all():
            raise ValueError("non-finite feature value")
        return vec

    def names(self, extractors: Sequence[str]) -> list[str]:
        table = self.extractor_names()
        return [n for name in extractors for n in table[name]]

    def all_extractors(self) -> list[str]:
        return list(self.extractor_names())
