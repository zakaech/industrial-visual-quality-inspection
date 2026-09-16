"""Training-time augmentations (train split only), applied to the ``uint8`` RGB canvas produced
by the :class:`Preprocessor`, *before* ``to_tensor``.

Only label-preserving, full-image transformations are allowed (risk R10): flips are exact pixel
permutations, photometric jitter is mild, and the optional translation replicates the border so
no defect is cut away. Rotations, crops, erasing, noise and blur are deliberately absent.

Randomness is keyed by ``(seed, epoch, index)`` so that an augmented sample is reproducible
regardless of the number of DataLoader workers or the batch order.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from vqi.config import Config


def rng_for(seed: int, epoch: int, index: int) -> np.random.Generator:
    """Independent, reproducible generator for one (epoch, sample)."""
    return np.random.default_rng(np.random.SeedSequence([int(seed), int(epoch), int(index)]))


@dataclass(frozen=True)
class Augmenter:
    hflip_p: float
    vflip_p: float
    photometric: bool
    brightness_limit: float
    contrast_limit: float
    photometric_p: float
    translation: bool
    max_shift_px: int
    translation_p: float

    @classmethod
    def from_config(cls, cfg: Config) -> Augmenter:
        a = cfg.augmentation
        return cls(
            hflip_p=a.horizontal_flip_p,
            vflip_p=a.vertical_flip_p,
            photometric=a.brightness_contrast.enabled,
            brightness_limit=a.brightness_contrast.brightness_limit,
            contrast_limit=a.brightness_contrast.contrast_limit,
            photometric_p=a.brightness_contrast.p,
            translation=a.translation.enabled,
            max_shift_px=a.translation.max_shift_px,
            translation_p=a.translation.p,
        )

    # ------------------------------------------------------------------ sampling
    def sample(self, rng: np.random.Generator) -> dict[str, float | bool | int]:
        """Draw every random decision once; the same draw is applied to image and mask."""
        draw: dict[str, float | bool | int] = {
            "hflip": bool(rng.random() < self.hflip_p),
            "vflip": bool(rng.random() < self.vflip_p),
            "photometric": bool(self.photometric and rng.random() < self.photometric_p),
            "brightness": 1.0,
            "contrast": 1.0,
            "dx": 0,
            "dy": 0,
        }
        if draw["photometric"]:
            draw["brightness"] = float(
                rng.uniform(1 - self.brightness_limit, 1 + self.brightness_limit)
            )
            draw["contrast"] = float(rng.uniform(1 - self.contrast_limit, 1 + self.contrast_limit))
        if self.translation and rng.random() < self.translation_p:
            draw["dx"] = int(rng.integers(-self.max_shift_px, self.max_shift_px + 1))
            draw["dy"] = int(rng.integers(-self.max_shift_px, self.max_shift_px + 1))
        return draw

    # ------------------------------------------------------------------ application
    @staticmethod
    def _geometric(arr: np.ndarray, draw: dict, border: int) -> np.ndarray:
        if draw["hflip"]:
            arr = arr[:, ::-1]
        if draw["vflip"]:
            arr = arr[::-1]
        if draw["dx"] or draw["dy"]:
            h, w = arr.shape[:2]
            shift = np.float32([[1, 0, draw["dx"]], [0, 1, draw["dy"]]])
            arr = cv2.warpAffine(
                np.ascontiguousarray(arr), shift, (w, h), flags=cv2.INTER_NEAREST, borderMode=border
            )
        return np.ascontiguousarray(arr)

    @staticmethod
    def _photometric(canvas: np.ndarray, draw: dict) -> np.ndarray:
        if not draw["photometric"]:
            return canvas
        img = canvas.astype(np.float32)
        mean = img.mean()
        img = (img - mean) * draw["contrast"] + mean * draw["brightness"]
        return np.clip(img, 0, 255).astype(np.uint8)

    def apply(self, canvas: np.ndarray, draw: dict) -> np.ndarray:
        out = self._geometric(canvas, draw, cv2.BORDER_REPLICATE)
        return self._photometric(out, draw)

    def apply_to_mask(self, mask: np.ndarray, draw: dict) -> np.ndarray:
        """Same geometric draw on a mask (tests and analysis only: the model never sees masks)."""
        return self._geometric(mask, draw, cv2.BORDER_CONSTANT)

    def __call__(self, canvas: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return self.apply(canvas, self.sample(rng))

    def describe(self) -> dict[str, float | bool | int]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}
