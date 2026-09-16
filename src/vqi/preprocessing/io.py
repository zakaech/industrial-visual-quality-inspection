"""Image decoding and format normalisation.

Every image — from disk during training or from an HTTP upload at inference — goes through
:func:`decode_image`, which always uses ``cv2.imdecode`` on a byte buffer so that both sources
follow byte-for-byte the same path. Output contract: ``uint8``, ``HxWx3``, RGB, C-contiguous.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class ImageDecodeError(ValueError):
    """The buffer is not a decodable image or has an unsupported layout."""


def read_bytes(path: str | Path) -> bytes:
    return Path(path).read_bytes()


def decode_image(source: str | Path | bytes | bytearray | memoryview) -> np.ndarray:
    """Decode an image file or buffer into an RGB ``uint8`` array of shape ``(H, W, 3)``.

    Accepts 8-bit and 16-bit PNG/JPEG/BMP with 1, 3 or 4 channels. 16-bit inputs are scaled to
    8-bit (``value // 257``); alpha is dropped; grayscale is replicated on three channels.
    """
    if isinstance(source, (str, Path)):
        buffer = np.frombuffer(read_bytes(source), dtype=np.uint8)
    else:
        buffer = np.frombuffer(bytes(source), dtype=np.uint8)
    if buffer.size == 0:
        raise ImageDecodeError("empty buffer")

    img = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ImageDecodeError("cv2.imdecode returned None (not an image or unsupported codec)")

    if img.dtype == np.uint16:
        img = (img.astype(np.uint32) // 257).astype(np.uint8)
    elif img.dtype != np.uint8:
        raise ImageDecodeError(f"unsupported dtype {img.dtype}")

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.ndim == 3 and img.shape[2] == 1:
        img = cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2RGB)
    elif img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        raise ImageDecodeError(f"unsupported image layout {img.shape}")
    return np.ascontiguousarray(img)


def to_color_mode(img_rgb: np.ndarray, color_mode: str) -> np.ndarray:
    """``rgb`` keeps the image; ``grayscale`` collapses to luminance replicated on 3 channels
    (so downstream code always sees 3 channels)."""
    if color_mode == "rgb":
        return img_rgb
    if color_mode == "grayscale":
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    raise ValueError(f"unknown color_mode: {color_mode}")


def to_gray(img_rgb: np.ndarray) -> np.ndarray:
    """Luminance (``uint8``, ``HxW``) of an RGB image."""
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
