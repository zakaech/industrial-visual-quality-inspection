"""Content hashes used for integrity and duplicate detection.

* :func:`sha256_file` — exact identity of a file (integrity of the archive, pixel-identical
  duplicates).
* :func:`phash` — perceptual hash (Zauner, 2010): the image is reduced to a small grayscale
  square, transformed with a DCT, and the sign of the low-frequency coefficients relative to
  their median gives a 64-bit fingerprint. Two images that differ only by noise, compression or
  a small shift have a small Hamming distance; unrelated images are ~32 bits apart.
  Implemented with OpenCV only (no extra dependency).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np

_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Hex SHA-256 digest of a file, streamed in chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def phash_bits(gray: np.ndarray, hash_size: int = 8, highfreq_factor: int = 4) -> np.ndarray:
    """Perceptual hash of a grayscale image as a flat boolean array of ``hash_size**2`` bits."""
    if gray.ndim != 2:
        raise ValueError("phash expects a single-channel image")
    side = hash_size * highfreq_factor
    small = cv2.resize(gray, (side, side), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(small)
    low = dct[:hash_size, :hash_size]
    median = np.median(low)
    return (low > median).reshape(-1)


def bits_to_hex(bits: np.ndarray) -> str:
    """Pack a boolean bit array into a hex string (left-padded, fixed width)."""
    packed = np.packbits(bits.astype(np.uint8))
    return packed.tobytes().hex()


def hex_to_bytes(hex_str: str) -> np.ndarray:
    return np.frombuffer(bytes.fromhex(hex_str), dtype=np.uint8)


def phash_hex(gray: np.ndarray, hash_size: int = 8, highfreq_factor: int = 4) -> str:
    return bits_to_hex(phash_bits(gray, hash_size, highfreq_factor))


def hamming_distance_matrix(hashes: list[str] | np.ndarray) -> np.ndarray:
    """Pairwise Hamming distances between packed hashes (hex strings), as a uint16 [N, N] array.

    Computed with XOR + a popcount lookup table on packed bytes: for N ≈ 3 000 and 64-bit
    hashes this is ~70 MB of intermediate memory, well within budget.
    """
    packed = np.stack([hex_to_bytes(h) for h in hashes])  # [N, B]
    xor = packed[:, None, :] ^ packed[None, :, :]  # [N, N, B]
    return _POPCOUNT[xor].sum(axis=2, dtype=np.uint16)


def hamming_distance(a: str, b: str) -> int:
    return int(_POPCOUNT[hex_to_bytes(a) ^ hex_to_bytes(b)].sum())
