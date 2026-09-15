import hashlib
from pathlib import Path

import numpy as np

from vqi.data.hashing import (
    hamming_distance,
    hamming_distance_matrix,
    phash_bits,
    phash_hex,
    sha256_file,
)


def _textured(rng: np.random.Generator, h: int = 128, w: int = 48) -> np.ndarray:
    # Smooth low-frequency structure + noise, so the DCT signature is meaningful.
    yy, xx = np.mgrid[0:h, 0:w]
    base = 40 + 20 * np.sin(yy / 17.0) * np.cos(xx / 9.0) + rng.normal(0, 4, size=(h, w))
    return np.clip(base, 0, 255).astype(np.uint8)


def test_sha256_file_matches_hashlib(tmp_path: Path):
    p = tmp_path / "f.bin"
    p.write_bytes(b"industrial" * 1000)
    assert sha256_file(p) == hashlib.sha256(b"industrial" * 1000).hexdigest()


def test_phash_is_64_bits_and_deterministic():
    rng = np.random.default_rng(1)
    img = _textured(rng)
    assert phash_bits(img).shape == (64,)
    assert phash_hex(img) == phash_hex(img)
    assert len(phash_hex(img)) == 16


def test_phash_identical_and_near_identical_images_are_close():
    rng = np.random.default_rng(2)
    img = _textured(rng)
    noisy = np.clip(img.astype(np.int16) + rng.integers(-3, 4, size=img.shape), 0, 255)
    assert hamming_distance(phash_hex(img), phash_hex(img)) == 0
    assert hamming_distance(phash_hex(img), phash_hex(noisy.astype(np.uint8))) <= 6


def test_phash_unrelated_images_are_far():
    rng = np.random.default_rng(3)
    a = _textured(rng)
    b = np.flipud(np.fliplr(_textured(np.random.default_rng(99), 128, 48)))
    assert hamming_distance(phash_hex(a), phash_hex(b)) >= 16


def test_hamming_matrix_is_symmetric_with_zero_diagonal():
    rng = np.random.default_rng(4)
    hashes = [phash_hex(_textured(rng)) for _ in range(5)]
    m = hamming_distance_matrix(hashes)
    assert m.shape == (5, 5)
    assert (m == m.T).all()
    assert (np.diag(m) == 0).all()
    assert m[0, 1] == hamming_distance(hashes[0], hashes[1])
