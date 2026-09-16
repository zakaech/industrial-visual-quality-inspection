"""Preprocessing contract: one code path, deterministic, correct shapes/channels/geometry."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from vqi.config import Config, load_config
from vqi.preprocessing import Preprocessor
from vqi.preprocessing.enhance import OPERATORS, apply_steps, validate_steps
from vqi.preprocessing.geometry import apply_geometry_to_mask, compute_geometry
from vqi.preprocessing.io import ImageDecodeError, decode_image

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cfg() -> Config:
    return load_config(REPO_ROOT / "configs" / "default.yaml")


def _textured(h: int, w: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(20, 80, size=(h, w, 3), dtype=np.uint8)


def _png_bytes(img_bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img_bgr)
    assert ok
    return buf.tobytes()


# ------------------------------------------------------------------------------ decoding


def test_decode_path_and_bytes_are_identical(tmp_path: Path):
    img = _textured(64, 24)
    p = tmp_path / "a.png"
    cv2.imwrite(str(p), img)
    from_path = decode_image(p)
    from_bytes = decode_image(p.read_bytes())
    assert np.array_equal(from_path, from_bytes)
    assert from_path.dtype == np.uint8 and from_path.shape == (64, 24, 3)
    # OpenCV stores BGR; the contract is RGB
    assert np.array_equal(from_path[:, :, ::-1], img)


@pytest.mark.parametrize("channels", [1, 3, 4])
def test_decode_normalises_channels(channels: int):
    base = _textured(32, 16)
    if channels == 1:
        src = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
    elif channels == 4:
        src = cv2.cvtColor(base, cv2.COLOR_BGR2BGRA)
    else:
        src = base
    out = decode_image(_png_bytes(src))
    assert out.shape == (32, 16, 3) and out.dtype == np.uint8


def test_decode_16bit_is_scaled_to_8bit():
    img16 = (np.full((8, 8), 40, dtype=np.uint16) * 257).astype(np.uint16)
    out = decode_image(_png_bytes(img16))
    assert out.dtype == np.uint8 and int(out[0, 0, 0]) == 40


def test_decode_rejects_garbage():
    with pytest.raises(ImageDecodeError):
        decode_image(b"not an image")
    with pytest.raises(ImageDecodeError):
        decode_image(b"")


# ------------------------------------------------------------------------------ geometry


@pytest.mark.parametrize("src", [(637, 229), (597, 184), (665, 241), (100, 400), (640, 224)])
def test_output_canvas_is_fixed_and_aspect_is_kept(cfg: Config, src: tuple[int, int]):
    pre = Preprocessor.from_config(cfg)
    h, w = src
    out = pre(_textured(h, w))
    assert out.shape == (cfg.preprocessing.input_size.height, cfg.preprocessing.input_size.width, 3)
    geom = pre.geometry_for(h, w)
    assert abs(geom.resized_h / geom.resized_w - h / w) < 0.02
    assert geom.resized_h <= geom.canvas_h and geom.resized_w <= geom.canvas_w
    # content is centred: symmetric padding up to 1 px
    assert abs(geom.top - (geom.canvas_h - geom.resized_h - geom.top)) <= 1
    assert abs(geom.left - (geom.canvas_w - geom.resized_w - geom.left)) <= 1


def test_padding_uses_image_median_by_default(cfg: Config):
    pre = Preprocessor.from_config(cfg)
    img = np.full((300, 224, 3), 57, dtype=np.uint8)
    out = pre(img)
    assert pre.pad_value == "median"
    assert int(out[0, 0, 0]) == 57 and int(out[-1, -1, 0]) == 57
    zero = Preprocessor.from_config(cfg, pad_value="zero")
    assert int(zero(img)[0, 0, 0]) == 0


def test_mask_follows_image_geometry(cfg: Config):
    pre = Preprocessor.from_config(cfg)
    h, w = 637, 229
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[300:310, 100:110] = 255
    on_canvas = pre.transform_mask(mask, h, w)
    assert on_canvas.shape == (640, 224)
    geom = pre.geometry_for(h, w)
    ys, xs = np.nonzero(on_canvas)
    assert abs(ys.mean() - (305 * geom.scale + geom.top)) < 2
    assert abs(xs.mean() - (105 * geom.scale + geom.left)) < 2
    assert set(np.unique(on_canvas)) <= {0, 255}


def test_stretch_mode_exists():
    geom = compute_geometry(100, 50, 640, 224, "stretch")
    assert (geom.resized_h, geom.resized_w, geom.top, geom.left) == (640, 224, 0, 0)
    m = apply_geometry_to_mask(np.ones((100, 50), dtype=np.uint8), geom)
    assert m.shape == (640, 224)


# ------------------------------------------------------------------------------ enhancement


@pytest.mark.parametrize("op", sorted(OPERATORS))
def test_each_operator_preserves_shape_and_dtype(op: str):
    img = _textured(96, 48)
    out = OPERATORS[op](img)
    assert out.shape == img.shape and out.dtype == np.uint8
    assert np.array_equal(out, OPERATORS[op](img))  # deterministic


def test_validate_steps_rejects_unknown_operator():
    with pytest.raises(ValueError, match="unknown operator"):
        validate_steps([{"op": "sepia"}])
    assert apply_steps(_textured(8, 8), []).shape == (8, 8, 3)


def test_every_config_variant_builds_and_runs(cfg: Config):
    img = _textured(637, 229)
    for name in cfg.preprocessing.enhancement.variants:
        pre = Preprocessor.from_config(cfg, variant=name)
        out = pre(img)
        assert out.shape == (640, 224, 3) and out.dtype == np.uint8
        assert pre.describe()["variant"] == name
    with pytest.raises(KeyError):
        Preprocessor.from_config(cfg, variant="does_not_exist")


def test_active_variant_is_what_the_api_will_use(cfg: Config):
    pre = Preprocessor.from_config(cfg)
    assert pre.variant == cfg.preprocessing.enhancement.active


# ------------------------------------------------------------------------------ model input


def test_to_tensor_shape_dtype_and_normalisation(cfg: Config):
    pre = Preprocessor.from_config(cfg)
    canvas = pre(_textured(637, 229))
    t = pre.to_tensor(canvas)
    assert tuple(t.shape) == (3, 640, 224)
    assert str(t.dtype) == "torch.float32"
    expected = (canvas[0, 0, 0] / 255.0 - pre.mean[0]) / pre.std[0]
    assert abs(float(t[0, 0, 0]) - expected) < 1e-5


def test_full_pipeline_path_vs_bytes_contract(cfg: Config, tmp_path: Path):
    """The API (bytes) and training (path) must produce identical tensors."""
    img = _textured(637, 229)
    p = tmp_path / "x.png"
    cv2.imwrite(str(p), img)
    pre = Preprocessor.from_config(cfg, variant="clahe")
    assert np.array_equal(pre(p), pre(p.read_bytes()))
