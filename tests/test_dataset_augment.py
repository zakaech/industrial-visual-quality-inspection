"""Dataset, canvas cache and augmentations: shapes, determinism, label preservation, guards."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from vqi.config import Config, load_config
from vqi.data.dataset import CanvasCache, VqiDataset, build_split, make_loader, training_rows
from vqi.data.download import raw_layout
from vqi.data.duplicates import analyse_duplicates, write_duplicates_report
from vqi.data.manifest import (
    build_manifest,
    check_expected_counts,
    read_manifest_meta,
    write_manifest,
)
from vqi.data.splits import ForbiddenSplitError, make_splits, write_splits
from vqi.preprocessing import Preprocessor
from vqi.preprocessing.augment import Augmenter, rng_for

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cfg() -> Config:
    return load_config(REPO_ROOT / "configs" / "default.yaml")


@pytest.fixture
def prepared(synthetic_config: Config, synthetic_dataset: dict):
    cfg = synthetic_config
    root = synthetic_dataset["root"]
    manifest, strays = build_manifest(cfg, raw_layout(cfg), root)
    manifest, report, nn = analyse_duplicates(manifest, cfg)
    write_manifest(manifest, cfg, check_expected_counts(manifest, cfg), strays=strays)
    write_duplicates_report(report, nn, cfg)
    splits = make_splits(manifest, cfg)
    write_splits(splits, manifest, cfg, manifest_sha256=read_manifest_meta(cfg)["manifest_sha256"])
    return cfg, root


# ------------------------------------------------------------------------------ augmentation


def _canvas(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(30, 60, size=(64, 32, 3), dtype=np.uint8)


def test_rng_is_keyed_by_seed_epoch_index():
    a = rng_for(42, 1, 7).random(3)
    assert np.array_equal(a, rng_for(42, 1, 7).random(3))
    assert not np.array_equal(a, rng_for(42, 2, 7).random(3))
    assert not np.array_equal(a, rng_for(42, 1, 8).random(3))


def test_flips_are_exact_permutations_and_involutive(cfg: Config):
    aug = Augmenter.from_config(cfg)
    canvas = _canvas()
    draw = {
        "hflip": True,
        "vflip": True,
        "photometric": False,
        "brightness": 1.0,
        "contrast": 1.0,
        "dx": 0,
        "dy": 0,
    }
    out = aug.apply(canvas, draw)
    assert out.shape == canvas.shape and out.dtype == np.uint8
    assert np.array_equal(np.sort(out.ravel()), np.sort(canvas.ravel()))  # same pixels
    assert np.array_equal(aug.apply(out, draw), canvas)  # flip twice = identity


def test_defect_survives_every_augmentation(cfg: Config):
    """A synthetic defect (mask) must be fully present after each augmentation draw."""
    aug = Augmenter.from_config(cfg)
    aug = Augmenter(**{**aug.describe(), "translation": True, "translation_p": 1.0})
    canvas = _canvas()
    mask = np.zeros((64, 32), dtype=np.uint8)
    mask[20:26, 10:15] = 255
    canvas[20:26, 10:15] = 200
    for k in range(50):
        draw = aug.sample(rng_for(1, 1, k))
        out = aug.apply(canvas, draw)
        m_out = aug.apply_to_mask(mask, draw)
        assert int((m_out > 0).sum()) == int((mask > 0).sum())  # no defect pixel lost
        # bright defect pixels stay brighter than the background after photometry
        assert out[m_out > 0].mean() > out[m_out == 0].mean()


def test_photometric_is_bounded_and_label_free(cfg: Config):
    aug = Augmenter.from_config(cfg)
    canvas = _canvas()
    draw = {
        "hflip": False,
        "vflip": False,
        "photometric": True,
        "brightness": 1.1,
        "contrast": 1.1,
        "dx": 0,
        "dy": 0,
    }
    out = aug.apply(canvas, draw)
    assert out.dtype == np.uint8 and out.shape == canvas.shape
    assert abs(float(out.mean()) - 1.1 * float(canvas.mean())) < 1.5
    assert float(((out == 0) | (out == 255)).mean()) == 0.0


# ------------------------------------------------------------------------------ dataset


def test_test_split_is_refused_for_training(prepared):
    cfg, _ = prepared
    with pytest.raises(ForbiddenSplitError):
        training_rows(cfg, "test")


def test_dataset_item_contract_and_val_determinism(prepared):
    cfg, root = prepared
    pre = Preprocessor.from_config(cfg)
    ds, cache, secs = build_split(cfg, "val", pre, root, None)
    x, y, image_id = ds[0]
    assert tuple(x.shape) == (3, 640, 224) and x.dtype == torch.float32
    assert y.dtype == torch.float32 and float(y) in (0.0, 1.0)
    assert isinstance(image_id, str)
    x2, _, _ = ds[0]
    assert torch.equal(x, x2)  # validation is deterministic
    assert cache is not None and secs is not None
    check = cache.verify(pre, root, n=5, seed=1)
    assert check["n_mismatch"] == 0 and check["max_abs_diff"] == 0


def test_augmentation_is_train_only_and_epoch_keyed(prepared):
    cfg, root = prepared
    pre = Preprocessor.from_config(cfg)
    aug = Augmenter.from_config(cfg)
    with pytest.raises(ValueError, match="train split"):
        build_split(cfg, "val", pre, root, aug)
    ds, _, _ = build_split(cfg, "train", pre, root, aug)
    ds.set_epoch(1)
    a, _, _ = ds[0]
    b, _, _ = ds[0]
    assert torch.equal(a, b)  # same (seed, epoch, index) → same sample
    e1 = [ds[i][0] for i in range(8)]
    ds.set_epoch(2)
    e2 = [ds[i][0] for i in range(8)]
    # over several indices at least one augmented sample differs between epochs
    assert any(not torch.equal(p, q) for p, q in zip(e1, e2, strict=True))


def test_loader_is_reproducible_and_cache_free_path_matches(prepared):
    cfg, root = prepared
    pre = Preprocessor.from_config(cfg)
    ds_cached, _, _ = build_split(cfg, "val", pre, root, None, use_cache=True)
    ds_live, _, _ = build_split(cfg, "val", pre, root, None, use_cache=False)
    for i in range(3):
        assert torch.equal(ds_cached[i][0], ds_live[i][0])
    loader = make_loader(ds_cached, 4, True, 0, seed=42)
    order1 = [ids for _, _, ids in loader]
    order2 = [ids for _, _, ids in make_loader(ds_cached, 4, True, 0, seed=42)]
    assert order1 == order2


def test_cache_key_changes_with_preprocessor(prepared):
    cfg, root = prepared
    rows = training_rows(cfg, "val")
    a = CanvasCache(cfg, Preprocessor.from_config(cfg), "val", rows)
    b = CanvasCache(cfg, Preprocessor.from_config(cfg, variant="clahe"), "val", rows)
    assert a.path != b.path
    ds = VqiDataset(rows, Preprocessor.from_config(cfg), root)
    assert len(ds) == len(rows)
