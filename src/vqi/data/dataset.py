"""PyTorch ``Dataset`` / ``DataLoader`` built on the shared :class:`Preprocessor`.

* :class:`CanvasCache` stores the *exact* ``uint8`` output of the preprocessor for one split as a
  ``.npy`` memmap (keyed by the preprocessor description and the split file hash), so training
  epochs pay for augmentation and tensor conversion only. It is an optimisation, not a new
  transformation: a check against the live preprocessor is part of the dry-run and of the tests.
* :class:`VqiDataset` yields ``(tensor float32 [3, H, W], label float32, image_id)``. Augmentation
  is applied only when an :class:`Augmenter` is given (train split), with an RNG keyed by
  ``(seed, epoch, index)``.
* The official test split cannot be loaded: rows come from
  :func:`vqi.data.splits.load_split_rows` with ``training.allowed_splits``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from vqi.config import Config
from vqi.data.splits import load_split_rows, read_splits_meta
from vqi.logging_utils import get_logger, log_event
from vqi.preprocessing.augment import Augmenter, rng_for
from vqi.preprocessing.pipeline import Preprocessor
from vqi.seed import make_generator, worker_init_fn

logger = get_logger(__name__)


def training_rows(cfg: Config, split: str) -> pd.DataFrame:
    return load_split_rows(cfg, split, cfg.training.allowed_splits)


# --------------------------------------------------------------------------------------------- #
# Canvas cache
# --------------------------------------------------------------------------------------------- #


class CanvasCache:
    """Memmapped ``uint8 [N, H, W, 3]`` canvases for one split, in the order of ``rows``."""

    def __init__(self, cfg: Config, pre: Preprocessor, split: str, rows: pd.DataFrame) -> None:
        self.rows = rows
        self.split = split
        key = hashlib.sha256(
            json.dumps(
                {
                    "preprocessor": pre.describe(),
                    "splits_sha256": read_splits_meta(cfg)["splits_sha256"],
                    "split": split,
                    "image_ids": rows["image_id"].tolist(),
                },
                sort_keys=True,
                default=str,
            ).encode()
        ).hexdigest()[:16]
        cfg.training.cache.dir.mkdir(parents=True, exist_ok=True)
        self.path = (
            cfg.training.cache.dir / f"{split}__{pre.variant}__pad-{pre.pad_value}__{key}.npy"
        )
        self.meta_path = self.path.with_suffix(".meta.json")
        self.shape = (len(rows), pre.height, pre.width, 3)

    def exists(self) -> bool:
        return self.path.exists() and self.meta_path.exists()

    def build(self, pre: Preprocessor, project_root: Path) -> float:
        """Write every canvas; returns the wall-clock seconds spent."""
        import time

        t0 = time.perf_counter()
        arr = np.lib.format.open_memmap(self.path, mode="w+", dtype=np.uint8, shape=self.shape)
        for i, path in enumerate(self.rows["image_path"]):
            arr[i] = pre(project_root / path)
        arr.flush()
        del arr
        seconds = time.perf_counter() - t0
        self.meta_path.write_text(
            json.dumps(
                {
                    "split": self.split,
                    "n": int(self.shape[0]),
                    "shape": list(self.shape),
                    "image_ids": self.rows["image_id"].tolist(),
                    "preprocessor": pre.describe(),
                    "build_seconds": seconds,
                },
                indent=2,
            ),
            encoding="utf-8",
            newline="\n",
        )
        log_event(
            logger, "cache.built", split=self.split, n=self.shape[0], seconds=round(seconds, 1)
        )
        return seconds

    def ensure(self, pre: Preprocessor, project_root: Path) -> float | None:
        """Build if missing; returns the build time or None when the cache was already there."""
        if self.exists():
            meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
            if meta["image_ids"] == self.rows["image_id"].tolist():
                return None
        return self.build(pre, project_root)

    def open(self) -> np.ndarray:
        return np.load(self.path, mmap_mode="r")

    def verify(self, pre: Preprocessor, project_root: Path, n: int, seed: int) -> dict[str, Any]:
        """Compare ``n`` random cached canvases with the live preprocessor (byte equality)."""
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(self.rows), size=min(n, len(self.rows)), replace=False)
        arr = self.open()
        mismatches = 0
        max_abs = 0
        for i in idx:
            live = pre(project_root / self.rows["image_path"].iloc[int(i)])
            diff = np.abs(arr[int(i)].astype(np.int16) - live.astype(np.int16))
            max_abs = max(max_abs, int(diff.max()))
            mismatches += int(diff.max() > 0)
        return {"n_checked": int(len(idx)), "n_mismatch": mismatches, "max_abs_diff": max_abs}


# --------------------------------------------------------------------------------------------- #
# Dataset / loader
# --------------------------------------------------------------------------------------------- #


class VqiDataset(Dataset):
    """One split of the manifest as model inputs. Picklable for spawn-based workers: the memmap is
    (re)opened lazily inside each process."""

    def __init__(
        self,
        rows: pd.DataFrame,
        pre: Preprocessor,
        project_root: Path,
        augmenter: Augmenter | None = None,
        cache_path: Path | None = None,
        seed: int = 42,
    ) -> None:
        self.image_ids = rows["image_id"].astype(str).tolist()
        self.paths = [str(project_root / p) for p in rows["image_path"]]
        self.labels = rows["label"].to_numpy().astype(np.float32)
        self.pre = pre
        self.augmenter = augmenter
        self.cache_path = cache_path
        self.seed = seed
        self.epoch = 0
        self._cache: np.ndarray | None = None

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.image_ids)

    def canvas(self, index: int) -> np.ndarray:
        if self.cache_path is not None:
            if self._cache is None:
                self._cache = np.load(self.cache_path, mmap_mode="r")
            return np.array(self._cache[index])  # materialise a private copy
        return self.pre(self.paths[index])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        canvas = self.canvas(index)
        if self.augmenter is not None:
            canvas = self.augmenter(canvas, rng_for(self.seed, self.epoch, index))
        return (
            self.pre.to_tensor(canvas),
            torch.tensor(self.labels[index], dtype=torch.float32),
            self.image_ids[index],
        )

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_cache"] = None  # never pickle the memmap
        return state


def make_loader(
    dataset: VqiDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    sampler: Sampler | None = None,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn if num_workers else None,
        generator=make_generator(seed),
        drop_last=False,
        pin_memory=False,
        persistent_workers=False,
    )


def build_split(
    cfg: Config,
    split: str,
    pre: Preprocessor,
    project_root: Path,
    augmenter: Augmenter | None,
    use_cache: bool | None = None,
) -> tuple[VqiDataset, CanvasCache | None, float | None]:
    """Rows → (dataset, cache, cache build seconds). Augmentation is the caller's decision, but
    it is refused for anything else than the train split."""
    if augmenter is not None and split != "train":
        raise ValueError("augmentation is only allowed on the train split")
    rows = training_rows(cfg, split)
    use_cache = cfg.training.cache.enabled if use_cache is None else use_cache
    cache: CanvasCache | None = None
    build_seconds: float | None = None
    if use_cache:
        cache = CanvasCache(cfg, pre, split, rows)
        build_seconds = cache.ensure(pre, project_root)
    ds = VqiDataset(
        rows,
        pre,
        project_root,
        augmenter=augmenter,
        cache_path=cache.path if cache else None,
        seed=cfg.reproducibility.seed,
    )
    return ds, cache, build_seconds


def class_counts(labels: Sequence[float]) -> tuple[int, int]:
    arr = np.asarray(labels)
    n_pos = int((arr == 1).sum())
    return n_pos, int(len(arr) - n_pos)
