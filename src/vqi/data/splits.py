"""Train / validation / test partition (decision D3).

* The **official test set is quarantined**: every non-corrupted official test image gets
  ``split == "test"`` and is evaluated once, at the very end.
* A **single stratified, group-aware hold-out** carves the validation set out of the official
  train set. scikit-learn has no ``StratifiedGroupShuffleSplit``, so the hold-out is obtained as
  *one* fold of a :class:`~sklearn.model_selection.StratifiedGroupKFold` partition with
  ``k = 1 / val_fraction``; the other folds are never used. This is not cross-validation.
* Images with a non-empty ``exclude_reason`` (corrupted, duplicates — see ``duplicates.py``)
  get ``split == "excluded"`` and never enter a DataLoader.

The resulting CSV is committed to git so the partition is reproducible without the data.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from vqi.config import Config
from vqi.data.hashing import sha256_file
from vqi.logging_utils import get_logger, log_event
from vqi.provenance import provenance

logger = get_logger(__name__)

SPLIT_COLUMNS = [
    "image_id",
    "official_split",
    "split",
    "label",
    "group_id",
    "exclude_reason",
    "acquisition_index",
    "defect_area_px",
    "sha256",
    "phash",
    "image_path",
    "mask_path",
]


class SplitError(RuntimeError):
    """A split invariant is violated."""


class ForbiddenSplitError(RuntimeError):
    """The official test split must not be touched before the final evaluation (Phase 4)."""


def _n_folds_for(val_fraction: float) -> int:
    k = round(1.0 / val_fraction)
    if abs(1.0 / k - val_fraction) > 1e-9:
        raise SplitError(
            f"val_fraction={val_fraction} must be 1/k for the group-aware stratified hold-out "
            f"(e.g. 0.2, 0.25, 0.1)"
        )
    return k


def make_splits(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Assign ``split`` ∈ {train, val, test, excluded} to every manifest row."""
    if cfg.split.strategy != "stratified_group_holdout":
        raise SplitError(f"Unsupported split strategy: {cfg.split.strategy}")
    if "exclude_reason" not in df.columns or "group_id" not in df.columns:
        raise SplitError("Run the duplicate analysis first (group_id / exclude_reason missing)")

    group_col = cfg.split.group_column or "group_id"
    if group_col not in df.columns:
        raise SplitError(f"group column '{group_col}' not in manifest")

    out = df.copy()
    out["split"] = "excluded"
    excluded = out["exclude_reason"].astype(str) != ""

    is_test = (out["official_split"] == "test") & ~excluded
    out.loc[is_test, "split"] = "test"

    pool = (out["official_split"] == "train") & ~excluded
    pool_idx = np.flatnonzero(pool.to_numpy())
    y = out.loc[pool, cfg.split.stratify_on].to_numpy()
    groups = out.loc[pool, group_col].to_numpy()

    k = _n_folds_for(cfg.split.val_fraction)
    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=cfg.reproducibility.seed)
    train_rel, val_rel = next(iter(sgkf.split(np.zeros(len(y)), y, groups)))
    out.loc[out.index[pool_idx[train_rel]], "split"] = "train"
    out.loc[out.index[pool_idx[val_rel]], "split"] = "val"

    out = out[SPLIT_COLUMNS].reset_index(drop=True)
    verify_splits(out, df)
    log_event(
        logger,
        "splits.made",
        strategy=cfg.split.strategy,
        k_folds_mechanism=k,
        fold_used=0,
        **{f"n_{s}": int((out["split"] == s).sum()) for s in ("train", "val", "test", "excluded")},
    )
    return out


def verify_splits(splits: pd.DataFrame, manifest: pd.DataFrame) -> dict[str, Any]:
    """Assert the invariants that make the partition leak-free; return the checks as a dict."""
    checks: dict[str, Any] = {}

    modelling = splits[splits["split"].isin(["train", "val"])]
    spanning = modelling.groupby("group_id")["split"].nunique()
    checks["groups_spanning_train_val"] = int((spanning > 1).sum())
    if checks["groups_spanning_train_val"]:
        raise SplitError("A duplicate group spans train and validation")

    official_test = set(
        manifest.loc[
            (manifest["official_split"] == "test")
            & (~manifest["is_corrupted"])
            & manifest["name_conforms"],
            "image_id",
        ]
    )
    checks["test_equals_official_test"] = (
        set(splits.loc[splits["split"] == "test", "image_id"]) == official_test
    )
    if not checks["test_equals_official_test"]:
        raise SplitError("The test split must equal the (non-corrupted) official test set")

    checks["val_in_official_train_only"] = bool(
        (splits.loc[splits["split"].isin(["train", "val"]), "official_split"] == "train").all()
    )
    if not checks["val_in_official_train_only"]:
        raise SplitError("train/val must only contain official train images")

    checks["no_duplicate_image_id"] = not bool(splits["image_id"].duplicated().any())
    if not checks["no_duplicate_image_id"]:
        raise SplitError("image_id appears more than once")

    n_train, n_val = int((splits["split"] == "train").sum()), int((splits["split"] == "val").sum())
    checks["val_fraction_actual"] = round(n_val / max(n_train + n_val, 1), 4)
    checks["positive_rate"] = {
        s: round(float((splits.loc[splits["split"] == s, "label"] == 1).mean()), 4)
        for s in ("train", "val", "test")
    }
    checks["counts"] = {
        s: {
            "total": int((splits["split"] == s).sum()),
            "defective": int(((splits["split"] == s) & (splits["label"] == 1)).sum()),
        }
        for s in ("train", "val", "test", "excluded")
    }
    return checks


def splits_paths(cfg: Config) -> tuple[Path, Path]:
    base = cfg.paths.splits_dir / cfg.split.manifest_version
    return base.with_suffix(".csv"), base.with_suffix(".meta.json")


def write_splits(
    splits: pd.DataFrame, manifest: pd.DataFrame, cfg: Config, manifest_sha256: str
) -> Path:
    csv_path, meta_path = splits_paths(cfg)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    splits.to_csv(csv_path, index=False, lineterminator="\n")
    meta = provenance(
        cfg,
        strategy=cfg.split.strategy,
        val_fraction=cfg.split.val_fraction,
        k_folds_mechanism=_n_folds_for(cfg.split.val_fraction),
        fold_used=0,
        stratify_on=cfg.split.stratify_on,
        group_column=cfg.split.group_column or "group_id",
        near_duplicate_threshold=cfg.split.duplicates.max_hamming_distance,
        manifest_sha256=manifest_sha256,
        splits_sha256=sha256_file(csv_path),
        checks=verify_splits(splits, manifest),
    )
    meta_path.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n"
    )
    log_event(logger, "splits.written", path=str(csv_path))
    return csv_path


def read_splits(cfg: Config) -> pd.DataFrame:
    csv_path, _ = splits_paths(cfg)
    if not csv_path.exists():
        raise FileNotFoundError(f"Splits not found: {csv_path} — run scripts/make_splits.py")
    str_cols = [
        "image_id",
        "group_id",
        "exclude_reason",
        "sha256",
        "phash",
        "image_path",
        "mask_path",
        "official_split",
        "split",
    ]
    df = pd.read_csv(csv_path, dtype=dict.fromkeys(str_cols, str))
    df[str_cols] = df[str_cols].fillna("")
    return df


def read_splits_meta(cfg: Config) -> dict[str, Any]:
    _, meta_path = splits_paths(cfg)
    if not meta_path.exists():
        raise FileNotFoundError(f"Split metadata not found: {meta_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def load_split_rows(cfg: Config, split: str, allowed_splits: Sequence[str]) -> pd.DataFrame:
    """Rows of one split, refusing anything outside ``allowed_splits`` (the test-set guard shared
    by the classical baseline and the CNN training)."""
    if split not in allowed_splits:
        raise ForbiddenSplitError(
            f"split '{split}' is not allowed here (allowed: {list(allowed_splits)})"
        )
    df = read_splits(cfg)
    rows = df[df["split"] == split].reset_index(drop=True)
    if rows.empty:
        raise ValueError(f"no rows for split '{split}'")
    return rows
