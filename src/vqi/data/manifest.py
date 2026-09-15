"""Build the dataset manifest: one row per image, the single source of truth for everything
downstream (labels, splits, exclusions, EDA statistics).

The manifest is a *view* over the immutable raw tree. Corrupted or duplicated images are never
deleted from ``data/raw``; they are flagged here and excluded by the split step.

Label rule (``dataset.label_rule == mask_non_empty``): an image is *defective* (label 1) iff its
ground-truth mask contains at least one positive pixel. The mask is read here only to derive the
label and the defect area; the training ``Dataset`` never opens it (leakage guard L7).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from vqi.config import Config
from vqi.data.download import RawLayout
from vqi.data.hashing import phash_hex, sha256_file
from vqi.logging_utils import get_logger, log_event
from vqi.provenance import provenance

logger = get_logger(__name__)

MANIFEST_FILENAME = "manifest.csv"
MANIFEST_META_FILENAME = "manifest.meta.json"

# Column order of the manifest; kept explicit so the CSV is stable across runs.
MANIFEST_COLUMNS = [
    "image_id",
    "official_split",
    "name_conforms",
    "acquisition_index",
    "image_path",
    "mask_path",
    "label",
    "defect_area_px",
    "defect_bbox_w",
    "defect_bbox_h",
    "n_defect_components",
    "height",
    "width",
    "channels",
    "channel_spread",
    "mean_luminance",
    "std_luminance",
    "min_luminance",
    "max_luminance",
    "mask_height",
    "mask_width",
    "mask_unique_values",
    "image_bytes",
    "sha256",
    "phash",
    "is_corrupted",
    "corruption_reason",
]


def _rel(path: Path, root: Path) -> str:
    """Path as stored in the manifest: POSIX, relative to the repository root."""
    if path.is_absolute():
        path = path.relative_to(root)
    return path.as_posix()


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _channel_spread(img: np.ndarray) -> float:
    """Mean absolute difference between colour channels; 0 for a truly grayscale image stored
    as 3 channels. Tells whether ``color_mode: rgb`` carries any information."""
    if img.ndim == 2 or img.shape[2] == 1:
        return 0.0
    bgr = img[:, :, :3].astype(np.int16)
    spread = (
        np.abs(bgr[:, :, 0] - bgr[:, :, 1]).mean()
        + np.abs(bgr[:, :, 1] - bgr[:, :, 2]).mean()
        + np.abs(bgr[:, :, 0] - bgr[:, :, 2]).mean()
    ) / 3.0
    return float(spread)


def _mask_stats(mask: np.ndarray) -> dict[str, Any]:
    positive = mask > 0
    area = int(positive.sum())
    stats: dict[str, Any] = {
        "defect_area_px": area,
        "defect_bbox_w": 0,
        "defect_bbox_h": 0,
        "n_defect_components": 0,
        "mask_unique_values": ",".join(str(v) for v in np.unique(mask)[:8]),
    }
    if area:
        ys, xs = np.nonzero(positive)
        stats["defect_bbox_w"] = int(xs.max() - xs.min() + 1)
        stats["defect_bbox_h"] = int(ys.max() - ys.min() + 1)
        n_labels, _ = cv2.connectedComponents(positive.astype(np.uint8), connectivity=8)
        stats["n_defect_components"] = int(n_labels - 1)
    return stats


def inspect_image(
    image_path: Path, mask_path: Path, cfg: Config, project_root: Path, name_conforms: bool = True
) -> dict[str, Any]:
    """Decode one image/mask pair and compute every manifest column for it.

    A file whose name does not follow the dataset convention (``name_conforms=False``) is still
    inventoried (bytes, hash, pixel statistics) so that the duplicate analysis can say what it
    is, but no label is derived for it and the split step excludes it.
    """
    dup = cfg.split.duplicates
    row: dict[str, Any] = {
        "image_id": image_path.stem,
        "name_conforms": bool(name_conforms),
        "image_path": _rel(image_path, project_root),
        "mask_path": _rel(mask_path, project_root) if mask_path.exists() else "",
        "label": -1,
        "is_corrupted": False,
        "corruption_reason": "",
    }
    try:
        row["acquisition_index"] = int(image_path.stem)
    except ValueError:
        row["acquisition_index"] = -1

    row["image_bytes"] = image_path.stat().st_size
    row["sha256"] = sha256_file(image_path)

    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED) if row["image_bytes"] else None
    if img is None:
        row.update(is_corrupted=True, corruption_reason="image_undecodable")
        return row

    gray = _to_gray(img)
    row.update(
        height=int(img.shape[0]),
        width=int(img.shape[1]),
        channels=int(1 if img.ndim == 2 else img.shape[2]),
        channel_spread=round(_channel_spread(img), 4),
        mean_luminance=round(float(gray.mean()), 3),
        std_luminance=round(float(gray.std()), 3),
        min_luminance=int(gray.min()),
        max_luminance=int(gray.max()),
        phash=phash_hex(gray, dup.phash_size, dup.phash_highfreq_factor),
    )

    if not name_conforms:
        return row  # no label for a file outside the naming convention
    if not mask_path.exists():
        row.update(is_corrupted=True, corruption_reason="mask_missing")
        return row
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        row.update(is_corrupted=True, corruption_reason="mask_undecodable")
        return row
    row.update(mask_height=int(mask.shape[0]), mask_width=int(mask.shape[1]))
    if mask.shape[:2] != gray.shape[:2]:
        row.update(is_corrupted=True, corruption_reason="mask_size_mismatch")
        return row

    row.update(_mask_stats(mask))
    if cfg.dataset.label_rule != "mask_non_empty":
        raise ValueError(f"Unsupported label rule: {cfg.dataset.label_rule}")
    row["label"] = int(row["defect_area_px"] > 0)
    return row


def list_pairs(
    directory: Path, image_ext: str, mask_suffix: str, image_id_pattern: str
) -> tuple[list[tuple[Path, Path, bool]], list[Path]]:
    """Inventory a raw directory.

    Returns ``(pairs, strays)`` where ``pairs`` holds ``(image_path, expected_mask_path,
    name_conforms)`` for every image-like file, and ``strays`` lists mask-like files that do not
    belong to a conforming image (e.g. ``10301_GT (copy).png``). Nothing is deleted.
    """
    pattern = re.compile(image_id_pattern)
    files = sorted(p for p in directory.iterdir() if p.suffix.lower() == image_ext.lower())
    conforming_ids = {p.stem for p in files if pattern.match(p.stem)}
    pairs, strays = [], []
    for path in files:
        stem = path.stem
        if mask_suffix in stem:  # mask-like
            owner = stem.split(mask_suffix)[0]
            if not (stem.endswith(mask_suffix) and owner in conforming_ids):
                strays.append(path)
            continue
        mask = path.with_name(f"{stem}{mask_suffix}{path.suffix}")
        pairs.append((path, mask, bool(pattern.match(stem))))
    return pairs, strays


def build_manifest(
    cfg: Config, layout: RawLayout, project_root: Path
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Walk the raw tree; return the manifest DataFrame and the list of stray files."""
    rows: list[dict[str, Any]] = []
    strays: list[dict[str, Any]] = []
    for split_name, directory in (("train", layout.train_dir), ("test", layout.test_dir)):
        pairs, stray_paths = list_pairs(
            directory, cfg.dataset.image_ext, cfg.dataset.mask_suffix, cfg.dataset.image_id_pattern
        )
        n_nonconforming = sum(1 for _, _, ok in pairs if not ok)
        log_event(
            logger,
            "manifest.scan",
            split=split_name,
            n_images=len(pairs),
            n_nonconforming_names=n_nonconforming,
            n_stray_files=len(stray_paths),
        )
        for image_path, mask_path, ok in pairs:
            row = inspect_image(image_path, mask_path, cfg, project_root, name_conforms=ok)
            row["official_split"] = split_name
            rows.append(row)
        for path in stray_paths:
            strays.append(
                {
                    "path": _rel(path, project_root),
                    "official_split": split_name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    df = pd.DataFrame(rows)
    for col in MANIFEST_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    df = df[MANIFEST_COLUMNS].sort_values(["official_split", "acquisition_index", "image_id"])
    df = df.reset_index(drop=True)
    if df["image_id"].duplicated().any():
        dupes = df.loc[df["image_id"].duplicated(), "image_id"].tolist()
        raise ValueError(f"image_id must be unique across splits; duplicates: {dupes[:5]}")
    return df, strays


def check_expected_counts(df: pd.DataFrame, cfg: Config) -> dict[str, Any]:
    """Compare observed counts with ``dataset.expected_counts``. Nothing is corrected here: a
    mismatch is reported so that a human decides whether the config or the data is wrong."""
    exp = cfg.dataset.expected_counts
    valid = df[(~df["is_corrupted"]) & df["name_conforms"]]
    observed = {
        "train_total": int((valid["official_split"] == "train").sum()),
        "train_defective": int(
            ((valid["official_split"] == "train") & (valid["label"] == 1)).sum()
        ),
        "test_total": int((valid["official_split"] == "test").sum()),
        "test_defective": int(((valid["official_split"] == "test") & (valid["label"] == 1)).sum()),
    }
    expected = {
        "train_total": exp.train_total,
        "train_defective": exp.train_defective,
        "test_total": exp.test_total,
        "test_defective": exp.test_defective,
    }
    mismatches = {k: (expected[k], observed[k]) for k in expected if expected[k] != observed[k]}
    return {
        "match": not mismatches,
        "expected": expected,
        "observed": observed,
        "mismatches": mismatches,
    }


def manifest_paths(cfg: Config) -> tuple[Path, Path]:
    return (
        cfg.paths.interim_dir / MANIFEST_FILENAME,
        cfg.paths.interim_dir / MANIFEST_META_FILENAME,
    )


def write_manifest(
    df: pd.DataFrame,
    cfg: Config,
    counts_check: dict[str, Any],
    strays: list[dict[str, Any]] | None = None,
) -> Path:
    csv_path, meta_path = manifest_paths(cfg)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False, lineterminator="\n")
    meta = provenance(
        cfg,
        n_rows=int(len(df)),
        n_nonconforming_names=int((~df["name_conforms"]).sum()),
        nonconforming_names=df.loc[~df["name_conforms"], "image_path"].tolist(),
        stray_files=strays or [],
        n_corrupted=int(df["is_corrupted"].sum()),
        corruption_reasons=df.loc[df["is_corrupted"], "corruption_reason"].value_counts().to_dict(),
        counts_check=counts_check,
        manifest_sha256=sha256_file(csv_path),
        archive_sha256=cfg.dataset.archive_sha256,
    )
    meta_path.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n"
    )
    log_event(
        logger,
        "manifest.written",
        path=str(csv_path),
        n_rows=len(df),
        counts_match=counts_check["match"],
    )
    return csv_path


def read_manifest(cfg: Config) -> pd.DataFrame:
    csv_path, _ = manifest_paths(cfg)
    if not csv_path.exists():
        raise FileNotFoundError(f"Manifest not found: {csv_path} — run scripts/build_manifest.py")
    str_cols = [
        "image_id",
        "image_path",
        "mask_path",
        "sha256",
        "phash",
        "mask_unique_values",
        "corruption_reason",
        "official_split",
        "group_id",
        "exclude_reason",
    ]
    df = pd.read_csv(csv_path, dtype=dict.fromkeys(str_cols, str))
    str_cols = [c for c in str_cols if c in df.columns]
    df[str_cols] = df[str_cols].fillna("")
    df["is_corrupted"] = df["is_corrupted"].astype(bool)
    df["name_conforms"] = df["name_conforms"].astype(bool)
    return df


def read_manifest_meta(cfg: Config) -> dict[str, Any]:
    _, meta_path = manifest_paths(cfg)
    return json.loads(meta_path.read_text(encoding="utf-8"))
