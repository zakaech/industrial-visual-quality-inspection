"""Integration tests of the ingestion chain on the synthetic dataset:
manifest → duplicates → expected counts → splits → EDA."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from vqi.config import Config, load_config
from vqi.data.download import raw_layout
from vqi.data.duplicates import analyse_duplicates, write_duplicates_report
from vqi.data.eda import run_eda
from vqi.data.manifest import (
    build_manifest,
    check_expected_counts,
    read_manifest,
    read_manifest_meta,
    write_manifest,
)
from vqi.data.splits import SplitError, make_splits, read_splits, verify_splits, write_splits


@pytest.fixture
def ingested(synthetic_config: Config, synthetic_dataset: dict):
    cfg = synthetic_config
    root = synthetic_dataset["root"]
    manifest, strays = build_manifest(cfg, raw_layout(cfg), root)
    manifest, report, nn = analyse_duplicates(manifest, cfg)
    counts = check_expected_counts(manifest, cfg)
    write_manifest(manifest, cfg, counts, strays=strays)
    write_duplicates_report(report, nn, cfg)
    return cfg, root, manifest, strays, report, nn, counts


def test_manifest_inventories_everything_without_deleting(ingested, synthetic_dataset):
    cfg, root, manifest, strays, report, nn, counts = ingested
    assert len(manifest) == 41 + 20  # 40 train + "(copy)" + 20 test
    assert (~manifest["name_conforms"]).sum() == 1
    assert manifest.loc[~manifest["name_conforms"], "image_id"].iloc[0] == "10003 (copy)"
    assert len(strays) == 1 and strays[0]["path"].endswith("10003_GT (copy).png")
    assert manifest["is_corrupted"].sum() == 0
    # raw tree untouched
    assert (synthetic_dataset["raw"] / "train" / "10003 (copy).png").exists()
    assert (synthetic_dataset["raw"] / "train" / "10003_GT (copy).png").exists()


def test_labels_come_from_masks(ingested, synthetic_dataset):
    _, _, manifest, *_ = ingested
    ok = manifest[manifest["name_conforms"]]
    labelled = set(ok.loc[ok["label"] == 1, "image_id"])
    assert labelled == synthetic_dataset["defective"]
    assert (ok.loc[ok["label"] == 1, "defect_area_px"] == 16).all()
    assert (ok.loc[ok["label"] == 0, "defect_area_px"] == 0).all()


def test_expected_counts_check(ingested, synthetic_config: Config, tmp_path: Path):
    cfg, _, manifest, _, _, _, counts = ingested
    assert counts["match"], counts
    # A wrong expectation is reported, not silently accepted.
    override = tmp_path / "wrong.yaml"
    override.write_text(
        yaml.safe_dump({"dataset": {"expected_counts": {"train_total": 41}}}), encoding="utf-8"
    )
    wrong_cfg = load_config(cfg.source_files[0], overrides=[*cfg.source_files[1:], override])
    wrong = check_expected_counts(manifest, wrong_cfg)
    assert not wrong["match"] and wrong["mismatches"] == {"train_total": (41, 40)}


def test_duplicate_policy(ingested):
    _, _, manifest, _, report, _, _ = ingested
    reasons = manifest.set_index("image_id")["exclude_reason"]
    assert reasons["10039"] == "exact_duplicate_of_train:10003"
    assert reasons["10038"] == "exact_duplicate_of_test:20007"
    assert reasons["10003 (copy)"] == "nonconforming_filename"
    assert reasons["20007"] == ""  # test images are never excluded
    assert reasons["10003"] == ""  # one copy is kept
    assert report["near"]["status"] == "disabled_until_threshold_chosen"
    assert report["exact"]["n_cross_split_groups"] == 1
    # duplicates share a group id
    groups = manifest.set_index("image_id")["group_id"]
    assert groups["10003"] == groups["10039"] == groups["10003 (copy)"]
    assert groups["10038"] == groups["20007"]


def test_splits_invariants_and_reproducibility(ingested, synthetic_config: Config):
    cfg, root, manifest, *_ = ingested
    splits = make_splits(manifest, cfg)
    checks = verify_splits(splits, manifest)
    assert checks["groups_spanning_train_val"] == 0
    assert checks["test_equals_official_test"]
    assert checks["val_in_official_train_only"]
    assert set(splits.loc[splits["split"] == "excluded", "image_id"]) == {
        "10038",
        "10039",
        "10003 (copy)",
    }
    pool = splits[splits["split"].isin(["train", "val"])]
    assert len(pool) == 38  # 40 conforming train images − 2 excluded duplicates
    assert abs(checks["val_fraction_actual"] - 0.2) < 0.06
    # stratification: positives on both sides
    assert (pool.loc[pool["split"] == "val", "label"] == 1).sum() >= 1
    assert (pool.loc[pool["split"] == "train", "label"] == 1).sum() >= 1

    # same seed → byte-identical CSV; different seed → different assignment
    meta = read_manifest_meta(cfg)
    write_splits(splits, manifest, cfg, manifest_sha256=meta["manifest_sha256"])
    first = read_splits(cfg)
    again = make_splits(read_manifest(cfg), cfg)
    pd.testing.assert_frame_equal(
        first[["image_id", "split"]].reset_index(drop=True),
        again[["image_id", "split"]].reset_index(drop=True),
    )
    other_seed = tmp_override(cfg, {"reproducibility": {"seed": 7}})
    different = make_splits(manifest, other_seed)
    assert not (different["split"].to_numpy() == splits["split"].to_numpy()).all()


def test_split_refuses_unsupported_fraction(ingested):
    cfg, _, manifest, *_ = ingested
    bad = tmp_override(cfg, {"split": {"val_fraction": 0.3}})
    with pytest.raises(SplitError, match="1/k"):
        make_splits(manifest, bad)


def test_eda_produces_report_and_figures(ingested):
    cfg, root, manifest, _, report, nn, _ = ingested
    splits = make_splits(manifest, cfg)
    meta = read_manifest_meta(cfg)
    write_splits(splits, manifest, cfg, manifest_sha256=meta["manifest_sha256"])
    from vqi.data.splits import read_splits_meta

    out = run_eda(
        read_manifest(cfg), meta, report, nn, read_splits(cfg), read_splits_meta(cfg), cfg, root
    )
    text = out.read_text(encoding="utf-8")
    assert "Analyse exploratoire" in text
    assert "Découpage train / validation / test" in text
    figures = out.parent / "figures"
    for name in (
        "contact_sheet.png",
        "closest_pairs.png",
        "phash_nn_distance.png",
        "dimensions.png",
        "defect_area.png",
        "luminance_contrast.png",
        "acquisition_order.png",
    ):
        assert (figures / name).exists(), name


def tmp_override(cfg: Config, override: dict) -> Config:
    """Reload ``cfg`` with one more override written next to its last source file."""
    path = Path(cfg.source_files[-1]).parent / f"override_{abs(hash(str(override)))}.yaml"
    path.write_text(yaml.safe_dump(override), encoding="utf-8")
    return load_config(cfg.source_files[0], overrides=[*cfg.source_files[1:], path])
