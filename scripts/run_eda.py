"""Exploratory data analysis: tables, figures, contact sheet and a Markdown report.

Reads data/interim/manifest.csv, the duplicate report and (if present) the split file; writes
everything under reports/eda/. Never modifies the data.

Usage (from the repository root):
    python scripts/run_eda.py [--config configs/default.yaml]
"""

from __future__ import annotations

import argparse
import sys

from vqi.cli import add_common_args, bootstrap
from vqi.data.duplicates import read_duplicates_report, read_nn_distances
from vqi.data.eda import run_eda
from vqi.data.manifest import read_manifest, read_manifest_meta
from vqi.data.splits import read_splits, read_splits_meta
from vqi.logging_utils import get_logger, log_event

logger = get_logger("scripts.run_eda")


def main() -> int:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()
    cfg, root = bootstrap(args)

    manifest = read_manifest(cfg)
    manifest_meta = read_manifest_meta(cfg)
    dup_report = read_duplicates_report(cfg)
    nn = read_nn_distances(cfg)
    try:
        splits, split_meta = read_splits(cfg), read_splits_meta(cfg)
    except FileNotFoundError:
        splits, split_meta = None, None
        log_event(logger, "eda.no_splits_yet", level=30)

    report = run_eda(manifest, manifest_meta, dup_report, nn, splits, split_meta, cfg, root)
    log_event(logger, "eda.done", report=str(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
