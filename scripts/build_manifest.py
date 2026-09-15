"""Build the manifest (one row per image) and run the duplicate analysis.

Outputs: data/interim/manifest.csv, manifest.meta.json, duplicates_report.json,
phash_nearest_neighbours.csv. Never modifies data/raw/.

Usage (from the repository root):
    python scripts/build_manifest.py [--config configs/default.yaml]
"""

from __future__ import annotations

import argparse
import sys

from vqi.cli import add_common_args, bootstrap
from vqi.data.download import raw_layout
from vqi.data.duplicates import analyse_duplicates, write_duplicates_report
from vqi.data.manifest import build_manifest, check_expected_counts, write_manifest
from vqi.logging_utils import get_logger, log_event

logger = get_logger("scripts.build_manifest")


def main() -> int:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()
    cfg, root = bootstrap(args)

    layout = raw_layout(cfg)
    if not layout.train_dir.is_dir() or not layout.test_dir.is_dir():
        log_event(logger, "manifest.raw_missing", level=40, root=str(layout.root))
        return 1

    manifest, strays = build_manifest(cfg, layout, root)
    manifest, report, nn = analyse_duplicates(manifest, cfg)
    counts = check_expected_counts(manifest, cfg)
    write_manifest(manifest, cfg, counts, strays=strays)
    write_duplicates_report(report, nn, cfg)

    if not counts["match"]:
        log_event(
            logger,
            "manifest.expected_counts_mismatch",
            level=30,
            mismatches=counts["mismatches"],
            hint="inspect the manifest, then update dataset.expected_counts with a "
            "justification; make_splits refuses to run until they match",
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
