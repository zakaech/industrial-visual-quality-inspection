"""Create the train / validation / test partition from the manifest (decision D3).

Refuses to run if the manifest's expected-count check failed: a partition built on a dataset
that does not match its documented statistics is not reproducible knowledge.

Usage (from the repository root):
    python scripts/make_splits.py [--config configs/default.yaml]
"""

from __future__ import annotations

import argparse
import sys

from vqi.cli import add_common_args, bootstrap
from vqi.data.manifest import read_manifest, read_manifest_meta
from vqi.data.splits import make_splits, write_splits
from vqi.logging_utils import get_logger, log_event

logger = get_logger("scripts.make_splits")


def main() -> int:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()
    cfg, _ = bootstrap(args)

    meta = read_manifest_meta(cfg)
    if not meta["counts_check"]["match"]:
        log_event(
            logger,
            "splits.refused",
            level=40,
            reason="manifest counts do not match dataset.expected_counts",
            mismatches=meta["counts_check"]["mismatches"],
        )
        return 1

    manifest = read_manifest(cfg)
    splits = make_splits(manifest, cfg)
    write_splits(splits, manifest, cfg, manifest_sha256=meta["manifest_sha256"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
