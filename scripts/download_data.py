"""Download the dataset archive, verify its SHA-256 and extract it under data/raw/.

Usage (from the repository root):
    python scripts/download_data.py [--config configs/default.yaml] [--force-download] [--overwrite]
"""

from __future__ import annotations

import argparse
import sys

from vqi.cli import add_common_args, bootstrap
from vqi.data.download import IntegrityError, ensure_dataset
from vqi.logging_utils import get_logger, log_event

logger = get_logger("scripts.download_data")


def main() -> int:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--force-download", action="store_true", help="re-download the archive")
    parser.add_argument(
        "--overwrite", action="store_true", help="delete and re-extract an existing extraction"
    )
    args = parser.parse_args()
    cfg, _ = bootstrap(args)
    try:
        layout = ensure_dataset(cfg, force_download=args.force_download, overwrite=args.overwrite)
    except IntegrityError as exc:
        log_event(logger, "download.failed", level=40, error=str(exc))
        return 1
    log_event(logger, "download.ready", root=str(layout.root))
    return 0


if __name__ == "__main__":
    sys.exit(main())
