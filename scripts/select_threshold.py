"""Phase 4 steps 3-5 — validation score analysis, calibration diagnostics, threshold selection,
and (with --freeze) the frozen bundle. Validation only; the test split is never loaded.

Usage (from the repository root):
    python scripts/select_threshold.py                # analyse + select (writes reports/phase4/)
    python scripts/select_threshold.py --freeze       # then write the frozen bundle (models/final/)
"""

from __future__ import annotations

import argparse
import sys

from vqi.cli import add_common_args, bootstrap
from vqi.evaluation.final import FreezeError, analyse_validation, freeze_bundle
from vqi.logging_utils import get_logger, log_event

logger = get_logger("scripts.select_threshold")


def main() -> int:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--freeze", action="store_true", help="write the frozen bundle")
    parser.add_argument(
        "--calibration", default="raw", help="calibration decision recorded in the bundle"
    )
    args = parser.parse_args()
    cfg, _ = bootstrap(args)
    analysis = analyse_validation(cfg)
    sel = analysis["threshold_selection"]
    print(
        f"threshold={sel['threshold']:.6f} (min_recall={sel['threshold_min_recall']:.6f}, "
        f"next positive={sel['next_lower_positive_score']}) at_threshold={sel['at_threshold']}"
    )
    if args.freeze:
        try:
            freeze_bundle(cfg, args.calibration)
        except FreezeError as exc:
            log_event(logger, "freeze.refused", level=40, error=str(exc))
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
