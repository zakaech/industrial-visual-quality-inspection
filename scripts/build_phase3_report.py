"""Assemble reports/phase3/phase3_report.md (+ figures) from the Phase 3 experiment JSONs.

Usage (from the repository root):
    python scripts/build_phase3_report.py [--config configs/default.yaml]
"""

from __future__ import annotations

import argparse
import sys

from vqi.cli import add_common_args, bootstrap
from vqi.training.report import build_report


def main() -> int:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()
    cfg, _ = bootstrap(args)
    build_report(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
