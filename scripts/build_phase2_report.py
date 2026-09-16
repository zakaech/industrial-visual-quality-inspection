"""Assemble reports/phase2/phase2_report.md from the Phase 2 artefacts (figures, CNR tables,
experiment results). Run after compare_preprocessing.py and run_baseline.py.

Usage (from the repository root):
    python scripts/build_phase2_report.py [--config configs/default.yaml]
"""

from __future__ import annotations

import argparse
import sys

from vqi.baseline.report import build_report
from vqi.cli import add_common_args, bootstrap


def main() -> int:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()
    cfg, _ = bootstrap(args)
    build_report(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
