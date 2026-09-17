"""Assemble reports/phase4/phase4_report.md from the Phase 4 artefacts (validation analysis,
frozen bundle, single test evaluation, error analysis, Grad-CAM).

Usage (from the repository root):
    python scripts/build_phase4_report.py [--config configs/default.yaml]
"""

from __future__ import annotations

import argparse
import sys

from vqi.cli import add_common_args, bootstrap
from vqi.evaluation.report_phase4 import build_report


def main() -> int:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()
    cfg, _ = bootstrap(args)
    build_report(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
