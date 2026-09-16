"""Phase 2 part A: compare the enhancement variants visually and with the defect CNR.

Uses train-split images only. Outputs figures and CSV tables under reports/phase2/.

Usage (from the repository root):
    python scripts/compare_preprocessing.py [--config configs/default.yaml]
"""

from __future__ import annotations

import argparse
import sys

from vqi.cli import add_common_args, bootstrap
from vqi.preprocessing.compare import run_comparison


def main() -> int:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()
    cfg, root = bootstrap(args)
    run_comparison(cfg, root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
