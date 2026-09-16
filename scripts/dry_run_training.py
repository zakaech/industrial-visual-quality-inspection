"""X0 — CPU dry-run before any training: real timings, memory, determinism, cache check.

Produces reports/phase3/dry_run.json and dry_run.md. Trains nothing beyond a few timed steps.

Usage (from the repository root):
    python scripts/dry_run_training.py [--config configs/default.yaml]
"""

from __future__ import annotations

import argparse
import sys

from vqi.cli import add_common_args, bootstrap
from vqi.training.dryrun import run_dry_run


def main() -> int:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()
    cfg, root = bootstrap(args)
    run_dry_run(cfg, root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
