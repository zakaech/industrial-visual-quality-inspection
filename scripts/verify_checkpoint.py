"""Phase 4 step 2 — load the candidate checkpoint, cross-check its provenance and reproduce the
recorded validation scores deterministically. Stops on any divergence. Never loads the test.

Usage (from the repository root):
    python scripts/verify_checkpoint.py [--config configs/default.yaml]
"""

from __future__ import annotations

import argparse
import sys

from vqi.cli import add_common_args, bootstrap
from vqi.evaluation.final import VerificationError, verify_checkpoint
from vqi.logging_utils import get_logger, log_event

logger = get_logger("scripts.verify_checkpoint")


def main() -> int:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()
    cfg, root = bootstrap(args)
    try:
        verify_checkpoint(cfg, root)
    except VerificationError as exc:
        log_event(logger, "verify.failed", level=40, error=str(exc))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
