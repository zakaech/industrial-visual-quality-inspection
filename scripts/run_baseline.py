"""Run the classical-baseline experiment matrix (Phase 2, parts B–E).

Fits on the train split, evaluates on the validation split, never touches the official test set.
Outputs: reports/phase2/experiments.csv and reports/phase2/experiments/<id>.json (+ CSVs).

Usage (from the repository root):
    python scripts/run_baseline.py [--config configs/default.yaml] [--only E1_lbp] [--no-cache]
"""

from __future__ import annotations

import argparse
import sys

from vqi.baseline.experiments import run_all
from vqi.cli import add_common_args, bootstrap
from vqi.logging_utils import get_logger, log_event

logger = get_logger("scripts.run_baseline")


def main() -> int:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--only", nargs="*", default=None, help="experiment ids to (re)run")
    parser.add_argument("--no-cache", action="store_true", help="recompute cached features")
    args = parser.parse_args()
    cfg, root = bootstrap(args)

    summary = run_all(cfg, root, only=args.only, use_cache=not args.no_cache)
    cols = [
        "id",
        "features",
        "preprocessing",
        "pad_value",
        "classifier",
        "class_weight",
        "val_ap",
        "val_roc_auc",
        "precision@r0.95",
        "fpr@r0.95",
    ]
    log_event(logger, "baseline.summary", n=len(summary))
    print(summary[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
