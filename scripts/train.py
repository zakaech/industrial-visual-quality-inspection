"""Run Phase 3 CNN experiments declared in phase3.experiments (fit on train, evaluate on val).

The official test split is refused by the data loader. Run the dry-run (X0) first.

Usage (from the repository root):
    python scripts/train.py --only X1_linear_probe [--batch-size 16] [--num-workers 2]
"""

from __future__ import annotations

import argparse
import sys

from vqi.cli import add_common_args, bootstrap
from vqi.logging_utils import get_logger, log_event
from vqi.training.experiments import run_experiment, summarise

logger = get_logger("scripts.train")


def main() -> int:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--only", nargs="*", default=None, help="experiment ids to run (in order)")
    parser.add_argument("--batch-size", type=int, default=None, help="override training.batch_size")
    parser.add_argument("--num-workers", type=int, default=None, help="override num_workers")
    args = parser.parse_args()
    cfg, root = bootstrap(args)

    specs = [s for s in cfg.phase3.experiments if not args.only or s.id in args.only]
    if args.only:
        missing = set(args.only) - {s.id for s in specs}
        if missing:
            log_event(logger, "train.unknown_experiments", level=40, ids=sorted(missing))
            return 1
    for spec in specs:
        run_experiment(spec, cfg, root, batch_size=args.batch_size, num_workers=args.num_workers)
    summary = summarise(cfg)
    cols = [
        "id",
        "stages",
        "pos_weight",
        "sampler",
        "val_ap",
        "precision@r0.95",
        "fpr@r0.95",
        "delta_ap_vs_svm",
        "delta_p95_vs_svm",
        "epochs",
        "total_seconds",
    ]
    print(summary[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
