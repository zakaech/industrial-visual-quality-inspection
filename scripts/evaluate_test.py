"""Phase 4 step 6 — the FIRST AND ONLY evaluation of the official test split with the frozen
bundle, followed by the error analysis and Grad-CAM (steps 7-8).

Guards: the bundle must be frozen (threshold.json); the script refuses to run if a test result
already exists. Nothing is tuned from the outcome.

Usage (from the repository root):
    python scripts/evaluate_test.py [--config configs/default.yaml]
"""

from __future__ import annotations

import argparse
import sys

from vqi.cli import add_common_args, bootstrap
from vqi.evaluation.errors import error_analysis
from vqi.evaluation.final import TestEvaluationError, evaluate_test_once
from vqi.inference.predictor import BundleError
from vqi.logging_utils import get_logger, log_event

logger = get_logger("scripts.evaluate_test")


def main() -> int:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()
    cfg, root = bootstrap(args)
    try:
        result = evaluate_test_once(cfg, root)
    except (TestEvaluationError, BundleError) as exc:
        log_event(logger, "test.refused", level=40, error=str(exc))
        return 1
    error_analysis(cfg, root)
    f = result["final"]["at_threshold"]
    print(
        f"TEST (n={result['n']}, pos={result['n_pos']}) "
        f"AP={result['final']['average_precision']:.4f} "
        f"ROC-AUC={result['final']['roc_auc']:.4f} @thr={result['threshold']:.6f}: "
        f"recall={f['recall']:.4f} precision={f['precision']:.4f} fpr={f['fpr']:.4f} "
        f"TP={f['tp']} FP={f['fp']} FN={f['fn']} TN={f['tn']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
