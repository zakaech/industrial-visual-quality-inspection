"""X0 complement — peak process memory of one training step per stage profile and batch size.

Run after dry_run_training.py; writes reports/phase3/dry_run_memory.json. Measures only.

Usage (from the repository root):
    python scripts/dry_run_memory.py [--config configs/default.yaml]
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import torch

from vqi.cli import add_common_args, bootstrap
from vqi.data.dataset import CanvasCache, class_counts, training_rows
from vqi.logging_utils import get_logger, log_event
from vqi.models.factory import build_model
from vqi.preprocessing.pipeline import Preprocessor
from vqi.provenance import provenance
from vqi.training.dryrun import STAGE_PROFILES, peak_rss_mb, time_train_step

logger = get_logger("scripts.dry_run_memory")


def main() -> int:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()
    cfg, root = bootstrap(args)
    d = cfg.training.dry_run
    seed = cfg.reproducibility.seed

    pre = Preprocessor.from_config(cfg)
    rows = training_rows(cfg, "train")
    n_pos, n_neg = class_counts(rows["label"])
    cache = CanvasCache(cfg, pre, "train", rows)
    cache.ensure(pre, root)
    arr = cache.open()
    torch.manual_seed(seed)
    model = build_model(cfg, n_pos, n_neg)

    report = {"rss_mb_after_model_build": peak_rss_mb(), "steps": []}
    for bs in d.batch_sizes:
        idx = np.random.default_rng(seed).choice(len(rows), size=bs, replace=False)
        x = torch.stack([pre.to_tensor(np.array(arr[int(i)])) for i in idx])
        y = torch.tensor(rows["label"].to_numpy()[idx], dtype=torch.float32)
        for profile, trainable in STAGE_PROFILES.items():
            before = peak_rss_mb()
            t_step = time_train_step(model, x, y, trainable, cfg, warmup=1, n_steps=1)
            after = peak_rss_mb()
            row = {
                "batch_size": bs,
                "profile": profile,
                "step_s": t_step,
                "peak_rss_mb_before": before,
                "peak_rss_mb_after": after,
            }
            report["steps"].append(row)
            log_event(logger, "memory.step", **row)
    report["peak_rss_mb_final"] = peak_rss_mb()
    report["provenance"] = provenance(cfg)
    out = cfg.paths.reports_dir / cfg.phase3.report_subdir / "dry_run_memory.json"
    out.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8", newline="\n")
    log_event(logger, "memory.done", path=str(out), peak_rss_mb=report["peak_rss_mb_final"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
