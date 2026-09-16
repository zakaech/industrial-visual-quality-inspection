"""Phase 3 experiment runner: one CNN experiment = a staged training recipe declared in
``phase3.experiments``. Writes ``runs/<run_id>/`` (checkpoints, history, frozen config) and
``reports/phase3/experiments/<id>.json`` with the same metric structure as Phase 2, plus the
paired comparison with the frozen Phase 2 reference. The official test split is never loaded.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from vqi.config import Config, Phase3ExperimentSpec, dump_config
from vqi.data.dataset import class_counts, training_rows
from vqi.evaluation.compare import align_scores, paired_bootstrap, recall_by_area
from vqi.evaluation.metrics import evaluate_scores, threshold_for_recall, threshold_sweep
from vqi.logging_utils import get_logger, log_event
from vqi.models.factory import build_model, model_summary
from vqi.preprocessing.augment import Augmenter
from vqi.preprocessing.pipeline import Preprocessor
from vqi.provenance import provenance
from vqi.training.loop import Trainer
from vqi.training.losses import build_loss, resolve_pos_weight

logger = get_logger(__name__)


def report_dir(cfg: Config) -> Path:
    return cfg.paths.reports_dir / cfg.phase3.report_subdir


def experiment_json(cfg: Config, exp_id: str) -> Path:
    return report_dir(cfg) / "experiments" / f"{exp_id}.json"


def load_result(cfg: Config, exp_id: str) -> dict[str, Any]:
    p = experiment_json(cfg, exp_id)
    if not p.exists():
        raise FileNotFoundError(f"experiment {exp_id} has no result yet ({p})")
    return json.loads(p.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------------------------- #
# Winner selection (criteria fixed before the runs: primary precision@recall≥0.95, then AP)
# --------------------------------------------------------------------------------------------- #


def selection_key(result: dict[str, Any], cfg: Config) -> tuple[float, float]:
    m = result["metrics"]
    by_target = {round(t["recall_target"], 2): t for t in m["at_recall_targets"]}
    primary, secondary = cfg.phase3.selection.primary, cfg.phase3.selection.secondary
    if primary != "precision_at_recall_0.95" or secondary != "average_precision":
        raise ValueError("selection criteria differ from the validated ones")
    return float(by_target[0.95]["precision"]), float(m["average_precision"])


def select_winner(cfg: Config, candidates: tuple[str, ...]) -> tuple[str, dict[str, Any]]:
    results = {cid: load_result(cfg, cid) for cid in candidates}
    ranked = sorted(results, key=lambda cid: selection_key(results[cid], cfg), reverse=True)
    return ranked[0], results[ranked[0]]


# --------------------------------------------------------------------------------------------- #
# Phase 2 reference
# --------------------------------------------------------------------------------------------- #


def phase2_reference(cfg: Config) -> tuple[str, pd.DataFrame, dict[str, Any]]:
    ref_id = cfg.phase3.reference_experiment
    base = cfg.paths.reports_dir / cfg.baseline.report_subdir / "experiments"
    scores = pd.read_csv(base / f"{ref_id}_val_scores.csv", dtype={"image_id": str})
    result = json.loads((base / f"{ref_id}.json").read_text(encoding="utf-8"))
    return ref_id, scores, result


# --------------------------------------------------------------------------------------------- #
# One experiment
# --------------------------------------------------------------------------------------------- #


def run_experiment(
    spec: Phase3ExperimentSpec,
    cfg: Config,
    project_root: Path,
    batch_size: int | None = None,
    num_workers: int | None = None,
) -> dict[str, Any]:
    torch.manual_seed(cfg.reproducibility.seed)
    np.random.seed(cfg.reproducibility.seed)
    t_start = time.perf_counter()

    train_rows = training_rows(cfg, "train")
    val_rows = training_rows(cfg, "val")
    n_pos, n_neg = class_counts(train_rows["label"])

    # inherit the imbalance recipe of the winner when continuing from it
    pos_weight_spec, sampler_name, winner_id, winner_ckpt = (
        spec.pos_weight,
        spec.sampler,
        None,
        None,
    )
    if spec.start_from_winner_of:
        winner_id, winner_result = select_winner(cfg, spec.start_from_winner_of)
        winner_ckpt = winner_result["run"]["best_checkpoint"]
        if pos_weight_spec == "inherit":
            pos_weight_spec = winner_result["spec"]["pos_weight"]
        if sampler_name == "inherit":
            sampler_name = winner_result["spec"]["sampler"]
    pos_weight = resolve_pos_weight(pos_weight_spec, n_pos, n_neg)
    loss_fn = build_loss(cfg.training.loss.name, pos_weight, cfg.training.loss.focal_gamma)

    pre = Preprocessor.from_config(cfg)
    model = build_model(cfg, n_pos, n_neg, pooling=spec.pooling)
    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{spec.id}_{cfg.hash}"
    run_dir = cfg.paths.runs_dir / run_id
    trainer = Trainer(
        cfg,
        model,
        pre,
        project_root,
        run_dir,
        loss_fn,
        sampler_name,
        pos_weight,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    if winner_ckpt:
        trainer.load_checkpoint(Path(winner_ckpt))
    dump_config(cfg, run_dir / "config.yaml")

    stages = [trainer.run_stage(name).as_dict() for name in spec.stages]
    history = [
        row
        for line in trainer.history_path.read_text(encoding="utf-8").splitlines()
        for row in [json.loads(line)]
    ]

    ids, probs, labels, val_loss = trainer.final_val_scores()
    ev = cfg.evaluation
    metrics = evaluate_scores(
        labels,
        probs,
        ev.report_recall_targets,
        0.5,
        ev.bootstrap.n_resamples,
        ev.bootstrap.confidence_level,
        cfg.reproducibility.seed,
    )
    thr95 = threshold_for_recall(labels, probs, 0.95)
    areas = val_rows.set_index("image_id").loc[list(ids), "defect_area_px"].to_numpy()
    metrics["recall_by_defect_area_at_r095"] = recall_by_area(
        labels, probs, areas, thr95, cfg.phase3.defect_area_bins
    )
    metrics["val_loss"] = val_loss

    ref_id, ref_scores, ref_result = phase2_reference(cfg)
    ids_aligned, a, b = align_scores(ref_scores["image_id"], ref_scores["score"], ids, probs)
    y_aligned = (
        val_rows.set_index("image_id").loc[list(ids_aligned), "label"].to_numpy().astype(int)
    )
    comparison = paired_bootstrap(
        y_aligned,
        a,
        b,
        cfg.phase3.paired_bootstrap.n_resamples,
        cfg.phase3.paired_bootstrap.confidence_level,
        cfg.reproducibility.seed,
    )
    comparison["reference"] = ref_id
    comparison["reference_ap"] = ref_result["metrics"]["average_precision"]

    best_ckpt = next((s["best_checkpoint"] for s in reversed(stages) if s["best_checkpoint"]), "")
    result = {
        "spec": {
            "id": spec.id,
            "stages": list(spec.stages),
            "pos_weight": pos_weight_spec,
            "pos_weight_value": pos_weight,
            "sampler": sampler_name,
            "start_from_winner_of": list(spec.start_from_winner_of or []),
            "winner": winner_id,
            "pooling": model.pooling,
            "batch_size": trainer.batch_size,
            "num_workers": trainer.num_workers,
            "augmentation": Augmenter.from_config(cfg).describe(),
            "loss": cfg.training.loss.name,
            "bn_policy": cfg.training.bn_policy,
        },
        "preprocessor": pre.describe(),
        "model": model_summary(model),
        "data": {
            "train": {"n": int(len(train_rows)), "n_pos": n_pos, "n_neg": n_neg},
            "val": {"n": int(len(val_rows)), "n_pos": int((val_rows["label"] == 1).sum())},
            "evaluated_on": "val",
            "test_used": False,
        },
        "run": {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "best_checkpoint": best_ckpt,
            "stages": stages,
            "cache_build_seconds": trainer.cache_seconds,
        },
        "history": history,
        "timing": {
            "total_seconds": time.perf_counter() - t_start,
            "seconds_per_epoch_by_stage": {s["name"]: s["seconds_per_epoch"] for s in stages},
        },
        "metrics": metrics,
        "comparison_vs_phase2": comparison,
        "provenance": provenance(cfg, torch=torch.__version__),
    }
    out = experiment_json(cfg, spec.id)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=float),
        encoding="utf-8",
        newline="\n",
    )
    (run_dir / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=float),
        encoding="utf-8",
        newline="\n",
    )
    pd.DataFrame({"image_id": ids, "label": labels, "score": probs}).to_csv(
        out.parent / f"{spec.id}_val_scores.csv", index=False, lineterminator="\n"
    )
    threshold_sweep(labels, probs).to_csv(
        out.parent / f"{spec.id}_threshold_sweep.csv", index=False, lineterminator="\n"
    )
    log_event(
        logger,
        "experiment.done",
        id=spec.id,
        val_ap=round(metrics["average_precision"], 4),
        precision_r095=round(selection_key(result, cfg)[0], 4),
        delta_ap_vs_phase2=round(comparison["delta_average_precision"]["point"], 4),
        seconds=round(result["timing"]["total_seconds"], 1),
    )
    return result


def summarise(cfg: Config) -> pd.DataFrame:
    rows = []
    for spec in cfg.phase3.experiments:
        p = experiment_json(cfg, spec.id)
        if not p.exists():
            continue
        r = json.loads(p.read_text(encoding="utf-8"))
        m = r["metrics"]
        by_t = {round(t["recall_target"], 2): t for t in m["at_recall_targets"]}
        cmp_ = r["comparison_vs_phase2"]
        rows.append(
            {
                "id": spec.id,
                "stages": "→".join(r["spec"]["stages"]),
                "pos_weight": r["spec"]["pos_weight_value"],
                "sampler": r["spec"]["sampler"],
                "pooling": r["spec"]["pooling"],
                "winner_of": r["spec"]["winner"],
                "val_ap": m["average_precision"],
                "val_ap_ci_low": m["average_precision_ci"]["low"],
                "val_ap_ci_high": m["average_precision_ci"]["high"],
                "val_roc_auc": m["roc_auc"],
                "precision@r0.90": by_t[0.9]["precision"],
                "precision@r0.95": by_t[0.95]["precision"],
                "fpr@r0.95": by_t[0.95]["fpr"],
                "precision@r0.98": by_t[0.98]["precision"],
                "recall@0.5": m["at_default_threshold"]["recall"],
                "precision@0.5": m["at_default_threshold"]["precision"],
                "f1@0.5": m["at_default_threshold"]["f1"],
                "mean_val_prob": r["history"][-1]["mean_val_prob"],
                "delta_ap_vs_svm": cmp_["delta_average_precision"]["point"],
                "delta_ap_ci_low": cmp_["delta_average_precision"]["low"],
                "delta_ap_ci_high": cmp_["delta_average_precision"]["high"],
                "delta_p95_vs_svm": cmp_["delta_precision_at_recall_0.95"]["point"],
                "delta_p95_ci_low": cmp_["delta_precision_at_recall_0.95"]["low"],
                "delta_p95_ci_high": cmp_["delta_precision_at_recall_0.95"]["high"],
                "epochs": sum(s["epochs_run"] for s in r["run"]["stages"]),
                "total_seconds": r["timing"]["total_seconds"],
                "run_id": r["run"]["run_id"],
            }
        )
    df = pd.DataFrame(rows)
    out = report_dir(cfg) / "experiments.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, lineterminator="\n")
    return df
