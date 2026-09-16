"""Staged trainer: freeze → unfreeze with per-group learning rates, cosine schedule with warm-up
per stage, early stopping on validation Average Precision, best/last checkpoints, JSON-lines
history. The validation split is used for selection only (never for gradients, never augmented).
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, TensorDataset

from vqi.config import Config, StageConfig
from vqi.data.dataset import CanvasCache, VqiDataset, build_split, make_loader
from vqi.evaluation.metrics import confusion_at, threshold_for_recall
from vqi.logging_utils import get_logger, log_event
from vqi.models.factory import (
    DefectClassifier,
    batchnorm_modules,
    param_groups,
    set_train_mode,
    set_trainable,
)
from vqi.preprocessing.augment import Augmenter
from vqi.preprocessing.pipeline import Preprocessor
from vqi.provenance import git_commit
from vqi.seed import make_generator
from vqi.training.features import frozen_features
from vqi.training.losses import build_sampler

logger = get_logger(__name__)

RECALL_TARGET_FOR_MONITORING = 0.95


class StageTransitionError(RuntimeError):
    """The stage's training mode alters the checkpoint outputs before any optimizer step."""


# --------------------------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------------------------- #


def cosine_with_warmup(total_steps: int, warmup_steps: int):
    """Per-step LR multiplier: linear warm-up then cosine decay to 0 at ``total_steps``."""

    def fn(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return fn


class EarlyStopper:
    def __init__(self, patience: int, min_delta: float, min_epochs: int, mode: str = "max") -> None:
        self.patience, self.min_delta, self.min_epochs = patience, min_delta, min_epochs
        self.sign = 1.0 if mode == "max" else -1.0
        self.best: float | None = None
        self.best_epoch = 0
        self.bad_epochs = 0

    def update(self, value: float, epoch: int) -> bool:
        """Returns True when ``value`` is a new best (improvement > min_delta)."""
        if self.best is None or self.sign * (value - self.best) > self.min_delta:
            self.best, self.best_epoch, self.bad_epochs = value, epoch, 0
            return True
        self.bad_epochs += 1
        return False

    def should_stop(self, epoch: int) -> bool:
        return epoch >= self.min_epochs and self.bad_epochs >= self.patience


def epoch_metrics(labels: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    thr = threshold_for_recall(labels, probs, RECALL_TARGET_FOR_MONITORING)
    c = confusion_at(labels, probs, thr)
    return {
        "val_average_precision": float(average_precision_score(labels, probs)),
        "val_precision_r095": c["precision"],
        "val_fpr_r095": c["fpr"],
        "val_threshold_r095": float(thr),
        "mean_val_prob": float(probs.mean()),
    }


@dataclass
class StageOutcome:
    name: str
    epochs_run: int = 0
    best_epoch: int = 0
    best_metric: float = float("nan")
    early_stopped: bool = False
    seconds: float = 0.0
    seconds_per_epoch: list[float] = field(default_factory=list)
    trainable: dict[str, int] = field(default_factory=dict)
    lr: dict[str, float] = field(default_factory=dict)
    best_checkpoint: str = ""
    last_checkpoint: str = ""
    history: list[dict[str, Any]] = field(default_factory=list)
    cached_features: bool = False
    feature_seconds: float | None = None
    transition_check: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        d.pop("history")
        return d


# --------------------------------------------------------------------------------------------- #
# Trainer
# --------------------------------------------------------------------------------------------- #


class Trainer:
    def __init__(
        self,
        cfg: Config,
        model: DefectClassifier,
        pre: Preprocessor,
        project_root: Path,
        run_dir: Path,
        loss_fn: nn.Module,
        sampler_name: str,
        pos_weight: float,
        use_cache: bool | None = None,
        batch_size: int | None = None,
        num_workers: int | None = None,
    ) -> None:
        self.cfg, self.model, self.pre, self.root = cfg, model, pre, project_root
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.loss_fn = loss_fn
        self.sampler_name = sampler_name
        self.pos_weight = pos_weight
        self.seed = cfg.reproducibility.seed
        self.device = cfg.training.device
        self.batch_size = batch_size or cfg.training.batch_size
        self.num_workers = cfg.training.num_workers if num_workers is None else num_workers
        self.use_cache = use_cache
        self.history_path = run_dir / "history.jsonl"
        self.augmenter = Augmenter.from_config(cfg)
        self.cache_seconds: dict[str, float | None] = {}
        self._datasets: dict[tuple[str, bool], tuple[VqiDataset, CanvasCache | None]] = {}

    # ------------------------------------------------------------------ data
    def dataset(self, split: str, augment: bool) -> VqiDataset:
        key = (split, augment)
        if key not in self._datasets:
            ds, cache, secs = build_split(
                self.cfg,
                split,
                self.pre,
                self.root,
                self.augmenter if augment else None,
                use_cache=self.use_cache,
            )
            self._datasets[key] = (ds, cache)
            if secs is not None:
                self.cache_seconds[split] = secs
        return self._datasets[key][0]

    def loaders(self, augment: bool) -> tuple[DataLoader, DataLoader]:
        train_ds = self.dataset("train", augment)
        val_ds = self.dataset("val", False)
        sampler = build_sampler(self.sampler_name, train_ds.labels, self.seed)
        train = make_loader(train_ds, self.batch_size, True, self.num_workers, self.seed, sampler)
        val = make_loader(val_ds, self.batch_size, False, self.num_workers, self.seed)
        return train, val

    # ------------------------------------------------------------------ evaluation
    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        self.model.eval()
        ids, probs, labels, losses = [], [], [], []
        for x, y, image_ids in loader:
            logits = self.model(x.to(self.device)).squeeze(1)
            losses.append(float(self.loss_fn(logits, y.to(self.device))) * len(y))
            probs.append(torch.sigmoid(logits).cpu().numpy())
            labels.append(y.numpy())
            ids.extend(image_ids)
        labels_arr = np.concatenate(labels)
        return (
            np.asarray(ids, dtype=str),
            np.concatenate(probs).astype(np.float64),
            labels_arr,
            float(sum(losses) / max(len(labels_arr), 1)),
        )

    # ------------------------------------------------------------------ checkpoints
    def save_checkpoint(self, path: Path, stage: str, epoch: int, metric: float) -> None:
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "stage": stage,
                "epoch": epoch,
                "metric": metric,
                "pooling": self.model.pooling,
                "preprocessor": self.pre.describe(),
                "config_hash": self.cfg.hash,
                "git_commit": git_commit(),
                "seed": self.seed,
                "pos_weight": self.pos_weight,
                "sampler": self.sampler_name,
            },
            path,
        )

    def load_checkpoint(self, path: Path) -> dict[str, Any]:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])
        return ckpt

    def _record(self, row: dict[str, Any]) -> None:
        with self.history_path.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(row, default=float) + "\n")

    # ------------------------------------------------------------------ stage transition
    @torch.no_grad()
    def transition_check(self, trainable: Sequence[str], tolerance: float = 1e-5) -> dict[str, Any]:
        """Before any optimizer step of a stage: the checkpoint we start from must produce the
        same validation outputs in the stage's *training mode* (BN policy applied, dropout
        disabled for the comparison) as in ``eval()``. A difference means the mode switch itself
        changes the network (e.g. BatchNorm using batch statistics) — the X2 incident.

        Returns the measured differences; raises :class:`StageTransitionError` on failure.
        """
        _, val_loader = self.loaders(False)
        self.model.eval()
        ref_feats, ref_logits = [], []
        for x, _, _ in val_loader:
            f = self.model.features(x.to(self.device))
            ref_feats.append(f)
            ref_logits.append(self.model.head_forward(f).squeeze(1))
        # stage training mode; head dropout switched off for a deterministic comparison
        set_train_mode(self.model, trainable, self.cfg.training.bn_policy)
        self.model.head.eval()
        max_feat, max_logit = 0.0, 0.0
        running_before = [m.running_mean.clone() for m in batchnorm_modules(self.model)]
        for (x, _, _), rf, rl in zip(val_loader, ref_feats, ref_logits, strict=True):
            f = self.model.features(x.to(self.device))
            max_feat = max(max_feat, float((f - rf).abs().max()))
            max_logit = max(
                max_logit, float((self.model.head_forward(f).squeeze(1) - rl).abs().max())
            )
        running_after = [m.running_mean for m in batchnorm_modules(self.model)]
        stats_drift = max(
            float((a - b).abs().max()) for a, b in zip(running_before, running_after, strict=True)
        )
        self.model.eval()
        result = {
            "n_images": int(sum(len(r) for r in ref_logits)),
            "bn_policy": self.cfg.training.bn_policy,
            "max_abs_feature_diff": max_feat,
            "max_abs_logit_diff": max_logit,
            "bn_running_mean_drift": stats_drift,
            "tolerance": tolerance,
            "passed": bool(max_feat <= tolerance and max_logit <= tolerance and stats_drift == 0.0),
        }
        log_event(logger, "stage.transition_check", **result)
        if not result["passed"]:
            raise StageTransitionError(
                f"training mode changes the checkpoint outputs before any update: {result}"
            )
        return result

    # ------------------------------------------------------------------ stages
    def run_stage(self, name: str) -> StageOutcome:
        stage = self.cfg.training.stage(name)
        outcome = StageOutcome(name=name, lr=dict(stage.lr), cached_features=stage.cached_features)
        outcome.trainable = set_trainable(self.model, stage.trainable)
        log_event(
            logger,
            "stage.start",
            stage=name,
            trainable=outcome.trainable["total_trainable"],
            lr=stage.lr,
            augment=stage.augment,
            cached_features=stage.cached_features,
        )
        t0 = time.perf_counter()
        if stage.cached_features:
            self._run_probe(stage, outcome)
        else:
            outcome.transition_check = self.transition_check(stage.trainable)
            self._record({"stage": name, "epoch": 0, "transition_check": outcome.transition_check})
            self._run_finetune(stage, outcome)
        outcome.seconds = time.perf_counter() - t0
        # continue from the best epoch of this stage
        if outcome.best_checkpoint:
            self.load_checkpoint(Path(outcome.best_checkpoint))
        log_event(
            logger,
            "stage.done",
            stage=name,
            epochs=outcome.epochs_run,
            best_epoch=outcome.best_epoch,
            best=round(outcome.best_metric, 4),
            seconds=round(outcome.seconds, 1),
        )
        return outcome

    def _epoch_end(
        self,
        stage: StageConfig,
        outcome: StageOutcome,
        stopper: EarlyStopper,
        epoch: int,
        train_loss: float,
        val: tuple,
        seconds: float,
        lr_now: dict[str, float],
    ) -> bool:
        _, probs, labels, val_loss = val
        m = epoch_metrics(labels, probs)
        row = {
            "stage": outcome.name,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **m,
            "lr": lr_now,
            "seconds": seconds,
        }
        outcome.history.append(row)
        outcome.seconds_per_epoch.append(seconds)
        self._record(row)
        monitor = m[self.cfg.training.early_stopping.monitor]
        improved = stopper.update(monitor, epoch)
        last = self.run_dir / f"stage_{outcome.name}_last.pt"
        self.save_checkpoint(last, outcome.name, epoch, monitor)
        outcome.last_checkpoint = str(last)
        if improved:
            best = self.run_dir / f"stage_{outcome.name}_best.pt"
            self.save_checkpoint(best, outcome.name, epoch, monitor)
            outcome.best_checkpoint = str(best)
            outcome.best_epoch, outcome.best_metric = epoch, monitor
        outcome.epochs_run = epoch
        log_event(
            logger,
            "epoch",
            stage=outcome.name,
            epoch=epoch,
            train_loss=round(train_loss, 4),
            val_loss=round(val_loss, 4),
            val_ap=round(m["val_average_precision"], 4),
            val_p_r095=round(m["val_precision_r095"], 4),
            seconds=round(seconds, 1),
            improved=improved,
        )
        if stopper.should_stop(epoch):
            outcome.early_stopped = True
            return True
        return False

    def _stopper(self, stage: StageConfig) -> EarlyStopper:
        es = self.cfg.training.early_stopping
        return EarlyStopper(stage.patience, es.min_delta, es.min_epochs, es.mode)

    # ---- full fine-tuning stage (B, C)
    def _run_finetune(self, stage: StageConfig, outcome: StageOutcome) -> None:
        train_loader, val_loader = self.loaders(stage.augment)
        optimizer = AdamW(
            param_groups(self.model, stage.lr, self.cfg.training.optimizer.weight_decay)
        )
        steps_per_epoch = len(train_loader)
        warmup = self.cfg.training.scheduler.warmup_epochs * steps_per_epoch
        total = stage.max_epochs * steps_per_epoch
        scheduler = LambdaLR(optimizer, cosine_with_warmup(total, warmup))
        stopper = self._stopper(stage)
        clip = self.cfg.training.grad_clip_norm
        for epoch in range(1, stage.max_epochs + 1):
            t0 = time.perf_counter()
            train_loader.dataset.set_epoch(epoch)
            set_train_mode(self.model, stage.trainable, self.cfg.training.bn_policy)
            running, n_seen = 0.0, 0
            for step, (x, y, _) in enumerate(train_loader):
                logits = self.model(x.to(self.device)).squeeze(1)
                loss = self.loss_fn(logits, y.to(self.device))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if clip:
                    nn.utils.clip_grad_norm_(
                        [p for g in optimizer.param_groups for p in g["params"]], clip
                    )
                optimizer.step()
                scheduler.step()
                running += loss.item() * len(y)
                n_seen += len(y)
                if step % self.cfg.training.log_every_n_steps == 0:
                    log_event(
                        logger,
                        "step",
                        stage=outcome.name,
                        epoch=epoch,
                        step=step,
                        loss=round(loss.item(), 4),
                    )
            val = self.evaluate(val_loader)
            lr_now = {g["name"]: g["lr"] for g in optimizer.param_groups}
            if self._epoch_end(
                stage,
                outcome,
                stopper,
                epoch,
                running / max(n_seen, 1),
                val,
                time.perf_counter() - t0,
                lr_now,
            ):
                break

    # ---- linear probe on cached frozen features (A)
    def _run_probe(self, stage: StageConfig, outcome: StageOutcome) -> None:
        train_ds = self.dataset("train", False)
        val_ds = self.dataset("val", False)
        train_loader = make_loader(train_ds, self.batch_size, False, self.num_workers, self.seed)
        val_loader = make_loader(val_ds, self.batch_size, False, self.num_workers, self.seed)
        tr = frozen_features(
            self.cfg, self.pre, self.model, "train", train_loader, train_ds.image_ids
        )
        va = frozen_features(self.cfg, self.pre, self.model, "val", val_loader, val_ds.image_ids)
        outcome.feature_seconds = (0.0 if tr.cached else tr.seconds) + (
            0.0 if va.cached else va.seconds
        )

        x_tr, y_tr = torch.from_numpy(tr.x), torch.from_numpy(tr.labels)
        x_va, y_va = torch.from_numpy(va.x), torch.from_numpy(va.labels)
        sampler = build_sampler(self.sampler_name, tr.labels, self.seed)
        probe_bs = min(64, len(y_tr))
        loader = DataLoader(
            TensorDataset(x_tr, y_tr),
            batch_size=probe_bs,
            shuffle=sampler is None,
            sampler=sampler,
            generator=make_generator(self.seed),
        )
        optimizer = AdamW(
            param_groups(self.model, stage.lr, self.cfg.training.optimizer.weight_decay)
        )
        steps_per_epoch = len(loader)
        scheduler = LambdaLR(
            optimizer,
            cosine_with_warmup(
                stage.max_epochs * steps_per_epoch,
                self.cfg.training.scheduler.warmup_epochs * steps_per_epoch,
            ),
        )
        stopper = self._stopper(stage)
        head = self.model.head
        for epoch in range(1, stage.max_epochs + 1):
            t0 = time.perf_counter()
            head.train()
            running, n_seen = 0.0, 0
            for xb, yb in loader:
                logits = self.model.head_forward(xb.to(self.device)).squeeze(1)
                loss = self.loss_fn(logits, yb.to(self.device))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                scheduler.step()
                running += loss.item() * len(yb)
                n_seen += len(yb)
            head.eval()
            with torch.no_grad():
                logits = self.model.head_forward(x_va.to(self.device)).squeeze(1)
                val_loss = float(self.loss_fn(logits, y_va.to(self.device)))
                probs = torch.sigmoid(logits).cpu().numpy().astype(np.float64)
            val = (va.image_ids, probs, va.labels, val_loss)
            lr_now = {g["name"]: g["lr"] for g in optimizer.param_groups}
            if self._epoch_end(
                stage,
                outcome,
                stopper,
                epoch,
                running / max(n_seen, 1),
                val,
                time.perf_counter() - t0,
                lr_now,
            ):
                break

    # ------------------------------------------------------------------ final scores
    def final_val_scores(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        _, val_loader = self.loaders(False)
        return self.evaluate(val_loader)
