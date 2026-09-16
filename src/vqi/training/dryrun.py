# ruff: noqa: E501  (Markdown rendering lines)
"""X0 — CPU dry-run: *measure* before training anything.

Every number written by this module is a wall-clock or memory measurement of this machine; it
produces no quality metric. Measured: canvas-cache build and equality with the live preprocessor,
DataLoader throughput per ``num_workers`` (with and without cache), forward and forward+backward
step time per training stage and batch size, peak process memory, an epoch-time estimate derived
from those measurements, bit-level determinism of two identical short runs, and the photometric
saturation introduced by the augmentation.
"""

from __future__ import annotations

import ctypes
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW

from vqi.config import Config
from vqi.data.dataset import CanvasCache, build_split, class_counts, make_loader, training_rows
from vqi.logging_utils import get_logger, log_event
from vqi.models.factory import build_model, param_groups, set_train_mode, set_trainable
from vqi.preprocessing.augment import Augmenter, rng_for
from vqi.preprocessing.pipeline import Preprocessor
from vqi.provenance import provenance
from vqi.training.losses import build_loss

logger = get_logger(__name__)

STAGE_PROFILES: dict[str, tuple[str, ...]] = {
    "A_head_only_live": ("head",),
    "B_layer4": ("layer4", "head"),
    "C_layer3_4": ("layer3", "layer4", "head"),
    "full_unfrozen": ("stem", "layer1", "layer2", "layer3", "layer4", "head"),
}


# --------------------------------------------------------------------------------------------- #
# Memory
# --------------------------------------------------------------------------------------------- #


def peak_rss_mb() -> float | None:
    """Peak resident set size of this process in MB (Windows: PeakWorkingSetSize; POSIX: ru_maxrss)."""
    try:
        if sys.platform == "win32":

            class _Counters(ctypes.Structure):
                _fields_ = [
                    ("cb", ctypes.c_ulong),
                    ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = _Counters()
            counters.cb = ctypes.sizeof(_Counters)
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            # Windows 7+ exports the PSAPI functions from kernel32 as K32*; psapi.dll is a shim
            fn = getattr(ctypes.windll.kernel32, "K32GetProcessMemoryInfo", None) or (
                ctypes.windll.psapi.GetProcessMemoryInfo
            )
            fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(_Counters), ctypes.c_ulong]
            fn.restype = ctypes.c_int
            ok = fn(handle, ctypes.byref(counters), counters.cb)
            return float(counters.PeakWorkingSetSize) / 2**20 if ok else None
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return float(rss) / (2**20 if sys.platform == "darwin" else 2**10)
    except Exception:  # noqa: BLE001 — measurement must never break the run
        return None


# --------------------------------------------------------------------------------------------- #
# Timed pieces
# --------------------------------------------------------------------------------------------- #


def time_loader(loader, n_batches: int) -> dict[str, float]:
    it = iter(loader)
    t0 = time.perf_counter()
    next(it)  # first batch includes worker start-up
    first = time.perf_counter() - t0
    t1 = time.perf_counter()
    n = 0
    for _ in range(n_batches - 1):
        try:
            next(it)
            n += 1
        except StopIteration:
            break
    steady = (time.perf_counter() - t1) / max(n, 1)
    return {"first_batch_s": first, "steady_ms_per_batch": steady * 1000.0, "n_batches": n + 1}


def time_forward(model: nn.Module, x: torch.Tensor, warmup: int, n_steps: int) -> float:
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        t0 = time.perf_counter()
        for _ in range(n_steps):
            model(x)
    return (time.perf_counter() - t0) / n_steps


def time_train_step(
    model,
    x: torch.Tensor,
    y: torch.Tensor,
    trainable: tuple[str, ...],
    cfg: Config,
    warmup: int,
    n_steps: int,
) -> float:
    set_trainable(model, trainable)
    set_train_mode(model, trainable, cfg.training.bn_policy)
    lr = {g: 1e-4 for g in trainable}
    optimizer = AdamW(param_groups(model, lr, cfg.training.optimizer.weight_decay))
    loss_fn = build_loss(cfg.training.loss.name, 1.0, cfg.training.loss.focal_gamma)

    def step() -> None:
        loss = loss_fn(model(x).squeeze(1), y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    for _ in range(warmup):
        step()
    t0 = time.perf_counter()
    for _ in range(n_steps):
        step()
    return (time.perf_counter() - t0) / n_steps


def determinism_check(
    cfg: Config, n_pos: int, n_neg: int, x: torch.Tensor, y: torch.Tensor, steps: int
) -> dict[str, Any]:
    """Two identical short training runs (stage B recipe) from the same seed: are the losses and
    the resulting head weights bit-identical on this CPU?"""

    def run() -> tuple[list[float], torch.Tensor]:
        torch.manual_seed(cfg.reproducibility.seed)
        model = build_model(cfg, n_pos, n_neg)
        trainable = ("layer4", "head")
        set_trainable(model, trainable)
        set_train_mode(model, trainable, cfg.training.bn_policy)
        optimizer = AdamW(
            param_groups(model, {"layer4": 1e-4, "head": 1e-3}, cfg.training.optimizer.weight_decay)
        )
        loss_fn = build_loss(cfg.training.loss.name, 1.0, cfg.training.loss.focal_gamma)
        losses = []
        for _ in range(steps):
            loss = loss_fn(model(x).squeeze(1), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        return losses, model.head[1].weight.detach().clone()

    l1, w1 = run()
    l2, w2 = run()
    return {
        "steps": steps,
        "max_abs_loss_diff": float(np.max(np.abs(np.asarray(l1) - np.asarray(l2)))),
        "max_abs_head_weight_diff": float((w1 - w2).abs().max()),
        "bit_identical": bool(l1 == l2 and torch.equal(w1, w2)),
        "torch_threads": torch.get_num_threads(),
    }


def saturation_check(cache: CanvasCache, augmenter: Augmenter, n: int, seed: int) -> dict[str, Any]:
    arr = cache.open()
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(cache.rows), size=min(n, len(cache.rows)), replace=False)
    before, after, mean_shift = [], [], []
    for k, i in enumerate(idx):
        canvas = np.array(arr[int(i)])
        aug = augmenter(canvas, rng_for(seed, 1, k))
        before.append(float(((canvas == 0) | (canvas == 255)).mean()))
        after.append(float(((aug == 0) | (aug == 255)).mean()))
        mean_shift.append(float(aug.mean()) - float(canvas.mean()))
    return {
        "n_images": int(len(idx)),
        "saturated_fraction_before_mean": float(np.mean(before)),
        "saturated_fraction_after_mean": float(np.mean(after)),
        "saturated_fraction_after_max": float(np.max(after)),
        "mean_intensity_shift_abs_max": float(np.max(np.abs(mean_shift))),
    }


# --------------------------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------------------------- #


def run_dry_run(cfg: Config, project_root: Path) -> dict[str, Any]:
    d = cfg.training.dry_run
    seed = cfg.reproducibility.seed
    out_dir = cfg.paths.reports_dir / cfg.phase3.report_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "machine": {
            "platform": platform.platform(),
            "cpu": platform.processor(),
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "python": sys.version.split()[0],
        },
        "rss_mb_at_start": peak_rss_mb(),
    }
    pre = Preprocessor.from_config(cfg)
    train_rows = training_rows(cfg, "train")
    val_rows = training_rows(cfg, "val")
    n_pos, n_neg = class_counts(train_rows["label"])
    report["data"] = {
        "train": {"n": len(train_rows), "n_pos": n_pos, "n_neg": n_neg},
        "val": {"n": len(val_rows)},
        "test_used": False,
    }

    # 1) canvas cache: build time + equality with the live preprocessor
    caches = {}
    for split, rows in (("train", train_rows), ("val", val_rows)):
        cache = CanvasCache(cfg, pre, split, rows)
        secs = cache.ensure(pre, project_root)
        caches[split] = cache
        meta = json.loads(cache.meta_path.read_text(encoding="utf-8"))
        report.setdefault("cache", {})[split] = {
            "path": str(cache.path),
            "n": meta["n"],
            "bytes": int(cache.path.stat().st_size),
            "build_seconds": secs if secs is not None else meta["build_seconds"],
            "built_now": secs is not None,
            "ms_per_image": 1000.0
            * (secs if secs is not None else meta["build_seconds"])
            / meta["n"],
            "check_vs_live": cache.verify(pre, project_root, d.n_cache_check, seed),
        }
    log_event(logger, "dryrun.cache", **{k: v["check_vs_live"] for k, v in report["cache"].items()})

    # 2) loader throughput (train split, augmentation on) per num_workers, cache on / off
    augmenter = Augmenter.from_config(cfg)
    loader_rows = []
    positive_workers = [w for w in d.num_workers_options if w > 0]
    live_workers = {0, min(positive_workers)} if positive_workers else {0}
    for use_cache in (True, False):
        for workers in d.num_workers_options:
            if not use_cache and workers not in live_workers:
                continue  # live preprocessing is measured for two worker counts only
            ds, _, _ = build_split(cfg, "train", pre, project_root, augmenter, use_cache=use_cache)
            loader = make_loader(ds, cfg.training.batch_size, True, workers, seed)
            t = time_loader(loader, d.n_loader_batches)
            loader_rows.append(
                {
                    "cache": use_cache,
                    "num_workers": workers,
                    "batch_size": cfg.training.batch_size,
                    **t,
                }
            )
            log_event(
                logger,
                "dryrun.loader",
                cache=use_cache,
                workers=workers,
                ms_per_batch=round(t["steady_ms_per_batch"], 1),
            )
    report["loader"] = loader_rows

    # 3) compute: forward and train steps per stage profile and batch size
    torch.manual_seed(seed)
    model = build_model(cfg, n_pos, n_neg)
    compute_rows = []
    arr = caches["train"].open()
    for bs in d.batch_sizes:
        idx = np.random.default_rng(seed).choice(len(train_rows), size=bs, replace=False)
        x = torch.stack([pre.to_tensor(np.array(arr[int(i)])) for i in idx])
        y = torch.tensor(train_rows["label"].to_numpy()[idx], dtype=torch.float32)
        fwd = time_forward(model, x, d.warmup_steps, d.n_steps)
        row = {
            "batch_size": bs,
            "forward_eval_s_per_batch": fwd,
            "forward_eval_ms_per_image": 1000.0 * fwd / bs,
        }
        for profile, trainable in STAGE_PROFILES.items():
            t_step = time_train_step(model, x, y, trainable, cfg, d.warmup_steps, d.n_steps)
            row[f"train_step_s_{profile}"] = t_step
            row[f"train_ms_per_image_{profile}"] = 1000.0 * t_step / bs
            log_event(
                logger,
                "dryrun.compute",
                batch_size=bs,
                profile=profile,
                s_per_step=round(t_step, 3),
            )
        row["peak_rss_mb_after"] = peak_rss_mb()
        compute_rows.append(row)
    report["compute"] = compute_rows

    # 4) epoch estimates from the measured step times (train pass + val forward pass)
    estimates = []
    for row in compute_rows:
        bs = row["batch_size"]
        n_train_batches = int(np.ceil(len(train_rows) / bs))
        n_val_batches = int(np.ceil(len(val_rows) / bs))
        val_pass = n_val_batches * row["forward_eval_s_per_batch"]
        for profile in STAGE_PROFILES:
            train_pass = n_train_batches * row[f"train_step_s_{profile}"]
            estimates.append(
                {
                    "batch_size": bs,
                    "profile": profile,
                    "train_pass_s": train_pass,
                    "val_pass_s": val_pass,
                    "epoch_s": train_pass + val_pass,
                    "epoch_min": (train_pass + val_pass) / 60.0,
                }
            )
    report["epoch_estimates"] = estimates

    # 5) determinism of two identical short runs (stage B recipe, batch 16 or first size)
    bs = cfg.training.batch_size if cfg.training.batch_size in d.batch_sizes else d.batch_sizes[0]
    idx = np.random.default_rng(seed).choice(len(train_rows), size=bs, replace=False)
    x = torch.stack([pre.to_tensor(np.array(arr[int(i)])) for i in idx])
    y = torch.tensor(train_rows["label"].to_numpy()[idx], dtype=torch.float32)
    report["determinism"] = determinism_check(cfg, n_pos, n_neg, x, y, d.determinism_steps)
    log_event(logger, "dryrun.determinism", **report["determinism"])

    # 6) photometric saturation after augmentation
    report["saturation"] = saturation_check(caches["train"], augmenter, d.n_saturation_check, seed)

    report["peak_rss_mb"] = peak_rss_mb()
    report["provenance"] = provenance(cfg)
    (out_dir / "dry_run.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=float),
        encoding="utf-8",
        newline="\n",
    )
    (out_dir / "dry_run.md").write_text(
        render_markdown(report, cfg), encoding="utf-8", newline="\n"
    )
    log_event(logger, "dryrun.done", path=str(out_dir / "dry_run.json"))
    return report


# --------------------------------------------------------------------------------------------- #
# Markdown
# --------------------------------------------------------------------------------------------- #


def _f(x: Any, nd: int = 1) -> str:
    if x is None:
        return "n/d"
    if isinstance(x, bool):
        return "oui" if x else "non"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    return f"{float(x):.{nd}f}".replace(".", ",")


def render_markdown(r: dict[str, Any], cfg: Config) -> str:
    lines = [
        "# X0 — Dry-run CPU (mesures réelles)",
        "",
        f"Machine : {r['machine']['platform']} · {r['machine']['cpu']} · torch {r['machine']['torch']} "
        f"· {r['machine']['torch_threads']} threads · Python {r['machine']['python']}",
        f"Données : train {r['data']['train']['n']} ({r['data']['train']['n_pos']} positives), "
        f"val {r['data']['val']['n']} · test chargé : non",
        "",
        "## 1. Cache des canevas (sortie exacte du Preprocessor)",
        "",
        "| split | n | Mo | construit maintenant | s total | ms/image | vérifiés vs live | écarts | diff max |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for split, c in r["cache"].items():
        v = c["check_vs_live"]
        lines.append(
            f"| {split} | {c['n']} | {_f(c['bytes'] / 2**20, 0)} | {_f(c['built_now'])} | "
            f"{_f(c['build_seconds'])} | {_f(c['ms_per_image'])} | {v['n_checked']} | "
            f"{v['n_mismatch']} | {v['max_abs_diff']} |"
        )
    lines += [
        "",
        f"## 2. Chargement (train, augmentation ON, batch {cfg.training.batch_size})",
        "",
        "| cache | num_workers | 1er batch (s) | régime (ms/batch) | batchs mesurés |",
        "|---|---|---|---|---|",
    ]
    for row in r["loader"]:
        lines.append(
            f"| {_f(row['cache'])} | {row['num_workers']} | {_f(row['first_batch_s'], 2)} | "
            f"{_f(row['steady_ms_per_batch'])} | {row['n_batches']} |"
        )
    lines += [
        "",
        "## 3. Calcul (s par pas, batch complet)",
        "",
        "| batch | forward eval | A tête (live) | B layer4 | C layer3+4 | tout dégelé | RSS pic (Mo) |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in r["compute"]:
        lines.append(
            f"| {row['batch_size']} | {_f(row['forward_eval_s_per_batch'], 3)} | "
            f"{_f(row['train_step_s_A_head_only_live'], 3)} | {_f(row['train_step_s_B_layer4'], 3)} | "
            f"{_f(row['train_step_s_C_layer3_4'], 3)} | {_f(row['train_step_s_full_unfrozen'], 3)} | "
            f"{_f(row['peak_rss_mb_after'], 0)} |"
        )
    lines += [
        "",
        "## 4. Estimation d'époque dérivée des mesures (passe train + passe val)",
        "",
        "| batch | profil | train (s) | val (s) | époque (min) |",
        "|---|---|---|---|---|",
    ]
    for e in r["epoch_estimates"]:
        lines.append(
            f"| {e['batch_size']} | {e['profile']} | {_f(e['train_pass_s'], 0)} | "
            f"{_f(e['val_pass_s'], 0)} | {_f(e['epoch_min'])} |"
        )
    det = r["determinism"]
    sat = r["saturation"]
    lines += [
        "",
        "## 5. Déterminisme (deux runs identiques, recette B, "
        f"{det['steps']} pas, {det['torch_threads']} threads)",
        "",
        f"- bit-identiques : {_f(det['bit_identical'])} · écart max des pertes : "
        f"{det['max_abs_loss_diff']:.3e} · écart max des poids de la tête : "
        f"{det['max_abs_head_weight_diff']:.3e}",
        "",
        f"## 6. Saturation photométrique après augmentation ({sat['n_images']} canevas train)",
        "",
        f"- fraction de pixels saturés (0 ou 255) : avant {sat['saturated_fraction_before_mean']:.5f}, "
        f"après {sat['saturated_fraction_after_mean']:.5f} (max {sat['saturated_fraction_after_max']:.5f}) ; "
        f"décalage max de l'intensité moyenne : {_f(sat['mean_intensity_shift_abs_max'])}/255",
        "",
        f"RSS de pointe du processus : {_f(r['peak_rss_mb'], 0)} Mo.",
    ]
    return "\n".join(lines) + "\n"
