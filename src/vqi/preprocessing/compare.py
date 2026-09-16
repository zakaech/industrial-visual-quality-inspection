"""Phase 2 part A: visual and objective comparison of the enhancement variants.

Uses **train-split images only**. Produces, under ``reports/phase2/``:

* ``figures/<family>_full.png`` / ``figures/<family>_zoom.png`` — representative images (healthy,
  small / median / large defect, dark, high- and low-contrast) under each variant of a family
  (contrast: equalize vs CLAHE; denoise: light vs strong; sharpen: unsharp vs Laplacian);
* ``figures/histograms.png`` — luminance histograms per variant (saturation shows up here);
* ``cnr_per_image.csv``, ``cnr_summary.csv``, ``figures/cnr_by_area.png`` — contrast-to-noise
  ratio of every defective train image under every variant, by defect-size stratum.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from vqi.config import Config  # noqa: E402
from vqi.data.manifest import read_manifest  # noqa: E402
from vqi.data.splits import read_splits  # noqa: E402
from vqi.logging_utils import get_logger, log_event  # noqa: E402
from vqi.preprocessing.io import decode_image, to_gray  # noqa: E402
from vqi.preprocessing.pipeline import Preprocessor  # noqa: E402
from vqi.preprocessing.quality import defect_cnr, image_stats  # noqa: E402
from vqi.provenance import provenance  # noqa: E402

logger = get_logger(__name__)

FAMILIES: dict[str, tuple[str, ...]] = {
    "contrast": ("none", "equalize", "clahe"),
    "denoise": ("none", "denoise_light", "denoise_strong"),
    "sharpen": ("none", "unsharp", "laplacian"),
}


@dataclass(frozen=True)
class Representative:
    role: str
    image_id: str
    image_path: str
    mask_path: str
    label: int
    defect_area_px: int


def train_rows(cfg: Config) -> pd.DataFrame:
    """Manifest rows of the *train* split (validation and test are never used for figures)."""
    manifest = read_manifest(cfg)
    splits = read_splits(cfg)
    train_ids = set(splits.loc[splits["split"] == "train", "image_id"])
    rows = manifest[manifest["image_id"].isin(train_ids)].copy()
    if rows.empty:
        raise ValueError("no train rows — run make_splits first")
    return rows


def select_representatives(cfg: Config, seed: int) -> list[Representative]:
    rows = train_rows(cfg)
    rep_cfg = cfg.baseline.representative_images
    healthy = rows[rows["label"] == 0]
    defective = rows[rows["label"] == 1].sort_values("defect_area_px")
    chosen: list[tuple[str, pd.Series]] = []

    sample = healthy.sample(n=rep_cfg.n_healthy, random_state=seed)
    chosen += [(f"healthy_{i + 1}", r) for i, (_, r) in enumerate(sample.iterrows())]

    small = defective[defective["defect_area_px"] <= rep_cfg.small_defect_max_area]
    chosen.append(("small_defect", (small if len(small) else defective).iloc[0]))
    median_area = defective["defect_area_px"].median()
    chosen.append(
        (
            "median_defect",
            defective.iloc[(defective["defect_area_px"] - median_area).abs().argmin()],
        )
    )
    chosen.append(("large_defect", defective.iloc[-1]))
    chosen.append(("darkest", rows.loc[rows["mean_luminance"].idxmin()]))
    chosen.append(("highest_contrast_healthy", healthy.loc[healthy["std_luminance"].idxmax()]))
    chosen.append(("lowest_contrast", rows.loc[rows["std_luminance"].idxmin()]))

    seen: set[str] = set()
    out = []
    for role, r in chosen:
        if r["image_id"] in seen:
            continue
        seen.add(r["image_id"])
        out.append(
            Representative(
                role,
                str(r["image_id"]),
                str(r["image_path"]),
                str(r["mask_path"]),
                int(r["label"]),
                int(r["defect_area_px"]),
            )
        )
    return out


# --------------------------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------------------------- #


def _canvas_and_mask(
    pre: Preprocessor, project_root: Path, rep: Representative
) -> tuple[np.ndarray, np.ndarray | None]:
    raw = decode_image(project_root / rep.image_path)
    canvas = pre(raw)
    mask = None
    if rep.label == 1 and rep.mask_path:
        m = cv2.imread(str(project_root / rep.mask_path), cv2.IMREAD_GRAYSCALE)
        mask = pre.transform_mask(m, raw.shape[0], raw.shape[1])
    return canvas, mask


def _zoom_box(mask: np.ndarray, side: int = 96) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask > 0)
    cy, cx = int(ys.mean()), int(xs.mean())
    h, w = mask.shape
    side = min(side, w)
    top = int(np.clip(cy - side // 2, 0, h - side))
    left = int(np.clip(cx - side // 2, 0, w - side))
    return top, left, top + side, left + side


def _draw_contour(rgb: np.ndarray, mask: np.ndarray, thickness: int = 1) -> np.ndarray:
    out = rgb.copy()
    contours, _ = cv2.findContours(
        (mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(out, contours, -1, (255, 0, 0), thickness)
    return out


def family_figures(
    cfg: Config, reps: list[Representative], project_root: Path, fig_dir: Path
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for family, variants in FAMILIES.items():
        pres = {v: Preprocessor.from_config(cfg, variant=v) for v in variants}
        # full images
        fig, axes = plt.subplots(
            len(reps), len(variants), figsize=(1.9 * len(variants), 4.3 * len(reps))
        )
        axes = np.atleast_2d(axes)
        for i, rep in enumerate(reps):
            for j, v in enumerate(variants):
                canvas, mask = _canvas_and_mask(pres[v], project_root, rep)
                shown = _draw_contour(canvas, mask) if mask is not None else canvas
                axes[i, j].imshow(shown)
                title = f"{v}" if i else f"{v}"
                axes[i, j].set_title(
                    f"{rep.role} {rep.image_id}\n{title}" if j == 0 else title, fontsize=7
                )
        for ax in axes.ravel():
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(
            f"Famille « {family} » — images du split train (contour du masque en rouge)",
            fontsize=10,
        )
        fig.tight_layout()
        p = fig_dir / f"{family}_full.png"
        fig.savefig(p, dpi=72, bbox_inches="tight")  # tall figure: keep the file small
        plt.close(fig)
        paths[f"{family}_full"] = p

        # zooms on defects
        defective = [r for r in reps if r.label == 1]
        fig, axes = plt.subplots(
            len(defective), len(variants), figsize=(2.6 * len(variants), 2.8 * len(defective))
        )
        axes = np.atleast_2d(axes)
        for i, rep in enumerate(defective):
            for j, v in enumerate(variants):
                canvas, mask = _canvas_and_mask(pres[v], project_root, rep)
                top, left, bottom, right = _zoom_box(mask)
                crop = _draw_contour(canvas, mask)[top:bottom, left:right]
                axes[i, j].imshow(crop)
                axes[i, j].set_title(
                    f"{rep.role} {rep.image_id} ({rep.defect_area_px} px)\n{v}" if j == 0 else v,
                    fontsize=7,
                )
        for ax in axes.ravel():
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(f"Famille « {family} » — zoom sur les défauts", fontsize=10)
        fig.tight_layout()
        p = fig_dir / f"{family}_zoom.png"
        fig.savefig(p, dpi=130, bbox_inches="tight")
        plt.close(fig)
        paths[f"{family}_zoom"] = p
    return paths


def histogram_figure(
    cfg: Config, reps: list[Representative], project_root: Path, fig_dir: Path
) -> Path:
    variants = list(cfg.preprocessing.enhancement.variants)
    picks = [r for r in reps if r.role in ("darkest", "median_defect", "highest_contrast_healthy")]
    fig, axes = plt.subplots(1, len(picks), figsize=(5.2 * len(picks), 3.6))
    axes = np.atleast_1d(axes)
    for ax, rep in zip(axes, picks, strict=True):
        raw = decode_image(project_root / rep.image_path)
        for v in variants:
            gray = to_gray(Preprocessor.from_config(cfg, variant=v)(raw))
            hist = np.bincount(gray.ravel(), minlength=256) / gray.size
            ax.plot(hist, linewidth=1, label=v)
        ax.set_title(f"{rep.role} {rep.image_id}", fontsize=9)
        ax.set_xlabel("luminance")
        ax.set_yscale("log")
    axes[0].legend(fontsize=7)
    fig.suptitle("Histogrammes de luminance après chaque variante (échelle log)", fontsize=10)
    fig.tight_layout()
    p = fig_dir / "histograms.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return p


# --------------------------------------------------------------------------------------------- #
# CNR analysis
# --------------------------------------------------------------------------------------------- #


def cnr_analysis(cfg: Config, project_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """CNR + saturation of every defective train image under every configured variant."""
    rows = train_rows(cfg)
    defective = rows[rows["label"] == 1]
    variants = list(cfg.baseline.cnr.variants)
    pres = {v: Preprocessor.from_config(cfg, variant=v) for v in variants}
    ring = cfg.baseline.cnr.ring_width
    bins = list(cfg.baseline.cnr.area_bins)
    labels = [f"[{bins[i]}, {bins[i + 1]})" for i in range(len(bins) - 1)]

    records = []
    for _, r in defective.iterrows():
        raw = decode_image(project_root / r["image_path"])
        mask = cv2.imread(str(project_root / r["mask_path"]), cv2.IMREAD_GRAYSCALE)
        for v in variants:
            canvas = pres[v](raw)
            m = pres[v].transform_mask(mask, raw.shape[0], raw.shape[1])
            gray = to_gray(canvas)
            records.append(
                {
                    "image_id": r["image_id"],
                    "defect_area_px": int(r["defect_area_px"]),
                    "variant": v,
                    "cnr": defect_cnr(gray, m, ring),
                    **image_stats(gray),
                }
            )
    per_image = pd.DataFrame(records)
    per_image["area_bin"] = pd.cut(
        per_image["defect_area_px"], bins=bins, right=False, labels=labels
    )
    base = per_image[per_image["variant"] == "none"].set_index("image_id")["cnr"]
    per_image["cnr_ratio_vs_none"] = per_image["cnr"] / per_image["image_id"].map(base)

    summary_rows = []
    for v in variants:
        sub = per_image[per_image["variant"] == v]
        for bin_label, grp in [("all", sub), *sub.groupby("area_bin", observed=True)]:
            summary_rows.append(
                {
                    "variant": v,
                    "area_bin": str(bin_label),
                    "n_images": int(len(grp)),
                    "median_cnr": float(grp["cnr"].median()),
                    "median_cnr_ratio_vs_none": float(grp["cnr_ratio_vs_none"].median()),
                    "share_improved": float((grp["cnr_ratio_vs_none"] > 1.0).mean()),
                    "mean_saturated_fraction": float(grp["saturated_fraction"].mean()),
                    "median_std": float(grp["std"].median()),
                }
            )
    summary = pd.DataFrame(summary_rows)
    return per_image, summary


def cnr_figure(summary: pd.DataFrame, fig_dir: Path) -> Path:
    sub = summary[summary["area_bin"] != "all"]
    bins = list(dict.fromkeys(sub["area_bin"]))
    variants = list(dict.fromkeys(sub["variant"]))
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2))
    width = 0.8 / len(variants)
    x = np.arange(len(bins))
    for k, v in enumerate(variants):
        vals = [
            float(
                sub[(sub["variant"] == v) & (sub["area_bin"] == b)][
                    "median_cnr_ratio_vs_none"
                ].iloc[0]
            )
            for b in bins
        ]
        axes[0].bar(x + k * width, vals, width, label=v)
        shares = [
            float(sub[(sub["variant"] == v) & (sub["area_bin"] == b)]["share_improved"].iloc[0])
            for b in bins
        ]
        axes[1].bar(x + k * width, shares, width, label=v)
    for ax, title in zip(
        axes,
        ("CNR médian relatif à « none » (1 = inchangé)", "part des images dont le CNR augmente"),
        strict=True,
    ):
        ax.set_xticks(x + width * (len(variants) - 1) / 2)
        ax.set_xticklabels(bins, rotation=20, fontsize=8)
        ax.set_xlabel("aire du défaut (px)")
        ax.set_title(title, fontsize=10)
        ax.axhline(1.0 if "relatif" in title else 0.5, color="black", linewidth=0.8, linestyle="--")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    p = fig_dir / "cnr_by_area.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return p


# --------------------------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------------------------- #


def run_comparison(cfg: Config, project_root: Path) -> dict[str, Any]:
    out_dir = cfg.paths.reports_dir / cfg.baseline.report_subdir
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    reps = select_representatives(cfg, cfg.reproducibility.seed)
    pd.DataFrame([r.__dict__ for r in reps]).to_csv(
        out_dir / "representative_images.csv", index=False, lineterminator="\n"
    )
    figures = family_figures(cfg, reps, project_root, fig_dir)
    figures["histograms"] = histogram_figure(cfg, reps, project_root, fig_dir)

    per_image, summary = cnr_analysis(cfg, project_root)
    per_image.to_csv(out_dir / "cnr_per_image.csv", index=False, lineterminator="\n")
    summary.to_csv(out_dir / "cnr_summary.csv", index=False, lineterminator="\n")
    figures["cnr_by_area"] = cnr_figure(summary, fig_dir)

    meta = provenance(
        cfg,
        representatives=[r.__dict__ for r in reps],
        figures={k: v.as_posix() for k, v in figures.items()},
        n_defective_train_images=int(per_image["image_id"].nunique()),
        variants=list(cfg.baseline.cnr.variants),
    )
    (out_dir / "compare_preprocessing.meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n"
    )
    log_event(logger, "compare.done", n_figures=len(figures), out=str(out_dir))
    return meta
