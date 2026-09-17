"""Phase 4 steps 7-8 — error analysis and Grad-CAM on the (already evaluated) official test set.

Everything here is post-hoc analysis of frozen decisions: nothing is tuned, nothing is re-scored
with a different threshold. Masks are read for display and for measuring where the Grad-CAM mass
falls — never as model input.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from vqi.config import Config  # noqa: E402
from vqi.data.manifest import read_manifest  # noqa: E402
from vqi.evaluation.final import report_dir  # noqa: E402
from vqi.evaluation.gradcam import GradCAM, cam_mask_overlap, overlay  # noqa: E402
from vqi.inference.predictor import Predictor  # noqa: E402
from vqi.logging_utils import get_logger, log_event  # noqa: E402
from vqi.preprocessing.io import decode_image  # noqa: E402
from vqi.provenance import provenance  # noqa: E402

logger = get_logger(__name__)

STAT_COLS = ["mean_luminance", "std_luminance", "channel_spread"]
DEFECT_COLS = ["defect_area_px", "defect_bbox_w", "defect_bbox_h", "n_defect_components"]


def _load(
    cfg: Config, project_root: Path, image_id: str, manifest: pd.DataFrame, pre
) -> tuple[np.ndarray, np.ndarray | None]:
    row = manifest.loc[image_id]
    raw = decode_image(project_root / row["image_path"])
    canvas = pre(raw)
    mask = None
    if row["label"] == 1 and row["mask_path"]:
        m = cv2.imread(str(project_root / row["mask_path"]), cv2.IMREAD_GRAYSCALE)
        mask = pre.transform_mask(m, raw.shape[0], raw.shape[1])
    return canvas, mask


def _contour(rgb: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    if mask is None:
        return rgb
    out = rgb.copy()
    contours, _ = cv2.findContours(
        (mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(out, contours, -1, (255, 0, 0), 1)
    return out


def _zoom(rgb: np.ndarray, mask: np.ndarray, side: int = 96) -> np.ndarray:
    ys, xs = np.nonzero(mask > 0)
    cy, cx = int(ys.mean()), int(xs.mean())
    h, w = mask.shape
    side = min(side, w)
    top = int(np.clip(cy - side // 2, 0, h - side))
    left = int(np.clip(cx - side // 2, 0, w - side))
    return _contour(rgb, mask)[top : top + side, left : left + side]


def _panel(items: list[dict[str, Any]], path: Path, title: str, with_zoom: bool) -> Path | None:
    if not items:
        return None
    n = len(items)
    rows = 2 if with_zoom else 1
    fig, axes = plt.subplots(rows, n, figsize=(1.9 * n, 4.6 * rows), squeeze=False)
    for j, it in enumerate(items):
        axes[0, j].imshow(_contour(it["canvas"], it["mask"]))
        axes[0, j].set_title(it["title"], fontsize=7)
        if with_zoom:
            if it["mask"] is not None:
                axes[1, j].imshow(_zoom(it["canvas"], it["mask"]))
            axes[1, j].set_title("zoom", fontsize=7)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


def _medians(df: pd.DataFrame, cols: list[str]) -> dict[str, float | None]:
    return {c: (float(df[c].median()) if len(df) else None) for c in cols if c in df.columns}


def error_analysis(cfg: Config, project_root: Path) -> dict[str, Any]:
    out_dir = report_dir(cfg)
    fig_dir = out_dir / "figures"
    scores = pd.read_csv(out_dir / "test_scores.csv", dtype={"image_id": str})
    manifest = read_manifest(cfg).set_index("image_id")
    predictor = Predictor.from_bundle(cfg)
    pre = predictor.pre
    ea = cfg.phase4.error_analysis

    groups = {k: scores[scores["outcome"] == k] for k in ("FN", "FP", "TP", "TN")}
    selection = {
        "FN": groups["FN"].sort_values("score", ascending=False).head(ea.n_fn_examples),
        "FP": groups["FP"].sort_values("score", ascending=False).head(ea.n_fp_examples),
        "TP": groups["TP"].sort_values("score").head(ea.n_tp_examples),  # the least confident TPs
        "TN": groups["TN"].sort_values("score", ascending=False).head(ea.n_tn_examples),
    }
    panels: dict[str, str | None] = {}
    loaded: dict[str, list[dict[str, Any]]] = {}
    for k, sub in selection.items():
        items = []
        for _, r in sub.iterrows():
            canvas, mask = _load(cfg, project_root, r["image_id"], manifest, pre)
            area = f", {int(r['defect_area_px'])} px" if r["label"] == 1 else ""
            items.append(
                {
                    "image_id": r["image_id"],
                    "canvas": canvas,
                    "mask": mask,
                    "score": float(r["score"]),
                    "title": f"{k} {r['image_id']}\np={r['score']:.3f}{area}",
                }
            )
        loaded[k] = items
        titles = {
            "FN": "Faux négatifs (test) — défauts manqués au seuil gelé",
            "FP": "Faux positifs (test) — scores les plus élevés",
            "TP": "Vrais positifs (test) — les moins confiants",
            "TN": "Vrais négatifs (test) — scores les plus élevés sous le seuil",
        }
        p = _panel(
            items, fig_dir / f"test_{k.lower()}_panel.png", titles[k], with_zoom=k in ("FN", "TP")
        )
        panels[k] = p.as_posix() if p else None

    comparison = {
        "FP_vs_TN_medians": {
            "FP": _medians(groups["FP"], STAT_COLS),
            "TN": _medians(groups["TN"], STAT_COLS),
            "n_FP": int(len(groups["FP"])),
            "n_TN": int(len(groups["TN"])),
        },
        "FN_vs_TP_medians": {
            "FN": _medians(groups["FN"], DEFECT_COLS + ["std_luminance", "score"]),
            "TP": _medians(groups["TP"], DEFECT_COLS + ["std_luminance", "score"]),
            "n_FN": int(len(groups["FN"])),
            "n_TP": int(len(groups["TP"])),
        },
        "FP_score_quantiles": {q: float(groups["FP"]["score"].quantile(q)) for q in (0.5, 0.9, 1.0)}
        if len(groups["FP"])
        else {},
        "FP_share_above_0.5": float((groups["FP"]["score"] >= 0.5).mean())
        if len(groups["FP"])
        else None,
    }

    # Grad-CAM on the selected examples
    gc = cfg.phase4.gradcam
    cam_model = GradCAM(predictor.model, gc.target_layer)
    cam_rows, cam_panels = [], {}
    for k, items in loaded.items():
        picks = items[: gc.n_per_category]
        if not picks:
            continue
        x = torch.stack([pre.to_tensor(it["canvas"]) for it in picks])
        cams, probs = cam_model(x)
        fig, axes = plt.subplots(2, len(picks), figsize=(1.9 * len(picks), 9), squeeze=False)
        for j, (it, cam, pr) in enumerate(zip(picks, cams, probs, strict=True)):
            ov = overlay(it["canvas"], cam)
            axes[0, j].imshow(_contour(it["canvas"], it["mask"]))
            axes[0, j].set_title(it["title"], fontsize=7)
            axes[1, j].imshow(_contour(ov, it["mask"]))
            info = {
                "outcome": k,
                "image_id": it["image_id"],
                "probability": float(pr),
                "cam_peak_row_col": [
                    int(v) for v in np.unravel_index(int(cam.argmax()), cam.shape)
                ],
            }
            if it["mask"] is not None:
                ovl = cam_mask_overlap(cam, it["mask"])
                info.update(ovl)
                share, area = ovl["cam_mass_in_mask"], ovl["mask_area_share"]
                axes[1, j].set_title(
                    f"CAM dans masque {share:.2f}\n(masque {area:.3f} de l'image)", fontsize=7
                )
            else:
                axes[1, j].set_title("Grad-CAM", fontsize=7)
            cam_rows.append(info)
        for ax in axes.ravel():
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(f"Grad-CAM ({gc.target_layer}) — {k} (test)", fontsize=10)
        fig.tight_layout()
        p = fig_dir / f"test_gradcam_{k.lower()}.png"
        fig.savefig(p, dpi=110, bbox_inches="tight")
        plt.close(fig)
        cam_panels[k] = p.as_posix()
    cam_model.close()

    cam_df = pd.DataFrame(cam_rows)
    cam_summary = {}
    for k in ("TP", "FN"):
        sub = (
            cam_df[(cam_df["outcome"] == k) & cam_df["cam_mass_in_mask"].notna()]
            if len(cam_df)
            else cam_df
        )
        if len(sub):
            cam_summary[k] = {
                "n": int(len(sub)),
                "median_cam_mass_in_mask": float(sub["cam_mass_in_mask"].median()),
                "median_concentration_ratio": float(sub["concentration_ratio"].median()),
                "share_peak_in_mask": float(sub["peak_in_mask"].mean()),
            }
    payload = {
        "counts": {k: int(len(v)) for k, v in groups.items()},
        "examples_shown": {k: [it["image_id"] for it in v] for k, v in loaded.items()},
        "panels": panels,
        "comparison": comparison,
        "gradcam": {
            "target_layer": gc.target_layer,
            "per_image": cam_rows,
            "summary": cam_summary,
            "panels": cam_panels,
            "limits": "résolution d'une cellule ≈ 32×32 px ; carte de sensibilité du logit, "
            "pas une segmentation ; dépend de la couche choisie ; ne dit rien de la calibration",
        },
        "test_used": True,
        "provenance": provenance(cfg),
    }
    (out_dir / "error_analysis.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=float),
        encoding="utf-8",
        newline="\n",
    )
    log_event(logger, "errors.analysed", **payload["counts"], gradcam_images=len(cam_rows))
    return payload
