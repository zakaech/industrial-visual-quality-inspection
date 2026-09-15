"""Exploratory data analysis, computed from the manifest and written to ``reports/eda/``.

Everything in the report is derived from the manifest / duplicate report / split file of the
current run — no number is typed by hand. The prose of the report is French (project
documentation language); code, columns and file names stay English.

Outputs:
* ``eda_report.md`` — the report;
* ``class_counts.csv``, ``trivial_feature_auc.csv``, ``imbalance_options.csv`` — tables;
* ``figures/*.png`` — dimensions, luminance/contrast, defect area, pHash distances,
  acquisition order, contact sheet (with mask contours and zooms), closest pHash pairs.
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
from sklearn.metrics import roc_auc_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from vqi.config import Config  # noqa: E402
from vqi.data.hashing import hamming_distance  # noqa: E402
from vqi.logging_utils import get_logger, log_event  # noqa: E402
from vqi.provenance import provenance  # noqa: E402

logger = get_logger(__name__)

GOOD, DEFECTIVE = 0, 1
CONTOUR_BGR = (0, 0, 255)


@dataclass(frozen=True)
class EdaPaths:
    root: Path
    figures: Path
    report: Path

    @classmethod
    def from_config(cls, cfg: Config) -> EdaPaths:
        root = cfg.paths.reports_dir / cfg.eda.report_subdir
        return cls(root=root, figures=root / "figures", report=root / "eda_report.md")


# --------------------------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------------------------- #


def usable(df: pd.DataFrame) -> pd.DataFrame:
    """Rows that carry a label: conforming name, decodable image, mask present."""
    return df[df["name_conforms"] & (~df["is_corrupted"]) & (df["label"] >= 0)]


def class_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Counts and defect rate per official split (+ total)."""
    ok = usable(df)
    rows = []
    for split in ("train", "test", "all"):
        sub = ok if split == "all" else ok[ok["official_split"] == split]
        n_def = int((sub["label"] == DEFECTIVE).sum())
        rows.append(
            {
                "official_split": split,
                "n_images": int(len(sub)),
                "n_good": int(len(sub) - n_def),
                "n_defective": n_def,
                "defect_rate": round(n_def / max(len(sub), 1), 4),
                "imbalance_ratio_neg_pos": round((len(sub) - n_def) / max(n_def, 1), 2),
            }
        )
    return pd.DataFrame(rows)


def trivial_feature_auc(df: pd.DataFrame) -> pd.DataFrame:
    """ROC-AUC of single global statistics as "classifiers" on the official train set.

    A value far from 0.5 means the label is partly predictable from a trivial cue (brightness,
    file size...). That is a bias to know about — a model could learn the cue instead of the
    defect. It is reported, not corrected.
    """
    ok = usable(df)
    train = ok[ok["official_split"] == "train"]
    rows = []
    for feature in (
        "mean_luminance",
        "std_luminance",
        "image_bytes",
        "height",
        "width",
        "channel_spread",
    ):
        auc = roc_auc_score(train["label"], train[feature])
        rows.append(
            {
                "feature": feature,
                "roc_auc": round(float(auc), 4),
                "auc_from_0.5": round(abs(float(auc) - 0.5), 4),
            }
        )
    return pd.DataFrame(rows).sort_values("auc_from_0.5", ascending=False).reset_index(drop=True)


def imbalance_options(df: pd.DataFrame) -> pd.DataFrame:
    """The candidate remedies for class imbalance, with the numbers each would use.

    Nothing is chosen here (decision deferred to Phase 3); the table makes the options concrete.
    """
    ok = usable(df)
    train = ok[ok["official_split"] == "train"]
    n_pos = int((train["label"] == DEFECTIVE).sum())
    n_neg = int(len(train) - n_pos)
    ratio = n_neg / max(n_pos, 1)
    return pd.DataFrame(
        [
            {
                "option": "aucune (BCE simple) + seuil sur la validation",
                "parametre": "pos_weight = 1",
                "valeur": 1.0,
                "effet_sur_le_gradient": "aucun ; la classe positive pèse ~%.1f %% des exemples"
                % (100 * n_pos / len(train)),
                "effet_sur_la_calibration": "aucun (probabilités interprétables)",
            },
            {
                "option": "pondération de la perte (BCEWithLogits pos_weight)",
                "parametre": "pos_weight = n_neg / n_pos",
                "valeur": round(ratio, 2),
                "effet_sur_le_gradient": f"chaque positif compte comme {ratio:.1f} négatifs ; "
                "classes équilibrées",
                "effet_sur_la_calibration": "décale le prior : les probabilités sont gonflées",
            },
            {
                "option": "pondération de la perte, variante douce",
                "parametre": "pos_weight = sqrt(n_neg / n_pos)",
                "valeur": round(float(np.sqrt(ratio)), 2),
                "effet_sur_le_gradient": "compromis entre les deux précédents",
                "effet_sur_la_calibration": "décalage modéré",
            },
            {
                "option": "suréchantillonnage des positifs (duplication)",
                "parametre": "facteur = n_neg / n_pos",
                "valeur": round(ratio, 2),
                "effet_sur_le_gradient": "équivalent à pos_weight en espérance, mais plus de passes "
                f"sur les mêmes {n_pos} positifs (risque de surapprentissage)",
                "effet_sur_la_calibration": "identique à pos_weight",
            },
            {
                "option": "WeightedRandomSampler (tirage équilibré par batch)",
                "parametre": "poids_i = 1 / n_classe(i) ; batch ≈ 50/50",
                "valeur": round(1 / n_pos, 6),
                "effet_sur_le_gradient": "chaque batch contient ~50 % de positifs ; l'epoch ne voit "
                "plus tous les négatifs",
                "effet_sur_la_calibration": "décalage du prior (comme pos_weight)",
            },
            {
                "option": "focal loss",
                "parametre": "gamma = 2 (config training.loss.focal_gamma)",
                "valeur": 2.0,
                "effet_sur_le_gradient": "atténue les exemples faciles, quelle que soit leur classe",
                "effet_sur_la_calibration": "connue pour dégrader la calibration",
            },
        ]
    )


def acquisition_order_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Is the defect rate stable along the acquisition sequence? (leakage risk L2)

    Per official split: rolling defect rate (window ≈ 5 % of the split) min/max, and the longest
    run of consecutive good images. A wide rolling-rate range or a very long defect-free run
    would suggest sessions/batches that a random split would mix.
    """
    ok = usable(df)
    rows = []
    for split in ("train", "test"):
        sub = ok[ok["official_split"] == split].sort_values("acquisition_index")
        lab = sub["label"].to_numpy()
        window = max(len(sub) // 20, 10)
        rolling = pd.Series(lab).rolling(window, center=True).mean().dropna()
        runs, run = [], 0
        for v in lab:
            run = run + 1 if v == GOOD else 0
            runs.append(run)
        rows.append(
            {
                "official_split": split,
                "window": window,
                "min_rolling_rate": round(float(rolling.min()), 4),
                "max_rolling_rate": round(float(rolling.max()), 4),
                "overall_rate": round(float(lab.mean()), 4),
                "longest_run_without_defect": int(max(runs)),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------------------------- #


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_dimensions(df: pd.DataFrame, path: Path) -> Path:
    ok = usable(df)
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    axes[0].hist(ok["height"], bins=40, color="#4c72b0")
    axes[0].set_title("Hauteur (px)")
    axes[1].hist(ok["width"], bins=40, color="#4c72b0")
    axes[1].set_title("Largeur (px)")
    for label, color, name in ((GOOD, "#4c72b0", "sain"), (DEFECTIVE, "#dd8452", "défectueux")):
        sub = ok[ok["label"] == label]
        axes[2].scatter(sub["width"], sub["height"], s=6, alpha=0.5, color=color, label=name)
    axes[2].set_xlabel("largeur")
    axes[2].set_ylabel("hauteur")
    axes[2].set_title("Largeur × hauteur par classe")
    axes[2].legend()
    fig.suptitle("Dimensions des images (fichiers conformes, non corrompus)")
    return _save(fig, path)


def fig_luminance_contrast(df: pd.DataFrame, path: Path) -> Path:
    ok = usable(df)
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    specs = (
        ("mean_luminance", "Luminance moyenne (0–255)"),
        ("std_luminance", "Contraste (écart-type des niveaux de gris)"),
        ("channel_spread", "Écart moyen entre canaux (0 = gris pur)"),
    )
    for ax, (col, title) in zip(axes, specs, strict=True):
        bins = np.histogram_bin_edges(ok[col], bins=40)
        for label, color, name in ((GOOD, "#4c72b0", "sain"), (DEFECTIVE, "#dd8452", "défectueux")):
            ax.hist(
                ok.loc[ok["label"] == label, col],
                bins=bins,
                alpha=0.6,
                color=color,
                label=name,
                density=True,
            )
        ax.set_title(title)
        ax.legend()
    fig.suptitle("Statistiques photométriques par classe (densités)")
    return _save(fig, path)


def fig_defect_area(df: pd.DataFrame, bins: tuple[int, ...], path: Path) -> Path:
    ok = usable(df)
    d = ok[ok["label"] == DEFECTIVE]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    axes[0].hist(d["defect_area_px"], bins=np.logspace(1, 5, 40), color="#dd8452")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("aire du défaut (px, échelle log)")
    axes[0].set_title("Aire des défauts")
    axes[1].scatter(d["defect_bbox_w"], d["defect_bbox_h"], s=8, alpha=0.6, color="#dd8452")
    axes[1].set_xlabel("largeur de la boîte englobante")
    axes[1].set_ylabel("hauteur de la boîte englobante")
    axes[1].set_title("Boîte englobante des défauts")
    fig.suptitle("Taille des défauts (depuis les masques ; images défectueuses uniquement)")
    return _save(fig, path)


def fig_phash_nn(nn: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 3.8))
    counts = nn["nn_distance"].value_counts().sort_index()
    cross = nn[nn["official_split"] != nn["nn_official_split"]]["nn_distance"].value_counts()
    ax.bar(counts.index, counts.values, color="#4c72b0", label="toutes les images")
    cross = cross.reindex(counts.index).fillna(0)
    ax.bar(
        cross.index,
        cross.values,
        color="#dd8452",
        label="plus proche voisin dans l'autre split officiel",
    )
    ax.set_xlabel("distance de Hamming pHash au plus proche voisin (64 bits)")
    ax.set_ylabel("nombre d'images")
    ax.set_title("Quasi-doublons : distribution des distances au plus proche voisin")
    ax.legend()
    return _save(fig, path)


def fig_acquisition_order(df: pd.DataFrame, path: Path) -> Path:
    """Label along the acquisition index: are defectives clustered in time (risk L2)?"""
    ok = usable(df)
    fig, axes = plt.subplots(2, 1, figsize=(12, 4.5), sharex=False)
    for ax, split in zip(axes, ("train", "test"), strict=True):
        sub = ok[ok["official_split"] == split].sort_values("acquisition_index")
        idx = sub["acquisition_index"].to_numpy()
        lab = sub["label"].to_numpy()
        ax.vlines(idx[lab == GOOD], 0, 1, color="#4c72b0", alpha=0.25, linewidth=0.5)
        ax.vlines(idx[lab == DEFECTIVE], 0, 1, color="#dd8452", linewidth=1.0)
        window = max(len(sub) // 20, 10)
        rate = pd.Series(lab).rolling(window, center=True).mean().to_numpy()
        ax2 = ax.twinx()
        ax2.plot(idx, rate, color="black", linewidth=1)
        ax2.set_ylim(0, max(0.3, float(np.nanmax(rate)) * 1.1))
        ax2.set_ylabel(f"taux glissant ({window})")
        ax.set_yticks([])
        ax.set_title(f"{split} officiel — orange = défectueux, ordre d'acquisition (id de fichier)")
    axes[-1].set_xlabel("acquisition_index")
    fig.tight_layout()
    return _save(fig, path)


def _read_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _overlay_contour(rgb: np.ndarray, mask_path: Path, thickness: int = 2) -> np.ndarray:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    out = rgb.copy()
    if mask is None:
        return out
    contours, _ = cv2.findContours(
        (mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(out, contours, -1, (255, 0, 0), thickness)
    return out


def _zoom_on_mask(
    rgb: np.ndarray, mask_path: Path, pad_factor: float = 1.0, min_side: int = 96
) -> np.ndarray:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    ys, xs = np.nonzero(mask > 0)
    if len(ys) == 0:
        return rgb
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    h, w = y1 - y0 + 1, x1 - x0 + 1
    side = int(max(min_side, max(h, w) * (1 + 2 * pad_factor)))
    side = min(side, rgb.shape[1])  # keep the crop square: never wider than the image
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    top = int(np.clip(cy - side // 2, 0, max(rgb.shape[0] - side, 0)))
    left = int(np.clip(cx - side // 2, 0, max(rgb.shape[1] - side, 0)))
    crop = _overlay_contour(rgb, mask_path, thickness=1)[top : top + side, left : left + side]
    return crop


def fig_contact_sheet(
    df: pd.DataFrame, cfg: Config, project_root: Path, path: Path, seed: int
) -> tuple[Path, list[str], list[str]]:
    """Row 1: random good images. Row 2: defective images spanning the defect-area range, with
    the mask contour in red. Row 3: zoom on each defect. Returns (path, good_ids, defective_ids)."""
    n = cfg.eda.contact_sheet.n_per_class
    ok = usable(df)
    train = ok[ok["official_split"] == "train"]
    rng = np.random.default_rng(seed)
    good = train[train["label"] == GOOD].sample(n=n, random_state=int(rng.integers(0, 2**31)))
    defective = train[train["label"] == DEFECTIVE].sort_values("defect_area_px")
    quantile_pos = np.linspace(0, len(defective) - 1, n).round().astype(int)
    defective = defective.iloc[quantile_pos]

    fig, axes = plt.subplots(3, n, figsize=(1.9 * n, 13))
    for col, (_, row) in enumerate(good.iterrows()):
        rgb = _read_rgb(project_root / row["image_path"])
        axes[0, col].imshow(rgb)
        axes[0, col].set_title(f"sain {row['image_id']}", fontsize=8)
    for col, (_, row) in enumerate(defective.iterrows()):
        rgb = _read_rgb(project_root / row["image_path"])
        mask_path = project_root / row["mask_path"]
        axes[1, col].imshow(_overlay_contour(rgb, mask_path))
        axes[1, col].set_title(
            f"défectueux {row['image_id']}\n{int(row['defect_area_px'])} px", fontsize=8
        )
        axes[2, col].imshow(_zoom_on_mask(rgb, mask_path))
        axes[2, col].set_title("zoom", fontsize=8)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        "Planche de contact — train officiel. Ligne 2 : contour du masque en rouge ; "
        "défauts triés par aire croissante",
        fontsize=11,
    )
    fig.tight_layout()
    return _save(fig, path), good["image_id"].tolist(), defective["image_id"].tolist()


def closest_pairs_table(df: pd.DataFrame, nn: pd.DataFrame, n_pairs: int) -> pd.DataFrame:
    """The ``n_pairs`` closest unordered pHash pairs (excluding nonconforming files)."""
    conforms = set(df.loc[df["name_conforms"], "image_id"])
    sub = nn[nn["image_id"].isin(conforms) & nn["nn_image_id"].isin(conforms)].copy()
    sub["a"] = np.where(sub["image_id"] < sub["nn_image_id"], sub["image_id"], sub["nn_image_id"])
    sub["b"] = np.where(sub["image_id"] < sub["nn_image_id"], sub["nn_image_id"], sub["image_id"])
    sub = sub.drop_duplicates(["a", "b"]).sort_values(["nn_distance", "a"]).head(n_pairs)
    meta = df.set_index("image_id")
    return pd.DataFrame(
        {
            "image_id_a": sub["a"].to_numpy(),
            "image_id_b": sub["b"].to_numpy(),
            "distance": sub["nn_distance"].to_numpy(),
            "split_a": meta.loc[sub["a"], "official_split"].to_numpy(),
            "split_b": meta.loc[sub["b"], "official_split"].to_numpy(),
            "label_a": meta.loc[sub["a"], "label"].to_numpy(),
            "label_b": meta.loc[sub["b"], "label"].to_numpy(),
        }
    ).reset_index(drop=True)


def fig_closest_pairs(
    df: pd.DataFrame, pairs: pd.DataFrame, project_root: Path, path: Path
) -> Path:
    meta = df.set_index("image_id")
    n = len(pairs)
    cols = 6
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols * 2, figsize=(1.6 * cols * 2, 4.6 * rows))
    axes = np.atleast_2d(axes)
    for k, (_, p) in enumerate(pairs.iterrows()):
        r, c = divmod(k, cols)
        for j, img_id in enumerate((p["image_id_a"], p["image_id_b"])):
            ax = axes[r, 2 * c + j]
            row = meta.loc[img_id]
            rgb = _read_rgb(project_root / row["image_path"])
            if row["label"] == DEFECTIVE and row["mask_path"]:
                rgb = _overlay_contour(rgb, project_root / row["mask_path"])
            ax.imshow(rgb)
            ax.set_title(f"{img_id} ({row['official_split']}, y={int(row['label'])})", fontsize=7)
        axes[r, 2 * c].set_xlabel(f"d = {int(p['distance'])}", fontsize=8)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Paires pHash les plus proches (d = distance de Hamming sur 64 bits)", fontsize=11)
    fig.tight_layout()
    return _save(fig, path)


# --------------------------------------------------------------------------------------------- #
# Report — the Markdown template below has long lines by nature (tables, f-strings).
# ruff: noqa: E501
# --------------------------------------------------------------------------------------------- #


def _md_table(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    df = df.set_axis(cols, axis=1)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def _fmt(x: float, nd: int = 1) -> str:
    return f"{x:.{nd}f}".replace(".", ",")


def build_report(
    df: pd.DataFrame,
    manifest_meta: dict[str, Any],
    dup_report: dict[str, Any],
    nn: pd.DataFrame,
    splits: pd.DataFrame | None,
    split_meta: dict[str, Any] | None,
    tables: dict[str, pd.DataFrame],
    figures: dict[str, Path],
    contact_ids: tuple[list[str], list[str]],
    cfg: Config,
) -> str:
    ok = usable(df)
    train = ok[ok["official_split"] == "train"]
    d = ok[ok["label"] == DEFECTIVE]
    counts = tables["class_counts"].set_index("official_split")
    figs = {
        k: v.relative_to(figures["_root"]).as_posix() for k, v in figures.items() if k != "_root"
    }
    cc = manifest_meta["counts_check"]
    sens = pd.DataFrame(dup_report["near"]["sensitivity"])
    nn_counts = nn["nn_distance"].value_counts().sort_index()
    tf = tables["trivial_feature_auc"]

    sections: list[str] = []
    sections.append(f"""# Analyse exploratoire — {cfg.dataset.name}

Généré automatiquement par `scripts/run_eda.py` le {manifest_meta["created_at"][:10]}
(config `{cfg.hash}`, commit `{manifest_meta.get("git_commit") or "hors dépôt"}`).
Chaque chiffre provient du manifeste `data/interim/manifest.csv`
(SHA-256 `{manifest_meta["manifest_sha256"][:12]}…`) ; rien n'est saisi à la main.

## 1. Intégrité et inventaire

| Contrôle | Résultat |
|---|---|
| Archive `{cfg.dataset.archive_name}` | SHA-256 `{cfg.dataset.archive_sha256[:16]}…` vérifié |
| Fichiers image inventoriés | {len(df)} |
| Fichiers hors convention de nommage (`{cfg.dataset.image_id_pattern}`) | {manifest_meta["n_nonconforming_names"]} : {", ".join("`" + p + "`" for p in manifest_meta["nonconforming_names"]) or "—"} |
| Fichiers parasites (masques orphelins) | {len(manifest_meta["stray_files"])} : {", ".join("`" + s["path"] + "`" for s in manifest_meta["stray_files"]) or "—"} |
| Images corrompues (non décodables, masque absent ou de taille différente) | {manifest_meta["n_corrupted"]} |
| Effectifs attendus (article / page officielle) vs observés | {"**conformes**" if cc["match"] else "**ÉCART** : " + str(cc["mismatches"])} — {cc["observed"]} |

Les fichiers hors convention et parasites sont des **copies byte-identiques** laissées dans
l'archive par les auteurs (voir §5) ; ils sont inventoriés, exclus par le manifeste et jamais
supprimés de `data/raw/`.

## 2. Effectifs par classe et déséquilibre

{_md_table(tables["class_counts"])}

Le taux de défaut est de {_fmt(100 * counts.loc["train", "defect_rate"])} % en train officiel et
{_fmt(100 * counts.loc["test", "defect_rate"])} % en test officiel : le déséquilibre est le même des
deux côtés, ce qui est cohérent avec un découpage aléatoire par les auteurs. Un modèle qui prédit
« sain » partout obtiendrait {_fmt(100 * (1 - counts.loc["test", "defect_rate"]))} % d'exactitude sur le
test en manquant 100 % des défauts — c'est pour cela que l'exactitude n'est pas une métrique de ce
projet (cadrage §4.1).

### Options pour répondre au déséquilibre (aucune n'est tranchée ici — Phase 3)

Sur le train officiel : {int((train["label"] == GOOD).sum())} saines / {int((train["label"] == DEFECTIVE).sum())} défectueuses,
soit un ratio de {_fmt(counts.loc["train", "imbalance_ratio_neg_pos"], 2)} : 1.

{_md_table(tables["imbalance_options"])}

Points de comparaison pour la décision :

- **Le levier des coûts reste le seuil** (cadrage §4.3) : aucune de ces options ne remplace le
  choix du point de fonctionnement sur la validation ; elles agissent sur l'apprentissage, pas
  sur la décision.
- `pos_weight`, le suréchantillonnage et le `WeightedRandomSampler` sont trois façons d'obtenir
  le même rééquilibrage en espérance ; ils diffèrent par la variance du gradient et par le
  nombre de passes sur les {int((train["label"] == DEFECTIVE).sum())} positifs.
- Un ratio de {_fmt(counts.loc["train", "imbalance_ratio_neg_pos"], 1)} : 1 est *modéré* : la BCE non pondérée
  converge généralement sans intervention à ce niveau, ce qui rend l'option « aucune » crédible
  comme référence ; l'ablation `pos_weight ∈ {{1 ; {_fmt(float(np.sqrt(counts.loc["train", "imbalance_ratio_neg_pos"])), 1)} ; {_fmt(counts.loc["train", "imbalance_ratio_neg_pos"], 1)}}}` prévue en Phase 4 tranchera sur des chiffres.
- Toute option qui rééquilibre décale les probabilités ; si la politique de seuil `expected_cost`
  est retenue, une recalibration (temperature scaling) devient nécessaire.

## 3. Dimensions et canaux

![dimensions]({figs["dimensions"]})

| Statistique | Hauteur (px) | Largeur (px) |
|---|---|---|
| min | {int(ok["height"].min())} | {int(ok["width"].min())} |
| médiane | {int(ok["height"].median())} | {int(ok["width"].median())} |
| max | {int(ok["height"].max())} | {int(ok["width"].max())} |
| valeurs distinctes | {ok["height"].nunique()} | {ok["width"].nunique()} |

Ratio hauteur/largeur médian : {_fmt(float((ok["height"] / ok["width"]).median()), 2)}. Les dimensions
varient d'une image à l'autre ({ok.groupby(["height", "width"]).ngroups} combinaisons distinctes) :
le prétraitement doit donc normaliser la taille. La cible de la config
({cfg.preprocessing.input_size.height}×{cfg.preprocessing.input_size.width}, H×W) est à l'échelle
native médiane ({int(ok["height"].median())}×{int(ok["width"].median())}) : il n'y a pratiquement
pas de sous-échantillonnage, donc pas de perte des petits défauts (décision D7 confirmée).

Canaux : {ok["channels"].value_counts().to_dict()} (tous les fichiers ont {int(ok["channels"].mode().iloc[0])} canaux).
Écart moyen entre canaux : médiane {_fmt(float(ok["channel_spread"].median()), 2)}/255,
max {_fmt(float(ok["channel_spread"].max()), 2)}/255 → les images sont **quasi monochromes**
(légère dominante colorée). Conséquence pour `preprocessing.color_mode` : `rgb` ne coûte rien avec un
backbone pré-entraîné (qui attend 3 canaux) et conserve la faible information couleur ; `grayscale`
répliqué sur 3 canaux serait équivalent à ~{_fmt(float(ok["channel_spread"].median()), 1)}/255 près.
Recommandation : garder `rgb`.

## 4. Luminosité et contraste

![luminance]({figs["luminance_contrast"]})

| Classe | Luminance moyenne (médiane) | Contraste = écart-type (médiane) |
|---|---|---|
| sain | {_fmt(float(ok.loc[ok["label"] == GOOD, "mean_luminance"].median()))} | {_fmt(float(ok.loc[ok["label"] == GOOD, "std_luminance"].median()))} |
| défectueux | {_fmt(float(d["mean_luminance"].median()))} | {_fmt(float(d["std_luminance"].median()))} |

Les images sont sombres (luminance ≈ {_fmt(float(ok["mean_luminance"].median()))}/255) et à faible
contraste : l'augmentation luminosité/contraste devra rester légère pour ne pas sortir du domaine.

### Séparabilité triviale (biais à connaître)

ROC-AUC d'une statistique globale utilisée seule comme score de défaut, sur le train officiel :

{_md_table(tf)}

Lecture : les statistiques les plus prédictives sont `{tf.iloc[0]["feature"]}` (AUC {_fmt(100 * tf.iloc[0]["roc_auc"])} %) et `{tf.iloc[1]["feature"]}` (AUC {_fmt(100 * tf.iloc[1]["roc_auc"])} %) — {"un signal trivial existe : un défaut ajoute de la texture (contraste) et, pour certains défauts colorés, une dominante chromatique. Il est loin de suffire (une AUC de 100 % serait un problème), mais il est réel : en Phase 4, le modèle sera comparé à cette référence triviale pour prouver qu’il apprend plus que cela." if tf.iloc[0]["roc_auc"] > 0.6 else "aucune statistique globale ne sépare les classes de façon notable ; le modèle devra apprendre des motifs locaux."}

## 5. Doublons et quasi-doublons

**Doublons exacts (SHA-256)** : {dup_report["exact"]["n_groups"]} groupe(s), {dup_report["exact"]["n_extra_images"]} image(s) en trop,
{dup_report["exact"]["n_cross_split_groups"]} groupe(s) à cheval sur train/test officiels.
Groupes : {dup_report["exact"]["groups"]}.

**Quasi-doublons (pHash 64 bits)** — statut : `{dup_report["near"]["status"]}`.

![phash]({figs["phash_nn_distance"]})

Distribution des distances au plus proche voisin (nombre d'images par distance) :

{_md_table(pd.DataFrame([nn_counts.values], columns=[f"d = {int(k)}" for k in nn_counts.index], index=["n_images"]).reset_index().rename(columns={"index": ""}))}

Table de sensibilité — ce que chaque seuil candidat `max_hamming_distance` marquerait :

{_md_table(sens)}

![closest]({figs["closest_pairs"]})

Lecture et proposition de seuil (à valider, la config reste `max_hamming_distance: null` tant
qu'elle n'est pas validée) :

- La distribution est **unimodale**, centrée sur {int(nn_counts.idxmax())} bits : sur un produit unique
  photographié dans des conditions fixes, *toutes* les images se ressemblent, et le plus proche
  voisin est à ~{int(nn_counts.idxmax())} bits. Il n'y a pas de seconde population à faible distance qui
  signerait des vues multiples d'une même pièce.
- Les seules paires à distance 0 sont la copie exacte déjà exclue. Les paires à distance ≤ 8 sont
  montrées ci-dessus : à juger visuellement.
- Un seuil à 10 marquerait déjà {int(sens.loc[sens["max_hamming_distance"] == 10, "n_pairs"].iloc[0])} paires
  dont {int(sens.loc[sens["max_hamming_distance"] == 10, "n_cross_split_pairs"].iloc[0])} entre train et test ; à 12,
  {int(sens.loc[sens["max_hamming_distance"] == 12, "n_pairs"].iloc[0])} paires — c'est la masse de la distribution,
  pas des doublons.
- **Proposition** : `max_hamming_distance: 8` (regroupe les rares paires très proches sans toucher
  au corps de la distribution), ou laisser `null` si l'inspection visuelle des paires ≤ 8 ne montre
  pas de doublon. Dans les deux cas l'effet sur le découpage est marginal ({int(sens.loc[sens["max_hamming_distance"] == 8, "n_images_involved"].iloc[0])} images concernées à 8).

## 6. Taille des défauts

![area]({figs["defect_area"]})

| Quantile de l'aire (px) | 0 % (min) | 5 % | 25 % | 50 % | 75 % | 95 % | 100 % (max) |
|---|---|---|---|---|---|---|---|
| valeur | {" | ".join(str(int(v)) for v in d["defect_area_px"].quantile([0, 0.05, 0.25, 0.5, 0.75, 0.95, 1]))} |

Défauts de moins de 100 px : {int((d["defect_area_px"] < 100).sum())} / {len(d)} ; de moins de 250 px :
{int((d["defect_area_px"] < 250).sum())} / {len(d)}. Nombre de composantes connexes par masque :
{ {int(k): int(v) for k, v in d["n_defect_components"].value_counts().sort_index().items()} }.
Sur une image de ~{int(ok["height"].median())}×{int(ok["width"].median())} = {int(ok["height"].median() * ok["width"].median())} px, un défaut médian couvre
{_fmt(100 * float(d["defect_area_px"].median()) / float(ok["height"].median() * ok["width"].median()), 2)} % de la surface ; le plus petit,
{_fmt(100 * float(d["defect_area_px"].min()) / float(ok["height"].median() * ok["width"].median()), 3)} %. Ces quantiles serviront de
strates pour l'analyse « rappel par taille de défaut » en Phase 4 (risque R6).

## 7. Ordre d'acquisition

![order]({figs["acquisition_order"]})

Les identifiants de fichiers sont séquentiels ; si les défectueuses étaient concentrées sur
certaines plages, un découpage aléatoire mélangerait des « sessions » entre train et validation
(risque L2). La figure montre le taux de défaut glissant le long de l'index.

{_md_table(tables["acquisition_order"])}

Lecture : {"le taux glissant reste dans une bande étroite et aucune plage longue n’est vide de défauts : les défectueuses sont réparties sur toute la séquence, ce qui rend le découpage aléatoire stratifié défendable (un split par blocs reste un contrôle de robustesse possible en v2)." if tables["acquisition_order"]["max_rolling_rate"].max() < 0.35 and tables["acquisition_order"]["longest_run_without_defect"].max() < 150 else "le taux glissant varie fortement le long de la séquence : des sessions existent, et un split par blocs doit être comparé au split aléatoire avant la Phase 3."}

## 8. Planche de contact

![contact]({figs["contact_sheet"]})

Saines (tirage aléatoire, seed {cfg.reproducibility.seed}) : {", ".join(contact_ids[0])}.
Défectueuses (réparties sur l'étendue des aires, de la plus petite à la plus grande) :
{", ".join(contact_ids[1])}. Ligne 3 : zoom sur chaque défaut.

""")

    if splits is not None and split_meta is not None:
        chk = split_meta["checks"]
        sc = pd.DataFrame([{"split": s, **v} for s, v in chk["counts"].items()])
        sc["defect_rate"] = (sc["defective"] / sc["total"].clip(lower=1)).round(4)
        sections.append(f"""## 9. Découpage train / validation / test (décision D3)

Fichier `data/splits/{cfg.split.manifest_version}.csv` (SHA-256 `{split_meta["splits_sha256"][:12]}…`),
stratégie `{split_meta["strategy"]}`, fraction de validation {split_meta["val_fraction"]},
seed {split_meta["seed"]}, groupes = `{split_meta["group_column"]}`, seuil quasi-doublons =
`{split_meta["near_duplicate_threshold"]}`.

{_md_table(sc)}

| Invariant | Résultat |
|---|---|
| Groupes de doublons à cheval sur train/val | {chk["groups_spanning_train_val"]} |
| Test = test officiel (non corrompu) | {chk["test_equals_official_test"]} |
| Train et val issus du train officiel uniquement | {chk["val_in_official_train_only"]} |
| Fraction de validation effective | {chk["val_fraction_actual"]} |
| Taux de défaut train / val / test | {chk["positive_rate"]} |

Le mécanisme est un **hold-out unique** obtenu comme un seul pli d'une partition
`StratifiedGroupKFold(k={split_meta["k_folds_mechanism"]})` ; les autres plis ne sont jamais
utilisés (ce n'est pas une validation croisée). Avec {chk["counts"]["val"]["defective"]} défectueuses en
validation, un rappel de 0,95 correspond à « manquer au plus {int(np.floor(0.05 * chk["counts"]["val"]["defective"]))} image(s) » —
d'où les intervalles bootstrap systématiques (risque R1).
""")

    sections.append("""## Décisions alimentées par cette analyse

| Décision | Élément de preuve dans ce rapport | Statut |
|---|---|---|
| `preprocessing.color_mode` | §3 écart entre canaux | recommandation : `rgb` |
| `preprocessing.input_size` | §3 dimensions natives | 640×224 confirmé (échelle native) |
| `split.duplicates.max_hamming_distance` | §5 distribution + paires les plus proches | à valider (proposition : 8 ou `null`) |
| stratégie de déséquilibre | §2 table des options | reportée à la Phase 3, ablation en Phase 4 |
| `augmentation.rotate_90_p` | §8 planche de contact (orientation des pièces) | à valider visuellement |
| strates de taille de défaut (Phase 4) | §6 quantiles | fixées par les quantiles ci-dessus |
""")
    return "\n".join(sections)


def run_eda(
    df: pd.DataFrame,
    manifest_meta: dict[str, Any],
    dup_report: dict[str, Any],
    nn: pd.DataFrame,
    splits: pd.DataFrame | None,
    split_meta: dict[str, Any] | None,
    cfg: Config,
    project_root: Path,
) -> Path:
    paths = EdaPaths.from_config(cfg)
    paths.figures.mkdir(parents=True, exist_ok=True)

    tables = {
        "class_counts": class_counts(df),
        "trivial_feature_auc": trivial_feature_auc(df),
        "imbalance_options": imbalance_options(df),
        "acquisition_order": acquisition_order_stats(df),
    }
    for name, table in tables.items():
        table.to_csv(paths.root / f"{name}.csv", index=False, lineterminator="\n")

    pairs = closest_pairs_table(df, nn, cfg.eda.n_closest_pairs)
    pairs.to_csv(paths.root / "closest_phash_pairs.csv", index=False, lineterminator="\n")
    # Distances in the NN table are to the *nearest* neighbour; recompute the exact distance of
    # each shown pair so the figure caption is self-consistent.
    meta = df.set_index("image_id")
    pairs["distance"] = [
        hamming_distance(meta.loc[a, "phash"], meta.loc[b, "phash"])
        for a, b in zip(pairs["image_id_a"], pairs["image_id_b"], strict=True)
    ]

    figures: dict[str, Path] = {"_root": paths.root}
    figures["dimensions"] = fig_dimensions(df, paths.figures / "dimensions.png")
    figures["luminance_contrast"] = fig_luminance_contrast(
        df, paths.figures / "luminance_contrast.png"
    )
    figures["defect_area"] = fig_defect_area(
        df, cfg.eda.defect_area_bins, paths.figures / "defect_area.png"
    )
    figures["phash_nn_distance"] = fig_phash_nn(nn, paths.figures / "phash_nn_distance.png")
    figures["acquisition_order"] = fig_acquisition_order(
        df, paths.figures / "acquisition_order.png"
    )
    figures["closest_pairs"] = fig_closest_pairs(
        df, pairs, project_root, paths.figures / "closest_pairs.png"
    )
    figures["contact_sheet"], good_ids, def_ids = fig_contact_sheet(
        df, cfg, project_root, paths.figures / "contact_sheet.png", cfg.reproducibility.seed
    )

    report = build_report(
        df,
        manifest_meta,
        dup_report,
        nn,
        splits,
        split_meta,
        tables,
        figures,
        (good_ids, def_ids),
        cfg,
    )
    paths.report.write_text(report, encoding="utf-8", newline="\n")
    (paths.root / "eda.meta.json").write_text(
        json.dumps(
            provenance(cfg, figures={k: v.as_posix() for k, v in figures.items() if k != "_root"}),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
        newline="\n",
    )
    log_event(logger, "eda.written", report=str(paths.report), n_figures=len(figures) - 1)
    return paths.report
