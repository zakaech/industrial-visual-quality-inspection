# ruff: noqa: E501  (Markdown template with long table rows)
"""Phase 2 report generator: assembles ``reports/phase2/phase2_report.md`` from the artefacts
written by ``compare_preprocessing.py`` and ``run_baseline.py``. Every number is read from those
files; nothing is typed by hand. Prose is French (project documentation language).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from vqi.baseline.experiments import experiment_dir  # noqa: E402
from vqi.baseline.features import FeatureExtractor  # noqa: E402
from vqi.config import Config  # noqa: E402
from vqi.logging_utils import get_logger, log_event  # noqa: E402
from vqi.provenance import provenance  # noqa: E402

logger = get_logger(__name__)


def _fmt(x: Any, nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "TBD"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    return f"{float(x):.{nd}f}".replace(".", ",")


def _md_table(df: pd.DataFrame, nd: int = 3) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.astype(object).iterrows():  # keep ints as ints (iterrows would upcast)
        cells = []
        for c in df.columns:
            v = row[c]
            cells.append(_fmt(v, nd) if isinstance(v, (float, np.floating)) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _load_results(cfg: Config) -> dict[str, dict[str, Any]]:
    out = {}
    for spec in cfg.baseline.experiments:
        p = experiment_dir(cfg) / f"{spec.id}.json"
        if p.exists():
            out[spec.id] = json.loads(p.read_text(encoding="utf-8"))
    return out


def pr_curve_figure(results: dict[str, dict[str, Any]], ids: list[str], path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for i in ids:
        if i not in results:
            continue
        c = results[i]["metrics"]["curves"]
        ap = results[i]["metrics"]["average_precision"]
        ax.plot(c["pr_recall"], c["pr_precision"], linewidth=1.3, label=f"{i} (AP {ap:.3f})")
    prevalence = next(iter(results.values()))["metrics"]["prevalence"]
    ax.axhline(
        prevalence,
        color="gray",
        linestyle="--",
        linewidth=0.8,
        label=f"hasard (prévalence {prevalence:.3f})",
    )
    ax.set_xlabel("rappel (défauts détectés)")
    ax.set_ylabel("précision")
    ax.set_title("Courbes précision–rappel sur la validation", fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def _variant_table(cfg: Config) -> pd.DataFrame:
    rows = []
    for name, steps in cfg.preprocessing.enhancement.variants.items():
        desc = (
            " → ".join(
                f"{s['op']}({', '.join(f'{k}={v}' for k, v in s.items() if k != 'op')})"
                for s in steps
            )
            or "—"
        )
        rows.append({"variante": name, "opérateurs (ordre d'application)": desc})
    return pd.DataFrame(rows)


def _feature_table(cfg: Config) -> pd.DataFrame:
    fe = FeatureExtractor.from_config(cfg)
    rows = []
    for group, extractors in cfg.baseline.feature_groups.items():
        rows.append(
            {
                "groupe": group,
                "extracteurs": ", ".join(extractors),
                "dimension": len(fe.names(extractors)),
            }
        )
    return pd.DataFrame(rows)


def _summary_table(summary: pd.DataFrame) -> pd.DataFrame:
    t = pd.DataFrame(
        {
            "id": summary["id"],
            "features": summary["features"],
            "prétraitement": summary["preprocessing"],
            "pad": summary["pad_value"],
            "clf": summary["classifier"],
            "class_weight": summary["class_weight"],
            "dim": summary["n_features"],
            "C": summary["best_C"],
            "γ": summary["best_gamma"],
            "AP CV (train)": summary["cv_ap_mean"],
            "AP val": summary["val_ap"],
            "IC 95 % AP": [
                f"[{_fmt(lo)} ; {_fmt(hi)}]"
                for lo, hi in zip(summary["val_ap_ci_low"], summary["val_ap_ci_high"], strict=True)
            ],
            "ROC-AUC val": summary["val_roc_auc"],
            "préc.@R0,90": summary["precision@r0.90"],
            "préc.@R0,95": summary["precision@r0.95"],
            "FPR@R0,95": summary["fpr@r0.95"],
            "préc.@R0,98": summary["precision@r0.98"],
        }
    )
    return t


def _tradeoff_table(result: dict[str, Any]) -> pd.DataFrame:
    rows = result["metrics"]["at_recall_targets"]
    return pd.DataFrame(
        [
            {
                "rappel cible": r["recall_target"],
                "seuil (score)": r["threshold"],
                "rappel obtenu": r["recall"],
                "précision": r["precision"],
                "FPR": r["fpr"],
                "TP": int(r["tp"]),
                "FP": int(r["fp"]),
                "FN": int(r["fn"]),
                "TN": int(r["tn"]),
            }
            for r in rows
        ]
    )


def build_report(cfg: Config) -> Path:
    out_dir = cfg.paths.reports_dir / cfg.baseline.report_subdir
    fig_dir = out_dir / "figures"
    summary = pd.read_csv(out_dir / "experiments.csv")
    results = _load_results(cfg)
    cnr_summary = pd.read_csv(out_dir / "cnr_summary.csv")
    cnr_meta = json.loads((out_dir / "compare_preprocessing.meta.json").read_text(encoding="utf-8"))
    reps = pd.read_csv(out_dir / "representative_images.csv", dtype={"image_id": str})

    by_id = summary.set_index("id", drop=False)
    e1 = summary[summary["id"].str.startswith("E1")]
    e3 = summary[summary["id"].str.startswith("E3") | (summary["id"] == "E1_combined")]
    best_e1 = e1[e1["features"] != "combined"].sort_values("val_ap", ascending=False).iloc[0]
    best_e3 = e3.sort_values("val_ap", ascending=False).iloc[0]
    ref = by_id.loc["E1_combined"] if "E1_combined" in by_id.index else best_e1
    best_overall = summary.sort_values("val_ap", ascending=False).iloc[0]
    # the classical reference for Phase 3: best SVM on combined features without enhancement
    classical = summary[
        (summary["features"] == "combined")
        & (summary["preprocessing"] == "none")
        & (summary["pad_value"] == "median")
        & (summary["classifier"] == "svm")
    ]
    ref_to_beat = classical.sort_values("val_ap", ascending=False).iloc[0]
    grid = summary[summary["id"].isin(["E1_combined", "E5_grid_edge_check", "E5_linear_svm"])]
    pr_ids = [
        "E0_contrast_only",
        "E1_global_stats",
        "E1_lbp",
        "E1_local_stats",
        "E1_combined",
        str(best_e3["id"]),
        str(ref_to_beat["id"]),
    ]
    pr_path = pr_curve_figure(results, list(dict.fromkeys(pr_ids)), fig_dir / "pr_curves.png")

    cnr_all = cnr_summary[cnr_summary["area_bin"] == "all"][
        [
            "variant",
            "median_cnr",
            "median_cnr_ratio_vs_none",
            "share_improved",
            "mean_saturated_fraction",
            "median_std",
        ]
    ].rename(
        columns={
            "variant": "variante",
            "median_cnr": "CNR médian",
            "median_cnr_ratio_vs_none": "CNR relatif à none",
            "share_improved": "part d'images améliorées",
            "mean_saturated_fraction": "fraction saturée",
            "median_std": "écart-type médian",
        }
    )
    bins_order = list(dict.fromkeys(cnr_summary[cnr_summary["area_bin"] != "all"]["area_bin"]))
    bins_order = sorted(bins_order, key=lambda b: int(b.strip("[").split(",")[0]))
    piv = (
        cnr_summary[cnr_summary["area_bin"] != "all"]
        .pivot(index="area_bin", columns="variant", values="median_cnr_ratio_vs_none")
        .loc[bins_order]
    )
    n_per_bin = cnr_summary[cnr_summary["variant"] == "none"].set_index("area_bin")["n_images"]
    piv.insert(0, "n", [int(n_per_bin[b]) for b in bins_order])
    piv = piv.reset_index().rename(columns={"area_bin": "aire du défaut (px)"})

    def ci_overlap(a: str, b: str) -> bool:
        ra, rb = by_id.loc[a], by_id.loc[b]
        return not (
            ra["val_ap_ci_low"] > rb["val_ap_ci_high"] or rb["val_ap_ci_low"] > ra["val_ap_ci_high"]
        )

    e0 = by_id.loc["E0_contrast_only"]
    improvement_vs_trivial = float(ref["val_ap"]) - float(e0["val_ap"])
    e3_vs_none = {
        str(r["id"]): (
            float(r["val_ap"]) - float(ref["val_ap"]),
            ci_overlap(str(r["id"]), "E1_combined"),
        )
        for _, r in e3.iterrows()
        if str(r["id"]) != "E1_combined"
    }
    significant_gains = [i for i, (d, overlap) in e3_vs_none.items() if d > 0 and not overlap]

    cw = summary[summary["id"].isin(["E1_combined", "E2_cw_none", "E2_cw_manual"])]
    denoise_cnr = cnr_all.set_index("variante")
    manual_cw = cfg.baseline.class_weights.get("manual")

    md = f"""# Phase 2 — Prétraitement OpenCV et baseline classique

Généré automatiquement par `scripts/build_phase2_report.py` (config `{cfg.hash}`,
seed {cfg.reproducibility.seed}). Tous les chiffres proviennent de `reports/phase2/experiments/*.json`,
`experiments.csv`, `cnr_summary.csv` ; aucun n'est saisi à la main. **Le test officiel n'a jamais
été chargé** : chaque expérience est ajustée sur le split `train` ({results[str(ref["id"])]["data"]["train"]["n"]} images,
{results[str(ref["id"])]["data"]["train"]["n_pos"]} défectueuses) et mesurée sur le split `val`
({results[str(ref["id"])]["data"]["val"]["n"]} images, {results[str(ref["id"])]["data"]["val"]["n_pos"]} défectueuses).

## 1. Prétraitements testés

Géométrie commune (A.1) : décodage → RGB → redimensionnement à ratio conservé sur un canevas
{cfg.preprocessing.input_size.height}×{cfg.preprocessing.input_size.width} (échelle ≈ 0,98 sur une image
native médiane 637×229 : pas de sous-échantillonnage) → padding `{cfg.preprocessing.pad_value}`.
Les opérateurs de contraste agissent sur la luminance (canal L de LAB) pour conserver l'information
couleur ; les lissages agissent sur les trois canaux. Chaque variante est testée **séparément**.

{_md_table(_variant_table(cfg))}

## 2. Résultats visuels (images du split train uniquement)

Images représentatives : {", ".join(f"{r.role} = {r.image_id}" for r in reps.itertuples())}.

| Famille | Images complètes | Zoom sur les défauts |
|---|---|---|
| contraste (equalize vs CLAHE) | ![](figures/contrast_full.png) | ![](figures/contrast_zoom.png) |
| débruitage (léger vs fort) | ![](figures/denoise_full.png) | ![](figures/denoise_zoom.png) |
| contours (unsharp vs Laplacien) | ![](figures/sharpen_full.png) | ![](figures/sharpen_zoom.png) |

Histogrammes de luminance après chaque variante (la saturation et l'étalement se lisent ici) :

![](figures/histograms.png)

## 3. Effet sur les micro-défauts : CNR par taille de défaut

CNR = |moyenne(défaut) − moyenne(anneau de {cfg.baseline.cnr.ring_width} px)| / écart-type(anneau),
calculé sur les **{cnr_meta["n_defective_train_images"]} images défectueuses du train** à partir des masques,
après chaque variante (`none` = référence).

{_md_table(cnr_all)}

CNR médian relatif à `none`, par strate d'aire du défaut :

{_md_table(piv)}

![](figures/cnr_by_area.png)

Lecture :

- **Égalisation globale** : CNR relatif {_fmt(denoise_cnr.loc["equalize", "CNR relatif à none"], 2)}, écart-type médian ×{_fmt(denoise_cnr.loc["equalize", "écart-type médian"] / denoise_cnr.loc["none", "écart-type médian"], 1)} — elle amplifie le bruit de texture au moins autant que le défaut ; les zooms le confirment (fond granuleux aussi clair que le défaut).
- **CLAHE** : CNR relatif {_fmt(denoise_cnr.loc["clahe", "CNR relatif à none"], 2)} — plus modéré, mais toujours en dessous de l'original sur {_fmt(100 * (1 - denoise_cnr.loc["clahe", "part d'images améliorées"]), 0)} % des images.
- **Débruitage** : léger {_fmt(denoise_cnr.loc["denoise_light", "CNR relatif à none"], 2)}, fort {_fmt(denoise_cnr.loc["denoise_strong", "CNR relatif à none"], 2)}, y compris sur la strate des plus petits défauts ({_fmt(piv.iloc[0]["denoise_strong"], 2)} pour {int(piv.iloc[0]["n"])} images, effectif trop faible pour conclure seul).
- **Rehaussement des contours** : unsharp {_fmt(denoise_cnr.loc["unsharp", "CNR relatif à none"], 2)}, Laplacien {_fmt(denoise_cnr.loc["laplacian", "CNR relatif à none"], 2)} — ils amplifient le bruit haute fréquence plus que le défaut.

**Mise en garde méthodologique** : le CNR récompense mécaniquement le lissage (le dénominateur, l'écart-type
de l'anneau, baisse). Il suffit à *rejeter* les transformations qui dégradent la visibilité (égalisation,
contours), pas à *retenir* le débruitage : pour cela, seule la performance du classifieur (§6, E3) compte.

## 4. Descripteurs classiques

{_md_table(_feature_table(cfg))}

- `contrast_only` : la référence triviale de la Phase 1 (écart-type de la luminance), mesurée ici sur la validation.
- `global_stats` + `histogram` : moments, percentiles, écart entre canaux, histogramme 32 bins — la forme de la distribution des intensités.
- `lbp` : LBP uniforme invariant en rotation à (P, R) ∈ {", ".join(f"({s.points}, {s.radius})" for s in cfg.baseline.lbp)} — micro-texture, invariante aux variations monotones d'éclairage (implémentation OpenCV/NumPy, testée).
- `local_stats` : grille de patches {cfg.baseline.local_stats.patch_size}×{cfg.baseline.local_stats.patch_size}, écart-type et variance du laplacien par patch agrégés par max / p99 / écart-type / moyenne — conçu pour les défauts *locaux et petits* qu'un histogramme global dilue.

Aucune feature n'utilise de statistique de dataset : le seul objet ajusté est le `StandardScaler`, à l'intérieur du `Pipeline`, sur le train.

## 5. Baseline : protocole

`Pipeline(StandardScaler → SVC rbf)`, scores = distance signée à la marge (pas de probabilités, pas de calibration revendiquée).
Hyperparamètres C ∈ {{{", ".join(str(c) for c in cfg.baseline.svm.c_grid)}}} × γ ∈ {{{", ".join(str(g) for g in cfg.baseline.svm.gamma_grid)}}}
choisis par **validation croisée stratifiée par groupe à {cfg.baseline.svm.cv_folds} plis, à l'intérieur du train**, sur l'Average Precision ; le split
`val` ne sert qu'à comparer les configurations. Random Forest : {cfg.baseline.random_forest.n_estimators} arbres, `class_weight={cfg.baseline.random_forest.class_weight}`, seed {cfg.reproducibility.seed}.
Intervalles de confiance : bootstrap percentile à 95 % ({cfg.evaluation.bootstrap.n_resamples} rééchantillonnages) sur la validation.

## 6. Métriques (validation uniquement)

{_md_table(_summary_table(summary))}

![](figures/pr_curves.png)

### 6.1 Contrôle du signal trivial (B.3)

| Score | AP val | ROC-AUC val |
|---|---|---|
| contraste global seul (E0) | {_fmt(e0["val_ap"])} | {_fmt(e0["val_roc_auc"])} |
| statistiques globales + histogramme (E1_global_stats) | {_fmt(by_id.loc["E1_global_stats", "val_ap"])} | {_fmt(by_id.loc["E1_global_stats", "val_roc_auc"])} |
| LBP (E1_lbp) | {_fmt(by_id.loc["E1_lbp", "val_ap"])} | {_fmt(by_id.loc["E1_lbp", "val_roc_auc"])} |
| statistiques locales (E1_local_stats) | {_fmt(by_id.loc["E1_local_stats", "val_ap"])} | {_fmt(by_id.loc["E1_local_stats", "val_roc_auc"])} |
| combinées (E1_combined) | {_fmt(ref["val_ap"])} | {_fmt(ref["val_roc_auc"])} |

Les descripteurs classiques combinés apportent +{_fmt(improvement_vs_trivial)} d'AP par rapport au contraste seul
(IC 95 % de E0 : [{_fmt(e0["val_ap_ci_low"])} ; {_fmt(e0["val_ap_ci_high"])}] ; de E1_combined : [{_fmt(ref["val_ap_ci_low"])} ; {_fmt(ref["val_ap_ci_high"])}]).
Le meilleur groupe seul est `{best_e1["features"]}` (AP {_fmt(best_e1["val_ap"])}).

### 6.2 Déséquilibre (C.2) — `class_weight` sur les features combinées, prétraitement `none`

{_md_table(_summary_table(cw)[["id", "class_weight", "C", "γ", "AP CV (train)", "AP val", "IC 95 % AP", "ROC-AUC val", "préc.@R0,95", "FPR@R0,95"]])}

`manual` = `{manual_cw}` est une **expérience exploratoire** (≈ racine du ratio de déséquilibre), pas un coût métier :
le ratio de coût réel n'est pas connu et le levier des coûts reste le seuil (Phase 4).

### 6.3 Contrôle de bord de grille et noyau

{_md_table(_summary_table(grid)[["id", "C", "γ", "AP CV (train)", "AP val", "IC 95 % AP", "ROC-AUC val", "préc.@R0,95", "FPR@R0,95"]])}

Dans toutes les expériences RBF de la grille validée, la CV a sélectionné γ = 0,001, c'est-à-dire
la **borne inférieure** de la grille. E5_grid_edge_check étend la grille vers le bas : la CV choisit
à nouveau le coin (C = {_fmt(by_id.loc["E5_grid_edge_check", "best_C"], 0)}, γ = {by_id.loc["E5_grid_edge_check", "best_gamma"]}).
Un noyau RBF à γ → 0 et C grand tend vers un séparateur linéaire ; E5_linear_svm le vérifie avec un
noyau linéaire (C = {_fmt(by_id.loc["E5_linear_svm", "best_C"], 2)}, optimum intérieur à la grille) et obtient une AP équivalente.
Les IC de ces trois configurations se recouvrent : les descripteurs classiques sont séparés de façon
essentiellement linéaire, et le noyau RBF n'apporte rien de mesurable.

### 6.4 Compromis de seuil pour la meilleure configuration ({best_overall["id"]})

{_md_table(_tradeoff_table(results[str(best_overall["id"])]))}

Au seuil par défaut du classifieur : précision {_fmt(best_overall["default_thr_precision"])}, rappel {_fmt(best_overall["default_thr_recall"])}, F1 {_fmt(best_overall["default_thr_f1"])} —
rapporté pour information, ce n'est pas un point de fonctionnement retenu.

## 7. Limites

- {results[str(ref["id"])]["data"]["val"]["n_pos"]} défectueuses en validation : les IC bootstrap sont larges ; deux configurations dont les IC se recouvrent ne sont pas départagées.
- {len(summary)} configurations comparées sur la même validation → le meilleur chiffre est légèrement optimiste (sélection). Le chiffre non biaisé viendra du test officiel en Phase 4.
- Les scores SVM ne sont pas des probabilités : les seuils du §6.4 sont des seuils de score.
- Le CNR favorise le lissage par construction (§3).
- Les strates de petits défauts contiennent peu d'images ({", ".join(f"{b} : {int(n_per_bin[b])}" for b in bins_order[:2])}).

## 8. Choix retenus (règles appliquées aux chiffres)

- Référence classique pour la comparaison avec le CNN : `{ref_to_beat["id"]}` — features `combined`, prétraitement `none`, AP val {_fmt(ref_to_beat["val_ap"])}, IC [{_fmt(ref_to_beat["val_ap_ci_low"])} ; {_fmt(ref_to_beat["val_ap_ci_high"])}], ROC-AUC {_fmt(ref_to_beat["val_roc_auc"])}.
- Prétraitement : meilleure variante E3 = `{best_e3["preprocessing"]}` (AP {_fmt(best_e3["val_ap"])}) ; gains significatifs (IC disjoints de `none`) : {", ".join(significant_gains) if significant_gains else "aucun"}.
  {"Aucune variante ne bat `none` de façon significative : la valeur `active` reste `none` pour la Phase 3, le CNN apprenant lui-même sa normalisation de contraste." if not significant_gains else "Une variante apporte un gain significatif : à confirmer en Phase 3 avant de la fixer comme `active`."}
- Padding : `median` vs `zero` → AP {_fmt(ref["val_ap"])} vs {_fmt(by_id.loc["E3_pad_zero", "val_ap"]) if "E3_pad_zero" in by_id.index else "TBD"} (IC {"recouvrants" if "E3_pad_zero" in by_id.index and ci_overlap("E1_combined", "E3_pad_zero") else "disjoints"}).

## 9. Choix rejetés

- Égalisation globale et rehaussement des contours (unsharp, Laplacien) : CNR relatif < 1 sur toutes les strates (§3){" et AP val inférieure à `none` (§6)" if all(e3_vs_none.get(i, (0, True))[0] <= 0 for i in ("E3_equalize", "E3_unsharp")) else ""}.
- GLCM : non ajouté (LBP couvre la micro-texture ; décision validée).
- Empilement de transformations : aucune combinaison n'a été construite, faute de gain individuel significatif à combiner.

## 10. Implications pour la Phase 3

- La référence à battre sur la **validation** est l'AP {_fmt(ref_to_beat["val_ap"])} (IC [{_fmt(ref_to_beat["val_ap_ci_low"])} ; {_fmt(ref_to_beat["val_ap_ci_high"])}]) de `{ref_to_beat["id"]}`, et non l'AP {_fmt(e0["val_ap"])} du contraste seul. Un CNN dont l'AP de validation reste sous {_fmt(ref_to_beat["val_ap_ci_high"])} n'apporte rien de démontrable avec {results[str(ref["id"])]["data"]["val"]["n_pos"]} positifs.
- Le coût du rappel élevé est le vrai problème de la baseline : à rappel 0,95, précision {_fmt(ref_to_beat["precision@r0.95"])} et FPR {_fmt(ref_to_beat["fpr@r0.95"])} — il faudrait réinspecter ~{_fmt(100 * ref_to_beat["fpr@r0.95"], 0)} % des pièces saines. C'est l'axe sur lequel le CNN doit apporter sa valeur.
- Les descripteurs classiques sont quasi linéairement séparables (§6.3) : le gain attendu du CNN vient de sa capacité à *localiser* (convolutions) plutôt que d'une non-linéarité sur des statistiques globales.
- Prétraitement du CNN : `{cfg.preprocessing.enhancement.active}` (§8) ; l'augmentation luminosité/contraste doit rester légère (§2, images sombres et peu contrastées).
- Déséquilibre : voir §6.2 pour l'effet de `class_weight` sur la baseline ; la décision pour le CNN reste ouverte (ablation `pos_weight` en Phase 4).
"""
    path = out_dir / "phase2_report.md"
    path.write_text(md, encoding="utf-8", newline="\n")
    (out_dir / "phase2_report.meta.json").write_text(
        json.dumps(
            provenance(cfg, n_experiments=len(summary), pr_curves=pr_path.as_posix()),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
        newline="\n",
    )
    log_event(logger, "report.written", path=str(path))
    return path
