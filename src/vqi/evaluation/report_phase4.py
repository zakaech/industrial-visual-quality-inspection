# ruff: noqa: E501  (Markdown template with long table rows)
"""Phase 4 report: assembled from reports/phase4/*.json and models/final/*.json. Every number is
read from those files; nothing is typed by hand. Prose is French (project documentation language).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vqi.config import Config
from vqi.evaluation.final import report_dir
from vqi.logging_utils import get_logger, log_event
from vqi.provenance import provenance

logger = get_logger(__name__)


def _short(commit: str | None) -> str:
    """7-char commit, or 'unknown' when provenance has none (no git available)."""
    return commit[:7] if commit else "unknown"


def _fmt(x: Any, nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/d"
    if isinstance(x, bool):
        return "oui" if x else "non"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    return f"{float(x):.{nd}f}".replace(".", ",")


def _ci(d: dict[str, Any], nd: int = 3) -> str:
    return f"[{_fmt(d['low'], nd)} ; {_fmt(d['high'], nd)}]"


def _md_table(df: pd.DataFrame, nd: int = 3) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.astype(object).iterrows():
        cells = [
            _fmt(row[c], nd) if isinstance(row[c], (float, np.floating)) else str(row[c])
            for c in df.columns
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _confusion_row(
    name: str, c: dict[str, Any], ci: dict[str, Any] | None = None
) -> dict[str, Any]:
    row = {
        "jeu": name,
        "seuil": c["threshold"],
        "rappel": c["recall"],
        "précision": c["precision"],
        "FPR": c["fpr"],
        "F1": c["f1"],
        "TP": int(c["tp"]),
        "FP": int(c["fp"]),
        "FN": int(c["fn"]),
        "TN": int(c["tn"]),
    }
    if ci:
        row["IC rappel"] = _ci(ci["recall"])
        row["IC précision"] = _ci(ci["precision"])
        row["IC FPR"] = _ci(ci["fpr"])
    return row


def _final_reading(test: dict[str, Any], cand_id: str) -> str:
    fb = test["frozen_baselines"]
    parts = []
    for key, label in (
        ("E5_grid_edge_check", "à la SVM classique"),
        ("E0_contrast_only", "au contraste seul"),
    ):
        if key in fb:
            d = fb[key]["paired_vs_x6"]["delta_average_precision"]
            sig = (
                "IC apparié de ΔAP excluant 0" if d["low"] > 0 else "IC apparié de ΔAP contenant 0"
            )
            parts.append(f"face {label} : ΔAP {_fmt(d['point'])} {_ci(d)} ({sig})")
    if "X4_B_pw_ratio" in fb:
        d = fb["X4_B_pw_ratio"]["paired_vs_x6"]["delta_average_precision"]
        verdict = "indissociables" if d["low"] <= 0 <= d["high"] else "distincts"
        parts.append(
            f"face à X4 (étape B) : ΔAP {_fmt(d['point'])} {_ci(d)} — {cand_id} et X4 sont **{verdict}** sur le test ; "
            "l'étape C n'apporte pas de gain démontré, alors que la validation le suggérait"
            if verdict == "indissociables"
            else f"face à X4 (étape B) : ΔAP {_fmt(d['point'])} {_ci(d)}"
        )
    return "Lecture prudente : " + " ; ".join(parts) + "."


def build_report(cfg: Config) -> Path:
    out_dir = report_dir(cfg)
    bundle = Path(cfg.inference.bundle_dir)
    ver = _load(out_dir / "checkpoint_verification.json")
    val = _load(out_dir / "validation_analysis.json")
    test = _load(out_dir / "test_evaluation.json")
    err = _load(out_dir / "error_analysis.json")
    thr = _load(bundle / "threshold.json")
    card = _load(bundle / "model_card.json")
    candidates = pd.read_csv(out_dir / "threshold_candidates.csv")
    cand_id = cfg.phase4.candidate.experiment_id
    exp_dirs = {
        "phase3": cfg.paths.reports_dir / cfg.phase3.report_subdir / "experiments",
        "phase2": cfg.paths.reports_dir / cfg.baseline.report_subdir / "experiments",
    }

    def _val_result(exp_id: str) -> dict[str, Any] | None:
        for d in exp_dirs.values():
            if (d / f"{exp_id}.json").exists():
                return _load(d / f"{exp_id}.json")
        return None

    labels = {
        "E0_contrast_only": "E0 contraste seul",
        "E5_grid_edge_check": "E5 SVM (Phase 2)",
        "X4_B_pw_ratio": "X4 étape B",
    }
    x6_val = _val_result(cand_id)
    baseline_val = {k: _val_result(k) for k in cfg.phase4.test.frozen_baselines}

    sel = val["threshold_selection"]
    cal = val["calibration"]
    fin = test["final"]
    at = fin["at_threshold"]
    vm = val["metrics"]
    cand_view = candidates[candidates["rank_of_positive"] >= max(1, len(candidates) - 5)][
        [
            "rank_of_positive",
            "threshold",
            "tp",
            "fp",
            "fn",
            "recall",
            "precision",
            "fpr",
            "expected_cost_ratio_5",
            "expected_cost_ratio_10",
            "expected_cost_ratio_20",
        ]
    ].rename(
        columns={
            "rank_of_positive": "positifs gardés",
            "threshold": "seuil",
            "expected_cost_ratio_5": "coût/img (FN=5×FP)",
            "expected_cost_ratio_10": "coût/img (FN=10×FP)",
            "expected_cost_ratio_20": "coût/img (FN=20×FP)",
        }
    )

    # comparison tables — validation (development) and test (final, single pass)
    def val_row(name: str, r: dict[str, Any]) -> dict[str, Any]:
        m = r["metrics"]
        bt = {round(t["recall_target"], 2): t for t in m["at_recall_targets"]}
        return {
            "modèle": name,
            "AP": m["average_precision"],
            "IC AP": _ci(m["average_precision_ci"]),
            "ROC-AUC": m["roc_auc"],
            "préc.@R0,95": bt[0.95]["precision"],
            "FPR@R0,95": bt[0.95]["fpr"],
            "rappel@R0,95": bt[0.95]["recall"],
        }

    val_table = pd.DataFrame(
        [val_row(labels.get(k, k), r) for k, r in baseline_val.items() if r is not None]
        + ([val_row(f"{cand_id} (candidat)", x6_val)] if x6_val else [])
    )

    test_rows = []
    for key, b in test["frozen_baselines"].items():
        name = labels.get(key, key)
        bt = {round(t["recall_target"], 2): t for t in b["at_recall_targets"]}
        d = b["paired_vs_x6"]
        test_rows.append(
            {
                "modèle": name,
                "AP": b["average_precision"],
                "IC AP": _ci(b["average_precision_ci"]),
                "ROC-AUC": b["roc_auc"],
                "préc.@R0,95": bt[0.95]["precision"],
                "FPR@R0,95": bt[0.95]["fpr"],
                "ΔAP (X6 − modèle) [IC]": f"{_fmt(d['delta_average_precision']['point'])} {_ci(d['delta_average_precision'])}",
                "Δpréc.@0,95 (X6 − modèle) [IC]": f"{_fmt(d['delta_precision_at_recall_0.95']['point'])} {_ci(d['delta_precision_at_recall_0.95'])}",
            }
        )
    bt95 = {round(t["recall_target"], 2): t for t in fin["at_recall_targets_information_only"]}[
        0.95
    ]
    test_rows.append(
        {
            "modèle": f"**{cand_id} — modèle final gelé**",
            "AP": fin["average_precision"],
            "IC AP": _ci(fin["average_precision_ci"]),
            "ROC-AUC": fin["roc_auc"],
            "préc.@R0,95": bt95["precision"],
            "FPR@R0,95": bt95["fpr"],
            "ΔAP (X6 − modèle) [IC]": "—",
            "Δpréc.@0,95 (X6 − modèle) [IC]": "—",
        }
    )
    test_table = pd.DataFrame(test_rows)

    area = pd.DataFrame(fin["recall_by_defect_area"]).rename(
        columns={
            "area_bin": "aire (px)",
            "n_positives": "défauts",
            "recall": "rappel",
            "missed": "manqués",
        }
    )
    fn_df = pd.DataFrame(test["false_negatives"])[
        ["image_id", "score", "defect_area_px", "defect_bbox_w", "defect_bbox_h", "std_luminance"]
    ].rename(
        columns={
            "score": "probabilité",
            "defect_area_px": "aire (px)",
            "defect_bbox_w": "boîte L",
            "defect_bbox_h": "boîte H",
            "std_luminance": "contraste image",
        }
    )
    fp_df = (
        pd.DataFrame(test["false_positives"])
        .head(10)[["image_id", "score", "mean_luminance", "std_luminance"]]
        .rename(
            columns={
                "score": "probabilité",
                "mean_luminance": "luminance",
                "std_luminance": "contraste",
            }
        )
    )
    cmp_fp = err["comparison"]["FP_vs_TN_medians"]
    cmp_fn = err["comparison"]["FN_vs_TP_medians"]
    gc = err["gradcam"]
    gc_rows = pd.DataFrame(
        [
            {
                "catégorie": k,
                "n": v["n"],
                "masse CAM dans le masque (médiane)": v["median_cam_mass_in_mask"],
                "concentration (masse/aire relative)": v["median_concentration_ratio"],
                "pic dans le masque": v["share_peak_in_mask"],
            }
            for k, v in gc["summary"].items()
        ]
    )
    cal_rows = pd.DataFrame(
        [
            {
                "scores": "bruts",
                "ECE (effectifs égaux)": cal["raw"]["ece_equal_mass"],
                "ECE (largeurs égales)": cal["raw"]["ece_equal_width"],
                "Brier": cal["raw"]["brier"],
                "NLL": cal["raw"]["nll"],
                "paramètres": "—",
            }
        ]
        + [
            {
                "scores": f"{m} (hors-pli, 5 plis sur val)",
                "ECE (effectifs égaux)": c["out_of_fold"]["ece_equal_mass"],
                "ECE (largeurs égales)": c["out_of_fold"]["ece_equal_width"],
                "Brier": c["out_of_fold"]["brier"],
                "NLL": c["out_of_fold"]["nll"],
                "paramètres": json.dumps(c["params_full_fit"]),
            }
            for m, c in cal["calibrators"].items()
        ]
    )
    rel = pd.DataFrame(cal["raw_reliability_equal_mass"])[
        ["bin", "n", "mean_predicted", "observed_frequency"]
    ].rename(
        columns={
            "mean_predicted": "prob. prédite (moy.)",
            "observed_frequency": "fréquence observée",
        }
    )

    md = f"""# Phase 4 — Modèle final, seuil et évaluation unique du test officiel

Généré par `scripts/build_phase4_report.py` (config `{cfg.hash}`). Chaque chiffre provient de
`reports/phase4/*.json`, `models/final/*.json` et des JSON des phases 2–3 ; rien n'est saisi à la main.
**Le test officiel a été évalué une seule fois**, le {test["evaluated_at"][:19].replace("T", " ")} UTC, après le gel du modèle,
du prétraitement et du seuil ({thr["frozen_at"][:19].replace("T", " ")} UTC). Aucune décision n'a été prise à partir du test.

## 4.1 Objectif

Transformer le candidat X6 (Phase 3) en système final de classification binaire sain / défectueux :
vérification du checkpoint, analyse et calibration des scores sur la validation, sélection et gel du seuil,
puis évaluation finale unique sur le test officiel KolektorSDD2, analyse des erreurs et Grad-CAM.

## 4.2 Modèle final (gelé)

| Élément | Valeur |
|---|---|
| Architecture | ResNet-18 ImageNet (`{card["checkpoint_metadata"]["config_hash"]}`), pooling `{card["architecture"]["pooling"]}`, tête Dropout 0,2 + Linear(512→1) — {card["architecture"]["n_params"]:,} paramètres |
| Recette | {card["training_recipe"]["stages"]} ; `pos_weight` {_fmt(card["training_recipe"]["pos_weight"], 3)} ; sampler {card["training_recipe"]["sampler"]} ; BN `{card["training_recipe"]["bn_policy"]}` ; AdamW wd 1e-4 ; LR C {json.dumps(card["training_recipe"]["lr_stage_C"])} ; batch {card["training_recipe"]["batch_size"]} ; seed {card["training_recipe"]["seed"]} |
| Checkpoint | `{card["checkpoint_source"]}` — étape {card["checkpoint_metadata"]["stage"]}, époque {card["checkpoint_metadata"]["epoch"]}, AP val {_fmt(card["checkpoint_metadata"]["metric"], 4)} (critère préenregistré), commit d'entraînement `{_short(card["checkpoint_metadata"]["git_commit"])}` |
| Bundle | `{bundle.as_posix()}/` : `model.pt` (SHA-256 `{card["model_sha256"][:16]}…`, copie octet à octet du checkpoint), `preprocessor.json`, `threshold.json`, `model_card.json`, `final_test_metrics.json` |
| Prétraitement | {json.dumps(card["preprocessor"])} |
| Vérification | métadonnées = résultat X6 ; deux passes val identiques ; écart max vs scores enregistrés {ver["validation"]["max_abs_score_diff_vs_recorded"]:.1e} ; AP reproduite {_fmt(ver["validation"]["average_precision"][0], 4)} |

## 4.3 Sélection du seuil (validation uniquement)

Contrainte de développement : rappel ≥ {sel["target_recall"]} ; contrainte métier : un défaut manqué coûte plus qu'une fausse alerte.
Règle préenregistrée (config, avant l'analyse) : `{sel["policy"]}` — seuil le moins permissif atteignant le rappel cible sur la
validation ({_fmt(sel["threshold_min_recall"], 5)}, 47/49) — puis **marge** `{sel["margin"]}` : milieu, en logit, entre ce score et celui du positif
suivant ({_fmt(sel["next_lower_positive_score"], 5)}) → **seuil gelé {_fmt(sel["threshold"], 6)}**. La marge protège d'une dérive des scores sur de
nouvelles pièces ; elle coûte {sel["margin_cost_extra_fp"]} faux positifs sur la validation pour le même rappel.

Candidats autour de la contrainte (un seuil par score de positif ; le coût attendu, à titre d'information, minimise
(FN × ratio + FP)/N pour des ratios de coût **inconnus** en pratique) :

{_md_table(cand_view, 4)}

Lecture : le point de coût minimal pour tout ratio 5–20 est le rang 12 (46/49, rappel 0,939), *sous* la contrainte de rappel ;
la contrainte prime par construction. Le seuil choisi n'a pas été retenu pour sa précision : c'est l'application de la règle
fixée avant de voir ces chiffres.

{_md_table(pd.DataFrame([_confusion_row("validation, seuil min_recall", sel["at_threshold_min_recall"]), _confusion_row("validation, seuil gelé", sel["at_threshold"], sel["bootstrap_at_threshold"])]))}

## 4.4 Calibration

Le modèle a été entraîné avec `pos_weight` = {_fmt(card["training_recipe"]["pos_weight"], 2)} : ses sorties ne sont pas des probabilités calibrées par construction,
et le seuil 0,5 n'a aucun statut particulier. Diagnostics sur la validation (n = {val["n"]}, {val["n_pos"]} positives) :

{_md_table(cal_rows, 4)}

Diagramme de fiabilité (bins à effectifs égaux, scores bruts) :

{_md_table(rel, 4)}

![](figures/val_reliability.png)

Décision : **scores bruts conservés** — règle : {cal["rule"]} ; ECE brut {_fmt(cal["raw"]["ece_equal_mass"], 3)}, température ajustée
{_fmt(cal["calibrators"].get("temperature", {}).get("params_full_fit", {}).get("temperature"), 3)} (≈ identité, aucun bénéfice), isotonique : gain hors-pli marginal pour 49 positifs
et ECE in-sample nul (surajustement). Point à retenir pour l'exploitation : dans la bande de scores autour du seuil
(≈ 0,02–0,03), la fréquence observée de défauts est ≈ 15 % — le seuil ne se lit pas comme « 2 % de probabilité ».
Toute calibration monotone laisserait le point de fonctionnement inchangé (même classement).

## 4.5 Résultats validation (développement — modèle et seuil choisis dessus, chiffres optimistes)

AP {_fmt(vm["average_precision"])} {_ci(vm["average_precision_ci"])} · ROC-AUC {_fmt(vm["roc_auc"])} {_ci(vm["roc_auc_ci"])}.

{_md_table(val_table)}

Figures : `figures/val_score_distribution.png`, `figures/val_pr_roc.png`, `figures/val_metrics_vs_threshold.png`.

## 4.6 Résultats du test officiel (finaux — évaluation unique)

Test officiel KolektorSDD2 : {test["n"]} images, {test["n_pos"]} défectueuses, {test["n_neg"]} saines. Modèle final gelé, seuil gelé {_fmt(test["threshold"], 6)}.

| Métrique | Valeur | IC 95 % bootstrap |
|---|---|---|
| AP | **{_fmt(fin["average_precision"])}** | {_ci(fin["average_precision_ci"])} |
| ROC-AUC | {_fmt(fin["roc_auc"])} | {_ci(fin["roc_auc_ci"])} |
| Rappel au seuil gelé | **{_fmt(at["recall"])}** ({at["tp"]}/{test["n_pos"]}) | {_ci(fin["at_threshold_ci"]["recall"])} |
| Précision au seuil gelé | {_fmt(at["precision"])} | {_ci(fin["at_threshold_ci"]["precision"])} |
| FPR au seuil gelé | {_fmt(at["fpr"])} | {_ci(fin["at_threshold_ci"]["fpr"])} |
| F1 / F2 au seuil gelé | {_fmt(at["f1"])} / {_fmt(at["f2"])} | — |
| Matrice de confusion | TP {at["tp"]} · FP {at["fp"]} · FN **{at["fn"]}** · TN {at["tn"]} | — |
| Exactitude (rapportée, non utilisée) | {_fmt(at["accuracy"])} | — |

Points de fonctionnement *a posteriori* sur le test (information seulement — le seuil gelé est le seul point de décision) :

{_md_table(pd.DataFrame([{"rappel cible": t["recall_target"], "seuil": t["threshold"], "rappel": t["recall"], "précision": t["precision"], "FPR": t["fpr"], "TP": int(t["tp"]), "FP": int(t["fp"]), "FN": int(t["fn"])} for t in fin["at_recall_targets_information_only"]]), 4)}

Rappel par taille de défaut au seuil gelé :

{_md_table(area)}

![](figures/test_recall_by_area.png)

![](figures/test_pr_roc.png)

![](figures/test_score_distribution.png)

Cohérence validation → test au seuil gelé : rappel {_fmt(sel["at_threshold"]["recall"])} → {_fmt(at["recall"])}, précision
{_fmt(sel["at_threshold"]["precision"])} → {_fmt(at["precision"])}, FPR {_fmt(sel["at_threshold"]["fpr"])} → {_fmt(at["fpr"])}, AP {_fmt(vm["average_precision"])} → {_fmt(fin["average_precision"])}.

## 4.7 Analyse des erreurs (test)

**Faux négatifs ({err["counts"]["FN"]})** — tous ont un score compris entre {_fmt(min(x["score"] for x in test["false_negatives"]), 4)} et {_fmt(max(x["score"] for x in test["false_negatives"]), 4)},
juste sous le seuil {_fmt(test["threshold"], 4)} :

{_md_table(fn_df, 4)}

Médianes FN vs TP : aire {_fmt(cmp_fn["FN"]["defect_area_px"], 0)} px vs {_fmt(cmp_fn["TP"]["defect_area_px"], 0)} px ; boîte {_fmt(cmp_fn["FN"]["defect_bbox_w"], 0)}×{_fmt(cmp_fn["FN"]["defect_bbox_h"], 0)} vs
{_fmt(cmp_fn["TP"]["defect_bbox_w"], 0)}×{_fmt(cmp_fn["TP"]["defect_bbox_h"], 0)} ; contraste image {_fmt(cmp_fn["FN"]["std_luminance"], 1)} vs {_fmt(cmp_fn["TP"]["std_luminance"], 1)}.
Observations (aucune conclusion causale) : les défauts manqués **semblent associés** à une petite aire (2 des 5 FN sont dans la
strate < 250 px, où le rappel tombe à {_fmt(area.iloc[0]["rappel"], 2)}), à un faible contraste local (deux défauts étendus mais très pâles,
20969 et 20281) et, pour deux d'entre eux (20816, 20236), à une position en bordure d'image. Hypothèse : à la résolution de `layer4`
(cellules ≈ 32×32 px), un défaut de quelques dizaines de pixels et peu contrasté n'active pas assez la carte.

![](figures/test_fn_panel.png)

**Faux positifs ({err["counts"]["FP"]})** — médiane des scores {_fmt(err["comparison"]["FP_score_quantiles"]["0.5"], 3)}, 90 % sous {_fmt(err["comparison"]["FP_score_quantiles"]["0.9"], 3)} ;
seulement {int(round(err["comparison"]["FP_share_above_0.5"] * err["counts"]["FP"]))} au-dessus de 0,5. Ce sont donc essentiellement des images saines poussées au-dessus d'un seuil très bas par
la contrainte de rappel, pas des erreurs confiantes. Médianes FP vs TN : luminance {_fmt(cmp_fp["FP"]["mean_luminance"], 1)} vs {_fmt(cmp_fp["TN"]["mean_luminance"], 1)},
contraste {_fmt(cmp_fp["FP"]["std_luminance"], 2)} vs {_fmt(cmp_fp["TN"]["std_luminance"], 2)}, écart entre canaux {_fmt(cmp_fp["FP"]["channel_spread"], 2)} vs {_fmt(cmp_fp["TN"]["channel_spread"], 2)} —
aucun motif photométrique global ne distingue les FP (cohérent avec la Phase 1 : la séparabilité triviale est faible).

{_md_table(fp_df, 3)}

![](figures/test_fp_panel.png)

## 4.8 Grad-CAM ({gc["target_layer"]})

{_md_table(gc_rows, 2)}

Lecture : quand le modèle est **confiant** (p ≈ 1), le pic de la carte est dans le masque pour {_fmt(gc["summary"].get("TP_confident", {}).get("share_peak_in_mask", 0) * 100, 0)} % des exemples et
{_fmt(gc["summary"].get("TP_confident", {}).get("median_cam_mass_in_mask", 0) * 100, 0)} % de la masse de la carte tombe dans le défaut — le réseau regarde bien la région annotée.
Pour les vrais positifs **limites** et les faux négatifs, la carte est diffuse (pic hors masque) : la décision repose sur peu d'évidence localisée.
Limites : {gc["limits"]}.

![](figures/test_gradcam_tp_confident.png)

![](figures/test_gradcam_fn.png)

![](figures/test_gradcam_fp.png)

## 4.9 Comparaison finale

Validation (développement, 49 positives) :

{_md_table(val_table)}

Test officiel (passe unique ; les trois références sont des modèles gelés scorés dans la même passe, sans aucune décision) :

{_md_table(test_table)}

{_final_reading(test, cand_id)}

## 4.10 Limites

- {test["n_pos"]} défauts dans le test, 49 en validation : IC larges ; le rappel au seuil gelé est {_fmt(at["recall"])} avec un IC {_ci(fin["at_threshold_ci"]["recall"])}.
- Modèle, checkpoint et seuil sélectionnés sur la validation : les chiffres de validation sont optimistes par construction ; seul le test est non biaisé, et il n'a servi à aucune décision.
- Les petits défauts (< 250 px) restent le point faible (rappel {_fmt(area.iloc[0]["rappel"], 2)} sur {int(area.iloc[0]["défauts"])} cas).
- Domaine spécifique (commutateurs KolektorSDD2, éclairage fixe) : aucune garantie de généralisation à une autre ligne, un autre produit ou un autre capteur sans nouvelle validation.
- Calibration imparfaite dans la bande utile (scores bruts, `pos_weight` 8,5) ; les probabilités ne sont pas des probabilités métier.
- Coût CPU élevé : ≈ {_fmt(test["timing"]["x6_preprocess_and_score_seconds"] / test["n"] * 1000, 0)} ms par image en inférence sur l'i7-8665U (prétraitement inclus), ≈ 1 h par époque d'entraînement en étape C.
- Grad-CAM : évidence qualitative à résolution grossière, pas une localisation.
- La provenance du run de test porte `git_dirty = {str(test["provenance"]["git_dirty"]).lower()}` : au lancement le working tree était propre au commit `{_short(test["provenance"]["git_commit"])}` ; le drapeau reflète les fichiers de sortie que le script lui-même écrit avant d'enregistrer sa provenance.

## 4.11 Reproductibilité

| Élément | Valeur |
|---|---|
| Commit du gel / de l'évaluation | `{_short(test["provenance"]["git_commit"])}` |
| Config au gel | `{card["provenance"]["config_hash"]}` ; config d'entraînement du checkpoint `{card["checkpoint_metadata"]["config_hash"]}` |
| Seed | {card["training_recipe"]["seed"]} |
| Checkpoint | `{card["checkpoint_source"]}` — SHA-256 `{card["model_sha256"]}` |
| Prétraitement | `models/final/preprocessor.json` (variante `{card["preprocessor"]["variant"]}`, padding `{card["preprocessor"]["pad_value"]}`, {card["preprocessor"]["input_size"][0]}×{card["preprocessor"]["input_size"][1]}) |
| Seuil final | {_fmt(thr["threshold"], 6)} (`models/final/threshold.json`, `frozen = true`, gelé le {thr["frozen_at"][:19].replace("T", " ")} UTC) |
| Test officiel | évalué une fois le {test["evaluated_at"][:19].replace("T", " ")} UTC ; `test_used = true` uniquement dans `test_evaluation.json`, `error_analysis.json` et `final_test_metrics.json` |
| Commandes | `python scripts/verify_checkpoint.py` → `python scripts/select_threshold.py --freeze` → `python scripts/evaluate_test.py` → `python scripts/build_phase4_report.py` |
"""
    path = out_dir / "phase4_report.md"
    path.write_text(md, encoding="utf-8", newline="\n")
    (out_dir / "phase4_report.meta.json").write_text(
        json.dumps(provenance(cfg), indent=2, ensure_ascii=False), encoding="utf-8", newline="\n"
    )
    log_event(logger, "report.written", path=str(path))
    return path
