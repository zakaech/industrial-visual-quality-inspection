# ruff: noqa: E501  (Markdown template with long table rows)
"""Phase 3 report: assembles ``reports/phase3/phase3_report.md`` and its figures from the
experiment JSONs written by ``training.experiments`` and the Phase 2 reference. Every number is
read from those files; nothing is typed by hand. Prose is French.
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

from vqi.config import Config  # noqa: E402
from vqi.logging_utils import get_logger, log_event  # noqa: E402
from vqi.provenance import provenance  # noqa: E402
from vqi.training.experiments import (  # noqa: E402
    experiment_json,
    phase2_reference,
    report_dir,
    summarise,
)

logger = get_logger(__name__)


def _fmt(x: Any, nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "TBD"
    if isinstance(x, bool):
        return "oui" if x else "non"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    return f"{float(x):.{nd}f}".replace(".", ",")


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


def load_results(cfg: Config) -> dict[str, dict[str, Any]]:
    out = {}
    for spec in cfg.phase3.experiments:
        p = experiment_json(cfg, spec.id)
        if p.exists():
            out[spec.id] = json.loads(p.read_text(encoding="utf-8"))
    return out


# --------------------------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------------------------- #


def pr_figure(cfg: Config, results: dict[str, dict[str, Any]], path: Path) -> Path:
    ref_id, _, ref_result = phase2_reference(cfg)
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    c = ref_result["metrics"]["curves"]
    ax.plot(
        c["pr_recall"],
        c["pr_precision"],
        color="black",
        linewidth=1.6,
        label=f"Phase 2 · {ref_id} (AP {ref_result['metrics']['average_precision']:.3f})",
    )
    for exp_id, r in results.items():
        c = r["metrics"]["curves"]
        ax.plot(
            c["pr_recall"],
            c["pr_precision"],
            linewidth=1.2,
            label=f"{exp_id} (AP {r['metrics']['average_precision']:.3f})",
        )
    prevalence = ref_result["metrics"]["prevalence"]
    ax.axhline(
        prevalence, color="gray", linestyle="--", linewidth=0.8, label=f"hasard ({prevalence:.3f})"
    )
    ax.axvline(0.95, color="tab:red", linestyle=":", linewidth=0.8, label="rappel 0,95")
    ax.set_xlabel("rappel (défauts détectés)")
    ax.set_ylabel("précision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(
        "Courbes précision–rappel sur la validation (466 images, 49 défectueuses)", fontsize=10
    )
    ax.legend(fontsize=7)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def history_figure(results: dict[str, dict[str, Any]], path: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for exp_id, r in results.items():
        h = pd.DataFrame([row for row in r["history"] if row["epoch"] > 0])
        h["global_epoch"] = range(1, len(h) + 1)
        axes[0].plot(h["global_epoch"], h["train_loss"], linewidth=1, label=f"{exp_id} train")
        axes[0].plot(
            h["global_epoch"], h["val_loss"], linewidth=1, linestyle="--", label=f"{exp_id} val"
        )
        axes[1].plot(h["global_epoch"], h["val_average_precision"], linewidth=1.2, label=exp_id)
        axes[2].plot(h["global_epoch"], h["val_precision_r095"], linewidth=1.2, label=exp_id)
        for ax in axes:
            for k in np.flatnonzero(h["stage"].ne(h["stage"].shift()).to_numpy())[1:]:
                ax.axvline(k + 0.5, color="gray", linewidth=0.6, linestyle=":")
    axes[0].set_title("perte (train, val --)", fontsize=9)
    axes[1].set_title("AP validation par époque", fontsize=9)
    axes[2].set_title("précision @ rappel ≥ 0,95 (validation)", fontsize=9)
    for ax in axes:
        ax.set_xlabel("époque (toutes étapes)")
        ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------------------------- #


def _tradeoff(r: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "rappel cible": t["recall_target"],
                "seuil (prob.)": t["threshold"],
                "rappel obtenu": t["recall"],
                "précision": t["precision"],
                "FPR": t["fpr"],
                "TP": int(t["tp"]),
                "FP": int(t["fp"]),
                "FN": int(t["fn"]),
                "TN": int(t["tn"]),
            }
            for t in r["metrics"]["at_recall_targets"]
        ]
    )


def _winner_paragraph(w: dict[str, Any] | None, n_val_pos: int) -> str:
    if w is None:
        return ""
    d_ap, d_p, d_f = (
        w["delta_average_precision"],
        w["delta_precision_at_recall_0.95"],
        w["delta_fpr_at_recall_0.95"],
    )
    return (
        f"Comparaison appariée avec le run de départ `{w['reference']}` : ΔAP {_fmt(d_ap['point'])} "
        f"[{_fmt(d_ap['low'])} ; {_fmt(d_ap['high'])}] ; Δprécision@0,95 {_fmt(d_p['point'])} "
        f"[{_fmt(d_p['low'])} ; {_fmt(d_p['high'])}] ; ΔFPR@0,95 {_fmt(d_f['point'])} "
        f"[{_fmt(d_f['low'])} ; {_fmt(d_f['high'])}] — un IC contenant 0 n'est pas départageable avec {n_val_pos} positifs."
    )


def _comparison_table(cfg: Config, results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    ref_id, _, ref = phase2_reference(cfg)
    by_t = {round(t["recall_target"], 2): t for t in ref["metrics"]["at_recall_targets"]}
    rows = [
        {
            "modèle": f"Phase 2 · {ref_id}",
            "AP": ref["metrics"]["average_precision"],
            "IC 95 % AP": f"[{_fmt(ref['metrics']['average_precision_ci']['low'])} ; {_fmt(ref['metrics']['average_precision_ci']['high'])}]",
            "préc.@R0,90": by_t[0.9]["precision"],
            "préc.@R0,95": by_t[0.95]["precision"],
            "FPR@R0,95": by_t[0.95]["fpr"],
            "préc.@R0,98": by_t[0.98]["precision"],
            "ROC-AUC": ref["metrics"]["roc_auc"],
            "ΔAP apparié [IC]": "—",
            "Δpréc.@0,95 apparié [IC]": "—",
        }
    ]
    for exp_id, r in results.items():
        m = r["metrics"]
        bt = {round(t["recall_target"], 2): t for t in m["at_recall_targets"]}
        cmp_ = r["comparison_vs_phase2"]
        d_ap, d_p = cmp_["delta_average_precision"], cmp_["delta_precision_at_recall_0.95"]
        rows.append(
            {
                "modèle": exp_id,
                "AP": m["average_precision"],
                "IC 95 % AP": f"[{_fmt(m['average_precision_ci']['low'])} ; {_fmt(m['average_precision_ci']['high'])}]",
                "préc.@R0,90": bt[0.9]["precision"],
                "préc.@R0,95": bt[0.95]["precision"],
                "FPR@R0,95": bt[0.95]["fpr"],
                "préc.@R0,98": bt[0.98]["precision"],
                "ROC-AUC": m["roc_auc"],
                "ΔAP apparié [IC]": f"{_fmt(d_ap['point'], 3)} [{_fmt(d_ap['low'])} ; {_fmt(d_ap['high'])}]",
                "Δpréc.@0,95 apparié [IC]": f"{_fmt(d_p['point'], 3)} [{_fmt(d_p['low'])} ; {_fmt(d_p['high'])}]",
            }
        )
    return pd.DataFrame(rows)


def winner_comparison(cfg: Config, r: dict[str, Any]) -> dict[str, Any] | None:
    """Paired bootstrap between an experiment and the run it started from (same val images)."""
    winner = r["spec"].get("winner")
    if not winner:
        return None
    from vqi.evaluation.compare import align_scores, paired_bootstrap

    base = report_dir(cfg) / "experiments"
    a = pd.read_csv(base / f"{winner}_val_scores.csv", dtype={"image_id": str})
    b = pd.read_csv(base / f"{r['spec']['id']}_val_scores.csv", dtype={"image_id": str})
    ids, sa, sb = align_scores(a["image_id"], a["score"], b["image_id"], b["score"])
    y = a.set_index("image_id").loc[list(ids), "label"].to_numpy().astype(int)
    out = paired_bootstrap(
        y,
        sa,
        sb,
        cfg.phase3.paired_bootstrap.n_resamples,
        cfg.phase3.paired_bootstrap.confidence_level,
        cfg.reproducibility.seed,
    )
    out["reference"] = winner
    return out


def build_report(cfg: Config) -> Path:
    out_dir = report_dir(cfg)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    results = load_results(cfg)
    if not results:
        raise FileNotFoundError("no Phase 3 experiment result found")
    summary = summarise(cfg)
    pr_path = pr_figure(cfg, results, fig_dir / "pr_curves.png")
    hist_path = history_figure(results, fig_dir / "training_curves.png")
    ref_id, _, ref = phase2_reference(cfg)
    n_val_pos = next(iter(results.values()))["data"]["val"]["n_pos"]

    sections = [
        f"""# Phase 3 — Résultats CNN (validation uniquement)

Généré par `scripts/build_phase3_report.py` (config `{cfg.hash}`, seed {cfg.reproducibility.seed}). Chaque chiffre
provient de `reports/phase3/experiments/*.json` ; le test officiel n'a été chargé par aucun run
(`test_used = false` dans chaque JSON). Validation : 466 images dont {n_val_pos} défectueuses — un écart de
quelques centièmes n'est pas interprétable ; les IC bootstrap et le bootstrap **apparié** contre la référence
Phase 2 (`{ref_id}`, AP {_fmt(ref["metrics"]["average_precision"])}) sont la seule base de comparaison.

## 1. Comparaison avec la référence Phase 2

{_md_table(_comparison_table(cfg, results))}

![]({pr_path.relative_to(out_dir).as_posix()})

## 2. Courbes d'entraînement

![]({hist_path.relative_to(out_dir).as_posix()})
"""
    ]
    for exp_id, r in results.items():
        stages = r["run"]["stages"]
        stage_rows = pd.DataFrame(
            [
                {
                    "étape": s["name"],
                    "entraînable": s["trainable"]["total_trainable"],
                    "LR": json.dumps(s["lr"]),
                    "époques": s["epochs_run"],
                    "meilleure": s["best_epoch"],
                    "AP val (best)": s["best_metric"],
                    "arrêt anticipé": s["early_stopped"],
                    "durée (s)": s["seconds"],
                    "features gelées (s)": s["feature_seconds"]
                    if s["feature_seconds"] is not None
                    else "cache",
                }
                for s in stages
            ]
        )
        hist = pd.DataFrame([row for row in r["history"] if row["epoch"] > 0])[
            [
                "stage",
                "epoch",
                "train_loss",
                "val_loss",
                "val_average_precision",
                "val_precision_r095",
                "val_fpr_r095",
                "mean_val_prob",
                "seconds",
            ]
        ]
        m = r["metrics"]
        area = pd.DataFrame(m["recall_by_defect_area_at_r095"])
        d05 = m["at_default_threshold"]
        cmp_ = r["comparison_vs_phase2"]
        sections.append(f"""## {exp_id}

Recette : étapes {" → ".join(r["spec"]["stages"])}, `pos_weight` = {r["spec"]["pos_weight"]} ({_fmt(r["spec"]["pos_weight_value"], 2)}),
sampler `{r["spec"]["sampler"]}`, pooling `{r["spec"]["pooling"]}`, batch {r["spec"]["batch_size"]}, workers {r["spec"]["num_workers"]},
biais initial de la tête {_fmt(r["model"].get("head_bias_init"), 3)} = log(n_pos/n_neg) = log({r["data"]["train"]["n_pos"]}/{r["data"]["train"]["n_neg"]}).
Run `{r["run"]["run_id"]}` — commit `{r["provenance"]["git_commit"]}`, config `{r["provenance"]["config_hash"]}`, torch {r["provenance"].get("torch")},
durée totale {_fmt(r["timing"]["total_seconds"] / 60, 1)} min, `test_used = {str(r["data"]["test_used"]).lower()}`.

{_md_table(stage_rows)}

Évolution par époque :

{_md_table(hist, 4)}

**Métriques finales (validation)** : AP {_fmt(m["average_precision"])} [{_fmt(m["average_precision_ci"]["low"])} ; {_fmt(m["average_precision_ci"]["high"])}],
ROC-AUC {_fmt(m["roc_auc"])} [{_fmt(m["roc_auc_ci"]["low"])} ; {_fmt(m["roc_auc_ci"]["high"])}], perte val {_fmt(m["val_loss"], 4)},
probabilité moyenne prédite {_fmt(hist.iloc[-1]["mean_val_prob"])} (prévalence {_fmt(m["prevalence"])}).

{_md_table(_tradeoff(r))}

Au seuil 0,5 (rapporté, non retenu) : précision {_fmt(d05["precision"])}, rappel {_fmt(d05["recall"])}, F1 {_fmt(d05["f1"])},
TP {d05["tp"]} / FP {d05["fp"]} / FN {d05["fn"]} / TN {d05["tn"]}.

Rappel par taille de défaut au seuil « rappel ≥ 0,95 » :

{_md_table(area)}

{_winner_paragraph(winner_comparison(cfg, r), n_val_pos)}

Comparaison appariée avec `{cmp_["reference"]}` ({cmp_["n_effective"]} rééchantillonnages) : ΔAP {_fmt(cmp_["delta_average_precision"]["point"])}
[{_fmt(cmp_["delta_average_precision"]["low"])} ; {_fmt(cmp_["delta_average_precision"]["high"])}] ; Δprécision@0,95 {_fmt(cmp_["delta_precision_at_recall_0.95"]["point"])}
[{_fmt(cmp_["delta_precision_at_recall_0.95"]["low"])} ; {_fmt(cmp_["delta_precision_at_recall_0.95"]["high"])}] ; ΔFPR@0,95 {_fmt(cmp_["delta_fpr_at_recall_0.95"]["point"])}
[{_fmt(cmp_["delta_fpr_at_recall_0.95"]["low"])} ; {_fmt(cmp_["delta_fpr_at_recall_0.95"]["high"])}]. Un IC de Δ contenant 0 = non départageable avec {n_val_pos} positifs.
""")
    path = out_dir / "phase3_report.md"
    path.write_text("\n".join(sections), encoding="utf-8", newline="\n")
    (out_dir / "phase3_report.meta.json").write_text(
        json.dumps(
            provenance(cfg, experiments=list(results), n=len(summary)), indent=2, ensure_ascii=False
        ),
        encoding="utf-8",
        newline="\n",
    )
    log_event(logger, "report.written", path=str(path), n_experiments=len(results))
    return path
