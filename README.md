# Industrial Visual Quality Inspection

Pipeline de deep learning pour la détection automatique de défauts de surface sur des pièces industrielles — de la
donnée brute jusqu'à un service d'inférence conteneurisé et appelable.

**État : Phase 2 livrée (prétraitement OpenCV partagé entraînement/API, baseline classique) — aucun réseau de
neurones n'a encore été entraîné. Tout chiffre de résultat ci-dessous provient d'une exécution journalisée ou est
marqué TBD.**

| Phase | Contenu | Document |
|---|---|---|
| 0 | Cadrage, choix du dataset, métriques, risques | [docs/00_phase0_framing.md](docs/00_phase0_framing.md) |
| 1 | Ingestion, manifeste, doublons, découpage, EDA | [docs/01_phase1_ingestion.md](docs/01_phase1_ingestion.md) · [rapport EDA](reports/eda/eda_report.md) |
| 2 | Prétraitement OpenCV, comparaison des transformations (CNR), baseline SVM | [docs/02_phase2_preprocessing_baseline.md](docs/02_phase2_preprocessing_baseline.md) · [rapport Phase 2](reports/phase2/phase2_report.md) |
| 3 | Entraînement CNN (ResNet-18) | à venir |
| 4 | Évaluation, seuil, évaluation finale unique sur le test | à venir |
| 5 | API FastAPI + Docker | à venir |

## Problème

Étant donné l'image d'une pièce, décider **ACCEPT** ou **REJECT**. Les deux erreurs n'ont pas le même coût : un
défaut manqué (faux négatif) expédie une pièce défectueuse ; une fausse alerte (faux positif) ne déclenche qu'une
réinspection humaine. Chaque choix de métrique, de fonction de perte et de seuil dans ce projet est fait sous cette
asymétrie — voir [docs/00_phase0_framing.md](docs/00_phase0_framing.md) §4.

## Jeu de données

KolektorSDD2 (Božič, Tabernik & Skočaj, 2021) : 3 335 images réelles de ligne de production de commutateurs
électriques, ~11 % défectueuses, masques au pixel, découpage train/test officiel. Justification et alternatives
étudiées : [docs/00_phase0_framing.md](docs/00_phase0_framing.md) §1.

Les données ne sont pas versionnées. La reproductibilité repose sur la somme de contrôle de l'archive
(`dataset.archive_sha256`) et sur le manifeste de découpage versionné dans `data/splits/v1.csv`
(train 1 865 / val 466 / test 1 004 images, seed 42, stratifié et group-aware — invariants dans
`v1.meta.json`). Fait notable : l'archive officielle contient une copie accidentelle
(`train/10301 (copy).png`), détectée et exclue via le manifeste sans toucher aux fichiers.

## Pipeline

Archive brute → vérification d'intégrité → manifeste → recherche de doublons → découpage stratifié → prétraitement →
fine-tuning ResNet-18 → choix du seuil sur la validation → évaluation finale unique sur le test → export → FastAPI →
Docker. Schéma et contrats par étape : [docs/pipeline.md](docs/pipeline.md).

## Résultats

Baseline classique (Phase 2 — validation uniquement, 49 défectueuses ; IC bootstrap 95 %) :

| Modèle | AP val | IC 95 % AP | ROC-AUC val | Précision à rappel 0,95 | FPR à rappel 0,95 |
|---|---|---|---|---|---|
| contraste global seul (référence triviale) | 0,402 | [0,268 ; 0,550] | 0,704 | 0,106 | 0,950 |
| 82 descripteurs classiques + SVM (`E5_grid_edge_check`) | 0,862 | [0,770 ; 0,935] | 0,945 | 0,233 | 0,372 |

Aucun prétraitement OpenCV (égalisation, CLAHE, débruitage, rehaussement) n'améliore la baseline de façon
significative → `preprocessing.enhancement.active: none`. Détails : [rapport Phase 2](reports/phase2/phase2_report.md).

CNN (Phase 3–4) :

| Partition | Average Precision | Rappel au point de fonctionnement | Précision au point de fonctionnement | FPR | Run id |
|---|---|---|---|---|---|
| validation | TBD | TBD | TBD | TBD | TBD |
| test (évaluation unique) | TBD | TBD | TBD | TBD | TBD |

Point de fonctionnement : politique de seuil `min_recall`, rappel cible 0,95 sur la validation — TBD.

## Organisation du dépôt

```
.
├── configs/            # configuration YAML (default.yaml + experiments/)
├── data/               # raw/ interim/ processed/ ignorés par git ; splits/ versionné
├── docs/               # cadrage, pipeline, une note par phase
├── notebooks/          # exploration uniquement ; rien dans le pipeline n'en dépend
├── reports/            # eda/, phase2/ : rapports Markdown générés + figures + résultats JSON/CSV
├── scripts/            # points d'entrée CLI (une action par script)
├── src/vqi/            # le package (vqi = visual quality inspection)
│   ├── data/           # téléchargement, intégrité, manifeste, doublons, découpage, EDA
│   ├── preprocessing/  # UN module OpenCV partagé par l'entraînement et l'inférence (io, geometry, enhance, pipeline)
│   ├── baseline/       # descripteurs classiques (stats, histogramme, LBP, stats locales) + SVM/RF + rapport
│   ├── evaluation/     # métriques, seuils, bootstrap — indépendants du modèle
│   ├── models/         # (Phase 3) fabrique de backbones, têtes
│   ├── training/       # (Phase 3) boucle, pertes, callbacks, checkpoints
│   ├── inference/      # (Phase 5) Predictor : charger le bundle → prétraiter → scorer → décider
│   └── api/            # (Phase 5) application FastAPI
├── tests/              # 58 tests, dataset synthétique, sans données réelles
├── runs/               # sorties d'entraînement, ignorées par git
└── Dockerfile          # (Phase 5)
```

## Stack

Python 3.11 · PyTorch / torchvision · OpenCV · scikit-learn · NumPy / pandas · FastAPI / Uvicorn · Docker
(+ matplotlib pour les figures). Machine cible sans GPU CUDA (tout tourne sur CPU).

## Démarrage

```bash
py -3.11 -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt && pip install -e ".[dev]"
python scripts/download_data.py           # télécharge (853 Mo), vérifie le SHA-256, extrait
python scripts/build_manifest.py          # manifeste + analyse des doublons
python scripts/make_splits.py             # partition train/val/test → data/splits/v1.csv
python scripts/run_eda.py                 # rapport + figures → reports/eda/
python scripts/compare_preprocessing.py   # Phase 2 : figures + CNR → reports/phase2/
python scripts/run_baseline.py            # Phase 2 : 16 expériences SVM/RF → reports/phase2/experiments.csv
python scripts/build_phase2_report.py     # Phase 2 : rapport Markdown
pytest
```

## Reproductibilité

Graine `42` partout (`reproducibility.seed` dans [configs/default.yaml](configs/default.yaml)), algorithmes torch
déterministes quand ils existent, workers de DataLoader seedés, configuration figée et hash git enregistrés avec
chaque artefact (`*.meta.json`, un JSON par expérience). Le non-déterminisme résiduel sur CPU est documenté dans le
cadrage (risque R9).
