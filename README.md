# Industrial Visual Quality Inspection

Pipeline de deep learning pour la détection automatique de défauts de surface sur des pièces industrielles — de la
donnée brute jusqu'à un service d'inférence conteneurisé et appelable.

**État : Phase 0 (cadrage) — aucun modèle n'a encore été entraîné. Tout chiffre ci-dessous est un emplacement à
remplir tant qu'il ne provient pas d'une exécution journalisée.**

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
├── docs/               # cadrage, pipeline, journal des décisions
├── notebooks/          # exploration uniquement (EDA) ; rien dans le pipeline n'en dépend
├── scripts/            # points d'entrée CLI : download, make_splits, train, evaluate, export, serve
├── src/vqi/            # le package (vqi = visual quality inspection)
│   ├── data/           # téléchargement, intégrité, manifeste, doublons, découpage, Dataset
│   ├── preprocessing/  # transformations OpenCV partagées par l'entraînement et l'inférence
│   ├── models/         # fabrique de backbones, têtes de classification
│   ├── training/       # boucle, pertes, callbacks, checkpoints
│   ├── evaluation/     # métriques, politiques de seuil, calibration, figures, analyse d'erreurs
│   ├── inference/      # Predictor : charger le bundle → prétraiter → scorer → décider
│   └── api/            # application FastAPI, schémas, routes
├── tests/
├── runs/               # sorties d'expériences, ignorées par git
└── Dockerfile
```

## Stack

Python 3.11 · PyTorch / torchvision · OpenCV · scikit-learn · NumPy / pandas · FastAPI / Uvicorn · Docker.
Machine cible sans GPU CUDA (entraînement sur CPU).

## Démarrage

```bash
py -3.11 -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt && pip install -e ".[dev]"
python scripts/download_data.py      # télécharge (853 Mo), vérifie le SHA-256, extrait
python scripts/build_manifest.py     # manifeste + analyse des doublons
python scripts/make_splits.py        # partition train/val/test → data/splits/v1.csv
python scripts/run_eda.py            # rapport + figures → reports/eda/
pytest                               # 18 tests, dataset synthétique, sans données réelles
```

## Reproductibilité

Graine `42` partout (`reproducibility.seed` dans [configs/default.yaml](configs/default.yaml)), algorithmes torch
déterministes quand ils existent, workers de DataLoader seedés, configuration figée et hash git enregistrés avec
chaque run. Le non-déterminisme résiduel sur CPU est documenté dans le cadrage (risque R9).
