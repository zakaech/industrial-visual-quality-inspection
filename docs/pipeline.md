# Pipeline de bout en bout — de l'image brute à la réponse de l'API

Une règle gouverne tout le schéma : **le code de prétraitement utilisé à l'inférence est le même objet que celui
utilisé à l'entraînement** (`src/vqi/preprocessing`). Tout le reste est une dérive entraînement/serving.

```mermaid
flowchart TB
    subgraph S0["0 · Acquisition — src/vqi/data"]
        A[("Archive KSDD2<br/>(téléchargement manuel)")] --> B{"SHA-256 == config ?"}
        B -- non --> B0["arrêt : fichier corrompu / incorrect"]
        B -- oui --> C["extraction → data/raw/"]
        C --> D["manifest.csv<br/>image_path · mask_path · label (masque non vide)<br/>official_split · sha256 · phash"]
        D --> D1{"effectifs == expected_counts ?"}
        D1 -- non --> B0
    end

    subgraph S1["1 · Découpage — src/vqi/data"]
        D1 -- oui --> E["recherche de doublons<br/>SHA-256 exacts · quasi-doublons pHash"]
        E --> F["découpage stratifié train/val<br/>seed 42 · par groupe · test officiel en quarantaine"]
        F --> G[("data/splits/v1.csv<br/>(versionné dans git)")]
    end

    subgraph S2["2 · Prétraitement — src/vqi/preprocessing"]
        G --> H["cv2.imread → BGR→RGB<br/>redimensionnement (ratio conservé) → float32 → normalisation ImageNet"]
        H --> H1["augmentation<br/>(split train uniquement, préserve le label)"]
    end

    subgraph S3["3 · Entraînement — src/vqi/models · src/vqi/training"]
        H1 --> I["ResNet-18 (ImageNet) + tête à 1 logit"]
        I --> J["BCEWithLogits · AdamW · LR cosinus<br/>arrêt anticipé sur l'Average Precision de validation"]
        J --> K[("runs/<run_id>/model.pt<br/>config.yaml · metrics.jsonl · hash git")]
    end

    subgraph S4["4 · Évaluation — src/vqi/evaluation"]
        K --> L["prédictions sur la validation"]
        L --> M["politique de seuil<br/>min_recall / expected_cost"]
        M --> N[("threshold.json")]
        L --> O["courbe PR · calibration · rappel par taille de défaut · Grad-CAM"]
        N --> P["Évaluation FINALE sur le test — une seule fois"]
        P --> Q[("report/ (métriques + IC + figures)")]
    end

    subgraph S5["5 · Serving — src/vqi/inference · src/vqi/api"]
        K --> R["bundle gelé models/final : model.pt + preprocessor.json + threshold.json + model_card.json"]
        N --> R
        R --> S["Predictor.from_bundle() au démarrage (une fois)"]
        T["POST /predict (octets de l'image)"] --> U["validation du content-type et de la taille"]
        U --> V["MÊME prétraitement (S2)"]
        V --> W["forward → sigmoïde → p"]
        W --> X{"p ≥ seuil ?"}
        X -- oui --> Y["décision : REJECT"]
        X -- non --> Z["décision : ACCEPT"]
        Y & Z --> AA["JSON : defect_probability · decision · threshold<br/>model_version · latency_ms"]
        S --> V
    end

    subgraph S6["6 · Packaging"]
        AA --> AB["Image Docker (python:3.11-slim, wheels torch CPU)<br/>uvicorn · GET /health · GET /model-info"]
    end
```

## Contrats par étape

| Étape | Entrée | Sortie (artefact) | Module | Déterministe ? |
|---|---|---|---|---|
| 0 Acquisition | archive + `dataset.archive_sha256` | `data/raw/`, `data/interim/manifest.csv` | `vqi.data.download`, `vqi.data.manifest` | oui (conditionné par la somme de contrôle) |
| 1 Découpage | manifeste, `split.*`, `reproducibility.seed` | `data/splits/v1.csv` (versionné) | `vqi.data.splits` | oui (seedé) |
| 2 Prétraitement | chemin d'image ou octets, `preprocessing.*` | `torch.Tensor` [C, H, W] | `vqi.preprocessing.transforms` | oui (augmentation seedée, train uniquement) |
| 3 Entraînement | csv de split, `model.*`, `training.*` | `runs/<id>/model.pt`, `metrics.jsonl`, `config.yaml` figée | `vqi.models.factory`, `vqi.training.loop` | au mieux sur CPU (voir risque R9) |
| 4 Évaluation | checkpoint, csv de split, `evaluation.*` | `threshold.json`, `report/` | `vqi.evaluation.*` | oui (bootstrap seedé) |
| 5 Serving | bundle exporté, `api.*` | JSON HTTP | `vqi.inference.predictor`, `vqi.api.app` | oui |
| 6 Packaging | dépôt + bundle exporté | image Docker | `Dockerfile` | oui (dépendances épinglées) |

## Contrat de l'API (v1, tel que livré en Phase 5)

| Endpoint | Requête | Réponse |
|---|---|---|
| `GET /health` | — | `{"status": "ok"}` |
| `GET /model-info` | — | `{"model_version", "model", "experiment_id", "architecture", "model_sha256", "preprocessor", "threshold": {value, policy, target_recall, margin, selected_on, calibration, frozen_at}, "decision_labels", "project_version", "bundle_frozen_at", "validation_summary", "final_test_summary"}` |
| `POST /predict` | `multipart/form-data`, champ `file` (png/jpeg/bmp, ≤ `api.max_upload_mb`) | `{"image_id": str, "probability": float, "decision": "ACCEPT" \| "REJECT", "threshold": float, "model_version": str, "latency_ms": float, "request_id": str}` |

Cas d'erreur : fichier vide → 400 ; fichier trop volumineux → 413 ; content-type non supporté → 415 ; champ absent
ou image non décodable → 422 ; modèle non chargé → 503 ; erreur interne → 500 sans traceback. Écarts par rapport
au contrat esquissé en Phase 0 : le champ `defect_probability` s'appelle `probability` (score brut, non calibré),
`image_id` et `request_id` ont été ajoutés pour la traçabilité, `/model-info` expose l'intégralité des
métadonnées du bundle gelé (`models/final/`) plutôt qu'un résumé, et le bundle est un `state_dict` PyTorch
(`model.pt`) et non un export TorchScript (décision de Phase 4). Détails : `docs/05_phase5_api.md`.

## Où vit chaque garde-fou anti-fuite

| Garde-fou | Emplacement |
|---|---|
| Somme de contrôle + effectifs attendus | étape 0, échoue bruyamment |
| Doublons exacts / quasi-doublons dans le même split | étape 1 |
| Jeu de test intouché jusqu'à la fin | étape 4 : le script de test exige un `threshold.json` existant produit depuis la validation |
| Masques jamais vus par le classifieur | le `Dataset` de l'étape 2 ne lit que les images ; les masques ne sont lus qu'à l'étape 4 |
| Prétraitement identique entraînement / serving | l'étape 2 est importée par les étapes 3 et 5 ; un test de contrat compare les tenseurs |
