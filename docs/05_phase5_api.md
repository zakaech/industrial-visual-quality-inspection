# Phase 5 — API d'inférence FastAPI

**État : livrée le 17/09/2026.** Ce document décrit l'architecture du service, ses endpoints, ses formats,
sa gestion des erreurs et la vérification réelle effectuée (serveur démarré, requêtes sur des images de
**validation**). Chaque chiffre cité provient d'une exécution réelle de cette session (journaux
`logs/api_serve.log`, sortie des tests) ; aucun chiffre n'est estimé.

## 5.1 Objectif

Exposer le modèle final de la Phase 4 (X6, bundle gelé `models/final/`) derrière une API HTTP minimale,
**sans rien réimplémenter** : le service appelle `Predictor.from_bundle()`, l'objet exact qui a produit les
métriques du test officiel. Le prétraitement, le modèle, les poids et le seuil `0,018752` sont inchangés ;
le test officiel n'a été utilisé ni pour développer ni pour démontrer l'API.

## 5.2 Architecture

```
client ──multipart/form-data (file)──▶ POST /predict
                                          │  validation d'entrée (type MIME, taille, vide)
                                          ▼
                              Predictor.from_bundle()   ← chargé UNE fois au démarrage (lifespan)
                                          │  bytes → cv2.imdecode → RGB → resize 640×224 (padding médiane)
                                          │  → normalisation ImageNet → ResNet-18 → sigmoïde
                                          ▼
                              score brut ≥ 0,018752 ? REJECT : ACCEPT
                                          │
                                          ▼
                              JSON {image_id, probability, decision, threshold, model_version, latency_ms, request_id}
```

| Fichier | Rôle |
|---|---|
| `src/vqi/api/app.py` | fabrique `create_app(cfg, bundle_dir)` : lifespan (chargement unique), middleware `X-Request-ID`, handlers d'erreur, point d'entrée ASGI `app` |
| `src/vqi/api/routes.py` | les trois endpoints ; validation d'entrée ; délégation au `Predictor` |
| `src/vqi/api/schemas.py` | modèles Pydantic v2 des réponses (documentés dans l'OpenAPI) |
| `scripts/serve_api.py` | lancement local via uvicorn, hôte/port lus dans la configuration |
| `tests/test_api.py` | 14 tests (bundle synthétique + contrat sur le bundle réel avec une image de validation) |

Décisions :

- **Un seul chemin d'inférence.** L'API ne connaît ni OpenCV ni PyTorch : elle passe des octets au
  `Predictor`, qui applique `Preprocessor` puis le modèle exactement comme en Phase 4. C'est ce qui rend
  vérifiable l'égalité API = `Predictor` (§ 5.8).
- **Chargement au démarrage, pas à la requête.** Le bundle est lu une fois dans le `lifespan` de l'application
  (0,93 s mesuré), stocké dans `app.state.predictor` ; les requêtes ne font que le lire. Un compteur
  `app.state.load_count` rend ce point testable.
- **Le service est configuré par le YAML existant** (section `api` : hôte, port, `max_upload_mb`,
  `allowed_content_types`, `decision_labels`, `model_version`) avec surcharge par variables d'environnement
  `VQI_CONFIG` (chemin de configuration) et `VQI_BUNDLE_DIR` (répertoire du bundle). Aucune valeur n'est codée
  en dur dans le service.
- **`model_version` = `<experiment_id>-<sha256 du model.pt>[:12]`** (`X6_C_winner-4233a9cd8053`), dérivée
  du bundle (`api.model_version: from_bundle`), ce qui lie chaque réponse au checkpoint exact.

## 5.3 Modèle servi

ResNet-18 ImageNet, `layer3` + `layer4` + tête fine-tunés (X6, étape C, époque 5), `model.pt` SHA-256
`4233a9cd80534b8efd20cb9abccb772911c9a7c8eb1cbba24fd27845b14e0cbb`, seuil gelé `0,018752005540198832`
(politique `min_recall` 0,95, marge `midpoint_logit`, choisi sur la validation, scores bruts sans calibration —
ECE 0,022 en Phase 4). Résultats de référence, rappelés par `/model-info` : validation AP 0,941 ; test officiel
(évaluation unique, Phase 4) AP 0,948, rappel 0,955, précision 0,536, FPR 0,102.

**Le champ `probability` est le score brut du modèle** (sortie sigmoïde), pas une probabilité calibrée : la
Phase 4 a montré que la calibration post-hoc n'apportait rien de significatif, et le score est en pratique très
proche de 0 pour la plupart des pièces saines (seuil 0,019). Le nom est conservé pour la lisibilité de l'API ;
la sémantique est documentée dans le schéma OpenAPI et ici.

## 5.4 Endpoints

| Méthode | Chemin | Rôle | Codes |
|---|---|---|---|
| GET | `/health` | sonde de vie (aucune inférence) | 200 |
| GET | `/model-info` | métadonnées gelées : version, SHA-256, prétraitement, seuil et sa règle, résumés validation / test | 200, 503 |
| POST | `/predict` | une image (`multipart/form-data`, champ `file`) → score + décision | 200, 400, 413, 415, 422, 503, 500 |
| GET | `/docs`, `/openapi.json` | documentation Swagger / schéma OpenAPI | 200 |

Chaque réponse porte l'en-tête `X-Request-ID` (repris de la requête s'il est fourni, sinon UUID4) ; le même
identifiant figure dans les journaux et dans le corps de `/predict`.

### `GET /health`

```json
{"status": "ok"}
```

### `GET /model-info` (extrait de la réponse réelle)

```json
{
  "model_version": "X6_C_winner-4233a9cd8053",
  "experiment_id": "X6_C_winner",
  "architecture": {"architecture": "resnet18", "pooling": "avg", "n_params": 11177025},
  "model_sha256": "4233a9cd80534b8efd20cb9abccb772911c9a7c8eb1cbba24fd27845b14e0cbb",
  "preprocessor": {"input_size": [640, 224], "resize_mode": "keep_aspect_pad", "pad_value": "median",
                   "interpolation": "area", "color_mode": "rgb", "variant": "none", "steps": [],
                   "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}},
  "threshold": {"value": 0.018752005540198832, "policy": "min_recall", "target_recall": 0.95,
                "margin": "midpoint_logit", "selected_on": "val (466 images, 49 defective)",
                "calibration": "raw", "frozen_at": "2026-09-17T15:26:49+00:00"},
  "decision_labels": {"positive": "REJECT", "negative": "ACCEPT"},
  "project_version": "0.1.0",
  "validation_summary": {"average_precision": 0.9408, "roc_auc": 0.9874,
                         "precision_at_recall_0.95": 0.5732, "fpr_at_recall_0.95": 0.0839},
  "final_test_summary": {"evaluated_at": "2026-09-17T16:09:09+00:00", "n": 1004, "n_pos": 110,
                         "average_precision": 0.9477, "roc_auc": 0.9788,
                         "at_threshold": {"recall": 0.9545, "precision": 0.5357, "fpr": 0.1018,
                                          "tp": 105, "fp": 91, "fn": 5, "tn": 803}}
}
```

(Valeurs arrondies ici à 4 décimales pour la lisibilité ; l'API renvoie les flottants complets.)

### `POST /predict`

Requête :

```bash
curl -X POST http://127.0.0.1:8000/predict \
     -F "file=@data/raw/kolektor_sdd2/train/10004.png;type=image/png"
```

Réponse réelle (image **de validation** `10004`, étiquette saine) :

```json
{"image_id": "10004.png", "probability": 0.016112767159938812, "decision": "ACCEPT",
 "threshold": 0.018752005540198832, "model_version": "X6_C_winner-4233a9cd8053",
 "latency_ms": 328.73, "request_id": "c5fceb66-e85d-4759-9ebd-617572627b83"}
```

`decision = REJECT` si et seulement si `probability >= threshold`. `latency_ms` mesure, côté serveur,
prétraitement + inférence (hors réception du fichier). `image_id` est le nom de fichier envoyé par le client.

## 5.5 Validation des entrées et codes HTTP

| Cas | Code | `detail` | Vérifié (réel) |
|---|---|---|---|
| champ `file` absent | 422 | `invalid request: expected a multipart field 'file'` | oui |
| type MIME hors `api.allowed_content_types` (png, jpeg, bmp) | 415 | `unsupported content type 'text/plain'; allowed: …` | oui |
| fichier vide | 400 | `empty file` | oui |
| fichier > `api.max_upload_mb` (10 Mo) | 413 | `file larger than 10485760 bytes` | oui (BMP de 11 Mo) |
| octets non décodables par `cv2.imdecode` | 422 | `the file could not be decoded as an image` | oui |
| modèle non chargé | 503 | `model not loaded` | test unitaire |
| exception interne | 500 | `internal error` (jamais de traceback dans la réponse) | test unitaire |

L'ordre des contrôles est : type MIME → lecture bornée (`limit + 1` octets) → vide → trop grand → décodage.
Le décodeur n'est donc jamais appelé sur un fichier hors limite. La vérification du type repose sur l'en-tête
`Content-Type` de la partie multipart, puis sur le décodage réel : un `text/plain` renommé en `image/png`
est rejeté en 422 par le décodeur.

## 5.6 Journalisation

Format JSON structuré existant (`vqi.logging_utils`), fichier `logs/vqi.log` + stdout. Événements :
`startup.model_loaded` (bundle, version, SHA-256 court, seuil, durée), `predict.done` (request_id, image_id,
taille en octets, type MIME, score arrondi, décision, latence), `predict.rejected` (request_id, raison),
`request.invalid`, `request.failed` (request_id, chemin, **nom de la classe d'exception seulement**),
`shutdown`. **Les octets de l'image ne sont jamais journalisés**, ni le traceback côté client.

Exemple réel :

```json
{"ts": "2026-09-17T18:00:53.505+00:00", "level": "INFO", "logger": "vqi.api", "msg": "startup.model_loaded",
 "bundle": "models\\final", "model_version": "X6_C_winner-4233a9cd8053", "model_sha256": "4233a9cd80534b8e",
 "threshold": 0.018752005540198832, "seconds": 0.93}
{"ts": "2026-09-17T18:01:16.290+00:00", "level": "INFO", "logger": "vqi.api", "msg": "predict.done",
 "request_id": "c5fceb66-e85d-4759-9ebd-617572627b83", "image_id": "10004.png", "bytes": 252277,
 "content_type": "image/png", "probability": 0.016113, "decision": "ACCEPT", "latency_ms": 328.7}
```

## 5.7 Lancement local

```bash
python scripts/serve_api.py                 # hôte/port depuis configs/default.yaml (0.0.0.0:8000)
# équivalent :
uvicorn vqi.api.app:app --host 0.0.0.0 --port 8000
# surcharges : VQI_CONFIG=chemin/config.yaml  VQI_BUNDLE_DIR=chemin/du/bundle  ou  --port / --bundle-dir
```

Swagger : <http://127.0.0.1:8000/docs> ; schéma : <http://127.0.0.1:8000/openapi.json>. Le démarrage échoue
explicitement (`startup.bundle_error`, puis exception) si le bundle est absent ou incohérent
(`preprocessor.json` différent du `Preprocessor` courant, seuil non gelé) : le service refuse de servir un
modèle qui ne serait pas celui de la Phase 4.

## 5.8 Vérification réelle (serveur démarré, images de validation)

Serveur lancé par `scripts/serve_api.py` (journal `logs/api_serve.log`, ignoré par git), un seul événement
`startup.model_loaded` sur toute la session (64 requêtes `/predict` journalisées : 58 servies, 6 rejetées), SHA-256 et seuil attendus. Comparaison entre
`Predictor.from_bundle(cfg).predict(path)` et `POST /predict` sur les mêmes octets, images de **validation**
uniquement (`split == "val"` dans `data/splits/v1.csv`) :

| Image (val) | Étiquette | Surface défaut | `Predictor` | API | \|Δ\| | Décision |
|---|---|---|---|---|---|---|
| 10004 | saine | — | 0,016112767160 | 0,016112767160 | 0,0 | ACCEPT / ACCEPT |
| 10045 | défectueuse | 23 px | 0,017811179161 | 0,017811179161 | 0,0 | ACCEPT / ACCEPT |
| 10952 | défectueuse | 14 727 px | 1,000000000000 | 1,000000000000 | 0,0 | REJECT / REJECT |

Écart nul (mêmes octets → même `imdecode` → même tenseur → même passe avant float32 sur CPU). Le test automatisé
utilise une tolérance de `1e-9`, justifiée par l'arrondi flottant du JSON et non par une variabilité du modèle.
Les trois scores sont en outre **identiques à ceux enregistrés en Phase 4** dans `reports/phase4/val_scores.csv`
(`10004` : 0,016112767159938812 ; `10045` : 0,017811179161071777 ; `10952` : 1,0). L'image `10045` (défaut de
23 px, score 0,0178 sous le seuil 0,0188) est l'un des deux faux négatifs de la validation relevés en Phase 4
(avec `11739`) : l'API reproduit fidèlement le comportement du modèle, y compris ses erreurs.

Latence côté serveur (`latency_ms`, 30 requêtes séquentielles, même image de validation, CPU i7-8665U,
sans GPU) : **médiane 255,7 ms, p95 273,2 ms, min 237,8 ms, max 366,7 ms** ; première requête après démarrage
328,7 ms. Temps de bout en bout mesuré par `curl` : 0,27–0,28 s. Note : un client `httpx` ouvrant une nouvelle
connexion par requête a mesuré ≈ 1,15 s de médiane sur la boucle locale Windows, surcoût de connexion côté client
(disparaît avec une connexion persistante : 272 ms) — il ne concerne pas le service.

Intégrité vérifiée après l'exercice : `models/final/model.pt` SHA-256 `4233a9cd80534b8e…` inchangé,
`threshold.json` inchangé (`0,018752`, `frozen: true`, `calibration_decision: raw`), aucun fichier de
`models/final/` ni de `reports/phase4/` modifié (`git status`).

## 5.9 Tests

`tests/test_api.py`, 14 tests, exécutés avec le reste de la suite : **99 tests passés** (85 antérieurs + 14),
`ruff check` et `ruff format --check` sans remarque.

- Bundle **synthétique** (poids ResNet-18 non entraînés, seuil 0,5, écrit dans un répertoire temporaire) :
  `/health`, `/model-info`, `/predict` valide (png / jpeg / bmp), champ absent (422), fichier vide (400),
  `text/plain` (415), image corrompue (422), fichier de 3 Mo avec limite 1 Mo (413, puis `/health` toujours
  vivant), erreur interne simulée (500 sans traceback, `X-Request-ID` présent), **modèle chargé une seule
  fois** (`load_count == 1`, même objet après 3 requêtes), égalité API = `Predictor` (tolérance `1e-9`),
  OpenAPI documentant les trois chemins et les codes d'erreur.
- Bundle **réel** (sauté si `models/final/model.pt` ou les données sont absents) : première image de
  validation de `v1.csv`, égalité API = `Predictor`, seuil `0,018752` et SHA-256 `4233a9cd80534b8e` exposés
  par `/model-info`. Aucune image du test officiel n'est lue.

## 5.10 Contraintes respectées

Pas de réentraînement, pas de X7, seuil / prétraitement / architecture / poids / calibration inchangés,
test officiel non utilisé, résultats de Phase 4 non modifiés (le seul fichier de configuration touché est
`api.model_version`, passé de `"TBD"` à `from_bundle`), pas de Docker ni de CI. Dépendance ajoutée :
`httpx` en extra `dev` uniquement (client de test FastAPI).

## 5.11 Limites connues et alternatives

- **Inférence synchrone dans la boucle d'événements.** `predict` est `async` (lecture du fichier) mais
  l'inférence (~250 ms) bloque le worker ; les requêtes sont donc servies une à la fois et `/health` attend
  pendant une inférence. Sur CPU, une inférence à la fois est le réglage le plus efficace (PyTorch occupe déjà
  tous les cœurs). Alternatives : `run_in_threadpool` + verrou, ou plusieurs workers uvicorn (chacun charge
  le modèle : mémoire ×N).
- **Limite de taille appliquée après réception.** Starlette lit le corps multipart avant l'endpoint ; le
  413 protège le décodeur et le modèle, pas la bande passante. En production, faire appliquer la limite par le
  reverse proxy (`client_max_body_size`).
- **Une image par requête**, pas de lot ; pas d'authentification, pas de TLS, pas de limitation de débit
  (hors périmètre d'un service de démonstration ; à placer derrière une passerelle).
- **Pas de détection de dérive** : le service ne vérifie pas que l'image ressemble au domaine KolektorSDD2
  (taille ≈ 230×630, fond métallique). Une image hors domaine reçoit un score sans avertissement ; un contrôle
  de plausibilité (dimensions, statistiques d'intensité) serait une extension naturelle.
- **Le score n'est pas calibré** (§ 5.3) et le seuil a été choisi pour un rappel ≥ 0,95 sur 49 défauts de
  validation : la précision attendue (~0,5) implique qu'environ une pièce rejetée sur deux est en réalité saine.
  C'est le compromis assumé depuis la Phase 0 (un défaut manqué coûte plus qu'un faux rejet).

## 5.12 Prochaine étape recommandée

Conteneurisation (Dockerfile CPU, image basée sur `requirements.txt`, `HEALTHCHECK` sur `/health`, bundle monté
en volume ou copié) puis intégration continue (lint + tests sur le bundle synthétique). Aucune de ces étapes
ne touche au modèle ni au seuil.
