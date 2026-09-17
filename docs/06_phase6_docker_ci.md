# Phase 6 — Dockerisation et intégration continue

**État : livrée le 17/09/2026.** Cette phase n'apporte aucune amélioration ML : le modèle, le seuil, le
prétraitement et les résultats de Phase 4 sont inchangés. Elle rend le service d'inférence (Phase 5)
reproductible et vérifiable : une image Docker CPU, non-root, contenant le bundle gelé, et une CI GitHub
Actions minimale (lint, tests, build Docker + smoke test). Chaque chiffre cité provient d'une exécution
réelle de cette session (journaux `logs/docker_build_*.log`, `logs/ci_tests_linux.log`, sorties `docker`).

## 6.1 Objectif

Industrialiser le service sans le complexifier : une commande pour construire, une pour lancer, et une
CI qui prouve à chaque commit que le dépôt reste lint-propre, testé, et constructible en image.

## 6.2 Architecture

```
GitHub Actions ── lint ── tests ── docker build ── smoke test (bundle synthétique)
                                        │
                                        ▼
                    image vqi-api  (python:3.11.9-slim-bookworm, CPU, utilisateur `app`)
                                        │
                              uvicorn vqi.api.app:app :8000
                                        │
                              Predictor.from_bundle()  ← chargé une fois au démarrage
                                        │
                     /app/models/final : model.pt (SHA-256 4233a9cd…), preprocessor.json,
                                        threshold.json (0,018752, gelé), model_card.json
```

| Fichier | Rôle |
|---|---|
| `Dockerfile` | image de service : dépendances épinglées, package installé, bundle copié, utilisateur non-root, `HEALTHCHECK`, commande uvicorn |
| `.dockerignore` | liste blanche : seuls `requirements-api.txt`, `pyproject.toml`, `README.md`, `src/`, `configs/default.yaml`, `models/final/` entrent dans le contexte |
| `requirements-api.txt` | dépendances d'exécution de l'API uniquement (sous-ensemble de `requirements.txt`, mêmes versions) |
| `compose.yaml` | un service `api`, port 8000 : `docker compose up --build` |
| `scripts/docker_smoke_test.sh` | test de fumée d'une image (mode `synthetic` pour la CI, `real` en local) |
| `src/vqi/inference/synthetic_bundle.py` | bundle **synthétique** (poids non entraînés, seuil 0,5) pour les tests et la CI |
| `.github/workflows/ci.yml` | jobs `lint`, `tests`, `docker` |

## 6.3 Stratégie du bundle : option A (bundle intégré à l'image)

**Choix : le bundle gelé `models/final/` est copié dans l'image.** Justification :

- l'image est autonome : `docker run` suffit, aucun volume à préparer, aucun téléchargement à l'exécution
  (le service ne télécharge jamais rien : ni le modèle, ni les poids ImageNet — voir § 6.9) ;
- reproductibilité forte : la version servie est liée à l'image ; `/model-info` expose le SHA-256 du
  `model.pt` réellement chargé, comparable à la valeur gelée en Phase 4 ;
- le surcoût est de 44,8 Mo sur une image de 1,86 Go (dominée par PyTorch CPU) — négligeable.

Contrainte réelle du dépôt et réponse : `model.pt` est ignoré par git (`.gitignore`), donc **un clone nu ne
peut pas construire l'image autonome**. Le `Dockerfile` échoue alors volontairement avec un message clair,
sauf si `--build-arg REQUIRE_MODEL=0` est passé : la CI construit ainsi une image « applicative » sans
`model.pt` et la teste avec un bundle synthétique monté en volume et sélectionné par la variable
`VQI_BUNDLE_DIR` (le mécanisme de la Phase 5). Cette même variable permet l'option B (bundle monté depuis
l'hôte) sans modifier l'image — documentée dans `compose.yaml`.

Ce que Docker **ne** redéfinit **pas** : aucun seuil, aucune taille d'entrée, aucun paramètre ML n'apparaît
dans le `Dockerfile` ou `compose.yaml` ; tout est lu dans `models/final/*.json` et `configs/default.yaml`.

## 6.4 Build

```bash
docker build -t vqi-api .
```

Mesures réelles (Docker Desktop 29.7.2, moteur linux/amd64, 8 CPU alloués) :

| Mesure | Valeur |
|---|---|
| premier build (cache vide : image de base + roues PyPI / index CPU PyTorch) | 439 s |
| rebuild après modification du package (couches de dépendances en cache) | 20 s |
| taille de l'image (`docker image ls`) | 1,86 Go |
| dont couche des dépendances Python | 1,23 Go (torch CPU 773 Mo, OpenCV headless 72 Mo, NumPy 45 Mo) |
| dont bundle `models/final/` | 44,8 Mo |
| Python dans l'image | 3.11.9 |
| dépendances principales | torch 2.14.0+cpu, torchvision 0.29.0+cpu, opencv-python-headless 5.0.0.93, numpy 2.4.6, fastapi 0.141.1, starlette 1.6.0, pydantic 2.13.5, uvicorn 0.53.0, python-multipart 0.0.32, PyYAML 6.0.3 |
| `pip check` dans l'image | « No broken requirements found » |

Pas de build multi-étapes : les dépendances sont des roues binaires (aucune compilation), l'image de base
`slim` est déjà minimale, et la taille est dictée par `libtorch_cpu`. Un multi-stage n'aurait retiré que
quelques mégaoctets au prix d'une complexité artificielle. Pas d'optimisation prématurée de la taille.

Choix d'installation : `pip install --no-deps .` installe le package `vqi` après les dépendances épinglées.
Pour que `pip check` soit propre, `pyproject.toml` distingue désormais les dépendances d'exécution de l'API
(`dependencies`) des outils d'entraînement/analyse (`[train]` : scikit-learn, pandas, matplotlib) et de
développement (`[dev]`) ; l'installation locale devient `pip install -e ".[train,dev]"`. Le chemin d'import
de l'API a été tracé (`sys.modules`) : il n'utilise ni scikit-learn, ni pandas, ni matplotlib.

## 6.5 Run

```bash
docker run --rm -p 8000:8000 vqi-api            # ou : docker compose up --build
curl http://127.0.0.1:8000/health               # {"status":"ok"}
```

Démarrage réel : `startup.model_loaded` en **0,79 s** dans le conteneur (`model_version`
`X6_C_winner-4233a9cd8053`, seuil `0,018752005540198832`). Surcharges possibles : `-e VQI_BUNDLE_DIR=/bundle
-v /chemin/bundle:/bundle:ro` (autre bundle), `-e VQI_CONFIG=...` (autre configuration).

## 6.6 Endpoints (vérifiés sur le conteneur réel)

| Endpoint | Résultat réel |
|---|---|
| `GET /health` | 200 `{"status":"ok"}`, en-tête `X-Request-ID` |
| `GET /model-info` | 200 ; `model_version` `X6_C_winner-4233a9cd8053`, `model_sha256` `4233a9cd80534b8efd20cb9abccb772911c9a7c8eb1cbba24fd27845b14e0cbb`, seuil `0,018752005540198832` (`min_recall` 0,95, `raw`), prétraitement `none` 640×224, `final_test_summary` présent |
| `POST /predict` | voir § 6.10 ; codes d'erreur inchangés (422 champ absent, 415 `text/plain`, 400 vide, 422 image corrompue) |

Latence côté serveur dans le conteneur (20 requêtes séquentielles, image de validation `10004`, hôte au
repos) : médiane **118,6 ms** (p95 145,1, min 89,6, max 152,0) à la première mesure et **133,7 ms** (p95 169,3)
sur l'image finale ; jusqu'à ~260–360 ms quand l'hôte est chargé (mesuré juste après la suite de tests).
Plus rapide que le même modèle sous Windows en Phase 5 (médiane 255,7 ms) : PyTorch Linux/oneDNN et 8 cœurs
alloués à la VM Docker. Mesure locale, non généralisable.

## 6.7 Healthcheck

```dockerfile
HEALTHCHECK --interval=30s --timeout=3s --start-period=30s --retries=3 \
    CMD python -c "... urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2) ..."
```

Léger (aucune inférence, aucune dépendance supplémentaire : `urllib` de la bibliothèque standard). Sémantique
vérifiée : uvicorn ne sert aucune requête tant que le `lifespan` n'a pas chargé le bundle, et un échec de
chargement (`BundleError`) fait échouer le démarrage — donc `200` sur `/health` signifie « processus vivant
**et** modèle chargé », pas seulement « processus vivant ». Pour rendre cette sémantique explicite, `/health`
renvoie désormais **503** si le prédicteur est absent (cas testé : `test_health_is_503_without_a_loaded_model`).
État observé : `docker inspect … .State.Health.Status` = `healthy` en local et dans le smoke test.

## 6.8 Utilisateur du conteneur

`useradd --uid 10001 --user-group --no-create-home --shell /usr/sbin/nologin app`, puis `USER app`. Vérifié :
`docker exec … id` → `uid=10001(app) gid=999(app)`. Le seul répertoire inscriptible par le service est
`/app/logs` (fichier `logging.file` de la configuration) ; le reste de `/app` appartient à root en lecture
seule pour `app`. Aucun mécanisme de capacités ou de sandbox supplémentaire (hors périmètre).

## 6.9 Correctif nécessaire à la conteneurisation (hors ML)

En Phase 5, `Predictor.from_bundle()` construisait le squelette ResNet-18 via `build_model(cfg)` avec
`model.pretrained: true` : torchvision cherchait alors les poids ImageNet dans le cache local
(`~/.cache/torch/hub/checkpoints`) — présents sur la machine de développement, **absents dans un conteneur**
— et aurait tenté un téléchargement au démarrage, juste avant d'écraser tous les poids par ceux de
`model.pt`. Correctif : `build_model(..., pretrained=False)` dans `from_bundle` et `from_checkpoint`
(paramètre optionnel, `None` = comportement de la configuration pour l'entraînement, inchangé). Le squelette
reçoit ensuite **tous** ses poids de `model.pt` (`load_state_dict(strict=True)`), donc le modèle servi est
strictement le même. Preuves :

- test `test_bundle_loading_never_requests_pretrained_weights` (torchvision est appelé avec `weights=None`) ;
- scores après correctif identiques bit à bit à `reports/phase4/val_scores.csv` (lecture CSV en
  `round_trip`) sur `10004`, `10045`, `10952`, `11739` ;
- le conteneur, sans cache ni réseau, démarre en 0,79 s.

## 6.10 Vérification du conteneur réel

Comparaison `Predictor.from_bundle()` local (Windows) ↔ `POST /predict` du conteneur (Linux), mêmes octets,
images de **validation** uniquement (jamais le test officiel) :

| Image (val) | Étiquette | `Predictor` local | Conteneur | \|Δ\| | Décision | Phase 4 (`val_scores.csv`) |
|---|---|---|---|---|---|---|
| 10004 | saine | 0,016112767159938812 | 0,016112767159938812 | 0,0 | ACCEPT / ACCEPT | 0,016112767159938812 |
| 10045 | défectueuse (23 px) | 0,017811179161071777 | 0,017811179161071777 | 0,0 | ACCEPT / ACCEPT | 0,017811179161071777 |
| 10952 | défectueuse (14 727 px) | 1,0 | 1,0 | 0,0 | REJECT / REJECT | 1,0 |

Écart maximal **0,0** (tolérance de la Phase 5 : `1e-9`). Le résultat est remarquable mais pas garanti en
général : deux plateformes (Windows / Linux, noyaux BLAS différents) peuvent différer de quelques ulp ; ici
elles coïncident. La tolérance `1e-9` reste la référence du test automatisé.

SHA-256 dans le conteneur : `sha256sum /app/models/final/model.pt` →
`4233a9cd80534b8efd20cb9abccb772911c9a7c8eb1cbba24fd27845b14e0cbb`, identique à la valeur enregistrée dans
`models/final/model_card.json` et exposée par `/model-info`. Le smoke test en mode `real` compare
automatiquement le SHA-256 et le seuil renvoyés par `/model-info` à ceux des fichiers de l'hôte.

## 6.11 CI GitHub Actions (`.github/workflows/ci.yml`)

Déclencheurs : `push` sur `main` et `pull_request`. Trois jobs indépendants sur `ubuntu-24.04`, sans
secret, sans GPU, sans données :

| Job | Étapes | Réseau requis |
|---|---|---|
| `lint` | Python 3.11, `pip install ruff==0.16.7`, `ruff check .`, `ruff format --check .` | PyPI |
| `tests` | Python 3.11, `pip install -r requirements.txt` (torch CPU via l'index PyTorch), `pip install --no-deps -e .`, `pytest -rs` | PyPI, index CPU PyTorch, et **download.pytorch.org pour le backbone ImageNet public** (`resnet18-f37072fd.pth`, contrôlé par hachage par torchvision, mis en cache par `actions/cache`) — utilisé par les tests d'entraînement, pas par le service |
| `docker` | `docker build --build-arg REQUIRE_MODEL=0 -t vqi-api:ci .`, puis `scripts/docker_smoke_test.sh vqi-api:ci synthetic` | PyPI, index CPU PyTorch |

Le job `tests` tourne sur les données **synthétiques** de `tests/conftest.py` et le bundle synthétique ; le
seul test qui lit des données réelles (`test_api_matches_predictor_on_real_bundle_with_a_validation_image`)
est **sauté** quand `models/final/model.pt` ou `data/splits/v1.csv` sont absents, ce qui est le cas en CI.
Le dataset officiel n'est jamais téléchargé ; le test officiel n'est jamais lu.

Smoke test (§ `scripts/docker_smoke_test.sh`) : le bundle synthétique est généré **à l'intérieur de l'image**
(`python -m vqi.inference.synthetic_bundle --out /bundle`, aucun téléchargement), monté en lecture seule et
sélectionné par `VQI_BUNDLE_DIR` ; le script attend `/health` = 200, vérifie que `/model-info` renvoie une
version `SYN-…`, qu'un `POST /predict` sur `sample.png` renvoie une décision, qu'un `text/plain` donne 415,
puis que le `HEALTHCHECK` Docker passe à `healthy`. Il s'arrête à la première anomalie et affiche les
journaux du conteneur.

État de validation : **le dépôt n'a pas de dépôt distant GitHub configuré** (`git remote -v` vide), donc le
workflow n'a pas encore été exécuté par GitHub Actions. Validation locale effectuée à la place :

- syntaxe YAML du workflow et de `compose.yaml` chargée sans erreur (`docker compose config` valide) ;
- job `lint` : `ruff check .` et `ruff format --check .` propres en local ;
- job `tests` : la suite complète a été exécutée **sous Linux** dans un conteneur `python:3.11.9-slim-bookworm`
  avec exactement les commandes du workflow (dépôt monté) — résultat en § 6.13 ;
- job `docker` : build avec `REQUIRE_MODEL=0` et `docker_smoke_test.sh … synthetic` exécutés en local avec le
  même script — `OK` en 16 s (hors build).

La première exécution réelle sur GitHub reste à observer après création du dépôt distant.

## 6.12 Reproductibilité

| Élément | Valeur figée |
|---|---|
| commit du code | voir `git log -1` (commit de Phase 6) ; le service expose `project_version` 0.1.0 |
| image de base | `python:3.11.9-slim-bookworm` (Python 3.11.9) |
| dépendances | `requirements-api.txt` (épinglées, roues CPU), sous-ensemble de `requirements.txt` |
| modèle | `models/final/model.pt`, SHA-256 `4233a9cd80534b8efd20cb9abccb772911c9a7c8eb1cbba24fd27845b14e0cbb` |
| seuil | `0,018752005540198832` (`threshold.json`, `frozen: true`, `raw`) |
| prétraitement | `preprocessor.json` (`none`, 640×224, padding médiane, ImageNet) — vérifié égal à la configuration au chargement |
| configuration | `configs/default.yaml` copié dans l'image ; `api.*` et `inference.bundle_dir` |

Ce qui **n'est pas** promis : une image bit à bit identique entre deux builds (horodatages des couches,
`.pyc` compilés à l'installation, image de base non épinglée par digest). Ce qui est vérifié : deux builds
successifs, dont un `--no-cache`, produisent le même contenu applicatif — mêmes versions (`pip freeze`), même
SHA-256 de `model.pt`, mêmes fichiers `vqi` — voir § 6.13.

## 6.13 Résultats des vérifications finales

| Vérification | Résultat réel |
|---|---|
| `ruff check .` / `ruff format --check .` (racine du dépôt) | propres (94 fichiers formatés) |
| `pytest` local (Windows, `.venv`) | **102 passés**, 0 échec, 0 sauté, 209,6 s (3 nouveaux tests : chargement sans poids ImageNet, carte du bundle synthétique, `/health` → 503 sans modèle) |
| `pytest` sous Linux (`python:3.11.9-slim-bookworm`, commandes du job `tests`, dépôt monté) | **102 passés**, 154,1 s — après un correctif (ci-dessous) |
| build `--no-cache` complet | 669 s (téléchargements inclus), succès |
| deux builds successifs (cache / `--no-cache`) | identifiants d'image différents (`bbfcc23171ed` / `c3c4f659661a` : non bit à bit), **contenu applicatif identique** : même Python 3.11.9, même `pip freeze` (27 paquets épinglés), mêmes SHA-256 de `model.pt`, `threshold.json`, `preprocessor.json`, `configs/default.yaml`, même empreinte de tous les `.py` du package `vqi`, même utilisateur |
| smoke test `synthetic` (mode CI) sur l'image finale | `OK` (`/health` 200, `/model-info` `SYN-…`, `/predict` 200, `text/plain` 415, `HEALTHCHECK` `healthy`, `uid=10001(app)`) |
| smoke test `real` sur l'image finale | `OK` (SHA-256 et seuil de `/model-info` = fichiers `models/final/` de l'hôte) |

Anomalie trouvée par la simulation Linux, et corrigée : le premier passage a donné **101 passés, 1 échec**
(`test_phase4_chain_on_synthetic_data`) — dans un conteneur sans `git`, `provenance()` renvoie
`git_commit = None` et le générateur de rapport de Phase 4 faisait `None[:7]`. Diagnostic confirmé en
installant `git` dans le conteneur (le test passe), puis correctif de robustesse dans
`src/vqi/evaluation/report_phase4.py` (`_short(commit)` → `unknown` si absent). Les artefacts de
`reports/phase4/` n'ont **pas** été régénérés : ils sont identiques au commit `42953ce` (`git status` propre
sur `reports/`). Sur un runner GitHub, `git` est présent, mais la suite ne doit pas en dépendre.

## 6.14 Sécurité minimale

Vérifié : aucun `.env`, aucun secret dans le dépôt (le service n'en a pas besoin) ; utilisateur non-root ;
contexte Docker en liste blanche (ni `.git`, ni `data/`, ni `reports/`, ni `logs/`, ni `tests/`) ; l'image ne
contient ni dataset, ni journaux historiques, ni artefact du test officiel autre que le résumé
`final_test_metrics.json` (métriques agrégées, déjà publiques dans le rapport de Phase 4). Pas d'audit de
sécurité complet (hors périmètre).

## 6.15 Limites

- **CPU uniquement**, latence ~120–135 ms par image dans le conteneur sur cette machine au repos ; pas de lot.
- **Service synchrone** : une inférence à la fois (choix de Phase 5), un seul worker uvicorn.
- **Pas de TLS, d'authentification ni de limitation de débit** : à placer derrière une passerelle.
- **Pas de déploiement cloud, pas de haute disponibilité, pas d'orchestrateur** (Kubernetes exclu par
  périmètre) ; `restart: unless-stopped` dans Compose est la seule résilience.
- **Image volumineuse** (1,86 Go) par construction : PyTorch CPU. Un export ONNX + `onnxruntime` réduirait
  l'image de plus de moitié, mais changerait le chemin d'inférence et exigerait une revalidation des scores —
  hors périmètre de cette phase.
- **CI non encore exécutée sur GitHub** (pas de dépôt distant) : validée localement, à confirmer au premier push.
- **Limite de taille appliquée après réception** et **pas de détection d'images hors domaine** (limites
  héritées de la Phase 5).
