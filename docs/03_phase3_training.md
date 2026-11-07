# Phase 3 — Deep learning : ResNet-18 par transfert, comparé à la baseline classique

**État : livrée le 17/09/2026 — X1 à X6 exécutés ; X7 (ablation du pooling) non lancé (non justifié par X6,
voir Phase 4).** Le rapport chiffré est dans [`reports/phase3/phase3_report.md`](../reports/phase3/phase3_report.md) ;
le dry-run dans [`reports/phase3/dry_run.md`](../reports/phase3/dry_run.md). Tous les chiffres ci-dessous en sont extraits.

## 1. La question de la phase

Le CNN apporte-t-il quelque chose de **démontrable** face à la référence Phase 2 (82 descripteurs + SVM,
AP 0,862 [0,770 ; 0,935] sur validation), et surtout dans le **régime haut rappel** où la baseline coûte
cher (précision 0,233 / FPR 0,372 à rappel 0,95) ?

## 2. Architecture logicielle

```
data/splits/v1.csv ──load_split_rows(allowed=[train, val])──► train (1 865 / 197 déf.) · val (466 / 49)   test → ForbiddenSplitError
      │
      ▼
CanvasCache (.npy memmap)  ← sortie EXACTE de Preprocessor(cfg) (variante none, padding médiane, 640×224) ; vérifiée contre le live
      │
      ▼
VqiDataset : canevas uint8 → [Augmenter, TRAIN seul, RNG(seed, epoch, index)] → Preprocessor.to_tensor()
      │
      ▼
DefectClassifier : ResNet-18 ImageNet → pooling (avg | max | avgmax) → Dropout 0,2 → Linear(→1)
      │            biais initial = logit(p) = log(p/(1−p)) = log(n_pos/n_neg) = log(197/1668) = −2,136
      ▼
BCEWithLogits(pos_weight ∈ {1, √ratio ≈ 2,91, ratio ≈ 8,47}) | WeightedRandomSampler
      │
      ▼
Trainer par étapes A → B → C (gel/dégel par groupe, LR par groupe, cosinus + warmup par étape,
      │                      early stopping sur AP val, best.pt / last.pt, history.jsonl)
      ▼
evaluate_scores (identique à la Phase 2) + rappel par taille de défaut + bootstrap APPARIÉ vs E5_grid_edge_check
```

Fichiers : `src/vqi/data/dataset.py`, `src/vqi/preprocessing/augment.py`, `src/vqi/models/factory.py`,
`src/vqi/training/{losses,features,loop,experiments,dryrun}.py`, `src/vqi/evaluation/compare.py`,
`scripts/{dry_run_training,train}.py`. Le garde-fou du test officiel est partagé
(`vqi.data.splits.load_split_rows`) et la baseline Phase 2 y délègue sans changement de comportement.

## 3. Choix défendables

| Choix | Décision | Pourquoi |
|---|---|---|
| Prétraitement | `Preprocessor` de Phase 2, `active: none`, padding médiane — aucune resize/normalisation ajoutée | une seule implémentation entraînement/API (R4) ; ResNet-18 accepte 640×224 (pooling adaptatif) |
| Cache | canevas uint8 memmap, invalidé par hash (Preprocessor + splits) | supprime décodage + resize de la boucle ; vérifié à l'octet près contre le live (dry-run et test) |
| Augmentations | flips H/V p 0,5 ; luminosité/contraste ×[0,9 ; 1,1] p 0,3 ; translation OFF ; rotation/crop/erasing/bruit rejetés | préservent le label (test : aucun pixel de défaut perdu) ; images sombres → jitter léger |
| Biais de tête | `log(n_pos/n_neg)` du train | le réseau prédit la prévalence avant tout apprentissage ; stabilise le début, probabilités sensées |
| Pooling | GAP par défaut ; GMP/avg+max en ablation conditionnelle X7 | la carte layer4 fait 20×7 : la moyenne dilue un défaut local (même mécanisme que l'échec de LBP en Phase 2) |
| BatchNorm | **`all_eval`** : toutes les BN restent en eval à toutes les étapes (statistiques ImageNet figées), γ/β des couches dégelées entraînables, dropout de la tête actif | incident X2 (16/09) : avec les BN de `layer4` en mode batch, le checkpoint X1 passait d'une perte val de 0,201 à 15,31 (|logit| 2,5 → 18,3) avant tout pas d'optimisation ; le contrôle de transition du `Trainer` (sorties val identiques en mode d'entraînement et en eval, dérive des `running_mean` nulle) l'interdit désormais et est testé sous les deux politiques |
| Étapes | A sonde linéaire (features gelées cachées, sans augmentation) → B layer4 → C layer3+4 ; layer1–2 jamais | coût CPU décroissant vers les couches basses, gain attendu faible ; A est un résultat en soi |
| LR | A tête 1e-3 ; B layer4 1e-4 / tête 1e-3 ; C layer3 5e-5 / layer4 1e-4 / tête 5e-4 ; AdamW wd 1e-4 ; cosinus + 1 époque de warmup par étape | LR décroissant avec la profondeur (couches basses génériques) ; pas de grille (49 positifs n'y suffiraient pas) |
| Déséquilibre | matrice à l'étape B : `pos_weight` 1 / √ratio / ratio + sampler équilibré ; **pas un coût métier** | le coût entre au seuil (Phase 4) ; on journalise la moyenne des probabilités val (déformation) |
| Sélection du gagnant B | **précision @ rappel ≥ 0,95**, puis AP ; FPR @ 0,95 rapporté — fixé avant les runs | contexte industriel haut rappel |
| Early stopping | AP val, patience 3 (B, C) / 5 (A), min_delta 0,002, ≥ 2 époques, best + last | AP par époque bruitée avec 49 positifs |
| Comparaison | bootstrap **apparié** (mêmes 466 images) sur ΔAP et Δprécision@0,95 vs `E5_grid_edge_check` | les IC marginaux se recouvrent trop facilement ; la référence n'est jamais retouchée |

## 4. Reproductibilité et provenance

Seed 42 (`seed_everything` : random, numpy, torch ; algorithmes déterministes `warn_only` ; 4 threads) ;
`DataLoader` avec générateur seedé et `worker_init_fn` ; augmentation keyée `(seed, epoch, index)` donc
indépendante du nombre de workers ; par run `runs/<horodatage>_<id>_<hash config>/` : `config.yaml` figée,
`history.jsonl`, `stage_<X>_best.pt` / `_last.pt` (state_dict + étape + époque + métrique + description du
Preprocessor + hash config + commit git + seed + pos_weight + sampler), `result.json` ; copie dans
`reports/phase3/experiments/<id>.json` avec la même structure de métriques que la Phase 2.

## 5. Dry-run X0 — voir `reports/phase3/dry_run.md`

Mesuré, jamais estimé : construction du cache et égalité avec le live, débit du chargement par
`num_workers` (cache ON/OFF), pas forward/backward par profil (A tête, B layer4, C layer3+4, tout dégelé)
et par batch (8 / 16 / 32), RSS de pointe, estimation d'époque dérivée des pas mesurés, déterminisme de deux
runs identiques (écart max des pertes et des poids), saturation photométrique après augmentation.

## 6. Matrice d'expériences (validée)

| Id | Étapes | Déséquilibre | Repart de |
|---|---|---|---|
| X1_linear_probe | A | pos_weight 1 | ImageNet |
| X2_B_pw1 | A → B | pos_weight 1 | — |
| X3_B_pw_sqrt_ratio | A → B | pos_weight √ratio | — |
| X4_B_pw_ratio | A → B | pos_weight ratio | — |
| X5_B_sampler | A → B | sampler équilibré | — |
| X6_C_winner | C | hérité du gagnant | best.pt du gagnant de X2–X5 (critère §3) |
| X7 (conditionnel) | C | hérité | ablation pooling, seulement si X6 et le dry-run le justifient |

## 7. Résultats (validation, 466 images / 49 défectueuses — développement)

| Run | Recette | AP | IC 95 % | Préc. @ R ≥ 0,95 | FPR @ 0,95 | Meilleure époque | Durée |
|---|---|---|---|---|---|---|---|
| X1 | sonde linéaire (features ImageNet gelées) | 0,791 | [0,679 ; 0,878] | 0,142 | 0,679 | 15/20 | 7 min |
| X2 | B, `pos_weight` 1 | 0,906 | [0,828 ; 0,970] | 0,187 | 0,489 | 7/8 | 95 min |
| X3 | B, `pos_weight` 2,91 | 0,904 | [0,828 ; 0,966] | 0,214 | 0,415 | 3/6 | 57 min |
| X4 | B, `pos_weight` 8,47 | 0,919 | [0,848 ; 0,976] | **0,251** | 0,336 | 7/8 | 59 min |
| X5 | B, sampler équilibré | 0,877 | [0,787 ; 0,953] | 0,110 | 0,911 | 4/7 | 52 min |
| X6 | C depuis X4 | **0,941** | [0,891 ; 0,979] | **0,573** | **0,084** | 5/8 | 113 min |
| réf. E5 SVM | Phase 2 | 0,862 | [0,770 ; 0,935] | 0,233 | 0,372 | — | — |

Gagnant de la matrice B selon le critère préenregistré (précision @ rappel ≥ 0,95, puis AP) : X4. Bootstrap
apparié vs la SVM : tous les IC de X1–X5 contiennent 0 ; X6 est la seule configuration dont les IC appariés
excluent 0 sur la validation (ΔAP +0,079 [+0,011 ; +0,153], Δprécision@0,95 +0,340 [+0,022 ; +0,659]) — chiffres
de développement, le test unique de la Phase 4 les remet en perspective (X6 et X4 y sont indissociables).

Incident méthodologique documenté : la première tentative de X2 a été interrompue avant tout résultat parce que la
politique BatchNorm `frozen_eval` détruisait la tête de la sonde à la transition A → B (perte val 0,20 → 15,3) ;
correctif `all_eval` + contrôle de transition (commit `9d704ec`), voir §3.

## 8. Tests

Tests (`pytest`) dont, pour cette phase : contrat du dataset (forme `(3, 640, 224)`, float32, refus du
test, val déterministe, cache == live, augmentation train seule et keyée par époque, ordre du loader
reproductible) ; augmentations (flips = permutations involutives, défaut intégralement conservé sous 50 tirages,
photométrie bornée) ; modèle (biais = log(n_pos/n_neg), sigmoïde initiale = prévalence, gel par groupe, BN
figée, groupes d'optimiseur, poolings) ; perte (`pos_weight` = formule manuelle, résolution ratio/√ratio),
sampler (~50 % de positifs), cosinus + early stopping ; bootstrap apparié ; dry-run et deux expériences de bout
en bout sur le dataset synthétique (checkpoints, history, result.json, sélection du gagnant).
