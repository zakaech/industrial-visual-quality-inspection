# Phase 4 — Modèle final, seuil et évaluation unique du test officiel

**État : livrée le 17/09/2026.** Le rapport chiffré, généré depuis les JSON d'exécution, est dans
[`reports/phase4/phase4_report.md`](../reports/phase4/phase4_report.md) ; ce document explique la *démarche*
et l'*interprétation*. Chaque chiffre cité provient de `reports/phase4/*.json` ou de `models/final/*.json`.

## 4.1 Objectif

Transformer le candidat X6 (Phase 3) en système final de classification binaire sain / défectueux, évalué
proprement : vérifier le checkpoint, analyser et calibrer les scores sur la validation, choisir et **geler**
le seuil, puis évaluer **une seule fois** le test officiel KolektorSDD2, analyser les erreurs, regarder où le
réseau regarde (Grad-CAM), et documenter.

L'ordre a été respecté strictement : audit → vérification → analyse validation → seuil → gel (commit `7ade5da`)
→ test unique → erreurs / Grad-CAM → rapport. Le test n'a été chargé qu'une fois, après le gel, et aucun
paramètre n'a été modifié ensuite.

## 4.2 Modèle final

ResNet-18 pré-entraîné ImageNet, `layer3` + `layer4` + tête (Dropout 0,2 + Linear 512→1) fine-tunés, pooling
moyen, 11 177 025 paramètres ; recette héritée de X4 : `pos_weight` 8,467, sans sampler, BatchNorm `all_eval`,
AdamW wd 1e-4, LR C `layer3` 5e-5 / `layer4` 1e-4 / tête 5e-4, batch 16, seed 42, augmentations flips +
photométrie légère (train seul). Checkpoint `stage_C_best.pt` (étape C, époque 5, AP val 0,9408 — critère
préenregistré), entraîné au commit `5b62df4`.

Bundle gelé `models/final/` : `model.pt` (copie octet à octet du checkpoint, SHA-256 `4233a9cd80534b8e…`,
ignoré par git), `preprocessor.json` (variante `none`, padding médiane, 640×224, normalisation ImageNet),
`threshold.json` (`frozen: true`), `model_card.json`, puis `final_test_metrics.json` après le test.
`Predictor.from_bundle()` (`src/vqi/inference/predictor.py`) est l'unique chemin d'inférence : il a servi
au test et servira à l'API.

**Vérification (étape 2)** : métadonnées du checkpoint = résultat X6 (étape, époque, métrique, `pos_weight`,
sampler, config, commit, seed, prétraitement) ; deux passes de validation bit-identiques ; écart maximal avec
les scores enregistrés en Phase 3 : 9,9 × 10⁻¹⁷ ; AP et matrice à rappel ≥ 0,95 reproduites à l'identique.

## 4.3 Sélection du seuil

Contrainte de développement : rappel ≥ 0,95 sur la validation ; contrainte métier : un faux négatif coûte plus
qu'un faux positif, avec un ratio **inconnu**.

Règle fixée dans la configuration *avant* l'analyse : `min_recall` — seuil le moins permissif atteignant le
rappel cible (0,01974 : 47/49 positifs) — puis **marge** au milieu (en logit) vers le positif suivant (0,01781)
→ **seuil gelé 0,018752**. La marge coûte 16 faux positifs de plus sur la validation (35 → 51) pour le même rappel,
et protège d'une dérive des scores de ±5 % en logit sur de nouvelles pièces. Sur la validation, au seuil gelé :
rappel 0,959 [0,898 ; 1,000], précision 0,480 [0,382 ; 0,576], FPR 0,122 [0,093 ; 0,153], TP 47 / FP 51 / FN 2 / TN 366.

Pourquoi pas le seuil « à meilleure précision » ni le coût attendu minimal : la table des candidats (rapport §4.3)
montre que, pour tout ratio de coût entre 5 et 20, le point de coût minimal garde 46/49 défauts (rappel 0,939),
*sous* la contrainte ; la contrainte de rappel prime par construction. Le seuil n'a donc pas été choisi en regardant
la précision, mais en appliquant la règle préenregistrée.

## 4.4 Calibration

Avec `pos_weight` 8,47, rien ne garantit que la sortie sigmoïde soit une probabilité. Diagnostics sur la
validation : ECE 0,022 (bins à effectifs égaux) / 0,0055 (largeurs égales), Brier 0,020, NLL 0,088, moyenne
prédite 0,101 pour une prévalence 0,105. Température ajustée par NLL : 1,013 (≈ identité) ; isotonique en
validation croisée 5 plis : ECE 0,010, Brier 0,019 — gain marginal pour 49 positifs et ECE in-sample nul
(surajustement). **Décision : scores bruts**, selon la règle « on garde les scores bruts sauf ECE > 0,10 ».
Deux précisions honnêtes : (1) les scores sont bien calibrés *en moyenne* mais pas dans la bande utile —
autour du seuil (scores 0,02–0,03) la fréquence observée de défauts est ≈ 15 %, le seuil ne se lit donc pas comme
« 2 % de probabilité » ; (2) toute calibration monotone laisserait le point de fonctionnement inchangé, la
question n'est que d'interprétabilité de la sortie.

## 4.5 Résultats validation (développement)

Chiffres optimistes par construction : le modèle, le checkpoint et le seuil ont été choisis dessus.
AP 0,941 [0,891 ; 0,979], ROC-AUC 0,987 ; au seuil gelé : rappel 0,959, précision 0,480, FPR 0,122.
Comparaison de développement (validation) : E0 contraste seul AP 0,402 ; E5 SVM 0,862 ; X4 étape B 0,919 ; X6 0,941.

## 4.6 Résultats du test officiel (finaux)

Évaluation unique le 17/09/2026 (16:09 UTC), 1 004 images dont 110 défectueuses, modèle et seuil gelés :

| Métrique | Test officiel | IC 95 % |
|---|---|---|
| AP | **0,948** | [0,909 ; 0,977] |
| ROC-AUC | 0,979 | [0,959 ; 0,994] |
| Rappel au seuil gelé | **0,955** (105/110, **5 défauts manqués**) | [0,907 ; 0,991] |
| Précision au seuil gelé | 0,536 | [0,464 ; 0,603] |
| FPR au seuil gelé | 0,102 (91/894 pièces saines à réinspecter) | [0,083 ; 0,121] |
| F1 / F2 | 0,686 / 0,825 | — |
| Matrice | TP 105 · FP 91 · FN 5 · TN 803 | — |

La validation avait annoncé rappel 0,959 / précision 0,480 / FPR 0,122 : le comportement se transporte au test
sans surprise, ce qui est la finalité de la marge et du seuil choisi sans regarder le test.

Références gelées, scorées dans la même passe (aucune décision) : E0 AP 0,384 ; E5 SVM AP 0,814 (précision 0,171
à rappel 0,95) ; X4 étape B AP 0,947. Bootstrap apparié sur le test : X6 − E5 ΔAP +0,133 [+0,089 ; +0,191] et
Δprécision@0,95 +0,373 [+0,045 ; +0,677] ; X6 − X4 ΔAP +0,001 [−0,022 ; +0,024] — **l'étape C n'apporte pas de
gain démontré sur le test**, alors que la validation le laissait espérer ; c'est précisément le genre de conclusion
qu'une évaluation unique et gelée permet d'énoncer sans se tromper soi-même.

## 4.7 Analyse des erreurs

Faux négatifs (5) : scores 0,0148–0,0185, tous juste sous le seuil 0,0188. Aire médiane 367 px contre 2 535 px pour
les vrais positifs ; 2 des 5 sont dans la strate < 250 px (rappel 3/5 dans cette strate, 102/105 au-delà).
Observations — *semblent associés à* : une petite aire (20816 : 181 px, 20544 : 199 px), un contraste local très
faible (20969 et 20281 : défauts étendus mais presque invisibles), une position en bordure d'image (20816, 20236).
Hypothèse, non démontrée : à la résolution de `layer4` (cellules ≈ 32×32 px), un défaut de quelques dizaines de pixels
peu contrasté n'active pas suffisamment la carte pour dépasser un seuil déjà très bas.

Faux positifs (91) : médiane des scores 0,024, 90 % sous 0,094, seulement 2 au-dessus de 0,5 (20358 : 0,84,
20719 : 0,79). Ce sont essentiellement des pièces saines poussées au-dessus d'un seuil très bas par la contrainte
de rappel, pas des erreurs confiantes. Aucun motif photométrique global ne les distingue des vrais négatifs
(luminance 44,1 vs 44,6, contraste 9,1 vs 9,0), cohérent avec la faible séparabilité triviale vue en Phase 1.
Aucune conclusion causale.

## 4.8 Grad-CAM

Grad-CAM sur `layer4`, masques utilisés seulement comme référence de mesure. Vrais positifs confiants (p ≈ 1, 6
exemples) : pic de la carte dans le masque pour 6/6, médiane de 64 % de la masse de la carte dans le défaut alors
que le défaut couvre 10–30 % de l'image — le réseau regarde bien la région annotée. Vrais positifs limites et faux
négatifs : cartes diffuses, pic hors masque (médiane 16 % et 6 % de masse dans le masque) — la décision repose sur
peu d'évidence localisée. Limites : résolution d'une cellule (≈ 32×32 px), sensibilité à la couche choisie, carte
de sensibilité du logit et non segmentation, aucune information sur la calibration.

## 4.9 Limites

- 110 défauts dans le test, 49 en validation : IC larges (rappel test [0,907 ; 0,991]).
- Sélection sur la validation → chiffres de validation optimistes ; seul le test est non biaisé et il n'a servi à
  aucune décision.
- Petits défauts (< 250 px) : point faible avéré (3/5 sur le test).
- Domaine spécifique (commutateurs, éclairage fixe) : aucune garantie de généralisation à une autre ligne, un autre
  produit ou un autre capteur sans nouvelle validation.
- Calibration imparfaite dans la bande utile ; probabilités non métier.
- Coût CPU : ≈ 164 ms par image en inférence (prétraitement inclus) sur l'i7-8665U ; ≈ 14 min par époque en étape C.
- Grad-CAM qualitatif à résolution grossière.

## 4.10 Reproductibilité

| Élément | Valeur |
|---|---|
| Commits | gel `7ade5da` (test non chargé) ; résultats de test et rapport dans le commit final de Phase 4 |
| Configuration | `configs/default.yaml` (section `phase4`), hash au gel `f9397663b72a` ; config d'entraînement du checkpoint `18eed599c81b` |
| Seed | 42 |
| Checkpoint | `runs/20260917_002044_X6_C_winner_18eed599c81b/stage_C_best.pt` → `models/final/model.pt`, SHA-256 `4233a9cd80534b8e…` |
| Prétraitement | `models/final/preprocessor.json` |
| Seuil final | 0,018752 (`models/final/threshold.json`, gelé le 17/09/2026 15:26 UTC) |
| Test officiel | évalué une fois le 17/09/2026 16:09 UTC ; `test_used = true` dans `reports/phase4/test_evaluation.json`, `error_analysis.json` et `models/final/final_test_metrics.json` uniquement ; `scripts/evaluate_test.py` refuse toute seconde exécution |
| Commandes | `verify_checkpoint.py` → `select_threshold.py --freeze` → `evaluate_test.py` → `build_phase4_report.py` |
