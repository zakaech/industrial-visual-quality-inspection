# Phase 2 — Prétraitement OpenCV et baseline classique

**État : livrée le 16/09/2026.** Le rapport chiffré, entièrement généré depuis les artefacts
d'exécution, est dans [`reports/phase2/phase2_report.md`](../reports/phase2/phase2_report.md).
Ce document explique la *conception* et l'*interprétation* ; les chiffres cités renvoient au rapport
(validation uniquement — **le test officiel n'a jamais été chargé**, le chargeur de la baseline lève une
exception si on le lui demande, et un test le vérifie).

## 1. La question de la phase

> Quel signal peut-on extraire de ces images *sans* deep learning, et quelle amélioration le CNN de la
> Phase 3 devra-t-il réellement démontrer ?

Réponse courte : des descripteurs classiques bien choisis atteignent une **AP de 0,86** sur la validation
(IC 95 % [0,77 ; 0,94]) contre 0,40 pour le contraste global seul ; mais à **rappel 0,95, la précision
n'est que de 0,23** (FPR 0,37 : il faudrait réinspecter 37 % des pièces saines). C'est là que le CNN
devra apporter sa valeur, pas sur l'AP brute qu'un intervalle de confiance large rend difficile à départager.

## 2. Un seul module de prétraitement (partie A)

```
src/vqi/preprocessing/
├── io.py         décodage cv2.imdecode (chemin ou octets → même chemin de code), 8/16 bits, 1/3/4 canaux → RGB uint8
├── geometry.py   redimensionnement à ratio conservé + padding centré ; masques transformés à l'identique
├── enhance.py    opérateurs purs RGB→RGB : equalize_hist, clahe, gaussian_blur, median_blur, unsharp_mask, laplacian_sharpen
├── pipeline.py   Preprocessor.from_config(cfg[, variant]) → __call__(source) → canevas uint8 ; to_tensor() pour le CNN
├── quality.py    CNR du défaut (analyse, jamais entraînement)
└── compare.py    figures et analyse CNR de la Phase 2
```

Décisions de conception défendables en entretien :

- **Une seule implémentation** pour l'entraînement (chemin) et l'API (octets) : `decode_image` passe *toujours*
  par `cv2.imdecode` sur un tampon, et un test de contrat vérifie l'égalité à l'octet près des deux sources
  (`test_full_pipeline_path_vs_bytes_contract`). C'est la parade au risque R4 (dérive entraînement/serving).
- **Géométrie** : `min(640/H, 224/W)` = 0,978 sur une image médiane 637×229 → 623×224 puis 17 lignes de padding.
  Sous-échantillonnage de 2 % : le plus petit défaut (23 px) survit. L'écrasement anisotrope (`stretch`) reste
  disponible mais rejeté (D7).
- **Padding à la médiane de l'image** plutôt qu'à zéro : un bord noir constant crée des patches artificiellement
  « anormaux ». Mesuré : AP 0,832 (médiane) contre 0,773 (zéro) — IC recouvrants, mais cohérent avec l'hypothèse ;
  la médiane est conservée.
- **Opérateurs sur la luminance (LAB)** pour les transformations de contraste, afin de conserver la faible
  information couleur (écart entre canaux : AUC 0,64 en Phase 1) ; lissages sur les trois canaux.
- Chaque variante est une **liste ordonnée d'opérateurs dans la config** (`preprocessing.enhancement.variants`) ;
  la variante servie par l'API est `enhancement.active`. Aucune combinaison n'a été empilée a priori.

## 3. Comment les prétraitements ont été jugés

Deux juges, dans cet ordre :

1. **CNR du défaut** (contraste-sur-bruit, depuis les masques, sur les 197 défectueuses du train, par strate
   de taille) — pour *rejeter* ce qui dégrade la visibilité : égalisation globale (CNR ×0,47, écart-type ×8),
   unsharp (×0,72), Laplacien (×0,39), CLAHE (×0,77). Les zooms confirment : le bruit de texture est amplifié
   au moins autant que le défaut.
2. **AP du classifieur sur la validation** (E3) — pour *retenir* : aucune variante ne bat `none`
   (0,832) ; `denoise_light` 0,826, `denoise_strong` 0,809, `unsharp` 0,789, `clahe` 0,786, `equalize` 0,737.
   Tous les IC se recouvrent avec celui de `none`.

Pourquoi deux juges : le CNR **récompense mécaniquement le lissage** (son dénominateur, l'écart-type de
l'anneau, baisse quand on floute) — d'où un CNR ×1,32 pour le filtre médian qui ne se traduit par *aucun* gain
de classification. Un seul critère aurait conduit à une mauvaise décision. Résultat : `active: none`.
Le CNN de la Phase 3 apprendra lui-même sa normalisation de contraste ; CLAHE est en outre à proscrire dans le
régime haut rappel (précision 0,11 à rappel 0,95 contre 0,20 sans prétraitement).

## 4. Les descripteurs et ce qu'ils ont appris

| Groupe | Dim | AP val | Lecture |
|---|---|---|---|
| `contrast_only` (E0) | 1 | 0,402 | la référence triviale ; AUC 0,70 sur validation (0,72 sur train officiel en Phase 1) |
| `global_stats` + histogramme | 46 | 0,776 | la forme de la distribution des intensités porte l'essentiel du signal global |
| `lbp` (P,R) = (8,1), (16,2) | 28 | 0,179 | **quasi inutile seul** : un histogramme LBP global est dominé par la texture du produit ; un défaut de 1,6 % de l'image ne le déplace pas |
| `local_stats` (grille 32×32) | 8 | 0,789 | le meilleur groupe seul, avec 8 dimensions : la bonne question est « existe-t-il *quelque part* un patch anormal ? » |
| `combined` | 82 | 0,832 → 0,862 (E5) | complémentarité réelle entre statistiques globales et locales |

L'échec de LBP est un résultat, pas un accident : il est *attendu* pour des défauts petits et locaux, et il
justifie a posteriori l'ajout de `local_stats` (validé avant exécution, pas après). GLCM n'a pas été ajouté.

## 5. Le classifieur et la discipline de sélection

- `Pipeline(StandardScaler → SVC)` : le scaler est ajusté sur le train **à l'intérieur du pipeline** ; un test
  vérifie que sa moyenne est celle du train et non de train+val.
- Hyperparamètres par **validation croisée stratifiée par groupe à 5 plis dans le train** ; la validation ne sert
  qu'à comparer les 16 configurations. Sans cela, le meilleur chiffre de validation serait sur-sélectionné.
- **Bord de grille** : la CV a choisi γ = 0,001 (borne basse) dans *toutes* les expériences RBF. Plutôt que
  d'ignorer le signal, deux contrôles : grille étendue (E5, choisit encore le coin : C = 1000, γ = 10⁻⁴,
  AP 0,862) et noyau **linéaire** (E5_linear_svm, optimum intérieur C = 0,1, AP 0,853). Conclusion : les
  descripteurs sont séparés quasi linéairement ; le RBF n'apporte rien. C'est le genre de constat qu'on doit
  savoir expliquer en entretien : γ → 0 rend le noyau RBF localement linéaire.
- Random Forest (E4, 500 arbres) : AP 0,805, IC recouvrant — comparaison secondaire, pas de gain.
- `class_weight` (E2) : none 0,846, balanced 0,832, manuel {1 : 3} 0,846 — **aucun effet mesurable sur le
  classement**. C'est cohérent avec le cadrage : la pondération déplace la marge, pas l'ordre des scores ;
  le levier des coûts reste le seuil. Le poids manuel est exploratoire et n'est pas un coût métier.

## 6. Évaluation

`src/vqi/evaluation/metrics.py` est écrit pour être réutilisé tel quel par le CNN : AP, ROC-AUC, IC bootstrap
(1 000 rééchantillonnages), table rappel cible → seuil / précision / FPR / TP / FP / FN / TN, balayage complet
des seuils, matrice au seuil par défaut (rapportée, jamais retenue). Les scores SVM sont des distances à la
marge : aucune calibration n'est revendiquée, le seuil opérationnel est un sujet de Phase 4.

## 7. Limites (à dire avant qu'on vous les oppose)

- 49 positifs en validation → IC de ±0,08 environ sur l'AP ; la plupart des écarts de la matrice sont dans le bruit.
- 16 configurations comparées sur la même validation → optimisme de sélection ; seul le test (Phase 4) est non biaisé.
- Le CNR a peu d'images dans les petites strates (3 et 6) ; le résultat sur les micro-défauts est indicatif.
- Le SVM n'est pas calibré ; on ne compare que des classements.

## 8. Ce que la Phase 3 hérite

| Décision | Valeur | Preuve |
|---|---|---|
| `preprocessing.enhancement.active` | `none` | rapport §6 (E3) et §3 (CNR) |
| `preprocessing.pad_value` | `median` | E3_pad_zero |
| Référence à battre (validation) | AP 0,862, IC [0,770 ; 0,935] (E5_grid_edge_check) | rapport §6.3 |
| Point faible à attaquer | précision 0,23 / FPR 0,37 à rappel 0,95 | rapport §6.4 |
| Déséquilibre | toujours ouvert ; `class_weight` sans effet sur la baseline | rapport §6.2 |
| Augmentation photométrique | légère (images sombres, peu contrastées) | rapport §2 |

## 9. Reproduire

```bash
python scripts/compare_preprocessing.py     # figures + CNR → reports/phase2/
python scripts/run_baseline.py              # 16 expériences → reports/phase2/experiments.csv (+ JSON par expérience)
python scripts/build_phase2_report.py       # rapport Markdown
pytest                                      # 58 tests
```

Temps mesurés (i7-8665U, CPU) : extraction des 83 descripteurs ≈ 83 ms/image (154 s pour le train) ;
sélection + ajustement d'un SVM (80 fits) ≈ 7–19 s ; Random Forest ≈ 2 s. Les matrices de descripteurs sont
mises en cache par variante (`data/processed/features/`, ignoré par git, invalidé par hash de contenu).
