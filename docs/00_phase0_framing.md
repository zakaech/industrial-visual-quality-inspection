# Phase 0 — Cadrage du projet

**État : validé le 15/09/2026 (D1–D10).** Les points marqués `[DECISION]` sont tracés dans la table du §5. Chaque chiffre de ce document est soit (a) une statistique publiée sur un dataset, avec sa source
et la mention « à vérifier au téléchargement », soit (b) un emplacement `TBD` explicite à remplir depuis une
exécution réelle. Aucune métrique n'est jamais inventée dans ce dépôt.

---

## 0. Environnement cible (mesuré le 15/09/2026)

| Élément | Valeur | Conséquence |
|---|---|---|
| CPU | Intel Core i7-8665U, 4 cœurs / 8 threads | Tout l'entraînement se fait sur CPU |
| RAM | 16 Go | Taille de batch et résolution d'entrée doivent rester modestes |
| GPU | Intel UHD 620 (pas de CUDA) | Aucune accélération GPU ; `device: cpu` dans la config |
| Python | 3.10 et 3.12 installés, 3.11 absent → **3.11.9 installé en Phase 1** (winget) | Parité avec l'image Docker (`python:3.11-slim`) |
| Docker | 29.7 | OK |
| Git | 2.47 | Dépôt initialisé en Phase 1 |

La contrainte matérielle est le facteur qui pèse le plus sur le choix du dataset et du modèle : le pipeline
complet (données → entraînement → évaluation → API) doit pouvoir s'exécuter de bout en bout en quelques heures sur
ce portable.

---

## 1. Choix du jeu de données

### 1.1 Candidats

Les chiffres ci-dessous proviennent des publications de référence / pages officielles des datasets et doivent être
recontrôlés sur les fichiers téléchargés (une vérification `expected_counts` est prévue dans la config à cet effet).

| Critère | NEU-CLS / NEU-DET | DAGM 2007 | MVTec AD | Severstal (Kaggle) | KolektorSDD (v1) | **KolektorSDD2** |
|---|---|---|---|---|---|---|
| Domaine | Surface de bandes d'acier laminées à chaud | Textures synthétiques | 15 objets/textures | Tôles d'acier, ligne de production | Commutateurs électriques | Commutateurs électriques, ligne de production |
| Images | 1 800 (200×200, niveaux de gris) | ~32k (512×512, niveaux de gris) test officiel inclus | 5 354 (700–1024 px, couleur) | 12 568 train (1600×256) + test non étiqueté | 399 (~500×1240) | 3 335 (~230×630) |
| Classes | 6 types de défauts, 300 chacun, **pas de classe saine** | 10 textures, chacune saine/défectueuse | sain vs 73 types de défauts | 4 types de défauts + sain | sain / défectueux | sain / défectueux |
| Annotation | label image (+ boîtes dans NEU-DET) | faible : ellipse autour du défaut | masques pixel (test uniquement) | masques RLE, polygones grossiers | masques pixel | masques pixel |
| Qualité d'annotation | bonne au niveau image | faible | excellente | bruitée | bonne | bonne |
| Réalisme | réel, mais recadrages basse résolution | **synthétique** | réel, conditions de laboratoire | production réelle | production réelle | production réelle |
| Classe saine | **non** | oui | oui (train = sain uniquement) | oui (~47 % ont un défaut) | oui (52/399 défectueuses) | oui (356/3 335 défectueuses ≈ 10,7 %) |
| Difficulté | classification quasi saturée (>99 % rapportés) | facile pour les CNN | difficile, cadre non supervisé | difficile, segmentation | modérée, dataset minuscule | modérée à difficile, défauts petits et subtils |
| Faisabilité CPU | excellente | acceptable | par catégorie seulement, features gelées | mauvaise | excellente | **bonne** |
| Risques de fuite connus | recadrages issus des mêmes bandes ; pas d'identifiant de pièce ; quasi-doublons possibles | synthétique → aucun | aucun pour le protocole standard | recadrages consécutifs d'une même bobine ; quasi-doublons documentés | 8 images par pièce physique → découpage par groupe obligatoire | pas d'identifiant de pièce fourni ; à vérifier empiriquement |
| Découpage officiel | non | oui | oui | train seulement (labels du test privés) | non (l'article utilise 3-fold par pièce) | **oui** (2 331 train / 1 004 test) |
| Licence (à confirmer au téléchargement) | recherche | recherche | CC BY-NC-SA 4.0 | règles de la compétition | recherche | recherche |

### 1.2 Pourquoi pas les autres

- **NEU-CLS/NEU-DET** — le classique attendu, mais il **n'a pas de classe saine** : la décision accepter/rejeter
  qui porte tout l'argument d'asymétrie des coûts ne peut pas s'y formuler. La classification multi-classes sur NEU
  est en outre un benchmark quasi saturé : les résultats seraient flatteurs mais peu informatifs. Il reste candidat
  pour une *extension v2* (typage du défaut), le code étant agnostique au dataset via la configuration.
- **DAGM 2007** — textures et défauts synthétiques. Adapté à un exercice pédagogique, peu représentatif
  d'un cas de vision « production ».
- **MVTec AD** — la référence de la *détection d'anomalies non supervisée*. Les images d'entraînement sont toutes
  saines : un classifieur supervisé ne pourrait être entraîné qu'en découpant le (petit, par catégorie) jeu de test.
  Il convient mieux à une approche par banque de mémoire (PatchCore) qu'à un pipeline de fine-tuning, et ses ~5 Go /
  images de 1024 px pèsent lourd sur ce portable. Conservé comme *comparaison v2* possible (référence non
  supervisée), pas comme dataset v1.
- **Severstal** — le plus réaliste, mais 12,5k images en 1600×256 est un problème de segmentation qui exige
  réalistement un GPU ; les labels du test sont privés et les masques sont notoirement bruités.
- **KolektorSDD v1** — réel, mais 52 images défectueuses au total ; toute métrique y a un intervalle de confiance
  trop large pour soutenir une décision de seuil.

### 1.3 Recommandation `[DECISION D1]`

**KolektorSDD2 (KSDD2)** — Božič, Tabernik & Skočaj, *Computers in Industry*, 2021.

Statistiques publiées (à vérifier au téléchargement) :

| Partition | Total | Défectueuses | Saines | Taux de défaut |
|---|---|---|---|---|
| Train officiel | 2 331 | 246 | 2 085 | 10,6 % |
| Test officiel | 1 004 | 110 | 894 | 11,0 % |

Pourquoi il convient à ce projet :

1. **Il contient les deux classes dont la décision métier a besoin** (accepter / rejeter), avec un taux de défaut
   réaliste de ~11 % qui rend la discussion sur le déséquilibre et l'asymétrie des coûts incontournable plutôt
   qu'académique.
2. **Images réelles de production** d'une pièce industrielle (commutateurs électriques), avec les familles de défauts
   visées par le projet : rayures, petites taches, imperfections de surface ; beaucoup de défauts sont petits et
   subtils → le problème n'est pas saturé.
3. **Masques au pixel** même si la v1 travaille au niveau image : ils permettent (a) le *rappel en fonction de la
   taille du défaut*, (b) une vérification Grad-CAM que le modèle regarde bien le défaut, (c) une extension
   segmentation en v2 — sans coût d'annotation supplémentaire.
4. **Découpage train/test officiel** des auteurs → le chiffre final est comparable à la littérature et le jeu de test
   peut être mis en quarantaine dès le premier jour.
5. **Résolution modeste (~230×630)** et ~3,3k images → le fine-tuning complet d'un petit CNN est faisable sur le CPU
   cible.

Réserves connues (traitées en §3 et §6) : aucun identifiant de pièce physique n'est fourni ; les labels du test sont
publics (discipline requise) ; les labels dérivent des masques, donc les défauts minuscules comptent comme positifs.

---

## 2. Cadrage de la tâche `[DECISION D2]`

| Option | Ce qu'elle exige | Ce qu'elle apporte | Implications |
|---|---|---|---|
| **A. Binaire supervisée** (sain vs défectueux, niveau image) | labels image pour les deux classes | un score P(défectueux) → accepter/rejeter avec un point de fonctionnement explicite | Correspondance directe avec la décision métier ; asymétrie gérée par une politique de seuil ; la plus simple à servir et à expliquer. |
| B. Multi-classes par type de défaut | labels typés (NEU, Severstal) | oriente la pièce vers la bonne reprise | KSDD2 n'a qu'un seul type « défectueux » → impossible ici. L'asymétrie devient une matrice de coûts ; « sain » doit rester une classe. |
| C. Détection d'anomalies non supervisée | images saines uniquement | robustesse aux types de défauts jamais vus | Features gelées (PatchCore) — peu à entraîner ; le seuil de fonctionnement exige *quand même* des défauts étiquetés en validation ; localisation des petits défauts plus difficile à basse résolution sur CPU. |

**Recommandation pour la v1 : option A.** C'est la seule option où chaque exigence du cadrage (seuillage à coût
asymétrique, choix de la perte, décision calibrée dans l'API) est exercée de bout en bout.
L'option C est la comparaison naturelle en v2 sur le *même* découpage KSDD2 (entraînement sur les images saines
uniquement, évaluation sur les mêmes jeux de validation/test), ce qui donnerait une comparaison supervisé /
non supervisé équitable.

Famille de modèle pour la v1 : **ResNet-18** pré-entraîné ImageNet avec une tête à un seul logit, fine-tuning
complet (`[DECISION D6]` ; EfficientNet-B0 / MobileNetV3-Large restent sélectionnables dans la config).
ResNet-18 est choisi pour son débit sur CPU et parce que son comportement est bien compris ; l'objectif de la v1 est
une *référence rigoureuse*, pas une entrée de classement.

Résolution d'entrée `[DECISION D7]` : conserver le ratio d'aspect natif (~640×224 après redimensionnement) plutôt
qu'écraser en 224×224 — les défauts sont petits, et un écrasement anisotrope les effacerait. La taille exacte est
fixée après que l'EDA a mesuré les dimensions réelles (elles varient légèrement d'une image KSDD2 à l'autre).

---

## 3. Découpage des données et contrôle des fuites

### 3.1 Protocole `[DECISION D3]`

```
train officiel (2 331)  ──découpage stratifié, seed 42──►  train (≈1 865)  +  validation (≈466, ≈49 défectueuses)
test officiel  (1 004)  ──mis en quarantaine────────────►  évalué UNE SEULE FOIS, à la toute fin, avec le seuil figé
```

- La **validation** sert à : l'arrêt anticipé, la sélection de modèle, les choix d'hyperparamètres *et le choix du
  seuil*.
- Le **test** sert à exactement une évaluation finale. Aucune décision n'est jamais prise à partir d'un chiffre du
  test. Si le résultat sur le test déçoit, c'est le résultat rapporté ; réajuster après l'avoir vu est une fuite.
- Alternative : validation croisée stratifiée 5-fold sur le train officiel. Pour : mutualise les 246 positifs pour
  le choix du seuil (prédictions out-of-fold). Contre : 5× le budget CPU.
  **Recommandation : hold-out pour la v1**, avec intervalles de confiance bootstrap pour rendre visible l'incertitude
  liée au petit échantillon ; le choix de seuil 5-fold OOF est une amélioration v2 documentée.

### 3.2 Risques de fuite propres à KSDD2 et parades

| # | Risque | Pourquoi c'est important ici | Parade (implémentée dans `src/vqi/data`) |
|---|---|---|---|
| L1 | Plusieurs images de la **même pièce physique** (KSDD v1 avait 8 faces par commutateur) | KSDD2 ne fournit pas d'identifiant de pièce ; si des faces d'une même pièce tombent en train *et* en validation, la validation est optimiste | Traiter comme *inconnu, pas absent* : **hachage perceptuel (pHash)** par paires sur toutes les images ; les paires sous un seuil de Hamming sont forcées dans le même split (mécanisme `group_column`, prêt aussi pour un vrai identifiant de pièce) |
| L2 | **Acquisitions consécutives** (identifiants de fichiers séquentiels, même session/éclairage) | Des identifiants voisins partagent les conditions d'acquisition → un découpage aléatoire laisse fuir des indices de « session » | Rapporter les métriques de validation sur le découpage aléatoire *et* sur un découpage par blocs contigus comme contrôle de robustesse (v2 si le budget CPU est court) |
| L3 | **Patches issus d'une même image mère** dans des splits différents | Sans objet en v1 (images entières), mais un futur tuilage le créerait | Le découpage se fait au niveau *identifiant d'image* et est stocké dans un manifeste ; tout tuilage doit hériter du split de l'image mère |
| L4 | **Doublons pixel-identiques** | Gonflent toutes les métriques | Déduplication par hash de fichier (SHA-256) à l'étape du manifeste |
| L5 | **Statistiques de prétraitement** calculées sur toutes les données | Normalisation / augmentation ajustées sur la validation ou le test | Uniquement les statistiques ImageNet (backbone pré-entraîné) ou des statistiques calculées sur le split train ; augmentation appliquée au train seul |
| L6 | **Seuil choisi sur le test** | La fuite silencieuse la plus fréquente derrière les « 99 % de rappel » | Le seuil est écrit dans `threshold.json` à partir de la validation *avant* que le script de test puisse tourner ; le script de test refuse de s'exécuter sans lui |
| L7 | **Fuite par les masques** | Les masques contiennent le label | Les masques ne sont jamais chargés par le `Dataset` de classification ; seul le module d'évaluation les lit (analyse par taille de défaut) |
| L8 | Labels du test publics | Tentation d'itérer sur le test | Évaluation finale unique, journalisée avec le hash du commit git et le hash de la config |

Les manifestes de découpage (`data/splits/*.csv` : identifiant d'image, label, split, sha256, phash) sont versionnés
dans Git : n'importe qui peut reproduire exactement la partition sans les données.

---

## 4. Métriques et politique de décision

### 4.1 Pourquoi l'exactitude est trompeuse

Avec ~11 % de pièces défectueuses, un modèle qui prédit « sain » pour tout atteint ~89 % d'exactitude tout en
manquant **100 % des défauts** — le pire résultat possible pour le métier. L'exactitude pondère les deux erreurs à
l'identique et est dominée par la classe majoritaire ; elle ne peut pas exprimer l'asymétrie.

Convention utilisée partout dans ce projet : **classe positive = défectueux**. Un faux négatif (FN) est une pièce
défectueuse expédiée ; un faux positif (FP) est une pièce saine envoyée en réinspection humaine.

### 4.2 Métriques primaires

| Métrique | Rôle | Pourquoi |
|---|---|---|
| **Rappel (taux de détection des défauts)** au point de fonctionnement | Métrique métier primaire | Mesure les échappées (FN), l'erreur coûteuse |
| **Précision** et **taux de faux positifs (FPR)** au point de fonctionnement | Coût de la politique | Charge de réinspection induite par le seuil choisi |
| **Average Precision (aire sous la courbe PR)** | Métrique primaire de sélection de modèle (indépendante du seuil) | Sous déséquilibre, la courbe PR est plus informative que la ROC (Saito & Rehmsmeier, 2015) ; c'est aussi la métrique rapportée par les auteurs de KSDD2 |
| ROC-AUC | Secondaire, pour comparabilité | Optimiste sous déséquilibre ; rapportée, pas optimisée |
| Table rappel → précision / FPR à rappel 0,90 / 0,95 / 0,98 / 0,99 | Communication avec les ingénieurs qualité | Répond à « combien coûte le fait d'attraper 99 % des défauts ? » |
| Calibration (diagramme de fiabilité, ECE, Brier) | Validité du raisonnement par seuil | Un seuil de probabilité n'a de sens que si les scores sont calibrés (Guo et al., 2017) |
| Rappel par taille de défaut (depuis les masques) | Analyse d'erreurs | Montre où sont les échappées (attendu : les plus petits défauts) |
| Intervalles de confiance bootstrap à 95 % sur tout ce qui précède | Honnêteté sur la taille d'échantillon | ~49 positifs en validation → intervalles larges ; ils sont rapportés, pas cachés |

Le F-bêta avec β = 2 est rapporté comme résumé en un chiffre mais n'est **pas** le critère de sélection : β est un
ratio de coût implicite, difficile à justifier.

### 4.3 Où entre l'asymétrie des coûts `[DECISION D4]` `[DECISION D5]`

Deux leviers existent, et ils ne sont pas interchangeables :

1. **La fonction de perte.** Pondérer la classe positive (`pos_weight` de la BCE) ou utiliser la focal loss change
   l'*équilibre des gradients* pendant l'entraînement. Cela traite surtout le déséquilibre de classes ; cela déforme
   aussi les probabilités de sortie (BCE pondérée ≡ décalage du prior), ce qui dégrade la calibration.
2. **Le seuil de décision.** Avec une probabilité calibrée p = P(défectueux | x), la théorie de la décision
   (Elkan, 2001) donne la règle optimale en coût : rejeter ssi p > C_FP / (C_FP + C_FN).
   Exemple : C_FN = 10 × C_FP → seuil ≈ 0,09 au lieu de 0,5.

**Recommandation :** entraîner avec une `BCEWithLogitsLoss` simple (`pos_weight = 1`) pour obtenir le meilleur
*classement* et les scores les plus *calibrés*, puis injecter l'asymétrie **au seuil**, choisi sur le jeu de
validation avec une politique explicite et configurable :

- `min_recall` (défaut) : plus petit seuil dont le rappel en validation ≥ cible (défaut 0,95 — voir le risque R1 sur
  le choix de ne pas viser 0,99 avec 49 positifs), puis rapporter la précision/le FPR résultants ;
- `expected_cost` : seuil minimisant C_FN·FN + C_FP·FP en validation, pour un ratio de coût énoncé ;
- `max_f_beta` : à titre de comparaison uniquement.

Pourquoi c'est le choix défendable : le ratio de coût est une *constante métier* qui peut changer ; avec l'approche
par seuil, on le change sans réentraîner, et le modèle reste un vrai classifieur probabiliste.
Une ablation `pos_weight` (1 / 3 / 9) est planifiée en phase d'évaluation pour que l'affirmation « la pondération
n'améliore pas le classement » soit *mesurée*, pas supposée.

### 4.4 Présentation du compromis

- Courbe précision–rappel sur la validation avec le point de fonctionnement marqué, et la courbe du test à côté
  (tracée une seule fois, à la fin).
- Table rappel → précision / FPR (4 cibles de rappel) sur la validation et le test.
- Courbe du coût attendu en fonction du seuil, pour le ratio de coût énoncé.
- Matrice de confusion au point de fonctionnement, avec IC bootstrap sur le rappel et la précision.
- Diagramme de fiabilité.
- Histogramme du rappel en fonction de l'aire du défaut.
- Planche Grad-CAM : détections, échappées, fausses alertes.

---

## 5. Décisions en attente de validation

| Id | Décision | Recommandation | Alternatives | Statut |
|---|---|---|---|---|
| D1 | Dataset | KolektorSDD2 | NEU-CLS (multi-classes, pas de classe saine), MVTec AD (non supervisé) | validé |
| D2 | Tâche v1 | Binaire supervisée accepter/rejeter | Détection d'anomalies non supervisée ; multi-classes | validé |
| D3 | Schéma de validation | Hold-out **unique** stratifié et group-aware, 20 % du train officiel, seed 42, IC bootstrap | 5-fold stratifié avec seuil OOF (5× le coût CPU) | validé |
| D4 | Perte | BCE simple (calibration d'abord), ablation `pos_weight` ensuite | BCE pondérée, focal loss, échantillonneur équilibré | validé |
| D5 | Politique de seuil | `min_recall`, cible 0,95 sur la validation | `expected_cost` avec un ratio de coût énoncé (exige un chiffre métier) | validé |
| D6 | Backbone | ResNet-18, pré-entraîné ImageNet, fine-tuning complet | EfficientNet-B0, MobileNetV3-Large ; sonde linéaire à backbone gelé comme référence bon marché | validé |
| D7 | Taille d'entrée | Ratio d'aspect natif, ≈ 640×224 (fixé après l'EDA) | Écrasement 224×224 (rejeté : petits défauts) | validé |
| D8 | Organisation du dépôt | Package `src/vqi` (src-layout, installable, testable) | Modules à plat à la racine | validé |
| D9 | Langue de la documentation | Français (code, identifiants, commits en anglais) | Anglais | **validé** |
| D10 | Interpréteur | Python 3.11.9 installé en local (parité avec l'image Docker) | 3.12 en local, 3.11 dans Docker (risque de disparité) | validé |

---

## 6. Registre des risques méthodologiques

| Id | Risque | Impact | Parade |
|---|---|---|---|
| R1 | Seulement ~49 images défectueuses en validation | Une cible de rappel 0,98 signifie « rater au plus 1 » ; le seuil est fixé par une seule image → instable | Cible 0,95 en v1 ; IC bootstrap rapportés ; v2 : mutualisation 5-fold OOF des 246 positifs |
| R2 | Labels du test publics | Réglage silencieux sur le test | Évaluation finale unique, conditionnée à un `threshold.json` figé, journalisée avec le hash du commit |
| R3 | Structure multi-vues / quasi-doublons inconnue | Validation optimiste | Déduplication pHash + SHA-256, politique « même split » (L1, L4) |
| R4 | Dérive entraînement/serving | L'API prétraite silencieusement autrement que l'entraînement | Un seul module de prétraitement utilisé par les deux ; un test de contrat vérifie l'égalité des tenseurs sur une image fixe |
| R5 | Budget temps CPU | Entraînement trop lent pour itérer | Dry-run d'une epoch chronométré avant de fixer le planning ; backbone plus petit / couches gelées en repli ; estimation : `TBD` min/epoch |
| R6 | Labels dérivés des masques ; certains défauts font quelques pixels | Plafond dur sur le rappel à FPR raisonnable ; bruit de label si les masques sont imparfaits | Analyse du rappel par taille de défaut ; rapporté comme un constat, pas de ré-étiquetage |
| R7 | Les réseaux de neurones sont sur-confiants | Un seuil `expected_cost` sur des scores mal calibrés est faux | Diagramme de fiabilité ; temperature scaling sur la validation si l'ECE est élevé (`TBD`) |
| R8 | Décalage de distribution entre train et test officiels | Validation ≠ test | Comparer les deux ; un écart est *rapporté*, jamais corrigé en touchant au test |
| R9 | Non-déterminisme résiduel sur CPU (ordonnancement des threads, certaines opérations) | Les runs diffèrent légèrement | Tout seeder, `torch.use_deterministic_algorithms`, workers de DataLoader seedés ; documenter la variance résiduelle sur 3 seeds si le budget le permet |
| R10 | Augmentations qui changent le label | Un flip est sûr ; un recadrage peut retirer le défaut | Uniquement des augmentations plein cadre qui préservent le label en v1 (flips, rotations de 90° si l'EDA confirme l'invariance à l'orientation, luminosité/contraste légers) |

---

## 7. Plan par phases

| Phase | Livrable | Critère de sortie |
|---|---|---|
| 0 | Ce document, schéma du pipeline, config, squelette du dépôt | Décisions D1–D10 validées |
| 1 | Environnement, chargeur de config, logging, seeding, téléchargement + vérification d'intégrité, manifeste, découpages, notebook EDA | `make_splits` reproductible ; l'EDA remplit les `TBD` (taille d'image, mode couleur, seuil pHash) |
| 2 | Prétraitement + `Dataset`/`DataLoader`, tests de contrat | Tenseurs identiques entre chemins d'entraînement et d'inférence |
| 3 | Fabrique de modèles, boucle d'entraînement, checkpoints, dry-run chronométré d'une epoch | Un entraînement complet avec courbes journalisées |
| 4 | Évaluation : métriques, politique de seuil, figures, analyse d'erreurs, évaluation finale sur le test | Rapport avec chiffres réels et IC |
| 5 | Module d'inférence, service FastAPI, Dockerfile, tests d'API | `docker run` → `POST /predict` renvoie une décision |
| 6 | README avec résultats, journal des décisions, notes de synthèse | Chaque chiffre traçable jusqu'à un run id |

---

## Références

- Božič, J., Tabernik, D., Skočaj, D. (2021). *Mixed supervision for surface-defect detection: from weakly to fully supervised learning.* Computers in Industry. (KolektorSDD2)
- Tabernik, D., et al. (2020). *Segmentation-based deep-learning approach for surface-defect detection.* J. Intell. Manuf. (KolektorSDD)
- Bergmann, P., et al. (2019). *MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection.* CVPR.
- Song, K., Yan, Y. (2013). *A noise robust method based on completed local binary patterns for hot-rolled steel strip surface defects.* Applied Surface Science. (NEU)
- Elkan, C. (2001). *The Foundations of Cost-Sensitive Learning.* IJCAI.
- Saito, T., Rehmsmeier, M. (2015). *The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets.* PLoS ONE.
- Guo, C., et al. (2017). *On Calibration of Modern Neural Networks.* ICML.
- Kaufman, S., et al. (2012). *Leakage in Data Mining: Formulation, Detection, and Avoidance.* ACM TKDD.
- Roth, K., et al. (2022). *Towards Total Recall in Industrial Anomaly Detection.* CVPR. (PatchCore, candidat v2)
