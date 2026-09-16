# Phase 1 — Ingestion et exploration des données

**État : livrée le 15/09/2026.** Le rapport chiffré est généré automatiquement dans
[`reports/eda/eda_report.md`](../reports/eda/eda_report.md) ; ce document décrit la *conception*
de la chaîne d'ingestion et les décisions qu'elle applique. Aucun chiffre ici n'est saisi à la main :
ceux qui sont cités renvoient au rapport.

## 1. Chaîne d'ingestion

```
scripts/download_data.py   →  data/raw/KolektorSDD2.zip  (SHA-256 vérifié)  →  data/raw/kolektor_sdd2/{train,test}
scripts/build_manifest.py  →  data/interim/manifest.csv + manifest.meta.json
                               data/interim/duplicates_report.json + phash_nearest_neighbours.csv
scripts/make_splits.py     →  data/splits/v1.csv + v1.meta.json        (versionnés dans git)
scripts/run_eda.py         →  reports/eda/eda_report.md + tables CSV + figures/
```

Chaque script prend `--config configs/default.yaml` (+ `--override` répétable), configure la
journalisation JSON, seede toutes les sources d'aléa, puis écrit un `*.meta.json` avec la
provenance : date, version du package, commit git, hash de la config, hash SHA-256 de l'artefact.

### 1.1 Principe : `data/raw/` est immuable

Rien dans le pipeline n'écrit ni ne supprime dans `data/raw/`. Les images corrompues, dupliquées
ou hors convention sont **marquées dans le manifeste** (`exclude_reason`) et ignorées par le
découpage. On peut donc toujours revenir aux octets d'origine, et la décision d'exclusion est
lisible et réversible.

### 1.2 Le manifeste est la source de vérité

Une ligne par fichier image, avec : identifiant, split officiel, conformité du nom, index
d'acquisition, chemins image/masque, **label dérivé du masque** (`mask_non_empty`), aire et boîte
englobante du défaut, dimensions, canaux, écart entre canaux, luminance, contraste, SHA-256, pHash,
indicateurs de corruption. Le masque n'est lu **que** pour produire le label et ces statistiques ;
le `Dataset` d'entraînement (Phase 2) ne l'ouvrira jamais (garde-fou L7).

### 1.3 Contrôle des effectifs attendus

`dataset.expected_counts` (article + page officielle) est confronté au manifeste. En cas d'écart,
`build_manifest` écrit quand même le manifeste (c'est un inventaire) mais `make_splits` **refuse de
tourner** : un découpage bâti sur un dataset qui ne correspond pas à sa description n'est pas une
connaissance reproductible. Ce contrôle a servi dès le premier run (voir §3).

## 2. Doublons : politique appliquée

| Cas | Détection | Décision |
|---|---|---|
| Fichier hors convention `^[0-9]+$` (ex. `10301 (copy).png`) | expression régulière | inventorié, `exclude_reason = nonconforming_filename`, jamais supprimé |
| Masque orphelin (ex. `10301_GT (copy).png`) | non rattaché à une image conforme | listé dans `manifest.meta.json` avec son SHA-256 |
| Doublon exact intra-train | SHA-256 identique | une copie conservée, les autres exclues |
| Doublon exact train ↔ test | SHA-256 identique | la copie **train** est exclue ; le test officiel reste intact |
| Doublon exact intra-test | SHA-256 identique | signalé, conservé (comparabilité avec la littérature) |
| Quasi-doublon (pHash) | distance de Hamming ≤ `max_hamming_distance` | **rien tant que le seuil est `null`** ; l'EDA produit la distribution des distances et une table de sensibilité pour décider |

Le pHash est implémenté avec OpenCV (redimensionnement 32×32 en niveaux de gris, DCT, signe des
8×8 coefficients basse fréquence par rapport à leur médiane → 64 bits). Les composantes connexes de
la relation « est un doublon de » forment des `group_id` que le découpage ne sépare jamais.

## 3. Ce que la Phase 1 a révélé

- **L'archive officielle contient une copie accidentelle** : `train/10301 (copy).png` et son masque,
  byte-identiques à `10301`. C'est la cause de l'écart entre les 2 332 fichiers image du dossier
  `train/` et les 2 331 annoncés. Une fois la copie exclue, les effectifs attendus sont
  **exactement confirmés** (2 331 / 246 ; 1 004 / 110).
- **Aucune image corrompue** ; masques strictement binaires (0/255).
- **Pas de population de quasi-doublons** : la distribution des distances pHash au plus proche
  voisin est unimodale (mode à 14 bits) ; les deux paires à 8 bits, inspectées visuellement, sont
  des pièces différentes. Proposition : laisser `max_hamming_distance: null` (ou 8, sans effet
  mesurable) — à valider.
- **Images quasi monochromes** (écart entre canaux ≈ 3,8/255), sombres (luminance ≈ 44/255), de
  dimensions variables (≈ 637×229 en médiane). La taille d'entrée 640×224 est à l'échelle native.
- **Un signal trivial existe** : le contraste global seul donne une AUC ≈ 0,72, l'écart entre canaux
  ≈ 0,64. Ce sont des références *à battre largement* en Phase 4, et un rappel que le modèle doit
  être évalué contre autre chose que le hasard.
- **Les défectueuses sont réparties sur toute la séquence d'acquisition** (taux glissant entre
  ≈ 4 % et ≈ 17 % en train) : le découpage aléatoire stratifié est défendable, le split par blocs
  reste un contrôle possible en v2.
- **Les défauts sont petits** : aire médiane ≈ 2 300 px (≈ 1,6 % de l'image), minimum 23 px.
  Les quantiles du rapport définissent les strates de l'analyse « rappel par taille » (Phase 4).

## 4. Découpage (décision D3, implémentée)

- Test = test officiel, intact, `split == "test"`, évalué une seule fois en fin de projet.
- Validation = hold-out **unique** de 20 % du train officiel, stratifié sur le label et respectant
  les `group_id`. Mécanisme : un seul pli d'un `StratifiedGroupKFold(k = 1 / val_fraction = 5,
  shuffle=True, random_state=seed)` ; les quatre autres plis ne sont jamais utilisés — ce n'est pas
  une validation croisée.
- Invariants vérifiés à l'écriture et enregistrés dans `v1.meta.json` : aucun groupe à cheval sur
  train/val, test = test officiel, train/val issus du train officiel seulement, identifiants uniques.
- Le CSV est versionné : n'importe qui peut reconstruire la partition sans les données, et la
  relancer avec la même seed produit un fichier byte-identique (testé).

Effectifs obtenus : voir §9 du rapport EDA (train 1 865 dont 197 défectueuses ; val 466 dont 49 ;
test 1 004 dont 110 ; 1 exclue).

## 5. Déséquilibre des classes — options présentées, aucune tranchée

Le rapport (§2) chiffre chaque option sur les effectifs réels : BCE simple + seuil, `pos_weight`
plein (≈ 8,5) ou doux (≈ 2,9), suréchantillonnage, `WeightedRandomSampler`, focal loss — avec
leur effet sur le gradient et sur la calibration. Rappel de cadrage : le levier des coûts reste le
seuil de décision ; ces options agissent sur l'apprentissage. La décision est prise en Phase 3 et
mesurée par ablation en Phase 4.

## 6. Tests

`pytest` (18 tests) couvre : le chargeur de config (clés inconnues, types, fusion, hash), le
hachage (SHA-256, pHash déterministe, proche pour des images quasi identiques, éloigné sinon,
matrice symétrique), et la chaîne complète sur un **dataset synthétique** reproduisant les cas
réels : copie « (copy) » + masque orphelin, doublon intra-train, doublon train ↔ test, labels
dérivés des masques, contrôle des effectifs, invariants du découpage, reproductibilité (même seed →
même CSV ; seed différente → partition différente), refus d'une fraction non 1/k, génération du
rapport et des figures.

## 7. Commandes

```bash
py -3.11 -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt && pip install -e ".[dev]"
python scripts/download_data.py
python scripts/build_manifest.py
python scripts/make_splits.py
python scripts/run_eda.py
pytest
```

## 8. Décisions ouvertes à l'issue de la Phase 1

| Id | Décision | Élément de preuve | Proposition |
|---|---|---|---|
| P1-1 | `split.duplicates.max_hamming_distance` | rapport §5 | `null` — **validé** (aucun quasi-doublon retenu) |
| P1-2 | `preprocessing.color_mode` | rapport §3 | `rgb` — **validé** |
| P1-3 | `augmentation.rotate_90_p` | planche de contact (pièces en portrait, défauts sans orientation privilégiée) | à valider : 0 en v1, flips seulement |
| P1-4 | stratégie de déséquilibre | rapport §2 | reportée à la Phase 3 |
