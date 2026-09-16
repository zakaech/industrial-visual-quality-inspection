# X0 — Dry-run CPU (mesures réelles)

Machine : Windows-10-10.0.26200-SP0 · Intel64 Family 6 Model 142 Stepping 12, GenuineIntel · torch 2.14.0+cpu · 4 threads · Python 3.11.9
Données : train 1865 (197 positives), val 466 · test chargé : non

## 1. Cache des canevas (sortie exacte du Preprocessor)

| split | n | Mo | construit maintenant | s total | ms/image | vérifiés vs live | écarts | diff max |
|---|---|---|---|---|---|---|---|---|
| train | 1865 | 765 | oui | 26,1 | 14,0 | 32 | 0 | 0 |
| val | 466 | 191 | oui | 5,9 | 12,7 | 32 | 0 | 0 |

## 2. Chargement (train, augmentation ON, batch 16)

| cache | num_workers | 1er batch (s) | régime (ms/batch) | batchs mesurés |
|---|---|---|---|---|
| oui | 0 | 0,15 | 213,3 | 8 |
| oui | 2 | 0,24 | 108,0 | 8 |
| oui | 4 | 0,33 | 54,1 | 8 |
| non | 0 | 0,33 | 360,6 | 8 |
| non | 2 | 0,36 | 161,7 | 8 |

## 3. Calcul (s par pas, batch complet)

| batch | forward eval | A tête (live) | B layer4 | C layer3+4 | tout dégelé | RSS pic (Mo) |
|---|---|---|---|---|---|---|
| 8 | 1,133 | 1,109 | 1,514 | 1,944 | 3,622 | n/d |
| 16 | 2,294 | 2,331 | 3,051 | 3,822 | 6,852 | n/d |
| 32 | 4,336 | 4,031 | 5,346 | 6,667 | 12,275 | n/d |

## 4. Estimation d'époque dérivée des mesures (passe train + passe val)

| batch | profil | train (s) | val (s) | époque (min) |
|---|---|---|---|---|
| 8 | A_head_only_live | 260 | 67 | 5,4 |
| 8 | B_layer4 | 354 | 67 | 7,0 |
| 8 | C_layer3_4 | 455 | 67 | 8,7 |
| 8 | full_unfrozen | 848 | 67 | 15,2 |
| 16 | A_head_only_live | 273 | 69 | 5,7 |
| 16 | B_layer4 | 357 | 69 | 7,1 |
| 16 | C_layer3_4 | 447 | 69 | 8,6 |
| 16 | full_unfrozen | 802 | 69 | 14,5 |
| 32 | A_head_only_live | 238 | 65 | 5,0 |
| 32 | B_layer4 | 315 | 65 | 6,3 |
| 32 | C_layer3_4 | 393 | 65 | 7,6 |
| 32 | full_unfrozen | 724 | 65 | 13,2 |

## 5. Déterminisme (deux runs identiques, recette B, 20 pas, 4 threads)

- bit-identiques : oui · écart max des pertes : 0.000e+00 · écart max des poids de la tête : 0.000e+00

## 6. Saturation photométrique après augmentation (200 canevas train)

- fraction de pixels saturés (0 ou 255) : avant 0.00002, après 0.00002 (max 0.00344) ; décalage max de l'intensité moyenne : 5,1/255

RSS de pointe du processus : n/d Mo.
