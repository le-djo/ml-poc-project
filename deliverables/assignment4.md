# Assignment 4 — Visualisations

**Rendu :** push sur le dépôt git  
**Figures :** 5 fichiers dans `deliverables/fig_*.png`  
**Reproduction :**

```bash
python scripts/build_features.py   # prérequis : data/Processed/processed.parquet
jupyter nbconvert --to notebook --execute --inplace notebooks/02_modeling.ipynb
```

Les 5 figures sont générées dans l'ordre par `notebooks/02_modeling.ipynb` (cellules 7–8 pour les figures de performance, cellules ajoutées pour les figures de données et de features).

---

## Visualisation 1 — Données brutes : spikes de volume autour de pump_ts

**Figure :** `deliverables/fig_raw_klines.png`

### Objectif
Montrer la diversité des événements pump dans les klines brutes et vérifier visuellement si le signal de volume précède ou coïncide avec `pump_ts`.

### Choix du graphique
Bar chart temporel, axe X en minutes relatives à `pump_ts`. Le volume est une variable de flux — les barres encodent l'intensité à chaque minute, la ligne verticale rouge matérialise `pump_ts`. Les barres pré-pump sont en bleu, post-pump en rouge, pour distinguer immédiatement les deux phases.

### Événements sélectionnés

| Événement | Symbole | vol_zscore_60m | Critère |
|-----------|---------|---------------|---------|
| Pump fort | DUSKBTC (oct. 2020) | 10.91 | Maximum global |
| Pump aléatoire | TNTBTC (juin 2018) | 0.08 | Tirage aléatoire |
| Pump faible | MODBTC (fév. 2018) | −0.72 | Minimum global |

### Interprétation

Les trois événements présentent le même schéma : **volume pré-pump quasi nul sur toute la fenêtre [−120, 0)**, spike massif survenant après `pump_ts`. Cette observation est uniforme quelle que soit la valeur de `vol_zscore_60m`.

**Pump fort (DUSK, vol_zscore=10,9) :** même avec le z-score maximal du dataset, la fenêtre pré-pump reste visuellement plate. Le `vol_zscore_60m` élevé reflète une élévation relative du dernier bar par rapport à un fond lui-même très faible — pas un pic d'accumulation détectable à l'œil.

**Pump aléatoire (TNT) et pump faible (MOD) :** schéma identique — silence pré-pump complet, explosion post-annonce. Les ratios post/pré observés (29× et 38× respectivement) confirment que l'intégralité de l'activité est déclenchée par l'annonce Telegram, pas anticipée.

**Interprétation structurelle :** les pumps coordonnés de 2018–2020 sur Binance ne laissent pas de signal d'accumulation détectable dans les klines 1-min avant l'annonce. La coordination se fait via des groupes privés (Telegram, Discord) avec diffusion quasi-simultanée — le délai entre la décision d'achat et l'annonce est inférieur à la résolution d'une barre 1-min.

### Pertinence
Cette observation est la **cause racine directe du Recall faible** à basse contamination (Recall@0,001 ≈ 0,003 pour tous les modèles) : ce n'est pas une défaillance des modèles, c'est une limite structurelle du signal disponible dans les klines 1-min. Les features pré-pump capturent un bruit de fond, pas un signal d'accumulation. Cela invalide l'hypothèse initiale selon laquelle les pumps seraient précédés d'une phase d'accumulation détectable à cette résolution temporelle.

---

## Visualisation 2 — Après feature engineering : corrélations entre features

**Figure :** `deliverables/fig_feature_heatmap.png`

### Objectif
Vérifier l'absence de multicollinéarité excessive, identifier les features redondantes, et guider le choix du modèle de détection d'anomalies.

### Choix du graphique
Heatmap de corrélation de Pearson (n=666). La matrice symétrique permet de lire simultanément toutes les paires ; la palette `coolwarm` encode direction et intensité ; les annotations numériques permettent une lecture exacte. Le format carré préserve la symétrie visuelle.

### Interprétation

Deux clusters distincts émergent :

**Cluster 1 — Momentum de prix** (corrélations r ≥ 0,90) :

| Paire | r |
|-------|---|
| ret_5m / ret_15m | 0,97 |
| ret_5m / price_impact_1m | 0,95 |
| ret_15m / ret_60m | 0,94 |
| ret_15m / price_impact_1m | 0,92 |
| ret_5m / ret_60m | 0,91 |
| std_ret_60m / ret_60m | 0,90 |

`ret_5m`, `ret_15m`, `ret_60m`, `price_impact_1m` et `std_ret_60m` mesurent tous des aspects du même phénomène : le mouvement de prix directionnnel dans les minutes précédant le pump. Cette redondance partielle n'est pas problématique pour l'Isolation Forest (robuste à la corrélation), mais elle distord les distances euclidiennes utilisées par LOF.

**Cluster 2 — Microstructure / activité** :

`vol_zscore_60m`, `ofi_proxy_1m`, `taker_buy_ratio_5m`, `conviction_ratio_1m` et `rush_order_count` présentent des corrélations faibles à modérées entre eux et avec le cluster 1. Ce sont des features orthogonales apportant un signal indépendant.

### Pertinence
La forte corrélation interne du cluster 1 explique partiellement la sous-performance du LOF (AUC=0,710) : les distances dans $\mathbb{R}^{10}$ sont dominées par les 5 features corrélées de momentum, compressant artificiellement les voisinages. L'Isolation Forest, qui partitionne chaque feature indépendamment, est immunisé contre cet effet — ce qui justifie son AUC supérieur (0,881).

---

## Visualisation 3 — Performances : distribution des scores d'anomalie

**Figure :** `deliverables/fig_score_distributions.png`

### Objectif
Visualiser la séparabilité des scores entre fenêtres contrôle (gt=0, heure calme) et fenêtres pré-pump (gt=1, heure d'accumulation) pour chaque modèle.

### Choix du graphique
Violin plot par modèle, deux violons côte à côte (gt=0 en bleu, gt=1 en rouge). Montre la distribution complète — forme, queue, bimodalité — là où une boxplot masquerait la structure interne.

### Interprétation

**Isolation Forest :** gt=0 concentré autour de −0,10 (distribution étroite), gt=1 étalé vers les valeurs positives avec une queue haute visible. Séparation partielle : les événements à signal fort (DUSK-like) s'isolent clairement, les événements faibles se mélangent avec le contrôle. Cohérent avec AUC=0,881.

**LOF :** distributions fortement chevauchantes sur toute la plage (≈1,0 à 4,5+). Le LOF n'arrive pas à distinguer les deux classes — les voisinages locaux sont perturbés par la multicollinéarité du cluster de momentum. Cohérent avec AUC=0,710.

**Z-score (vol_zscore_60m) :** gt=0 très concentré autour de 0 (distribution étroite — le volume pré-pump de la fenêtre calme est dans sa propre norme), gt=1 très étalé de 0 à +7 avec structure bimodale. La bimodalité reflète exactement les deux types d'événements vus dans fig_raw_klines : pumps avec accumulation préalable (queue haute) et pumps sans signal pré-pump (pic à 0). Cohérent avec AUC=0,766.

### Pertinence
La figure explique directement l'ordre AUC : IF (0,881) > Z-score (0,766) > LOF (0,710). L'IF bénéficie des 10 dimensions pour isoler les événements ; le Z-score, univarié, capture les cas extrêmes ; le LOF perd de l'information à cause de la structure de corrélation.

---

## Visualisation 4 — Performances : courbes ROC

**Figure :** `deliverables/fig_roc_comparison.png`

### Objectif
Comparer le pouvoir discriminant des 3 modèles indépendamment du seuil de contamination, sur toute la gamme de seuils possibles.

### Choix du graphique
Courbes ROC superposées sur les mêmes axes, diagonale en pointillés (baseline aléatoire AUC=0,5). La surface sous la courbe (AUC) résume la performance en un scalaire ; les courbes révèlent le comportement à différents points de fonctionnement.

### Interprétation

**IF (AUC=0,881) :** monte rapidement vers le coin supérieur gauche aux faibles FPR — à 5 % de faux positifs, l'IF capture déjà ~60 % des vrais pumps. C'est le comportement souhaité pour une alerte à haute précision.

**Z-score (AUC=0,766) :** progression plus régulière, intermédiaire sur tout le spectre. Meilleur que LOF sur tous les seuils, mais gagnant en recall aux hauts FPR grâce aux événements à vol_zscore extrême.

**LOF (AUC=0,710) :** courbe proche de la diagonale — le modèle est à peine au-dessus du hasard. La géométrie locale dans $\mathbb{R}^{10}$ n'est pas discriminante sur ce feature set à 3 666 observations.

### Pertinence
Justifie le choix de l'Isolation Forest comme modèle recommandé pour un déploiement Kaiko : meilleur AUC et comportement favorable aux faibles contaminations, là où une alerte de surveillance doit être précise.

---

## Visualisation 5 — Performances : Recall@LaMargia vs contamination

**Figure :** `deliverables/fig_recall_vs_contamination.png`

### Objectif
Analyser la sensibilité des 3 modèles au paramètre de contamination — critique car la prévalence réelle des pumps en production est inconnue (0,054 % dans La Morgia, potentiellement différente en temps réel).

### Choix du graphique
Axe X logarithmique : la contamination varie sur 2 ordres de grandeur (0,0005 à 0,05). L'échelle log révèle le comportement aux faibles valeurs — là où le déploiement opère réellement. Les points marquent les valeurs testées ({0,0005, 0,001, 0,005, 0,01, 0,05}).

### Interprétation

**À c=0,001 (ligne rose) :** tous les modèles atteignent Recall ≈ 0,003 — 1 seul événement flaggé sur 333. Comportement attendu : 0,001 × 666 = 0,66 flag, arrondi à 1. Le recall reflète la contrainte arithmétique, pas une défaillance des modèles.

**À c=0,005 :** IF et Z-score montent à 0,012 (4 événements sur 333), LOF à 0,006. La différence commence à se creuser.

**À c=0,05 :** Z-score atteint Recall=0,10 (33 événements), IF suit de près. Le Z-score surpasse l'IF en recall à haute contamination malgré un AUC inférieur : son score univarié est bien calibré pour les cas extrêmes (vol_zscore_60m très élevé = pump évident), mais ne discrimine pas les cas intermédiaires.

**Observation principale :** aucun modèle n'atteint un recall élevé à faible contamination. Ce n'est pas une défaillance des modèles — c'est la conséquence directe de l'absence de signal pré-pump dans les klines 1-min, confirmée visuellement dans fig_raw_klines.png : les pumps 2018–2020 ne laissent pas de trace d'accumulation détectable avant l'annonce Telegram. Ce constat contraste avec l'évaluation initiale sur La Morgia (AUC=0,9976) documentée comme artefact dans `deliverables/assignment2.md` — les features La Morgia étaient calculées *pendant* le pump, rendant le problème trivialement séparable.

### Pertinence
La courbe de recall en fonction de la contamination est l'outil de calibration pour le déploiement : elle permet de choisir le point de fonctionnement (seuil de contamination) selon le coût relatif des faux positifs et des faux négatifs dans le contexte Kaiko.

---

## Localisation et reproduction

| Figure | Fichier | Génération |
|--------|---------|-----------|
| Données brutes | `deliverables/fig_raw_klines.png` | Cellule ajoutée dans `02_modeling.ipynb` |
| Corrélation features | `deliverables/fig_feature_heatmap.png` | Cellule ajoutée dans `02_modeling.ipynb` |
| Distributions scores | `deliverables/fig_score_distributions.png` | Cellule 7 de `02_modeling.ipynb` |
| Courbes ROC | `deliverables/fig_roc_comparison.png` | Cellule 7 de `02_modeling.ipynb` |
| Recall vs contamination | `deliverables/fig_recall_vs_contamination.png` | Cellule 7 de `02_modeling.ipynb` |

**Commande de reproduction complète :**

```bash
python scripts/build_features.py
jupyter nbconvert --to notebook --execute --inplace notebooks/02_modeling.ipynb
```

Les figures sont écrasées à chaque exécution — résultats déterministes (`random_state=42` pour IF).
