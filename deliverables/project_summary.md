# Pump & Dump Detection — Résumé complet du projet

---

## 1. Le problème

### Qu'est-ce qu'un pump & dump ?

Un pump & dump est une manipulation de marché coordonnée : un groupe d'individus — typiquement via un canal Telegram privé — se met d'accord pour acheter massivement un actif crypto au même instant. Ce rush d'achat fait monter le cours artificiellement (le "pump"). Les organisateurs vendent alors leurs positions au sommet, faisant chuter le cours (le "dump"), tandis que les retardataires qui ont acheté au pic se retrouvent piégés avec des pertes sèches.

Sur Binance entre 2018 et 2021, ces schémas étaient documentés à grande échelle : La Morgia et al. (2020) en ont recensé 317 en quelques mois sur une seule plateforme. Le signal caractéristique est un spike de volume et de prix sur une fenêtre de secondes à quelques minutes, déclenché à la seconde de l'annonce.

### Pourquoi c'est difficile à détecter automatiquement

La difficulté principale est **temporelle** : le signal d'achat coordonné arrive en même temps que (ou après) l'annonce Telegram, pas avant. Dans les données de marché à 1 minute, il est souvent impossible de distinguer l'accumulation pré-pump d'un mouvement ordinaire. Sur nos 333 événements analysés, la grande majorité ne laisse aucune trace détectable dans les klines avant l'annonce.

Deuxième difficulté : **l'hétérogénéité**. Les pumps ciblent des centaines de coins différents, avec des profils de volume, des capitalisations et des liquidités très variés. Un modèle doit généraliser à des actifs qu'il n'a jamais vus.

### Pourquoi l'approche non supervisée et pas supervisée

Un classifieur supervisé nécessite des labels au moment de l'inférence. En production, ces labels n'existent pas — on ne sait pas qu'un pump va se produire avant qu'il se produise. De plus, les pumps *privés* (Discord, groupes fermés) ne sont jamais répertoriés dans les bases publiques comme La Morgia : un modèle supervisé entraîné sur ces labels ne les détectera pas.

L'approche non supervisée est différente : le modèle apprend ce qu'est un marché **normal**, puis produit un score d'anomalie pour chaque nouvelle observation. Les labels ne servent qu'à évaluer le modèle après coup — ils ne sont jamais passés à `fit()`. Ce protocole est conforme au déploiement réel.

### Contexte réglementaire MiCA

Le règlement européen *Markets in Crypto-Assets* (MiCA, entré en vigueur en 2024) impose aux plateformes crypto une **surveillance active des manipulations de marché**, sous peine d'amendes pouvant atteindre €5M ou 3% du chiffre d'affaires annuel. Kaiko, fournisseur de données de marché institutionnelles, couvre 150+ exchanges et est directement concerné par cette exigence. Ce projet s'inscrit dans ce contexte : construire un système de détection automatisée opérationnel sur les données Binance.

---

## 2. Les données

### Source des événements : La Morgia et al. 2020

Le point de départ est le dataset de La Morgia et al. (2020), qui recense 317 événements pump & dump documentés sur Binance entre 2018 et 2020. Chaque ligne contient : la date, l'heure approximative, le symbole (ex. `BRDBTC`), et l'exchange. On filtre sur Binance : **520 événements** dans le fichier source (`data/RAW/pump_telegram.csv`), dont **333 ont produit des données klines suffisantes** (≥ 30 barres dans la fenêtre pré-pump).

### Ce qu'on a collecté : klines 1-min via l'API Binance

Pour chaque événement, on récupère les klines 1-minute sur la fenêtre `[pump_ts − 2h, pump_ts + 30min]` via l'endpoint REST Binance `GET /api/v3/klines`. Chaque fichier parquet contient ~150 lignes × 12 colonnes (OHLCV + taker_buy_base + num_trades, etc.).

**Résultat de la collecte :** 333 fichiers parquet dans `data/RAW/klines/`, soit ~50k barres au total.

### Ce qu'on n'a PAS réussi à collecter et pourquoi

**aggTrades (données tick par tick) :** deux approches tentées :
1. API REST Binance `GET /aggTrades` — retourne 0 lignes pour les coins délistés après leur pump. Binance purge ces données.
2. Fichiers historiques `data.binance.vision` — retourne HTTP 404 pour les mêmes coins sur les mêmes dates. Confirmé sur toutes les paires testées.

**Cause :** les coins ciblés par les pumps 2018-2020 étaient majoritairement des micro-caps qui ont depuis été délistés de Binance. Binance ne conserve pas les données aggTrades pour les symboles délistés. Cette limitation est définitive.

**CoinGecko (market cap, rang) :** l'API CoinGecko ne renvoie pas de données historiques de rang pour des coins délistés sur la période 2018-2020. Fonctionnel pour BTC/ETH, inutilisable pour les micro-caps.

**Décision :** l'OFI (Order Flow Imbalance) est proxifié depuis les klines (`taker_buy_base` / `volume`), qui est disponible même pour les coins délistés. Cette approximation est documentée comme limitation.

### Dataset final

| Composante | Lignes | Labels |
|---|---|---|
| Fenêtres pré-pump (333 événements La Morgia) | 333 | gt=1 |
| Fenêtres marché normal (BTC/ETH/BNB/XRP/LTC, 2018-2021) | 3 333 | gt=0 |
| **Total** | **3 666** | **9,1% positifs** |

Les fenêtres marché normal ont été collectées via l'API Binance sur des dates aléatoires — c'est la correction H1 décrite en section 5.2.

---

## 3. Les features

### 3.1 Features collectées (Binance klines brutes)

Ces features sont calculées directement à partir des colonnes OHLCV retournées par l'API Binance. Aucune donnée externe n'est nécessaire.

| Feature | Formule | Interprétation |
|---|---|---|
| `ret_5m` | ln(close₋₁ / close₋₆) | Log-return sur les 5 dernières minutes. Capture le momentum court terme. |
| `ret_15m` | ln(close₋₁ / close₋₁₆) | Log-return sur 15 minutes. Momentum moyen terme. |
| `ret_60m` | ln(close₋₁ / close₋₆₁) | Log-return sur 60 minutes. Momentum long terme. Imputé à 0 pour les fenêtres gt=0 (60 barres insuffisantes). |
| `std_ret_60m` | σ(ln(closeₜ / closeₜ₋₁)) sur 60 min | Écart-type des log-returns minute. Mesure la volatilité réalisée. Un pump crée un spike de volatilité. |
| `vol_zscore_60m` | (vol₋₁ − μ_vol) / σ_vol | Z-score du volume de la dernière barre par rapport à la baseline de la fenêtre. Détecte les anomalies de volume instantanées. |

### 3.2 Features construites (feature engineering)

Ces features n'existent pas dans les données brutes. Elles ont été construites à partir des colonnes klines par analogie avec la littérature de microstructure.

| Feature | Formule | Source | Justification |
|---|---|---|---|
| `taker_buy_ratio_5m` | Σ taker_buy_base₋₅ / Σ volume₋₅ | — | Mesure la fraction des 5 dernières minutes dominée par des ordres d'achat agressifs (market orders). Un pump coordonné génère un ratio proche de 1. |
| `ofi_proxy_1m` | (2 × taker_buy_base₋₁ − volume₋₁) / (volume₋₁ + ε) | Cont et al. (2014) — *Price impact of order flow imbalance* | Proxy du déséquilibre d'ordre : +1 = achat pur, −1 = vente pure. Approxime l'OFI exact (non disponible depuis aggTrades délistés). |
| `price_impact_1m` | \|ln(close₋₁ / open₋₁)\| | Amihud (1986) — *Illiquidity and stock returns* | Ratio d'Amihud simplifié : retour log absolu de la dernière bougie. Mesure l'impact prix unitaire, indépendant de l'échelle du cours. |
| `conviction_ratio_1m` | \|close₋₁ − open₋₁\| / (high₋₁ − low₋₁ + ε) | — | Fraction de l'amplitude de la bougie capturée par le mouvement net. Proche de 1 = bougie directionnelle forte (tous les trades dans le même sens). |
| `rush_order_count` | Σ num_trades₋₂ | — | Nombre de trades sur les 2 dernières minutes. Proxy de l'intensité d'activité, corrélé aux rafales d'ordres coordonnés. |

### 3.3 Features testées et rejetées

**`vol_zscore_5m`, `vol_zscore_15m`, `vol_zscore_30m` :** trois z-scores de volume sur fenêtres glissantes (même formule que `vol_zscore_60m` mais sur 5, 15 et 30 barres). Testés dans l'hypothèse que l'accélération multi-fenêtre du volume serait discriminante. Résultat : AUC IF −0,010 par rapport à la version sans ces features. Cause : ces z-scores sont redondants avec `vol_zscore_60m` et ajoutent du bruit plutôt que du signal pour l'Isolation Forest. Rejetés.

**`vol_acceleration` (vol_zscore_5m − vol_zscore_30m) :** mesure la vitesse d'accélération du volume. Sep_ratio = 0,61× (moins séparant que la baseline). Signal marginal. Rejeté.

**`market_cap_rank` (CoinGecko) :** rang de capitalisation boursière au moment du pump. Hypothèse : les pumps ciblent préférentiellement les petites capitalisations. API CoinGecko non fonctionnelle pour coins délistés sur la période 2018-2020. Non collecté.

**`aggTrades OFI exact` :** OFI calculé trade par trade (vs proxifié depuis klines). Plus précis, mais API Binance retourne 0 lignes pour tous les coins délistés. 404 confirmé sur `data.binance.vision`. Non collecté.

**`PCA prix (price_PC1)` :** première composante principale des 5 features de momentum prix (ret_5m, ret_15m, ret_60m, std_ret_60m, price_impact_1m). Hypothèse H2 : compresser les features corrélées (r ≥ 0,87) en une seule dimension réduirait la redondance. Résultat : AUC IF −0,087 (0,8299 → 0,7433). Cause : l'Isolation Forest exploite précisément les petites divergences entre features corrélées — les 4,8% de variance résiduelle inter-features contiennent du signal discriminant. Comprimer les détruit. Rejeté.

**Fenêtre gt=1 réduite à 30min :** test d'une fenêtre d'observation pré-pump de seulement 30 minutes (au lieu de 2h). Hypothèse : le signal est concentré dans les 30 dernières minutes. Résultat : AUC IF −0,207 (0,8299 → 0,6233). Cause : le signal est distribué sur toute la fenêtre de 2h, pas concentré. Revenu à la fenêtre 2h.

---

## 4. Les décisions méthodologiques

### Choix de l'approche non supervisée

**Décision :** entraîner les modèles sur X sans jamais passer y à `fit()`. Les labels servent uniquement à évaluer les scores post-hoc.

**Pourquoi :** en production réelle, les labels n'existent pas au moment de l'inférence. Un classifieur supervisé (Random Forest, XGBoost) entraîné sur les 333 événements La Morgia ne généralisera pas aux pumps privés et ne peut pas scorer des fenêtres inconnues en temps réel. L'approche non supervisée est la seule cohérente avec le cadre de déploiement.

**Alternative rejetée :** classification supervisée avec train/test split temporel. Rejetée car les labels La Morgia ne représentent qu'une fraction des pumps réels et biaiseraient le modèle vers les patterns connus.

### Choix de RobustScaler

**Décision :** normaliser les features par la médiane et l'IQR (interquartile range) plutôt que par la moyenne et l'écart-type.

**Pourquoi :** les features financières (vol_zscore, ret_5m, price_impact) présentent des queues lourdes et des valeurs extrêmes. Un StandardScaler serait distordu par ces extrêmes. Le RobustScaler est insensible aux outliers — ce qui est critique quand les outliers sont précisément les pumps qu'on cherche à détecter.

**Alternative rejetée :** MinMaxScaler (sensible aux valeurs extrêmes), StandardScaler (même problème), pas de scaling (l'IF est théoriquement invariant à l'échelle mais les distances pour le LOF ne le sont pas).

### Choix de l'évaluation temporelle stricte

**Décision :** toutes les features sont calculées à partir de lignes vérifiant `open_time < window_end_ms`. Une assertion explicite est ajoutée dans `build_features.py` :
```python
assert int(pre["open_time"].max()) < window_end_ms
```

**Pourquoi :** toute donnée postérieure à pump_ts introduit un leakage temporel. C'est précisément l'erreur de l'évaluation A2 (AUC=0,9976 — voir section 5.1). L'assertion rend cette discipline non négociable et détectable immédiatement si elle est violée.

**Alternative rejetée :** cross-validation shufflée (standard en ML classique). Inutilisable ici : le modèle est non supervisé, il n'y a pas de train/test split. La discipline temporelle s'applique uniquement à la construction des features, pas à l'entraînement.

### Choix de Recall@LaMargia comme métrique primaire

**Décision :** utiliser le recall (taux de vrais positifs) comme métrique principale, à contamination c fixée (budget d'alertes).

**Pourquoi :** dans un contexte de surveillance réglementaire, manquer un vrai pump (faux négatif) est pire que générer une fausse alerte (faux positif). L'accuracy est aveugle : à 0,054% de taux réel de pumps, un modèle qui dit "toujours normal" atteint 99,946% d'accuracy. Le F1 dépend du seuil et de la précision, difficiles à calibrer sans données de déploiement. Le recall à contamination fixée est directement interprétable comme "budget d'alertes analyste" : à c=0,5%, on lève 18 alertes et on capture 19 vrais pumps.

**Métrique secondaire :** AUC-ROC, indépendante du seuil, pour comparer les modèles sur tout le spectre.

### Choix du dataset La Morgia comme ground truth

**Décision :** utiliser les événements La Morgia comme seuls labels positifs.

**Pourquoi :** c'est la seule source de labels pump & dump validée académiquement pour Binance 2018-2020. Ces événements proviennent de groupes Telegram publics — leur authenticité est vérifiable.

**Limite connue :** les pumps privés (Discord, groupes fermés) ne sont pas représentés. Le recall mesuré est donc un recall sur les pumps *connus publiquement*, potentiellement sous-estimé par rapport au phénomène réel.

---

## 5. Les problèmes rencontrés et solutions trouvées

### 5.1 L'artefact AUC=0.9976

**Symptôme :** L'évaluation initiale (A2) produisait AUC=0,9976, Recall≈100%. Un chiffre trop bon pour être honnête.

**Investigation — audit en 5 étapes :**

1. **Séparabilité triviale :** calcul du sep_ratio = μ(gt=1) / μ(gt=0) pour chaque feature La Morgia. Les features `std_volume` et `std_rush_order_count` séparent les classes avec un ratio de 145×. Conclusion : les features séparent trivialement les classes — ce n'est pas du leakage, c'est une séparabilité structurelle excessive.

2. **Leakage La Morgia :** vérification que les features La Morgia sont calculées sur une fenêtre `[pump_ts−24h, pump_ts+24h]`. Les lignes gt=1 coïncident avec le pic de ces features. Confirmation : le modèle détectait le pump *pendant qu'il se produisait*, pas avant. Ce n'est pas de la prédiction, c'est de la détection rétrospective.

3. **Baseline univariée :** AUC d'un simple Z-score sur les features La Morgia = 0,9987. Le problème est dans les features, pas dans le modèle.

4. **Généralisation klines :** test sur les features klines calculées strictement avant pump_ts. AUC chute à 0,8299. Confirmation que les features La Morgia étaient la cause.

5. **Contamination train/test :** vérification qu'il n'y a pas de fuite entre les données d'entraînement et d'évaluation. Aucune fuite — le problème est purement dans la définition des features.

**Cause racine :** les features La Morgia (`std_volume`, `std_rush_order_count`, etc.) sont calculées sur la fenêtre `[pump_ts−24h, pump_ts+24h]`, qui **inclut le moment du pump**. Les lignes gt=1 correspondent exactement au pic de ces features. Le modèle n'apprend pas à anticiper un pump — il apprend à reconnaître un pump en cours. Inutilisable en production.

**Solution :** reconstruction complète du pipeline. Les features sont recalculées depuis les klines Binance sur la fenêtre strictement pré-pump `[pump_ts−2h, pump_ts)`, avec assertion temporelle explicite dans le code.

**Résultat :** AUC 0,9976 → **0,8299** (honnête). Δ = −0,169 — le coût de l'honnêteté méthodologique.

### 5.2 La baseline gt=0 biaisée (hypothèse H1)

**Symptôme :** AUC=0,8299, Recall@0,005=0,012. La décomposition des scores révèle un problème plus profond.

**Hypothèses formulées et testées :**

- **H-fenêtre (fenêtre gt=1 réduite à 30min) :** hypothèse que le signal est concentré dans les 30 dernières minutes avant le pump. Résultat : AUC −0,207. Le signal est distribué sur 2h. Rejeté.

- **H-vol (nouvelles features volume multi-fenêtre) :** ajout de vol_zscore_5m/15m/30m + vol_acceleration. Résultat : AUC −0,010. Le signal volumique redondant nuit à l'IF. Rejeté.

- **H2 (PCA sur cluster momentum prix) :** compression des 5 features corrélées (r≥0,87) en une composante principale. Hypothèse : réduire la redondance améliore la discrimination. Résultat : AUC −0,087 (0,8299 → 0,7433). L'IF exploite précisément les 4,8% de variance résiduelle entre features corrélées. Rejeté.

- **H1 (correction baseline gt=0) :** examen des scores moyens par groupe.

| Groupe | Score moyen IF |
|--------|---------------|
| Marché normal réel (BTC/ETH, dates aléatoires) | +0,101 |
| Ancien gt=0 (heure calme, même événement pump) | +0,027 |
| Fenêtre pré-pump gt=1 | +0,021 |

**Cause racine :** les fenêtres gt=0 de la version initiale provenaient des *mêmes événements pump* que les gt=1 (fenêtre `[pump_ts−2h, pump_ts−1h)`). Ces fenêtres "calmes" étaient déjà légèrement anormales (mêmes coins rares, même période de marché tendue, contexte pré-manipulation). Leur score moyen (+0,027) était quasi-identique au score des pumps (+0,021). L'IF comparait anormal vs légèrement-moins-anormal — une frontière impossible à apprendre.

**Solution :** collecte de **3 000 fenêtres de marché aléatoires** via l'API Binance publique sur 5 coins liquides (BTCUSDT, ETHUSDT, BNBUSDT, XRPUSDT, LTCUSDT), période 2018-2021, dates tirées au hasard. Ces fenêtres représentent un marché genuinement normal. Score moyen = +0,101 — clairement séparé des pumps.

**Résultat :** AUC 0,8299 → **0,8815** (+0,052), Recall@0,005 : 0,012 → **0,057** (×4,75). La séparation est réelle.

### 5.3 L'échec des données contextuelles

**aggTrades — approche 1 (API REST Binance) :** `GET /api/v3/aggTrades?symbol=BRDBTC&startTime=...&endTime=...` retourne une liste vide pour tous les coins délistés testés. Binance ne conserve pas les données tick par tick après délisting. Résultat : 0 lignes pour 100% des événements pump.

**aggTrades — approche 2 (data.binance.vision) :** l'archive publique Binance pour les données historiques retourne HTTP 404 sur les mêmes paires. La purge est complète — les données n'existent nulle part.

**Impact :** l'OFI exact (Order Flow Imbalance, calculé trade par trade) n'est pas disponible. On le proxifie depuis `taker_buy_base` des klines : `ofi_proxy = (2 × takerBuyBase − volume) / volume`. Corrélé à l'OFI exact sur les coins encore listés (BTC/ETH), mais potentiellement moins précis sur les micro-caps.

**CoinGecko :** l'API historique CoinGecko ne renvoie pas de données de rang pour des tokens délistés sur la période 2018-2020. La feature `market_cap_rank` — hypothèse que les pumps ciblent préférentiellement les plus petites capitalisations — n'a pas pu être testée.

---

## 6. Les modèles

### Isolation Forest (AUC = 0,881)

**Principe :** l'Isolation Forest construit 200 arbres de décision aléatoires. À chaque arbre, il choisit aléatoirement une feature et une valeur de coupure. L'idée centrale : une observation anormale (un pump) est *rare et extrême* — elle s'isole en **peu de coupures**. Une observation normale nécessite beaucoup plus de coupures pour être isolée. Le score est la profondeur moyenne d'isolation sur les 200 arbres.

**Hypothèse implicite :** les pumps sont rares *et* occupent une région de l'espace de features différente du marché normal. Ni l'une ni l'autre n'est vraie à 100% (certains pumps ressemblent à du marché normal), mais c'est vrai pour une fraction suffisante.

**Avantages :**
- Complexité O(n log n), scalable
- Aucune hypothèse distributionnelle (pas de gaussianité requise)
- Immune à la multicollinéarité : partitionne chaque feature indépendamment, donc les 5 features corrélées du cluster momentum (r≥0,87) contribuent chacune séparément au signal
- Contamination directement paramétrable

**Limites :**
- Boîte noire — ne produit pas d'explication par feature sans SHAP
- Ne capte pas les anomalies *contextuelles* (normales globalement, anormales localement)

**Résultat :** AUC=0,8815 | Recall@c=0,001 : 0,012 | Recall@c=0,005 : 0,057 | Recall@c=0,05 : 0,387

### Local Outlier Factor (AUC = 0,710)

**Principe :** le LOF compare la densité locale d'un point à celle de ses k=20 plus proches voisins. Un point isolé dans une région dense globalement — mais entouré de voisins eux-mêmes bien espacés — reçoit un score élevé. Conçu pour les anomalies *contextuelles*.

**Hypothèse implicite :** les pumps forment des clusters localement peu denses dans l'espace de features.

**Avantages :**
- Capture des anomalies locales que l'IF rate
- Aucune hypothèse distributionnelle

**Limites :**
- Les distances euclidiennes en ℝ¹⁴ sont dominées par les 5 features de momentum corrélées. Les 4,8% de variance résiduelle entre ces features — qui contiennent le signal discriminant pour l'IF — sont noyés dans la distance euclidienne totale. Le LOF ne peut pas les exploiter.
- `novelty=False` : ne peut pas scorer de nouvelles observations post-entraînement — ne se sauvegarde pas en joblib pour déploiement
- Recall@c=0,005 = 0,012 vs 0,057 pour l'IF

**Résultat :** AUC=0,7103 | Recall@c=0,001 : 0,009 | Recall@c=0,005 : 0,012 | Recall@c=0,05 : 0,159

### Z-score rolling (AUC = 0,766)

**Principe :** score univarié directement issu du feature engineering : `vol_zscore_60m = (vol₋₁ − μ_vol) / σ_vol`. Aucun entraînement. La décision est immédiate : une barre dont le volume dépasse la baseline de la fenêtre par X écarts-types est suspecte.

**Hypothèse implicite :** les pumps génèrent systématiquement un spike de volume anormal par rapport à la baseline de la fenêtre. Vraie pour les pumps avec accumulation visible, fausse pour les pumps avec annonce simultanée (spike post-annonce invisible dans la fenêtre pré-pump).

**Avantages :**
- Interprétable directement : "ce coin a un volume 8× supérieur à sa baseline des 2h précédentes"
- Déployable en streaming sans infrastructure ML, à coût zéro
- Adaptif aux régimes de marché (normalisé par la fenêtre courante)
- Recall@0,005 = 0,057 — aussi bon que l'IF à haute contamination, sur les pumps avec gros spike

**Limites :**
- Ignore les 13 autres features (OFI, taker ratio, momentum)
- Aveugle aux pumps dont le signal s'exprime sur le momentum prix ou la microstructure plutôt que sur le volume brut

**Résultat :** AUC=0,7660 | Recall@c=0,001 : 0,012 | Recall@c=0,005 : 0,057 | Recall@c=0,05 : 0,345

### Pourquoi l'Isolation Forest gagne

La structure de corrélation des features explique directement la hiérarchie des modèles. Les 5 features de momentum prix (ret_5m, ret_15m, ret_60m, std_ret_60m, price_impact_1m) forment un cluster fortement corrélé (r ≥ 0,87) — elles mesurent en grande partie le même phénomène. Pour le LOF, ces 5 features "écrasent" les distances euclidiennes et masquent le signal des 9 autres features orthogonales. L'IF, en partitionnant chaque feature indépendamment, utilise les 5 axes du cluster comme **5 sources de signal indépendantes**, chacune contribuant à réduire la profondeur d'isolation des pumps.

---

## 7. Résultats finaux

### Tableau comparatif complet

| Configuration | Dataset | AUC | Recall@0,005 | Statut |
|---|---|---|---|---|
| La Morgia features (A2) | 584k lignes | 0,9976 | ~100% | ❌ Artefact — features calculées pendant le pump |
| Klines pré-pump, baseline initiale | 666 lignes | 0,8299 | 0,012 | ⚠️ Baseline gt=0 biaisée (même événements) |
| **H1 final (IF)** | **3 666 lignes** | **0,8815** | **0,057** | ✅ Honnête |
| H1 final (LOF) | 3 666 lignes | 0,7103 | 0,012 | ✅ Honnête |
| H1 final (Z-score) | 3 666 lignes | 0,7660 | 0,057 | ✅ Honnête |

### Tableau de sensibilité à la contamination (modèle IF)

| Contamination | Budget d'alertes (sur 3 666) | Vrais pumps détectés | Recall |
|---|---|---|---|
| 0,05% | 2 alertes | 2 | 0,006 |
| 0,10% | 4 alertes | 4 | 0,012 |
| 0,50% | 18 alertes | 19 | 0,057 |
| 1,00% | 37 alertes | 37 | 0,111 |
| 5,00% | 183 alertes | 129 | 0,387 |

### Interprétation du Recall

À c=0,5% (contamination réaliste pour un analyste de surveillance) : le modèle analyse 3 666 fenêtres de marché, lève les 18 alertes les plus suspectes, et capture 19 des 333 vrais pumps connus dans le dataset. Un modèle aléatoire à la même contamination capturerait en moyenne 0,5% × 333 ≈ 1,7 pumps. L'IF est donc **×12 mieux que le hasard** à ce budget d'alertes.

---

## 8. Limites

### Signal structurel absent dans la majorité des pumps

C'est la limite fondamentale du projet. Sur les 3 exemples visualisés dans `fig_raw_klines.png` (DUSK, TNT, MOD), le spike de volume arrive **après** pump_ts dans les 3 cas — y compris pour DUSK dont `vol_zscore_60m = 10,9` (le maximum du dataset). Ce z-score élevé reflète une élévation relative de la dernière barre par rapport à un fond lui-même très faible, pas un signal d'accumulation pré-annonce.

Les pumps coordonnés 2018-2020 sur Binance opèrent avec un délai entre décision et exécution inférieur à la résolution d'une barre 1-min. La coordination se fait simultanément sur des milliers de membres — il n'y a pas de phase d'accumulation détectable. Recall@0,001 = 0,012 n'est pas une défaillance du modèle : c'est la réalité du signal disponible.

### Données 2018-2021, marché moderne différent

Les patterns de pump & dump de 2018-2021 ciblaient des micro-caps BTC. Le marché crypto 2024+ est différent : stablecoins, marchés perpétuels, hybridation CEX/DEX, nouvelles plateformes. Le modèle n'a pas été validé sur des données récentes — une validation out-of-time sur 2022-2024 serait nécessaire avant tout déploiement.

### Features mono-exchange

Toutes les features proviennent de Binance uniquement. Les manipulations cross-exchange (accumulation sur Binance, dump sur OKX, arbitrage simultané) sont invisibles dans ce pipeline. Or, c'est précisément l'avantage compétitif de Kaiko : surveiller 150+ exchanges simultanément et calculer des features cross-exchange (spread inter-venues, divergences de prix, flux net agrégé).

---

## 9. Pistes d'amélioration

### Données tick-level sur coins actifs

Les coins délistés après leur pump ont perdu leurs données aggTrades définitivement. Pour les coins encore listés, les données tick-level permettraient un OFI exact (plutôt que proxifié depuis klines), une résolution sub-minute, et des features de microstructure plus précises (queue imbalance, trade size distribution).

### Features cross-exchange (avantage unique de Kaiko)

Le pipeline actuel est limité à Binance. Les features cross-exchange — spread bid-ask inter-venues, divergences de prix entre plateformes, corrélation des flux d'ordres — exploiteraient la couverture unique de Kaiko (150+ exchanges) et rendraient le modèle difficile à contourner. Un acteur malveillant peut masquer son accumulation sur un seul exchange ; masquer une accumulation simultanée sur 10 exchanges est beaucoup plus difficile.

### Détection en streaming sur fenêtre glissante

Le pipeline actuel est batch : il analyse une fenêtre fixe de 2h. Une fenêtre glissante en temps réel (via WebSocket Binance, mise à jour chaque minute) permettrait une alerte quasi-instantanée. Un score anormal croissant sur 3-5 barres consécutives serait un signal bien plus fort qu'un score ponctuel.

### Active learning avec feedback analyste

Chaque alerte générée est soit confirmée (vrai pump) soit rejetée (faux positif) par un analyste. Ces feedbacks constituent des labels de qualité sur données récentes. Un cycle d'active learning — réentraînement périodique avec les nouvelles annotations — permettrait au modèle de s'adapter à l'évolution des techniques de manipulation et de corriger ses faux positifs récurrents.

### Audit trail SHAP pour conformité MiCA

Le règlement MiCA Article 62 impose une documentation des décisions algorithmiques. SHAP (SHapley Additive exPlanations) permet de décomposer le score de l'Isolation Forest par feature pour chaque alerte : "ce score de +0,14 est dû à 60% au vol_zscore_60m = 8,2 et à 25% au taker_buy_ratio_5m = 0,91". Indispensable pour un déploiement en environnement régulé.
