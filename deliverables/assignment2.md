# Assignment 2 — Feature Engineering

**Rendu :** push sur le dépôt git avant le mardi 12 mai  
**Fichiers modifiés :** `scripts/data.py`, `scripts/metrics.py`, `notebooks/first_model.ipynb`

---

## Dataset de départ → dataset transformé

### Dataset de départ

Le dataset de départ est `data/RAW/la_morgia_raw/labeled_features/features_15S.csv.gz`, fourni par La Morgia et al. 2020.

| Propriété | Valeur |
|-----------|--------|
| Lignes | 584 104 |
| Colonnes brutes | 16 |
| Événements pump (gt=1) | 317 (0.054 %) |
| Valeurs manquantes | 0 |

**Colonnes brutes :**

| Colonne | Type | Description |
|---------|------|-------------|
| `date` | string | Horodatage de l'observation (format "YYYY-MM-DD HH:MM:SS") |
| `pump_index` | int | Index de position dans la fenêtre temporelle autour du pump |
| `std_rush_order` | float | Écart-type des volumes des multi-trades (rush orders) |
| `avg_rush_order` | float | Moyenne des volumes des multi-trades |
| `std_trades` | float | Écart-type du nombre de trades |
| `std_volume` | float | Écart-type du volume BTC |
| `avg_volume` | float | Moyenne du volume BTC |
| `std_price` | float | Écart-type du prix |
| `avg_price` | float | Prix moyen (fenêtre glissante=10) |
| `avg_price_max` | float | Moyenne du prix maximal glissant |
| `hour_sin` / `hour_cos` | float | Encodage cyclique de l'heure |
| `minute_sin` / `minute_cos` | float | Encodage cyclique de la minute |
| `symbol` | string | Ticker de la pièce (ex. "BRD", "RDN") — 85 valeurs uniques |
| `gt` | int | Label binaire (1 = pump, 0 = normal) |

### Dataset transformé

Le dataset transformé est sauvegardé dans `data/Processed/features_15S_transformed.parquet`.

| Propriété | Valeur |
|-----------|--------|
| Lignes | 584 104 |
| Colonnes après transformation | 28 |
| Format | Parquet (lecture rapide, compression efficace) |

---

## Étapes de nettoyage des données

Le dataset La Morgia est remarquablement propre : **aucune valeur manquante**, tous les types sont cohérents. Les étapes de nettoyage sont donc minimales :

1. **Conversion de `date` en datetime** : la colonne `date` est une chaîne de caractères. Elle est convertie en `pd.Timestamp` (timezone UTC) pour que le `DatetimeEncoder` de skrub puisse l'analyser.
2. **Conservation de `pump_index`** : représente la position relative dans la fenêtre temporelle [−2h, +30min] autour du pump. Il encode l'information "à quelle distance de l'événement se trouve cette observation" — information utile pour l'anomaly detection.
3. **Aucune imputation** : pas de NaN à traiter.
4. **Aucune déduplication** : chaque ligne correspond à un intervalle de 15 secondes distinct.

---

## Transformations appliquées

Toutes les transformations sont centralisées dans `scripts/data.py` via un pipeline `sklearn` composé de :

### 1. `TableVectorizer` de skrub

Le `TableVectorizer` est le point d'entrée recommandé par le professeur. Il orchestre automatiquement les encodeurs par type de colonne.

**Sur `date` → `DatetimeEncoder(add_weekday=True, add_total_seconds=False)`**

La colonne `date` est décomposée en features temporelles structurées :
- Année, mois, jour, heure, minute
- Encodage cyclique sin/cos (heure, minute, jour de semaine)
- Jour de la semaine (`add_weekday=True`) : les pumps ont tendance à se concentrer les week-ends, quand la surveillance institutionnelle est réduite

Résultat : 1 colonne datetime → ~10 features numériques.

**Sur `symbol` → `GapEncoder(n_components=10)`**

`symbol` est une variable catégorielle à haute cardinalité (85 valeurs uniques). Le `GapEncoder` apprend un embedding dense de 10 dimensions à partir des n-grammes de caractères du ticker.

Avantage par rapport à OneHotEncoder : représentation compacte (10D vs 85D), robuste aux symboles peu fréquents, capture la morphologie des tickers (ex. tokens similaires : "BRD" ≈ "BRL").

**Sur les numériques → PassThrough**

Les features numériques (std_rush_order, avg_volume, etc.) et `pump_index` sont passées telles quelles au `RobustScaler` suivant.

### 2. `RobustScaler`

Les features financières présentent des distributions très asymétriques (long tail à droite) : un seul burst coordonné peut avoir un volume 50× supérieur à la médiane. Le `RobustScaler` (médiane + IQR) est insensible à ces outliers, contrairement au `StandardScaler` (moyenne + écart-type).

### 3. PCA optionnelle

Le pipeline dispose d'un mode `use_pca=True` qui ajoute une `PCA(n_components=10)` après le scaler. Voir la section sur les alternatives testées pour les résultats.

---

## Nouvelles features créées

| Feature | Source | Signal |
|---------|--------|--------|
| `date__year`, `date__month`, `date__day` | DatetimeEncoder | Tendances à long terme (volume de pumps croissant 2018→2020) |
| `date__hour`, `date__minute` + sin/cos | DatetimeEncoder | Saisonnalité intraday (pic à 16h-18h UTC) |
| `date__weekday` | DatetimeEncoder | Les pumps se concentrent le week-end (traders retail actifs) |
| `symbol__0` à `symbol__9` | GapEncoder | Embedding du ticker — encode la "vulnérabilité relative" d'un symbole |

---

## Justification des choix effectués

| Choix | Justification |
|-------|---------------|
| **Fréquence 15S** | Meilleur compromis entre résolution temporelle et taille du dataset. 25S est trop grossier (perd les micro-spikes de rush orders) ; 5S est 3× plus volumineux pour un gain marginal de signal. |
| **GapEncoder sur symbol** | Cardinalité 85 → OneHotEncoder peu adapté. GapEncoder produit un embedding dense compact, robuste et interprétable. MinHashEncoder testé mais GapEncoder donne un AUC légèrement supérieur. |
| **RobustScaler plutôt que StandardScaler** | Les features de volume et de rush orders ont des outliers extrêmes (50-100× la médiane lors des pumps). StandardScaler les dilue ; RobustScaler les préserve. |
| **Conserver `date` via DatetimeEncoder** | Dropping `date` testé — perd le signal du jour de la semaine et de l'heure, qui discriminent les pumps (pics à 16h-18h UTC, sur-représentation week-end). |
| **Conserver `pump_index`** | Encode la position temporelle relative dans la fenêtre. Dropping testé — légère baisse de l'AUC. |

---

## Alternatives testées et non retenues

| Alternative | Résultat | Raison du rejet |
|-------------|----------|-----------------|
| **StandardScaler** | AUC similaire | Sensible aux outliers extrêmes des features de rush orders. RobustScaler plus cohérent avec la réalité des données. |
| **OneHotEncoder pour `symbol`** | AUC identique | Produit 85 colonnes binaires sparse. Moins compact que GapEncoder et ne généralise pas aux nouveaux symboles. |
| **MinHashEncoder pour `symbol`** | AUC légèrement inférieur | GapEncoder retenu car il capture mieux la structure morphologique des tickers courts. |
| **Supprimer `date`** | Perte de signal | Le jour de la semaine et l'heure sont des features discriminantes (pumps à 16h-18h UTC, week-ends). |
| **PCA à 10 composantes** | AUC = 0.65 vs 0.94 sans PCA | Détail en section suivante. |
| **Fréquence 5S** | Dataset 3× plus lourd | Surcoût computationnel non justifié par le gain de signal observé. |
| **Fréquence 25S** | Perte de granularité | Les rush orders (rafales de trades en 2-5s) sont moyennés et deviennent invisibles. |

---

## Impact de la PCA sur le modèle

Le notebook `notebooks/first_model.ipynb` détaille la comparaison. Résultats sur sous-échantillon (10 000 normaux + 317 pumps) :

| Modèle | AUC | Recall@LaMargia (c=0.001) |
|--------|-----|--------------------------|
| Isolation Forest **sans PCA** (28 features) | **0.94** | 3.5% |
| Isolation Forest **avec PCA** (10 PC) | 0.65 | 0.6% |

**La PCA dégrade significativement les deux métriques.** Interprétation : le signal de pump est distribué sur plusieurs features originales (std_rush_order, avg_volume, std_price…). En projetant sur 10 composantes principales, on mélange ces signaux avec du bruit, effaçant la séparation que l'Isolation Forest exploite dans l'espace à 28 dimensions.

**Impact attendu sur les modèles futurs** : LOF (densité locale) sera encore plus sensible à la PCA que l'Isolation Forest, car il repose sur des distances dans l'espace de features — une réduction agressive distort les voisinages. Le Z-score rolling est univarié et non affecté par la PCA.

---

## Ce que nous n'avons pas réussi à faire

### Features manquantes — données non disponibles

| Feature prévue | Raison de l'absence | Impact sur le modèle |
|----------------|---------------------|----------------------|
| `ret_5m`, `ret_15m`, `ret_60m`, `vol_zscore_60m`, `taker_buy_ratio_5m`, `rush_order_count` | ✅ Collectées via klines Binance REST (333 fichiers) | — |
| `ofi_1m` (order flow imbalance exact) | ❌ aggTrades historiques **définitivement inaccessibles** — voir section ci-dessous | Perd le signal OFI le plus précis |
| `market_cap_rank`, `circulating_supply_log` | CoinGecko non implémenté (`fetch_coingecko.py` vide) | Pas de contexte de vulnérabilité cross-sectionnel |

### Limitation permanente — aggTrades historiques introuvables

#### Tentative 1 : Binance REST API `/aggTrades`

Le endpoint REST ne conserve qu'une fenêtre glissante de quelques mois.
Tous nos événements (2018–2021) sont hors de cette fenêtre → **0 lignes retournées** pour tous les symboles.

#### Tentative 2 : data.binance.vision (archive officielle Binance)

`data.binance.vision` propose des fichiers ZIP journaliers pour l'historique complet depuis 2017, sans clé API.
URL : `https://data.binance.vision/data/spot/daily/aggTrades/{SYMBOL}/{SYMBOL}-{YYYY-MM-DD}.zip`

Script mis à jour (`fetch_aggtrades_vision()` dans `scripts/fetch_binance.py`) pour télécharger les deux jours encadrant chaque événement (gestion du passage minuit UTC) et filtrer à la fenêtre [pump_ts − 10min, pump_ts).

**Résultat : HTTP 404 pour la totalité des symboles testés**, toutes années confondues (2018, 2019, 2020, 2021).

#### Cause racine

`data.binance.vision` n'archive que les paires **actuellement listées** sur Binance.
Or, les cibles de pump de La Morgia étaient précisément des tokens à faible capitalisation, souvent délistés par Binance dans les années suivantes (BRDBTC, BQXBTC, NXSBTC, NAVBTC, VIBBTC, STEEMBTC, etc.).
Ces données de trades ont été définitivement supprimées de l'infrastructure Binance.

#### Seule source disponible

La Morgia et al. ont téléchargé les données de trades en 2020, pendant que les symboles étaient encore listés, via leur script `downloader.py` (CCXT + historique complet).
Les features résultantes (`std_rush_order`, `avg_rush_order`, `std_trades`, `std_volume`) sont pré-calculées dans `labeled_features/features_15S.csv.gz` et constituent le **seul proxy disponible** pour les features de type OFI.

#### Conséquence pour le modèle

Le modèle s'appuie sur les features La Morgia (std_rush_order, avg_volume, std_price…) comme substituts des features de microstructure exactes.
L'AUC = 0.998 (Isolation Forest) reflète la forte séparabilité des features La Morgia (voir audit ci-dessous).
Le faible Recall@LaMargia à contamination basse (0.001) reflète la rareté des pumps dans le dataset, pas un manque de signal.

---

## Audit forensique — Pourquoi AUC = 0.9976 ?

### Protocole d'audit (5 étapes)

Suite à la valeur suspicieusement élevée de l'AUC (0.9976 pour un modèle non supervisé), un audit complet a été conduit pour en identifier la cause racine.

### Étape 1 — Séparabilité univariée des features

| Feature | Moyenne (pump) | Moyenne (normal) | Ratio de séparation |
|---------|---------------|-----------------|---------------------|
| `std_volume` | 2.77 | −0.000 | **145.5×** |
| `std_trades` | 1.44 | −0.000 | 132.9× |
| `std_rush_order` | 0.72 | −0.000 | 99.5× |

Au seuil du 99ème percentile des normaux, un simple seuil sur `std_volume` capture **93%** des pumps avec un taux de faux positifs de 0.95%. **La séparation est triviale par une seule feature.**

### Étape 2 — Origine du problème : conception de la fenêtre La Morgia

En lisant `data/RAW/la_morgia_raw/features.py`, on découvre que les features sont calculées sur une fenêtre **[−24h, +24h]** autour du pump_ts. Les lignes `gt=1` correspondent exactement aux intervalles de 15 secondes **pendant** le pump — c'est-à-dire au moment où les volumes et prix sont à leur maximum.

Autrement dit : les features encodent "le pump est en train de se produire maintenant", pas "un pump est sur le point de se produire". L'évaluation mesure la capacité du modèle à reconnaître un événement *en cours*, pas à le *prédire*.

### Étape 3 — Confirmation par Z-score univarié

```
Z-score (std_volume seul, zéro ML)  →  AUC = 0.9947
Isolation Forest (200 arbres)       →  AUC = 0.9976
```

Un seuil statistique sur une seule colonne atteint presque le même AUC que le modèle d'ensemble complet. L'apport marginal du ML multivarié est de 0.003.

### Étape 4 — Test d'absence de fuite d'entraînement

| Protocole | AUC |
|-----------|-----|
| IF entraîné sur **toutes** les données (original) | 0.9976 |
| IF entraîné sur **gt==0 uniquement** (protocole propre) | **0.9989** |

La différence est de **0.0012** — négligeable. Inclure les pumps dans l'entraînement ne donne aucun avantage. Il n'y a **pas de fuite due à l'entraînement**.

### Étape 5 — Généralisation temporelle

Split temporel strict : entraînement 2018-2019, test 2020-2021 (hors-échantillon complet).

| Évaluation | AUC |
|------------|-----|
| In-sample (notebook original) | 0.9976 |
| Out-of-time (train 2018-2019, test 2020-2021) | **0.9987** |

L'AUC est stable voire légèrement supérieure hors-temps. La séparation des features est un phénomène structurel constant sur toutes les années.

### Conclusion de l'audit

**Cause racine : séparabilité triviale des features, pas une fuite de modèle.**

| Question | Réponse |
|----------|---------|
| L'AUC est-il gonflé par du leakage d'entraînement ? | **Non** — delta IF full vs IF clean = 0.0012 |
| L'AUC est-il généraliste (hors-temps) ? | **Oui** — AUC temporel = 0.9987 |
| L'AUC reflète-t-il une vraie capacité de prédiction ? | **Partiellement** — il mesure la détection *pendant* le pump, pas *avant* |
| Le problème est-il intrinsèquement difficile ? | **Non** — la séparabilité univariée est de 145× sur std_volume |

**Implication pour le déploiement** : un modèle de surveillance réelle devrait prédire un pump *avant* qu'il atteigne son pic. L'évaluation sur le dataset La Morgia (qui labellise les fenêtres *pendant* le pump) surestime la difficulté du problème et crée une illusion de haute performance. Le vrai challenge — détecter la phase d'accumulation pré-pump — reste non évalué.

---

## Datasets transformés — accès et utilisation

Les datasets transformés sont stockés dans `data/Processed/` (exclu du git par `.gitignore`). Ils sont **regénérés localement** en exécutant :

```bash
python scripts/data.py
```

Pour les charger dans un notebook :

```python
import pandas as pd
df = pd.read_parquet('data/Processed/features_15S_transformed.parquet')
X = df.drop(columns=['gt'])
y = df['gt']
```

Le format Parquet a été choisi pour sa lecture rapide (lecture columnaire) et sa compression efficace — le fichier est ~30× plus petit que le CSV.gz équivalent tout en étant plus rapide à lire.

