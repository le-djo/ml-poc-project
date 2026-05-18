# Assignment 3 — Modélisation et Évaluation

**Rendu :** push sur le dépôt git  
**Fichiers modifiés :** `scripts/build_features.py`, `notebooks/02_modeling.ipynb`  
**Reproduction :** `python scripts/build_features.py` puis exécuter `notebooks/02_modeling.ipynb` cellule par cellule

---

## 1. Définition du problème ML

### Cadre : détection d'anomalies non supervisée

Le problème est formulé comme de la **détection d'anomalies non supervisée** : à partir d'un vecteur de features calculées sur une fenêtre de marché, un modèle doit produire un score d'anomalie sans jamais avoir accès aux labels.

**Pourquoi pas supervisé ?**

En production, aucun label n'est disponible au moment de l'inférence. Les pumps annoncés publiquement sur Telegram (source La Morgia) ne couvrent que les événements coordonnés *publics* — les pumps privés (Discord, groupes fermés) ne sont pas étiquetés. Un classifieur supervisé entraîné sur les labels Telegram ne généralisera pas à ces cas.

**Formulation formelle**

$$X \in \mathbb{R}^{3666 \times 14}, \quad f : \mathbb{R}^{14} \to \mathbb{R}$$

où $f$ est le score d'anomalie (plus bas = plus anormal pour IF et LOF), et $y \in \{0, 1\}^{3666}$ est utilisé **uniquement post-hoc** pour évaluer les scores — jamais passé à `fit()`.

**Taux de pump**

Dans le dataset construit : 9,1 % (333 gt=1 sur 3 666 lignes — 333 fenêtres pré-pump + 3 333 fenêtres de marché genuines).  
Dans la réalité Binance : **0,054 %** (317 pumps sur 584 104 fenêtres de 15 secondes, La Morgia et al. 2020).  
Ce déséquilibre extrême justifie le choix des métriques ci-dessous.

---

## 2. Définition de la métrique d'évaluation

### Métrique primaire : Recall@LaMargia

$$\text{Recall} = \frac{\sum_{i : y_i = 1} \mathbf{1}[\hat{y}_i = 1]}{\sum_{i} y_i}$$

où $\hat{y}_i = \mathbf{1}[f(x_i) \leq \tau_c]$, $\tau_c$ étant le percentile $c$ des scores (les $c$ % les plus anomaux sont flaggés).

**Pourquoi pas l'accuracy ?**  
À 0,054 % de pumps, un modèle constant qui prédit *normal* atteint 99,946 % d'accuracy. La métrique est aveugle au problème.

**Pourquoi pas le F1 ?**  
Le F1 dépend du seuil et de la précision — deux grandeurs difficiles à calibrer sans données de déploiement réelles. Le Recall seul permet de comparer des modèles à des échelles de score différentes (IF, LOF, z-score) en faisant varier $c$.

**En surveillance réglementaire**, rater un vrai pump (faux négatif) est pire qu'une fausse alerte (faux positif) : le Recall est donc la métrique naturelle.

### Métrique secondaire : AUC-ROC

$$\text{AUC} = \int_0^1 \text{TPR}(\text{FPR}^{-1}(t)) \, dt$$

Mesure indépendante du seuil : probabilité que le modèle attribue un score plus anomal à un vrai pump qu'à une observation normale. Permet la comparaison inter-modèles sans fixer $c$.

---

## 3. Protocole d'évaluation

### Discipline temporelle (non négociable)

Toutes les features sont calculées à partir de lignes vérifiant `open_time < window_end_ms`. Une assertion explicite est ajoutée dans `scripts/build_features.py` :

```python
assert int(pre["open_time"].max()) < window_end_ms
```

Aucune donnée postérieure à `pump_ts` n'entre dans le calcul des features.

### Correction méthodologique par rapport à A2

L'évaluation A2 utilisait les features La Morgia (std_volume, std_rush_order, etc.) calculées sur la fenêtre `[-24h, +24h]` autour du pump. Les lignes `gt=1` coïncident *exactement* avec le pic des features — le modèle mesure la détection **pendant** le pump, pas avant. AUC=0,9976 est un artefact.

L'évaluation corrigée utilise des features calculées **strictement avant** `pump_ts` depuis les klines Binance. AUC=**0,881** est le résultat honnête.

### Design du dataset — hypothèse H1 (négatifs genuins)

| Label | Source | N | Signification |
|-------|--------|---|---------------|
| `gt=1` | Événements pump | 333 | Features de `[pump_ts − 2h, pump_ts)` |
| `gt=0` | Fenêtres aléatoires (BTC/ETH/BNB/XRP/LTC, 2018–2021) | 3 333 | Marché normal — pas adjacent à un pump |

3 666 lignes totales, 0 valeur manquante après imputation. Les fenêtres gt=0 ne proviennent plus des mêmes événements que gt=1 — biais d'adjacence corrigé (hypothèse H1 confirmée : +0,052 AUC).

### Protocole non supervisé

Pas de train/test split : les modèles sont entraînés sur $X$ sans accès à $y$, puis les scores sont évalués post-hoc contre $y$. Ce protocole est conforme au cadre non supervisé — en production, $y$ n'existe pas.

### Analyse de sensibilité

La contamination $c$ varie dans $\{0{,}0005 ;\ 0{,}001 ;\ 0{,}005\}$ pour évaluer la robustesse du Recall au choix du seuil.

### Reproduction

```bash
python scripts/build_features.py        # génère data/Processed/processed.parquet
jupyter nbconvert --to notebook --execute --inplace notebooks/02_modeling.ipynb
```

---

## 4. Présentation des trois modèles

### 4.1 Isolation Forest

**Hypothèses principales**  
Les pumps sont *rares* et *isolés* dans l'espace de features : un arbre de décision aléatoire les isole en moins de partitions que les observations normales.

**Avantages**  
- Complexité O(n log n), scalable sur de grands volumes
- Aucune hypothèse distributionnelle
- La contamination est directement paramétrable

**Limites**  
- Insensible aux corrélations locales (anomalies contextuelles dans des sous-espaces denses)
- Peut échouer si les features redondantes masquent la structure

**Adéquation au problème**  
Les pumps coordonnés produisent des spikes simultanés sur plusieurs features (volume, taker_buy_ratio, OFI) — exactement la signature qu'Isolation Forest détecte en peu de partitions.

**Paramètres**

```python
IsolationForest(n_estimators=200, contamination='auto', random_state=42)
```

**Résultat** : AUC = **0,881** | Recall@0,005 = **0,057**

---

### 4.2 Local Outlier Factor (LOF)

**Hypothèses principales**  
Les pumps présentent une densité locale inférieure à celle de leurs $k$ plus proches voisins — anomalies *contextuelles* plutôt que globales.

**Avantages**  
- Capture des anomalies que les méthodes globales ratent (région dense globalement, mais localement isolée)
- Aucune hypothèse distributionnelle

**Limites**  
- Complexité naïve O(n²) en mémoire
- Sensible au choix de $k$
- `novelty=False` : pas de `predict()` sur de nouvelles données — ne se sauvegarde pas en joblib

**Adéquation au problème**  
Complète IF en détectant les événements qui semblent normaux à l'échelle globale mais aberrants localement (pumps de petite amplitude précédés d'une accumulation discrète).

**Paramètres**

```python
LocalOutlierFactor(n_neighbors=20, novelty=False, contamination=0.001)
```

**Résultat** : AUC = **0,710** | Recall@0,005 = **0,012** — le plus faible des trois : la densité locale est moins discriminante sur ce jeu de features klines.

---

### 4.3 Z-score Glissant (baseline statistique)

**Hypothèses principales**  
Les pumps induisent des déviations statistiquement significatives par rapport à la baseline récente. La feature `vol_zscore_60m` encode déjà cette déviation : $(v_{-1} - \bar{v}) / \sigma_v$.

**Avantages**  
- Score immédiatement interprétable (nombre d'écarts-types)
- Adaptatif aux régimes de marché (normalisé par la fenêtre courante)
- Streamable en temps réel sans infrastructure ML
- Zéro coût d'entraînement

**Limites**  
- Univarié — ignore les autres features (ret_5m, OFI, taker_buy_ratio)
- Suppose une stationnarité locale sur 60 minutes
- Insensible aux pumps dont le signal s'exprime sur des features autres que le volume

**Adéquation au problème**  
Baseline naturelle déployable dans un Market Surveyor sans pipeline ML. Permet de quantifier l'apport marginal des modèles multivariés.

**Implémentation**

```python
scores_z = X[:, feat_cols.index('vol_zscore_60m')]   # higher = more anomalous
```

**Résultat** : AUC = **0,766** | Recall@0,005 = **0,057**

---

## 5. Justification du choix des trois modèles

### Tableau comparatif

| Modèle | Paradigme | Force principale | Coût computationnel | AUC |
|--------|-----------|-----------------|---------------------|-----|
| Isolation Forest | Global / Arbres ensemble | Isolation multi-feature | O(n log n) | **0,881** |
| LOF | Local / Densité géométrique | Anomalies contextuelles | O(n²) | 0,710 |
| Z-score rolling | Temporel / Statistique | Interprétabilité, streamable | O(1) | 0,766 |

### Complémentarité

Les trois modèles couvrent des paradigmes orthogonaux. Leur conjonction constitue un système de vote multi-signal :

- **IF seul détecte** → isolation globale multi-feature (signature pump classique)
- **LOF seul détecte** → accumulation discrète localement aberrante
- **Z-score seul détecte** → spike de volume isolé, sans coordination multi-feature
- **Consensus des trois** → haute confiance, alerte prioritaire

### Quantification de l'apport du ML

| Modèle | AUC | Delta vs baseline |
|--------|-----|-------------------|
| Z-score (zéro ML) | 0,766 | — |
| LOF | 0,710 | −0,094 (moins bon sur ce dataset) |
| Isolation Forest | 0,881 | **+0,120** |

L'Isolation Forest apporte +0,120 AUC par rapport à une simple règle de seuil statistique. Le LOF reste *moins performant* que la baseline univariée — sensibilité aux corrélations du cluster momentum et à la dimensionnalité. Résultat honnête.

---

## 6. Fichiers modifiés

| Fichier | Action | Description |
|---------|--------|-------------|
| `scripts/build_features.py` | **Réécrit** | Deux fenêtres par événement (gt=0/gt=1), 14 features dont 7 nouvelles (`ofi_proxy_1m`, `price_impact_1m`, `conviction_ratio_1m`, `vol_zscore_5m`, `vol_zscore_15m`, `vol_zscore_30m`, `vol_acceleration`), assertion temporelle explicite, checkpoint sur `processed.parquet`, log des événements skippés dans `data/RAW/skipped_events.txt` |
| `scripts/metrics.py` | Inchangé | Fonctions complètes depuis A2 (`recall_at_lamorgia`, `anomaly_auc`, `evaluate`) |
| `scripts/data.py` | Inchangé | Pipeline skrub A2 non modifié |

### Features calculées (`build_features.py`)

| Feature | Calcul | Signal |
|---------|--------|--------|
| `ret_5m` | $\ln(c_{-1}/c_{-6})$ | Momentum court terme |
| `ret_15m` | $\ln(c_{-1}/c_{-16})$ | Momentum moyen terme |
| `ret_60m` | $\ln(c_{-1}/c_{-61})$ | Momentum long terme (0 si < 61 barres) |
| `std_ret_60m` | $\sigma(\ln c_t/c_{t-1})_{60\text{min}}$ | Spike de volatilité |
| `vol_zscore_60m` | $(v_{-1} - \bar{v})/\sigma_v$ | Anomalie de volume vs baseline 2h |
| `vol_zscore_5m` | $(\bar{v}_{-5} - \bar{v})/\sigma_v$ | Z-score volume sur 5 min |
| `vol_zscore_15m` | $(\bar{v}_{-15} - \bar{v})/\sigma_v$ | Z-score volume sur 15 min |
| `vol_zscore_30m` | $(\bar{v}_{-30} - \bar{v})/\sigma_v$ | Z-score volume sur 30 min |
| `vol_acceleration` | $\text{vol\_zscore\_5m} - \text{vol\_zscore\_30m}$ | Accélération du volume (5min vs 30min) |
| `taker_buy_ratio_5m` | $\sum \text{tb}_{-5}/\sum v_{-5}$ | Pression d'achat coordonnée |
| `ofi_proxy_1m` | $(2\cdot\text{tb}_{-1} - v_{-1})/(v_{-1}+\varepsilon)$ | Order flow imbalance estimé |
| `price_impact_1m` | $\|\ln(c_{-1}/o_{-1})\|$ | Impact prix (ratio d'Amihud simplifié) |
| `conviction_ratio_1m` | $\|c_{-1}-o_{-1}\|/(h_{-1}-l_{-1}+\varepsilon)$ | Directionnalité du dernier bar |
| `rush_order_count` | $\sum \text{num\_trades}_{-2}$ | Fréquence de trades (proxy OFI) |

---

## 7. Reproduction

### Prérequis

```bash
pip install scikit-learn pandas numpy pyarrow joblib matplotlib
```

### Étape 1 — Générer le dataset de features

```bash
python scripts/build_features.py
# Output: data/Processed/processed.parquet (3 666 lignes × 20 colonnes)
```

Si `processed.parquet` existe déjà avec les bonnes colonnes, le script saute le calcul (checkpoint).

### Étape 2 — Exécuter le notebook de modélisation

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/02_modeling.ipynb
```

Le notebook exécute les 8 cellules dans l'ordre :
1. Setup et chargement de `processed.parquet`
2. Preprocessing (`RobustScaler`) → `models/scaler_klines.joblib`
3. Isolation Forest → `models/isolation_forest.joblib`
4. Local Outlier Factor
5. Z-score rolling (`vol_zscore_60m`)
6. Analyse de sensibilité (tableau Recall × contamination)
7. Figures → `deliverables/fig_roc_comparison.png`, `fig_recall_vs_contamination.png`, `fig_score_distributions.png`
8. Tableau récapitulatif et comparaison avant/après correction

### Résultats attendus

| Modèle | AUC | Recall@0,005 |
|--------|-----|-------------|
| Isolation Forest | 0,881 | 0,057 |
| Z-score rolling | 0,766 | 0,057 |
| LOF (n=20) | 0,710 | 0,012 |

**Comparaison méthodologique :**

| Évaluation | Dataset | AUC | Ce que ça mesure |
|------------|---------|-----|------------------|
| A2 — La Morgia (artifact) | 584k lignes | 0,9976 | Détection *pendant* le pump |
| A3 — Klines pré-pump (honnête) | 3 666 lignes | 0,881 | Détection *avant* le pump |
