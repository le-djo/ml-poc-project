# Assignment 1 — Phase 0 : Exploration et cadrage du projet

---

### Description du projet

Ce projet porte sur la **détection automatique des schémas de pump & dump** dans les marchés de crypto-actifs. Un pump & dump consiste à coordonner artificiellement l'achat massif d'un actif peu liquide (souvent via des groupes Telegram) afin d'en gonfler le cours, puis à revendre rapidement au détriment des acheteurs tardifs.

Ce travail s'inscrit dans un double contexte :

1. **Académique** : projet de cours en machine learning appliqué aux marchés financiers.
2. **Préparation à un stage chez Kaiko** : Kaiko est un fournisseur de données institutionnelles de marché crypto. Ce projet prépare à l'intégration d'un module de surveillance de marché (_Market Surveyor_) dans leur infrastructure.

La pertinence réglementaire est directe : le **Règlement MiCA (Markets in Crypto-Assets), Article 62**, impose aux fournisseurs de services d'actifs numériques de détecter et signaler les abus de marché, dont les manipulations de cours. La capacité à scorer en temps réel les transactions anormales est donc à la fois une exigence de conformité et un avantage concurrentiel.

---

### Définition du problème

Le problème est formulé comme une **détection d'anomalies non supervisée**, et non comme une classification supervisée. Ce choix est délibéré et fondé sur deux contraintes :

1. **Absence d'étiquettes à l'inférence** : en production, aucun système ne dispose en temps réel d'une liste des événements de manipulation passés. Les labels ne peuvent donc pas être utilisés pour entraîner un classifieur déployé en production.
2. **Rareté des événements** : les pompes représentent une fraction infime du flux transactionnel global. Les algorithmes de détection d'anomalies sont conçus précisément pour identifier des observations rares dans un espace de haute dimension, sans avoir besoin de classes équilibrées.

La validation est **post-hoc** : après avoir généré des scores d'anomalie, on mesure si les vrais événements de pump (annotés par La Morgia et al.) ont bien été détectés. Les labels servent uniquement à évaluer le rappel, jamais à entraîner le modèle.

---

### Description du dataset

Trois sources de données sont utilisées, avec des rôles bien distincts :

| Source | Type | Contenu | Usage |
|--------|------|---------|-------|
| **La Morgia et al. 2020** | Log d'événements | 1 110 événements de pump annotés (symbol, group, date, hour, exchange) | Référence ground truth pour la validation post-hoc |
| **Binance REST API** | Données de marché | Klines 1 min + aggTrades sur la fenêtre [pump_ts−2h, pump_ts+30min] | Construction de toutes les features numériques |
| **CoinGecko API** | Contexte cross-sectionnel | market_cap_rank, circulating_supply, catégories | Features de vulnérabilité structurelle de la pièce |

**Point critique** : La Morgia et al. 2020 ne contient **aucune donnée de prix ni de volume**. C'est un journal d'annonces Telegram transformées en horodatages d'événements. Toutes les features numériques exploitables (rendements, volumes, pression acheteuse, etc.) sont **produites par notre propre code** à partir des APIs Binance et CoinGecko.

Le sous-ensemble retenu pour ce projet est filtré sur **Binance uniquement** (520 événements sur 1 110), correspondant à la période 2018–2021. La base symbolique de La Morgia ne contient pas la paire de cotation (ex. : "BRD" et non "BRDBTC") ; en 2018-2021, les cibles de pump sur Binance étaient quasi-exclusivement cotées en BTC, ce qui justifie l'équivalence du filtre Binance ≈ filtre /BTC pour ce dataset.

---

### Description des features

Toutes les features sont calculées sur la fenêtre [pump_ts−2h, pump_ts) — strictement **avant** l'événement, sans fuite de données.

| Feature | Signal attendu |
|---------|---------------|
| `ret_5m` | Log-rendement sur 5 min — détecte l'accélération de prix en fin de phase de pompe |
| `ret_15m` | Log-rendement sur 15 min — vélocité de prix à horizon intermédiaire |
| `ret_60m` | Log-rendement sur 60 min — tendance de prix sur la fenêtre complète |
| `std_ret_60m` | Écart-type des rendements sur 60 min — pic de volatilité anormal |
| `vol_zscore_60m` | Z-score du volume sur 60 min vs baseline historique — anomalie volumique absolue |
| `vol_zscore_dynamic` | Z-score volume avec contrôle heure-du-jour — corrige la saisonnalité intraday pour éviter les faux positifs aux heures de fort volume habituel |
| `taker_buy_ratio_5m` | Part des volumes côté acheteur sur les 5 dernières minutes (aggTrades) — pression d'achat coordonnée |
| `ofi_1m` | Order Flow Imbalance sur 1 min — signal le plus fort pour une coordination d'achat : déséquilibre entre ordres d'achat et de vente agressifs |
| `rush_order_count` | Nombre de trades dans les 2 dernières minutes — spike de fréquence transactionnelle typique d'une ruée |
| `market_cap_rank` | Rang de capitalisation boursière (CoinGecko) — les petites caps sont plus vulnérables à la manipulation |
| `circulating_supply_log` | Log de l'offre en circulation — proxy de liquidité structurelle |

---

### Premières analyses EDA

Les analyses exploratoires sont détaillées dans le notebook [`notebooks/01_eda.ipynb`](../notebooks/01_eda.ipynb). Les quatre observations principales sont les suivantes :

1. **Distribution temporelle croissante** : le nombre d'événements de pump sur Binance augmente d'année en année (111 en 2018, 162 en 2019, 225 en 2020), suggérant une professionnalisation des groupes de manipulation ou un biais de détection croissant lié à l'extension du dataset.

2. **Concentration sur quelques symboles** : BRD, RDN, NXS et NAV dominent avec 20–23 événements chacun sur 520. Ces actifs de faible capitalisation et de faible liquidité sont ciblés de façon répétée par les mêmes groupes, ce qui indique une logique de recyclage des cibles.

3. **Pic d'activité en fin d'après-midi UTC (16h–18h)** : 135 événements à 16h UTC et 101 à 18h UTC. Cette concentration horaire coïncide avec le chevauchement entre les sessions européenne et américaine — période de liquidité maximale favorable à l'exécution rapide d'une vente.

4. **Dominance de deux groupes Telegram** : BPF (139 événements) et CPI (96 événements) représentent à eux deux 45 % des pumps Binance. La concentration du pouvoir de manipulation dans un nombre restreint de groupes confirme la faisabilité d'une détection centrée sur les patterns comportementaux répétitifs.

---

### Objectif business

L'objectif opérationnel est d'intégrer un **module de scoring d'anomalie en temps réel** dans le produit _Kaiko Market Surveyor_. Ce module devrait :

- Produire un score d'anomalie par trade, agrégé sur une fenêtre glissante.
- Déclencher une alerte lorsqu'un seuil de score est dépassé sur un symbole donné.
- Fournir une explication par feature (type SHAP) pour chaque alerte, afin de satisfaire les exigences d'explicabilité imposées aux outils de conformité.

Dans le cadre du **Règlement MiCA, Article 62**, les prestataires de services sur actifs numériques (PSAD) ont l'obligation légale de détecter et signaler les abus de marché, incluant les manipulations de cours telles que les pump & dump. Un outil de scoring automatique répond directement à cette obligation.

**Avantage concurrentiel de Kaiko** : Kaiko agrège des données de marché provenant de plus de 100 exchanges. Notre modèle actuel est limité à Binance (signal monochange), ce qui constitue à la fois une **limite** (angle mort sur les coordinations cross-exchange) et une **opportunité business** : l'extension du modèle aux données multi-exchange de Kaiko permettrait de détecter les schémas de manipulation coordonnés entre plusieurs plateformes — un signal que les acteurs mono-exchange ne peuvent pas observer.

---

### Contexte ML

L'approche retenue est entièrement **non supervisée** :

| Modèle | Type | Caractéristiques |
|--------|------|-----------------|
| **Isolation Forest** | Arbre (ensemble) | Rapide, scalable, efficace en haute dimension. Isole les anomalies en partitionnant aléatoirement l'espace des features. |
| **Local Outlier Factor (LOF)** | Densité (voisinage local) | Détecte les anomalies relatives à leur voisinage local — utile pour des clusters de comportements normaux hétérogènes. |
| **Z-score glissant** | Statistique univariée | Baseline simple sur une feature à la fois (ex. `vol_zscore_60m`). Sert de référence pour évaluer l'apport des modèles multivariés. |

Il n'y a **pas de découpage train/test** au sens classique : l'entraînement se fait sur l'ensemble du dataset, et la sensibilité du modèle est évaluée via une analyse sur le **paramètre de contamination** (proportion estimée d'anomalies dans le dataset). La validation se fait ensuite en comparant les alertes produites avec les labels La Morgia.

---

### Métrique

**Métrique principale : Recall@LaMargia**

$$\text{Recall@LaMargia} = \frac{\text{Nombre d'événements La Morgia détectés comme anomalie}}{\text{Nombre total d'événements La Morgia dans le sous-ensemble Binance}}$$

Cette métrique mesure la proportion des vrais pumps qui ont été signalés par le modèle. Elle est prioritaire car, dans un système de surveillance réglementaire, **ne pas détecter un abus est plus grave que de générer une fausse alerte**.

**Métrique secondaire : AUC (Area Under the ROC Curve)**

Le score d'anomalie continu est comparé à la variable binaire "est un pump" (d'après La Morgia). L'AUC mesure la capacité discriminante globale du score, indépendamment du seuil choisi.

**Pourquoi ne pas utiliser l'accuracy ni le F1 ?**

- **Accuracy** : inutilisable en présence d'un déséquilibre sévère de classes. Avec ~520 pumps sur des millions de ticks de marché, un modèle qui prédit "normal" à 100 % du temps aurait une accuracy proche de 1 sans aucune valeur détective.
- **F1-score** : suppose un seuil de classification optimal connu à l'avance et des classes équilibrées. En détection d'anomalies non supervisée, il n'y a pas de seuil appris — on fait varier le paramètre de contamination — et le F1 dépend fortement de ce choix arbitraire. Le Recall@LaMargia est plus robuste et plus directement aligné sur l'objectif opérationnel.

---

### Hypothèses, risques, limites

#### Hypothèses

- **H1** : Les pumps produisent des spikes détectables de volume et de rendement dans les 2 heures précédant l'annonce publique. C'est la prémisse principale du projet : si la pompe commence avant l'annonce (accumulation discrète), le signal doit être visible dans les features de marché.
- **H2** : Les features comportementales (OFI, taker buy ratio) sont des signaux plus forts que le prix seul. La coordination d'achat laisse une empreinte caractéristique dans le carnet d'ordres, indépendamment de l'amplitude du mouvement de prix.

#### Risques

- **Délisting Binance** : de nombreux symboles présents dans La Morgia (2018-2019) ont été retirés de Binance. Cela entraîne une perte de données irrémédiable. Le pourcentage de symboles perdus sera documenté et inclus dans les résultats.
- **Mapping symbol → CoinGecko** : la correspondance entre le ticker Binance et l'ID CoinGecko est heuristique (ex. : "BRD" → "bread"). Des erreurs de mapping peuvent introduire des features cross-sectionnelles corrompues pour certains symboles.
- **Vue monochange** : le modèle ne voit que Binance. Des schémas de pump coordonnés entre plusieurs exchanges (achat sur Bittrex, vente sur Binance) seraient invisibles à notre système.

#### Limites

- **Vintage des données** : le dataset La Morgia couvre principalement 2018-2020. Les tactiques de manipulation ont pu évoluer (utilisation de DEX, de nouveaux vecteurs de coordination, etc.). Le modèle entraîné sur ces données peut sous-performer sur des pumps plus récents.
- **Biais du survivant** : seuls les symboles encore listés sur Binance au moment de la collecte des klines peuvent être analysés. Les pièces retirées — qui étaient potentiellement parmi les plus manipulées — sont exclues de facto.
- **Labels Telegram uniquement** : La Morgia n'annote que les pumps annoncés publiquement sur Telegram. Les pumps organisés via d'autres canaux (Discord, groupes privés, signaux OTC) ne sont pas dans le ground truth. Le recall calculé est donc une borne inférieure du rappel réel.
