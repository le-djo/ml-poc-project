# Assignment 5 — Application de démonstration

**Rendu :** push sur le dépôt git  
**Fichiers créés :** `scripts/main.py`, `app.py`  
**Lancement :** `python scripts/main.py`

---

## 1. Description de l'application

Application Streamlit interactive de démonstration du pipeline complet de détection de pump & dump non supervisée sur les klines Binance 2018-2021. Conçue pour une présentation orale de 6 minutes : 3-4 minutes de présentation structurée + démo live en temps réel.

---

## 2. Objectif

Présenter de façon cohérente et interactive l'ensemble du pipeline ML :

- **Problème** → pourquoi la détection de P&D est difficile et réglementairement importante (MiCA)
- **Données & features** → 14 features microstructurelles calculées strictement avant `pump_ts`
- **Modèles** → justification des 3 approches non supervisées, résultats honnêtes
- **Démo live** → scorer n'importe quelle fenêtre de marché en temps réel via l'API Binance publique

---

## 3. Fonctionnalités

### Tab 1 — 🎯 Problème & Données
Présentation du contexte : définition du pump & dump, justification de l'approche non supervisée (absence de labels en production, pumps privés non étiquetés), cadre réglementaire MiCA. Stats du dataset (3 666 lignes, 333 événements pump, 14 features, 5 coins). Visualisation des klines brutes (`fig_raw_klines.png`) illustrant l'absence de signal pré-pump détectable.

### Tab 2 — ⚙️ Features & Pipeline
Heatmap de corrélation Pearson (`fig_feature_heatmap.png`) révélant deux clusters : momentum prix (r ≥ 0,87) et microstructure. Table des 14 features en deux colonnes. Encadré d'interprétation sur la complémentarité des clusters.

### Tab 3 — 📊 Résultats des modèles
Métriques finales H1 en `st.metric` :

| Modèle | AUC | Recall@0,005 |
|---|---|---|
| Isolation Forest | 0,881 | 0,057 |
| LOF (n=20) | 0,710 | 0,012 |
| Z-score vol | 0,766 | 0,057 |

Courbes ROC et recall vs contamination côte à côte. Encadré résumant l'hypothèse H1 confirmée (+0,052 AUC, Recall ×4,75).

### Tab 4 — 🔍 Démo live
Score en temps réel d'une fenêtre de marché arbitraire via l'API publique Binance.

---

## 4. Inputs utilisateur (Tab 4)

| Input | Type | Défaut | Description |
|---|---|---|---|
| Symbole Binance | `st.text_input` | `BTCUSDT` | Paire de trading Binance (ex. ETHUSDT, BNBUSDT) |
| Date | `st.date_input` | 2021-06-01 | Date de début de la fenêtre de 120 minutes |

---

## 5. Outputs (Tab 4)

| Output | Format | Description |
|---|---|---|
| Score d'anomalie | `st.metric` | `-IF.decision_function(x)` — positif = anomalie |
| Niveau de risque | Couleur (rouge/orange/vert) | Seuils : >0 élevé, >−0,05 modéré, sinon normal |
| Volume 1-min | `st.bar_chart` | Série temporelle des 120 barres récupérées |
| Features calculées | `st.dataframe` | Dict retourné par `compute_features()` |

---

## 6. Structure technique

```
scripts/main.py          ← point d'entrée obligatoire
    └── subprocess → streamlit run app.py

app.py                   ← application Streamlit (4 tabs)
    ├── scripts/build_features.py :: compute_features()
    ├── models/isolation_forest.joblib
    ├── models/scaler_klines.joblib
    └── deliverables/fig_*.png
```

**Discipline temporelle préservée dans la démo :** `window_end_ms = max(open_time) + 60 000 ms` — aucune donnée future utilisée pour le scoring.

---

## 7. Lancement

```bash
# Installation
pip install -r requirements.txt

# Lancement
python scripts/main.py
```

L'application s'ouvre automatiquement dans le navigateur par défaut sur `http://localhost:8501`.

---

## 8. Limites et pistes d'amélioration

**Limites structurelles :**
- Le signal pré-pump dans les klines 1-min est quasi absent (spike uniquement post-annonce Telegram) — le Recall reste faible aux faibles contaminations (Recall@0,001 ≈ 0,012).
- L'Isolation Forest est entraîné sur 2018-2021 : dérive de régime possible sur le marché actuel.
- La démo live utilise l'API publique Binance sans authentification — limite de débit 1 200 poids/min.

**Pistes d'amélioration :**
- Données aggTrades (ordre par ordre) pour une résolution sub-minute
- Fenêtre glissante en streaming (WebSocket Binance) pour détection quasi-temps-réel
- Modèle de séries temporelles (LSTM Autoencoder) pour capturer les patterns séquentiels
- Recalibration périodique du modèle sur données récentes (drift adaptation)
