import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "scripts"))
from build_features import compute_features  # noqa: E402

DELIVERABLES = ROOT / "deliverables"
MODELS_DIR   = ROOT / "models"
DATA_PATH    = ROOT / "data" / "Processed" / "processed.parquet"

FEAT_COLS = [
    "ret_5m", "ret_15m", "ret_60m", "std_ret_60m",
    "vol_zscore_60m", "vol_zscore_5m", "vol_zscore_15m", "vol_zscore_30m",
    "vol_acceleration",
    "taker_buy_ratio_5m",
    "ofi_proxy_1m", "price_impact_1m", "conviction_ratio_1m",
    "rush_order_count",
]

KNOWN_EVENTS = {
    # --- DÉTECTIONS RÉUSSIES (dans le dataset d'entraînement, score IF confirmé positif) ---
    "HCBTC — 25 jan 2019 (score IF 0.135, meilleure détection)":               ("HCBTC",   "2019-01-25", "2019-01-25 17:59:00"),
    "BRDBTC — 03 oct 2019 (score IF 0.132, détection forte)":                  ("BRDBTC",  "2019-10-03", "2019-10-03 14:59:00"),
    # --- LIMITE DU MODÈLE (pump réel mais non ou faiblement détecté) ---
    "DUSKBTC — 30 oct 2020 (pump fort, signal pré-pump modéré)":               ("DUSKBTC", "2020-10-30", "2020-10-30 16:00:00"),
    "TNTBTC — 23 jun 2018 (pump documenté, score faible)":                     ("TNTBTC",  "2018-06-23", "2018-06-23 15:59:00"),
    "BRDBTC — 04 oct 2018 (pump réel, NON détecté)":                           ("BRDBTC",  "2018-10-04", "2018-10-04 18:59:00"),
    # --- MARCHÉ NORMAL (modèle dit NORMAL, correct) ---
    "BTCUSDT — 01 jun 2020 (marché calme, doit scorer NORMAL)":                ("BTCUSDT", "2020-06-01"),
    "ETHUSDT — 15 mar 2019 (marché normal, référence)":                        ("ETHUSDT", "2019-03-15"),
    # --- SYMBOLE PERSONNALISÉ ---
    "Symbole personnalisé...":                                                  None,
}
NORMAL_KEY = "BTCUSDT — 01 jun 2020 (marché calme, doit scorer NORMAL)"


# ── Cached helpers ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def build_heatmaps():
    if not DATA_PATH.exists():
        return None, None
    df = pd.read_parquet(DATA_PATH)
    results = {}
    for name, feats, title, tmp in [
        ("price", ["ret_5m", "ret_15m", "ret_60m", "std_ret_60m", "price_impact_1m"],
         "Cluster momentum prix (r≥0.87)\n— même signal, 5 axes de partitionnement",
         "/tmp/heatmap_price.png"),
        ("micro", ["vol_zscore_60m", "taker_buy_ratio_5m", "ofi_proxy_1m",
                   "conviction_ratio_1m", "rush_order_count"],
         "Cluster microstructure\n— signal indépendant",
         "/tmp/heatmap_micro.png"),
    ]:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(df[feats].corr(), annot=True, fmt=".2f", cmap="coolwarm",
                    vmin=-1, vmax=1, ax=ax, square=True)
        ax.set_title(title, fontsize=9)
        plt.tight_layout()
        fig.savefig(tmp, dpi=120, bbox_inches="tight")
        plt.close(fig)
        results[name] = tmp
    return results["price"], results["micro"]


@st.cache_resource(show_spinner=False)
def load_models():
    scaler = joblib.load(MODELS_DIR / "scaler_klines.joblib")
    model  = joblib.load(MODELS_DIR / "isolation_forest.joblib")
    return scaler, model


@st.cache_data(show_spinner=False)
def load_pump_stats():
    df = pd.read_parquet(DATA_PATH)
    pumps = df[df["gt"] == 1][FEAT_COLS]
    return pumps.describe(percentiles=[0.25, 0.5, 0.75])


@st.cache_resource(show_spinner=False)
def load_shap_explainer():
    import shap as _shap
    _, model = load_models()
    return _shap.TreeExplainer(model)


# ── Page config & sidebar ──────────────────────────────────────────────────────

st.set_page_config(page_title="Pump & Dump Detection", layout="wide",
                   initial_sidebar_state="expanded")

PAGES = [
    "🎯 Le problème",
    "🔬 Notre approche",
    "📊 Résultats",
    "🔍 Démo live",
    "⚠️ Limites & Perspectives",
    "📖 Glossaire",
]
page = st.sidebar.radio("Navigation", PAGES)
st.sidebar.markdown("---")
st.sidebar.caption("Pump & Dump Detection — Binance 2018-2021")
st.sidebar.caption("IF AUC=0.881 | Recall ×4.75")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Le problème
# ══════════════════════════════════════════════════════════════════════════════
if page == "🎯 Le problème":
    st.title("Détecter les manipulations de marché avant qu'elles se produisent")

    st.info(
        "En 2018-2021, des groupes Telegram coordonnaient des pump & dump sur Binance : "
        "achat massif sur signal → cours artificiel → vente immédiate → retardataires piégés. "
        "MiCA (2024) impose désormais aux plateformes crypto une surveillance active — "
        "amende jusqu'à €5M ou 3% du CA annuel en cas de manquement. "
        "Kaiko, fournisseur de données institutionnelles, a besoin d'un système de détection automatisé."
    )

    k1, k2, k3 = st.columns(3)
    k1.metric("Manipulations analysées", "333 événements")
    k2.metric("Pouvoir de détection", "AUC 0.881", delta="+0.052 vs baseline initiale")
    k3.metric("Gain en recall", "×4.75", delta="après correction méthodologique")

    st.subheader("Pourquoi c'est difficile")
    c_left, c_right = st.columns(2)
    with c_left:
        st.error(
            "❌ **Approche supervisée impossible**\n\n"
            "En production, aucun label n'est disponible en temps réel. "
            "Les pumps privés (Discord, groupes fermés) ne sont jamais répertoriés. "
            "Un classifieur supervisé ne généralisera pas."
        )
    with c_right:
        st.success(
            "✅ **Notre approche : détection d'anomalies non supervisée**\n\n"
            "Le modèle apprend ce qu'est un marché normal, "
            "puis flagge ce qui s'en écarte — sans jamais voir de labels."
        )

    st.subheader("Le défi central : le signal arrive APRÈS l'annonce")
    st.image(str(DELIVERABLES / "fig_raw_klines.png"),
             caption="**Sur ces 3 exemples réels, le spike de volume arrive APRÈS pump_ts "
                     "(ligne rouge = moment de l'annonce Telegram). Le modèle doit détecter "
                     "les rares cas où une accumulation pré-annonce est visible.**")
    st.warning(
        "C'est pourquoi notre Recall est de 5.7% à faible contamination — non pas une défaillance "
        "du modèle, mais une limite structurelle du signal disponible dans les klines 1-min."
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Notre approche
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔬 Notre approche":
    st.title("Pipeline de détection : de la donnée brute au score d'anomalie")

    st.markdown("""
**Pipeline complet :**
`API Binance` → `333 klines parquet` → `14 features` → `Isolation Forest` → `Score d'anomalie`

**Données d'entraînement :**
- 333 fenêtres pré-pump (gt=1) : 2h avant chaque événement La Morgia
- 3 333 fenêtres marché normal (gt=0) : BTCUSDT/ETHUSDT/BNBUSDT/XRPUSDT/LTCUSDT, dates aléatoires 2018-2021
- Total : **3 666 observations**, taux de pump réel : **9.1%**
""")

    st.subheader("Découverte clé : notre baseline était biaisée")
    with st.expander("🔍 Comprendre le problème de baseline (cliquer pour développer)"):
        st.markdown("""
**Première version :** nos fenêtres "normales" (gt=0) venaient des mêmes événements pump —
l'heure calme juste avant chaque pump.

**Le problème :** ces fenêtres "calmes" étaient déjà légèrement anormales
(mêmes coins, même période, contexte pré-manipulation).

| Groupe | Score moyen IF |
|--------|---------------|
| Marché normal réel (BTC/ETH) | +0.101 |
| Notre ancien "normal" (heure calme pump) | +0.027 |
| Fenêtre pré-pump (gt=1) | +0.021 |

L'IF comparait anomal vs légèrement-moins-anomal → frontière impossible à apprendre.

**La correction :** collecter 3 000 fenêtres de marché réel via l'API Binance.
Résultat : AUC 0.8299 → **0.8815**, Recall ×4.75.
""")

    st.subheader("Features")
    fc_left, fc_right = st.columns(2)
    with fc_left:
        st.markdown("📥 **Features collectées (Binance klines brutes)**")
        st.table(pd.DataFrame([
            ("ret_5m",         "(close_t - close_{t-5}) / close_{t-5}", "Momentum 5 min"),
            ("ret_15m",        "idem 15 min",                           "Momentum 15 min"),
            ("ret_60m",        "idem 60 min",                           "Momentum 60 min"),
            ("std_ret_60m",    "écart-type returns sur 60 min",         "Volatilité réalisée"),
            ("vol_zscore_60m", "(vol - μ_60) / σ_60",                   "Anomalie volumique"),
        ], columns=["Feature", "Calcul", "Signal"]))
    with fc_right:
        st.markdown("⚙️ **Features construites (feature engineering)**")
        st.table(pd.DataFrame([
            ("taker_buy_ratio_5m",  "takerBuyBase / volume",               "Pression acheteuse"),
            ("ofi_proxy_1m",        "(2×takerBuyBase - volume) / volume",  "Order Flow Imbalance (Cont et al.)"),
            ("price_impact_1m",     "abs(close-open) / volume",            "Illiquidité d'Amihud (1986)"),
            ("conviction_ratio_1m", "abs(close-open) / (high-low)",        "Directionnalité"),
            ("rush_order_count",    "nb trades dans les 2 dernières min",  "Rafales agressives"),
        ], columns=["Feature", "Formule", "Inspiration"]))

    st.subheader("Structure de corrélation des features")
    hmap_price, hmap_micro = build_heatmaps()
    if hmap_price:
        hc1, hc2 = st.columns(2)
        with hc1:
            st.image(hmap_price, use_container_width=True)
        with hc2:
            st.image(hmap_micro, use_container_width=True)
    else:
        st.image(str(DELIVERABLES / "fig_feature_heatmap.png"), use_container_width=True)
    st.info("""
**Ce qu'on retient :** Les 5 features de prix bougent ensemble (r≥0.87) — elles mesurent le même phénomène sous 5 angles.
L'Isolation Forest en profite : chaque axe redondant lui donne une chance supplémentaire d'isoler une anomalie.
Supprimer ces corrélations (PCA) détruit ce signal → AUC −0.087. On les garde toutes.
""")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Résultats
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Résultats":
    st.title("Performances des 3 modèles")

    m1, m2, m3 = st.columns(3)
    m1.metric("Isolation Forest — AUC", "0.881", delta="+0.052 vs baseline")
    m2.metric("LOF (n=20) — AUC", "0.710")
    m3.metric("Z-score vol — AUC", "0.766")

    fig_left, fig_right = st.columns(2)
    with fig_left:
        st.image(str(DELIVERABLES / "fig_roc_comparison.png"),
                 caption="Courbes ROC — IF domine sur tout le spectre de seuils.",
                 use_container_width=True)
    with fig_right:
        st.image(str(DELIVERABLES / "fig_recall_vs_contamination.png"),
                 caption="Recall@LaMargia vs contamination — comportement aux faibles prévalences.",
                 use_container_width=True)

    st.subheader("Comprendre le Recall@LaMargia")
    st.info("""
**Comment ça marche concrètement :**
Le modèle ne connaît pas les labels. On lui donne 3 666 fenêtres à scorer.
On lui fixe un budget : lever les top 0.5% des scores les plus suspects = **~18 alertes**.
Parmi ces alertes : combien tombent sur un vrai pump ? C'est le Recall.
À c=0.5% → **19 vrais pumps détectés sur 333** → ×12 mieux qu'un tirage aléatoire (qui en trouverait ~1.6).
""")
    with st.expander("📐 Voir la formule exacte"):
        st.latex(
            r"\text{Recall} = \frac{\sum_{i: y_i=1} \mathbf{1}[\hat{y}_i=1]}{\sum_i y_i}"
            r"\quad \text{où} \quad"
            r"\hat{y}_i = \mathbf{1}[f(x_i) \leq \tau_c], \quad \tau_c = \text{percentile}_c(\mathbf{f})"
        )

    st.subheader("Analyse par modèle")
    e1, e2, e3 = st.columns(3)
    with e1:
        with st.expander("🌲 Isolation Forest — AUC 0.881", expanded=True):
            st.markdown("""
Partitionne chaque feature indépendamment par arbres aléatoires.
Un pump s'isole en **moins de coupures** qu'une observation normale.

- ✅ Immune à la multicollinéarité du cluster momentum
- ✅ À FPR=5%, capture déjà ~60% des pumps
- ✅ Meilleur recall à toutes les contaminations
- ⚠️ Boîte noire — pas d'explication par feature
""")
    with e2:
        with st.expander("📍 LOF (n=20) — AUC 0.710", expanded=True):
            st.markdown("""
Compare la densité locale d'un point à celle de ses k=20 voisins.

- ✅ Détecte les anomalies locales invisibles à l'IF
- ❌ Distances distordues par les 5 features corrélées de momentum
- ❌ Courbe ROC proche de la diagonale aux faibles FPR
- ❌ Recall@0.005 = 0.012 vs 0.057 pour l'IF
""")
    with e3:
        with st.expander("📈 Z-score vol — AUC 0.766", expanded=True):
            st.markdown("""
Score univarié sur `vol_zscore_60m`. Zéro entraînement.

- ✅ Interprétable directement (nb d'écarts-types)
- ✅ Déployable en streaming sans infrastructure ML
- ❌ Ignore les 13 autres features
- ❌ Aveugle aux pumps sans spike de volume
""")

    st.markdown("""
**Comment lire ce graphe :** Chaque violon montre la distribution des scores d'anomalie.
Bleu = fenêtres de marché normal. Rose = fenêtres pré-pump.
**Plus les deux violons sont séparés, mieux le modèle distingue les deux.**
Isolation Forest : normaux concentrés à −0.10, pumps étalés vers +0.25 → meilleure séparation.
LOF : les deux se chevauchent → quasi-inutile sur ce problème.
""")
    st.image(str(DELIVERABLES / "fig_score_distributions.png"),
             caption="Distribution des scores : IF montre la meilleure séparation gt=0 vs gt=1",
             use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Démo live
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Démo live":
    st.title("Scorer un événement en temps réel")

    pump_stats = load_pump_stats()

    event_choice  = st.selectbox("Événement à analyser", options=list(KNOWN_EVENTS.keys()))
    event_val     = KNOWN_EVENTS[event_choice]
    is_known_pump = (event_val is not None) and (len(event_val) > 2)

    # Reset state when the selection changes
    if st.session_state.get("_last_event") != event_choice:
        st.session_state["scored"] = False
        st.session_state["reveal"] = False
        st.session_state["_last_event"] = event_choice

    if event_val is None:
        ci1, ci2 = st.columns(2)
        with ci1:
            symbol = st.selectbox("Symbole Binance", [
                "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "LTCUSDT",
                "SOLUSDT", "ADAUSDT", "DOTUSDT", "MATICUSDT", "LINKUSDT",
                "XRPBTC", "ETHBTC", "BNBBTC", "LTCBTC",
            ])
        with ci2:
            date = st.date_input("Date", value=datetime(2021, 6, 1))
    else:
        symbol = event_val[0]
        date   = datetime.strptime(event_val[1], "%Y-%m-%d").date()
        if len(event_val) > 2:
            _pts = pd.Timestamp(event_val[2], tz="UTC")
            _sdt = _pts - pd.Timedelta(hours=2)
            st.caption(
                f"Symbole : **{symbol}** — "
                f"Fenêtre : **{_sdt.strftime('%H:%M')} – {_pts.strftime('%H:%M')} UTC** ({date})"
            )
        else:
            st.caption(f"Symbole : **{symbol}** — Date : **{date}** (fenêtre midi UTC)")

    if st.button("🔍 Analyser"):
        st.session_state["reveal"] = False
        if event_val is not None and len(event_val) > 2:
            _pump_ts_ms = int(pd.Timestamp(event_val[2], tz="UTC").timestamp() * 1000)
            start_ms    = _pump_ts_ms - 2 * 3600 * 1000
        else:
            _pump_ts_ms = None
            start_ms    = int(datetime(date.year, date.month, date.day, 12, 0).timestamp() * 1000)

        with st.spinner("Récupération des klines Binance..."):
            try:
                resp = requests.get(
                    "https://api.binance.com/api/v3/klines",
                    params={"symbol": symbol.upper(), "interval": "1m",
                            "startTime": start_ms, "limit": 120},
                    timeout=10,
                )
                resp.raise_for_status()
                raw = resp.json()
            except Exception as e:
                st.error(f"Erreur API Binance : {e}")
                st.stop()

        if not isinstance(raw, list) or len(raw) < 30:
            st.error("Données insuffisantes pour ce symbole/date (< 30 barres retournées). "
                     "Ce coin est peut-être délisté sur Binance.")
            st.stop()

        klines = pd.DataFrame(raw, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_vol", "num_trades",
            "taker_buy_base", "taker_buy_quote", "ignore",
        ])
        for _c in ["open_time", "close_time"]:
            klines[_c] = klines[_c].astype(np.int64)
        for _c in ["open", "high", "low", "close", "volume",
                   "taker_buy_base", "taker_buy_quote", "quote_asset_vol"]:
            klines[_c] = klines[_c].astype(float)
        klines["num_trades"] = klines["num_trades"].astype(int)

        window_end_ms = _pump_ts_ms if _pump_ts_ms is not None else int(klines["open_time"].max()) + 60_000
        feats = compute_features(klines, window_end_ms)

        if feats is None:
            st.error("Données insuffisantes (compute_features a retourné None — < 30 barres valides).")
            st.stop()

        try:
            scaler, model = load_models()
        except FileNotFoundError:
            st.error("Modèles introuvables dans models/. Exécutez d'abord notebooks/02_modeling.ipynb.")
            st.stop()

        feat_vec    = np.array([[feats[c] for c in FEAT_COLS]], dtype=float)
        feat_scaled = scaler.transform(feat_vec)
        raw_score   = float(model.decision_function(feat_scaled)[0])

        st.session_state["scored"]         = True
        st.session_state["_klines"]        = klines
        st.session_state["_feats"]         = feats
        st.session_state["_anomaly_score"] = -raw_score
        st.session_state["_window_end_ms"] = window_end_ms
        st.session_state["_is_known_pump"] = is_known_pump
        st.session_state["_scored_symbol"] = symbol.upper()

    # ── Display results (persists across reruns via session_state) ─────────────
    if st.session_state.get("scored"):
        klines        = st.session_state["_klines"]
        feats         = st.session_state["_feats"]
        anomaly_score = st.session_state["_anomaly_score"]
        window_end_ms = st.session_state["_window_end_ms"]
        is_known_pump = st.session_state["_is_known_pump"]
        scored_symbol = st.session_state["_scored_symbol"]

        if anomaly_score > 0:
            st.error(f"🔴 ANOMALIE DÉTECTÉE — score : {anomaly_score:.4f}")
        elif anomaly_score > -0.05:
            st.warning(f"🟡 SIGNAL MODÉRÉ — score : {anomaly_score:.4f}")
        else:
            st.success(f"🟢 NORMAL — score : {anomaly_score:.4f}")

        st.caption(
            "Le score IF est négatif pour les observations normales (faciles à isoler) "
            "et positif pour les anomalies (isolées en peu de coupures)."
        )

        if is_known_pump and anomaly_score <= 0:
            st.info(
                "ℹ️ **Résultat attendu pour ce pump** — Le modèle analyse la fenêtre PRÉ-pump "
                "(2h avant l'annonce Telegram). "
                "Ce pump n'a pas laissé de signal d'accumulation détectable dans les klines 1-min avant l'annonce. "
                "La figure ci-dessous le confirme : le spike de volume est post-annonce (à droite de la ligne rouge). "
                "C'est la limite structurelle documentée dans **Limites & Perspectives**."
            )

        pump_ts_dt  = pd.to_datetime(window_end_ms, unit="ms", utc=True)
        pump_ts_str = pump_ts_dt.strftime("%H:%M UTC")
        timestamps  = pd.to_datetime(klines["open_time"], unit="ms", utc=True)

        fig_vol = go.Figure()
        fig_vol.add_trace(go.Bar(x=timestamps, y=klines["volume"],
                                 marker_color="#4C9BE8", name="Volume"))
        fig_vol.add_shape(type="line",
                          x0=pump_ts_dt, x1=pump_ts_dt, y0=0, y1=1, yref="paper",
                          line=dict(color="red", dash="dash", width=2))
        fig_vol.add_annotation(x=pump_ts_dt, y=0.97, yref="paper",
                               text="pump_ts", showarrow=False,
                               font=dict(color="red", size=12), xanchor="left")
        fig_vol.update_layout(
            title=f"Volume 1-min — ⏰ ligne rouge = moment de l'annonce Telegram ({pump_ts_str})",
            xaxis_title="Heure (UTC)", yaxis_title="Volume",
            showlegend=False, margin=dict(t=50, b=20),
        )
        st.plotly_chart(fig_vol, use_container_width=True)

        n_rows = len(klines)
        st.info(
            f"Analyse : {n_rows} minutes de klines Binance | 14 features calculées | "
            "comparé à 3 333 fenêtres de marché normal (BTC/ETH/BNB/XRP/LTC, 2018-2021)."
        )

        # Volume trend (change 2C)
        if n_rows >= 60:
            first_50_mean = klines["volume"].iloc[:50].mean()
            last_10_mean  = klines["volume"].iloc[-10:].mean()
            ratio = last_10_mean / (first_50_mean + 1e-9)
            if ratio > 3:
                st.warning(f"📈 Volume en hausse dans les 10 dernières minutes "
                           f"(×{ratio:.1f} vs début de fenêtre) — signal d'accumulation visible.")
            elif ratio > 1.5:
                st.info(f"📊 Légère hausse de volume en fin de fenêtre (×{ratio:.1f}) — signal faible.")
            else:
                st.success(f"📉 Volume stable sur toute la fenêtre (×{ratio:.1f}) — "
                           "pas d'accumulation détectable.")

        # ── Feature 1 : Reveal post-pump ──────────────────────────────────────
        if st.button("👁️ Révéler ce qui s'est passé après l'annonce"):
            st.session_state["reveal"] = True

        if st.session_state.get("reveal"):
            extended_end_ms = window_end_ms + 30 * 60 * 1000
            with st.spinner("Récupération des klines post-annonce..."):
                try:
                    resp_ext = requests.get(
                        "https://api.binance.com/api/v3/klines",
                        params={"symbol": scored_symbol, "interval": "1m",
                                "startTime": int(klines["open_time"].min()),
                                "endTime":   extended_end_ms,
                                "limit":     1000},
                        timeout=10,
                    )
                    resp_ext.raise_for_status()
                    raw_ext = resp_ext.json()
                except Exception as e:
                    st.error(f"Erreur API Binance (révélation) : {e}")
                    raw_ext = None

            if raw_ext and isinstance(raw_ext, list) and len(raw_ext) > 0:
                kl_ext = pd.DataFrame(raw_ext, columns=[
                    "open_time", "open", "high", "low", "close", "volume",
                    "close_time", "quote_asset_vol", "num_trades",
                    "taker_buy_base", "taker_buy_quote", "ignore",
                ])
                kl_ext["open_time"]      = kl_ext["open_time"].astype(np.int64)
                kl_ext["volume"]         = kl_ext["volume"].astype(float)
                kl_ext["taker_buy_base"] = kl_ext["taker_buy_base"].astype(float)

                ts_all    = pd.to_datetime(kl_ext["open_time"], unit="ms", utc=True)
                mask_pre  = kl_ext["open_time"] < window_end_ms
                mask_post = ~mask_pre
                max_vol   = float(kl_ext["volume"].max())

                sell_ratio_post = 1.0 - (
                    kl_ext.loc[mask_post, "taker_buy_base"] /
                    (kl_ext.loc[mask_post, "volume"] + 1e-9)
                )

                fig_rev = go.Figure()
                fig_rev.add_trace(go.Bar(
                    x=ts_all[mask_pre],  y=kl_ext.loc[mask_pre,  "volume"],
                    marker_color="#4C9BE8", name="Pré-pump (vu par le modèle)",
                ))
                fig_rev.add_trace(go.Bar(
                    x=ts_all[mask_post], y=kl_ext.loc[mask_post, "volume"],
                    marker_color="#E84C4C", name="Post-annonce (révélé)",
                ))
                fig_rev.add_trace(go.Scatter(
                    x=ts_all[mask_post], y=sell_ratio_post * max_vol,
                    name="Pression vendeuse proxy (1−taker_buy_ratio)",
                    mode="lines", line=dict(color="orange", width=2),
                ))
                fig_rev.add_shape(type="line",
                                  x0=pump_ts_dt, x1=pump_ts_dt, y0=0, y1=1, yref="paper",
                                  line=dict(color="red", dash="dash", width=2))
                fig_rev.add_annotation(
                    x=pump_ts_dt, y=max_vol * 0.9,
                    text="← Le modèle n'a vu que cette partie",
                    showarrow=False, font=dict(color="red", size=11),
                )
                fig_rev.update_layout(
                    title="Révélation : volume AVANT et APRÈS l'annonce Telegram",
                    xaxis_title="Heure (UTC)", yaxis_title="Volume",
                    margin=dict(t=50, b=20),
                )
                st.plotly_chart(fig_rev, use_container_width=True)
                st.caption(
                    "La ligne orange = pression vendeuse estimée (1 − taker_buy_ratio). "
                    "Montée post-spike = les holders vendent sur le pic = signature du dump."
                )

                pre_vols  = kl_ext.loc[mask_pre,  "volume"]
                post_vols = kl_ext.loc[mask_post, "volume"]
                if len(post_vols) > 0 and len(pre_vols) > 0 and pre_vols.mean() > 0:
                    if post_vols.max() > 3 * pre_vols.mean():
                        if anomaly_score > 0:
                            st.error("🔴 Spike confirmé post-annonce — le modèle avait raison de lever une alerte.")
                        else:
                            st.warning(
                                "⚠️ En rétrospection, une hausse de volume a bien eu lieu après l'annonce — "
                                "mais le signal pré-annonce était insuffisant pour lever une alerte ferme."
                            )
                    else:
                        st.info("ℹ️ Pas de spike majeur post-annonce sur cette fenêtre 30 min.")

        # ── Feature 2 : Enriched feature table ────────────────────────────────
        st.subheader("Features calculées")

        # Threshold legend (change 2A)
        col1, col2, col3 = st.columns(3)
        col1.error("🔴 Élevé = au-dessus du 75e percentile pump\n(valeur rare, typique des vrais pumps)")
        col2.warning("🟡 Modéré = entre médiane et 75e percentile\n(signal présent mais pas extrême)")
        col3.success("🟢 Normal = en dessous de la médiane pump\n(comportement habituel du marché)")

        rows = []
        for feat in FEAT_COLS:
            val = feats.get(feat, float("nan"))
            p25 = pump_stats.loc["25%", feat]
            p50 = pump_stats.loc["50%", feat]
            p75 = pump_stats.loc["75%", feat]
            if val > p75:
                signal = "🔴 Élevé — typique d'un pump"
            elif val > p50:
                signal = "🟡 Au-dessus de la médiane pump"
            elif val > p25:
                signal = "🟢 Dans la norme"
            else:
                signal = "🟢 Faible"
            rows.append({
                "Feature":      feat,
                "Valeur":       f"{val:.2f}",
                "Médiane pump": f"{p50:.2f}",
                "Signal":       signal,
            })

        feat_table = pd.DataFrame(rows)
        st.dataframe(feat_table, hide_index=True, use_container_width=True)
        st.caption(
            "Interprétation basée sur la distribution des 333 pumps documentés (La Morgia 2020). "
            "Plusieurs signaux 🔴 simultanés renforcent la fiabilité d'une alerte."
        )

        n_red = sum(
            1 for feat in FEAT_COLS
            if feats.get(feat, float("nan")) > pump_stats.loc["75%", feat]
        )
        _interp = (
            "— pattern composite détecté, alerte justifiée." if n_red >= 4
            else "— signal insuffisant pour une alerte ferme." if n_red < 2
            else "— signal modéré, analyse complémentaire recommandée."
        )
        st.markdown(f"""
**{n_red} / {len(FEAT_COLS)} features au-dessus du 75e percentile pump**

| Seuil | Interprétation |
|-------|----------------|
| 0–1 🟢 | Signal faible — probablement du bruit |
| 2–3 🟡 | Signal modéré — surveiller, pas encore concluant |
| 4+ 🔴 | Signal fort — pattern composite typique d'un pump |

→ **Situation actuelle : {n_red} feature(s) 🔴** {_interp}
""")

        if -0.05 < anomaly_score <= 0:
            st.subheader("🔍 Guide d'analyse — signal modéré")
            st.markdown("""
Le modèle hésite. Voici les 3 questions à poser pour décider :

**1. Combien de features 🔴 simultanées ?**
≥ 3 features au-dessus du 75e percentile pump = pattern suspect, même avec score modéré.

**2. Y a-t-il une accélération volumétrique récente ?**
`vol_acceleration > 0` **ET** `taker_buy_ratio_5m > 0.6` = pression acheteuse qui s'accélère = signal fort.

**3. Le graphe montre-t-il une montée progressive dans les 30 dernières minutes ?**
Un pump avec accumulation visible se traduit par des barres croissantes en fin de fenêtre pré-pump.
Si oui → escalader vers un analyste senior.
""")
            vol_acc = feats.get("vol_acceleration", 0.0)
            tbr_5m  = feats.get("taker_buy_ratio_5m", 0.0)
            if vol_acc > 0 and tbr_5m > 0.6:
                st.warning("⚠️ Volume accélérant + pression acheteuse élevée détectés — surveiller de près.")
            elif n_red >= 3:
                st.warning(f"⚠️ {n_red} features au-dessus du 75e percentile pump — signal composite non négligeable.")
            else:
                st.info("ℹ️ Signal modéré sans pattern composite clair — probablement du bruit de marché.")

        # ── Feature 4 : SHAP ──────────────────────────────────────────────────
        st.subheader("Explication SHAP — pourquoi ce score ?")
        try:
            _explainer = load_shap_explainer()
            _shap_vals = _explainer.shap_values(feat_scaled)[0]   # shape (14,)
            _shap_df   = pd.DataFrame({
                "Feature": FEAT_COLS,
                "SHAP":    _shap_vals,
            }).sort_values("SHAP")
            _fig_shap = go.Figure(go.Bar(
                x=_shap_df["SHAP"],
                y=_shap_df["Feature"],
                orientation="h",
                marker_color=["#E84C4C" if v > 0 else "#4C9BE8" for v in _shap_df["SHAP"]],
            ))
            _fig_shap.update_layout(
                title="Contribution de chaque feature au score d'anomalie (SHAP)",
                xaxis_title="Valeur SHAP (positif = pousse vers anomalie, négatif = pousse vers normal)",
                margin=dict(t=55, b=20, l=180),
                height=420,
            )
            st.plotly_chart(_fig_shap, use_container_width=True)
            st.caption(
                "SHAP explique POURQUOI ce score. "
                "Une feature rouge a poussé le modèle vers 'anomalie'. "
                "Une feature bleue a poussé vers 'normal'."
            )
        except Exception as _e:
            st.caption(f"SHAP indisponible : {_e}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — Limites & Perspectives
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚠️ Limites & Perspectives":
    st.title("Ce qu'on n'a pas résolu — et comment aller plus loin")

    st.subheader("Limites identifiées")

    with st.expander("📉 Limite 1 : la plupart des pumps ne laissent pas de signal pré-pump"):
        st.markdown("""
Sur les 3 exemples raw_klines, le spike de volume arrive après pump_ts dans les 3 cas.
Le modèle détecte les pumps avec accumulation pré-annonce — une minorité.

**Recall@0.005 = 5.7% : 19 pumps détectés sur 333.**

Ce n'est pas une défaillance — c'est la réalité du signal disponible dans les klines 1-min.
Les groupes Telegram diffusaient le signal à des milliers de membres simultanément,
le délai entre décision d'achat et annonce est inférieur à la résolution d'une barre 1-min.
""")

    with st.expander("⏰ Limite 2 : données de 2018-2021, marché moderne différent"):
        st.markdown("""
Les pumps 2018-2021 ciblaient des micro-caps BTC sur Binance.
Le marché crypto 2024+ est différent : stablecoins, perpétuels, CEX/DEX mixtes.

Le modèle n'a pas été validé sur des données récentes.
Une validation out-of-time sur 2022-2024 serait nécessaire avant tout déploiement.
""")

    with st.expander("🔗 Limite 3 : features mono-exchange"):
        st.markdown("""
Toutes les features viennent de Binance uniquement.
Les manipulations cross-exchange (achat sur Binance, vente sur OKX) sont invisibles.

Kaiko surveille 150+ exchanges — la vraie valeur ajoutée est dans les features cross-exchange :
spread bid-ask inter-venues, divergences de prix entre plateformes, flux d'ordre net agrégé.
""")

    st.subheader("Pistes d'amélioration")
    p1, p2, p3 = st.columns(3)
    with p1:
        st.markdown("🔬 **Données**")
        st.markdown("""
- Données tick-level (aggTrades) sur coins encore listés
- Features cross-exchange : spread bid-ask inter-venues
- Labels récents (2022-2024) pour validation out-of-time
""")
    with p2:
        st.markdown("⚙️ **Modélisation**")
        st.markdown("""
- Fenêtre glissante en streaming (détection en temps réel)
- Autoencoder LSTM pour capturer les patterns temporels
- Active learning : l'analyste confirme/rejette les alertes → amélioration continue
""")
    with p3:
        st.markdown("🏢 **Déploiement Kaiko**")
        st.markdown("""
- Score continu par trade dans Market Surveyor
- Seuil adaptatif par exchange (contamination calibrée par venue)
- Audit trail SHAP pour conformité MiCA Article 62
""")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — Glossaire
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📖 Glossaire":
    st.title("Référence technique")

    st.subheader("1. Features collectées (source : Binance klines 1-min)")
    st.table(pd.DataFrame([
        ("ret_5m",         "(close_t − close_{t−5}) / close_{t−5}", "Momentum prix sur 5 min"),
        ("ret_15m",        "idem 15 min",                           "Momentum prix sur 15 min"),
        ("ret_60m",        "idem 60 min",                           "Momentum prix sur 60 min"),
        ("std_ret_60m",    "écart-type des returns minute sur 60 min", "Volatilité réalisée"),
        ("vol_zscore_60m", "(vol_t − μ₆₀) / σ₆₀",                  "Anomalie volumique vs baseline 60 min"),
    ], columns=["Feature", "Calcul", "Interprétation"]))

    st.subheader("2. Features construites (feature engineering)")
    st.table(pd.DataFrame([
        ("taker_buy_ratio_5m",  "takerBuyBase / volume",               "Pression acheteuse"),
        ("ofi_proxy_1m",        "(2×takerBuyBase − volume) / volume",  "Order Flow Imbalance (Cont et al.)"),
        ("price_impact_1m",     "abs(close − open) / volume",          "Illiquidité d'Amihud (1986)"),
        ("conviction_ratio_1m", "abs(close − open) / (high − low)",    "Directionnalité de la bougie"),
        ("rush_order_count",    "nb trades dans les 2 dernières min",  "Rafales agressives"),
    ], columns=["Feature", "Formule", "Source"]))

    st.subheader("3. Features testées et rejetées")
    st.table(pd.DataFrame([
        ("vol_zscore_5m/15m/30m",  "AUC IF −0.010 — volume redondant noie le signal IF"),
        ("vol_acceleration",       "sep_ratio 0.61× — signal marginal"),
        ("market_cap_rank (CoinGecko)", "API indisponible pour coins délistés 2018-2020"),
        ("aggTrades OFI exact",    "Binance purge aggTrades coins délistés — 404 confirmé sur toutes les paires"),
        ("PCA prix (price_PC1)",   "AUC −0.087 — IF exploite les 4.8% variance résiduelle inter-features"),
        ("Fenêtre 30min gt=1",     "AUC −0.21 — signal distribué sur 2h, pas concentré"),
    ], columns=["Feature", "Raison du rejet"]))

    st.subheader("4. Métriques")
    st.table(pd.DataFrame([
        ("AUC-ROC",          "P(score_pump > score_normal) pour une paire aléatoire",
         "Comparaison threshold-free entre modèles"),
        ("Recall@LaMargia",  "Pumps détectés / total pumps vrais, à budget d'alertes fixé",
         "Coût asymétrique : manquer un pump > fausse alarme"),
        ("Contamination",    "Fraction du dataset flaggée comme anomalie",
         "Budget d'alertes analyste — paramètre opérationnel"),
    ], columns=["Métrique", "Définition", "Pourquoi ce projet"]))
    with st.expander("📐 Formule AUC-ROC"):
        st.latex(r"\text{AUC} = \int_0^1 \text{TPR}(\text{FPR}^{-1}(t))\, dt")
    with st.expander("📐 Formule Recall@LaMargia"):
        st.latex(
            r"\text{Recall} = \frac{\sum_{i: y_i=1} \mathbf{1}[\hat{y}_i=1]}{\sum_i y_i}"
            r"\quad \text{où} \quad \hat{y}_i = \mathbf{1}[f(x_i) \leq \tau_c],\;"
            r"\tau_c = \text{percentile}_c(\mathbf{f})"
        )

    st.subheader("5. Décisions méthodologiques")
    with st.expander("🔍 Pourquoi non supervisé ?"):
        st.markdown("""
En production, aucun label n'est disponible au moment de l'inférence.
Les pumps annoncés publiquement sur Telegram (source La Morgia) ne couvrent que les événements
coordonnés *publics* — les pumps privés (Discord, groupes fermés) ne sont pas étiquetés.

Un classifieur supervisé entraîné sur les labels Telegram ne généralisera pas à ces cas.
L'approche non supervisée produit un score d'anomalie sans jamais accéder aux labels.
""")
    with st.expander("⚠️ L'artefact AUC=0.9976"):
        st.markdown("""
L'évaluation A2 utilisait les features La Morgia calculées sur la fenêtre `[pump_ts−24h, pump_ts+24h]`.
Les lignes `gt=1` coïncident *exactement* avec le pic des features — le modèle détectait
**pendant** le pump, pas avant. AUC=0.9976 est un artefact de data leakage temporel.

L'évaluation corrigée utilise des features calculées **strictement avant** `pump_ts` :
`assert int(pre["open_time"].max()) < window_end_ms`.
""")
    with st.expander("✅ La correction baseline (H1)"):
        st.markdown("""
**Problème :** gt=0 biaisés (score moyen 0.027 ≈ pump score 0.021) → frontière indistinguable.

**Correction :** collecter 3 000 fenêtres de marché réel (BTC/ETH/BNB/XRP/LTC, 2018-2021).
Score moyen marché réel = +0.101, clairement séparé des pumps (+0.021).

**Résultat :** AUC 0.8299 → **0.8815** (+0.052), Recall@0.005 : 0.012 → **0.057** (×4.75).
""")
