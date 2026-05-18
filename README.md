# Pump & Dump Detection in Crypto Markets

Unsupervised anomaly detection system for identifying pump-and-dump manipulation schemes on Binance. The pipeline extracts microstructure features (log returns, volume z-scores, order flow imbalance, taker buy pressure) from Binance klines over a strict pre-event window, then scores each observation using an ensemble of Isolation Forest, Local Outlier Factor, and rolling Z-score models. Ground-truth labels from La Morgia et al. 2020 are used exclusively for post-hoc recall validation (Recall@LaMargia), never during training. The project is developed as an academic deliverable and as preparation for integration into Kaiko's Market Surveyor product, with direct applicability to MiCA Article 62 market abuse surveillance requirements.

---

## Results

| Model | AUC | Recall@0.005 |
|-------|-----|-------------|
| Isolation Forest | **0.881** | **0.057** |
| Z-score (vol_zscore_60m) | 0.766 | 0.057 |
| LOF (n=20) | 0.710 | 0.012 |

Key finding: baseline gt=0 was biased (pump-adjacent windows ≈ pump windows). Adding 3000 genuine market windows via Binance REST API improved AUC from 0.8299 → 0.881, Recall ×4.75.

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/le-djo/ml-poc-project.git
cd ml-poc-project
pip install -r requirements.txt
```

### 2. Download La Morgia dataset

The labeled event log is not included in the repo. Clone it into `data/RAW/`:

```bash
mkdir -p data/RAW
git clone https://github.com/nicolasmarcello/la-morgia-pump-dump.git data/RAW/la_morgia_raw
```

Then copy the pump event list:
```bash
cp data/RAW/la_morgia_raw/pump_telegram.csv data/RAW/pump_telegram.csv
```

### 3. Collect Binance klines (pump events)

Fetches 1-min klines for each of the 333 pump events from the Binance public REST API (no API key required). Takes ~10 minutes. Checkpointed — safe to interrupt and resume.

```bash
python scripts/fetch_binance.py
```

Output: `data/RAW/klines/` — 333 parquet files.

### 4. Collect normal market windows (required for H1 baseline)

Normal market window collection is integrated into `scripts/fetch_binance.py` — the same script handles both pump event klines (step 3) and the 3000 random control windows. No separate script needed.

Output: `data/RAW/normal_windows/` — 30 chunk parquet files.

### 5. Build feature matrix

Computes 14 features strictly before pump_ts for each event. Combines pump events + normal windows into the final dataset.

```bash
python scripts/build_features.py
```

Output: `data/Processed/processed.parquet` — 3666 rows, 14 features.

### 6. Train models and generate figures

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/02_modeling.ipynb
```

Output: `models/isolation_forest.joblib`, `models/scaler_klines.joblib`, `deliverables/fig_*.png`.

### 7. Launch the Streamlit app

```bash
python scripts/main.py
```

Opens at `http://localhost:8501`. No data files needed at runtime — the app fetches live klines from Binance REST API for the demo tab.

---

## Project structure

```
ml-poc-project/
├── app.py                    # Streamlit app (6 pages)
├── scripts/
│   ├── main.py               # Entry point: python scripts/main.py
│   ├── fetch_binance.py      # Collect pump event klines + normal windows (H1)
│   ├── build_features.py     # Feature engineering pipeline
│   ├── metrics.py            # recall_at_lamorgia(), anomaly_auc()
│   └── data.py               # La Morgia feature pipeline (A2)
├── notebooks/
│   ├── 01_eda.ipynb          # Exploratory data analysis
│   └── 02_modeling.ipynb     # Model training and evaluation
├── models/
│   ├── isolation_forest.joblib
│   └── scaler_klines.joblib
├── deliverables/
│   ├── assignment1.md → assignment5.md
│   ├── project_summary.md    # Complete project documentation
│   └── fig_*.png             # Figures for assignments
├── data/                     # gitignored — generated locally
│   ├── RAW/klines/           # 333 pump event parquet files
│   ├── RAW/normal_windows/   # 3000 normal market windows
│   ├── RAW/la_morgia_raw/    # La Morgia dataset (cloned separately)
│   └── Processed/            # processed.parquet (3666 rows)
└── requirements.txt
```

---

## Notes

- **No API key required** — all data from Binance public REST API (`/api/v3/klines`)
- **Reproducible** — all steps checkpointed, deterministic with `random_state=42`
- **Data not included** — `data/` is gitignored; follow setup steps 2-6 to regenerate
- **Runtime** — full pipeline takes ~25 minutes (mostly API calls)
