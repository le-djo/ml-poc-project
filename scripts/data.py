"""
data.py — load, clean, and transform the La Morgia labeled features dataset.

Starting dataset : data/RAW/la_morgia_raw/labeled_features/features_{freq}.csv.gz
Output dataset   : data/Processed/features_{freq}_transformed.parquet

Pipeline (in order):
  1. Load one of the three time-frequency files (25S / 15S / 5S)
  2. Parse the `date` column with skrub DatetimeEncoder → cyclical + ordinal time features
  3. Encode `symbol` (85 unique values) with skrub GapEncoder → dense embedding
  4. Scale all numerical features with RobustScaler (heavy-tailed financial distributions)
  5. Optionally apply PCA for dimensionality reduction
  6. Separate X / y and persist processed data to data/Processed/

Alternatives tested but not retained — see assignment2.md for full justification:
  - StandardScaler  : rejected (sensitive to the extreme outliers in rush-order features)
  - OneHotEncoder   : rejected for symbol (85 dims, sparse, noisy for tree models)
  - Dropping `date` : rejected after DatetimeEncoder showed day-of-week signal
  - freq=25S / 5S   : tested; 15S retained as best precision/recall trade-off baseline

What we could NOT do (data unavailable):
  - Binance klines not fetched  → OFI, taker_buy_ratio, vol_zscore features absent
  - CoinGecko not fetched       → market_cap_rank, circulating_supply absent
"""

import os
import pathlib

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from skrub import DatetimeEncoder, GapEncoder, TableVectorizer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "RAW" / "la_morgia_raw" / "labeled_features"
PROCESSED_DIR = PROJECT_ROOT / "data" / "Processed"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_features(freq: str = "15S") -> pd.DataFrame:
    """Load raw labeled features for the given time frequency (25S / 15S / 5S)."""
    path = RAW_DIR / f"features_{freq}.csv.gz"
    if not path.exists():
        raise FileNotFoundError(f"Features file not found: {path}")
    df = pd.read_csv(path)
    return df


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal cleaning pass.

    - `date`       : kept as-is; DatetimeEncoder will parse it downstream.
    - `pump_index` : kept as a relative-position feature (step within the ±window).
    - `symbol`     : kept as-is; GapEncoder handles it downstream.
    - No missing values in this dataset, so no imputation needed.
    """
    df = df.copy()
    # DatetimeEncoder requires a proper datetime dtype, not a string
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df


# ---------------------------------------------------------------------------
# Transformation pipeline (skrub + sklearn)
# ---------------------------------------------------------------------------

def build_vectorizer() -> TableVectorizer:
    """
    TableVectorizer with explicit column overrides:

    - date   → DatetimeEncoder  : extracts year, month, day, hour, minute,
                                   plus cyclical sin/cos encodings. Better than
                                   dropping because day-of-week carries signal
                                   (pumps cluster on weekends).
    - symbol → GapEncoder       : learns a character n-gram embedding per symbol;
                                   robust to unseen tokens and captures ticker
                                   morphology. MinHashEncoder was also tested but
                                   GapEncoder produced slightly higher AUC.
    - numerics → passthrough    : already in float64, scaled separately.
    """
    return TableVectorizer(
        specific_transformers=[
            (DatetimeEncoder(add_total_seconds=False, add_weekday=True), ["date"]),
            (GapEncoder(n_components=10), ["symbol"]),
        ],
    )


def build_pipeline(use_pca: bool = False, n_components: int = 10) -> Pipeline:
    """
    Full sklearn Pipeline:
      TableVectorizer → RobustScaler → [optional PCA]

    RobustScaler is preferred over StandardScaler because rush-order and
    volume features are heavily right-skewed: a single coordinated burst
    can be 50× the median, which would inflate StandardScaler's variance
    estimate and shrink all other features toward zero.
    """
    steps = [
        ("vectorizer", build_vectorizer()),
        ("scaler", RobustScaler()),
    ]
    if use_pca:
        steps.append(("pca", PCA(n_components=n_components, random_state=42)))
    return Pipeline(steps)


# ---------------------------------------------------------------------------
# End-to-end transform + persist
# ---------------------------------------------------------------------------

def transform_and_save(freq: str = "15S", use_pca: bool = False) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load → clean → transform → save to data/Processed/.

    Returns
    -------
    X : pd.DataFrame  (transformed features)
    y : pd.Series     (binary label: 1 = pump event)
    """
    df = load_features(freq)
    df = clean(df)

    y = df["gt"]
    X_raw = df.drop(columns=["gt"])

    pipeline = build_pipeline(use_pca=use_pca)
    X_arr = pipeline.fit_transform(X_raw)

    # Recover column names after vectorization
    try:
        feature_names = pipeline.named_steps["vectorizer"].get_feature_names_out()
        if use_pca:
            n = pipeline.named_steps["pca"].n_components_
            feature_names = [f"PC{i+1}" for i in range(n)]
    except Exception:
        feature_names = [f"f{i}" for i in range(X_arr.shape[1])]

    X = pd.DataFrame(X_arr, columns=feature_names)

    # Persist
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "_pca" if use_pca else ""
    out_path = PROCESSED_DIR / f"features_{freq}{suffix}_transformed.parquet"
    X.assign(gt=y.values).to_parquet(out_path, index=False)
    print(f"Saved → {out_path}  shape={X.shape}")

    return X, y


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for use_pca in (False, True):
        X, y = transform_and_save(freq="15S", use_pca=use_pca)
        print(f"PCA={use_pca}  X={X.shape}  pumps={y.sum()}/{len(y)}")
