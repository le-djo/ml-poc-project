"""
build_features.py — compute microstructure features from collected klines.

Input  : data/RAW/klines/{SYMBOL}BTC_{unix_seconds}.parquet  (151 rows × 12 cols each)
Output : data/Processed/features_klines.parquet

Feature window: strictly [pump_ts - 2h, pump_ts) — rows where open_time < pump_ts_ms.
That gives 120 one-minute bars per event.

Features computed (all from klines — aggTrades unavailable for 2018-2020 history):
  ret_5m            log(close[-1] / close[-6])          price velocity last 5 min
  ret_15m           log(close[-1] / close[-16])         price velocity last 15 min
  ret_60m           log(close[-1] / close[-61])         price velocity last 60 min
  std_ret_60m       std of 1m log returns, last 60 min  volatility spike
  vol_zscore_60m    (vol[-1] - mean(vol[-120:])) /       volume anomaly vs 2h baseline
                    std(vol[-120:])
  vol_zscore_dynamic same but normalised by global       hour-of-day adjusted anomaly
                    hour-of-day median/mad across
                    all collected events
  taker_buy_ratio_5m sum(taker_buy_base[-5:]) /          coordinated buying pressure
                    sum(volume[-5:])
  rush_order_count  sum(num_trades[-2:])                trade frequency spike proxy
  ofi_1m            NaN (aggTrades unavailable)         — see assignment2.md
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

KLINES_DIR    = Path("data/RAW/klines")
PROCESSED_DIR = Path("data/Processed")
OUT_PATH      = PROCESSED_DIR / "features_klines.parquet"
EPS           = 1e-8   # avoid division by zero


# ---------------------------------------------------------------------------
# Per-event feature computation
# ---------------------------------------------------------------------------

def compute_features(df: pd.DataFrame, pump_ts_ms: int) -> dict:
    """
    Compute all features from a single kline DataFrame.
    Uses only rows strictly before pump_ts (open_time < pump_ts_ms).
    """
    pre = df[df["open_time"] < pump_ts_ms].copy()

    if len(pre) < 10:
        return None  # not enough data to compute anything useful

    close     = pre["close"].values.astype(float)
    volume    = pre["volume"].values.astype(float)
    tb_base   = pre["taker_buy_base"].values.astype(float)
    n_trades  = pre["num_trades"].values.astype(int)

    # 1-minute log returns (length = len(close) - 1)
    log_ret = np.log(close[1:] / (close[:-1] + EPS))

    def _ret(n: int) -> float:
        return float(np.log(close[-1] / (close[-n - 1] + EPS))) if len(close) > n else np.nan

    return {
        "ret_5m":              _ret(5),
        "ret_15m":             _ret(15),
        "ret_60m":             _ret(60),
        "std_ret_60m":         float(np.std(log_ret[-60:])) if len(log_ret) >= 60 else np.nan,
        "vol_zscore_60m":      float(
            (volume[-1] - np.mean(volume)) / (np.std(volume) + EPS)
        ),
        "taker_buy_ratio_5m":  float(
            np.sum(tb_base[-5:]) / (np.sum(volume[-5:]) + EPS)
        ) if len(volume) >= 5 else np.nan,
        "rush_order_count":    int(np.sum(n_trades[-2:])) if len(n_trades) >= 2 else np.nan,
        "ofi_1m":              np.nan,   # aggTrades unavailable for 2018-2020
        # raw volume kept for dynamic z-score computation in second pass
        "_last_volume":        float(volume[-1]),
        "_hour_utc":           int(pd.Timestamp(pump_ts_ms, unit="ms", tz="UTC").hour),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(KLINES_DIR.glob("*.parquet"))
    print(f"Kline files found: {len(files)}")

    rows = []
    skipped = 0

    for path in files:
        # Parse symbol and pump_ts from filename: SYMBOLBTC_unixseconds.parquet
        stem = path.stem                        # e.g. "BRDBTC_1545498000"
        parts = stem.rsplit("_", 1)
        if len(parts) != 2:
            continue
        binance_symbol, ts_str = parts
        pump_ts_unix = int(ts_str)
        pump_ts_ms   = pump_ts_unix * 1000
        symbol       = binance_symbol.replace("BTC", "")  # strip quote currency

        df = pd.read_parquet(path)
        feats = compute_features(df, pump_ts_ms)

        if feats is None:
            skipped += 1
            continue

        feats["symbol"]       = symbol
        feats["binance_symbol"] = binance_symbol
        feats["pump_ts"]      = pd.Timestamp(pump_ts_unix, unit="s", tz="UTC")
        feats["pump_ts_unix"] = pump_ts_unix
        feats["gt"]           = 1   # all collected windows are pump events
        rows.append(feats)

    print(f"Events processed: {len(rows)}  |  Skipped (< 10 bars): {skipped}")

    feat_df = pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Second pass: vol_zscore_dynamic
    # Normalise each event's last-minute volume by the median and MAD
    # of all events in the same UTC hour slot.
    # ------------------------------------------------------------------
    hour_stats = (
        feat_df.groupby("_hour_utc")["_last_volume"]
        .agg(median="median", mad=lambda x: np.median(np.abs(x - np.median(x))))
        .reset_index()
    )
    feat_df = feat_df.merge(hour_stats, on="_hour_utc", how="left")
    feat_df["vol_zscore_dynamic"] = (
        (feat_df["_last_volume"] - feat_df["median"]) / (feat_df["mad"] + EPS)
    )
    feat_df.drop(columns=["_last_volume", "_hour_utc", "median", "mad"], inplace=True)

    # ------------------------------------------------------------------
    # Persist
    # ------------------------------------------------------------------
    feat_df.to_parquet(OUT_PATH, index=False)
    print(f"\nSaved → {OUT_PATH}")
    print(f"Shape : {feat_df.shape}")
    print("\nFeature summary (non-null counts):")
    print(feat_df[["ret_5m","ret_15m","ret_60m","std_ret_60m",
                   "vol_zscore_60m","vol_zscore_dynamic",
                   "taker_buy_ratio_5m","rush_order_count"]].describe().round(4).to_string())
    print(f"\nMissing values:\n{feat_df.isnull().sum()}")


if __name__ == "__main__":
    main()
