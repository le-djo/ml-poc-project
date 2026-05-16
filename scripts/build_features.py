"""
build_features.py — compute microstructure features from collected klines.

Input  : data/RAW/klines/{SYMBOL}BTC_{unix_seconds}.parquet  (151 rows × 12 cols each)
Output : data/Processed/processed.parquet

Design: two rows per event, enabling proper binary AUC evaluation.
  gt=0 : features from [pump_ts-2h, pump_ts-1h)  — quiet control period
  gt=1 : features from [pump_ts-2h, pump_ts)     — full pre-pump accumulation window

Temporal discipline: every feature uses only rows where open_time < window_end_ms.
Assertion added to each compute call.

Features (all from klines — aggTrades unavailable for 2018-2020):
  ret_5m              log(close[-1] / close[-6])
  ret_15m             log(close[-1] / close[-16])
  ret_60m             log(close[-1] / close[-61])   — NaN for gt=0 windows (<61 bars)
  std_ret_60m         std of last 60 1-min log returns
  vol_zscore_60m      (vol[-1] - mean(vol)) / std(vol)    — anomaly vs window baseline
  vol_zscore_5m       (mean(vol[-5:])  - mean(vol)) / std(vol)
  vol_zscore_15m      (mean(vol[-15:]) - mean(vol)) / std(vol)
  vol_zscore_30m      (mean(vol[-30:]) - mean(vol)) / std(vol)
  vol_acceleration    vol_zscore_5m - vol_zscore_30m
  taker_buy_ratio_5m  sum(taker_buy_base[-5:]) / sum(volume[-5:])
  ofi_proxy_1m        (2*taker_buy_base[-1] - volume[-1]) / (volume[-1] + eps)
  price_impact_1m     |close[-1]-open[-1]| / (volume[-1] + eps)   [Amihud ratio]
  conviction_ratio_1m |close[-1]-open[-1]| / (high[-1]-low[-1] + eps)
  rush_order_count    sum(num_trades[-2:])
"""

from pathlib import Path

import numpy as np
import pandas as pd

KLINES_DIR    = Path("data/RAW/klines")
PROCESSED_DIR = Path("data/Processed")
OUT_PATH      = PROCESSED_DIR / "processed.parquet"
SKIPPED_LOG   = Path("data/RAW/skipped_events.txt")
EPS           = 1e-9
MIN_ROWS      = 30


FEAT_COLS = [
    "ret_5m", "ret_15m", "ret_60m", "std_ret_60m",
    "vol_zscore_60m", "vol_zscore_5m", "vol_zscore_15m", "vol_zscore_30m",
    "vol_acceleration",
    "taker_buy_ratio_5m",
    "ofi_proxy_1m", "price_impact_1m", "conviction_ratio_1m",
    "rush_order_count",
]


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def compute_features(
    df: pd.DataFrame, window_end_ms: int, window_start_ms: int | None = None
) -> dict | None:
    """
    Compute features using only rows where open_time < window_end_ms.
    If window_start_ms is provided, also filters open_time >= window_start_ms.

    Returns None if fewer than MIN_ROWS bars are available.
    Raises AssertionError if temporal discipline is violated.
    """
    pre = df[df["open_time"] < window_end_ms].copy()
    if window_start_ms is not None:
        pre = pre[pre["open_time"] >= window_start_ms]

    if len(pre) < MIN_ROWS:
        return None

    # Temporal discipline — non-negotiable
    assert int(pre["open_time"].max()) < window_end_ms, (
        f"Temporal leak: max open_time {pre['open_time'].max()} >= window_end_ms {window_end_ms}"
    )

    close    = pre["close"].values.astype(float)
    volume   = pre["volume"].values.astype(float)
    tb_base  = pre["taker_buy_base"].values.astype(float)
    n_trades = pre["num_trades"].values.astype(int)
    high_    = pre["high"].values.astype(float)
    low_     = pre["low"].values.astype(float)
    open_    = pre["open"].values.astype(float)

    log_ret = np.log(close[1:] / (close[:-1] + EPS))

    def _ret(n: int) -> float:
        return float(np.log(close[-1] / (close[-n - 1] + EPS))) if len(close) > n else np.nan

    # OFI proxy from klines: +1 = all aggressive buy, -1 = all sell
    ofi_1m = float((2.0 * tb_base[-1] - volume[-1]) / (volume[-1] + EPS))

    # Amihud price impact: |log-return of last bar| (fractional move, scale-free)
    price_impact_1m = float(abs(np.log(close[-1] / (open_[-1] + EPS))))

    # Bar conviction: fraction of bar range captured by net move
    conviction_1m = float(
        abs(close[-1] - open_[-1]) / ((high_[-1] - low_[-1]) + EPS)
    )

    # Change 2 — multi-window volume z-scores vs full window baseline
    vol_mean_baseline = np.mean(volume)
    vol_std_baseline  = np.std(volume) + EPS

    def _vol_zscore(n: int) -> float:
        return float((np.mean(volume[-n:]) - vol_mean_baseline) / vol_std_baseline)

    vol_zscore_5m  = _vol_zscore(5)  if len(volume) >= 5  else np.nan
    vol_zscore_15m = _vol_zscore(15) if len(volume) >= 15 else np.nan
    vol_zscore_30m = _vol_zscore(30) if len(volume) >= 30 else np.nan

    # Change 3 — acceleration: is volume speeding up in last 5min vs 30min?
    vol_acceleration = (
        float(vol_zscore_5m - vol_zscore_30m)
        if not (np.isnan(vol_zscore_5m) or np.isnan(vol_zscore_30m))
        else np.nan
    )

    return {
        "ret_5m":               _ret(5),
        "ret_15m":              _ret(15),
        "ret_60m":              _ret(60),
        "std_ret_60m":          float(np.std(log_ret[-60:])) if len(log_ret) >= 60 else np.nan,
        "vol_zscore_60m":       float((volume[-1] - vol_mean_baseline) / vol_std_baseline),
        "vol_zscore_5m":        vol_zscore_5m,
        "vol_zscore_15m":       vol_zscore_15m,
        "vol_zscore_30m":       vol_zscore_30m,
        "vol_acceleration":     vol_acceleration,
        "taker_buy_ratio_5m":   float(np.sum(tb_base[-5:]) / (np.sum(volume[-5:]) + EPS))
                                if len(volume) >= 5 else np.nan,
        "ofi_proxy_1m":         ofi_1m,
        "price_impact_1m":      price_impact_1m,
        "conviction_ratio_1m":  conviction_1m,
        "rush_order_count":     int(np.sum(n_trades[-2:])) if len(n_trades) >= 2 else np.nan,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Checkpoint
    if OUT_PATH.exists():
        existing = pd.read_parquet(OUT_PATH)
        if set(FEAT_COLS).issubset(existing.columns):
            print(f"Checkpoint: {OUT_PATH} already exists ({existing.shape}). Skipping.")
            print(existing[FEAT_COLS].describe().round(4).to_string())
            return

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(KLINES_DIR.glob("*.parquet"))
    print(f"Kline files found: {len(files)}")

    rows = []
    skipped = []

    for path in files:
        stem  = path.stem
        parts = stem.rsplit("_", 1)
        if len(parts) != 2:
            continue

        binance_symbol, ts_str = parts
        pump_ts_unix = int(ts_str)
        pump_ts_ms   = pump_ts_unix * 1000
        symbol       = binance_symbol.removesuffix("BTC")
        pump_ts      = pd.Timestamp(pump_ts_unix, unit="s", tz="UTC")

        df = pd.read_parquet(path)

        # gt=1 : full 2h pre-pump window → [pump_ts-2h, pump_ts)
        gt1_end = pump_ts_ms
        feats1  = compute_features(df, gt1_end)
        if feats1 is None:
            skipped.append(f"{stem},gt1,insufficient_rows")
            continue

        # gt=0 : quiet control window → [pump_ts-2h, pump_ts-1h)
        gt0_end = pump_ts_ms - 3_600_000
        feats0  = compute_features(df, gt0_end)
        if feats0 is None:
            skipped.append(f"{stem},gt0,insufficient_rows")
            # Still include the gt=1 row even if gt=0 fails
            feats1.update({
                "event_id":       stem,
                "symbol":         symbol,
                "binance_symbol": binance_symbol,
                "pump_ts":        pump_ts,
                "pump_ts_unix":   pump_ts_unix,
                "gt":             1,
            })
            rows.append(feats1)
            continue

        # gt=0 row
        feats0.update({
            "event_id":       stem,
            "symbol":         symbol,
            "binance_symbol": binance_symbol,
            "pump_ts":        pump_ts,
            "pump_ts_unix":   pump_ts_unix,
            "gt":             0,
        })
        rows.append(feats0)

        # gt=1 row
        feats1.update({
            "event_id":       stem,
            "symbol":         symbol,
            "binance_symbol": binance_symbol,
            "pump_ts":        pump_ts,
            "pump_ts_unix":   pump_ts_unix,
            "gt":             1,
        })
        rows.append(feats1)

    feat_df = pd.DataFrame(rows)

    # Fill NaN for gt=0 windows (60 bars — insufficient for 60min lookback)
    # Imputed with 0: no net move / no vol detected = neutral baseline
    for col in ("ret_60m", "std_ret_60m"):
        n_nan = feat_df[col].isna().sum()
        feat_df[col] = feat_df[col].fillna(0.0)
        print(f"{col} NaN imputed to 0: {n_nan} rows (expected ≤ #gt0 events)")

    # Log skipped events
    if skipped:
        SKIPPED_LOG.parent.mkdir(parents=True, exist_ok=True)
        with SKIPPED_LOG.open("w") as f:
            f.write("\n".join(skipped) + "\n")
        print(f"Skipped events logged → {SKIPPED_LOG}  ({len(skipped)} entries)")

    feat_df.to_parquet(OUT_PATH, index=False)
    print(f"\nSaved → {OUT_PATH}")
    print(f"Shape : {feat_df.shape}")
    print(f"gt=0   : {(feat_df['gt']==0).sum()} rows  |  gt=1 : {(feat_df['gt']==1).sum()} rows")
    print(f"\nMissing values after imputation:\n{feat_df[FEAT_COLS].isnull().sum().to_string()}")
    print(f"\nFeature summary:\n{feat_df[FEAT_COLS + ['gt']].describe().round(4).to_string()}")


if __name__ == "__main__":
    main()
