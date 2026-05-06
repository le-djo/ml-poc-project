"""
fetch_binance.py — collect Binance klines and aggTrades for every pump event.

Usage
-----
  python scripts/fetch_binance.py            # dry run: aggtrades vision, first 3 events
  python scripts/fetch_binance.py --dry-run  # same
  python scripts/fetch_binance.py --full     # all 520 events

Output
------
  data/RAW/klines/          {SYMBOL}BTC_{unix_seconds}.parquet  (1m klines, 150 rows)
  data/RAW/aggtrades/       {SYMBOL}BTC_{unix_seconds}.parquet  (aggTrades via vision)
  data/RAW/failed_symbols.txt                                   (klines 400 errors)
  data/RAW/failed_aggtrades.txt                                 (vision 404 errors)

aggTrades strategy
------------------
The Binance REST /aggTrades endpoint only retains a rolling window of a few months.
All pump events (2018-2020) fall outside this window — the REST endpoint returns 0 rows.
Fix: data.binance.vision, Binance's official static archive (full history since 2017).
URL: https://data.binance.vision/data/spot/daily/aggTrades/{SYMBOL}/{SYMBOL}-{YYYY-MM-DD}.zip
No API key required. 0.5s sleep between downloads.
"""

import argparse
import io
import time
import zipfile
from datetime import timedelta
from pathlib import Path

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_URL    = "https://api.binance.com/api/v3"
VISION_BASE = "https://data.binance.vision/data/spot/daily/aggTrades"

KLINES_DIR          = Path("data/RAW/klines")
AGGTRADES_DIR       = Path("data/RAW/aggtrades")
FAILED_LOG          = Path("data/RAW/failed_symbols.txt")
FAILED_AGGTRADES_LOG = Path("data/RAW/failed_aggtrades.txt")

WEIGHT_PER_CALL  = 2
SAFE_WEIGHT_LIMIT = 500   # proactive throttle — well below the 1200/min hard limit

_AGGTRADE_VISION_COLS = [
    "agg_id", "price", "qty", "first_trade_id",
    "last_trade_id", "timestamp", "is_buyer_maker",
]


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

class RateLimiter:
    """
    Sliding 60-second window weight tracker.

    Proactively sleeps when accumulated weight approaches SAFE_WEIGHT_LIMIT,
    preventing 429 errors before they happen.
    """

    def __init__(self, safe_limit: int = SAFE_WEIGHT_LIMIT):
        self._safe_limit = safe_limit
        self._used = 0
        self._window_start = time.monotonic()

    def _reset_if_new_window(self):
        if time.monotonic() - self._window_start >= 60:
            self._used = 0
            self._window_start = time.monotonic()

    def check_and_wait(self, weight: int = WEIGHT_PER_CALL):
        """Sleep until the current window resets if adding `weight` would exceed the limit."""
        self._reset_if_new_window()
        if self._used + weight > self._safe_limit:
            elapsed = time.monotonic() - self._window_start
            sleep_for = max(0.0, 60.0 - elapsed + 0.5)
            print(f"  [rate limiter] weight={self._used}/{self._safe_limit} — sleeping {sleep_for:.1f}s")
            time.sleep(sleep_for)
            self._used = 0
            self._window_start = time.monotonic()
        self._used += weight


# ---------------------------------------------------------------------------
# Event list loader
# ---------------------------------------------------------------------------

def load_events() -> pd.DataFrame:
    """
    Load pump_telegram.csv, filter Binance, reconstruct pump_ts, add Binance symbol
    and pre-computed ms timestamps for the two fetch windows.
    """
    df = pd.read_csv("data/RAW/pump_telegram.csv")
    df = df[df["exchange"].str.lower() == "binance"].copy()

    # hour column is already "HH:MM" — append ":00" to form a valid datetime string
    df["pump_ts"] = pd.to_datetime(
        df["date"] + " " + df["hour"].astype(str) + ":00", utc=True
    )
    df["binance_symbol"] = df["symbol"] + "BTC"

    # Unix milliseconds for each window boundary
    def _ms(ts):
        return int(ts.timestamp() * 1000)

    df["kline_start_ms"] = df["pump_ts"].apply(lambda t: _ms(t - pd.Timedelta(hours=2)))
    df["kline_end_ms"]   = df["pump_ts"].apply(lambda t: _ms(t + pd.Timedelta(minutes=30)))
    df["agg_start_ms"]   = df["pump_ts"].apply(lambda t: _ms(t - pd.Timedelta(minutes=10)))
    df["agg_end_ms"]     = df["pump_ts"].apply(lambda t: _ms(t + pd.Timedelta(minutes=5)))

    df = df.reset_index(drop=True)
    print(f"Events loaded: {len(df)} Binance events")
    print(df[["symbol", "binance_symbol", "pump_ts"]].head(3).to_string())
    return df


# ---------------------------------------------------------------------------
# Fetch helpers
# ---------------------------------------------------------------------------

_KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_asset_vol", "num_trades",
    "taker_buy_base", "taker_buy_quote", "ignore",
]
_NUMERIC_KLINE_COLS = ["open", "high", "low", "close", "volume",
                       "quote_asset_vol", "taker_buy_base", "taker_buy_quote"]

_AGTRADE_COLS = ["agg_id", "price", "qty", "first_trade_id",
                 "last_trade_id", "timestamp", "is_buyer_maker"]
_AGTRADE_MAP  = {"a": "agg_id", "p": "price", "q": "qty",
                 "f": "first_trade_id", "l": "last_trade_id",
                 "T": "timestamp", "m": "is_buyer_maker"}


def _get(url: str, params: dict, rl: RateLimiter) -> list:
    """
    Execute a single GET request with rate limiting and error handling.

    Returns the parsed JSON list.
    Raises ValueError on 400 (delisted symbol).
    Raises RuntimeError on 429 after one retry.
    """
    rl.check_and_wait(WEIGHT_PER_CALL)
    resp = requests.get(url, params=params, timeout=15)

    if resp.status_code == 200:
        return resp.json()

    if resp.status_code == 400:
        raise ValueError(f"HTTP 400 — symbol likely delisted: {params.get('symbol')}")

    if resp.status_code == 429:
        print("  [429] Rate limited — sleeping 60s and retrying once")
        time.sleep(60)
        rl.check_and_wait(WEIGHT_PER_CALL)
        resp2 = requests.get(url, params=params, timeout=15)
        if resp2.status_code == 200:
            return resp2.json()
        raise RuntimeError(f"HTTP 429 persists after retry for {params.get('symbol')}")

    resp.raise_for_status()
    return []  # unreachable but keeps type checker happy


def fetch_klines(symbol: str, start_ms: int, end_ms: int, rl: RateLimiter) -> pd.DataFrame:
    """
    Fetch 1-minute klines for `symbol` over [start_ms, end_ms].
    Returns a DataFrame with 12 named columns; numeric columns cast to float.
    """
    url = f"{BASE_URL}/klines"
    params = {
        "symbol": symbol,
        "interval": "1m",
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": 1000,
    }
    data = _get(url, params, rl)
    df = pd.DataFrame(data, columns=_KLINE_COLS)
    for col in _NUMERIC_KLINE_COLS:
        df[col] = df[col].astype(float)
    df["num_trades"] = df["num_trades"].astype(int)
    return df


def fetch_aggtrades(symbol: str, start_ms: int, end_ms: int, rl: RateLimiter) -> pd.DataFrame:
    """
    Fetch aggregate trades for `symbol` over [start_ms, end_ms].
    Handles pagination automatically (Binance returns ≤1000 trades per call).
    Returns a DataFrame with 7 named columns.
    """
    url = f"{BASE_URL}/aggTrades"
    pages = []
    cursor = start_ms

    while True:
        params = {"symbol": symbol, "startTime": cursor, "endTime": end_ms}
        data = _get(url, params, rl)
        if not data:
            break
        pages.extend(data)
        if len(data) < 1000:
            break
        # Advance cursor to just after the last returned trade
        cursor = data[-1]["T"] + 1
        if cursor > end_ms:
            break

    if not pages:
        return pd.DataFrame(columns=_AGTRADE_COLS)

    df = pd.DataFrame(pages).rename(columns=_AGTRADE_MAP)[_AGTRADE_COLS]
    df["price"] = df["price"].astype(float)
    df["qty"]   = df["qty"].astype(float)
    return df


# ---------------------------------------------------------------------------
# Single-event collector
# ---------------------------------------------------------------------------

def collect_event(row: pd.Series, rl: RateLimiter, n_failed: list) -> None:
    """
    Fetch and persist klines + aggTrades for one event.
    `n_failed` is a one-element list used as a mutable counter.
    """
    symbol   = row["binance_symbol"]
    ts_unix  = int(row["pump_ts"].timestamp())

    kline_path = KLINES_DIR    / f"{symbol}_{ts_unix}.parquet"
    agg_path   = AGGTRADES_DIR / f"{symbol}_{ts_unix}.parquet"

    # Checkpoint — skip if both outputs already exist
    if kline_path.exists() and agg_path.exists():
        print(f"  skip {symbol} {row['pump_ts'].date()}")
        return

    try:
        klines = fetch_klines(symbol, row["kline_start_ms"], row["kline_end_ms"], rl)
        klines.to_parquet(kline_path, index=False)

        trades = fetch_aggtrades(symbol, row["agg_start_ms"], row["agg_end_ms"], rl)
        trades.to_parquet(agg_path, index=False)

    except ValueError as exc:
        # 400 = symbol delisted on Binance
        print(f"  [400] {exc} — logging and skipping")
        with FAILED_LOG.open("a") as f:
            f.write(f"{symbol},{row['pump_ts']}\n")
        n_failed[0] += 1

    time.sleep(0.1)   # polite gap between events


# ---------------------------------------------------------------------------
# aggTrades via data.binance.vision (historical archive — replaces REST API)
# ---------------------------------------------------------------------------

def fetch_aggtrades_vision(symbol: str, pump_ts: pd.Timestamp) -> pd.DataFrame:
    """
    Download daily aggTrades ZIP files from data.binance.vision and return
    the rows that fall in [pump_ts - 10min, pump_ts).

    Downloads up to two days (day-before + day-of) to handle events that
    span midnight UTC.  Each 404 is logged to FAILED_AGGTRADES_LOG.
    """
    window_start_ms = int((pump_ts - pd.Timedelta(minutes=10)).timestamp() * 1000)
    window_end_ms   = int(pump_ts.timestamp() * 1000)

    date_of  = pump_ts.date()
    date_pre = date_of - timedelta(days=1)

    frames = []
    for date in (date_pre, date_of):
        url = f"{VISION_BASE}/{symbol}/{symbol}-{date}.zip"
        print(f"  GET {url}")
        try:
            resp = requests.get(url, timeout=30)
        except requests.RequestException as exc:
            print(f"  [network error] {exc}")
            continue

        if resp.status_code == 404:
            with FAILED_AGGTRADES_LOG.open("a") as f:
                f.write(f"{symbol},{date},404\n")
            print(f"  404 — not in archive")
            time.sleep(0.5)
            continue

        resp.raise_for_status()
        time.sleep(0.5)   # respectful throttle after each successful download

        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            csv_name = zf.namelist()[0]
            with zf.open(csv_name) as f:
                df = pd.read_csv(
                    f, header=None, names=_AGGTRADE_VISION_COLS,
                    dtype={"price": float, "qty": float, "timestamp": "int64",
                           "is_buyer_maker": bool},
                )
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=_AGGTRADE_VISION_COLS)

    combined = pd.concat(frames, ignore_index=True)
    # Filter to [pump_ts - 10min, pump_ts) — strictly before pump
    filtered = combined[
        (combined["timestamp"] >= window_start_ms) &
        (combined["timestamp"] <  window_end_ms)
    ].reset_index(drop=True)
    return filtered


def collect_aggtrades_loop(events: pd.DataFrame, dry_run: bool) -> None:
    """
    Iterate events and collect aggTrades via data.binance.vision.

    Checkpoint: skip if parquet already exists AND has rows > 0
    (empty parquets from the failed REST attempt are re-downloaded).
    """
    subset = events.head(3) if dry_run else events
    total  = len(subset)
    n_failed = 0

    print(f"\n--- aggTrades via data.binance.vision ({'DRY RUN — 3 events' if dry_run else f'{total} events'}) ---")

    for i, (_, row) in enumerate(subset.iterrows(), 1):
        symbol  = row["binance_symbol"]
        pump_ts = row["pump_ts"]
        ts_unix = int(pump_ts.timestamp())
        agg_path = AGGTRADES_DIR / f"{symbol}_{ts_unix}.parquet"

        # Skip only if file exists and is non-empty
        if agg_path.exists():
            existing = pd.read_parquet(agg_path)
            if len(existing) > 0:
                print(f"  skip {symbol} {pump_ts.date()} ({len(existing)} rows already saved)")
                continue

        print(f"\n[{i}/{total}] {symbol} @ {pump_ts}")
        try:
            df = fetch_aggtrades_vision(symbol, pump_ts)
            df.to_parquet(agg_path, index=False)
            print(f"  → saved {len(df)} rows to {agg_path.name}")
        except Exception as exc:
            print(f"  [ERROR] {exc}")
            with FAILED_AGGTRADES_LOG.open("a") as f:
                f.write(f"{symbol},{pump_ts},error,{exc}\n")
            n_failed += 1

        if i % 10 == 0:
            print(f"Progress: {i}/{total} | Failed: {n_failed}")

    print(f"\naggTrades done. {total - n_failed}/{total} processed. Failed: {n_failed}")

    if dry_run:
        saved = [p for p in AGGTRADES_DIR.glob("*.parquet")
                 if pd.read_parquet(p).shape[0] > 0]
        print(f"\nNon-empty aggtrade files: {len(saved)}")
        for p in sorted(saved)[:3]:
            df = pd.read_parquet(p)
            print(f"  {p.name}: {df.shape}")
        if n_failed == 0:
            print("\nDRY RUN OK — ready for full collection")
            print("Run:  python scripts/fetch_binance.py --full")
        else:
            print(f"\nDRY RUN finished with {n_failed} failures — check {FAILED_AGGTRADES_LOG}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(dry_run: bool = True) -> None:
    events = load_events()
    KLINES_DIR.mkdir(parents=True, exist_ok=True)
    AGGTRADES_DIR.mkdir(parents=True, exist_ok=True)

    # Klines are already collected (333 files from previous run).
    # This script now focuses on fixing aggTrades via data.binance.vision.
    kline_count = len(list(KLINES_DIR.glob("*.parquet")))
    print(f"Klines already collected: {kline_count} files (checkpoint will skip all)")

    collect_aggtrades_loop(events, dry_run=dry_run)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch Binance klines + aggTrades for pump events")
    parser.add_argument("--dry-run", action="store_true", help="fetch first 3 events only (default)")
    parser.add_argument("--full",    action="store_true", help="fetch all 520 events")
    args = parser.parse_args()

    main(dry_run=not args.full)
