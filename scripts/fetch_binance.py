"""
fetch_binance.py — collect Binance klines and aggTrades for every pump event.

Usage
-----
  python scripts/fetch_binance.py            # dry run (first 3 events only)
  python scripts/fetch_binance.py --dry-run  # same
  python scripts/fetch_binance.py --full     # all 520 events

Output
------
  data/RAW/klines/     {SYMBOL}BTC_{unix_seconds}.parquet   (1m klines, 150 rows each)
  data/RAW/aggtrades/  {SYMBOL}BTC_{unix_seconds}.parquet   (15-min aggTrades window)
  data/RAW/failed_symbols.txt                               (symbols that returned 400)
"""

import argparse
import time
from pathlib import Path

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_URL = "https://api.binance.com/api/v3"
KLINES_DIR = Path("data/RAW/klines")
AGGTRADES_DIR = Path("data/RAW/aggtrades")
FAILED_LOG = Path("data/RAW/failed_symbols.txt")

WEIGHT_PER_CALL = 2
SAFE_WEIGHT_LIMIT = 500   # proactive throttle — well below the 1200/min hard limit


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
# Main
# ---------------------------------------------------------------------------

def main(dry_run: bool = True) -> None:
    events = load_events()
    KLINES_DIR.mkdir(parents=True, exist_ok=True)
    AGGTRADES_DIR.mkdir(parents=True, exist_ok=True)

    rl = RateLimiter()
    n_failed = [0]

    if dry_run:
        subset = events.head(3)
        print("\n--- DRY RUN (first 3 events) ---")
        print("URLs that will be called:")
        for _, row in subset.iterrows():
            sym = row["binance_symbol"]
            print(f"  klines:    {BASE_URL}/klines?symbol={sym}&interval=1m"
                  f"&startTime={row['kline_start_ms']}&endTime={row['kline_end_ms']}&limit=1000")
            print(f"  aggTrades: {BASE_URL}/aggTrades?symbol={sym}"
                  f"&startTime={row['agg_start_ms']}&endTime={row['agg_end_ms']}")
        print()

        for _, row in subset.iterrows():
            print(f"Fetching {row['binance_symbol']} @ {row['pump_ts']} ...")
            collect_event(row, rl, n_failed)

        # Verification
        kline_files = sorted(KLINES_DIR.glob("*.parquet"))
        agg_files   = sorted(AGGTRADES_DIR.glob("*.parquet"))
        print(f"\nKlines files saved    : {len(kline_files)}")
        print(f"AggTrades files saved : {len(agg_files)}")
        for p in kline_files[:3]:
            df = pd.read_parquet(p)
            print(f"  {p.name}: {df.shape}")
        for p in agg_files[:3]:
            df = pd.read_parquet(p)
            print(f"  {p.name}: {df.shape}")

        if n_failed[0] == 0 and len(kline_files) >= 3 and len(agg_files) >= 3:
            print("\nDRY RUN OK — ready for full collection")
            print("Run:  python scripts/fetch_binance.py --full")
        else:
            print(f"\nDRY RUN finished with {n_failed[0]} failures — check failed_symbols.txt")
        return

    # Full run
    total = len(events)
    print(f"\n--- FULL RUN ({total} events) ---")
    for i, (_, row) in enumerate(events.iterrows(), 1):
        collect_event(row, rl, n_failed)
        if i % 10 == 0:
            print(f"Progress: {i}/{total} | Failed: {n_failed[0]}")

    print(f"\nDone. {total - n_failed[0]}/{total} events collected. "
          f"Failed: {n_failed[0]} (see {FAILED_LOG})")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch Binance klines + aggTrades for pump events")
    parser.add_argument("--dry-run", action="store_true", help="fetch first 3 events only (default)")
    parser.add_argument("--full",    action="store_true", help="fetch all 520 events")
    args = parser.parse_args()

    main(dry_run=not args.full)
