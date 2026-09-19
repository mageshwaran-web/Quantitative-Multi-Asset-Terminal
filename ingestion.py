"""
Layer 1 — Data ingestion and SQLite storage.

Fetches daily OHLCV from yfinance and stores it in data/market.db.
Each asset keeps its NATIVE trading calendar (BTC trades 7 days/week,
GC=F and NVDA do not). No calendar alignment happens here; alignment is
done downstream in indicators.get_correlation_matrix().

Usage:
    python ingestion.py            # create schema + fetch/append new rows
    python ingestion.py --status   # show what is currently in the DB
"""

from __future__ import annotations

import argparse
import os
import sqlite3
from datetime import date, datetime

import pandas as pd

DB_PATH = os.path.join("data", "market.db")
START_DATE = "2015-01-01"

# Logical asset name -> yfinance ticker
TICKERS: dict[str, str] = {
    "GOLD": "GC=F",
    "BITCOIN": "BTC-USD",
    "NVIDIA": "NVDA",
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS prices (
    asset TEXT NOT NULL,
    date DATE NOT NULL,
    open REAL, high REAL, low REAL, close REAL, volume REAL,
    PRIMARY KEY (asset, date)
);

CREATE TABLE IF NOT EXISTS backtest_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy TEXT NOT NULL,
    params_json TEXT NOT NULL,
    asset TEXT NOT NULL,
    sharpe REAL, max_drawdown REAL, final_value REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


# ----------------------------------------------------------------------
# DB plumbing
# ----------------------------------------------------------------------
def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Open a connection, creating the parent directory if needed."""
    parent = os.path.dirname(db_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return sqlite3.connect(db_path)


def init_db(db_path: str = DB_PATH) -> None:
    """Create tables if they don't already exist. Safe to re-run."""
    with get_connection(db_path) as conn:
        conn.executescript(SCHEMA)


def get_last_date(asset: str, db_path: str = DB_PATH) -> str | None:
    """Most recent stored date for an asset, as 'YYYY-MM-DD', or None."""
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT MAX(date) FROM prices WHERE asset = ?", (asset,)
        ).fetchone()
    return row[0] if row and row[0] else None


# ----------------------------------------------------------------------
# Fetch + normalise
# ----------------------------------------------------------------------
def _normalise(raw: pd.DataFrame, asset: str) -> pd.DataFrame:
    """yfinance frame -> tidy frame: asset, date, open, high, low, close, volume."""
    if raw is None or raw.empty:
        return pd.DataFrame(
            columns=["asset", "date", "open", "high", "low", "close", "volume"]
        )

    df = raw.copy()

    # yfinance returns a MultiIndex column frame when given a ticker list.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "datetime" in df.columns and "date" not in df.columns:
        df = df.rename(columns={"datetime": "date"})

    keep = ["date", "open", "high", "low", "close", "volume"]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise ValueError(f"{asset}: missing columns from yfinance: {missing}")

    df = df[keep].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.strftime("%Y-%m-%d")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # A row with no close is useless; volume is allowed to be NaN/0 (GC=F gaps).
    df = df.dropna(subset=["close"])
    df = df.drop_duplicates(subset=["date"]).sort_values("date")
    df.insert(0, "asset", asset)
    return df.reset_index(drop=True)


def fetch_asset(asset: str, start: str, end: str | None = None) -> pd.DataFrame:
    """Download one asset from yfinance. Imported lazily so the rest of the
    package works without a network connection."""
    import yfinance as yf

    ticker = TICKERS[asset]
    raw = yf.download(
        ticker,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    return _normalise(raw, asset)


# ----------------------------------------------------------------------
# Write
# ----------------------------------------------------------------------
def store_prices(df: pd.DataFrame, db_path: str = DB_PATH) -> int:
    """Insert rows, skipping any (asset, date) that already exists.

    Returns the number of rows actually inserted. Idempotent: re-running
    with the same frame inserts 0 rows.
    """
    if df is None or df.empty:
        return 0

    rows = df[["asset", "date", "open", "high", "low", "close", "volume"]].where(
        pd.notnull(df), None
    )
    records = [tuple(r) for r in rows.itertuples(index=False, name=None)]

    with get_connection(db_path) as conn:
        before = conn.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
        conn.executemany(
            "INSERT OR IGNORE INTO prices "
            "(asset, date, open, high, low, close, volume) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            records,
        )
        after = conn.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
    return after - before


def refresh_data(
    assets: list[str] | None = None,
    db_path: str = DB_PATH,
    start: str = START_DATE,
) -> dict[str, int]:
    """Fetch and append only new rows for each asset.

    Re-fetches from a few days before the last stored date so that late
    vendor corrections are picked up; INSERT OR IGNORE makes the overlap
    free of duplicates.
    """
    init_db(db_path)
    assets = assets or list(TICKERS)
    today = date.today().isoformat()
    inserted: dict[str, int] = {}

    for asset in assets:
        last = get_last_date(asset, db_path)
        if last:
            fetch_from = (
                pd.Timestamp(last) - pd.Timedelta(days=5)
            ).strftime("%Y-%m-%d")
        else:
            fetch_from = start

        try:
            df = fetch_asset(asset, start=fetch_from)
        except Exception as exc:  # network down, rate limit, bad ticker
            print(f"  {asset:<8} FETCH FAILED: {type(exc).__name__}: {exc}")
            inserted[asset] = 0
            continue

        n = store_prices(df, db_path)
        inserted[asset] = n
        span = f"{df['date'].min()} -> {df['date'].max()}" if not df.empty else "no rows"
        print(f"  {asset:<8} fetched {len(df):>5} rows ({span}), inserted {n} new")

    print(f"  as of {today}")
    return inserted


# ----------------------------------------------------------------------
# Read (used by the dashboard)
# ----------------------------------------------------------------------
def load_prices(
    asset: str,
    db_path: str = DB_PATH,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Read one asset back out as a tidy, date-ascending DataFrame."""
    query = "SELECT date, open, high, low, close, volume FROM prices WHERE asset = ?"
    params: list = [asset]
    if start:
        query += " AND date >= ?"
        params.append(str(start))
    if end:
        query += " AND date <= ?"
        params.append(str(end))
    query += " ORDER BY date ASC"

    with get_connection(db_path) as conn:
        df = pd.read_sql_query(query, conn, params=params)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


def status(db_path: str = DB_PATH) -> pd.DataFrame:
    """One row per asset: row count and date coverage."""
    if not os.path.exists(db_path):
        print(f"No database at {db_path}. Run: python ingestion.py")
        return pd.DataFrame()
    with get_connection(db_path) as conn:
        return pd.read_sql_query(
            "SELECT asset, COUNT(*) AS rows, MIN(date) AS first_date, "
            "MAX(date) AS last_date FROM prices GROUP BY asset ORDER BY asset",
            conn,
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest daily OHLCV into data/market.db")
    ap.add_argument("--db", default=DB_PATH)
    ap.add_argument("--status", action="store_true", help="print coverage and exit")
    ap.add_argument("--assets", nargs="*", default=None, choices=list(TICKERS))
    args = ap.parse_args()

    if args.status:
        print(status(args.db).to_string(index=False))
        return

    print(f"Refreshing {args.db} at {datetime.now():%Y-%m-%d %H:%M}")
    refresh_data(assets=args.assets, db_path=args.db)
    print()
    df = status(args.db)
    print(df.to_string(index=False) if not df.empty else "DB is empty.")


if __name__ == "__main__":
    main()
