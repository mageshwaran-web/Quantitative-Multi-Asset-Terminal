"""
REST backend — exposes the same engine over HTTP.

The analytics live in indicators.py and backtester.py; this file only reads
SQLite, validates input, and serialises results. It adds no maths of its own,
so the API and the Streamlit dashboard can never drift apart.

Run:
    python server.py                 # http://127.0.0.1:8000
    uvicorn server:app --reload      # same, with autoreload

Interactive docs: http://127.0.0.1:8000/docs

Endpoints
    GET  /health                       service + database status
    GET  /assets                       available assets and coverage
    GET  /prices/{asset}               OHLCV rows (optional date filter)
    GET  /metrics/{asset}              volatility, Sharpe, max drawdown, returns
    GET  /indicators/{asset}           SMA / EMA / returns / drawdown / rolling vol
    GET  /correlation                  correlation matrix across assets
    GET  /correlation/rolling          rolling correlation for a pair
    GET  /strategies                   strategy names and their parameters
    POST /backtest                     run one backtest
    POST /robustness                   sweep a parameter grid
    POST /regimes                      compare labelled date ranges
    GET  /backtest/history             previously saved runs
"""

from __future__ import annotations

import json
import os
import sqlite3
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import backtester as bt
import indicators as ind

DB_PATH = os.path.join("data", "market.db")
ASSETS = ["GOLD", "BITCOIN", "NVIDIA"]
PERIODS_PER_YEAR = {"BITCOIN": 365, "GOLD": 252, "NVIDIA": 252}

app = FastAPI(
    title="Quantitative Multi-Asset Backtesting API",
    version="1.0.0",
    description=(
        "Historical simulation only. Results describe how a rule would have "
        "behaved on one past sample after costs; they are not forecasts and "
        "carry no guarantee of future returns."
    ),
)

# Open CORS so a React/Vue/mobile client on another port can call this.
# Narrow allow_origins to your real frontend before deploying anywhere public.
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=False,
    allow_methods=["*"], allow_headers=["*"],
)


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _clean(value: Any) -> Any:
    """JSON has no NaN or Infinity. Convert them to null."""
    if isinstance(value, (np.floating, float)):
        f = float(value)
        return None if not np.isfinite(f) else f
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    return value


def series_to_json(s: pd.Series, dropna: bool = True) -> dict[str, list]:
    """Series -> {"dates": [...], "values": [...]} for easy charting."""
    if s is None or len(s) == 0:
        return {"dates": [], "values": []}
    s = s.dropna() if dropna else s
    idx = pd.to_datetime(s.index)
    return {
        "dates": [d.strftime("%Y-%m-%d") for d in idx],
        "values": [_clean(v) for v in s.to_numpy()],
    }


def frame_to_records(df: pd.DataFrame) -> list[dict]:
    if df is None or df.empty:
        return []
    return [{k: _clean(v) for k, v in row.items()}
            for row in df.to_dict(orient="records")]


def load_prices(asset: str, start: str | None = None,
                end: str | None = None) -> pd.DataFrame:
    """Read one asset from SQLite. 404 if the asset or database is absent."""
    asset = asset.upper()
    if asset not in ASSETS:
        raise HTTPException(404, f"Unknown asset '{asset}'. Available: {ASSETS}")
    if not os.path.exists(DB_PATH):
        raise HTTPException(
            503, f"No database at {DB_PATH}. Run `python ingestion.py` first.")

    sql = "SELECT date, open, high, low, close, volume FROM prices WHERE asset = ?"
    params: list = [asset]
    if start:
        sql += " AND date >= ?"
        params.append(start)
    if end:
        sql += " AND date <= ?"
        params.append(end)
    sql += " ORDER BY date ASC"

    try:
        with sqlite3.connect(DB_PATH) as conn:
            df = pd.read_sql_query(sql, conn, params=params)
    except sqlite3.Error as exc:
        raise HTTPException(500, f"Database error: {exc}") from exc

    if df.empty:
        raise HTTPException(404, f"No rows for {asset} in the requested range.")
    df["date"] = pd.to_datetime(df["date"])
    return df


DISCLAIMER = (
    "Historical simulation after costs. Not a forecast; past performance does "
    "not guarantee future returns."
)


# ----------------------------------------------------------------------
# request models
# ----------------------------------------------------------------------
class BacktestRequest(BaseModel):
    asset: str = Field(..., description="GOLD, BITCOIN or NVIDIA")
    strategy: str = Field(..., description="one of /strategies")
    params: dict = Field(default_factory=dict)
    initial_capital: float = 10_000.0
    transaction_cost_pct: float = Field(0.001, ge=0, lt=1)
    start: str | None = None
    end: str | None = None
    include_curves: bool = True


class RobustnessRequest(BaseModel):
    asset: str
    strategy: str
    param_grid: dict[str, list] = Field(
        ..., description='e.g. {"fast": [10, 20], "slow": [50, 100]}')
    initial_capital: float = 10_000.0
    transaction_cost_pct: float = Field(0.001, ge=0, lt=1)
    start: str | None = None
    end: str | None = None


class RegimeRequest(BaseModel):
    asset: str
    strategy: str
    params: dict = Field(default_factory=dict)
    regimes: dict[str, list[str]] = Field(
        ..., description='e.g. {"2022 Bear": ["2022-01-01", "2022-12-31"]}')
    initial_capital: float = 10_000.0
    transaction_cost_pct: float = Field(0.001, ge=0, lt=1)


# ----------------------------------------------------------------------
# meta
# ----------------------------------------------------------------------
@app.get("/health", tags=["meta"])
def health() -> dict:
    exists = os.path.exists(DB_PATH)
    rows = 0
    if exists:
        try:
            with sqlite3.connect(DB_PATH) as conn:
                rows = conn.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
        except sqlite3.Error:
            rows = 0
    return {
        "status": "ok" if rows else "no_data",
        "database": DB_PATH,
        "database_exists": exists,
        "price_rows": rows,
        "hint": None if rows else "Run `python ingestion.py` to populate the database.",
    }


@app.get("/assets", tags=["meta"])
def assets() -> dict:
    if not os.path.exists(DB_PATH):
        raise HTTPException(503, f"No database at {DB_PATH}. Run `python ingestion.py`.")
    with sqlite3.connect(DB_PATH) as conn:
        cov = pd.read_sql_query(
            "SELECT asset, COUNT(*) AS rows, MIN(date) AS first_date, "
            "MAX(date) AS last_date FROM prices GROUP BY asset", conn)
    return {"assets": ASSETS, "coverage": frame_to_records(cov),
            "periods_per_year": PERIODS_PER_YEAR}


@app.get("/strategies", tags=["meta"])
def strategies() -> dict:
    return {
        "strategies": list(bt.STRATEGIES),
        "parameters": {
            "sma_crossover": {"fast": 20, "slow": 50},
            "ema_trend": {"window": 50},
            "momentum": {"lookback": 60},
            "mean_reversion": {"window": 20, "std_threshold": 2.0},
        },
        "execution_model": (
            "Signal formed at day t close is filled at day t+1 open. Transaction "
            "cost is charged on both sides of every fill. Position sizing is "
            "all-in/all-out and capped by available cash."
        ),
    }


# ----------------------------------------------------------------------
# prices, metrics, indicators
# ----------------------------------------------------------------------
@app.get("/prices/{asset}", tags=["data"])
def prices(asset: str, start: str | None = None, end: str | None = None,
           limit: int = Query(0, ge=0, description="0 = all rows")) -> dict:
    df = load_prices(asset, start, end)
    if limit:
        df = df.tail(limit)
    out = df.copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    return {"asset": asset.upper(), "rows": len(out),
            "prices": frame_to_records(out)}


@app.get("/metrics/{asset}", tags=["analytics"])
def metrics(asset: str, start: str | None = None, end: str | None = None,
            risk_free_rate: float = 0.0) -> dict:
    df = load_prices(asset, start, end)
    ppy = PERIODS_PER_YEAR[asset.upper()]
    rets = ind.get_daily_returns(df)
    cum = ind.get_cumulative_returns(df)
    return {
        "asset": asset.upper(),
        "start": df["date"].min().strftime("%Y-%m-%d"),
        "end": df["date"].max().strftime("%Y-%m-%d"),
        "sessions": len(df),
        "periods_per_year": ppy,
        "last_close": _clean(df["close"].iloc[-1]),
        "cumulative_return_pct": _clean(cum.iloc[-1] * 100),
        "annualised_volatility_pct": _clean(
            ind.get_volatility(rets, periods_per_year=ppy) * 100),
        "daily_volatility_pct": _clean(
            ind.get_volatility(rets, annualize=False) * 100),
        "sharpe_ratio": _clean(
            ind.get_sharpe_ratio(rets, risk_free_rate, periods_per_year=ppy)),
        "max_drawdown_pct": _clean(ind.get_max_drawdown(df["close"]) * 100),
        "disclaimer": DISCLAIMER,
    }


@app.get("/indicators/{asset}", tags=["analytics"])
def indicators_endpoint(
    asset: str,
    sma: str = Query("20,50", description="comma-separated SMA windows"),
    ema: str = Query("20,50", description="comma-separated EMA windows"),
    rolling_return_window: int = Query(252, ge=1),
    rolling_vol_window: int = Query(30, ge=2),
    start: str | None = None,
    end: str | None = None,
) -> dict:
    df = load_prices(asset, start, end)
    ppy = PERIODS_PER_YEAR[asset.upper()]

    def windows(raw: str) -> list[int]:
        out = []
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            if not part.isdigit() or int(part) < 1:
                raise HTTPException(422, f"Invalid window '{part}'.")
            out.append(int(part))
        return out

    rets = ind.get_daily_returns(df)
    payload = {
        "asset": asset.upper(),
        "close": series_to_json(
            pd.Series(df["close"].to_numpy(), index=df["date"])),
        "sma": {str(w): series_to_json(ind.get_sma(df, w)) for w in windows(sma)},
        "ema": {str(w): series_to_json(ind.get_ema(df, w)) for w in windows(ema)},
        "daily_returns": series_to_json(rets),
        "cumulative_returns": series_to_json(ind.get_cumulative_returns(df)),
        "rolling_returns": series_to_json(
            ind.get_rolling_returns(df, rolling_return_window)),
        "rolling_volatility": series_to_json(
            ind.get_rolling_volatility(rets, rolling_vol_window,
                                       periods_per_year=ppy)),
        "drawdown": series_to_json(
            ind.get_drawdown_series(
                pd.Series(df["close"].to_numpy(), index=df["date"]))),
    }
    return payload


# ----------------------------------------------------------------------
# correlation
# ----------------------------------------------------------------------
@app.get("/correlation", tags=["analytics"])
def correlation(assets: str = Query(",".join(ASSETS)),
                start: str | None = None, end: str | None = None) -> dict:
    names = [a.strip().upper() for a in assets.split(",") if a.strip()]
    if len(names) < 2:
        raise HTTPException(422, "Provide at least two assets.")
    frames = {n: load_prices(n, start, end) for n in names}
    matrix = ind.get_correlation_matrix(frames)
    if matrix.empty:
        raise HTTPException(404, "Not enough overlapping trading days.")
    return {
        "assets": names,
        "matrix": {r: {c: _clean(matrix.loc[r, c]) for c in matrix.columns}
                   for r in matrix.index},
        "note": ("Daily returns are inner-joined on shared trading days before "
                 "correlating, so Bitcoin's weekend sessions are excluded rather "
                 "than padded with fabricated flat days."),
    }


@app.get("/correlation/rolling", tags=["analytics"])
def rolling_correlation(asset_a: str, asset_b: str, window: int = Query(90, ge=2),
                        start: str | None = None, end: str | None = None) -> dict:
    if asset_a.upper() == asset_b.upper():
        raise HTTPException(422, "Pick two different assets.")
    a = load_prices(asset_a, start, end)
    b = load_prices(asset_b, start, end)
    rolling = ind.get_rolling_correlation(a, b, window).dropna()
    if rolling.empty:
        raise HTTPException(404, "Not enough shared history for that window.")
    return {
        "pair": [asset_a.upper(), asset_b.upper()],
        "window": window,
        "mean": _clean(rolling.mean()),
        "min": _clean(rolling.min()),
        "max": _clean(rolling.max()),
        "series": series_to_json(rolling),
    }


# ----------------------------------------------------------------------
# backtesting
# ----------------------------------------------------------------------
@app.post("/backtest", tags=["backtest"])
def backtest(req: BacktestRequest) -> dict:
    df = load_prices(req.asset, req.start, req.end)
    ppy = PERIODS_PER_YEAR[req.asset.upper()]
    try:
        res = bt.run_backtest(
            df, req.strategy, req.params,
            initial_capital=req.initial_capital,
            transaction_cost_pct=req.transaction_cost_pct,
            periods_per_year=ppy)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    if res.get("warning"):
        raise HTTPException(422, res["warning"])

    eq, bench = res["equity_curve"], res["benchmark_curve"]
    payload = {
        "asset": req.asset.upper(),
        "strategy": req.strategy,
        "params": req.params,
        "initial_capital": req.initial_capital,
        "transaction_cost_pct": req.transaction_cost_pct,
        "metrics": {
            "sharpe": _clean(res["sharpe"]),
            "benchmark_sharpe": _clean(res["benchmark_sharpe"]),
            "total_return_pct": _clean(res["total_return_pct"]),
            "benchmark_total_return_pct": _clean(res["benchmark_total_return_pct"]),
            "volatility_pct": _clean(
                ind.get_volatility(eq.pct_change(), periods_per_year=ppy) * 100),
            "benchmark_volatility_pct": _clean(
                ind.get_volatility(bench.pct_change(), periods_per_year=ppy) * 100),
            "max_drawdown_pct": _clean(res["max_drawdown"] * 100),
            "benchmark_max_drawdown_pct": _clean(res["benchmark_max_drawdown"] * 100),
            "final_value": _clean(res["final_value"]),
            "benchmark_final_value": _clean(res["benchmark_final_value"]),
            "num_trades": res["num_trades"],
            "exposure_pct": _clean(res["exposure_pct"]),
        },
        "trades": [
            {"date": t["date"].strftime("%Y-%m-%d"), "action": t["action"],
             "price": _clean(t["price"]), "shares": _clean(t["shares"])}
            for t in res["trades"]
        ],
        "disclaimer": DISCLAIMER,
    }
    if req.include_curves:
        payload["equity_curve"] = series_to_json(eq, dropna=False)
        payload["benchmark_curve"] = series_to_json(bench, dropna=False)
        payload["drawdown_curve"] = series_to_json(ind.get_drawdown_series(eq))
    return payload


@app.post("/robustness", tags=["backtest"])
def robustness(req: RobustnessRequest) -> dict:
    df = load_prices(req.asset, req.start, req.end)
    try:
        table = bt.run_robustness_test(
            df, req.strategy, req.param_grid,
            initial_capital=req.initial_capital,
            transaction_cost_pct=req.transaction_cost_pct,
            periods_per_year=PERIODS_PER_YEAR[req.asset.upper()])
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    if table.empty:
        raise HTTPException(422, "No valid parameter combination produced a result.")

    sharpe = table["sharpe"].dropna()
    return {
        "asset": req.asset.upper(),
        "strategy": req.strategy,
        "combinations": len(table),
        "sharpe_best": _clean(sharpe.max()) if len(sharpe) else None,
        "sharpe_worst": _clean(sharpe.min()) if len(sharpe) else None,
        "sharpe_spread": _clean(sharpe.max() - sharpe.min()) if len(sharpe) else None,
        "sharpe_std": _clean(sharpe.std()) if len(sharpe) > 1 else None,
        "results": frame_to_records(table),
        "interpretation": (
            "Read the spread, not the best row. A strategy whose Sharpe collapses "
            "when a window shifts slightly was fitted to noise; only a broad plateau "
            "of similar results is weak evidence of a real effect."
        ),
        "disclaimer": DISCLAIMER,
    }


@app.post("/regimes", tags=["backtest"])
def regimes(req: RegimeRequest) -> dict:
    df = load_prices(req.asset)
    windows: dict[str, tuple] = {}
    for label, pair in req.regimes.items():
        if len(pair) != 2:
            raise HTTPException(422, f"Regime '{label}' needs exactly [start, end].")
        windows[label] = (pair[0], pair[1])
    try:
        table = bt.run_regime_analysis(
            df, req.strategy, req.params, windows,
            initial_capital=req.initial_capital,
            transaction_cost_pct=req.transaction_cost_pct,
            periods_per_year=PERIODS_PER_YEAR[req.asset.upper()])
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return {
        "asset": req.asset.upper(),
        "strategy": req.strategy,
        "params": req.params,
        "regimes": frame_to_records(table),
        "note": ("Each regime restarts from the initial capital and restarts the "
                 "indicator warm-up, so short windows can spend much of their length "
                 "flat — check exposure_pct before reading much into one regime."),
        "disclaimer": DISCLAIMER,
    }


@app.get("/backtest/history", tags=["backtest"])
def history(limit: int = Query(50, ge=1, le=500)) -> dict:
    if not os.path.exists(DB_PATH):
        raise HTTPException(503, f"No database at {DB_PATH}.")
    try:
        with sqlite3.connect(DB_PATH) as conn:
            df = pd.read_sql_query(
                "SELECT id, strategy, params_json, asset, sharpe, max_drawdown, "
                "final_value, created_at FROM backtest_results "
                "ORDER BY created_at DESC, id DESC LIMIT ?", conn, params=(limit,))
    except sqlite3.Error as exc:
        raise HTTPException(500, f"Database error: {exc}") from exc
    records = frame_to_records(df)
    for r in records:
        try:
            r["params"] = json.loads(r.pop("params_json"))
        except (TypeError, ValueError):
            r["params"] = {}
    return {"count": len(records), "results": records}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
