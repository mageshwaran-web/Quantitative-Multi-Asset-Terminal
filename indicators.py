"""
Layer 2 — Indicators, risk metrics and correlation.

Pure pandas/numpy. No SQLite, no Streamlit, no network.

Every function takes a DataFrame with columns
    date, open, high, low, close, volume
sorted ascending by date. Returned Series are indexed by date when a
'date' column is present, otherwise by the frame's own index.

Conventions:
  * Returns are simple (arithmetic) daily returns, not log returns.
  * `periods_per_year` defaults to 252 (equities/futures). Pass 365 for
    Bitcoin, which trades every calendar day.
  * `risk_free_rate` is an ANNUAL rate (0.04 = 4%) and is de-annualised
    internally.
  * get_max_drawdown returns a NEGATIVE fraction: -0.35 means -35%.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ----------------------------------------------------------------------
# internals
# ----------------------------------------------------------------------
def _series(df: pd.DataFrame, column: str = "close") -> pd.Series:
    """Pull one column out of a price frame, indexed by date if available."""
    if column not in df.columns:
        raise KeyError(f"DataFrame has no '{column}' column (got {list(df.columns)})")
    s = pd.Series(df[column].to_numpy(dtype=float), name=column)
    if "date" in df.columns:
        s.index = pd.to_datetime(df["date"]).to_numpy()
    else:
        s.index = df.index
    return s


def _clean(returns: pd.Series) -> pd.Series:
    """Drop NaN/inf so a single bad tick can't poison a whole statistic."""
    r = pd.Series(returns).astype(float)
    return r.replace([np.inf, -np.inf], np.nan).dropna()


# ----------------------------------------------------------------------
# trend indicators
# ----------------------------------------------------------------------
def get_sma(df: pd.DataFrame, window: int) -> pd.Series:
    """Simple moving average of close. NaN until `window` observations exist."""
    if window < 1:
        raise ValueError("window must be >= 1")
    return _series(df).rolling(window=window, min_periods=window).mean().rename(
        f"sma_{window}"
    )


def get_ema(df: pd.DataFrame, window: int) -> pd.Series:
    """Exponential moving average of close (span=window, no warm-up bias
    correction, so it matches what a trader sees on a chart)."""
    if window < 1:
        raise ValueError("window must be >= 1")
    return _series(df).ewm(span=window, adjust=False).mean().rename(f"ema_{window}")


# ----------------------------------------------------------------------
# returns
# ----------------------------------------------------------------------
def get_daily_returns(df: pd.DataFrame) -> pd.Series:
    """Day-over-day simple return of close. First element is NaN."""
    return _series(df).pct_change().rename("daily_return")


def get_cumulative_returns(df: pd.DataFrame) -> pd.Series:
    """Growth of 1 unit since the first row, expressed as a fraction.
    0.25 means +25% since inception."""
    r = get_daily_returns(df).fillna(0.0)
    return ((1.0 + r).cumprod() - 1.0).rename("cumulative_return")


def get_rolling_returns(df: pd.DataFrame, window: int) -> pd.Series:
    """Trailing `window`-day total return, e.g. window=252 -> 1y return."""
    if window < 1:
        raise ValueError("window must be >= 1")
    return _series(df).pct_change(periods=window).rename(f"rolling_return_{window}")


# ----------------------------------------------------------------------
# risk metrics
# ----------------------------------------------------------------------
def get_volatility(
    returns: pd.Series,
    annualize: bool = True,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    """Standard deviation of returns (sample, ddof=1), annualised by default."""
    r = _clean(returns)
    if len(r) < 2:
        return float("nan")
    vol = float(r.std(ddof=1))
    return vol * np.sqrt(periods_per_year) if annualize else vol


def get_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    """Annualised Sharpe ratio. `risk_free_rate` is annual and is converted
    to a per-period rate before subtracting. Returns NaN if volatility is
    zero or there are fewer than two observations."""
    r = _clean(returns)
    if len(r) < 2:
        return float("nan")
    rf_per_period = risk_free_rate / periods_per_year
    excess = r - rf_per_period
    sd = float(excess.std(ddof=1))
    if sd == 0 or not np.isfinite(sd):
        return float("nan")
    return float(excess.mean() / sd * np.sqrt(periods_per_year))


def get_max_drawdown(prices: pd.Series) -> float:
    """Largest peak-to-trough decline of a price or equity series.
    Returned as a negative fraction (-0.35 == a 35% drawdown)."""
    p = pd.Series(prices).astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if p.empty:
        return float("nan")
    running_peak = p.cummax()
    drawdown = p / running_peak - 1.0
    return float(drawdown.min())


def get_rolling_volatility(
    returns: pd.Series,
    window: int = 30,
    annualize: bool = True,
    periods_per_year: int = TRADING_DAYS,
) -> pd.Series:
    """Trailing `window`-day standard deviation of returns, annualised by
    default. Used for the volatility-over-time chart and for eyeballing
    high- vs low-volatility regimes."""
    if window < 2:
        raise ValueError("window must be >= 2")
    r = pd.Series(returns).astype(float).replace([np.inf, -np.inf], np.nan)
    vol = r.rolling(window=window, min_periods=window).std(ddof=1)
    if annualize:
        vol = vol * np.sqrt(periods_per_year)
    return vol.rename(f"rolling_vol_{window}")


def get_drawdown_series(prices: pd.Series) -> pd.Series:
    """Underwater curve: percentage below the running peak at every point.
    Always <= 0. get_max_drawdown() is the minimum of this series."""
    p = pd.Series(prices).astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if p.empty:
        return pd.Series(dtype=float, name="drawdown")
    return (p / p.cummax() - 1.0).rename("drawdown")


# ----------------------------------------------------------------------
# cross-asset
# ----------------------------------------------------------------------
def get_correlation_matrix(asset_dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Correlation of daily returns across assets.

    Assets are stored on their native calendars (Bitcoin trades weekends,
    Gold and NVIDIA do not), so returns are inner-joined on shared dates
    first. Correlating unaligned series, or forward-filling weekends into
    Gold, would manufacture spurious zero-return days and bias the result
    toward zero.
    """
    if not asset_dfs:
        return pd.DataFrame()

    cols = {}
    for name, df in asset_dfs.items():
        if df is None or df.empty:
            continue
        cols[name] = get_daily_returns(df)
    if not cols:
        return pd.DataFrame()

    joined = pd.concat(cols, axis=1, join="inner").dropna()
    if joined.empty or joined.shape[0] < 2:
        return pd.DataFrame(index=list(cols), columns=list(cols), dtype=float)
    return joined.corr()


def get_rolling_correlation(
    df1: pd.DataFrame, df2: pd.DataFrame, window: int
) -> pd.Series:
    """Rolling correlation of two assets' daily returns over `window` shared
    trading days. Inner-joined on date before the rolling window is applied,
    so the window always covers `window` days both assets actually traded."""
    if window < 2:
        raise ValueError("window must be >= 2")
    a = get_daily_returns(df1).rename("a")
    b = get_daily_returns(df2).rename("b")
    joined = pd.concat([a, b], axis=1, join="inner").dropna()
    if joined.empty:
        return pd.Series(dtype=float, name=f"rolling_corr_{window}")
    return (
        joined["a"]
        .rolling(window=window, min_periods=window)
        .corr(joined["b"])
        .rename(f"rolling_corr_{window}")
    )
