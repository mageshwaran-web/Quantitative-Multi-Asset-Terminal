"""
Portfolio Optimization — Modern Portfolio Theory (Markowitz).

Pure pandas/numpy/scipy. No SQLite, no Streamlit, no network — same rule as
indicators.py and backtester.py.

CRITICAL CAVEAT (read before trusting any output here)
--------------------------------------------------------
Every number in this module is computed from ONE historical sample of
returns. The "optimal" weights that maximised Sharpe over the last N years
are the single most over-fit number in quantitative finance: they are
optimal *for that specific sample*, not for the future. Correlations and
volatilities both drift over time (see indicators.get_rolling_correlation),
so a portfolio optimized on 2015-2020 data can be badly wrong for 2021-2026.

This module does not predict, forecast, or recommend an allocation. It
answers a narrower question: "given how these assets moved together in the
past, what weight combinations would have produced the best historical
risk/return trade-off?" That is backward-looking analysis, not investment
advice.

Inputs
------
Every function takes a dict[str, pd.DataFrame] of price frames (the same
shape produced by ingestion.load_prices / dashboard.load_prices), each with
at least 'date' and 'close' columns. Assets are aligned on shared trading
days via an inner join — same convention as indicators.get_correlation_matrix
— so Bitcoin's weekend sessions never get forward-filled into Gold or NVIDIA.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from indicators import TRADING_DAYS, get_daily_returns


# ----------------------------------------------------------------------
# shared return matrix
# ----------------------------------------------------------------------
def get_aligned_returns(asset_dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Daily returns for every asset, inner-joined on shared trading days.

    Mirrors indicators.get_correlation_matrix's alignment rule: assets keep
    their native calendars everywhere else, and are only aligned here,
    at the point where a covariance matrix actually requires it.
    """
    if not asset_dfs:
        return pd.DataFrame()
    cols = {name: get_daily_returns(df) for name, df in asset_dfs.items()
            if df is not None and not df.empty}
    if len(cols) < 2:
        return pd.DataFrame()
    return pd.concat(cols, axis=1, join="inner").dropna()


def annualized_mean_cov(
    returns: pd.DataFrame, periods_per_year: int = TRADING_DAYS
) -> tuple[pd.Series, pd.DataFrame]:
    """Annualised mean return vector and covariance matrix."""
    mu = returns.mean() * periods_per_year
    cov = returns.cov() * periods_per_year
    return mu, cov


# ----------------------------------------------------------------------
# portfolio math
# ----------------------------------------------------------------------
def portfolio_performance(
    weights: np.ndarray, mu: pd.Series, cov: pd.DataFrame,
    risk_free_rate: float = 0.0,
) -> dict[str, float]:
    """Expected return, volatility and Sharpe of one weight vector."""
    w = np.asarray(weights, dtype=float)
    ret = float(w @ mu.to_numpy())
    vol = float(np.sqrt(w @ cov.to_numpy() @ w))
    sharpe = (ret - risk_free_rate) / vol if vol > 0 else float("nan")
    return {"return": ret, "volatility": vol, "sharpe": sharpe}


def _constraints(allow_short: bool, n: int):
    bounds = [(-1.0, 1.0) if allow_short else (0.0, 1.0)] * n
    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    return bounds, cons


def max_sharpe_portfolio(
    mu: pd.Series, cov: pd.DataFrame, risk_free_rate: float = 0.0,
    allow_short: bool = False,
) -> dict:
    """Weights that maximise historical Sharpe ratio, subject to weights
    summing to 1 (and to being non-negative unless allow_short=True)."""
    n = len(mu)
    bounds, cons = _constraints(allow_short, n)
    x0 = np.full(n, 1.0 / n)

    def neg_sharpe(w):
        perf = portfolio_performance(w, mu, cov, risk_free_rate)
        return -perf["sharpe"] if np.isfinite(perf["sharpe"]) else 1e6

    res = minimize(neg_sharpe, x0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-10})
    weights = res.x if res.success else x0
    perf = portfolio_performance(weights, mu, cov, risk_free_rate)
    return {"weights": dict(zip(mu.index, weights)), **perf,
            "converged": bool(res.success)}


def min_volatility_portfolio(
    mu: pd.Series, cov: pd.DataFrame, allow_short: bool = False,
) -> dict:
    """Weights that minimise historical portfolio volatility."""
    n = len(mu)
    bounds, cons = _constraints(allow_short, n)
    x0 = np.full(n, 1.0 / n)

    def vol(w):
        return float(np.sqrt(w @ cov.to_numpy() @ w))

    res = minimize(vol, x0, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter": 500, "ftol": 1e-10})
    weights = res.x if res.success else x0
    perf = portfolio_performance(weights, mu, cov)
    return {"weights": dict(zip(mu.index, weights)), **perf,
            "converged": bool(res.success)}


def target_return_portfolio(
    mu: pd.Series, cov: pd.DataFrame, target_return: float,
    allow_short: bool = False,
) -> dict | None:
    """Minimum-volatility weights that achieve at least `target_return`.
    Returns None if the target is outside the feasible range (e.g. above
    what the highest-returning single asset could deliver)."""
    n = len(mu)
    bounds, cons_base = _constraints(allow_short, n)
    cons = cons_base + (
        {"type": "ineq", "fun": lambda w: w @ mu.to_numpy() - target_return},
    )
    x0 = np.full(n, 1.0 / n)

    def vol(w):
        return float(np.sqrt(w @ cov.to_numpy() @ w))

    res = minimize(vol, x0, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter": 500, "ftol": 1e-10})
    if not res.success:
        return None
    perf = portfolio_performance(res.x, mu, cov)
    return {"weights": dict(zip(mu.index, res.x)), **perf, "converged": True}


def efficient_frontier(
    mu: pd.Series, cov: pd.DataFrame, n_points: int = 40,
    allow_short: bool = False,
) -> pd.DataFrame:
    """Sweep target returns from the min-vol portfolio's return up to the
    single best-performing asset's return, solving min-vol at each point.
    Returns a DataFrame of (return, volatility, sharpe, weights...) rows,
    sorted by volatility — the frontier a chart would plot."""
    min_vol = min_volatility_portfolio(mu, cov, allow_short=allow_short)
    lo = min_vol["return"]
    hi = float(mu.max())
    if hi <= lo:
        hi = lo + 1e-6

    rows = []
    for target in np.linspace(lo, hi, n_points):
        port = target_return_portfolio(mu, cov, target, allow_short=allow_short)
        if port is None:
            continue
        row = {"return": port["return"], "volatility": port["volatility"],
               "sharpe": port["sharpe"]}
        row.update({f"w_{k}": v for k, v in port["weights"].items()})
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("volatility").reset_index(drop=True)


def equal_weight_portfolio(mu: pd.Series, cov: pd.DataFrame) -> dict:
    """The naive 1/N benchmark every optimizer should be compared against."""
    n = len(mu)
    weights = np.full(n, 1.0 / n)
    perf = portfolio_performance(weights, mu, cov)
    return {"weights": dict(zip(mu.index, weights)), **perf, "converged": True}


def optimize_portfolio(
    asset_dfs: dict[str, pd.DataFrame],
    risk_free_rate: float = 0.0,
    allow_short: bool = False,
    periods_per_year: int = TRADING_DAYS,
    frontier_points: int = 40,
) -> dict:
    """Convenience entry point: run everything a dashboard tab needs in one
    call. Returns max-Sharpe, min-vol and equal-weight portfolios, plus the
    efficient frontier and the return/covariance inputs they were built from.
    """
    returns = get_aligned_returns(asset_dfs)
    if returns.empty or len(returns) < 30:
        return {
            "warning": "Not enough overlapping trading days across the "
                       "selected assets to optimize a portfolio.",
            "assets": list(asset_dfs),
        }

    mu, cov = annualized_mean_cov(returns, periods_per_year)
    max_sharpe = max_sharpe_portfolio(mu, cov, risk_free_rate, allow_short)
    min_vol = min_volatility_portfolio(mu, cov, allow_short)
    equal_weight = equal_weight_portfolio(mu, cov)
    frontier = efficient_frontier(mu, cov, frontier_points, allow_short)

    return {
        "assets": list(mu.index),
        "n_observations": len(returns),
        "start_date": str(returns.index.min().date()),
        "end_date": str(returns.index.max().date()),
        "expected_returns": mu.to_dict(),
        "covariance_matrix": cov,
        "correlation_matrix": returns.corr(),
        "max_sharpe": max_sharpe,
        "min_volatility": min_vol,
        "equal_weight": equal_weight,
        "efficient_frontier": frontier,
    }
