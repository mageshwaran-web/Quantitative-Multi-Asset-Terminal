"""
Layer 3 — Event-driven backtesting engine.

No SQLite, no Streamlit, no network. Consumes a price DataFrame and the
metric functions from indicators.py.

EXECUTION MODEL (read this before trusting any number below)
------------------------------------------------------------
  * A signal for day t is computed from data available up to and including
    day t's CLOSE.
  * That signal can only be acted on from day t+1 onward. Concretely,
    position[t+1] = signal[t], and the resulting order is filled at day
    t+1's OPEN price. Choosing next-day open (rather than next-day close)
    means the fill price is never contaminated by information that arrived
    after the decision was made.
  * Consequence: the very first bar is never traded, and the last signal in
    the series is never acted on. Both are correct.
  * transaction_cost_pct is charged on the full notional of every fill, on
    both buys and sells. A 0.001 cost means a round trip costs ~0.2%.
  * Position sizing is all-in / all-out (target weight 0 or 1). On a buy,
    share count is capped at cash / (price * (1 + cost)), so cash can never
    go negative. Fractional shares are permitted.
  * final_value marks the portfolio to the last close and does NOT charge a
    liquidation cost on any still-open position. The benchmark is treated
    identically, so the comparison stays apples-to-apples.
  * The benchmark is a real buy-and-hold simulation over the same bars:
    it buys at the same first tradable open, pays the same entry cost, and
    is marked to close daily.

Backtest results describe how a rule would have behaved on one historical
sample. They are not a forecast.
"""

from __future__ import annotations

import itertools
from typing import Any, Callable

import numpy as np
import pandas as pd

from indicators import (
    TRADING_DAYS,
    get_daily_returns,
    get_ema,
    get_max_drawdown,
    get_rolling_returns,
    get_sharpe_ratio,
    get_sma,
)

STRATEGIES = ("sma_crossover", "ema_trend", "momentum", "mean_reversion")


# ----------------------------------------------------------------------
# preparation
# ----------------------------------------------------------------------
def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Validate, sort, and index a price frame by date."""
    required = {"date", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"price frame missing columns: {sorted(missing)}")

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values("date").drop_duplicates(subset="date")
    out = out.dropna(subset=["close"]).reset_index(drop=True)

    # Some vendors leave gaps in 'open' (GC=F in particular). Falling back to
    # that same bar's close is the conservative choice: it never uses a price
    # from a later bar.
    if "open" not in out.columns:
        out["open"] = out["close"]
    out["open"] = pd.to_numeric(out["open"], errors="coerce").fillna(out["close"])
    out = out.set_index("date")
    return out


# ----------------------------------------------------------------------
# signals — each returns a 0/1 target position for day t using data <= t
# ----------------------------------------------------------------------
def _signal_sma_crossover(px: pd.DataFrame, params: dict) -> pd.Series:
    fast, slow = int(params.get("fast", 20)), int(params.get("slow", 50))
    if fast >= slow:
        raise ValueError(f"fast ({fast}) must be < slow ({slow})")
    frame = px.reset_index()
    f, s = get_sma(frame, fast), get_sma(frame, slow)
    sig = (f > s).astype(float)
    sig[f.isna() | s.isna()] = 0.0  # flat during warm-up
    return pd.Series(sig.to_numpy(), index=px.index)


def _signal_ema_trend(px: pd.DataFrame, params: dict) -> pd.Series:
    window = int(params.get("window", 50))
    frame = px.reset_index()
    ema = get_ema(frame, window)
    close = frame["close"].to_numpy(dtype=float)
    sig = (close > ema.to_numpy()).astype(float)
    sig[:window] = 0.0  # ignore the EMA's unconverged warm-up
    return pd.Series(sig, index=px.index)


def _signal_momentum(px: pd.DataFrame, params: dict) -> pd.Series:
    lookback = int(params.get("lookback", 60))
    frame = px.reset_index()
    roll = get_rolling_returns(frame, lookback)
    sig = (roll > 0).astype(float)
    sig[roll.isna()] = 0.0
    return pd.Series(sig.to_numpy(), index=px.index)


def _signal_mean_reversion(px: pd.DataFrame, params: dict) -> pd.Series:
    """Buy when close is `std_threshold` sigmas below its moving average;
    exit when it has reverted back to the average. Stateful, so it is
    built with an explicit forward pass — which also makes it obvious
    that bar i only ever reads z-scores at or before i."""
    window = int(params.get("window", 20))
    threshold = float(params.get("std_threshold", 2.0))
    close = px["close"]
    ma = close.rolling(window, min_periods=window).mean()
    sd = close.rolling(window, min_periods=window).std(ddof=1)
    z = ((close - ma) / sd.replace(0.0, np.nan)).to_numpy()

    sig = np.zeros(len(px))
    holding = 0.0
    for i, zi in enumerate(z):
        if np.isnan(zi):
            holding = 0.0
        elif holding == 0.0 and zi <= -threshold:
            holding = 1.0
        elif holding == 1.0 and zi >= 0.0:
            holding = 0.0
        sig[i] = holding
    return pd.Series(sig, index=px.index)


_DISPATCH: dict[str, Callable[[pd.DataFrame, dict], pd.Series]] = {
    "sma_crossover": _signal_sma_crossover,
    "ema_trend": _signal_ema_trend,
    "momentum": _signal_momentum,
    "mean_reversion": _signal_mean_reversion,
}


def generate_signal(df: pd.DataFrame, strategy_name: str, params: dict) -> pd.Series:
    """Target position (0 or 1) for each bar, decided at that bar's close."""
    if strategy_name not in _DISPATCH:
        raise ValueError(
            f"unknown strategy '{strategy_name}'. Options: {list(_DISPATCH)}"
        )
    return _DISPATCH[strategy_name](_prepare(df), params or {})


def min_bars_required(strategy_name: str, params: dict) -> int:
    """Warm-up bars a strategy needs before it can produce a live signal."""
    p = params or {}
    if strategy_name == "sma_crossover":
        return int(p.get("slow", 50))
    if strategy_name == "ema_trend":
        return int(p.get("window", 50))
    if strategy_name == "momentum":
        return int(p.get("lookback", 60))
    if strategy_name == "mean_reversion":
        return int(p.get("window", 20))
    return 1


# ----------------------------------------------------------------------
# core engine
# ----------------------------------------------------------------------
def _empty_result(initial_capital: float) -> dict[str, Any]:
    empty = pd.Series(dtype=float)
    return {
        "equity_curve": empty,
        "benchmark_curve": empty,
        "sharpe": float("nan"),
        "benchmark_sharpe": float("nan"),
        "max_drawdown": float("nan"),
        "benchmark_max_drawdown": float("nan"),
        "final_value": float(initial_capital),
        "total_return_pct": 0.0,
        "num_trades": 0,
        "trades": [],
        "warning": "Not enough data to run this strategy over this period.",
    }


def run_backtest(
    df: pd.DataFrame,
    strategy_name: str,
    params: dict,
    initial_capital: float = 10_000,
    transaction_cost_pct: float = 0.001,
    periods_per_year: int = TRADING_DAYS,
) -> dict[str, Any]:
    """Simulate one strategy against buy-and-hold. See module docstring for
    the execution model. Returns the dict described in the project spec."""
    if initial_capital <= 0:
        raise ValueError("initial_capital must be positive")
    if not 0 <= transaction_cost_pct < 1:
        raise ValueError("transaction_cost_pct must be in [0, 1)")
    if strategy_name not in _DISPATCH:
        raise ValueError(
            f"unknown strategy '{strategy_name}'. Options: {list(_DISPATCH)}"
        )

    px = _prepare(df)
    needed = min_bars_required(strategy_name, params) + 2
    if len(px) < needed:
        return _empty_result(initial_capital)

    signal = _DISPATCH[strategy_name](px, params or {})

    # THE line that prevents look-ahead: today's position was decided by
    # yesterday's close. Bar 0 is therefore always flat.
    position = signal.shift(1).fillna(0.0).to_numpy()

    opens = px["open"].to_numpy(dtype=float)
    closes = px["close"].to_numpy(dtype=float)
    dates = px.index
    cost = float(transaction_cost_pct)

    cash = float(initial_capital)
    shares = 0.0
    held = 0.0
    equity = np.empty(len(px), dtype=float)
    trades: list[dict] = []

    for i in range(len(px)):
        target = position[i]
        fill = opens[i]

        if target != held and fill > 0:
            if target > 0:  # enter long at today's open
                budget = cash / (fill * (1.0 + cost))
                if budget > 0:
                    shares = budget
                    cash -= shares * fill * (1.0 + cost)
                    cash = max(cash, 0.0)  # guard float dust
                    held = 1.0
                    trades.append(
                        {
                            "date": dates[i],
                            "action": "BUY",
                            "price": float(fill),
                            "shares": float(shares),
                        }
                    )
            else:  # exit to cash at today's open
                if shares > 0:
                    cash += shares * fill * (1.0 - cost)
                    trades.append(
                        {
                            "date": dates[i],
                            "action": "SELL",
                            "price": float(fill),
                            "shares": float(shares),
                        }
                    )
                    shares = 0.0
                held = 0.0

        equity[i] = cash + shares * closes[i]

    equity_curve = pd.Series(equity, index=dates, name="strategy")

    # Buy-and-hold over the same bars, same entry bar, same entry cost.
    bench_entry = opens[1] if len(opens) > 1 else opens[0]
    bench_shares = initial_capital / (bench_entry * (1.0 + cost))
    bench_cash = initial_capital - bench_shares * bench_entry * (1.0 + cost)
    bench = bench_cash + bench_shares * closes
    bench[0] = initial_capital  # not yet invested on bar 0
    benchmark_curve = pd.Series(bench, index=dates, name="buy_and_hold")

    strat_returns = equity_curve.pct_change()
    bench_returns = benchmark_curve.pct_change()
    final_value = float(equity_curve.iloc[-1])

    return {
        "equity_curve": equity_curve,
        "benchmark_curve": benchmark_curve,
        "sharpe": get_sharpe_ratio(strat_returns, periods_per_year=periods_per_year),
        "benchmark_sharpe": get_sharpe_ratio(
            bench_returns, periods_per_year=periods_per_year
        ),
        "max_drawdown": get_max_drawdown(equity_curve),
        "benchmark_max_drawdown": get_max_drawdown(benchmark_curve),
        "final_value": final_value,
        "total_return_pct": (final_value / initial_capital - 1.0) * 100.0,
        "num_trades": len(trades),
        "trades": trades,
        "exposure_pct": float(np.mean(position > 0) * 100.0),
        "benchmark_final_value": float(benchmark_curve.iloc[-1]),
        "benchmark_total_return_pct": (
            float(benchmark_curve.iloc[-1]) / initial_capital - 1.0
        )
        * 100.0,
    }


# ----------------------------------------------------------------------
# robustness
# ----------------------------------------------------------------------
def run_robustness_test(
    df: pd.DataFrame,
    strategy_name: str,
    param_grid: dict[str, list],
    initial_capital: float = 10_000,
    transaction_cost_pct: float = 0.001,
    periods_per_year: int = TRADING_DAYS,
) -> pd.DataFrame:
    """Run every combination in `param_grid` and return a comparison table.

    The point is dispersion, not the top row. A rule whose Sharpe collapses
    when a window moves by five days was fitted to noise; a broad plateau of
    similar results is the only thing that suggests otherwise. Sorting by
    Sharpe and reporting the winner is how backtests get oversold.
    """
    if not param_grid:
        raise ValueError("param_grid is empty")

    keys = list(param_grid)
    rows = []
    for combo in itertools.product(*(param_grid[k] for k in keys)):
        params = dict(zip(keys, combo))
        try:
            res = run_backtest(
                df,
                strategy_name,
                params,
                initial_capital=initial_capital,
                transaction_cost_pct=transaction_cost_pct,
                periods_per_year=periods_per_year,
            )
        except ValueError:
            continue  # e.g. fast >= slow
        row = dict(params)
        row.update(
            {
                "params": ", ".join(f"{k}={v}" for k, v in params.items()),
                "sharpe": res["sharpe"],
                "total_return_pct": res["total_return_pct"],
                "max_drawdown_pct": res["max_drawdown"] * 100.0,
                "final_value": res["final_value"],
                "num_trades": res["num_trades"],
            }
        )
        rows.append(row)

    if not rows:
        return pd.DataFrame(
            columns=[
                "params",
                "sharpe",
                "total_return_pct",
                "max_drawdown_pct",
                "final_value",
                "num_trades",
            ]
        )

    out = pd.DataFrame(rows)
    front = ["params", "sharpe", "total_return_pct", "max_drawdown_pct",
             "final_value", "num_trades"]
    return out[front + [c for c in out.columns if c not in front]]


def run_regime_analysis(
    df: pd.DataFrame,
    strategy_name: str,
    params: dict,
    regime_dates: dict[str, tuple],
    initial_capital: float = 10_000,
    transaction_cost_pct: float = 0.001,
    periods_per_year: int = TRADING_DAYS,
) -> pd.DataFrame:
    """Re-run the same strategy inside each labelled date window.

    Each regime is backtested standalone from `initial_capital`, so regimes
    are directly comparable to one another and to buy-and-hold over the same
    window. Note that each slice restarts the indicator warm-up, so a short
    regime may spend much of its length flat — 'n_bars' and 'exposure_pct'
    are there to make that visible rather than hidden.
    """
    px = _prepare(df).reset_index()
    rows = []

    for label, (start, end) in regime_dates.items():
        mask = (px["date"] >= pd.Timestamp(start)) & (px["date"] <= pd.Timestamp(end))
        window = px.loc[mask]
        if window.empty:
            rows.append({"regime": label, "start": start, "end": end, "n_bars": 0,
                         "note": "no data in range"})
            continue

        res = run_backtest(
            window,
            strategy_name,
            params,
            initial_capital=initial_capital,
            transaction_cost_pct=transaction_cost_pct,
            periods_per_year=periods_per_year,
        )
        rows.append(
            {
                "regime": label,
                "start": str(window["date"].min().date()),
                "end": str(window["date"].max().date()),
                "n_bars": len(window),
                "strategy_return_pct": res["total_return_pct"],
                "benchmark_return_pct": res.get("benchmark_total_return_pct", np.nan),
                "sharpe": res["sharpe"],
                "benchmark_sharpe": res.get("benchmark_sharpe", np.nan),
                "max_drawdown_pct": res["max_drawdown"] * 100.0,
                "benchmark_max_drawdown_pct": res["benchmark_max_drawdown"] * 100.0
                if pd.notna(res["benchmark_max_drawdown"])
                else np.nan,
                "num_trades": res["num_trades"],
                "exposure_pct": res.get("exposure_pct", np.nan),
                "note": res.get("warning", ""),
            }
        )

    return pd.DataFrame(rows)
