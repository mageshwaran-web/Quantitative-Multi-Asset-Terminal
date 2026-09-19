# Feature checklist — problem statement → implementation

Every row below was executed during build verification, not just written.

## Multi-Asset Data Processing
| Requirement | Where | Status |
|---|---|---|
| Collect Gold, Bitcoin, NVIDIA | `ingestion.py` → `TICKERS`, `refresh_data()` | ✅ |
| Normalise historical data | `ingestion.py` → `_normalise()` | ✅ |
| Optionally extend to other assets | add to `TICKERS`; the rest is asset-agnostic | ✅ |

## Quantitative Indicator Engine
| Requirement | Function | Dashboard | API |
|---|---|---|---|
| SMA | `get_sma` | Tab 1 overlay | `/indicators` |
| EMA | `get_ema` | Tab 1 overlay | `/indicators` |
| Daily returns | `get_daily_returns` | Tab 1 returns chart | `/indicators` |
| Cumulative returns | `get_cumulative_returns` | Tab 1 card + chart | `/indicators` |
| Historical volatility | `get_volatility(annualize=False)` | — | `/metrics` |
| Annualised volatility | `get_volatility` | Tab 1 card + rolling chart | `/metrics` |
| Sharpe ratio | `get_sharpe_ratio` | Tabs 1, 3, 4 | `/metrics`, `/backtest` |
| Maximum drawdown | `get_max_drawdown` | Tabs 1, 3, 4 | `/metrics`, `/backtest` |
| Rolling returns | `get_rolling_returns` | Tab 1 chart | `/indicators` |
| Rolling volatility (added) | `get_rolling_volatility` | Tab 1 chart | `/indicators` |
| Drawdown series (added) | `get_drawdown_series` | Tabs 1, 3 underwater charts | `/indicators` |

## Cross-Asset Correlation Analysis
| Requirement | Where | Status |
|---|---|---|
| Correlation matrix | `get_correlation_matrix` → Tab 2 heatmap, `/correlation` | ✅ |
| Rolling correlation | `get_rolling_correlation` → Tab 2 chart, `/correlation/rolling` | ✅ |
| Handles differing calendars | inner join on shared dates, never forward-fill | ✅ |

## Strategy Backtesting Engine
| Strategy | Signal function | Status |
|---|---|---|
| SMA Crossover | `_signal_sma_crossover` | ✅ |
| EMA Trend | `_signal_ema_trend` | ✅ |
| Momentum | `_signal_momentum` | ✅ |
| Mean Reversion | `_signal_mean_reversion` | ✅ |
| Simulates portfolio, not just signals | full equity curve in `run_backtest` | ✅ |

## Realistic Trading Simulation
| Requirement | Implementation | Status |
|---|---|---|
| Initial capital | `initial_capital` parameter | ✅ |
| Position sizing | all-in/all-out, capped at `cash / (price × (1+cost))` | ✅ |
| Transaction costs | charged on both sides of every fill | ✅ |
| Entry and exit prices | next-day **open**, recorded per trade | ✅ |
| Portfolio value | daily mark-to-close equity curve | ✅ |
| Number of trades | `num_trades` + full trade log | ✅ |

## Strategy vs Benchmark Comparison
| Metric | Status |
|---|---|
| Return | ✅ Tab 3 cards + table, `/backtest` |
| Sharpe ratio | ✅ |
| Volatility | ✅ (comparison table + API) |
| Maximum drawdown | ✅ (cards + overlaid drawdown chart) |
| Benchmark is a real buy-and-hold simulation | ✅ same entry bar, same entry cost |

## Strategy Robustness Testing
| Requirement | Where | Status |
|---|---|---|
| Vary moving-average periods | Tab 4 grid + `/robustness` | ✅ |
| Vary transaction costs | cost slider feeds the whole sweep | ✅ |
| Vary backtesting periods | Tab 3 date window; regimes in Tab 4 | ✅ |
| Surfaces over-optimisation | reports best/worst/spread/std, plus a Sharpe heatmap | ✅ |

## Market Regime Analysis
| Regime | Preset | Status |
|---|---|---|
| Bull market | 2021, 2023 | ✅ |
| Bear market | 2022 | ✅ |
| High volatility | 2020 COVID crash | ✅ |
| Low volatility | 2024-25 recent (compare rolling-vol chart) | ✅ |
| Custom ranges | any range via `/regimes` | ✅ |

## Interactive Financial Dashboard
| Visual | Location | Status |
|---|---|---|
| Price trends | Tab 1 | ✅ |
| SMA / EMA indicators | Tab 1 (both, toggleable) | ✅ |
| Returns | Tab 1 daily-returns bars, green/red | ✅ |
| Volatility | Tab 1 rolling 30-day annualised chart | ✅ |
| Drawdowns | Tab 1 + Tab 3 underwater charts | ✅ |
| Correlation heatmap | Tab 2 | ✅ |
| Buy/Sell signals | Tab 3 triangle markers on the equity curve | ✅ |
| Backtesting equity curves | Tab 3 | ✅ |
| Strategy vs benchmark | Tab 3 cards, curve overlay, table | ✅ |

## Financial Considerations
| Requirement | How it is handled |
|---|---|
| Not presented as guaranteed future returns | Disclaimer in the masthead, in every API payload, and in-context notes on Tabs 3 and 4 |
| Look-ahead bias | `position = signal.shift(1)` + fills at next-day open. Verified by scrambling all prices after bar 121 — pre-cut trades and equity stayed bit-identical |
| Data leakage | No forward-fill across calendars; correlations inner-join; each regime restarts its own warm-up |
| Unrealistic trade execution | Costs both sides, cash never negative, no same-bar fills, fills at open not close |
| Over-optimisation | Robustness tab reports spread/std and a parameter heatmap rather than only the winning row |

## Not implemented (listed as Future Scope in the PDF)
Portfolio optimisation · Monte Carlo · Value at Risk · ML regime detection ·
paper trading · real-time data · AI research assistant.

Also still outside the model: slippage and market impact, bid-ask spread beyond
the flat cost, borrow costs, dividends and futures roll yield, taxes. All of
these push live results **below** the simulated figures.
