# Quantitative Multi-Asset Intelligence & Backtesting Platform

Gold (`GC=F`), Bitcoin (`BTC-USD`) and NVIDIA (`NVDA`) — ingestion, indicators,
backtesting and a Streamlit dashboard.

## Setup

```bash
pip install -r requirements.txt
python ingestion.py          # populates data/market.db (needs internet)
python ingestion.py --status # row counts and date coverage per asset
streamlit run dashboard.py
```

`ingestion.py` is re-runnable: it re-fetches from a few days before the last
stored date and relies on `INSERT OR IGNORE` against the `(asset, date)`
primary key, so repeated runs append only genuinely new rows.

## Layer boundaries

| File | Does | Never touches |
|---|---|---|
| `ingestion.py` | yfinance → SQLite | indicators, backtests, Streamlit |
| `indicators.py` | pure pandas/numpy maths | SQLite, Streamlit, network |
| `backtester.py` | simulation, imports `indicators` | SQLite, Streamlit, network |
| `dashboard.py` | sqlite3 reads + Plotly rendering | yfinance; computes no maths itself |

## Execution model (how look-ahead bias is prevented)

A signal for day *t* is computed from data through day *t*'s **close**. It is
acted on from day *t+1*, and the fill happens at day *t+1*'s **open**. In code
this is the single line `position = signal.shift(1)` in `run_backtest`, plus
filling at `opens[i]` rather than `closes[i]`. Choosing next-day open means the
fill price cannot contain information that arrived after the decision.

Consequences that are correct, not bugs: the first bar is never traded, the
final signal is never acted on, and entries lag turning points by at least one
bar — a V-shaped test series is bought *after* the bottom, never at it.

Other guarantees:
- `transaction_cost_pct` is charged on the full notional of every fill, both sides.
- Buy size is capped at `cash / (price * (1 + cost))`, so cash cannot go negative.
- Assets keep their native calendars in the DB; correlation inner-joins on shared
  dates rather than forward-filling Bitcoin's weekends into Gold, which would
  manufacture fake flat days and pull correlations toward zero.
- `final_value` marks to the last close without charging a liquidation cost, and
  the benchmark is treated identically so the comparison stays symmetric.
- Bitcoin annualises with 365 periods/year, Gold and NVIDIA with 252.

## What these numbers are not

Everything the dashboard shows is a simulation of one rule over one historical
sample. It is not a forecast and not a guarantee of future returns. Three
things in particular are deliberately surfaced rather than hidden:

- **The robustness tab reports the spread of Sharpe across a parameter grid, not
  just the best row.** A rule that only works at one setting is fitted to noise.
- **The regime tab shows `exposure_pct` and `n_bars`** because each slice restarts
  the indicator warm-up, so a short regime can spend most of its length flat.
- **The backtest tab says so plainly when the strategy underperforms buy-and-hold.**

Real-world frictions still absent from the model: slippage and market impact,
bid-ask spread beyond the flat cost, borrow costs, dividends and futures roll
yield, survivorship and vendor-revision effects, and taxes. Live results would
be worse than what you see here, not better.

## Verification performed

- DB schema, idempotent re-runs (0 duplicate rows), overlapping appends, read-back.
- Indicators checked against hand-computed values on a 6-row frame, and Sharpe/
  volatility against manual formulas; correlation of two independent random walks
  came out at 0.009 with a unit diagonal and symmetry.
- Backtester: fill lands on the bar *after* the first true signal at that bar's
  open; scrambling every price after bar 121 left all pre-cut trades and the
  pre-cut equity curve bit-identical (the look-ahead probe); cash stays
  non-negative; higher costs strictly reduce final value.
- Dashboard: imported against stubbed `streamlit`/`plotly` with a synthetic DB so
  all four tabs and every button path executed; missing-DB and empty-DB cases
  warn and stop instead of crashing.

---

# REST API (server.py)

The same engine over HTTP, so a React/Vue/mobile client can use it without
Streamlit. `server.py` reads SQLite, validates input and serialises results —
it adds no maths, so the API and the dashboard can never drift apart.

```powershell
python -m pip install -r requirements.txt
python ingestion.py
python server.py            # http://127.0.0.1:8000  (docs at /docs)
streamlit run dashboard.py  # separate terminal, http://localhost:8501
```

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | service + database status |
| GET | `/assets` | available assets and coverage |
| GET | `/prices/{asset}` | OHLCV rows |
| GET | `/metrics/{asset}` | volatility, Sharpe, max drawdown, returns |
| GET | `/indicators/{asset}` | SMA, EMA, returns, rolling vol, drawdown |
| GET | `/correlation` | correlation matrix |
| GET | `/correlation/rolling` | rolling correlation for a pair |
| GET | `/strategies` | strategy names, default params, execution model |
| POST | `/backtest` | run one backtest |
| POST | `/robustness` | sweep a parameter grid |
| POST | `/regimes` | compare labelled date ranges |
| GET | `/backtest/history` | previously saved runs |

Example:

```bash
curl -X POST http://127.0.0.1:8000/backtest -H "Content-Type: application/json" \
  -d '{"asset":"NVIDIA","strategy":"sma_crossover","params":{"fast":20,"slow":50}}'
```

CORS is wide open for local development. Narrow `allow_origins` in `server.py`
before exposing this anywhere public.

# Dashboard notes

Dark terminal theme: charcoal base, glassmorphic metric cards, cyan/blue
accents, green and red reserved for gains and losses. Plotly charts use a
transparent background with a 78px top margin so the title, the legend and
Plotly's modebar (zoom / pan / reset / download, top-right of every chart)
never overlap.

Backtest, robustness and regime results are held in `st.session_state`, so they
stay on screen while you change other controls and only recompute when you press
the button again — the earlier version discarded results on every rerun, which
is what made the buttons feel unresponsive.


