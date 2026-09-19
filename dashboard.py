"""
Layer 4 — Streamlit dashboard (dark quant-terminal build).

Reads ONLY from data/market.db via sqlite3. Never calls yfinance.
Computes no analytics of its own: every number and every series on screen
comes from indicators.py or backtester.py.

Run with:  streamlit run dashboard.py

FEATURE MAP (problem statement -> where it appears)
  Multi-asset processing ......... sidebar coverage panel, all tabs
  SMA / EMA ...................... Tab 1 overlay selector (both plotted)
  Daily + cumulative returns ..... Tab 1 returns chart + metric card
  Volatility (hist + annualised) . Tab 1 metric card + rolling-vol chart
  Sharpe ratio ................... Tab 1 card, Tab 3 comparison, Tab 4 tables
  Maximum drawdown ............... Tab 1 card + underwater chart, Tab 3, Tab 4
  Rolling returns ................ Tab 1 rolling-performance chart
  Correlation matrix ............. Tab 2 heatmap
  Rolling correlation ............ Tab 2 chart
  4 strategies ................... Tab 3 strategy selector
  Realistic simulation ........... Tab 3 (capital, costs, sizing, fills, trades)
  Strategy vs benchmark .......... Tab 3 cards + equity + drawdown overlay
  Buy/sell signals ............... Tab 3 markers on the equity curve
  Robustness testing ............. Tab 4 param grid + Sharpe surface
  Market regime analysis ......... Tab 4 (bull / bear / high-vol / low-vol)
"""

from __future__ import annotations

import json
import os
import sqlite3

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import backtester as bt
import indicators as ind
import portfolio as pf
import ai_assistant as ai

DB_PATH = os.path.join("data", "market.db")
ASSETS = ["GOLD", "BITCOIN", "NVIDIA"]
PERIODS_PER_YEAR = {"BITCOIN": 365, "GOLD": 252, "NVIDIA": 252}

# palette
BG = "#0B0E14"
PANEL = "rgba(255,255,255,0.035)"
GRID = "rgba(255,255,255,0.06)"
TEXT = "#E6EAF2"
MUTED = "#8A93A6"
CYAN = "#22D3EE"
BLUE = "#5B8DEF"
VIOLET = "#A78BFA"
GREEN = "#34D399"
RED = "#F87171"
AMBER = "#FBBF24"

st.set_page_config(
    page_title="Quant Terminal", page_icon="◧", layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {{ font-family:'Inter',system-ui,sans-serif; }}
.stApp {{ background:{BG}; color:{TEXT}; }}
.block-container {{ padding:2.2rem 2.6rem 4rem; max-width:1500px; }}
#MainMenu, footer {{ visibility:hidden; }}

h1,h2,h3,h4 {{ color:{TEXT}; letter-spacing:-0.02em; font-weight:600; }}
h1 {{ font-size:1.9rem !important; margin-bottom:.2rem !important; }}
h3 {{ font-size:1.05rem !important; margin:1.6rem 0 .5rem !important; }}
p, label, .stMarkdown {{ color:{MUTED}; }}

.hero {{
  border:1px solid rgba(255,255,255,.08); border-radius:18px; padding:1.5rem 1.8rem;
  background:linear-gradient(135deg, rgba(34,211,238,.07), rgba(91,141,239,.03) 55%, transparent);
  backdrop-filter:blur(14px); margin-bottom:.4rem;
}}
.hero .eyebrow {{
  font-family:'JetBrains Mono',monospace; font-size:.68rem; letter-spacing:.22em;
  text-transform:uppercase; color:{CYAN}; margin-bottom:.45rem;
}}
.hero .sub {{ color:{MUTED}; font-size:.87rem; line-height:1.55; max-width:78ch; margin:0; }}

.cardrow {{ display:flex; gap:14px; flex-wrap:wrap; margin:.5rem 0 .3rem; }}
.card {{
  flex:1 1 170px; min-width:160px; padding:16px 18px; border-radius:16px;
  border:1px solid rgba(255,255,255,.08); background:{PANEL};
  backdrop-filter:blur(12px); box-shadow:0 8px 26px rgba(0,0,0,.34);
}}
.card .k {{
  font-family:'JetBrains Mono',monospace; font-size:.63rem; letter-spacing:.15em;
  text-transform:uppercase; color:{MUTED}; margin-bottom:.5rem; white-space:nowrap;
  overflow:hidden; text-overflow:ellipsis;
}}
.card .v {{ font-size:1.5rem; font-weight:600; line-height:1.15; letter-spacing:-.02em; }}
.card .n {{ font-size:.72rem; color:{MUTED}; margin-top:.35rem; }}
.pos {{ color:{GREEN}; }} .neg {{ color:{RED}; }} .acc {{ color:{CYAN}; }} .neu {{ color:{TEXT}; }}

.stTabs [data-baseweb="tab-list"] {{ gap:6px; border-bottom:1px solid rgba(255,255,255,.07); }}
.stTabs [data-baseweb="tab"] {{
  height:44px; padding:0 20px; background:transparent; border-radius:10px 10px 0 0;
  color:{MUTED}; font-size:.87rem; font-weight:500;
}}
.stTabs [aria-selected="true"] {{ background:rgba(34,211,238,.09); color:{CYAN} !important; }}

.stSelectbox div[data-baseweb="select"] > div, .stMultiSelect div[data-baseweb="select"] > div,
.stNumberInput input, .stDateInput input {{
  background:rgba(255,255,255,.04) !important; border:1px solid rgba(255,255,255,.1) !important;
  border-radius:8px !important; color:{TEXT} !important;
}}
.stDateInput {{ width: 100%; }}
.stDateInput div[data-testid="dateInputContainer"] {{ display: flex; flex-direction: column; gap: 8px; }}
[data-testid="stDateInput"] input {{ cursor: pointer; }}
.stSlider [data-baseweb="slider"] div[role="slider"] {{ background:{CYAN} !important; }}

.stButton>button {{
  width:100%; border-radius:8px; padding:10px 16px; font-weight:600; font-size:.9rem;
  border: 1px solid {CYAN}; color:{BG};
  background-color: {CYAN}; transition: all .15s ease;
}}
.stButton>button:hover {{ 
  background-color: {BLUE};
  border-color: {BLUE};
  box-shadow: 0 4px 12px rgba(34,211,238,.3);
}}
.stButton>button:active {{ opacity: 0.95; }}

.stDataFrame {{ border:1px solid rgba(255,255,255,.08); border-radius:14px; overflow:hidden; }}
[data-testid="stSidebar"] {{ background:#080A10; border-right:1px solid rgba(255,255,255,.06); }}
hr {{ border-color:rgba(255,255,255,.07); }}
.note {{
  font-size:.78rem; color:{MUTED}; border-left:2px solid rgba(34,211,238,.45);
  padding:.45rem 0 .45rem .8rem; margin:.7rem 0 .2rem; line-height:1.6;
}}
</style>
""",
    unsafe_allow_html=True,
)

PLOT_CFG = {
    "displayModeBar": True,
    "displaylogo": False,
    "scrollZoom": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    "toImageButtonOptions": {"format": "png", "scale": 2},
}


def style_fig(fig: go.Figure, title: str = "", height: int = 420) -> go.Figure:
    """Apply the dark terminal theme. Generous top margin keeps the title,
    the legend and Plotly's modebar from overlapping each other."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=TEXT), x=0, xanchor="left",
                   pad=dict(b=14)),
        height=height,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=MUTED, size=12),
        margin=dict(l=8, r=14, t=78, b=8),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#11161F", bordercolor="rgba(255,255,255,.14)",
                        font=dict(color=TEXT, size=12)),
        legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0, xanchor="left",
                    bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID),
    )
    return fig


def cards(items: list) -> None:
    """items: (label, value, tone in {pos,neg,acc,neu}, footnote)"""
    html = "".join(
        f'<div class="card"><div class="k">{k}</div>'
        f'<div class="v {tone}">{v}</div>'
        + (f'<div class="n">{note}</div>' if note else "")
        + "</div>"
        for k, v, tone, note in items
    )
    st.markdown(f'<div class="cardrow">{html}</div>', unsafe_allow_html=True)


def tone_of(x) -> str:
    if x is None or pd.isna(x):
        return "neu"
    return "pos" if x > 0 else ("neg" if x < 0 else "neu")


def fmt(x, suffix: str = "", dp: int = 2, dash: str = "—") -> str:
    if x is None or pd.isna(x):
        return dash
    return f"{x:,.{dp}f}{suffix}"


# ----------------------------------------------------------------------
# data access
# ----------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_prices(asset: str, db_path: str = DB_PATH) -> pd.DataFrame:
    if not os.path.exists(db_path):
        return pd.DataFrame()
    try:
        with sqlite3.connect(db_path) as conn:
            df = pd.read_sql_query(
                "SELECT date, open, high, low, close, volume FROM prices "
                "WHERE asset = ? ORDER BY date ASC",
                conn, params=(asset,),
            )
    except sqlite3.Error:
        return pd.DataFrame()
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(show_spinner=False)
def load_coverage(db_path: str = DB_PATH) -> pd.DataFrame:
    if not os.path.exists(db_path):
        return pd.DataFrame()
    try:
        with sqlite3.connect(db_path) as conn:
            return pd.read_sql_query(
                "SELECT asset, COUNT(*) AS rows, MIN(date) AS first_date, "
                "MAX(date) AS last_date FROM prices GROUP BY asset",
                conn,
            )
    except sqlite3.Error:
        return pd.DataFrame()


def save_backtest(strategy: str, params: dict, asset: str, result: dict) -> None:
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                "INSERT INTO backtest_results (strategy, params_json, asset, "
                "sharpe, max_drawdown, final_value) VALUES (?, ?, ?, ?, ?, ?)",
                (strategy, json.dumps(params), asset, float(result["sharpe"]),
                 float(result["max_drawdown"]), float(result["final_value"])),
            )
    except sqlite3.Error as exc:
        st.warning(f"Could not save result: {exc}")


def slice_dates(df: pd.DataFrame, start, end) -> pd.DataFrame:
    if df.empty:
        return df
    m = (df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))
    return df.loc[m].reset_index(drop=True)


# ----------------------------------------------------------------------
# shell
# ----------------------------------------------------------------------
st.markdown(
    '<div class="hero"><div class="eyebrow">Quantitative Multi-Asset Terminal</div>'
    "<h1>Gold · Bitcoin · NVIDIA</h1>"
    '<p class="sub">Indicator engine, cross-asset correlation, and a next-day-open '
    "backtester with transaction costs. Everything below is a historical simulation "
    "of one rule over one sample — it is not a forecast, and past performance carries "
    "no guarantee of future returns.</p></div>",
    unsafe_allow_html=True,
)

if not os.path.exists(DB_PATH):
    st.warning(f"No database at `{DB_PATH}`. Run `python ingestion.py`, then reload.")
    st.stop()

coverage = load_coverage()
if coverage.empty:
    st.warning("`data/market.db` exists but `prices` is empty. Run `python ingestion.py`.")
    st.stop()

with st.sidebar:
    st.markdown("### Data coverage")
    st.dataframe(coverage, hide_index=True, use_container_width=True)
    st.markdown(
        '<div class="note">Refresh with <code>python ingestion.py</code>. '
        "Assets keep their native calendars — Bitcoin trades weekends, Gold and "
        "NVIDIA do not.</div>",
        unsafe_allow_html=True,
    )
    st.markdown("### Execution model")
    st.markdown(
        '<div class="note">Signal formed at day <i>t</i> close → filled at day '
        "<i>t+1</i> <b>open</b>, costs charged both sides. No bar trades on its own "
        "or later information.</div>",
        unsafe_allow_html=True,
    )

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["  Asset Explorer  ", "  Correlation  ", "  Backtest  ",
     "  Robustness & Regimes  ", "  Portfolio  ", "  AI Assistant  "]
)

# ======================================================================
# Tab 1 — Asset Explorer
# ======================================================================
with tab1:
    c1, c2, c3 = st.columns([1.1, 2, 1.4])
    asset = c1.selectbox("Asset", ASSETS, key="ex_asset")
    full = load_prices(asset)

    if full.empty:
        st.warning(f"No rows stored for {asset}.")
    else:
        lo, hi = full["date"].min().date(), full["date"].max().date()
        rng = c2.date_input("Window", value=(lo, hi), min_value=lo, max_value=hi,
                            key="ex_dates")
        start, end = rng if isinstance(rng, tuple) and len(rng) == 2 else (lo, hi)
        overlays = c3.multiselect(
            "Overlays", ["SMA 20", "SMA 50", "EMA 20", "EMA 50"],
            default=["SMA 20", "SMA 50", "EMA 20"], key="ex_overlays",
        )

        df = slice_dates(full, start, end)
        if df.empty:
            st.warning("No trading days in the selected range.")
        else:
            ppy = PERIODS_PER_YEAR[asset]
            rets = ind.get_daily_returns(df)
            cum = ind.get_cumulative_returns(df)
            vol = ind.get_volatility(rets, periods_per_year=ppy)
            shrp = ind.get_sharpe_ratio(rets, periods_per_year=ppy)
            mdd = ind.get_max_drawdown(df["close"])
            dd = ind.get_drawdown_series(df["close"])
            rvol = ind.get_rolling_volatility(rets, 30, periods_per_year=ppy)
            roll_window = int(min(252, max(20, len(df) // 4)))
            rroll = ind.get_rolling_returns(df, roll_window)

            cards([
                ("Last close", fmt(df["close"].iloc[-1]), "neu",
                 f"{df['date'].iloc[-1]:%d %b %Y}"),
                ("Cumulative return", fmt(cum.iloc[-1] * 100, "%"),
                 tone_of(cum.iloc[-1]), f"{len(df)} sessions"),
                ("Annualised volatility", fmt(vol * 100, "%"), "acc",
                 f"{ppy} periods/yr"),
                ("Sharpe (buy & hold)", fmt(shrp), tone_of(shrp), "rf = 0%"),
                ("Max drawdown", fmt(mdd * 100, "%"), "neg", "peak to trough"),
            ])

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["date"], y=df["close"], name="Close",
                                     line=dict(color=CYAN, width=1.8),
                                     fill="tozeroy",
                                     fillcolor="rgba(34,211,238,0.06)"))
            spec = {
                "SMA 20": (ind.get_sma(df, 20), BLUE, "dot"),
                "SMA 50": (ind.get_sma(df, 50), VIOLET, "dot"),
                "EMA 20": (ind.get_ema(df, 20), AMBER, "solid"),
                "EMA 50": (ind.get_ema(df, 50), GREEN, "solid"),
            }
            for label in overlays:
                series, colour, dash = spec[label]
                fig.add_trace(go.Scatter(
                    x=df["date"], y=series.to_numpy(), name=label,
                    line=dict(color=colour, width=1.2, dash=dash)))
            st.plotly_chart(
                style_fig(fig, f"{asset} · price with SMA / EMA overlays", 460),
                use_container_width=True, config=PLOT_CFG)

            g1, g2 = st.columns(2)
            ddfig = go.Figure()
            ddfig.add_trace(go.Scatter(
                x=dd.index, y=dd.to_numpy() * 100, name="Drawdown",
                line=dict(color=RED, width=1.3), fill="tozeroy",
                fillcolor="rgba(248,113,113,0.13)"))
            g1.plotly_chart(style_fig(ddfig, "Underwater curve (% below peak)", 330),
                            use_container_width=True, config=PLOT_CFG)

            vfig = go.Figure()
            vfig.add_trace(go.Scatter(
                x=rvol.index, y=rvol.to_numpy() * 100, name="30d annualised vol",
                line=dict(color=AMBER, width=1.3)))
            g2.plotly_chart(
                style_fig(vfig, "Rolling 30-day annualised volatility (%)", 330),
                use_container_width=True, config=PLOT_CFG)

            g3, g4 = st.columns(2)
            rfig = go.Figure()
            rcol = [GREEN if v >= 0 else RED for v in rets.fillna(0).to_numpy()]
            rfig.add_trace(go.Bar(x=rets.index, y=rets.to_numpy() * 100,
                                  name="Daily return", marker_color=rcol,
                                  marker_line_width=0))
            g3.plotly_chart(style_fig(rfig, "Daily returns (%)", 330),
                            use_container_width=True, config=PLOT_CFG)

            pfig = go.Figure()
            pfig.add_trace(go.Scatter(x=cum.index, y=cum.to_numpy() * 100,
                                      name="Cumulative",
                                      line=dict(color=CYAN, width=1.6)))
            pfig.add_trace(go.Scatter(x=rroll.index, y=rroll.to_numpy() * 100,
                                      name=f"Rolling {roll_window}d",
                                      line=dict(color=VIOLET, width=1.3, dash="dot")))
            pfig.add_hline(y=0, line_dash="dot", line_color=MUTED, opacity=0.6)
            g4.plotly_chart(
                style_fig(pfig, "Cumulative vs rolling performance (%)", 330),
                use_container_width=True, config=PLOT_CFG)

# ======================================================================
# Tab 2 — Correlation
# ======================================================================
with tab2:
    frames = {a: load_prices(a) for a in ASSETS}
    frames = {a: d for a, d in frames.items() if not d.empty}

    if len(frames) < 2:
        st.warning("At least two assets with stored data are needed to correlate.")
    else:
        c1, c2, c3 = st.columns([1.2, 1.2, 1.6])
        years = c1.slider("Lookback (years)", 1, 12, 5, key="co_years")
        cutoff = pd.Timestamp.today().normalize() - pd.DateOffset(years=years)
        windowed = {a: d.loc[d["date"] >= cutoff].reset_index(drop=True)
                    for a, d in frames.items()}
        windowed = {a: d for a, d in windowed.items() if len(d) > 2}
        matrix = ind.get_correlation_matrix(windowed)

        names = list(frames)
        p1 = c2.selectbox("Pair A", names, index=0, key="co_a")
        p2 = c3.selectbox("Pair B", names, index=1 if len(names) > 1 else 0, key="co_b")
        window = c1.slider("Rolling window (shared days)", 20, 250, 90, step=10,
                           key="co_window")

        h1, h2 = st.columns([1, 1.25])
        if matrix.empty:
            h1.warning("Not enough overlapping trading days.")
        else:
            heat = go.Figure(go.Heatmap(
                z=matrix.to_numpy(), x=list(matrix.columns), y=list(matrix.index),
                zmin=-1, zmax=1,
                colorscale=[[0, RED], [0.5, "#161B26"], [1, CYAN]],
                text=matrix.round(2).to_numpy(), texttemplate="%{text}",
                textfont=dict(size=14, color=TEXT),
                colorbar=dict(outlinewidth=0, tickfont=dict(color=MUTED, size=10),
                              thickness=10),
                hoverongaps=False))
            h1.plotly_chart(
                style_fig(heat, f"Daily-return correlation · last {years}y", 400),
                use_container_width=True, config=PLOT_CFG)

        if p1 == p2:
            h2.info("Pick two different assets for a rolling correlation.")
        else:
            rolling = ind.get_rolling_correlation(frames[p1], frames[p2], window).dropna()
            if rolling.empty:
                h2.warning("Not enough shared history for that window.")
            else:
                rc = go.Figure()
                rc.add_trace(go.Scatter(x=rolling.index, y=rolling.to_numpy(),
                                        name=f"{p1} vs {p2}",
                                        line=dict(color=CYAN, width=1.5)))
                rc.add_hline(y=0, line_dash="dot", line_color=MUTED, opacity=0.7)
                rc.update_layout(yaxis_range=[-1, 1])
                h2.plotly_chart(
                    style_fig(rc, f"{window}-day rolling correlation · {p1} vs {p2}", 400),
                    use_container_width=True, config=PLOT_CFG)

                cards([
                    ("Mean correlation", fmt(rolling.mean()), "acc",
                     f"{window}-day window"),
                    ("Minimum", fmt(rolling.min()), "neu", "most decoupled"),
                    ("Maximum", fmt(rolling.max()), "acc", "most coupled"),
                    ("Shared sessions", f"{len(rolling):,}", "neu", "inner-joined"),
                ])

        st.markdown(
            '<div class="note">Returns are inner-joined on shared trading days before '
            "correlating. Forward-filling Bitcoin's weekends into Gold would invent flat "
            "days and drag every correlation toward zero. The rolling chart exists because "
            "a single full-sample number hides how unstable these relationships are.</div>",
            unsafe_allow_html=True,
        )

# ======================================================================
# Tab 3 — Backtest
# ======================================================================
PARAM_SPECS = {
    "sma_crossover": [("fast", 2, 100, 20), ("slow", 5, 300, 50)],
    "ema_trend": [("window", 5, 250, 50)],
    "momentum": [("lookback", 5, 250, 60)],
    "mean_reversion": [("window", 5, 120, 20)],
}
LABELS = {
    "sma_crossover": "SMA Crossover", "ema_trend": "EMA Trend",
    "momentum": "Momentum", "mean_reversion": "Mean Reversion",
}

with tab3:
    c1, c2, c3, c4 = st.columns(4)
    bt_asset = c1.selectbox("Asset", ASSETS, key="bt_asset")
    strategy = c2.selectbox("Strategy", list(bt.STRATEGIES), key="bt_strategy",
                            format_func=lambda s: LABELS.get(s, s))
    capital = c3.number_input("Initial capital", 100.0, 10_000_000.0, 10_000.0,
                              step=1000.0, key="bt_capital")
    cost_bps = c4.slider("Cost per fill (bps)", 0, 100, 10, key="bt_cost")
    transaction_cost_pct = cost_bps / 10_000.0

    pcols = st.columns(4)
    params: dict = {}
    for i, (name, lo_v, hi_v, default) in enumerate(PARAM_SPECS[strategy]):
        params[name] = pcols[i % 4].slider(name, lo_v, hi_v, default,
                                           key=f"bt_p_{strategy}_{name}")
    if strategy == "mean_reversion":
        params["std_threshold"] = pcols[1].slider("std_threshold", 0.5, 4.0, 2.0,
                                                  step=0.1, key="bt_p_mr_thresh")

    bt_full = load_prices(bt_asset)
    if bt_full.empty:
        st.warning(f"No rows stored for {bt_asset}.")
    else:
        lo, hi = bt_full["date"].min().date(), bt_full["date"].max().date()
        b_start = pcols[2].date_input("From", value=lo, min_value=lo, max_value=hi, key="bt_start")
        b_end = pcols[3].date_input("To", value=hi, min_value=lo, max_value=hi, key="bt_end")
        run = st.button("Run Backtest", type="primary", key="bt_run")
        bt_df = slice_dates(bt_full, b_start, b_end)

        if run:
            if bt_df.empty:
                st.warning("No trading days in the selected window.")
                st.session_state.pop("bt_result", None)
            else:
                try:
                    st.session_state["bt_result"] = bt.run_backtest(
                        bt_df, strategy, params, initial_capital=float(capital),
                        transaction_cost_pct=transaction_cost_pct,
                        periods_per_year=PERIODS_PER_YEAR[bt_asset])
                    st.session_state["bt_meta"] = (bt_asset, strategy, dict(params))
                except ValueError as exc:
                    st.warning(f"Could not run backtest: {exc}")
                    st.session_state.pop("bt_result", None)

        result = st.session_state.get("bt_result")
        if result is None:
            st.markdown(
                '<div class="note">Set the controls above and press <b>Run Backtest</b>. '
                "Results stay on screen while you explore and only recompute when you "
                "press the button again.</div>", unsafe_allow_html=True)
        elif result.get("warning"):
            st.warning(result["warning"])
        else:
            r_asset, r_strat, r_params = st.session_state.get(
                "bt_meta", (bt_asset, strategy, params))
            r_ppy = PERIODS_PER_YEAR[r_asset]
            eq, bench = result["equity_curve"], result["benchmark_curve"]
            edge = result["total_return_pct"] - result["benchmark_total_return_pct"]

            cards([
                ("Final value", fmt(result["final_value"]), tone_of(edge),
                 f"benchmark {result['benchmark_final_value']:,.0f}"),
                ("Total return", fmt(result["total_return_pct"], "%"),
                 tone_of(result["total_return_pct"]),
                 f"benchmark {result['benchmark_total_return_pct']:,.2f}%"),
                ("Sharpe", fmt(result["sharpe"]), tone_of(result["sharpe"]),
                 f"benchmark {result['benchmark_sharpe']:.2f}"),
                ("Max drawdown", fmt(result["max_drawdown"] * 100, "%"), "neg",
                 f"benchmark {result['benchmark_max_drawdown'] * 100:.2f}%"),
                ("Trades", f"{result['num_trades']}", "acc",
                 f"{result['exposure_pct']:.0f}% time in market"),
            ])

            eqfig = go.Figure()
            eqfig.add_trace(go.Scatter(x=eq.index, y=eq.to_numpy(), name="Strategy",
                                       line=dict(color=CYAN, width=1.9)))
            eqfig.add_trace(go.Scatter(x=bench.index, y=bench.to_numpy(),
                                       name="Buy & hold",
                                       line=dict(color=MUTED, width=1.3, dash="dash")))
            buys = [t for t in result["trades"] if t["action"] == "BUY"]
            sells = [t for t in result["trades"] if t["action"] == "SELL"]
            for group, colour, symbol, nm in (
                (buys, GREEN, "triangle-up", "Buy"),
                (sells, RED, "triangle-down", "Sell"),
            ):
                if group:
                    dts = [t["date"] for t in group]
                    eqfig.add_trace(go.Scatter(
                        x=dts, y=[float(eq.loc[d]) for d in dts], mode="markers",
                        name=nm, marker=dict(color=colour, size=9, symbol=symbol,
                                             line=dict(width=1, color=BG)),
                        hovertemplate=nm + " @ %{x|%d %b %Y}<extra></extra>"))
            st.plotly_chart(
                style_fig(eqfig,
                          f"{r_asset} · {LABELS.get(r_strat, r_strat)} vs buy & hold "
                          "— markers are fills at next-day open", 470),
                use_container_width=True, config=PLOT_CFG)

            d1, d2 = st.columns([1.35, 1])
            ddf = go.Figure()
            ddf.add_trace(go.Scatter(
                x=eq.index, y=ind.get_drawdown_series(eq).to_numpy() * 100,
                name="Strategy", line=dict(color=CYAN, width=1.3),
                fill="tozeroy", fillcolor="rgba(34,211,238,0.10)"))
            ddf.add_trace(go.Scatter(
                x=bench.index, y=ind.get_drawdown_series(bench).to_numpy() * 100,
                name="Buy & hold", line=dict(color=RED, width=1.1, dash="dash")))
            d1.plotly_chart(style_fig(ddf, "Drawdown comparison (%)", 340),
                            use_container_width=True, config=PLOT_CFG)

            comp = pd.DataFrame({
                "Metric": ["Sharpe", "Total return %", "Volatility %",
                           "Max drawdown %", "Final value", "Trades"],
                "Strategy": [
                    round(result["sharpe"], 2),
                    round(result["total_return_pct"], 2),
                    round(ind.get_volatility(eq.pct_change(),
                                             periods_per_year=r_ppy) * 100, 2),
                    round(result["max_drawdown"] * 100, 2),
                    round(result["final_value"], 2), result["num_trades"]],
                "Buy & hold": [
                    round(result["benchmark_sharpe"], 2),
                    round(result["benchmark_total_return_pct"], 2),
                    round(ind.get_volatility(bench.pct_change(),
                                             periods_per_year=r_ppy) * 100, 2),
                    round(result["benchmark_max_drawdown"] * 100, 2),
                    round(result["benchmark_final_value"], 2), 1],
            })
            d2.markdown("### Strategy vs benchmark")
            d2.dataframe(comp, hide_index=True, use_container_width=True)
            if result["sharpe"] < result["benchmark_sharpe"]:
                d2.markdown(
                    '<div class="note">Lower risk-adjusted return than simply holding '
                    "the asset over this window.</div>", unsafe_allow_html=True)

            st.markdown(f"### Trade log · {result['num_trades']} fills")
            if result["trades"]:
                trades = pd.DataFrame(result["trades"])
                trades["date"] = pd.to_datetime(trades["date"]).dt.date
                trades["value"] = (trades["price"] * trades["shares"]).round(2)
                trades["price"] = trades["price"].round(2)
                trades["shares"] = trades["shares"].round(6)
                st.dataframe(trades, hide_index=True, use_container_width=True,
                             height=300)
            else:
                st.info("The strategy never entered a position in this window.")

            if st.button("Save run to backtest_results", key="bt_save"):
                save_backtest(r_strat, r_params, r_asset, result)
                st.success("Saved to data/market.db.")

# ======================================================================
# Tab 4 — Robustness & Regimes
# ======================================================================
PRESET_REGIMES = {
    "2020 COVID crash (high vol)": ("2020-02-01", "2020-04-30"),
    "2021 Bull market": ("2021-01-01", "2021-12-31"),
    "2022 Bear market": ("2022-01-01", "2022-12-31"),
    "2023 Bull market": ("2023-01-01", "2023-12-31"),
    "2024-25 Recent": ("2024-01-01", "2025-12-31"),
}
GRIDS = {
    "sma_crossover": {"fast": [10, 20, 50], "slow": [50, 100, 200]},
    "ema_trend": {"window": [20, 50, 100, 200]},
    "momentum": {"lookback": [20, 60, 120, 250]},
    "mean_reversion": {"window": [10, 20, 50], "std_threshold": [1.0, 1.5, 2.0, 2.5]},
}

with tab4:
    c1, c2, c3 = st.columns(3)
    r_asset = c1.selectbox("Asset", ASSETS, key="rb_asset")
    r_strategy = c2.selectbox("Strategy", list(bt.STRATEGIES), key="rb_strategy",
                              format_func=lambda s: LABELS.get(s, s))
    r_cost = c3.slider("Cost per fill (bps)", 0, 100, 10, key="rb_cost") / 10_000.0
    r_df = load_prices(r_asset)
    ppy = PERIODS_PER_YEAR[r_asset]

    if r_df.empty:
        st.warning(f"No rows stored for {r_asset}.")
    else:
        st.markdown("### Parameter robustness")
        st.markdown(
            '<div class="note">Read the <b>spread</b>, not the winner. A rule that only '
            "works at one setting is fitted to noise; a broad plateau of similar Sharpes "
            "is the weaker but more honest signal.</div>", unsafe_allow_html=True)

        if st.button("Run robustness test", key="rb_run"):
            with st.spinner("Sweeping the parameter grid..."):
                st.session_state["rb_table"] = bt.run_robustness_test(
                    r_df, r_strategy, GRIDS[r_strategy],
                    transaction_cost_pct=r_cost, periods_per_year=ppy)
                st.session_state["rb_grid_keys"] = list(GRIDS[r_strategy])

        table = st.session_state.get("rb_table")
        if table is not None and not table.empty:
            spread = table["sharpe"].max() - table["sharpe"].min()
            cards([
                ("Combinations", f"{len(table)}", "neu", "valid runs"),
                ("Best Sharpe", fmt(table["sharpe"].max()), "pos",
                 "not an expectation"),
                ("Worst Sharpe", fmt(table["sharpe"].min()), "neg",
                 "same rule, other params"),
                ("Sharpe spread", fmt(spread), "acc" if spread < 0.5 else "neg",
                 "wide spread means fragile"),
            ])
            st.dataframe(
                table.round({"sharpe": 2, "total_return_pct": 2,
                             "max_drawdown_pct": 2, "final_value": 2}),
                hide_index=True, use_container_width=True, height=330)

            keys = st.session_state.get("rb_grid_keys", [])
            if len(keys) == 2 and all(k in table.columns for k in keys):
                pivot = table.pivot_table(index=keys[0], columns=keys[1],
                                          values="sharpe")
                surf = go.Figure(go.Heatmap(
                    z=pivot.to_numpy(),
                    x=[str(c) for c in pivot.columns],
                    y=[str(i) for i in pivot.index],
                    colorscale=[[0, RED], [0.5, "#161B26"], [1, CYAN]],
                    text=pivot.round(2).to_numpy(), texttemplate="%{text}",
                    textfont=dict(size=13, color=TEXT),
                    colorbar=dict(outlinewidth=0, thickness=10,
                                  tickfont=dict(color=MUTED, size=10))))
                surf.update_layout(xaxis_title=keys[1], yaxis_title=keys[0])
                st.plotly_chart(
                    style_fig(surf, "Sharpe across the parameter grid", 380),
                    use_container_width=True, config=PLOT_CFG)
        elif table is not None:
            st.warning("No valid parameter combinations produced a result.")

        st.divider()
        st.markdown("### Market regime analysis")
        chosen = st.multiselect("Regimes", list(PRESET_REGIMES),
                                default=list(PRESET_REGIMES)[:4], key="rb_regimes")
        rp = st.columns(4)
        reg_params: dict = {}
        for i, (name, lo_v, hi_v, default) in enumerate(PARAM_SPECS[r_strategy]):
            reg_params[name] = rp[i % 4].slider(name, lo_v, hi_v, default,
                                                key=f"rb_p_{r_strategy}_{name}")
        if r_strategy == "mean_reversion":
            reg_params["std_threshold"] = rp[1].slider("std_threshold", 0.5, 4.0, 2.0,
                                                       step=0.1, key="rb_p_mr_thresh")

        if rp[3].button("Run regime analysis", key="rb_regime_run"):
            if not chosen:
                st.warning("Select at least one regime.")
            else:
                try:
                    st.session_state["rb_reg_out"] = bt.run_regime_analysis(
                        r_df, r_strategy, reg_params,
                        {k: PRESET_REGIMES[k] for k in chosen},
                        transaction_cost_pct=r_cost, periods_per_year=ppy)
                except ValueError as exc:
                    st.warning(f"Could not run regime analysis: {exc}")

        reg = st.session_state.get("rb_reg_out")
        if reg is not None and not reg.empty:
            st.dataframe(reg.round(2), hide_index=True, use_container_width=True)
            plot = reg.dropna(subset=["strategy_return_pct"])
            if not plot.empty:
                gfig = go.Figure()
                gfig.add_trace(go.Bar(x=plot["regime"], y=plot["strategy_return_pct"],
                                      name="Strategy", marker_color=CYAN,
                                      marker_line_width=0))
                gfig.add_trace(go.Bar(x=plot["regime"], y=plot["benchmark_return_pct"],
                                      name="Buy & hold", marker_color=MUTED,
                                      marker_line_width=0))
                gfig.update_layout(barmode="group", hovermode="closest")
                st.plotly_chart(style_fig(gfig, "Return by market regime (%)", 400),
                                use_container_width=True, config=PLOT_CFG)
            st.markdown(
                '<div class="note">Each regime restarts from the initial capital and '
                "restarts the indicator warm-up, so a short window can spend most of its "
                "length flat — check <code>exposure_pct</code> before reading much into "
                "any single regime.</div>", unsafe_allow_html=True)
        elif reg is not None:
            st.warning("No regime produced a result.")

# ======================================================================
# Tab 5 — Portfolio Optimization
# ======================================================================
with tab5:
    st.markdown(
        '<div class="note">Modern Portfolio Theory on historical returns. The '
        "weights below are optimal <b>for the past sample shown</b> — correlations "
        "and volatilities drift over time, so this is backward-looking analysis, "
        "not an investment recommendation for the future.</div>",
        unsafe_allow_html=True,
    )

    p_frames = {a: load_prices(a) for a in ASSETS}
    p_frames = {a: d for a, d in p_frames.items() if not d.empty}

    if len(p_frames) < 2:
        st.warning("At least two assets with stored data are needed to optimize.")
    else:
        pc1, pc2, pc3 = st.columns(3)
        chosen_assets = pc1.multiselect("Assets", list(p_frames), default=list(p_frames),
                                        key="pf_assets")
        p_years = pc2.slider("Lookback (years)", 1, 12, 5, key="pf_years")
        rf = pc3.number_input("Risk-free rate (annual %)", 0.0, 15.0, 0.0, step=0.25,
                              key="pf_rf") / 100.0
        allow_short = st.checkbox("Allow short positions (negative weights)",
                                  value=False, key="pf_short")

        if len(chosen_assets) < 2:
            st.warning("Select at least two assets.")
        else:
            cutoff = pd.Timestamp.today().normalize() - pd.DateOffset(years=p_years)
            windowed = {
                a: p_frames[a].loc[p_frames[a]["date"] >= cutoff].reset_index(drop=True)
                for a in chosen_assets
            }

            if st.button("Run optimization", key="pf_run", type="primary"):
                st.session_state["pf_result"] = pf.optimize_portfolio(
                    windowed, risk_free_rate=rf, allow_short=allow_short)

            result = st.session_state.get("pf_result")
            if result is None:
                st.markdown(
                    '<div class="note">Choose assets and press <b>Run optimization</b>.'
                    "</div>", unsafe_allow_html=True)
            elif result.get("warning"):
                st.warning(result["warning"])
            else:
                st.caption(
                    f"{result['n_observations']} shared trading days · "
                    f"{result['start_date']} to {result['end_date']}"
                )

                ms, mv, ew = result["max_sharpe"], result["min_volatility"], result["equal_weight"]
                cards([
                    ("Max Sharpe — return", fmt(ms["return"] * 100, "%"), "acc",
                     f"Sharpe {ms['sharpe']:.2f}"),
                    ("Max Sharpe — volatility", fmt(ms["volatility"] * 100, "%"), "neu",
                     "annualised"),
                    ("Min volatility — return", fmt(mv["return"] * 100, "%"), "neu",
                     f"vol {mv['volatility']*100:.2f}%"),
                    ("Equal weight (1/N) — Sharpe", fmt(ew["sharpe"]), tone_of(ew["sharpe"]),
                     "naive benchmark"),
                ])

                w1, w2 = st.columns(2)
                wfig = go.Figure()
                assets_list = result["assets"]
                for label, port, colour in [
                    ("Max Sharpe", ms, CYAN), ("Min Volatility", mv, VIOLET),
                    ("Equal Weight", ew, MUTED),
                ]:
                    wfig.add_trace(go.Bar(
                        x=assets_list,
                        y=[port["weights"][a] * 100 for a in assets_list],
                        name=label))
                wfig.update_layout(barmode="group", yaxis_title="Weight (%)")
                w1.plotly_chart(style_fig(wfig, "Portfolio weights by method", 380),
                                use_container_width=True, config=PLOT_CFG)

                frontier = result["efficient_frontier"]
                if not frontier.empty:
                    ffig = go.Figure()
                    ffig.add_trace(go.Scatter(
                        x=frontier["volatility"] * 100, y=frontier["return"] * 100,
                        mode="lines", name="Efficient frontier",
                        line=dict(color=CYAN, width=2)))
                    ffig.add_trace(go.Scatter(
                        x=[ms["volatility"] * 100], y=[ms["return"] * 100],
                        mode="markers", name="Max Sharpe",
                        marker=dict(color=GREEN, size=12, symbol="star")))
                    ffig.add_trace(go.Scatter(
                        x=[mv["volatility"] * 100], y=[mv["return"] * 100],
                        mode="markers", name="Min Volatility",
                        marker=dict(color=VIOLET, size=11, symbol="diamond")))
                    ffig.add_trace(go.Scatter(
                        x=[ew["volatility"] * 100], y=[ew["return"] * 100],
                        mode="markers", name="Equal Weight",
                        marker=dict(color=MUTED, size=10, symbol="circle")))
                    ffig.update_layout(xaxis_title="Volatility (%)",
                                       yaxis_title="Expected return (%)")
                    w2.plotly_chart(style_fig(ffig, "Efficient frontier", 380),
                                    use_container_width=True, config=PLOT_CFG)

                st.markdown("### Weights detail")
                detail = pd.DataFrame({
                    "Asset": assets_list,
                    "Max Sharpe %": [round(ms["weights"][a] * 100, 1) for a in assets_list],
                    "Min Volatility %": [round(mv["weights"][a] * 100, 1) for a in assets_list],
                    "Equal Weight %": [round(ew["weights"][a] * 100, 1) for a in assets_list],
                })
                st.dataframe(detail, hide_index=True, use_container_width=True)

                st.markdown("### Correlation used in this optimization")
                corr = result["correlation_matrix"]
                cfig = go.Figure(go.Heatmap(
                    z=corr.to_numpy(), x=list(corr.columns), y=list(corr.index),
                    zmin=-1, zmax=1, colorscale=[[0, RED], [0.5, "#161B26"], [1, CYAN]],
                    text=corr.round(2).to_numpy(), texttemplate="%{text}",
                    textfont=dict(size=13, color=TEXT),
                    colorbar=dict(outlinewidth=0, thickness=10,
                                  tickfont=dict(color=MUTED, size=10))))
                st.plotly_chart(style_fig(cfig, "Return correlation (same sample)", 360),
                                use_container_width=True, config=PLOT_CFG)

# ======================================================================
# Tab 6 — AI Research Assistant
# ======================================================================
with tab6:
    st.markdown(
        '<div class="note">This assistant explains numbers the platform has already '
        "computed. It never predicts prices, never generates trading signals, and "
        "never recommends an allocation — it only narrates results already on screen. "
        "Powered by Featherless.</div>",
        unsafe_allow_html=True,
    )

    if not ai.is_configured():
        st.warning(
            "No Featherless API key found. Set the `FEATHERLESS_API_KEY` environment "
            "variable, then restart the app to use this tab.\n\n"
            "Windows (PowerShell): `$env:FEATHERLESS_API_KEY=\"your-key-here\"`\n\n"
            "Then re-run `streamlit run dashboard.py` from the same terminal."
        )
    else:
        st.success("Featherless API key detected.")

        context_choice = st.selectbox(
            "What should the assistant look at?",
            ["Last backtest result (Tab 3)", "Last robustness sweep (Tab 4)",
             "Last regime analysis (Tab 4)", "Last portfolio optimization (Tab 5)"],
            key="ai_context",
        )

        default_q = ""
        bt_result = st.session_state.get("bt_result")
        rb_table = st.session_state.get("rb_table")
        rb_reg = st.session_state.get("rb_reg_out")
        pf_result = st.session_state.get("pf_result")

        available = {
            "Last backtest result (Tab 3)": bt_result is not None and not bt_result.get("warning"),
            "Last robustness sweep (Tab 4)": rb_table is not None and not rb_table.empty,
            "Last regime analysis (Tab 4)": rb_reg is not None and not rb_reg.empty,
            "Last portfolio optimization (Tab 5)": pf_result is not None and not pf_result.get("warning"),
        }

        if not available[context_choice]:
            st.info(
                f"No result to explain yet for '{context_choice}'. Run it in its tab "
                "first, then come back here."
            )
        else:
            question = st.text_area(
                "Ask a question (optional — leave blank for a general summary)",
                value=default_q, key="ai_question", height=90,
                placeholder="e.g. Why did the strategy underperform buy-and-hold?",
            )
            if st.button("Ask the assistant", key="ai_ask", type="primary"):
                with st.spinner("Thinking..."):
                    q = question.strip() or None
                    if context_choice == "Last backtest result (Tab 3)":
                        r_asset, r_strat, r_params = st.session_state.get(
                            "bt_meta", ("", "", {}))
                        answer = ai.explain_backtest(bt_result, r_asset, r_strat,
                                                     r_params, q)
                    elif context_choice == "Last robustness sweep (Tab 4)":
                        answer = ai.explain_robustness(
                            rb_table.to_dict(orient="records"),
                            st.session_state.get("rb_strategy", ""),
                            st.session_state.get("rb_asset", ""), q)
                    elif context_choice == "Last regime analysis (Tab 4)":
                        answer = ai.explain_regimes(
                            rb_reg.to_dict(orient="records"),
                            st.session_state.get("rb_strategy", ""),
                            st.session_state.get("rb_asset", ""), q)
                    else:
                        answer = ai.explain_portfolio(pf_result, q)
                    st.session_state["ai_answer"] = answer

            answer = st.session_state.get("ai_answer")
            if answer:
                st.markdown("### Response")
                st.markdown(f'<div class="note">{answer}</div>', unsafe_allow_html=True)
