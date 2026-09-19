"""
AI Quantitative Research Assistant — narrates results, never predicts them.

Uses Featherless (an OpenAI-compatible LLM inference API) purely as a
text-generation layer over numbers that indicators.py, backtester.py and
portfolio.py have already computed. This module:

  * NEVER asks the model to produce a price forecast, a trading signal, or
    a recommended allocation. Every prompt hands the model a finished
    result (metrics, a trade table, a robustness table) and asks it to
    explain or summarise that result in plain English.
  * ALWAYS includes the platform's standard disclaimer in the system
    prompt, so the model is primed to describe historical behaviour rather
    than promise future performance.
  * Fails soft: with no API key, no network, or a bad response, callers get
    a clear string explaining why, never a stack trace and never a silent
    fabrication.

Reads the key from the FEATHERLESS_API_KEY environment variable. Never
hardcode a key in this file or in dashboard.py.
"""

from __future__ import annotations

import json
import os

import numpy as np
import requests

FEATHERLESS_URL = "https://api.featherless.ai/v1/chat/completions"
DEFAULT_MODEL = "microsoft/phi-4"
REQUEST_TIMEOUT = 30

SYSTEM_PROMPT = """You are a quantitative research assistant embedded in a \
backtesting dashboard for Gold, Bitcoin and NVIDIA.

Rules you must follow in every reply:
1. You explain numbers that have already been computed by the platform's \
own backtesting and statistics engine. You never invent numbers, and you \
never calculate anything yourself — treat every figure given to you as \
ground truth from the platform.
2. You NEVER predict future prices, future returns, or where a market is \
headed. You NEVER recommend buying, selling, or holding anything.
3. You NEVER suggest an asset allocation or portfolio weights beyond \
describing the ones already computed by the optimizer.
4. Everything you describe is a historical simulation on one past sample. \
If the person's question implies certainty about the future, gently \
correct that framing before answering.
5. Be concise. Use plain English a non-quant could follow. Avoid jargon \
without explaining it once.
6. If the data given to you doesn't support an answer, say so directly \
instead of guessing.
"""


def is_configured() -> bool:
    return bool(os.environ.get("FEATHERLESS_API_KEY"))


def _call_featherless(messages: list[dict], model: str = DEFAULT_MODEL,
                      max_tokens: int = 500) -> str:
    """Low-level call. Raises RuntimeError with a human-readable message
    on any failure — never a raw exception a dashboard would need to catch
    by type."""
    api_key = os.environ.get("FEATHERLESS_API_KEY")
    if not api_key:
        raise RuntimeError(
            "No Featherless API key found. Set the FEATHERLESS_API_KEY "
            "environment variable and restart the app."
        )

    try:
        resp = requests.post(
            FEATHERLESS_URL,
            headers={"Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"},
            json={"model": model, "messages": messages,
                  "max_tokens": max_tokens, "temperature": 0.15,
                  "repetition_penalty": 1.15, "top_p": 0.9},
            timeout=REQUEST_TIMEOUT,
        )
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            "Could not reach Featherless (network blocked or offline). "
            f"Details: {exc}"
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise RuntimeError(
            f"Featherless request timed out after {REQUEST_TIMEOUT}s."
        ) from exc

    if resp.status_code == 401:
        raise RuntimeError("Featherless rejected the API key (401 Unauthorized).")
    if resp.status_code == 429:
        raise RuntimeError("Featherless rate limit hit (429). Try again shortly.")
    if resp.status_code >= 400:
        raise RuntimeError(f"Featherless error {resp.status_code}: {resp.text[:300]}")

    try:
        data = resp.json()
        text = data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, json.JSONDecodeError) as exc:
        # Status 200 but no 'choices' usually means Featherless embedded an
        # error inside a 200 response (e.g. unknown/unavailable model). Show
        # the raw body so the real reason isn't hidden behind a generic
        # "unexpected shape" message.
        raise RuntimeError(
            f"Featherless returned status {resp.status_code} but no usable "
            f"'choices' field (model='{model}'). Raw response: {resp.text[:400]}"
        ) from exc

    if _looks_degenerate(text):
        raise RuntimeError(
            f"Model '{model}' returned degenerate/looping output instead of a "
            "real answer. This is a hosting/model issue, not a data issue — "
            "try a different model (see the model list at /v1/models)."
        )
    return text


def _looks_degenerate(text: str) -> bool:
    """Heuristic check for the repeating-fragment / mixed-script garbage a
    broken model or mismatched chat template can produce (observed in
    practice: short tokens strung together, each followed by a stray '?',
    with words from more than one script mixed in). A plain distinct-word
    ratio is too easily fooled — the garbage uses many different tokens,
    just in a nonsense pattern — so this checks two more specific signals
    instead: how much of the text is punctuation noise, and whether more
    than one writing script appears in a short answer."""
    if not text or len(text) < 5:
        return True
    words = text.split()
    if len(words) < 8:
        return False  # too short to judge reliably, let it through

    stray_marks = sum(1 for w in words if w.strip("?!.,\"'") == "")
    trailing_q = sum(1 for w in words if w.endswith("?") and len(w) < 12)
    if (stray_marks + trailing_q) / len(words) > 0.3:
        return True

    has_latin = any("a" <= c.lower() <= "z" for c in text)
    has_cyrillic = any("\u0400" <= c <= "\u04FF" for c in text)
    if has_latin and has_cyrillic:
        return True

    distinct_ratio = len(set(words)) / len(words)
    return distinct_ratio < 0.25


def _safe_round(value, dp=3):
    """json.dumps chokes on numpy scalars/NaN; make every payload JSON-safe."""
    import math

    try:
        f = float(value)
        return None if not math.isfinite(f) else round(f, dp)
    except (TypeError, ValueError):
        return value


def _clean_payload(obj):
    """Recursively convert numpy scalars, NaN and Infinity into plain,
    JSON-safe Python types. json.dumps raises on numpy int64/float64 and
    on non-finite floats by default, and a bad backtest metric (e.g. an
    infinite Sharpe from zero volatility) must never crash the assistant."""
    if isinstance(obj, dict):
        return {k: _clean_payload(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean_payload(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return _safe_round(obj)
    return obj


# ----------------------------------------------------------------------
# public entry points — one per kind of thing the dashboard might explain
# ----------------------------------------------------------------------
def explain_backtest(result: dict, asset: str, strategy: str, params: dict,
                     question: str | None = None) -> str:
    """Explain one backtest result. `result` is the dict run_backtest()
    returns (equity_curve/benchmark_curve are dropped before sending —
    only the summary metrics and trade count go to the model)."""
    summary = {
        "asset": asset, "strategy": strategy, "params": params,
        "sharpe": result.get("sharpe"),
        "benchmark_sharpe": result.get("benchmark_sharpe"),
        "total_return_pct": result.get("total_return_pct"),
        "benchmark_total_return_pct": result.get("benchmark_total_return_pct"),
        "max_drawdown_pct": (result.get("max_drawdown") or 0) * 100,
        "benchmark_max_drawdown_pct": (result.get("benchmark_max_drawdown") or 0) * 100,
        "num_trades": result.get("num_trades"),
        "exposure_pct": result.get("exposure_pct"),
    }
    summary = _clean_payload(summary)
    user_q = question or (
        "Explain this backtest result in plain English: how did the strategy "
        "do versus buy-and-hold, and what stands out?"
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"Here is one backtest result, already computed by the platform "
            f"(all figures are historical, after transaction costs):\n"
            f"{json.dumps(summary, indent=2)}\n\n{user_q}"
        )},
    ]
    try:
        return _call_featherless(messages)
    except RuntimeError as exc:
        return f"AI assistant unavailable: {exc}"


def explain_robustness(table_records: list[dict], strategy: str, asset: str,
                       question: str | None = None) -> str:
    """Explain a robustness sweep — emphasise spread over the winning row,
    matching the same framing the dashboard itself uses."""
    trimmed = table_records[:30]  # keep the prompt small and cheap
    user_q = question or (
        "Summarise what this parameter sweep shows. Focus on how stable or "
        "fragile the strategy looks across settings, not just the best row."
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"Robustness sweep for {strategy} on {asset} "
            f"({len(table_records)} parameter combinations, showing up to 30):\n"
            f"{json.dumps(_clean_payload(trimmed), indent=2)}\n\n{user_q}"
        )},
    ]
    try:
        return _call_featherless(messages)
    except RuntimeError as exc:
        return f"AI assistant unavailable: {exc}"


def explain_regimes(regime_records: list[dict], strategy: str, asset: str,
                    question: str | None = None) -> str:
    """Explain a regime comparison table."""
    user_q = question or (
        "Summarise how this strategy behaved across these market regimes "
        "compared to buy-and-hold. Note any regime where exposure was low."
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"Regime comparison for {strategy} on {asset}:\n"
            f"{json.dumps(_clean_payload(regime_records), indent=2)}\n\n{user_q}"
        )},
    ]
    try:
        return _call_featherless(messages)
    except RuntimeError as exc:
        return f"AI assistant unavailable: {exc}"


def explain_portfolio(portfolio_result: dict, question: str | None = None) -> str:
    """Explain a portfolio-optimization result (max-Sharpe / min-vol /
    equal-weight comparison). The covariance matrix and frontier are
    summarised rather than sent in full, to keep the prompt small."""
    if portfolio_result.get("warning"):
        return portfolio_result["warning"]

    summary = {
        "assets": portfolio_result.get("assets"),
        "period": f"{portfolio_result.get('start_date')} to "
                  f"{portfolio_result.get('end_date')}",
        "max_sharpe_portfolio": {
            "weights": portfolio_result["max_sharpe"]["weights"],
            "expected_return_pct": portfolio_result["max_sharpe"]["return"] * 100,
            "volatility_pct": portfolio_result["max_sharpe"]["volatility"] * 100,
            "sharpe": portfolio_result["max_sharpe"]["sharpe"],
        },
        "min_volatility_portfolio": {
            "weights": portfolio_result["min_volatility"]["weights"],
            "expected_return_pct": portfolio_result["min_volatility"]["return"] * 100,
            "volatility_pct": portfolio_result["min_volatility"]["volatility"] * 100,
        },
        "equal_weight_portfolio": {
            "weights": portfolio_result["equal_weight"]["weights"],
            "expected_return_pct": portfolio_result["equal_weight"]["return"] * 100,
            "volatility_pct": portfolio_result["equal_weight"]["volatility"] * 100,
        },
    }
    user_q = question or (
        "Explain what these three portfolios show and how they trade off "
        "return against risk. Remind me these weights are backward-looking."
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"Portfolio optimization result (Modern Portfolio Theory, "
            f"historical data only):\n{json.dumps(_clean_payload(summary), indent=2)}"
            f"\n\n{user_q}"
        )},
    ]
    try:
        return _call_featherless(messages)
    except RuntimeError as exc:
        return f"AI assistant unavailable: {exc}"


def free_form_question(context: dict, question: str) -> str:
    """Answer an arbitrary question about whatever context dict the
    dashboard currently has on screen (a mix of metrics/tables). Still
    bound by the same system prompt — no predictions, no advice."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"Context currently on screen:\n{json.dumps(_clean_payload(context), indent=2)}"
            f"\n\nQuestion: {question}"
        )},
    ]
    try:
        return _call_featherless(messages)
    except RuntimeError as exc:
        return f"AI assistant unavailable: {exc}"
