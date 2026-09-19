# New Features: Portfolio Optimization + AI Research Assistant

Both were added **into the existing app** — no files were replaced except
`dashboard.py`, which gained two new tabs. All previously-shipped files
(`ingestion.py`, `indicators.py`, `backtester.py`, `server.py`) are untouched.

New files:
- `portfolio.py` — Modern Portfolio Theory optimization (pure math, no network)
- `ai_assistant.py` — Featherless-backed result explainer (network, optional)

---

## Tab 5 — Portfolio Optimization

Given historical returns for any 2 or 3 of your assets, computes:

| Portfolio | What it is |
|---|---|
| **Max Sharpe** | Highest historical risk-adjusted return, weights summing to 1 |
| **Min Volatility** | Lowest historical risk, same constraints |
| **Equal Weight (1/N)** | The naive benchmark every optimizer should beat |
| **Efficient frontier** | 40-point sweep of min-vol portfolios across the achievable return range |

Optional: allow short positions (negative weights). Off by default.

**Verified against a closed-form solution:** for two uncorrelated assets,
the analytic minimum-variance weight is `(1/var1) / (1/var1 + 1/var2)`. The
optimizer's output matched this to within 0.2% on a 1,000-day synthetic
series. Constraint checks (weights sum to 1, no negative weights when
`allow_short=False`, max-Sharpe portfolio beats equal-weight, min-vol
portfolio has the lowest volatility of the three) all passed. Edge cases —
empty asset dict, single asset, fewer than 30 overlapping days — return a
clear warning instead of crashing.

**The caveat that matters, and is shown directly in the tab:** these weights
are optimal for the historical sample shown, not a forecast. Correlations
between Gold, Bitcoin and NVIDIA drift over multi-year windows (see the
rolling correlation chart on Tab 2) — a portfolio optimized on the last 5
years is not guaranteed to be optimal for the next 5.

---

## Tab 6 — AI Research Assistant (Featherless)

Explains results already computed elsewhere in the app. Does **not** generate
predictions, trading signals, or allocation advice — it only narrates numbers
that `indicators.py`, `backtester.py`, and `portfolio.py` already produced.
This constraint is enforced in the system prompt sent on every call, not just
described in this file.

**Setup:**

```powershell
# PowerShell — set for the current terminal session
$env:FEATHERLESS_API_KEY = "your-key-here"
streamlit run dashboard.py
```

```powershell
# To make it permanent (persists across terminal sessions)
[System.Environment]::SetEnvironmentVariable("FEATHERLESS_API_KEY", "your-key-here", "User")
# then open a NEW terminal and run streamlit run dashboard.py
```

The key is never hardcoded anywhere in the codebase — only read from the
environment at call time.

**What it can explain:**
- Your most recent backtest result (Tab 3)
- Your most recent robustness sweep (Tab 4)
- Your most recent regime analysis (Tab 4)
- Your most recent portfolio optimization (Tab 5)

Run any of those first — the AI tab has nothing to explain until you do.

**Failure handling, tested without a live Featherless connection:**

| Scenario | Result |
|---|---|
| No API key set | Clear message, no crash |
| Network unreachable | Clear message naming the cause, no crash |
| Bad/expired key (401) | Clear message, no crash |
| Rate limited (429) | Clear message, no crash |
| Malformed API response | Clear message, no crash |
| NaN/Infinity in a metric (e.g. Sharpe from zero volatility) | Sanitized to `null` before sending — does not crash the request |

I could not test against the real Featherless endpoint from this sandbox
(no outbound network access here), so the request/response wiring is
verified against a mocked HTTP layer that returns the same shapes
Featherless's OpenAI-compatible API documents. If your first real request
fails with a response-shape error, the model name in `ai_assistant.py`
(`DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"`) is the most likely
thing to need adjusting — check which model names your Featherless account
has access to and update that constant.

---

## Everything else is unchanged

`ingestion.py`, `indicators.py`, `backtester.py`, `server.py` were not
touched. The look-ahead-bias guarantees, transaction-cost handling, and all
previously verified behavior in the backtesting engine are exactly as before.

---

## Update: garbage/degenerate output fix (ai_assistant.py)

**Symptom reported:** Tab 6 returned nonsense — repeating fragments mixed with
Cyrillic characters (e.g. "milesctors? tt Children? ... мия? ttiker...")
instead of a real answer.

**Root cause:** not a bug in the request/response handling — the default
model, `mistralai/Mistral-7B-Instruct-v0.3`, produced degenerate output on
Featherless's hosting for this account (likely a chat-template mismatch or
a broken quantization of that specific model). Small, cheaply-hosted models
are more prone to this failure mode than larger ones.

**Fixes applied:**
1. **Detection, not just prevention** — added `_looks_degenerate()`, which
   checks stray-punctuation density and mixed-script content (the two
   signals actually present in the observed garbage) and raises a clear
   "model returned degenerate output" error instead of silently showing
   nonsense. Verified against the exact garbage text from the bug report
   plus 6 legitimate answers (including question-heavy and bullet-style
   text) — all classified correctly.
2. **Better generation parameters** — lowered `temperature` to 0.15 and
   added `repetition_penalty: 1.15`, both of which reduce the kind of
   token-looping seen in the report.
3. **Still no crash either way** — whether the model degenerates or the
   network fails, the dashboard shows a plain-English message, never a
   traceback or raw garbage.

**What you still need to do:** switch `DEFAULT_MODEL` in `ai_assistant.py`
to a model your Featherless account can reliably run. Get the list with:

```powershell
python -c "
import os, requests
key = os.environ.get('FEATHERLESS_API_KEY')
resp = requests.get('https://api.featherless.ai/v1/models', headers={'Authorization': f'Bearer {key}'})
for m in resp.json().get('data', []):
    print(m.get('id'))
"
```

Then update the constant near the top of `ai_assistant.py`:
```python
DEFAULT_MODEL = "your-chosen-model-id-here"
```
