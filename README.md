# trade_trooper

An agentic trading-intelligence pipeline built on **Alpaca's data APIs**
(simulated/paper trading environment). Agents collect data in parallel, then
sequentially predict, quantify risk, and decide — all deterministic math stays
in Python; local LLM reasoning only orchestrates and interprets.

---

## Architecture (4-phase cycle, ~10s per cycle)

```
PHASE 1 (Parallel data collection - 4 seconds):
  ├─ News Collection Agent
  │   ├─ Calls: fetch_news()
  │   ├─ Calls: sentiment_analysis()
  │   └─ Returns: news_signals, sentiment_score
  ├─ Market Data Agent
  │   ├─ Calls: get_current_price()
  │   ├─ Calls: get_volatility()
  │   └─ Returns: market_data, current_iv
  └─ Historical Data Agent
      ├─ Calls: get_price_history()
      ├─ Calls: calculate_returns()
      └─ Returns: historical_trends, volatility_history

PHASE 2 (Sequential prediction - 3 seconds):
  └─ Prediction Agent (input: all Phase 1 results)
      ├─ Calls: calculate_technical_indicators()
      ├─ Calls: forecast_volatility()
      ├─ Calls: estimate_price_move()
      └─ Returns: price_forecast, iv_forecast, confidence

PHASE 3 (Sequential risk - 2 seconds):
  └─ Risk Agent (input: Phase 2 predictions)
      ├─ Calls: calculate_greeks()          (option chain greeks + IV, BS fallback)
      ├─ Calls: calculate_position_size()   (equity + option, risk-scaled)
      ├─ Calls: calculate_max_loss()        (stop + gap-inflated + VaR/CVaR tail)
      ├─ Calls: calculate_risk_score()      (composite 0-100 → risk level)
      └─ Returns: risk_metrics, position_recommendation

PHASE 4 (Sequential decision - 1 second):
  └─ Decision Agent (input: all previous results)
      ├─ Calls: synthesize_signals()        (weighted signal vote → bias + agreement)
      ├─ Calls: rank_opportunities()        (gates + scores call/put/equity candidates)
      └─ Returns: trade_decision, confidence_score

Total: ~10 seconds per trading cycle
```

The orchestrator (`orchestrator/pipeline.py`) additionally runs web research
(when warranted) and a final LLM synthesis into the intelligence report.
Because every phase is deterministic, the final LLM synthesis is given the
Phase 2–4 decision context so its narrative and `council_input` stay consistent
with the computed decision.

### Repo layout

| Module | Purpose |
|-------|---------|
| `agents/` | The 6 agents (phase 1 collection; phase 2-4 all implemented) + LLM client |
| `tools/` | Deterministic tool suite + `registry.py` (tool specs) |
| `schemas/` | Pydantic models (news / market / historical / prediction / risk / decision / pipeline) |
| `orchestrator/` | Phase-based pipeline + phase timing budgets |
| `alpaca/` | Alpaca data layer (news, market data, historical, options) |
| `web/` | Web research (Ollama search/fetch) |
| `utils/` | Config, logging, stage timing |

The **Historical Data Agent** (Phase 1, fully implemented) fetches price
history, dividends, and rolling realized volatility from Alpaca and produces
`historical_trends` + `volatility_history` plus a technical indicator bundle
(MA/RSI/MACD/Bollinger/Stochastic/ATR/ADX/OBV), support/resistance, core chart
patterns, volatility regimes, drawdown/VaR, gaps, events and a signal-voting
summary. Alpaca has no earnings endpoint — `get_earnings_history` is a
documented stub, and earnings-like events are inferred from bar signatures.

The **Risk Agent** (Phase 3, implemented) fetches the option chain via
`GET /v1beta1/options/snapshots/{symbol}` (real greeks + implied volatility +
bid/ask) and computes a stop (ATR + support), target, position size (equity and
long-only options, scaled by confidence / IV quality / spread / drawdown), a
gap-inflated max loss with a VaR/CVaR tail, and a composite 0–100 risk score.
When the options feed is unavailable (no subscription) it degrades gracefully
to Phase 2's estimated IV and local Black-Scholes greeks.

The **Decision Agent** (Phase 4, implemented) is the final deterministic
stage. `synthesize_signals` votes across every directional signal (news
sentiment, the historical technical summary, the historical trend, the Phase 2
composite signal, and market trend) into a `composite_bias` plus an
`agreement_score` and an auditable `divergences` list. `rank_opportunities`
then gates and scores up to three trade candidates — **long call**, **long
put**, and **long equity** — and returns a `trade_decision`
(`long_call | long_put | long_equity | hold | avoid`) with a
`confidence_score`.

The agent is:
- **bullish** → long call (or long equity), sized by Phase 3;
- **bearish** → long put. The Decision Agent re-sizes the put itself using
  Phase 3's position-size tool with the put premium + |put delta| (Phase 3
  only sizes calls), so bearish plays carry defined-risk premium;
- **neutral / gated-out** → `hold`; **very high risk** → `avoid`;
  **no price / prediction error** → `hold` with `status="insufficient_data"`.

Deterministic gates: `prediction.confidence >= MIN_CONFIDENCE`,
`risk_reward_ratio >= MIN_RISK_REWARD`, and `risk_level != very_high`. Each
candidate is scored 0–100 (signal/agreement 40%, reward:risk 25%, risk quality
20%, execution quality 15%) so the best instrument wins, and the result
carries a `decision_metrics` audit dict for transparency.

---

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com) running locally with `gemma4:e4b` pulled
- An [Alpaca](https://alpaca.markets/) account (free tier is sufficient)

## Setup

```bash
cd market_intel_agent
python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
ollama pull gemma4:e4b
```

Copy `.env.example` to `.env` and fill in your keys:

| Variable | Required | Purpose |
|----------|----------|---------|
| `ALPACA_API_KEY` | yes | Alpaca API key ID |
| `ALPACA_API_SECRET` | yes | Alpaca API secret |
| `ALPACA_DATA_FEED` | no | `iex` (free) or `sip` (paid); default `iex` |
| `ALPACA_OPTIONS_FEED` | no | `indicative` (free, delayed) or `opra` (paid); default `indicative` |
 | `ACCOUNT_CAPITAL` | no | capital base for position sizing; default `100000` |
| `RISK_PER_TRADE_PCT` | no | fraction of capital risked per trade; default `0.01` |
| `MAX_POSITION_PCT` | no | cap on position as a fraction of capital; default `0.05` |
| `MIN_RISK_REWARD` | no | minimum reward/risk gate for the Decision Agent; default `1.0` |
| `MIN_CONFIDENCE` | no | minimum prediction confidence gate for the Decision Agent; default `0.35` |
| `OLLAMA_BASE_URL` | no | local Ollama base URL (default `http://localhost:11434`) |
| `OLLAMA_MODEL` | no | model tag (default `gemma4:e4b`) |
| `OLLAMA_API_KEY` | no* | API key for Ollama web search (or run `ollama signin`) |
| `OLLAMA_WEB_SEARCH_URL` | no | default `http://localhost:11434/api/experimental/web_search` |
| `OLLAMA_WEB_FETCH_URL` | no | default `http://localhost:11434/api/experimental/web_fetch` |

\* Required for web research only; the core pipeline works without it.

## Run

```bash
python main.py NVDA                       # full 4-phase pipeline
python main.py NVDA --historical          # Phase 1 Historical Agent only
python main.py NVDA --news-limit 5 --lookback-hours 24 --verbose
```

The human-readable report is printed first, followed by the full
machine-readable JSON. Historical output includes trends, volatility history,
technical signals, and the signal-voting summary.

## Tests, lint, typecheck

```bash
python -m pytest -q        # mocked HTTP (respx) — no credentials required
ruff check .               # lint (uses .venv/bin/ruff)
pyright                    # typecheck (uses .venv/bin/pyright)
```

`opencode` LSP config lives in `.opencode/opencode.json` (pyright + ruff server
from the project `.venv`).

---

## Design notes

- **Anti-hallucination** — the system prompt forbids inventing sources, quotes,
  transactions, numbers, or dates; models must say "Insufficient evidence."
  or "Sources conflict." when appropriate, and lower confidence for
  unverifiable claims.
- **LLM never computes indicators** — all indicators/levels/risk math is
  deterministic Python (`tools/historical/*`, `ta`/`quantstats`/`hurst`).
- **Graceful degradation** — a failure in any one fetch produces a partial but
  still-valid result instead of a crash.
- **Async parallelism** — Phase 1 agents, Alpaca requests, search queries, and
  page fetches run concurrently; every stage is timed (`orchestrator/timings.py`).
- **No order endpoints** — only `data.alpaca.markets` is ever called; trading
  integration comes in later phases.

## Disclaimer

This is a research/information tool only. It does not place trades, does not
manage a portfolio, and does not use real money. Nothing it outputs is
investment advice.
