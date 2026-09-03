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
| `TRADING_ENABLED` | no | **kill switch** for the paper-trading loop; default `false` |
| `TRADING_INTERVAL_MIN` | no | paper-loop cycle interval in minutes; default `30` |
| `MAX_OPEN_POSITIONS` | no | max simultaneous positions; default `1` |
| `DAILY_LOSS_LIMIT_PCT` | no | stop new entries for the day if this drawdown is hit; default `0.02` |
| `ORDER_TYPE` | no | `limit` or `market`; default `limit` |
| `TRADE_HORIZON_DAYS` | no | max holding period before forced exit; default `5` |
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
python main.py NVDA --trade               # paper-trading loop (TRADING_ENABLED=true)
python main.py NVDA --backtest --months 6 # deterministic Phase 2-4 historical replay
python main.py --stats                    # trade/backtest statistics report
```

The human-readable report is printed first, followed by the full
machine-readable JSON. Historical output includes trends, volatility history,
technical signals, and the signal-voting summary.

## Paper-trading loop (Phase 5 / execution)

Set `TRADING_ENABLED=true` in `.env`, then run the loop in the background (it
only executes during US market hours, keeps one position at a time, respects a
daily-loss limit, and is **paper-account only**):

```bash
nohup .venv/bin/python main.py NVDA --trade > trade.log 2>&1 &
```

Every cycle it manages open positions (stop/target, horizon expiry, or a flip
to `avoid`), runs the pipeline, and — if the decision is a trade and there is
room — places an order:

- `long_equity` → server-side **bracket** order (buy shares with a stop-loss and
  take-profit attached).
- `long_call` / `long_put` → **buy-to-open** limit order at (mid ± slippage) for
  the exact ATM contract Phase 3 selected. All options orders are defined-risk
  (max loss = premium).
- `hold` / `avoid` → no order.

Order IDs are deterministic per cycle (`client_order_id`), so a restart can
never double-fire an order. Everything is journaled to a SQLite database
(`trading_journal.db` by default, `--journal` to override).

### Backtest

`python main.py NVDA --backtest --months 6` replays the deterministic Phase 2-4
chain over daily bars (no LLM, no live option chain — Phase 3 uses its
Black-Scholes fallback) into the same journal, producing hundreds of simulated
trades in seconds. **Limitation:** the news-sentiment leg is LLM-based and is
turned off in the backtest (news-neutral), so the backtest measures the pure
technical/prediction edge; the forward paper run adds the news effect.

### Statistics

`python main.py --stats` reports realized P&L, win rate, profit factor,
expectancy, average win/loss, max drawdown, decision distribution, a
per-instrument breakdown and the latest equity snapshot.

### Tuning harness

Every numeric weight/threshold in Phases 2-4 lives in `tuning.py`
(`TuningConfig`). `scripts/tune.py` sweeps them against the backtest so they
can be fitted rather than hand-edited:

```bash
# baseline over a diversified universe (fetches bars once per ticker)
python scripts/tune.py evaluate --universe NVDA,AMD,SPY --months 12

# sweep a grid of knobs (JSON maps knob -> list of candidate values)
python scripts/tune.py sweep --grid grid.json --months 12 --max-dd-cap 0.15
```

Keys in a grid may be any `TuningConfig` field (e.g. `momentum_weights`,
`signal_weights`, `factor_weights`, `news_weight`, `min_confidence`) or any
`Settings` field (gates/sizing). Output is a table of aggregated profit factor,
expectancy, win rate and max drawdown, ranked by expectancy.

The backtest prices options with Black-Scholes (Phase 3 estimated IV, constant
over the holding period) so option P&L reflects premium/delta/gamma/theta
rather than a raw `underlying × 100` proxy.

## Risk assessment and alternatives

The agent is a long-only (-premium) volatility-and-momentum system. Honest
caveats:

**Strategy risk**
- The live decision edge is thin: observed reward/risk ~1.03 vs a ~2% option
  spread, so many cycles are `hold`s. In a high-IV regime (vol forecast 60%+)
  *buying* long options is expensive (theta decay can be ~8%/day of premium).
- The thresholds (`MIN_CONFIDENCE=0.35`, `MIN_RISK_REWARD=1.0`) are reasoned
  guesses, not fitted parameters; they are unlikely to be near-optimal.
- Single ticker, single position — correlated to one name's regime/news shocks.
- Short forward runs give a *sample* of tens of trades, **not** statistical
  significance. The backtest is where the count lives; only long runs (hundreds
  of trades) can meaningfully support a claim.

**Operational risk**
- Paper fills are simulated (NBBO-ish, `indicative` options feed is
  delayed/modified) and ignore commissions/OCC fees — real results would be
  worse.
- Bracket orders (equity) are monitored by Alpaca server-side; options are
  managed by the loop, so a crash mid-position relies on the loop resuming
  (idempotent, so it re-checks every cycle). Daily-loss limit and max positions
  bound the damage.
- The API key/secret are only ever used against the **paper** endpoint; the
  runner refuses to start if the key looks live (`AK` prefix). Rotate the key
  after any hackathon if it was shared.

**How downside is bounded**: paper-only, 1% capital risk per trade, ≤5% position
cap, defined-risk premium on options, one position, daily loss limit, kill
switch.

**Alternatives to consider**
1. **Backtest first** (built-in) — most evidence for the least cost.
2. **Debit spreads** instead of naked long options — cut theta/IV cost (Phase 5
   orders are long-call/long-put today; spreads are a follow-up).
3. **Raise the R:R gate** (e.g. `MIN_RISK_REWARD=1.5`) and/or skip entries when
   IV percentile is very high to avoid overpaying for premium.
4. **Equity-only** mode (bracket orders, no options) for a cleaner first cut.
5. **Multi-ticker diversification** once the single-name loop is stable.

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
- **Paper-only execution, kill-switched** — order endpoints are only ever hit
  against `paper-api.alpaca.markets`, gated by `TRADING_ENABLED` and refusing to
  start with a live (`AK`) key.

## Disclaimer

This is a research/information tool that, in its execution mode, trades a
**simulated paper account only**. It does not trade real money and nothing it
outputs is investment advice. Past/paper performance does not guarantee future
results.
