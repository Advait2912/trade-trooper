# market_intel_agent

A small but production-oriented, research-only AI agent that demonstrates the
core intelligence layer of an AI trading system — **without** placing any
orders or managing any money.

Given a ticker such as `NVDA`, it:

1. Fetches recent news from the **Alpaca News API**
2. Applies deterministic filtering (relevance, duplicates, recency)
3. Asks a local **Gemma 4 E4B** (`gemma4:e4b`) model to identify important events
4. Optionally researches/verifies the event via **Ollama web search**
5. Fetches current **Alpaca market data** and computes technical indicators
   deterministically in Python
6. Asks Gemma to synthesize everything into a structured intelligence report

The agent **never executes trades**, calls no order endpoints, and is
explicitly anti-hallucination by design.

---

## Architecture

```
observe → investigate → verify → contextualize → assess uncertainty → report
```

| Stage | Responsibility | Module |
|-------|----------------|--------|
| 1 | Alpaca news fetch + normalization | `alpaca/news.py` |
| 2 | Deterministic relevance/dedup/recency filter | `alpaca/news.py` |
| 3 | Initial Gemma analysis (structured JSON) | `agent/analyst.py` |
| 4 | Web search (controller-decided) | `web/search.py` |
| 5 | Web fetch + HTML text extraction | `web/fetch.py` |
| 6 | Market data + deterministic indicators | `alpaca/market_data.py`, `analysis/indicators.py` |
| 7 | Final Gemma synthesis | `agent/analyst.py`, `agent/prompts.py` |

The async controller orchestrating all stages lives in `agent/pipeline.py`.

Boundaries are deliberately clean:

- **data collection** — `alpaca/`, `web/`
- **deterministic analysis** — `analysis/indicators.py`
- **LLM reasoning** — `agent/analyst.py` (validated by `agent/schemas.py`)
- **synthesis** — `agent/pipeline.py`

---

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com) running locally with `gemma4:e4b` pulled
- An [Alpaca](https://alpaca.markets/) account (free tier is sufficient)

## Setup

```bash
cd market_intel_agent
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
ollama pull gemma4:e4b
```

Copy `.env.example` to `.env` and fill in your keys (a `.env` already exists
with blank values for you to complete):

| Variable | Required | Purpose |
|----------|----------|---------|
| `ALPACA_API_KEY` | yes | Alpaca API key ID |
| `ALPACA_API_SECRET` | yes | Alpaca API secret |
| `ALPACA_DATA_FEED` | no | `iex` (free) or `sip` (paid); default `iex` |
| `OLLAMA_BASE_URL` | no | local Ollama base URL (default `http://localhost:11434`) |
| `OLLAMA_MODEL` | no | model tag (default `gemma4:e4b`) |
| `OLLAMA_API_KEY` | no* | API key for Ollama web search (or run `ollama signin`) |
| `OLLAMA_WEB_SEARCH_URL` | no | default `http://localhost:11434/api/experimental/web_search` |
| `OLLAMA_WEB_FETCH_URL` | no | default `http://localhost:11434/api/experimental/web_fetch` |

\* Required for web research only; the core pipeline works without it.

## Run

```bash
python main.py NVDA
python main.py NVDA --news-limit 5 --lookback-hours 24 --verbose
```

The human-readable report is printed first, followed by the full
machine-readable JSON.

## Tests

Tests use mocked API responses (`respx`) and never require real credentials or
a running Ollama server:

```bash
python -m pytest -q
```

---

## Design notes

- **Anti-hallucination** — the system prompt forbids inventing sources, quotes,
  transactions, numbers, or dates; models must say "Insufficient evidence."
  or "Sources conflict." when appropriate, and lower confidence for
  unverifiable claims.
- **LLM never computes indicators** — SMA/RSI/ATR/returns/volatility are pure
  Python in `analysis/indicators.py`.
- **Graceful degradation** — a failure in news, Ollama, web research, or market
  data produces a partial but still-valid report instead of a crash.
- **Async parallelism** — independent Alpaca requests, search queries, and page
  fetches run concurrently; every stage is timed.
- **No order endpoints** — only `data.alpaca.markets` is ever called.

## Example output

```
========================================
MARKET INTELLIGENCE REPORT — NVDA
========================================

LATEST RELEVANT NEWS
--------------------
Headline: ...
Source: ...
Published: ...

EVENT
-----
...

NEWS SENTIMENT
--------------
Bullish

NEWS IMPACT
-----------
+0.55

WEB RESEARCH
------------
✓ Additional sources found
✓ 3 source(s) reviewed
✓ Event partially verified

MARKET
------
Price: $182.40
1D: +3.10%
5D: +8.40%
RSI: 64.2
SMA20: 174.10
SMA50: 161.80
Volume vs average: 1.42x
Trend: Bullish

ASSESSMENT
----------
...

ACTIONABILITY
-------------
Medium

CONFIDENCE
----------
0.72

COUNCIL INPUT
-------------
Bullish bias
Confidence: 0.68
========================================
```

## Disclaimer

This is a research/information tool only. It does not place trades, does not
manage a portfolio, and does not use real money. Nothing it outputs is
investment advice.
