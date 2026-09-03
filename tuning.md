# Tuning guide

How weight/threshold tuning works in this repo, how to tune a single universe,
and how the per-industry weights DB (`data/weights_db.json`) lets you give every
sector its own config.

## What is tunable

Every numeric knob in the deterministic Phase 2-4 chain lives in one of two
places:

- **`TuningConfig`** (`tuning.py`) — weights and thresholds consumed by the
  tools, e.g. `momentum_weights`, `signal_weights`, `factor_weights`,
  `news_weight`, `equity_only`, `conf_base`.
- **`Settings`** (`utils/config.py`) — gates and sizing, e.g.
  `min_confidence`, `min_risk_reward`, `trade_horizon_days`.

`scripts/tune.py` treats them uniformly: any key that exists on either class can
be set, swept, or optimized, and the harness routes it with
`_apply_overrides` (`settings`, `tuning`, overrides).

## How a config is scored

A config is evaluated by replaying the backtest over a universe (bars are
fetched once per ticker and reused across every config/trial). The pooled
trades produce headline stats: trade count, win rate, profit factor (PF),
expectancy, and max drawdown.

The optimizer maximizes a composite score (`_objective_score` in `tune.py`):

```
0.4 * min(PF, 3)/3  +  0.3 * win_rate  +  0.3 * sigmoid(20 * expectancy / R)
```

where `R = account_capital * risk_per_trade_pct`. Configs with fewer than
`--min-trades` trades get a heavy penalty, and drawdowns beyond
`--max-dd-cap` (fraction of starting equity) subtract from the score.

## Commands

### `evaluate` — score one (or a few) configs

```bash
# baseline over a universe
python scripts/tune.py evaluate --universe NVDA,AMD,SPY --months 12

# preset + inline override
python scripts/tune.py evaluate --preset equity_only --set min_confidence=0.45

# per-industry weights (each ticker uses its industry's config)
python scripts/tune.py evaluate --weights-db data/weights_db.json \
    --universe NVDA,AAPL,JPM --news-cache data/news_cache.db
```

`--set key=v` fixes a key; `--config overrides.json` loads a whole dict.
`--news-cache` makes the backtest news-aware (see below).

**Historical train/test splits**: every command accepts `--end-date YYYY-MM-DD`,
which re-anchors the backtest window to that date instead of today. Tune on a
past window and then evaluate on a later one to measure out-of-sample
generalization:

```bash
# train: last 6 months of 2024
python scripts/tune.py optimize-stocks --weights-db data/weights_db.json \
    --tickers NVDA --months 6 --end-date 2024-12-31 --n-trials 50 --news-cache data/news_cache.db
# test: calendar 2025 with the trained config
python scripts/tune.py evaluate --weights-db data/weights_db.json --universe NVDA \
    --months 12 --end-date 2025-12-31 --news-cache data/news_cache.db
```

The fetched bar history always includes the ~1-year indicator warm-up before
the tradeable window, so a backtest needs news coverage over the window's own
dates to be news-aware.

### `sweep` — grid-search a few knobs

```bash
python scripts/tune.py sweep --universe NVDA,AMD,SPY --preset equity_only \
    --set min_confidence=0.35,0.5 --set min_risk_reward=1.0,1.5
```

`--set key=v1,v2` sweeps that key; `--set key=v` fixes it. Results are ranked
by expectancy.

### `optimize` — Optuna (TPE) over one universe

```bash
python scripts/tune.py optimize --universe NVDA,AMD,SPY --n-trials 100 \
    --validate-universe TSLA,TLT --news-cache data/news_cache.db
```

Samples `signal_weights` + `momentum_weights` (simplex-normalized),
`min_confidence`, `min_risk_reward`, `trade_horizon_days`, and `equity_only`.
Prints the best config and (optionally) validates it on held-out tickers.
Requires `optuna` (`pip install -r requirements-ml.txt`).

### `optimize-industries` — tune once per industry, save to a weights DB

```bash
python scripts/tune.py optimize-industries --weights-db data/weights_db.json \
    --industries Technology,Financials --n-trials 50 --news-cache data/news_cache.db
```

For each industry it runs the same Optuna search over `INDUSTRY_STOCKS[name]`
tickers, then writes the best config into `data/weights_db.json` under that industry
key and prints it. The DB is saved **after each industry**, so an interrupted
run keeps everything tuned so far. Use `--industries all` for every industry in
the table.

### `optimize-stocks` — tune one ticker at a time

```bash
python scripts/tune.py optimize-stocks --weights-db data/weights_db.json \
    --tickers NVDA,TSLA --n-trials 50 --min-trades 5 --news-cache data/news_cache.db
```

Runs the same Optuna search over a **single** ticker and writes the best config
under `data/weights_db.json`'s reserved `"tickers"` namespace
(`db["tickers"]["NVDA"]`). The DB is saved **after each ticker**, so an
interrupted run keeps everything tuned so far.

Because a single ticker trades far less than a whole universe,
`optimize-stocks` defaults `--min-trades` to **5** (the universe-wide commands
default to 30) — pass a higher value if your ticker produces more trades.
Stock-specific configs **override** the industry config for that ticker, so use
them sparingly for names that genuinely diverge from their sector.

## The industry weights DB

`weights_db.py` maps tickers to industries and stores one tuned config per
industry as JSON:

```json
{
  "Technology": {
    "equity_only": true,
    "min_confidence": 0.40,
    "min_risk_reward": 0.95,
    "trade_horizon_days": 7,
    "momentum_weights": { "macd": 0.08, "adx": 0.22, "...": 0.0 },
    "signal_weights":   { "news_sentiment": 0.28, "historical_trend": 0.39, "...": 0.0 }
  },
  "Financials": { "...": 0.0 },
  "default": { "min_confidence": 0.52, "...": 0.0 },
  "tickers": { "NVDA": { "min_confidence": 0.30, "...": 0.0 } }
}
```

`"tickers"` is a reserved key holding per-stock overrides (populated by
`optimize-stocks`); it is optional and backward compatible.

### Resolution order

When `evaluate --weights-db` runs, each ticker's overrides are merged in this
order (later layers win):

1. `--preset` / `--set` / `--config` — the global base layer, applied to every
   ticker.
2. the `default` entry — the tuned config that applies to any industry without
   an entry.
3. the ticker's own industry entry (`weights_db.INDUSTRY_STOCKS` → inverse map
   `TICKER_TO_INDUSTRY`).
4. the ticker's own stock-specific entry (`db["tickers"][TICKER]`).

An unknown ticker resolves to the `default` entry (plus any stock-specific
entry). Entries may contain both `TuningConfig` and `Settings` keys — they are
applied per ticker with fresh `Settings`/`TuningConfig` instances, so each
ticker runs the backtest with its own gates, horizon, and weights. `evaluate`
annotates a ticker with `*` (e.g. `NVDA [Technology]*`) when a stock-specific
override is active.

### The industry table

`INDUSTRY_STOCKS` in `weights_db.py` currently covers 11 industries, 6 tickers
each: Technology, Financials, Healthcare, Consumer Discretionary, Industrials,
Communication Services, Consumer Staples, Energy, Materials, Utilities, and
Real Estate. Edit that dict to change coverage; the inverse map is derived
automatically.

## News-aware tuning

The backtest is news-neutral by default. To include the news-sentiment leg,
build a historical sentiment cache first (FinBERT, GPU recommended; requires
`requirements-ml.txt`):

```bash
python scripts/build_news_cache.py NVDA,AAPL,JPM --start 2025-01-01 --end 2026-09-03
```

The cache lands in `data/news_cache.db` (the default) and is picked up
**automatically** by `evaluate`/`sweep`/`optimize`/`optimize-industries` —
`--news-cache <path>` overrides it, and a missing cache falls back to
news-neutral. Re-running is idempotent (only newly-fetched articles are
scored) and recomputes each ticker's per-day aggregate, so you can keep
extending one DB as you add tickers. **Align `--start`/`--end` with the months
you backtest** — days with no cached sentiment fall back to news-neutral.

## Recommended workflow

1. Build/extend the news cache for the tickers you care about.
2. Tune a couple of industries first to sanity-check the pipeline:
   `optimize-industries --industries Technology,Financials --n-trials 50`.
3. Inspect the saved `data/weights_db.json` entries and the printed train scores.
4. For names that diverge from their sector, tune them individually:
   `optimize-stocks --tickers NVDA --n-trials 50 --min-trades 5`.
5. Backtest a **mixed** universe with `evaluate --weights-db data/weights_db.json
   --universe <mix>` to see how industry- and stock-specific weights generalize
   together.
6. Expand to more industries (`--industries all`), then re-evaluate.

## Caveats

- Tuning is **in-sample**: Optuna searches on the same window it scores, so
  train scores overstate real-world edge. Use a held-out `--validate-universe`
  (or fresh dates) before trusting a config.
- Stock-specific tuning overfits the most — one ticker yields few trades, so a
  per-stock config can chase noise. Prefer the industry layer unless the name
  clearly behaves differently from its sector.
- The backtest simulates fills (next-open + 0.05% slippage) and ignores
  commissions/fees.
- Industry weights are only meaningful when the industry's tickers share a
  regime; a lone mega-cap (e.g. NVDA) may not represent its whole sector.
- Untuned industries fall back to `default`, so tune the sectors you actually
  trade.