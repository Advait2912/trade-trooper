# Trade-Trooper — Findings & Diagnostics

Date: 2026-09-04 · Branch: `master` · Paper-trading only

---

## 1. Backtest P&L diagnosis across month windows

`scripts/diagnose_backtest.py` runs the deterministic Phase 2–4 backtest
(default weights, news-aware FinBERT cache) over the live 11-ticker universe
(`NVDA, AAPL, MSFT, AMD, JPM, BAC, V, GS, TSLA, XOM, KO`) across 3/6/9/12/18/24
month windows, breaking trades down by ticker, exit reason and instrument.

### Headline: the options leg was the P&L killer

| Window | Options PF | Options P&L | Equity-only PF | Equity-only P&L |
|--------|-----------|-------------|----------------|-----------------|
| 3m     | 1.06      | +$946       | **1.70**       | **+$2,342**     |
| 6m     | 0.99      | -$376       | **1.63**       | **+$3,860**     |
| 9m     | 0.94      | -$2,605     | **1.38**       | **+$3,448**     |
| 12m    | 0.89      | **-$6,295** | **1.17**       | **+$2,064**     |
| 18m    | 0.99      | -$986       | **1.17**       | **+$3,034**     |
| 24m    | 1.02      | +$2,423     | **1.20**       | **+$4,113**     |

**Equity-only is profitable in every window; options mode is breakeven or
losing everywhere.**

### Why options lose: theta decay

Every backtest trade was a long option (call/put), priced with Black-Scholes
at the Phase 3 estimated IV, constant over the holding period. Time decay
bleeds premium daily. Verified on an ATM call at 45% IV:

| Days left | Premium |
|-----------|---------|
| 5         | $3.34   |
| 4         | $2.99   |
| 3         | $2.58   |
| 2         | $2.10   |
| 1         | $1.48   |

≈ 56% of premium lost to theta in 5 days if the underlying does not move.
The README's "high-IV regime → buying long options is expensive" caveat is
exactly what the backtest measures.

### Exit structure is inverted (24-month window, options mode)

| Exit reason | Trades | P&L | Win rate | Avg win | Avg loss |
|-------------|-------:|----:|---------:|--------:|---------:|
| target      | 80 (7.7%) | **+$43,042** | 98.8% | $546 | -$80 |
| horizon     | 765 (74%) | **-$11,062** | 33.1% | $255 | -$147 |
| stop        | 190 (18%) | **-$29,557** | 2.1% | $6 | -$159 |

- Target hits produce all the profit but are rare (7.7%).
- 74% of trades ride the 5-day horizon and close flat-to-slightly-down
  (including theta drag).
- Stops hit exactly as sized (-$159 avg = the risked amount).

### Win-rate / reward profile (options mode)

- Win rate 32–37%, avg win $274–$320, avg loss $150–$152 → realized R:R ≈ 1.9–2.1.
- Breakeven win rate at R:R 2.0 is 33.3% → the strategy sits **right at
  breakeven** in options mode.
- Equity-only mode: win rate ~52–55%, avg win ~$90–$120, avg loss ~$80–$87.

### Chronic losers / winners by ticker (12-month window)

| Ticker | Options P&L | Equity-only P&L |
|--------|------------:|----------------:|
| NVDA   | -$2,092     | -$869           |
| AAPL   | +$1,302     | +$910           |
| MSFT   | -$948       | -$166           |
| AMD    | +$774       | +$285           |
| JPM    | -$2,331     | -$302           |
| BAC    | +$2,764     | +$675           |
| V      | -$1,709     | -$1             |
| GS     | -$488       | +$758           |
| TSLA   | -$266       | +$58            |
| XOM    | -$482       | +$695           |
| KO     | **-$2,819** | **+$308**       |

Consistent losers in options mode: KO, V, NVDA, MSFT, JPM. Consistent
winners: BAC, AMD, AAPL. KO flips from worst (-$2,819) to positive (+$308)
in equity-only mode.

---

## 2. Fix applied

`data/weights_db.json` now ships:

```json
{ "default": { "equity_only": true } }
```

The live pipeline (`orchestrator/pipeline.py` → `weights_db.resolve_config`)
applies `equity_only=True` to every ticker, so the live runner and all
backtests trade **equity** (server-side bracket orders with stop/target)
instead of paying option theta.

Verified through the official path:

```
== evaluate (11 tickers, 12 months, news-aware) ==
AGGREGATE   trades= 201  win= 50.7%  PF= 1.24  exp=$ 11.70  pnl=$ 2,351.25
```

vs options mode on the same window: PF 0.89, -$6,295 → a ~$8.6k swing.

The tuned weights (Optuna, all 11 industries) are preserved at
`data/weights_db_tuned_backup.json`; the Tuning tab in the UI reads that file.

---

## 3. Out-of-sample validation (previous session)

Gate rule: held-out PF ≥ 1.5 to go live with tuned weights.

- Train: 2025-01-01..2025-12-31 (Optuna, 15 trials/industry, news-aware)
- Test: 2026-01-05..2026-09-04 (8 months, held out)
- Universe: NVDA, AAPL, JPM, XOM, KO, TSLA, LLY, GS

| Config | PF | P&L | Trades |
|--------|---:|----:|-------:|
| 2025-trained weights | 0.51 | -$2,724 | 56 |
| Default weights | **1.38** | +$2,833 | 93 |

**Conclusion: the Optuna tuning overfit.** Default weights generalized better
and the live runner runs with defaults (+ the equity-only fix). This is the
honest engineering decision — in-sample tuning (README's PF 1.71 claim) does
not transfer out-of-sample.

---

## 4. Known limitations (honest)

- Paper fills are simulated (NBBO-ish, `indicative` options feed is
  delayed/modified); commissions/OCC fees are not modeled.
- The backtest prices options with Black-Scholes at constant Phase 3 IV —
  real option outcomes vary (and the theta drag is real for long premium).
- 12-month equity-only PF 1.24 on 201 trades is promising but not
  statistically significant; longer forward runs are needed.
- Single-account, paper-only. No live capital is involved.

---

## 5. Reproducing

```bash
# Windowed P&L diagnosis (options vs equity-only, per ticker / exit reason)
.venv/bin/python scripts/diagnose_backtest.py

# Single-window evaluation, equity-only (official path)
.venv/bin/python scripts/tune.py evaluate \
  --weights-db data/weights_db.json \
  --universe NVDA,AAPL,MSFT,AMD,JPM,BAC,V,GS,TSLA,XOM,KO \
  --news-cache data/news_cache.db --months 12

# Compare against options mode
.venv/bin/python scripts/tune.py evaluate \
  --universe NVDA,AAPL,MSFT,AMD,JPM,BAC,V,GS,TSLA,XOM,KO \
  --news-cache data/news_cache.db --months 12
```