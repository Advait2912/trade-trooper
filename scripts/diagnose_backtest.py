"""Diagnose backtest P&L across multiple month windows.

Runs the deterministic Phase 2-4 backtest (default weights, news-aware) over
several windows on the live 11-ticker universe, then breaks down the pooled
trades by window, ticker, instrument, exit reason and decision to surface
what is driving profits/losses.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from alpaca.client import AlpacaClient
from alpaca.historical import get_price_history
from trading.backtest import run_backtest
from trading.journal import TradeJournal
from trading.news_cache import NewsCache
from utils.config import load_settings

UNIVERSE = ["NVDA", "AAPL", "MSFT", "AMD", "JPM", "BAC", "V", "GS", "TSLA", "XOM", "KO"]
WINDOWS = [3, 6, 9, 12, 18, 24]

_WARMUP = 260
_DAY = 252


def _stats(trades: list[dict]) -> dict:
    if not trades:
        return {"trades": 0}
    pnl = [t["pnl"] for t in trades]
    wins = [p for p in pnl if p > 0]
    losses = [p for p in pnl if p < 0]
    gw, gl = sum(wins), abs(sum(losses))
    pf = gw / gl if gl > 0 else (float("inf") if gw > 0 else 0.0)
    return {
        "trades": len(trades),
        "pnl": round(sum(pnl), 2),
        "win_rate": round(len(wins) / len(pnl), 4),
        "pf": round(pf, 2) if pf != float("inf") else "inf",
        "expectancy": round(sum(pnl) / len(pnl), 2),
        "avg_win": round(sum(wins) / len(wins), 2) if wins else 0.0,
        "avg_loss": round(sum(losses) / len(losses), 2) if losses else 0.0,
    }


async def run_window(settings, cache, months: int, equity_only: bool) -> dict:
    from tuning import TuningConfig

    tuning = TuningConfig(equity_only=equity_only) if equity_only else None
    days_back = _WARMUP + int(_DAY * max(1, months) / 12.0)
    bars_cache: dict[str, list] = {}
    async with AlpacaClient(settings) as client:
        for t in UNIVERSE:
            try:
                bars_cache[t] = await get_price_history(
                    client, t, days_back=days_back, interval="1d",
                    feed=settings.alpaca_data_feed,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  ! {t} fetch failed: {exc}")

    all_trades: list[dict] = []
    per_ticker: dict[str, list[dict]] = {t: [] for t in UNIVERSE}
    for t in UNIVERSE:
        if t not in bars_cache:
            continue
        j = TradeJournal(":memory:")
        await run_backtest(settings, t, months=months, journal=j,
                           bars=bars_cache[t], news_cache=cache, tuning=tuning)
        trades = j.trades()
        j.close()
        all_trades.extend(trades)
        per_ticker[t] = trades
    return {"all": all_trades, "per_ticker": per_ticker}


async def main() -> None:
    settings = load_settings()
    cache = NewsCache("data/news_cache.db")

    for months in WINDOWS:
        print(f"\n===== WINDOW: last {months} months =====")
        for label, eq in (("OPTIONS (default)", False), ("EQUITY ONLY", True)):
            res = await run_window(settings, cache, months, eq)
            all_trades = res["all"]
            s = _stats(all_trades)
            print(f"[{label}] AGGREGATE: {s}")

        # detailed breakdown for the default (options) mode only
        res = await run_window(settings, cache, months, False)
        all_trades = res["all"]
        per_ticker = res["per_ticker"]

        print("-- per ticker (options mode) --")
        for t in UNIVERSE:
            s = _stats(per_ticker[t])
            if s["trades"]:
                print(f"  {t:<5} pnl=${s['pnl']:>10,.2f}  n={s['trades']:>3}  "
                      f"win={s['win_rate']*100:>5.1f}%  pf={s['pf']}")

        print("-- per exit reason (options mode) --")
        reasons: dict[str, list] = {}
        for tr in all_trades:
            reasons.setdefault(tr["exit_reason"], []).append(tr)
        for reason, ts in sorted(reasons.items()):
            s = _stats(ts)
            print(f"  {reason:<10} n={s['trades']:>3}  pnl=${s['pnl']:>10,.2f}  "
                  f"win={s['win_rate']*100:>5.1f}%  avg_win=${s.get('avg_win',0):,.0f}  "
                  f"avg_loss=${s.get('avg_loss',0):,.0f}")

    cache.close()


if __name__ == "__main__":
    asyncio.run(main())