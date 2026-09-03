"""Weight-tuning harness for the deterministic Phase 2-4 chain.

Usage (from the repo root):

    # baseline / evaluate a single (possibly overridden) config over a universe
    python scripts/tune.py evaluate --universe NVDA,AMD,SPY --months 12
    python scripts/tune.py evaluate --config overrides.json --months 12

    # sweep a grid of knobs and print a ranked table
    python scripts/tune.py sweep --grid grid.json --months 12 --max-dd-cap 0.15

``grid.json`` is a JSON object mapping knob names to lists of candidate values;
each combination is run over every ticker in the universe and the aggregated
trade metrics are printed, sorted by expectancy.  Keys may be ``TuningConfig``
fields (any weight / threshold) or ``Settings`` fields (gates / sizing).

Bars are fetched once per ticker and reused across every config so a sweep is
cheap after the first fetch.
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from alpaca.client import AlpacaClient
from alpaca.historical import get_price_history
from trading.backtest import run_backtest
from trading.journal import TradeJournal
from tuning import TuningConfig
from utils.config import Settings, load_settings

DEFAULT_UNIVERSE = ["NVDA", "AMD", "TSLA", "AAPL", "MSFT", "SPY", "QQQ", "KO", "TLT", "XOM"]

_WARMUP = 260
_DAY = 252


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def aggregate_stats(trades: list[dict[str, Any]]) -> dict[str, Any]:
    """Pool trades (across tickers) into headline metrics."""
    pnl = [t["pnl"] for t in trades]
    wins = [p for p in pnl if p > 0]
    losses = [p for p in pnl if p < 0]
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else (float("inf") if gross_win > 0 else 0.0)

    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for p in pnl:
        equity += p
        peak = max(peak, equity)
        max_dd = min(max_dd, equity - peak)

    return {
        "trades": len(trades),
        "pnl": round(sum(pnl), 2),
        "win_rate": round(len(wins) / len(trades), 4) if trades else 0.0,
        "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else float("inf"),
        "expectancy": round(sum(pnl) / len(trades), 2) if trades else 0.0,
        "avg_win": round(sum(wins) / len(wins), 2) if wins else 0.0,
        "avg_loss": round(sum(losses) / len(losses), 2) if losses else 0.0,
        "max_dd": round(max_dd, 2),
    }


async def _run_config(
    settings: Settings,
    universe: list[str],
    months: int,
    tuning: TuningConfig,
    bars_cache: dict[str, list[Any]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Run one config over the universe; return pooled trades + per-ticker stats."""
    all_trades: list[dict[str, Any]] = []
    per_ticker: dict[str, dict[str, Any]] = {}
    for ticker in universe:
        journal = TradeJournal(":memory:")
        await run_backtest(
            settings, ticker, months=months, journal=journal, tuning=tuning,
            bars=bars_cache.get(ticker),
        )
        trades = journal.trades()
        journal.close()
        per_ticker[ticker] = aggregate_stats(trades)
        all_trades.extend(trades)
    return all_trades, per_ticker


def _apply_overrides(settings: Settings, tuning: TuningConfig, overrides: dict[str, Any]) -> None:
    for key, value in overrides.items():
        if hasattr(tuning, key):
            setattr(tuning, key, value)
        elif hasattr(settings, key):
            setattr(settings, key, value)
        else:
            raise ValueError(f"Unknown override key: {key!r}")


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------
async def _fetch_bars_for(settings: Settings, ticker: str, months: int) -> list[Any]:
    days_back = _WARMUP + int(_DAY * max(1, months) / 12.0)
    async with AlpacaClient(settings) as client:
        return await get_price_history(
            client, ticker, days_back=days_back, interval="1d", feed=settings.alpaca_data_feed
        )


async def _fetch_all(settings: Settings, universe: list[str], months: int) -> dict[str, list[Any]]:
    results = await asyncio.gather(
        *(_fetch_bars_for(settings, t, months) for t in universe), return_exceptions=True
    )
    cache: dict[str, list[Any]] = {}
    for ticker, res in zip(universe, results):
        if isinstance(res, BaseException):
            print(f"  ! {ticker}: fetch failed ({res}); skipping", file=sys.stderr)
        else:
            cache[ticker] = res
    return cache


# ---------------------------------------------------------------------------
# Table output
# ---------------------------------------------------------------------------
def _print_aggregate(label: str, agg: dict[str, Any], cap: float | None = None) -> None:
    dd_flag = ""
    if cap is not None and agg["trades"] and abs(agg["max_dd"]) > cap * 100_000:
        dd_flag = "  [MAX-DD CAP EXCEEDED]"
    pf = agg["profit_factor"]
    pf_s = "∞" if pf == float("inf") else f"{pf:.2f}"
    print(
        f"{label:<40} trades={agg['trades']:>4}  win={agg['win_rate']*100:>5.1f}%  "
        f"PF={pf_s:>5}  exp=${agg['expectancy']:>9}  pnl=${agg['pnl']:>10}  "
        f"maxDD=${agg['max_dd']:>9}{dd_flag}"
    )


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
async def _evaluate(args: argparse.Namespace) -> int:
    settings = load_settings()
    universe = [t.strip().upper() for t in args.universe.split(",") if t.strip()]
    overrides = json.loads(Path(args.config).read_text()) if args.config else {}
    tuning = TuningConfig.from_overrides(
        {k: v for k, v in overrides.items() if hasattr(TuningConfig(), k)}
    )
    _apply_overrides(settings, tuning, overrides)

    bars = await _fetch_all(settings, universe, args.months)
    trades, per_ticker = await _run_config(settings, universe, args.months, tuning, bars)

    print(f"\n== evaluate ({len(universe)} tickers, {args.months} months) ==")
    for ticker, agg in per_ticker.items():
        _print_aggregate(f"  {ticker}", agg)
    print("-" * 110)
    _print_aggregate("AGGREGATE", aggregate_stats(trades), cap=args.max_dd_cap)
    return 0


async def _sweep(args: argparse.Namespace) -> int:
    grid = json.loads(Path(args.grid).read_text())
    universe = [t.strip().upper() for t in args.universe.split(",") if t.strip()]
    if not grid:
        print("Grid is empty; nothing to sweep.", file=sys.stderr)
        return 2

    base_settings = load_settings()
    bars = await _fetch_all(base_settings, universe, args.months)

    keys = list(grid)
    combos = list(itertools.product(*(grid[k] for k in keys)))
    rows: list[tuple[str, dict[str, Any]]] = []

    print(f"\n== sweep: {len(combos)} configs x {len(universe)} tickers x {args.months} months ==")
    for idx, combo in enumerate(combos, start=1):
        overrides = dict(zip(keys, combo))
        settings = load_settings()  # fresh per config (gates/sizing may differ)
        tuning = TuningConfig.from_overrides(
            {k: v for k, v in overrides.items() if hasattr(TuningConfig(), k)}
        )
        _apply_overrides(settings, tuning, overrides)

        trades, _ = await _run_config(settings, universe, args.months, tuning, bars)
        agg = aggregate_stats(trades)
        rows.append((json.dumps(overrides, sort_keys=True), agg))
        _print_aggregate(f"[{idx}/{len(combos)}] {json.dumps(overrides, sort_keys=True)}", agg, cap=args.max_dd_cap)

    # ranked table (by expectancy, then profit factor)
    rows.sort(key=lambda r: (-r[1]["expectancy"], -r[1]["profit_factor"]))
    print("\n== ranked by expectancy ==")
    for label, agg in rows:
        _print_aggregate(label, agg, cap=args.max_dd_cap)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tune", description="Sweep/evaluate trading weights via backtests.")
    sub = parser.add_subparsers(dest="command", required=True)

    common = dict(
        universe=dict(default=",".join(DEFAULT_UNIVERSE),
                      help="comma-separated tickers (default: diversified set)"),
        months=dict(type=int, default=12, help="lookback months per ticker"),
        max_dd_cap=dict(type=float, default=0.15,
                        help="fraction of starting equity treated as max-drawdown cap"),
    )

    p = sub.add_parser("evaluate")
    for flag, kw in common.items():
        p.add_argument(f"--{flag.replace('_', '-')}", **kw)
    p.add_argument("--config", default=None, help="JSON file of overrides (default: none)")

    p = sub.add_parser("sweep")
    for flag, kw in common.items():
        p.add_argument(f"--{flag.replace('_', '-')}", **kw)
    p.add_argument("--grid", required=True, help="JSON file mapping knob -> list of values")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "evaluate":
        return asyncio.run(_evaluate(args))
    if args.command == "sweep":
        return asyncio.run(_sweep(args))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
