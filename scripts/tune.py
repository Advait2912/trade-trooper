"""Weight-tuning harness for the deterministic Phase 2-4 chain.

Everything is driven from the command line â€” no JSON files required:

    # baseline over a universe
    python scripts/tune.py evaluate --universe NVDA,AMD,SPY --months 12

    # evaluate a single config with inline overrides and/or a preset
    python scripts/tune.py evaluate --preset equity_only --set min_confidence=0.45

    # evaluate with per-industry weights (each ticker uses its industry's config)
    python scripts/tune.py evaluate --weights-db data/weights_db.json \
        --universe NVDA,AAPL,JPM --news-cache data/news_cache.db

    # sweep knobs: a single value fixes it, comma-separated values sweep it
    python scripts/tune.py sweep --universe NVDA,AMD,SPY --preset equity_only \
        --set min_confidence=0.35,0.5 --set min_risk_reward=1.0,1.5

``--set key=v`` fixes ``key``; ``--set key=v1,v2,...`` sweeps ``key`` over that
list.  ``--preset NAME`` applies a named override from ``tuning.PRESETS``
(``default``, ``equity_only``, ``conservative``, ``aggressive``,
``signal_prediction_led``, ``signal_technical_led``).  Keys may be
``TuningConfig`` fields (weights/thresholds) or ``Settings`` fields
(gates/sizing).  Values are auto-coerced (bool / int / float / str).

Per-industry weights are stored in a JSON DB (``data/weights_db.json``).  The
``evaluate`` ``--weights-db`` flag resolves each ticker's industry config via
``weights_db.resolve_overrides`` (``--preset``/``--set`` are the global base
layer, then ``default``, then the industry entry, then any stock-specific entry
under ``weights_db.TICKER_NAMESPACE``).  ``optimize-industries`` runs the Optuna
search per industry and writes each industry's best config back into the DB;
``optimize-stocks`` does the same per individual ticker.

A JSON file is still available for dict-valued sweeps you can't express as a
flag: ``--config overrides.json`` (evaluate) or ``--grid grid.json`` (sweep).

Bars are fetched once per ticker and reused across every config.
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import math
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from alpaca.client import AlpacaClient
from alpaca.historical import get_price_history
from trading.backtest import run_backtest
from trading.journal import TradeJournal
from trading.news_cache import NewsCache
from tuning import PRESETS, TuningConfig
from utils.config import Settings, load_settings
from utils.paths import data_path
from weights_db import (
    INDUSTRY_STOCKS,
    TICKER_NAMESPACE,
    load_weights_db,
    resolve_config,
    resolve_industry,
    save_weights_db,
)

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
    config_for: Callable[[str], tuple[Settings, TuningConfig]],
    universe: list[str],
    months: int,
    bars_cache: dict[str, list[Any]],
    news_cache: NewsCache | None = None,
    journal: TradeJournal | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Run one config per ticker over the universe; return pooled + per-ticker stats.

    ``config_for(ticker)`` returns the (Settings, TuningConfig) for that ticker,
    so a weights DB can hand each ticker its industry's config while a plain
    evaluate/sweep passes a uniform pair.  ``journal`` (optional) persists every
    simulated trade to one SQLite file for the dashboard; ``start_date``/
    ``end_date`` clip the tradeable window.
    """
    all_trades: list[dict[str, Any]] = []
    per_ticker: dict[str, dict[str, Any]] = {}
    for ticker in universe:
        settings, tuning = config_for(ticker)
        if journal is not None:
            trades = journal.trades()
            seen = len(trades)
            await run_backtest(
                settings, ticker, months=months, journal=journal, tuning=tuning,
                bars=bars_cache.get(ticker), news_cache=news_cache,
                start_date=_parse_date(start_date), end_date=_parse_date(end_date),
            )
            new_trades = journal.trades()[seen:]
        else:
            j = TradeJournal(":memory:")
            await run_backtest(
                settings, ticker, months=months, journal=j, tuning=tuning,
                bars=bars_cache.get(ticker), news_cache=news_cache,
                start_date=_parse_date(start_date), end_date=_parse_date(end_date),
            )
            new_trades = j.trades()
            j.close()
        per_ticker[ticker] = aggregate_stats(new_trades)
        all_trades.extend(new_trades)
    return all_trades, per_ticker


def _build_config(base_overrides: dict[str, Any]) -> tuple[Settings, TuningConfig]:
    """Build a fresh (Settings, TuningConfig) from (possibly partial) overrides."""
    settings = load_settings()
    tuning = TuningConfig()
    _apply_overrides(settings, tuning, base_overrides)
    return settings, tuning


def _apply_overrides(settings: Settings, tuning: TuningConfig, overrides: dict[str, Any]) -> None:
    for key, value in overrides.items():
        if hasattr(tuning, key):
            setattr(tuning, key, value)
        elif hasattr(settings, key):
            setattr(settings, key, value)
        else:
            raise ValueError(f"Unknown override key: {key!r}")


# ---------------------------------------------------------------------------
# Override parsing (presets + inline --set)
# ---------------------------------------------------------------------------
def _coerce(value: str) -> Any:
    value = value.strip()
    if value.lower() in ("true", "yes", "1"):
        return True
    if value.lower() in ("false", "no", "0"):
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _parse_set_item(item: str) -> tuple[str, list[Any]]:
    if "=" not in item:
        raise ValueError(f"Invalid --set {item!r}; expected key=value")
    key, _, raw = item.partition("=")
    key = key.strip()
    values = [_coerce(v) for v in raw.split(",") if v.strip() != ""]
    if not key or not values:
        raise ValueError(f"Invalid --set {item!r}")
    return key, values


def _preset_overrides(names: list[str]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for name in names:
        if name not in PRESETS:
            raise ValueError(f"Unknown preset {name!r}; available: {sorted(PRESETS)}")
        merged.update(PRESETS[name])
    return merged


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------
async def _fetch_bars_for(
    settings: Settings,
    ticker: str,
    months: int,
    end_date: str | None = None,
    start_date: str | None = None,
) -> list[Any]:
    end = _parse_date(end_date)
    start = _parse_date(start_date)
    if start is not None and end is not None:
        span_trading = int((end - start).days * _DAY / 365.0)
        days_back = _WARMUP + span_trading + 20
        fetch_start = start - timedelta(days=400)  # ~285 trading days of warm-up
    else:
        days_back = _WARMUP + int(_DAY * max(1, months) / 12.0)
        fetch_start = None
    async with AlpacaClient(settings) as client:
        return await get_price_history(
            client, ticker, days_back=days_back, interval="1d",
            feed=settings.alpaca_data_feed, end_date=end, start_date=fetch_start,
        )


def _parse_date(value: str | None) -> date | None:
    """Parse --end-date / --start-date YYYY-MM-DD into a date (None when omitted)."""
    if not value:
        return None
    return date.fromisoformat(value)


async def _fetch_all(
    settings: Settings,
    universe: list[str],
    months: int,
    end_date: str | None = None,
    start_date: str | None = None,
) -> dict[str, list[Any]]:
    results = await asyncio.gather(
        *(_fetch_bars_for(settings, t, months, end_date, start_date) for t in universe),
        return_exceptions=True,
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
    pf_s = "inf" if pf == float("inf") else f"{pf:.2f}"
    print(
        f"{label:<44} trades={agg['trades']:>4}  win={agg['win_rate']*100:>5.1f}%  "
        f"PF={pf_s:>5}  exp=${agg['expectancy']:>9}  pnl=${agg['pnl']:>10}  "
        f"maxDD=${agg['max_dd']:>9}{dd_flag}"
    )


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def _universe(args: argparse.Namespace) -> list[str]:
    return [t.strip().upper() for t in args.universe.split(",") if t.strip()]


def _open_news_cache(args: argparse.Namespace) -> NewsCache | None:
    """Open the news cache, or None when absent (news-neutral backtest)."""
    if not args.news_cache:
        return None
    path = Path(args.news_cache)
    if not path.exists():
        print(f"  ! news cache {path} not found; running news-neutral", file=sys.stderr)
        return None
    return NewsCache(path)


async def _evaluate(args: argparse.Namespace) -> int:
    universe = _universe(args)
    base = _preset_overrides(args.preset)
    for item in args.set:
        key, values = _parse_set_item(item)
        if len(values) != 1:
            print(f"Error: evaluate --set takes a single value (got {item!r}).", file=sys.stderr)
            return 2
        base[key] = values[0]
    if args.config:
        base.update(json.loads(Path(args.config).read_text()))

    db = load_weights_db(args.weights_db) if args.weights_db else None
    base_settings = load_settings()

    def config_for(ticker: str) -> tuple[Settings, TuningConfig]:
        if db is not None:
            return resolve_config(ticker, db, base_settings, base)
        return _build_config(base)

    journal = TradeJournal(args.journal) if args.journal else None
    news_cache = _open_news_cache(args)
    try:
        bars = await _fetch_all(base_settings, universe, args.months,
                                args.end_date, args.start_date)
        trades, per_ticker = await _run_config(
            config_for, universe, args.months, bars, news_cache,
            journal=journal, start_date=args.start_date, end_date=args.end_date,
        )
    finally:
        if news_cache:
            news_cache.close()
        if journal:
            journal.close()

    print(f"\n== evaluate ({len(universe)} tickers, {args.months} months) ==")
    for ticker, agg in per_ticker.items():
        if db is not None:
            stock = (db.get(TICKER_NAMESPACE) or {}).get(ticker)
            star = "*" if stock else ""
            label = f"  {ticker} [{resolve_industry(ticker)}{star}]"
        else:
            label = f"  {ticker}"
        _print_aggregate(label, agg)
    print("-" * 112)
    _print_aggregate("AGGREGATE", aggregate_stats(trades), cap=args.max_dd_cap)
    return 0


async def _sweep(args: argparse.Namespace) -> int:
    universe = _universe(args)

    base = _preset_overrides(args.preset)
    axes: dict[str, list[Any]] = {}
    for item in args.set:
        key, values = _parse_set_item(item)
        if len(values) == 1:
            base[key] = values[0]      # fixed value
        else:
            axes[key] = values         # sweep axis
    if args.grid:
        axes.update(json.loads(Path(args.grid).read_text()))

    if not axes:
        print("Nothing to sweep: give --set key=v1,v2,... or --grid grid.json.", file=sys.stderr)
        return 2

    base_settings = load_settings()
    news_cache = _open_news_cache(args)
    bars = await _fetch_all(base_settings, universe, args.months, args.end_date, args.start_date)

    keys = list(axes)
    combos = list(itertools.product(*(axes[k] for k in keys)))
    rows: list[tuple[str, dict[str, Any]]] = []

    print(f"\n== sweep: {len(combos)} configs x {len(universe)} tickers x {args.months} months "
          f"(fixed: {json.dumps(base, sort_keys=True)}) ==")
    try:
        for idx, combo in enumerate(combos, start=1):
            overrides = dict(base)
            overrides.update(dict(zip(keys, combo)))
            settings = load_settings()  # fresh per config (gates/sizing may differ)
            tuning = TuningConfig()
            _apply_overrides(settings, tuning, overrides)

            trades, _ = await _run_config(
                lambda ticker: (settings, tuning), universe, args.months, bars, news_cache,
                start_date=args.start_date, end_date=args.end_date,
            )
            agg = aggregate_stats(trades)
            rows.append((json.dumps(overrides, sort_keys=True), agg))
            _print_aggregate(f"[{idx}/{len(combos)}] {json.dumps(overrides, sort_keys=True)}", agg, cap=args.max_dd_cap)
    finally:
        if news_cache:
            news_cache.close()

    rows.sort(key=lambda r: (-r[1]["expectancy"], -r[1]["profit_factor"]))
    print("\n== ranked by expectancy ==")
    for label, agg in rows:
        _print_aggregate(label, agg, cap=args.max_dd_cap)
    return 0


# ---------------------------------------------------------------------------
# Optimizer (Optuna TPE) â€” learns weights against a composite loss
# ---------------------------------------------------------------------------
def _objective_score(
    agg: dict[str, Any],
    min_trades: int = 30,
    dd_cap: float = 0.15,
    capital: float = 100_000.0,
    risk_pct: float = 0.01,
) -> float:
    """Composite score to maximize: 0.4*PF + 0.3*win_rate + 0.3*expectancy,
    penalized for too-few trades and excessive drawdown."""
    trades = agg["trades"]
    if trades < min_trades:
        return -1000.0 + trades  # heavy penalty, still ordered by trade count
    pf_score = min(agg["profit_factor"], 3.0) / 3.0
    wr = agg["win_rate"]
    exp_r = agg["expectancy"] / (capital * risk_pct)  # R multiples per trade
    exp_score = 1.0 / (1.0 + math.exp(-exp_r * 20.0))
    score = 0.4 * pf_score + 0.3 * wr + 0.3 * exp_score
    dd = abs(agg["max_dd"]) / capital
    if dd > dd_cap:
        score -= (dd - dd_cap) * 2.0
    return score


_SIGNAL_KEYS = ["news_sentiment", "technical_summary", "historical_trend", "prediction_signal", "market_trend"]
_MOMENTUM_KEYS = ["macd", "adx", "rsi", "bollinger", "obv", "stochastic"]


def _sample_simplex(trial: Any, keys: list[str], prefix: str) -> dict[str, float]:
    raw = {k: trial.suggest_float(f"{prefix}_{k}", 0.0, 1.0) for k in keys}
    total = sum(raw.values()) or 1.0
    return {k: round(v / total, 6) for k, v in raw.items()}


def _sample_trial(trial: Any) -> dict[str, Any]:
    overrides: dict[str, Any] = {
        "signal_weights": _sample_simplex(trial, _SIGNAL_KEYS, "sw"),
        "momentum_weights": _sample_simplex(trial, _MOMENTUM_KEYS, "mw"),
        "min_confidence": trial.suggest_float("min_confidence", 0.0, 0.7),
        "min_risk_reward": trial.suggest_float("min_risk_reward", 0.5, 2.5),
        "trade_horizon_days": trial.suggest_int("trade_horizon_days", 2, 7),
        "equity_only": trial.suggest_categorical("equity_only", [False, True]),
    }
    return overrides


def _render_progress(done: int, total: int, msg: str) -> None:
    """Live progress bar on a TTY, or a clean per-trial line in a log."""
    if sys.stderr.isatty():
        width = 30
        filled = int(width * done / total)
        bar = "#" * filled + "-" * (width - filled)
        sys.stderr.write(f"\r[{bar}] {done}/{total} ({100 * done // total}%) {msg}")
        sys.stderr.flush()
    else:
        sys.stderr.write(f"[{done}/{total}] {msg}\n")
        sys.stderr.flush()


async def _optimize_universe(
    train: list[str],
    months: int,
    n_trials: int,
    bars: dict[str, list[Any]],
    news_cache: NewsCache | None,
    min_trades: int,
    dd_cap: float,
    start_date: str | None = None,
    end_date: str | None = None,
    progress_file: str | None = None,
) -> tuple[dict[str, Any], float] | None:
    """Run Optuna TPE over ``train``; return (best overrides, best score)."""
    try:
        import optuna  # noqa: PLC0415 - optional dependency
    except ImportError:
        print("optuna is required for tuning. Run `pip install optuna`.", file=sys.stderr)
        return None

    sampler = optuna.samplers.TPESampler(seed=0)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    for n in range(n_trials):
        trial = study.ask()
        overrides = _sample_trial(trial)
        trial.set_user_attr("config", json.dumps(overrides, sort_keys=True))
        settings = load_settings()
        tuning = TuningConfig()
        _apply_overrides(settings, tuning, overrides)

        trades, _ = await _run_config(
            lambda ticker: (settings, tuning), train, months, bars, news_cache,
            start_date=start_date, end_date=end_date,
        )
        agg = aggregate_stats(trades)
        score = _objective_score(agg, min_trades, dd_cap)
        study.tell(trial, score)
        msg = (f"score={score:.3f} PF={agg['profit_factor']:.2f} "
               f"win={agg['win_rate'] * 100:.1f}% exp=${agg['expectancy']:.2f} "
               f"trades={agg['trades']}")
        _render_progress(n + 1, n_trials, msg)
        if progress_file:
            try:
                Path(progress_file).write_text(json.dumps({
                    "done": n + 1,
                    "total": n_trials,
                    "message": msg,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }))
            except OSError:
                pass
    if sys.stderr.isatty():
        sys.stderr.write("\n")

    best = study.best_trial
    best_overrides = json.loads(best.user_attrs["config"])
    return best_overrides, best.value


async def _optimize(args: argparse.Namespace) -> int:
    train = _universe(args)
    validate = [t.strip().upper() for t in (args.validate_universe or "").split(",") if t.strip()]

    train_bars = await _fetch_all(load_settings(), train, args.months, args.end_date, args.start_date)
    news_cache = _open_news_cache(args)
    try:
        print(f"\n== optimize: {args.n_trials} trials x {len(train)} tickers x {args.months} months ==")
        result = await _optimize_universe(
            train, args.months, args.n_trials, train_bars, news_cache,
            args.min_trades, args.max_dd_cap, args.start_date, args.end_date,
            args.progress_file,
        )
    finally:
        if news_cache:
            news_cache.close()
    if result is None:
        return 2
    best_overrides, best_value = result

    print(f"\n== best config (train score={best_value:.3f}) ==")
    print(json.dumps(best_overrides, indent=2, sort_keys=True))

    if validate:
        vcache = _open_news_cache(args)
        try:
            val_bars = await _fetch_all(load_settings(), validate, args.months, args.end_date, args.start_date)
            settings, tuning = _build_config(best_overrides)
            trades, per_ticker = await _run_config(
                lambda ticker: (settings, tuning), validate, args.months, val_bars, vcache,
                start_date=args.start_date, end_date=args.end_date,
            )
        finally:
            if vcache:
                vcache.close()
        print("\n== validation (held-out tickers) ==")
        for ticker, agg in per_ticker.items():
            _print_aggregate(f"  {ticker}", agg)
        _print_aggregate("VALIDATION AGGREGATE", aggregate_stats(trades), cap=args.max_dd_cap)
    return 0


async def _optimize_industries(args: argparse.Namespace) -> int:
    db_path = Path(args.weights_db)
    db = load_weights_db(db_path)

    industries = [name.strip() for name in args.industries.split(",") if name.strip()]
    if len(industries) == 1 and industries[0].lower() == "all":
        industries = list(INDUSTRY_STOCKS)
    for name in industries:
        if name not in INDUSTRY_STOCKS:
            print(f"Unknown industry {name!r}; available: {sorted(INDUSTRY_STOCKS)}", file=sys.stderr)
            return 2

    news_cache = _open_news_cache(args)
    try:
        for industry in industries:
            tickers = INDUSTRY_STOCKS[industry]
            print(f"\n== optimize-industries: {industry} ({len(tickers)} tickers, {args.months} months) ==")
            bars = await _fetch_all(load_settings(), tickers, args.months, args.end_date, args.start_date)
            available = [t for t in tickers if t in bars]
            if not available:
                print(f"  ! no bars fetched for {industry}; skipping", file=sys.stderr)
                continue
            result = await _optimize_universe(
                available, args.months, args.n_trials, bars, news_cache,
                args.min_trades, args.max_dd_cap, args.start_date, args.end_date,
                args.progress_file,
            )
            if result is None:
                return 2
            best_overrides, best_value = result
            db[industry] = best_overrides
            save_weights_db(db_path, db)
            print(f"\n== {industry} best config (train score={best_value:.3f}) ==")
            print(json.dumps(best_overrides, indent=2, sort_keys=True))
            print(f"  -> saved to {db_path}")
    finally:
        if news_cache:
            news_cache.close()
    return 0


async def _optimize_stocks(args: argparse.Namespace) -> int:
    db_path = Path(args.weights_db)
    db = load_weights_db(db_path)
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]

    ticker_db = db.setdefault(TICKER_NAMESPACE, {})
    news_cache = _open_news_cache(args)
    try:
        for ticker in tickers:
            print(f"\n== optimize-stocks: {ticker} ({args.months} months) ==")
            bars = await _fetch_all(load_settings(), [ticker], args.months, args.end_date, args.start_date)
            if ticker not in bars:
                print(f"  ! no bars fetched for {ticker}; skipping", file=sys.stderr)
                continue
            result = await _optimize_universe(
                [ticker], args.months, args.n_trials, bars, news_cache,
                args.min_trades, args.max_dd_cap, args.start_date, args.end_date,
                args.progress_file,
            )
            if result is None:
                return 2
            best_overrides, best_value = result
            ticker_db[ticker] = best_overrides
            save_weights_db(db_path, db)
            print(f"\n== {ticker} best config (train score={best_value:.3f}) ==")
            print(json.dumps(best_overrides, indent=2, sort_keys=True))
            print(f"  -> saved to {db_path} [{TICKER_NAMESPACE}.{ticker}]")
    finally:
        if news_cache:
            news_cache.close()
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
        end_date=dict(default=None,
                      help="window end date YYYY-MM-DD (default: today) â€” for historical train/test"),
        start_date=dict(default=None,
                        help="window start date YYYY-MM-DD (default: derived from months)"),
    )
    presets = dict(preset=dict(action="append", default=[], help="named preset override (repeatable)"),
                   set=dict(action="append", default=[], dest="set",
                            help="key=value (fix) or key=v1,v2 (sweep axis); repeatable"))
    cache = dict(news_cache=dict(default=str(data_path("news_cache.db")),
                                 help="path to a built news-sentiment cache (default data/news_cache.db)"))

    for name in ("evaluate", "sweep"):
        p = sub.add_parser(name)
        for flag, kw in common.items():
            p.add_argument(f"--{flag.replace('_', '-')}", **kw)
        for flag, kw in presets.items():
            p.add_argument(f"--{flag.replace('_', '-')}", **kw)
        for flag, kw in cache.items():
            p.add_argument(f"--{flag.replace('_', '-')}", **kw)

    p = next(sp for sp in sub.choices.values() if sp.prog.endswith("evaluate"))
    p.add_argument("--config", default=None, help="JSON file of overrides (optional)")
    p.add_argument("--weights-db", default=str(data_path("weights_db.json")),
                   help="path to per-industry weights JSON (default data/weights_db.json); "
                        "resolves each ticker's industry/stock config")
    p.add_argument("--journal", default=None,
                   help="persist all simulated trades to a SQLite journal at this path "
                        "(default: in-memory, not written)")

    p = next(sp for sp in sub.choices.values() if sp.prog.endswith("sweep"))
    p.add_argument("--grid", default=None, help="JSON file mapping knob -> list of values (optional)")

    p = sub.add_parser("optimize")
    for flag, kw in common.items():
        p.add_argument(f"--{flag.replace('_', '-')}", **kw)
    for flag, kw in cache.items():
        p.add_argument(f"--{flag.replace('_', '-')}", **kw)
    p.add_argument("--n-trials", type=int, default=100, help="number of Optuna trials")
    p.add_argument("--min-trades", type=int, default=30, help="min trades before the loss penalizes")
    p.add_argument("--validate-universe", default=None,
                   help="held-out tickers to validate the best config on (comma-separated)")
    p.add_argument("--progress-file", default=None,
                   help="write {done, total, message} JSON after each trial (for dashboards)")

    p = sub.add_parser("optimize-industries")
    p.add_argument("--industries", default="all",
                   help="comma-separated industry names from weights_db.INDUSTRY_STOCKS, or 'all'")
    p.add_argument("--weights-db", default=str(data_path("weights_db.json")),
                   help="path to weights_db.json to read/write (default data/weights_db.json)")
    p.add_argument("--months", type=int, default=12, help="lookback months per ticker")
    p.add_argument("--max-dd-cap", type=float, default=0.15,
                   help="fraction of starting equity treated as max-drawdown cap")
    p.add_argument("--end-date", default=None,
                   help="window end date YYYY-MM-DD (default: today) â€” for historical train/test")
    p.add_argument("--start-date", default=None,
                   help="window start date YYYY-MM-DD (default: derived from months)")
    p.add_argument("--news-cache", default=str(data_path("news_cache.db")),
                   help="path to a built news-sentiment cache (default data/news_cache.db)")
    p.add_argument("--n-trials", type=int, default=100, help="number of Optuna trials per industry")
    p.add_argument("--min-trades", type=int, default=30, help="min trades before the loss penalizes")
    p.add_argument("--progress-file", default=None,
                   help="write {done, total, message} JSON after each trial (for dashboards)")

    p = sub.add_parser("optimize-stocks")
    p.add_argument("--tickers", required=True,
                   help="comma-separated tickers to tune individually (each written to the DB)")
    p.add_argument("--weights-db", default=str(data_path("weights_db.json")),
                   help="path to weights_db.json to read/write (default data/weights_db.json)")
    p.add_argument("--months", type=int, default=12, help="lookback months per ticker")
    p.add_argument("--max-dd-cap", type=float, default=0.15,
                   help="fraction of starting equity treated as max-drawdown cap")
    p.add_argument("--end-date", default=None,
                   help="window end date YYYY-MM-DD (default: today) â€” for historical train/test")
    p.add_argument("--start-date", default=None,
                   help="window start date YYYY-MM-DD (default: derived from months)")
    p.add_argument("--news-cache", default=str(data_path("news_cache.db")),
                   help="path to a built news-sentiment cache (default data/news_cache.db)")
    p.add_argument("--n-trials", type=int, default=100, help="number of Optuna trials per ticker")
    p.add_argument("--min-trades", type=int, default=5,
                   help="min trades before the loss penalizes (single tickers trade less)")
    p.add_argument("--progress-file", default=None,
                   help="write {done, total, message} JSON after each trial (for dashboards)")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "evaluate":
        return asyncio.run(_evaluate(args))
    if args.command == "sweep":
        return asyncio.run(_sweep(args))
    if args.command == "optimize":
        return asyncio.run(_optimize(args))
    if args.command == "optimize-industries":
        return asyncio.run(_optimize_industries(args))
    if args.command == "optimize-stocks":
        return asyncio.run(_optimize_stocks(args))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
