import asyncio
import concurrent.futures
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import optuna

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.tune import _fetch_all
from trading.backtest import (
    _WARMUP, _DAY, _SLIPPAGE,
    _hist_from_bars, _bundle, _predict, _risk_compute, _decision_decide,
    _check_exit, _option_pnl
)
from trading.news_cache import NewsCache
from tuning import TuningConfig
from utils.config import Settings, load_settings
from utils.paths import data_path
from weights_db import INDUSTRY_STOCKS

SECTOR_ANCHORS = {
    "Technology": ["NVDA", "AAPL", "MSFT", "AMD"],
    "Financials": ["BAC", "JPM", "GS", "V"],
    "Consumer Discretionary": ["TSLA", "AMZN", "NKE"],
    "Energy": ["XOM", "CVX", "COP"],
    "Consumer Staples": ["KO", "COST", "PG"],
    "Communication Services": ["GOOGL", "META", "NFLX"],
    "Healthcare": ["LLY", "JNJ", "UNH"],
    "Industrials": ["CAT", "GE", "BA"],
    "Materials": ["FCX", "NUE", "LIN"],
    "Utilities": ["NEE", "DUK", "SO"],
    "Real Estate": ["PLD", "AMT", "SPG"],
}

def _precompute_worker(ticker: str, bars: list[Any], news_cache_path: str):
    cache = NewsCache(news_cache_path)
    daily_map = cache.load_daily_map(ticker)
    cache.close()

    precomputed = []
    for i in range(_WARMUP, len(bars) - 1):
        window = bars[: i + 1]
        cur_date = str(window[-1].date)[:10]
        day_sentiment = daily_map.get(cur_date)
        hist = _hist_from_bars(window)
        bundle = _bundle(ticker, window, hist, day_sentiment)
        precomputed.append((i, cur_date, bundle, day_sentiment))
    return ticker, precomputed

def simulate_tickers(tickers: list[str], bars_dict: dict, precomputed_dict: dict, settings: Settings, tuning: TuningConfig):
    all_trades = []

    for ticker in tickers:
        if ticker not in bars_dict or ticker not in precomputed_dict:
            continue
        bars = bars_dict[ticker]
        items = precomputed_dict[ticker]

        active = None
        entry_ts = ""

        for i, cur_date, bundle, day_sentiment in items:
            nxt = bars[i + 1]
            if active is None:
                prediction = _predict(bundle, tuning, day_sentiment)
                risk = _risk_compute(bundle, prediction, [], settings, [], tuning)
                decision = _decision_decide(bundle, prediction, risk, settings, [], tuning)

                if decision.trade_decision in ("long_call", "long_put"):
                    entry_price = float(nxt.open) * (1.0 + _SLIPPAGE)
                    greeks_info = (risk.risk_metrics or {}).get("calculate_greeks", {}) or {}
                    active = {
                        "symbol": ticker, "entry": entry_price, "entry_i": i + 1,
                        "stop": decision.stop_loss, "target": decision.take_profit,
                        "decision": decision.trade_decision,
                        "qty_shares": decision.position_shares,
                        "qty_contracts": decision.option_contracts,
                        "option_type": decision.option_type,
                        "strike": float(greeks_info.get("strike", 0.0) or 0.0),
                        "iv": float(risk.iv_used or 0.0) / 100.0,
                        "t_total": max(1, settings.trade_horizon_days) / _DAY,
                    }
                    entry_ts = str(nxt.date)
            else:
                stop_hit, target_hit, exit_price, reason = _check_exit(active, bars[i + 1])
                horizon_reached = (i + 1 - active["entry_i"]) >= max(1, settings.trade_horizon_days)
                if stop_hit or target_hit or horizon_reached:
                    if not (stop_hit or target_hit):
                        exit_price = float(bars[i + 1].open)
                    exit_px = round(exit_price * (1.0 - _SLIPPAGE), 4)
                    qty = active["qty_contracts"]
                    pnl, pnl_pct = _option_pnl(active, exit_px, i + 1)
                    all_trades.append({
                        "ticker": ticker, "pnl": pnl, "pnl_pct": pnl_pct,
                        "option_type": active["option_type"],
                    })
                    active = None

    wins = [t for t in all_trades if t["pnl"] > 0]
    losses = [t for t in all_trades if t["pnl"] < 0]
    gw = sum(t["pnl"] for t in wins)
    gl = abs(sum(t["pnl"] for t in losses))
    pf = (gw / gl) if gl > 0 else (99.0 if gw > 0 else 0.0)

    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for t in all_trades:
        equity += t["pnl"]
        peak = max(peak, equity)
        max_dd = min(max_dd, equity - peak)

    net_pnl = sum(t["pnl"] for t in all_trades)
    return {
        "trades": len(all_trades),
        "win_rate": (len(wins) / len(all_trades) * 100.0) if all_trades else 0.0,
        "profit_factor": pf,
        "net_pnl": net_pnl,
        "return_pct": (net_pnl / 100_000.0) * 100.0,
        "max_dd_pct": (max_dd / 100_000.0) * 100.0,
    }

def objective_score(m: dict[str, Any], min_trades: int = 15) -> float:
    if m["trades"] < min_trades:
        return -200.0 + m["trades"]
    pf_score = min(m["profit_factor"], 3.0) / 3.0
    wr_score = m["win_rate"] / 100.0
    ret_score = min(m["return_pct"], 100.0) / 100.0
    score = 0.40 * pf_score + 0.30 * wr_score + 0.30 * ret_score
    dd = abs(m["max_dd_pct"]) / 100.0
    if dd > 0.20:
        score -= (dd - 0.20) * 2.5
    return score

def sample_trial(trial: optuna.Trial) -> tuple[Settings, TuningConfig]:
    s = load_settings()
    s.risk_per_trade_pct = trial.suggest_float("risk_per_trade_pct", 0.020, 0.038)
    s.max_position_pct = 0.15
    s.min_confidence = trial.suggest_float("min_confidence", 0.38, 0.62)
    s.min_risk_reward = trial.suggest_float("min_risk_reward", 0.75, 1.80)
    s.trade_horizon_days = trial.suggest_int("trade_horizon_days", 3, 6)

    sw_keys = ["news_sentiment", "technical_summary", "historical_trend", "prediction_signal", "market_trend"]
    raw_sw = {k: trial.suggest_float(f"sw_{k}", 0.05, 1.0) for k in sw_keys}
    tot_sw = sum(raw_sw.values())
    signal_weights = {k: round(v / tot_sw, 5) for k, v in raw_sw.items()}

    mw_keys = ["macd", "adx", "rsi", "bollinger", "obv", "stochastic", "squeeze"]
    raw_mw = {k: trial.suggest_float(f"mw_{k}", 0.05, 1.0) for k in mw_keys}
    tot_mw = sum(raw_mw.values())
    momentum_weights = {k: round(v / tot_mw, 5) for k, v in raw_mw.items()}

    t = TuningConfig(
        equity_only=False,
        options_only=True,
        signal_weights=signal_weights,
        momentum_weights=momentum_weights,
    )
    return s, t

async def main():
    print("=" * 85, flush=True)
    print("      3-YEAR OPTUNA SECTOR WEIGHTS TUNER (11 SECTORS, PURE OPTIONS)      ", flush=True)
    print("=" * 85, flush=True)

    settings = load_settings()
    news_cache_path = str(data_path("news_cache.db"))

    # Initial DB structure with our proven 3-Year Optuna Global baseline
    db_out = {
        "default": {
            "equity_only": False,
            "options_only": True,
            "risk_per_trade_pct": 0.0328,
            "max_position_pct": 0.15,
            "min_confidence": 0.4131,
            "min_risk_reward": 0.8961,
            "trade_horizon_days": 4,
            "momentum_weights": {
                "bollinger": 0.3340, "macd": 0.1967, "stochastic": 0.1496,
                "adx": 0.1382, "rsi": 0.0841, "squeeze": 0.0627, "obv": 0.0348
            },
            "signal_weights": {
                "news_sentiment": 0.3178, "market_trend": 0.3071,
                "historical_trend": 0.2686, "prediction_signal": 0.0718,
                "technical_summary": 0.0346
            },
        },
        "tickers": {},
    }

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print("\nTraining Optuna TPE sector by sector (low-memory footprint)...", flush=True)
    print("-" * 85, flush=True)
    print(f"{'Sector':<26} | {'Anchors':<20} | {'Trades':<7} | {'Win Rate':<9} | {'PF':<6} | {'3Y P&L ($)':<12}", flush=True)
    print("-" * 85, flush=True)

    for sector, anchors in SECTOR_ANCHORS.items():
        t_sec0 = time.time()
        # Fetch bars for this sector's anchors only
        bars_dict = await _fetch_all(settings, anchors, months=36)
        valid_anchors = [a for a in anchors if a in bars_dict]
        if not valid_anchors:
            continue

        # Precompute daily bundles for this sector with max_workers=4 (lean memory)
        precomputed_dict = {}
        with concurrent.futures.ProcessPoolExecutor(max_workers=min(4, len(valid_anchors))) as executor:
            futures = [
                executor.submit(_precompute_worker, t, bars_dict[t], news_cache_path)
                for t in valid_anchors
            ]
            for f in concurrent.futures.as_completed(futures):
                ticker, pre = f.result()
                precomputed_dict[ticker] = pre

        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        n_trials = 30
        for _ in range(n_trials):
            trial = study.ask()
            s_t, t_t = sample_trial(trial)
            m = simulate_tickers(valid_anchors, bars_dict, precomputed_dict, s_t, t_t)
            score = objective_score(m, min_trades=10)
            study.tell(trial, score)

        # Extract best params
        bp = study.best_trial.params
        sw_keys = ["news_sentiment", "technical_summary", "historical_trend", "prediction_signal", "market_trend"]
        raw_sw = {k: bp[f"sw_{k}"] for k in sw_keys}
        tot_sw = sum(raw_sw.values())
        best_sw = {k: round(v / tot_sw, 5) for k, v in raw_sw.items()}

        mw_keys = ["macd", "adx", "rsi", "bollinger", "obv", "stochastic", "squeeze"]
        raw_mw = {k: bp[f"mw_{k}"] for k in mw_keys}
        tot_mw = sum(raw_mw.values())
        best_mw = {k: round(v / tot_mw, 5) for k, v in raw_mw.items()}

        s_best = load_settings()
        s_best.risk_per_trade_pct = bp["risk_per_trade_pct"]
        s_best.max_position_pct = 0.15
        s_best.min_confidence = bp["min_confidence"]
        s_best.min_risk_reward = bp["min_risk_reward"]
        s_best.trade_horizon_days = bp["trade_horizon_days"]

        t_best = TuningConfig(
            equity_only=False,
            options_only=True,
            signal_weights=best_sw,
            momentum_weights=best_mw,
        )

        res = simulate_tickers(valid_anchors, bars_dict, precomputed_dict, s_best, t_best)

        db_out[sector] = {
            "equity_only": False,
            "options_only": True,
            "risk_per_trade_pct": s_best.risk_per_trade_pct,
            "max_position_pct": 0.15,
            "min_confidence": s_best.min_confidence,
            "min_risk_reward": s_best.min_risk_reward,
            "trade_horizon_days": s_best.trade_horizon_days,
            "momentum_weights": best_mw,
            "signal_weights": best_sw,
        }

        # Save progressively so data is never lost
        out_file = data_path("weights_db.json")
        with open(out_file, "w") as f:
            json.dump(db_out, f, indent=2)

        anchors_str = ",".join(valid_anchors)
        pf_str = f"{res['profit_factor']:.2f}" if res['profit_factor'] < 90 else "Inf"
        print(
            f"{sector:<26} | {anchors_str:<20} | {res['trades']:<7} | {res['win_rate']:>7.1f}% | "
            f"{pf_str:>6} | ${res['net_pnl']:>10,.2f} ({time.time() - t_sec0:.1f}s)",
            flush=True,
        )

    print("-" * 85, flush=True)
    print(f"\nAll 11 sector 3-Year Optuna options configurations saved to {out_file}!", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
