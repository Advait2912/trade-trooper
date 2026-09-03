"""Phase 5 - Historical backtest engine.

Replays the **deterministic** Phase 2-4 chain over historical daily bars.
No LLM and no live option chain: Phase 3 uses the same Black-Scholes fallback
the live path uses when the options feed is unavailable.

For each bar (after warm-up) it builds a ``Phase1Bundle`` from the window of
bars, then reuses the *actual* production functions:

    PredictionAgent.run  -> PredictionResult
    RiskAgent._compute_risk(chain=[]) -> RiskResult   (BS fallback)
    DecisionAgent._decide -> DecisionResult

If the decision is a trade, it fills at the next bar's open + slippage and then
walks forward checking intrabar high/low against stop/target, closing at the
exit bar's open; a position also closes when the horizon is reached.

Assumption / limitation (documented): the news-sentiment leg is LLM-based and
cannot be cheaply replayed for every historical day, so the backtest runs
news-neutral (``adjusted_momentum = momentum``). The forward paper run captures
the news effect; the backtest quantifies the pure technical/prediction edge.
Fills are simulated (next-open + slippage); commissions/fees are not modeled.
"""

from __future__ import annotations

import logging
from typing import Any

from agents.decision_agent import _decide as _decision_decide
from agents.risk_agent import _compute_risk as _risk_compute
from alpaca.client import AlpacaClient
from alpaca.historical import get_price_history
from schemas.historical import HistoricalAgentResult
from schemas.market import MarketData
from schemas.news import NewsCollectionResult
from schemas.prediction import Phase1Bundle, PredictionResult
from tools.historical.events import identify_trading_events
from tools.historical.indicators import (
    calculate_adx,
    calculate_atr,
    calculate_bollinger_bands,
    calculate_macd,
    calculate_moving_averages,
    calculate_obv,
    calculate_rsi,
)
from tools.historical.levels import (
    identify_support_resistance,
    identify_trend,
)
from tools.historical.risk_stats import (
    analyze_gaps,
    calculate_drawdown,
    calculate_returns,
    calculate_value_at_risk,
)
from tools.historical.summary import generate_technical_summary
from tools.historical.volatility import (
    analyze_mean_reversion,
    calculate_historical_volatility,
    detect_volatility_regimes,
)
from tools.prediction_tools.price_move import estimate_price_move
from tools.prediction_tools.technical import calculate_technical_indicators
from tools.prediction_tools.volatility_forecast import forecast_volatility
from trading.journal import TradeJournal
from utils.config import Settings

log = logging.getLogger("market_intel_agent.backtest")

_WARMUP = 260
_SLIPPAGE = 0.0005
_DAY = 252
_START_EQUITY = 100_000.0


def _predict(bundle: Phase1Bundle) -> PredictionResult:
    """News-neutral Phase 2 replica (no LLM)."""
    tech = calculate_technical_indicators(bundle.historical.technical or {})
    vol = forecast_volatility(
        bundle.historical.volatility or {}, bundle.historical.historical_trends or {}
    )
    move = estimate_price_move(
        market=bundle.market,
        adjusted_momentum=tech["momentum_score"],
        adx_trend_strength=tech["adx_trend_strength"],
        vol_regime=vol["vol_regime"],
        mean_reversion_score=vol["mean_reversion_score"],
    )
    return PredictionResult(
        price_forecast=move["price_forecast"],
        price_forecast_low=move["price_forecast_low"],
        price_forecast_high=move["price_forecast_high"],
        expected_move_pct=move["expected_move_pct"],
        forecast_horizon_days=move["forecast_horizon_days"],
        iv_forecast=vol["iv_forecast"],
        vol_regime=vol["vol_regime"],
        vol_percentile=vol["vol_percentile"],
        composite_signal=tech["composite_signal"],
        momentum_score=tech["momentum_score"],
        adjusted_momentum=tech["momentum_score"],
        confidence=move["confidence"],
        rsi_signal=tech["rsi_signal"],
        macd_signal=tech["macd_signal"],
        adx_trend_strength=tech["adx_trend_strength"],
        adx_trend_direction=tech["adx_trend_direction"],
        bollinger_regime=tech["bollinger_regime"],
        obv_confirmation=tech["obv_confirmation"],
        mean_reversion_score=vol["mean_reversion_score"],
    )


def _hist_from_bars(bars: list[Any]) -> dict[str, Any]:
    """Compute the Phase 1 historical bundle for a window of bars."""
    closes = [float(b.close) for b in bars]
    highs = [float(b.high) for b in bars]
    lows = [float(b.low) for b in bars]
    volumes = [int(b.volume) for b in bars]
    bar_dicts = [{"date": b.date, "open": b.open, "high": b.high,
                  "low": b.low, "close": b.close} for b in bars]

    tech = {
        "calculate_moving_averages": calculate_moving_averages(closes),
        "calculate_rsi": calculate_rsi(closes),
        "calculate_macd": calculate_macd(closes),
        "calculate_bollinger_bands": calculate_bollinger_bands(closes),
        "calculate_atr": calculate_atr(highs, lows, closes),
        "calculate_adx": calculate_adx(highs, lows, closes),
        "calculate_obv": calculate_obv(closes, volumes),
    }
    levels = identify_support_resistance(bar_dicts)
    trend = identify_trend(closes)
    summary = generate_technical_summary("", {**tech, "identify_support_resistance": levels,
                                              "identify_trend": trend})

    return {
        "levels": levels,
        "trend": trend,
        "technical": tech,
        "summary": summary,
        "events": identify_trading_events(bar_dicts, volumes)["events"],
        "gaps": analyze_gaps(bar_dicts),
        "drawdown": calculate_drawdown(closes),
        "var": calculate_value_at_risk([(closes[i] / closes[i - 1]) - 1.0
                                        for i in range(1, len(closes)) if closes[i - 1] > 0]),
        "hist_vol": calculate_historical_volatility(closes, 20),
        "regime": detect_volatility_regimes([float(calculate_historical_volatility(closes, 20)["historical_vol"])] * 30),
        "reversion": analyze_mean_reversion(closes),
        "returns": calculate_returns(closes),
    }


def _bundle(ticker: str, bars: list[Any], hist: dict[str, Any]) -> Phase1Bundle:
    spot = float(bars[-1].close)
    atr = float(hist["technical"]["calculate_atr"]["atr"] or 0.0)
    return Phase1Bundle(
        news=NewsCollectionResult(ticker=ticker, sentiment_score=0.0),
        market=MarketData(price=spot, atr14=atr),
        historical=HistoricalAgentResult(
            symbol=ticker, bars_count=len(bars),
            levels=hist["levels"],
            technical=hist["technical"],
            volatility={
                "calculate_historical_volatility": hist["hist_vol"],
                "detect_volatility_regimes": hist["regime"],
                "analyze_mean_reversion": hist["reversion"],
            },
            risk={
                "calculate_drawdown": hist["drawdown"],
                "analyze_gaps": hist["gaps"],
                "calculate_value_at_risk": hist["var"],
            },
            summary=hist["summary"],
            historical_trends={
                "trend": hist["trend"]["trend"],
                "trend_class": hist["trend"]["trend_class"],
            },
        ),
    )


def _check_exit(active: dict[str, Any], bar: Any) -> tuple[bool, bool, float, str]:
    """Return (stop_hit, target_hit, exit_price, reason) for the current bar."""
    stop = float(active["stop"] or 0.0)
    target = float(active["target"] or 0.0)
    high = float(bar.high)
    low = float(bar.low)
    if active["decision"] in ("long_call", "long_equity"):
        if stop > 0 and low <= stop:
            return True, False, stop, "stop"
        if target > 0 and high >= target:
            return False, True, target, "target"
    elif active["decision"] == "long_put":
        if stop > 0 and high >= stop:
            return True, False, stop, "stop"
        if target > 0 and low <= target:
            return False, True, target, "target"
    return False, False, float(bar.close), "horizon"


async def _fetch_bars(settings: Settings, ticker: str, days_back: int) -> list[Any]:
    async with AlpacaClient(settings) as client:
        return await get_price_history(client, ticker, days_back=days_back,
                                       interval="1d", feed=settings.alpaca_data_feed)


async def run_backtest(
    settings: Settings, ticker: str, months: int = 6, journal: TradeJournal | None = None,
) -> dict[str, Any]:
    """Run a deterministic backtest and journal every cycle + simulated trade."""
    journal = journal or TradeJournal("backtest_journal.db")
    days_back = _WARMUP + int(_DAY * max(1, months) / 12.0)
    bars = await _fetch_bars(settings, ticker, days_back)

    if len(bars) < _WARMUP + 10:
        return {"summary": f"Insufficient history for {ticker} ({len(bars)} bars).",
                "trades": 0, "journal": str(journal.path)}

    active: dict[str, Any] | None = None
    trades = 0
    entry_ts = ""

    for i in range(_WARMUP, len(bars) - 1):
        window = bars[: i + 1]
        nxt = bars[i + 1]
        if active is None:
            hist = _hist_from_bars(window)
            bundle = _bundle(ticker, window, hist)
            prediction = _predict(bundle)
            risk = _risk_compute(bundle, prediction, [], settings, [])
            decision = _decision_decide(bundle, prediction, risk, settings, [])

            journal.record_cycle(str(window[-1].date), ticker, decision.trade_decision,
                                 decision.composite_bias, decision.confidence_score,
                                 {"price": float(window[-1].close), "iter": i})

            if decision.trade_decision in ("long_equity", "long_call", "long_put"):
                entry_price = float(nxt.open) * (1.0 + _SLIPPAGE)
                active = {
                    "symbol": ticker, "entry": entry_price, "entry_i": i + 1,
                    "stop": decision.stop_loss, "target": decision.take_profit,
                    "decision": decision.trade_decision,
                    "qty_shares": decision.position_shares,
                    "qty_contracts": decision.option_contracts,
                }
                entry_ts = str(nxt.date)
                qty = max(active["qty_shares"], active["qty_contracts"])
                journal.record_order(cycle_ts=str(window[-1].date),
                                     client_order_id=f"bt-{i}", symbol=ticker,
                                     side="buy", qty=qty, order_type="market",
                                     limit_price=None, status="filled",
                                     filled_avg_price=entry_price, order_id=None, reason="entry")
        else:
            stop_hit, target_hit, exit_price, reason = _check_exit(active, bars[i + 1])
            horizon_reached = (i + 1 - active["entry_i"]) >= max(1, settings.trade_horizon_days)
            if stop_hit or target_hit or horizon_reached:
                if not (stop_hit or target_hit):
                    exit_price = float(bars[i + 1].open)  # close at open on horizon
                exit_px = round(exit_price * (1.0 - _SLIPPAGE), 4)
                instrument = "option" if active["decision"] in ("long_call", "long_put") else "equity"
                qty = active["qty_contracts"] if instrument == "option" else active["qty_shares"]
                mult = 100.0 if instrument == "option" else 1.0
                pnl = round((exit_px - active["entry"]) * qty * mult, 2)
                journal.record_trade(
                    opened_ts=entry_ts, closed_ts=str(bars[i + 1].date), ticker=ticker,
                    instrument=instrument, option_type=("call" if "call" in active["decision"] else "put") if instrument == "option" else "",
                    symbol=ticker, quantity=qty, entry_price=active["entry"],
                    exit_price=exit_px, pnl=pnl,
                    pnl_pct=round((exit_px - active["entry"]) / active["entry"], 4),
                    exit_reason=reason, cycle_ts=str(bars[i + 1].date),
                )
                trades += 1
                active = None

    ending = _START_EQUITY + sum(t["pnl"] for t in journal.trades())
    return {
        "summary": (
            f"Backtest {ticker} over {len(bars)} bars: {trades} simulated "
            f"trades; equity ${ending:,.0f} (from ${_START_EQUITY:,.0f}). "
            f"Details in {journal.path}."
        ),
        "trades": trades,
        "ending_equity": round(ending, 2),
        "journal": str(journal.path),
    }
