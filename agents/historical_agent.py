"""Phase 1 - Historical Data Agent.

Collects price/dividend history from Alpaca, computes rolling realized
volatility, then runs the deterministic analysis tools (indicators, levels,
volatility, risk stats, events, summary) via the tool suite. Returns a
``HistoricalAgentResult`` consumed by the Phase 2 Prediction Agent.

Data availability note: Alpaca has **no earnings endpoint**, so
``get_earnings_history`` is a documented stub; earnings-like signals are
inferred from bar signatures (gaps, volume surges) by ``identify_trading_events``.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, cast

from agents.base import BaseAgent
from alpaca.client import AlpacaClient, AlpacaError
from alpaca.historical import (
    get_dividends_history,
    get_earnings_history,
    get_price_history,
    get_volatility_history,
)
from schemas.historical import (
    DividendRecord,
    EarningsRecord,
    HistoricalAgentResult,
    PriceBar,
    VolatilityPoint,
)
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
    detect_chart_patterns,
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
from utils.config import Settings
from utils.logging import StageTimer

log = logging.getLogger("market_intel_agent.historical_agent")

DEFAULT_DAYS_BACK = 252
DEFAULT_INTERVAL = "1d"
DEFAULT_VOL_WINDOW = 20


class HistoricalAgent(BaseAgent):
    """Collects historical data and computes trends/volatility (Phase 1)."""

    name = "historical"
    description = (
        "Fetches price/dividend history from Alpaca and computes historical "
        "trends, volatility history, technical indicators and risk stats."
    )
    phase = 1
    tools = [
        "get_price_history",
        "get_dividends_history",
        "get_earnings_history",
        "get_volatility_history",
        "get_returns",
    ] + [
        "calculate_moving_averages",
        "calculate_rsi",
        "calculate_macd",
        "calculate_bollinger_bands",
        "calculate_atr",
        "calculate_adx",
        "calculate_obv",
        "identify_support_resistance",
        "identify_trend",
        "detect_chart_patterns",
        "calculate_historical_volatility",
        "detect_volatility_regimes",
        "analyze_mean_reversion",
        "calculate_drawdown",
        "analyze_gaps",
        "calculate_value_at_risk",
        "generate_technical_summary",
        "identify_trading_events",
    ]

    def __init__(
        self,
        settings: Settings,
        days_back: int = DEFAULT_DAYS_BACK,
        interval: str = DEFAULT_INTERVAL,
        vol_window: int = DEFAULT_VOL_WINDOW,
    ) -> None:
        self.settings = settings
        self.days_back = days_back
        self.interval = interval
        self.vol_window = vol_window

    async def run(self, ticker: str) -> HistoricalAgentResult:
        """Run all data collection (parallel) then compute the analysis bundle."""
        errors: list[str] = []

        bars: list[PriceBar] | None = None
        dividends: list[DividendRecord] = []
        earnings: list[EarningsRecord] = []
        vol_points: list[VolatilityPoint] = []
        try:
            with StageTimer("Historical data fetched", log):
                async with AlpacaClient(self.settings) as client:
                    outcomes = await asyncio.gather(
                        get_price_history(
                            client,
                            ticker,
                            days_back=self.days_back,
                            interval=self.interval,
                            feed=self.settings.alpaca_data_feed,
                        ),
                        get_dividends_history(
                            client,
                            ticker,
                            years_back=5,
                            feed=self.settings.alpaca_data_feed,
                        ),
                        get_earnings_history(client, ticker, quarters=8),
                        get_volatility_history(
                            client,
                            ticker,
                            days_back=self.days_back,
                            period=self.vol_window,
                            feed=self.settings.alpaca_data_feed,
                        ),
                        return_exceptions=True,
                    )
        except AlpacaError as exc:
            log.warning("Historical data fetch failed: %s", exc)
            errors.append(f"data fetch failed: {exc}")
            outcomes = [None, [], [], []]

        names = ("price_history", "dividends", "earnings", "volatility_history")
        for index, name in enumerate(names):
            outcome = outcomes[index] if index < len(outcomes) else None
            if isinstance(outcome, BaseException):
                log.warning("%s fetch failed: %s", name, outcome)
                errors.append(f"{name} failed: {outcome}")
                outcome = None
            if name == "price_history":
                bars = cast(list[PriceBar] | None, outcome)
            elif name == "dividends":
                dividends = cast(list[DividendRecord], outcome or [])
            elif name == "earnings":
                earnings = cast(list[EarningsRecord], outcome or [])
            elif name == "volatility_history":
                vol_points = cast(list[VolatilityPoint], outcome or [])

        if not bars:
            return HistoricalAgentResult(
                symbol=ticker,
                status="partial",
                errors=errors or ["No price history returned."],
            )

        with StageTimer("Historical analysis computed", log):
            computed = await asyncio.to_thread(
                _compute_answer, ticker, bars, dividends, earnings, vol_points, self.vol_window
            )

        return HistoricalAgentResult(
            symbol=ticker,
            status="ok" if not errors else "partial",
            bars_count=len(bars),
            historical_trends=computed["historical_trends"],
            volatility_history=vol_points,
            dividends=dividends,
            earnings=earnings,
            technical=computed["technical"],
            volatility=computed["volatility"],
            risk=computed["risk"],
            levels=computed["levels"],
            patterns=computed["patterns"],
            summary=computed["summary"],
            errors=errors,
            # Raw arrays for Phase 2 re-use (same data, no extra fetch)
            closes=computed["closes"],
            highs=computed["highs"],
            lows=computed["lows"],
            volumes=computed["volumes"],
        )


def _compute_answer(
    symbol: str,
    bars: list[Any],
    dividends: list[Any],
    earnings: list[Any],
    vol_points: list[Any],
    vol_window: int,
) -> dict[str, Any]:
    """Synchronous computation bundle (runs in a worker thread)."""
    closes = [float(b.close) for b in bars]
    highs = [float(b.high) for b in bars]
    lows = [float(b.low) for b in bars]
    volumes = [int(b.volume) for b in bars]
    bar_dicts = [
        {
            "date": b.date,
            "open": b.open,
            "high": b.high,
            "low": b.low,
            "close": b.close,
        }
        for b in bars
    ]
    vol_series = [
        float(p.rolling_vol_20d)
        for p in vol_points
        if float(p.rolling_vol_20d) > 0
    ]

    technical: dict[str, Any] = {
        "calculate_moving_averages": calculate_moving_averages(closes),
        "calculate_rsi": calculate_rsi(closes),
        "calculate_macd": calculate_macd(closes),
        "calculate_bollinger_bands": calculate_bollinger_bands(closes),
        "calculate_atr": calculate_atr(highs, lows, closes),
        "calculate_adx": calculate_adx(highs, lows, closes),
        "calculate_obv": calculate_obv(closes, volumes),
    }

    levels_result = identify_support_resistance(bar_dicts)
    patterns_result = detect_chart_patterns(bar_dicts)
    trend_result = identify_trend(closes)
    gap_result = analyze_gaps(bar_dicts)
    returns_result = calculate_returns(closes)

    hist_vol = calculate_historical_volatility(closes, vol_window)
    regime_result = detect_volatility_regimes(vol_series or [float(hist_vol["historical_vol"])] * 30)
    reversion_result = analyze_mean_reversion(closes)

    volatility_bundle: dict[str, Any] = {
        "calculate_historical_volatility": hist_vol,
        "detect_volatility_regimes": regime_result,
        "analyze_mean_reversion": reversion_result,
    }

    all_indicators: dict[str, Any] = {
        **technical,
        "identify_support_resistance": levels_result,
        "identify_trend": trend_result,
    }
    summary_result = generate_technical_summary(symbol, all_indicators)

    returns_history = [
        (closes[i] / closes[i - 1]) - 1.0
        for i in range(1, len(closes))
        if closes[i - 1] > 0
    ]
    risk_bundle: dict[str, Any] = {
        "calculate_drawdown": calculate_drawdown(closes),
        "analyze_gaps": gap_result,
        "calculate_value_at_risk": calculate_value_at_risk(returns_history),
        "calculate_returns": returns_result,
    }

    events_result = identify_trading_events(
        bar_dicts, volumes, earnings_dates=[]  # Alpaca has no earnings calendar
    )

    dd = risk_bundle["calculate_drawdown"]
    historical_trends: dict[str, Any] = {
        "trend": trend_result["trend"],
        "trend_class": trend_result["trend_class"],
        "trend_strength": trend_result["trend_strength"],
        "angle": trend_result["angle"],
        "days_in_trend": trend_result["days_in_trend"],
        "returns": returns_result["return_1w"],
        "returns_1m": returns_result["return_1m"],
        "returns_3m": returns_result["return_3m"],
        "returns_1y": returns_result["return_1y"],
        "max_drawdown": dd["max_drawdown"],
        "current_drawdown": dd["current_drawdown"],
        "historical_vol": hist_vol["historical_vol"],
        "vol_percentile": hist_vol["vol_percentile"],
        "mean_reversion_score": reversion_result["mean_reversion_score"],
        "current_price": closes[-1] if closes else 0.0,
        "earnings_note": "Earnings data is not available from Alpaca; "
        "earnings-like events are inferred from bar signatures.",
    }

    return {
        "historical_trends": historical_trends,
        "technical": technical,
        "volatility": volatility_bundle,
        "risk": risk_bundle,
        "levels": levels_result,
        "patterns": patterns_result["patterns_found"],
        "summary": {**summary_result, "events": events_result["events"]},
        # Raw arrays passed through for Phase 2 use
        "closes": closes,
        "highs": highs,
        "lows": lows,
        "volumes": volumes,
    }
