"""HistoricalAgent integration tests (respx-mocked endpoints)."""

import respx

from agents.historical_agent import HistoricalAgent
from tests.conftest import (
    historical_bars,
    mock_historical,
    mock_historical_error,
)


@respx.mock
async def test_historical_agent_happy_path(settings):
    mock_historical("NVDA")
    agent = HistoricalAgent(settings, days_back=120)

    result = await agent.run("NVDA")

    assert result.status == "ok"
    assert result.symbol == "NVDA"
    assert result.bars_count > 0
    assert result.historical_trends["trend"] in (
        "strong_uptrend", "uptrend", "weak_uptrend", "ranging",
        "weak_downtrend", "downtrend", "strong_downtrend",
    )
    assert result.historical_trends["current_price"] > 0
    assert result.historical_trends["earnings_note"] != ""
    assert len(result.volatility_history) > 0
    assert len(result.dividends) == 3
    assert result.summary["overall_signal"] in (
        "strong_bullish", "bullish", "neutral", "bearish", "strong_bearish"
    )
    assert result.summary["events"] is not None
    assert isinstance(result.technical["calculate_rsi"]["rsi"], float)
    assert isinstance(result.risk["calculate_drawdown"]["max_drawdown"], float)
    assert isinstance(result.levels["resistance_levels"], list)
    assert result.patterns == [] or isinstance(result.patterns[0], dict)
    assert result.errors == []


@respx.mock
async def test_historical_agent_custom_bars_deterministic(settings):
    # Strong one-sided uptrend -> bullish summary, no crash.
    bars = []
    price = 100.0
    for i in range(120):
        price *= 1.003
        bars.append(
            {
                "t": f"2024-{i // 28 + 1:02d}-{i % 28 + 1:02d}T13:30:00Z",
                "o": price / 1.003,
                "h": price * 1.001,
                "l": price / 1.003 * 0.999,
                "c": price,
                "v": 1_000_000 + i,
            }
        )
    mock_historical("NVDA", bars={"bars": bars, "symbol": "NVDA"})
    agent = HistoricalAgent(settings, days_back=120)

    result = await agent.run("NVDA")

    assert result.status == "ok"
    assert result.historical_trends["trend"] in ("uptrend", "strong_uptrend")
    assert result.historical_trends["trend_class"] == "bullish"
    assert result.summary["overall_signal"] == "strong_bullish"
    assert result.summary["bullish_signals"] >= 5


@respx.mock
async def test_historical_agent_api_failure_degrades(settings):
    mock_historical_error(500)
    agent = HistoricalAgent(settings)

    result = await agent.run("NVDA")

    assert result.status == "partial"
    assert result.bars_count == 0
    assert result.errors


@respx.mock
async def test_historical_agent_dividend_failure_does_not_kill(settings):
    import httpx

    from tests.conftest import BARS_URL_RE, CORPORATE_ACTIONS_URL_RE

    respx.get(BARS_URL_RE).mock(
        return_value=httpx.Response(200, json={"bars": historical_bars("NVDA", n=80), "symbol": "NVDA"})
    )
    respx.get(CORPORATE_ACTIONS_URL_RE).mock(
        return_value=httpx.Response(404, json={"message": "not found"})
    )

    agent = HistoricalAgent(settings, days_back=80)
    result = await agent.run("NVDA")

    # Bars succeeded -> analysis still produced; dividends degrade.
    assert result.status == "partial"
    assert result.bars_count == 80
    assert result.dividends == []
    assert any("dividends" in e for e in result.errors)


def test_registry_import_handlers_resolve():
    from tools.registry import import_handlers

    handlers = import_handlers(
        ["get_price_history", "calculate_rsi", "generate_technical_summary", "identify_trend"]
    )
    assert callable(handlers["get_price_history"])
    assert callable(handlers["calculate_rsi"])
    assert callable(handlers["generate_technical_summary"])
    assert callable(handlers["identify_trend"])


def test_registry_import_handlers_all_phases():
    from tools.registry import import_handlers

    handlers = import_handlers(
        [
            "calculate_technical_indicators",
            "forecast_volatility",
            "estimate_price_move",
            "calculate_greeks",
            "calculate_position_size",
            "calculate_max_loss",
            "calculate_risk_score",
            "synthesize_signals",
            "rank_opportunities",
        ]
    )
    for name in [
        "calculate_technical_indicators",
        "forecast_volatility",
        "estimate_price_move",
        "calculate_greeks",
        "calculate_position_size",
        "calculate_max_loss",
        "calculate_risk_score",
        "synthesize_signals",
        "rank_opportunities",
    ]:
        assert callable(handlers[name]), f"{name} handler not resolved"
