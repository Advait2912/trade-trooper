"""Tool registry: name -> spec (schema + handler).

Every deterministic tool in the suite is declared here so agents, the
orchestrator and documentation share one source of truth. Handlers are
imported lazily via ``import_handlers`` so loading a spec never requires an
optional dependency.
"""

from __future__ import annotations

from typing import Any, Callable

_TOOL_HANDLERS: dict[str, Callable[..., Any]] = {}

TOOL_SPECS: dict[str, dict[str, Any]] = {
    # ------------------------------------------------------------------
    # Phase 1 - News Collection Agent
    # ------------------------------------------------------------------
    "fetch_news": {
        "phase": 1,
        "agent": "news_collection",
        "description": "Fetch and deterministically filter Alpaca news for a symbol",
    },
    "sentiment_analysis": {
        "phase": 1,
        "agent": "news_collection",
        "description": "Score sentiment of collected news signals",
    },
    # ------------------------------------------------------------------
    # Phase 1 - Market Data Agent
    # ------------------------------------------------------------------
    "get_current_price": {
        "phase": 1,
        "agent": "market_data",
        "description": "Retrieve the current price for a symbol",
    },
    "get_volatility": {
        "phase": 1,
        "agent": "market_data",
        "description": "Retrieve current volatility for a symbol",
    },
    "get_market_data": {
        "phase": 1,
        "agent": "market_data",
        "description": "Fetch snapshot + daily bars and compute indicators",
    },
    # ------------------------------------------------------------------
    # Phase 1 - Historical Data Agent
    # ------------------------------------------------------------------
    "get_price_history": {
        "phase": 1,
        "agent": "historical",
        "description": "Retrieve historical price data (OHLCV) for a symbol",
    },
    "get_dividends_history": {
        "phase": 1,
        "agent": "historical",
        "description": "Retrieve historical dividend data",
    },
    "get_earnings_history": {
        "phase": 1,
        "agent": "historical",
        "description": "Get historical earnings data and dates (Alpaca-unavailable stub)",
    },
    "get_volatility_history": {
        "phase": 1,
        "agent": "historical",
        "description": "Get historical realized volatility over time",
    },
    "calculate_moving_averages": {
        "phase": 1,
        "agent": "historical",
        "description": "Calculate simple and exponential moving averages",
    },
    "calculate_rsi": {
        "phase": 1,
        "agent": "historical",
        "description": "Calculate Relative Strength Index (momentum indicator)",
    },
    "calculate_macd": {
        "phase": 1,
        "agent": "historical",
        "description": "Calculate MACD (trend-following momentum indicator)",
    },
    "calculate_bollinger_bands": {
        "phase": 1,
        "agent": "historical",
        "description": "Calculate Bollinger Bands (volatility indicator)",
    },
    "calculate_stochastic": {
        "phase": 1,
        "agent": "historical",
        "description": "Calculate Stochastic Oscillator (%K, %D)",
    },
    "calculate_atr": {
        "phase": 1,
        "agent": "historical",
        "description": "Calculate Average True Range (volatility measure)",
    },
    "calculate_adx": {
        "phase": 1,
        "agent": "historical",
        "description": "Calculate ADX (trend strength indicator)",
    },
    "calculate_obv": {
        "phase": 1,
        "agent": "historical",
        "description": "Calculate On-Balance Volume (volume indicator)",
    },
    "identify_support_resistance": {
        "phase": 1,
        "agent": "historical",
        "description": "Identify key support and resistance levels",
    },
    "identify_trend": {
        "phase": 1,
        "agent": "historical",
        "description": "Identify current trend direction and strength",
    },
    "detect_chart_patterns": {
        "phase": 1,
        "agent": "historical",
        "description": "Detect common chart patterns (head & shoulders, triangles, etc.)",
    },
    "calculate_historical_volatility": {
        "phase": 1,
        "agent": "historical",
        "description": "Calculate realized volatility from historical prices",
    },
    "detect_volatility_regimes": {
        "phase": 1,
        "agent": "historical",
        "description": "Detect if volatility is in expansion or contraction phase",
    },
    "analyze_mean_reversion": {
        "phase": 1,
        "agent": "historical",
        "description": "Analyze if price is mean-reverting or trending",
    },
    "calculate_correlation": {
        "phase": 1,
        "agent": "historical",
        "description": "Calculate correlation between two assets",
    },
    "calculate_drawdown": {
        "phase": 1,
        "agent": "historical",
        "description": "Calculate maximum drawdown and recovery time",
    },
    "analyze_gaps": {
        "phase": 1,
        "agent": "historical",
        "description": "Analyze price gaps in history",
    },
    "calculate_value_at_risk": {
        "phase": 1,
        "agent": "historical",
        "description": "Calculate Value at Risk (VaR) for historical data",
    },
    "calculate_returns": {
        "phase": 1,
        "agent": "historical",
        "description": "Calculate returns over various periods",
    },
    "generate_technical_summary": {
        "phase": 1,
        "agent": "historical",
        "description": "Generate comprehensive technical analysis summary",
    },
    "identify_trading_events": {
        "phase": 1,
        "agent": "historical",
        "description": "Identify significant trading events (earnings gaps, vol spikes, etc.)",
    },
    # ------------------------------------------------------------------
    # Phase 2 - Prediction Agent (placeholder)
    # ------------------------------------------------------------------
    "calculate_technical_indicators": {
        "phase": 2,
        "agent": "prediction",
        "description": "Calculate technical indicators for prediction",
    },
    "forecast_volatility": {
        "phase": 2,
        "agent": "prediction",
        "description": "Forecast volatility",
    },
    "estimate_price_move": {
        "phase": 2,
        "agent": "prediction",
        "description": "Estimate expected price move",
    },
    # ------------------------------------------------------------------
    # Phase 3 - Risk Agent (placeholder)
    # ------------------------------------------------------------------
    "calculate_greeks": {
        "phase": 3,
        "agent": "risk",
        "description": "Calculate option greeks",
    },
    "calculate_position_size": {
        "phase": 3,
        "agent": "risk",
        "description": "Calculate position size",
    },
    "calculate_max_loss": {
        "phase": 3,
        "agent": "risk",
        "description": "Calculate maximum loss",
    },
    # ------------------------------------------------------------------
    # Phase 4 - Decision Agent (placeholder)
    # ------------------------------------------------------------------
    "synthesize_signals": {
        "phase": 4,
        "agent": "decision",
        "description": "Synthesize all signals into one view",
    },
    "rank_opportunities": {
        "phase": 4,
        "agent": "decision",
        "description": "Rank trading opportunities",
    },
}


def get_tool(name: str) -> dict[str, Any]:
    """Return the spec for a tool by name (raises KeyError if unknown)."""
    return TOOL_SPECS[name]


def import_handlers(tool_names: list[str] | None = None) -> dict[str, Callable[..., Any]]:
    """Lazily import handler functions and return the name -> callable map."""
    import importlib

    module_map = {
        "fetch_news": "tools.news_tools",
        "sentiment_analysis": "tools.news_tools",
        "get_current_price": "tools.market_data_tools",
        "get_volatility": "tools.market_data_tools",
        "get_market_data": "tools.market_data_tools",
        "get_price_history": "tools.historical.data_tools",
        "get_dividends_history": "tools.historical.data_tools",
        "get_earnings_history": "tools.historical.data_tools",
        "get_volatility_history": "tools.historical.data_tools",
        "calculate_moving_averages": "tools.historical.indicators",
        "calculate_rsi": "tools.historical.indicators",
        "calculate_macd": "tools.historical.indicators",
        "calculate_bollinger_bands": "tools.historical.indicators",
        "calculate_stochastic": "tools.historical.indicators",
        "calculate_atr": "tools.historical.indicators",
        "calculate_adx": "tools.historical.indicators",
        "calculate_obv": "tools.historical.indicators",
        "identify_support_resistance": "tools.historical.levels",
        "identify_trend": "tools.historical.levels",
        "detect_chart_patterns": "tools.historical.levels",
        "calculate_historical_volatility": "tools.historical.volatility",
        "detect_volatility_regimes": "tools.historical.volatility",
        "analyze_mean_reversion": "tools.historical.volatility",
        "calculate_correlation": "tools.historical.volatility",
        "calculate_drawdown": "tools.historical.risk_stats",
        "analyze_gaps": "tools.historical.risk_stats",
        "calculate_value_at_risk": "tools.historical.risk_stats",
        "calculate_returns": "tools.historical.risk_stats",
        "generate_technical_summary": "tools.historical.summary",
        "identify_trading_events": "tools.historical.events",
    }

    wanted = list(tool_names) if tool_names is not None else list(TOOL_SPECS)
    for name in wanted:
        module = module_map.get(name)
        if module is None:
            continue
        mod = importlib.import_module(module)
        fn = getattr(mod, name, None)
        if callable(fn):
            _TOOL_HANDLERS[name] = fn
    return _TOOL_HANDLERS
