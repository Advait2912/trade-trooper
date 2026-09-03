"""Tests for the industry-scoped weights DB.

Offline only — no HTTP, no LLM, no torch.
"""

from __future__ import annotations

from tools.prediction_tools.technical import calculate_technical_indicators
from tuning import TuningConfig
from weights_db import (
    DEFAULT_INDUSTRY,
    INDUSTRY_STOCKS,
    TICKER_NAMESPACE,
    TICKER_TO_INDUSTRY,
    load_weights_db,
    resolve_industry,
    resolve_overrides,
    save_weights_db,
)


class TestIndustryMapping:

    def test_industry_stocks_integrity(self):
        industries = list(INDUSTRY_STOCKS)
        assert len(industries) == 11
        assert "Technology" in INDUSTRY_STOCKS
        assert "Financials" in INDUSTRY_STOCKS
        all_tickers = [t for tickers in INDUSTRY_STOCKS.values() for t in tickers]
        assert len(all_tickers) == 66
        assert len(all_tickers) == len(set(all_tickers))  # no ticker in two industries

    def test_ticker_to_industry_inverse(self):
        for industry, tickers in INDUSTRY_STOCKS.items():
            for t in tickers:
                assert TICKER_TO_INDUSTRY[t] == industry

    def test_resolve_industry_known_and_fallback(self):
        assert resolve_industry("NVDA") == "Technology"
        assert resolve_industry("nvda") == "Technology"  # case-insensitive
        assert resolve_industry("JPM") == "Financials"
        assert resolve_industry("ZZZZ") == DEFAULT_INDUSTRY
        assert resolve_industry("") == DEFAULT_INDUSTRY


class TestWeightsDbIo:

    def test_save_load_roundtrip(self, tmp_path):
        path = tmp_path / "w.json"
        data = {"Technology": {"equity_only": True}, DEFAULT_INDUSTRY: {"min_confidence": 0.4}}
        save_weights_db(path, data)
        assert load_weights_db(path) == data

    def test_save_load_ticker_namespace_roundtrip(self, tmp_path):
        path = tmp_path / "w.json"
        data = {
            "Technology": {"equity_only": True},
            TICKER_NAMESPACE: {"NVDA": {"min_confidence": 0.30}},
        }
        save_weights_db(path, data)
        loaded = load_weights_db(path)
        assert loaded[TICKER_NAMESPACE]["NVDA"]["min_confidence"] == 0.30
        assert resolve_overrides("NVDA", loaded)["min_confidence"] == 0.30

    def test_load_missing_returns_empty(self, tmp_path):
        assert load_weights_db(tmp_path / "nope.json") == {}


class TestResolveOverrides:

    def test_merge_precedence(self):
        db = {
            DEFAULT_INDUSTRY: {"min_confidence": 0.40, "trade_horizon_days": 7},
            "Technology": {"min_confidence": 0.55},
        }
        merged = resolve_overrides("NVDA", db, base={"equity_only": True})
        assert merged["min_confidence"] == 0.55       # industry wins over default
        assert merged["trade_horizon_days"] == 7      # default fills the gap
        assert merged["equity_only"] is True          # base layer preserved

    def test_unknown_ticker_uses_default(self):
        db = {DEFAULT_INDUSTRY: {"min_confidence": 0.40}}
        assert resolve_overrides("ZZZZ", db)["min_confidence"] == 0.40

    def test_empty_db_returns_base(self):
        assert resolve_overrides("NVDA", {}, base={"equity_only": True}) == {"equity_only": True}

    def test_stock_entry_wins_over_industry(self):
        db = {
            "Technology": {"min_confidence": 0.55, "trade_horizon_days": 7},
            TICKER_NAMESPACE: {"NVDA": {"min_confidence": 0.30}},
        }
        merged = resolve_overrides("NVDA", db)
        assert merged["min_confidence"] == 0.30       # stock beats industry
        assert merged["trade_horizon_days"] == 7      # industry fills the gap

    def test_stock_entry_preserves_default_and_base(self):
        db = {
            DEFAULT_INDUSTRY: {"equity_only": True},
            "Technology": {"trade_horizon_days": 5},
            TICKER_NAMESPACE: {"NVDA": {"min_confidence": 0.3}},
        }
        merged = resolve_overrides("NVDA", db, base={"min_risk_reward": 1.2})
        assert merged["min_confidence"] == 0.3
        assert merged["trade_horizon_days"] == 5
        assert merged["equity_only"] is True
        assert merged["min_risk_reward"] == 1.2

    def test_stock_entry_case_insensitive(self):
        db = {TICKER_NAMESPACE: {"NVDA": {"min_confidence": 0.2}}}
        assert resolve_overrides("nvda", db)["min_confidence"] == 0.2  # input uppercased

    def test_missing_stock_entry_falls_back_to_industry(self):
        db = {"Technology": {"min_confidence": 0.55}}
        assert resolve_overrides("NVDA", db)["min_confidence"] == 0.55


class TestIndustryWeightsThreading:

    def _divergent_technical(self) -> dict:
        """RSI oversold (bullish) but MACD bearish-strong (bearish)."""
        return {
            "calculate_rsi": {"rsi": 25.0, "signal": "oversold"},
            "calculate_macd": {"trend": "bearish", "momentum_strength": "strong"},
        }

    def test_industry_momentum_weights_flow_into_indicators(self):
        db = {
            "Technology": {
                "momentum_weights": {
                    "macd": 1.0, "adx": 0.0, "rsi": 0.0,
                    "bollinger": 0.0, "obv": 0.0, "stochastic": 0.0,
                },
            },
        }
        overrides = resolve_overrides("NVDA", db)
        tuning = TuningConfig.from_overrides({"momentum_weights": overrides["momentum_weights"]})
        result = calculate_technical_indicators(self._divergent_technical(), tuning=tuning)
        assert result["momentum_score"] < 0  # MACD-led -> bearish

    def test_industry_weights_change_default_behavior(self):
        base = calculate_technical_indicators(self._divergent_technical())
        rsi_led = {
            "momentum_weights": {
                "macd": 0.0, "adx": 0.0, "rsi": 1.0,
                "bollinger": 0.0, "obv": 0.0, "stochastic": 0.0,
            },
        }
        db = {"Financials": rsi_led}
        overrides = resolve_overrides("JPM", db)
        tuning = TuningConfig.from_overrides({"momentum_weights": overrides["momentum_weights"]})
        result = calculate_technical_indicators(self._divergent_technical(), tuning=tuning)
        assert result["momentum_score"] > 0
        assert result["momentum_score"] != base["momentum_score"]
