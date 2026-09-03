"""Configuration and ticker validation."""

import pytest

from utils.config import ConfigError, Settings, validate_ticker


def test_valid_tickers():
    assert validate_ticker("nvda") == "NVDA"
    assert validate_ticker("BRK.B") == "BRK.B"
    assert validate_ticker(" Aapl ") == "AAPL"


def test_invalid_tickers_rejected():
    for bad in ["", "123", "TOOLONG", "NVD A", "NVDA!"]:
        with pytest.raises(ConfigError):
            validate_ticker(bad)


def test_settings_normalizes_urls():
    s = Settings(alpaca_api_key="k", alpaca_api_secret="s")
    assert s.ollama_base_url.endswith("/")
    assert s.has_alpaca_credentials is True


def test_settings_no_credentials():
    s = Settings()
    assert s.has_alpaca_credentials is False


def test_settings_defaults_risk_and_decision():
    s = Settings()
    assert s.min_risk_reward == 1.0
    assert s.min_confidence == 0.35
    assert s.risk_per_trade_pct == 0.01
    assert s.max_position_pct == 0.05
    assert s.account_capital == 100_000.0


def test_settings_custom_risk_and_decision():
    s = Settings(
        min_risk_reward=2.0, min_confidence=0.5, alpaca_options_feed="OPRA"
    )
    assert s.min_risk_reward == 2.0
    assert s.min_confidence == 0.5
    assert s.alpaca_options_feed == "opra"


def test_settings_invalid_feed_normalized():
    s = Settings(alpaca_options_feed="bogus")
    assert s.alpaca_options_feed == "indicative"
