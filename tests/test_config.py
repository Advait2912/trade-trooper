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
