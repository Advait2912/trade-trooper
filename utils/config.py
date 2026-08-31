"""Configuration loading and validation.

All settings are read from environment variables (loaded via python-dotenv).
No secrets are ever hardcoded.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from dotenv import load_dotenv

# A US equity ticker: 1-5 uppercase letters (optionally followed by a share
# class suffix such as .A, -B, etc. which Alpaca supports for some symbols).
_TICKER_RE = re.compile(r"^[A-Za-z]{1,5}([.\-][A-Za-z])?$")


class ConfigError(RuntimeError):
    """Raised when configuration is missing or invalid."""


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


@dataclass
class Settings:
    """Runtime configuration for the agent."""

    alpaca_api_key: str = ""
    alpaca_api_secret: str = ""
    alpaca_data_feed: str = "iex"

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "gemma4:e4b"
    ollama_api_key: str = ""
    ollama_web_search_url: str = "http://localhost:11434/api/experimental/web_search"
    ollama_web_fetch_url: str = "http://localhost:11434/api/experimental/web_fetch"

    news_limit: int = 5
    lookback_hours: int = 24

    # Web research guardrails.
    max_search_rounds: int = 3
    max_fetch_pages: int = 5
    max_web_chars_per_page: int = 4000
    max_web_chars_total: int = 12000

    # Timeouts (seconds).
    http_timeout: float = 15.0
    ollama_timeout: float = 120.0

    def __post_init__(self) -> None:
        self._normalize()

    def _normalize(self) -> None:
        if not self.ollama_base_url.endswith("/"):
            self.ollama_base_url += "/"
        if not self.ollama_web_search_url.startswith("http"):
            raise ConfigError(
                f"Invalid OLLAMA_WEB_SEARCH_URL: {self.ollama_web_search_url!r}"
            )
        if not self.ollama_web_fetch_url.startswith("http"):
            raise ConfigError(
                f"Invalid OLLAMA_WEB_FETCH_URL: {self.ollama_web_fetch_url!r}"
            )
        self.ollama_web_search_url = self.ollama_web_search_url.rstrip("/")
        self.ollama_web_fetch_url = self.ollama_web_fetch_url.rstrip("/")

    @property
    def has_alpaca_credentials(self) -> bool:
        return bool(self.alpaca_api_key and self.alpaca_api_secret)


def validate_ticker(ticker: str) -> str:
    """Validate and normalize a ticker symbol.

    Raises ConfigError for invalid input; returns the normalized uppercase
    root symbol (share-class suffix preserved).
    """
    ticker = ticker.strip()
    if not ticker:
        raise ConfigError("Ticker must not be empty.")
    if not _TICKER_RE.match(ticker):
        raise ConfigError(
            f"Invalid ticker {ticker!r}: expected 1-5 letters, optionally "
            "followed by a share-class suffix (e.g. BRK.B)."
        )
    return ticker.upper()


def load_settings() -> Settings:
    """Load settings from the environment, applying the .env file if present."""
    load_dotenv()

    try:
        news_limit = int(_env("NEWS_LIMIT", "5"))
        lookback_hours = int(_env("LOOKBACK_HOURS", "24"))
    except ValueError as exc:  # pragma: no cover - defensive
        raise ConfigError("NEWS_LIMIT and LOOKBACK_HOURS must be integers.") from exc

    return Settings(
        alpaca_api_key=_env("ALPACA_API_KEY"),
        alpaca_api_secret=_env("ALPACA_API_SECRET"),
        alpaca_data_feed=_env("ALPACA_DATA_FEED", "iex").lower(),
        ollama_base_url=_env("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_model=_env("OLLAMA_MODEL", "gemma4:e4b"),
        ollama_api_key=_env("OLLAMA_API_KEY"),
        ollama_web_search_url=_env(
            "OLLAMA_WEB_SEARCH_URL",
            "http://localhost:11434/api/experimental/web_search",
        ),
        ollama_web_fetch_url=_env(
            "OLLAMA_WEB_FETCH_URL",
            "http://localhost:11434/api/experimental/web_fetch",
        ),
        news_limit=news_limit,
        lookback_hours=lookback_hours,
    )
