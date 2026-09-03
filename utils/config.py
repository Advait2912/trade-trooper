"""Configuration loading and validation.

All settings are read from environment variables (loaded via python-dotenv).
No secrets are ever hardcoded.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

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
    alpaca_options_feed: str = "indicative"  # opra (paid) | indicative (free)

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

    # Risk (Phase 3) — sizing / capital assumptions.
    account_capital: float = 100_000.0
    risk_per_trade_pct: float = 0.01    # fraction of capital risked per trade
    max_position_pct: float = 0.05      # cap on position as a fraction of capital
    min_risk_reward: float = 1.0        # minimum reward/risk (surfaced to Phase 4)

    # Decision (Phase 4).
    min_confidence: float = 0.35        # minimum prediction confidence to trade

    # Trading (execution) — paper only, kill-switched.
    trading_enabled: bool = False       # set true to allow the paper runner
    trading_interval_min: int = 30      # cycle interval in minutes
    max_open_positions: int = 1         # one position at a time
    daily_loss_limit_pct: float = 0.02  # stop trading for the day if hit (fraction)
    order_type: str = "limit"           # limit | market
    limit_slippage_pct: float = 0.0     # extra % away from mid for limit fills
    trade_horizon_days: int = 5         # max holding period (calendar days)

    def __post_init__(self) -> None:
        self._normalize()

    def _normalize(self) -> None:
        if not self.ollama_base_url.endswith("/"):
            self.ollama_base_url += "/"
        self.alpaca_options_feed = self.alpaca_options_feed.lower()
        if self.alpaca_options_feed not in ("opra", "indicative"):
            self.alpaca_options_feed = "indicative"
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

    try:
        account_capital = float(_env("ACCOUNT_CAPITAL", "100000"))
        risk_per_trade_pct = float(_env("RISK_PER_TRADE_PCT", "0.01"))
        max_position_pct = float(_env("MAX_POSITION_PCT", "0.05"))
        min_risk_reward = float(_env("MIN_RISK_REWARD", "1.0"))
        min_confidence = float(_env("MIN_CONFIDENCE", "0.35"))
        daily_loss_limit_pct = float(_env("DAILY_LOSS_LIMIT_PCT", "0.02"))
    except ValueError as exc:  # pragma: no cover - defensive
        raise ConfigError(
            "ACCOUNT_CAPITAL, RISK_PER_TRADE_PCT, MAX_POSITION_PCT, "
            "MIN_RISK_REWARD, MIN_CONFIDENCE and DAILY_LOSS_LIMIT_PCT must be numbers."
        ) from exc

    trading_enabled = _env("TRADING_ENABLED", "false").lower() in ("1", "true", "yes")
    try:
        trading_interval_min = int(_env("TRADING_INTERVAL_MIN", "30"))
        max_open_positions = int(_env("MAX_OPEN_POSITIONS", "1"))
        trade_horizon_days = int(_env("TRADE_HORIZON_DAYS", "5"))
    except ValueError as exc:  # pragma: no cover - defensive
        raise ConfigError(
            "TRADING_INTERVAL_MIN, MAX_OPEN_POSITIONS and TRADE_HORIZON_DAYS "
            "must be integers."
        ) from exc

    order_type = _env("ORDER_TYPE", "limit").lower()
    if order_type not in ("limit", "market"):
        order_type = "limit"

    return Settings(
        alpaca_api_key=_env("ALPACA_API_KEY"),
        alpaca_api_secret=_env("ALPACA_API_SECRET"),
        alpaca_data_feed=_env("ALPACA_DATA_FEED", "iex").lower(),
        alpaca_options_feed=_env("ALPACA_OPTIONS_FEED", "indicative").lower(),
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
        account_capital=account_capital,
        risk_per_trade_pct=risk_per_trade_pct,
        max_position_pct=max_position_pct,
        min_risk_reward=min_risk_reward,
        min_confidence=min_confidence,
        trading_enabled=trading_enabled,
        trading_interval_min=trading_interval_min,
        max_open_positions=max_open_positions,
        daily_loss_limit_pct=daily_loss_limit_pct,
        order_type=order_type,
        limit_slippage_pct=float(_env("LIMIT_SLIPPAGE_PCT", "0")),
        trade_horizon_days=trade_horizon_days,
    )
