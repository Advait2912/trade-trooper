"""Historical data schemas (Phase 1 - Historical Data Agent).

The deterministic tools in ``tools/historical`` return plain JSON-ready dicts;
the models below are the well-typed containers the agent assembles them into
so downstream phases (Prediction, Risk, Decision) receive stable shapes.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PriceBar(BaseModel):
    """One OHLCV bar."""

    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int


class DividendRecord(BaseModel):
    """One cash dividend event (via Alpaca corporate actions)."""

    date: str = ""
    dividend_amount: float = 0.0
    ex_date: str = ""


class EarningsRecord(BaseModel):
    """One earnings datapoint.

    Alpaca does not provide earnings data; this model exists so downstream
    consumers can handle the records when a future data source is added.
    """

    date: str
    eps_actual: float | None = None
    eps_estimate: float | None = None
    revenue: float | None = None
    surprise_percent: float | None = None


class VolatilityPoint(BaseModel):
    """A single point of realized volatility history."""

    date: str
    realized_vol: float = 0.0
    rolling_vol_20d: float = 0.0
    rolling_vol_60d: float = 0.0


class HistoricalAgentResult(BaseModel):
    """Output of the Historical Data Agent (Phase 1 collection).

    ``historical_trends`` and ``volatility_history`` are the primary outputs
    consumed by the Prediction Agent (Phase 2). All other fields are
    deterministic tool results bundled for convenience.
    """

    symbol: str = ""
    status: str = "ok"  # ok | partial | unavailable
    historical_trends: dict[str, Any] = Field(default_factory=dict)
    volatility_history: list[VolatilityPoint] = Field(default_factory=list)
    dividends: list[DividendRecord] = Field(default_factory=list)
    earnings: list[EarningsRecord] = Field(default_factory=list)
    technical: dict[str, Any] = Field(default_factory=dict)
    volatility: dict[str, Any] = Field(default_factory=dict)
    risk: dict[str, Any] = Field(default_factory=dict)
    levels: dict[str, Any] = Field(default_factory=dict)
    patterns: list[dict[str, Any]] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)
    bars_count: int = 0
    errors: list[str] = Field(default_factory=list)
