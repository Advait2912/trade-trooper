"""Enums shared across pipeline schemas.

Every enum also carries its JSON string value (``str, Enum``) so LLM output
and serialized reports stay clean.
"""

from __future__ import annotations

from enum import Enum


class Materiality(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Sentiment(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    UNCERTAIN = "uncertain"


class EvidenceQuality(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Trend(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    UNCERTAIN = "uncertain"


class Actionability(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class TimeHorizon(str, Enum):
    IMMEDIATE = "immediate"
    ONE_TO_FIVE_DAYS = "1-5_days"
    ONE_TO_FOUR_WEEKS = "1-4_weeks"
    LONG_TERM = "long_term"
    UNCERTAIN = "uncertain"


class Bias(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    UNCERTAIN = "uncertain"


class SourceRelevance(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
