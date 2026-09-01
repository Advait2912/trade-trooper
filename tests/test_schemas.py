"""Pydantic schema validation."""

import pytest
from pydantic import ValidationError

from schemas import (
    Analysis,
    Bias,
    EventAssessment,
    EvidenceQuality,
    FinalReport,
    InitialAnalysis,
    Materiality,
    Sentiment,
)


def test_initial_analysis_valid():
    a = InitialAnalysis(
        ticker="NVDA",
        event="x",
        relevance=0.5,
        materiality=Materiality.HIGH,
        sentiment=Sentiment.BULLISH,
        evidence_quality=EvidenceQuality.MEDIUM,
        needs_web_research=True,
        research_questions=["q"],
    )
    assert a.materiality.value == "high"


def test_relevance_out_of_bounds_rejected():
    with pytest.raises(ValidationError):
        InitialAnalysis(ticker="NVDA", event="x", relevance=1.5)


def test_invalid_enum_rejected():
    with pytest.raises(ValidationError):
        InitialAnalysis(
            ticker="NVDA",
            event="x",
            relevance=0.5,
            sentiment="happy",  # type: ignore[arg-type]
        )


def test_news_impact_bounds():
    with pytest.raises(ValidationError):
        Analysis(news_impact=1.5)
    with pytest.raises(ValidationError):
        Analysis(news_impact=-1.5)


def test_final_report_roundtrip():
    report = FinalReport(
        ticker="NVDA",
        timestamp="2026-01-01T00:00:00Z",
        event=EventAssessment(description="d", relevance=0.5),
    )
    data = report.model_dump()
    again = FinalReport.model_validate(data)
    assert again.ticker == "NVDA"
    assert again.event.relevance == 0.5


def test_bias_default():
    from schemas import CouncilInput

    ci = CouncilInput()
    assert ci.recommended_bias is Bias.NEUTRAL
