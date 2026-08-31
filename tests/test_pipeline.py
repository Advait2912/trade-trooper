"""End-to-end pipeline tests using mocked HTTP (respx).

No real credentials or Ollama server required.
"""

import respx

from agent.pipeline import Pipeline
from tests.conftest import (
    FINAL_JSON,
    INITIAL_JSON,
    mock_market_data,
    mock_market_data_error,
    mock_news,
    mock_news_empty,
    mock_news_error,
    mock_ollama,
    mock_ollama_unavailable,
    mock_web_research_full,
    mock_web_search_error,
)


async def _run(settings):
    pipeline = Pipeline(settings, verbose=False)
    return await pipeline.run("NVDA")


@respx.mock
async def test_bullish_news_full_pipeline(settings):
    mock_news(["Nvidia signs a major new GPU supply agreement"])
    mock_market_data("NVDA")
    mock_ollama()
    mock_web_research_full()

    report = await _run(settings)

    assert report.ticker == "NVDA"
    assert report.event.sentiment.value == "bullish"
    assert report.analysis.news_impact > 0
    assert report.web_research.performed is True
    assert report.market_context.price > 0


@respx.mock
async def test_bearish_news(settings):
    bearish_initial = {
        **INITIAL_JSON,
        "event": "Nvidia faces a major export ban.",
        "sentiment": "bearish",
        "needs_web_research": True,
    }
    bearish_final = {
        **FINAL_JSON,
        "event": {
            **FINAL_JSON["event"],
            "description": "Nvidia faces a major export ban.",
            "sentiment": "bearish",
        },
        "analysis": {**FINAL_JSON["analysis"], "news_impact": -0.6},
    }

    mock_news(["Nvidia faces a major export ban"])
    mock_market_data("NVDA")
    mock_ollama(initial=bearish_initial, final=bearish_final)
    mock_web_research_full()

    report = await _run(settings)
    assert report.event.sentiment.value == "bearish"
    assert report.analysis.news_impact < 0


@respx.mock
async def test_ambiguous_headline(settings):
    ambiguous_initial = {
        **INITIAL_JSON,
        "event": "Nvidia announces something unclear.",
        "sentiment": "uncertain",
        "materiality": "low",
        "relevance": 0.3,
        "needs_web_research": False,
        "research_questions": [],
    }
    ambiguous_final = {
        **FINAL_JSON,
        "event": {
            **FINAL_JSON["event"],
            "description": "Nvidia announces something unclear.",
            "sentiment": "uncertain",
            "materiality": "low",
            "relevance": 0.3,
        },
        "analysis": {
            **FINAL_JSON["analysis"],
            "news_impact": 0.0,
            "actionability": "none",
            "confidence": 0.3,
        },
    }

    mock_news(["Nvidia makes an announcement"])
    mock_market_data("NVDA")
    mock_ollama(initial=ambiguous_initial, final=ambiguous_final)

    report = await _run(settings)
    assert report.event.sentiment.value == "uncertain"
    assert report.web_research.performed is False


@respx.mock
async def test_empty_news_feed(settings):
    mock_news_empty()

    report = await _run(settings)
    assert "Insufficient evidence" in report.analysis.summary
    assert report.analysis.actionability.value == "none"


@respx.mock
async def test_alpaca_api_failure(settings):
    mock_news_error(500)

    report = await _run(settings)
    assert report.analysis.confidence == 0.0
    assert "Insufficient evidence" in report.analysis.summary


@respx.mock
async def test_ollama_unavailable(settings):
    mock_news(["Nvidia signs a major new GPU supply agreement"])
    mock_market_data("NVDA")
    mock_ollama_unavailable(500)

    report = await _run(settings)
    assert report.analysis.actionability.value == "none"
    assert report.analysis.confidence == 0.0


@respx.mock
async def test_web_search_failure(settings):
    mock_news(["Nvidia signs a major new GPU supply agreement"])
    mock_market_data("NVDA")
    mock_ollama()  # initial analysis says needs_web_research=True
    mock_web_search_error(500)

    report = await _run(settings)
    # Research was attempted but produced no sources; pipeline must not crash.
    assert report.web_research.performed is True
    assert report.web_research.sources == []


@respx.mock
async def test_market_data_failure_degrades(settings):
    mock_news(["Nvidia signs a major new GPU supply agreement"])
    mock_market_data_error(500)
    mock_ollama(initial={**INITIAL_JSON, "needs_web_research": False})

    report = await _run(settings)
    assert report.market_context.price == 0.0
    assert report.analysis.summary  # still produced a synthesis
