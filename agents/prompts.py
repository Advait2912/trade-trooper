"""Prompt templates and the shared system prompt.

The system prompt encodes the anti-hallucination rules, source hierarchy and
scoring rules from the specification so the local model behaves like a
research analyst rather than a chatbot.
"""

from __future__ import annotations

from schemas.news import NewsArticle

SYSTEM_PROMPT = """\
You are a junior financial research analyst working inside an intelligence \
system. You NEVER place trades and NEVER recommend specific orders. Your job \
is to observe, investigate, verify, contextualize, assess uncertainty, and \
report — not to translate a headline into a buy/sell call.

STRICT ANTI-HALLUCINATION RULES:
- Never invent a source, quote, transaction, company statement, customer, \
financial number, or publication date.
- Never claim something was "confirmed" when only a headline or summary is \
available.
- Never claim a stock will rise or fall with certainty.
- When evidence is insufficient, say "Insufficient evidence."
- When sources conflict, say "Sources conflict."
- When information cannot be verified, lower your confidence.
- Clearly separate FACTS (explicitly known) from INFERENCES (reasonably \
inferred) from UNCERTAINTIES (unknown).

SOURCE HIERARCHY (higher = more weight):
1. SEC filings / regulatory documents
2. Company investor relations / official announcements
3. Official government sources
4. Reputable financial news
5. Major technology/business publications
6. Other credible sources
7. Low-quality aggregators / social media

SCORING RULES:
- news_impact ranges -1.0 (extremely bearish) to +1.0 (extremely bullish) and \
describes the information's potential directional impact, NOT a predicted \
percentage move.
- confidence is how confident you are in the interpretation (0.0 to 1.0).
- actionability is how useful the information currently is for a trading \
decision (none/low/medium/high). Highly bullish news can still have low \
evidence quality and low actionability.

Always respond ONLY with valid JSON matching the requested schema. Do not add \
commentary, markdown fences, or extra text.
"""

INITIAL_ANALYSIS_PROMPT = """\
You are given the headline and summary of a news article tagged with ticker \
{ticker}.

Answer these questions:
1. What happened?
2. Is this directly relevant to the company behind {ticker}?
3. Could the event materially affect the company or its stock?
4. What facts are explicitly known?
5. What information is missing?
6. Does this event need further web research?

Do NOT decide whether to buy or sell.

Return a JSON object with exactly these fields:
{{
  "ticker": "{ticker}",
  "event": "<one-sentence factual description of the event>",
  "relevance": <0.0 to 1.0>,
  "materiality": "low" | "medium" | "high",
  "sentiment": "bullish" | "bearish" | "neutral" | "uncertain",
  "evidence_quality": "low" | "medium" | "high",
  "needs_web_research": <true | false>,
  "research_questions": ["<specific question>", "..."]
}}

Article:
- Headline: {headline}
- Source: {source}
- Summary: {summary}
"""

FINAL_SYNTHESIS_PROMPT = """\
Synthesize all the evidence below for ticker {ticker} into a structured \
assessment.

INPUTS:
A. Original news:
{news_block}

B. Initial analysis:
{initial_block}

C. Web research findings:
{web_block}

D. Market data and technical indicators:
{market_block}

E. Historical data and technical analysis:
{historical_block}

F. Deterministic prediction / risk / decision (Phases 2-4):
{decision_block}

Requirements:
- Distinguish FACTS, INFERENCES, and UNCERTAINTIES.
- Do NOT translate bullish news automatically into a buy. You may conclude \
bullish-but-not-actionable, bearish-but-not-actionable, neutral, conflicting \
evidence, or insufficient evidence.
- Apply the anti-hallucination and scoring rules.
- "web_research.performed" must be {performed} (a JSON boolean) and \
"web_research.key_findings" / "web_research.sources" must reflect the research \
above (empty if none was performed).
- Use the historical context to corroborate or challenge the news-driven \
narrative, but never invent numbers.
- Your "council_input" must be consistent with the deterministic Phase 4 \
decision block: if it says long_call/long_put/long_equity, your \
recommended_bias should agree (bullish/bearish) rather than contradict it. \
Never invent prices or levels that are not already in the inputs.

Return a JSON object with exactly these fields:
{{
  "event": {{
    "description": "<string>",
    "relevance": <0.0 to 1.0>,
    "materiality": "low" | "medium" | "high",
    "sentiment": "bullish" | "bearish" | "neutral" | "uncertain"
  }},
  "evidence": {{
    "quality": "low" | "medium" | "high",
    "facts": ["<string>"],
    "inferences": ["<string>"],
    "uncertainties": ["<string>"]
  }},
  "web_research": {{
    "performed": {performed},
    "key_findings": ["<string>"],
    "sources": [
      {{
        "title": "<string>",
        "source": "<string>",
        "url": "<string>",
        "relevance": "high" | "medium" | "low"
      }}
    ]
  }},
  "market_trend": "bullish" | "bearish" | "neutral" | "uncertain",
  "analysis": {{
    "news_impact": <-1.0 to 1.0>,
    "actionability": "none" | "low" | "medium" | "high",
    "time_horizon": "immediate" | "1-5_days" | "1-4_weeks" | "long_term" | "uncertain",
    "confidence": <0.0 to 1.0>,
    "summary": "<string>"
  }},
  "council_input": {{
    "recommended_bias": "bullish" | "bearish" | "neutral" | "uncertain",
    "confidence": <0.0 to 1.0>,
    "key_reason": "<string>",
    "should_council_consider": <true | false>
  }}
}}
"""


def build_initial_prompt(ticker: str, article: NewsArticle) -> str:
    return INITIAL_ANALYSIS_PROMPT.format(
        ticker=ticker,
        headline=article.headline,
        source=article.source or "unknown",
        summary=article.summary or "(no summary provided)",
    )


def build_final_prompt(
    ticker: str,
    news_block: str,
    initial_block: str,
    web_block: str,
    market_block: str,
    historical_block: str,
    decision_block: str,
    performed: bool,
) -> str:
    return FINAL_SYNTHESIS_PROMPT.format(
        ticker=ticker,
        news_block=news_block,
        initial_block=initial_block,
        web_block=web_block,
        market_block=market_block,
        historical_block=historical_block,
        decision_block=decision_block,
        performed="true" if performed else "false",
    )
