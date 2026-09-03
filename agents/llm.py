"""Ollama inference client with structured-output validation.

Uses the local Ollama HTTP API (/api/chat) directly — never shelling out to
the `ollama` CLI. Responses are validated with Pydantic and repaired once if
invalid.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Type, TypeVar
from urllib.parse import urljoin

import httpx
from pydantic import BaseModel, ValidationError

from agents import prompts
from schemas.news import InitialAnalysis, NewsArticle
from schemas.pipeline import FinalSynthesis
from utils.config import Settings

log = logging.getLogger("market_intel_agent.ollama")

T = TypeVar("T", bound=BaseModel)

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


class OllamaError(RuntimeError):
    """Base class for Ollama errors."""


class OllamaUnavailableError(OllamaError):
    """Could not reach the local Ollama server."""


class OllamaParseError(OllamaError):
    """Model output could not be parsed into valid JSON after a repair attempt."""


class OllamaClient:
    """Async wrapper around the local Ollama chat API."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._chat_url = urljoin(settings.ollama_base_url, "api/chat")
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "OllamaClient":
        self._client = httpx.AsyncClient(timeout=self._settings.ollama_timeout)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # ------------------------------------------------------------------
    # High-level analysis entry points
    # ------------------------------------------------------------------
    async def analyze_initial(self, ticker: str, article: NewsArticle) -> InitialAnalysis:
        prompt = prompts.build_initial_prompt(ticker, article)
        return await self._complete(
            prompt=prompt,
            system=prompts.SYSTEM_PROMPT,
            response_model=InitialAnalysis,
        )

    async def synthesize(
        self,
        ticker: str,
        news_block: str,
        initial_block: str,
        web_block: str,
        market_block: str,
        historical_block: str,
        decision_block: str,
        performed: bool,
    ) -> FinalSynthesis:
        prompt = prompts.build_final_prompt(
            ticker=ticker,
            news_block=news_block,
            initial_block=initial_block,
            web_block=web_block,
            market_block=market_block,
            historical_block=historical_block,
            decision_block=decision_block,
            performed=performed,
        )
        return await self._complete(
            prompt=prompt,
            system=prompts.SYSTEM_PROMPT,
            response_model=FinalSynthesis,
        )

    # ------------------------------------------------------------------
    # Low-level completion with validation + repair
    # ------------------------------------------------------------------
    async def _complete(
        self,
        prompt: str,
        system: str,
        response_model: Type[T],
    ) -> T:
        if self._client is None:
            raise OllamaError("OllamaClient used outside of async context manager.")

        schema = _json_schema(response_model)

        # First attempt with JSON-schema format; fall back to plain JSON mode.
        for fmt in (schema, "json"):
            try:
                content = await self._chat(prompt, system, fmt)
                return _parse_model(content, response_model)
            except (ValidationError, ValueError) as exc:
                log.warning("Initial parse failed (%s); attempting repair.", exc)
                try:
                    content = await self._chat(prompt, system, fmt, repair=str(exc))
                    return _parse_model(content, response_model)
                except (ValidationError, ValueError):
                    continue
            except _FormatUnsupportedError:
                continue

        raise OllamaParseError(
            f"Ollama output could not be validated as {response_model.__name__}."
        )

    async def _chat(
        self,
        prompt: str,
        system: str,
        fmt: Any,
        repair: str | None = None,
    ) -> str:
        assert self._client is not None
        messages: list[dict] = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        if repair:
            messages.append(
                {
                    "role": "assistant",
                    "content": "(previous output was invalid)",
                }
            )
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Your previous output failed validation: "
                        f"{repair}\nReturn ONLY valid JSON matching the schema."
                    ),
                }
            )

        payload: dict[str, Any] = {
            "model": self._settings.ollama_model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.0},
            "format": fmt,
        }

        try:
            resp = await self._client.post(self._chat_url, json=payload)
        except httpx.HTTPError as exc:
            raise OllamaUnavailableError(
                f"Could not reach Ollama at {self._chat_url}: {exc}"
            ) from exc

        if resp.status_code >= 400:
            body = resp.text[:300]
            if _looks_like_unsupported_format(resp.status_code, body):
                raise _FormatUnsupportedError(body)
            raise OllamaError(f"Ollama error {resp.status_code}: {body}")

        try:
            data = resp.json()
        except (ValueError, TypeError) as exc:
            raise OllamaError("Ollama returned malformed JSON.") from exc

        content = data.get("message", {}).get("content", "")
        if not content:
            raise ValueError("Empty model output.")
        return content


class _FormatUnsupportedError(Exception):
    pass


def _looks_like_unsupported_format(status: int, body: str) -> bool:
    if status != 400:
        return False
    lowered = body.lower()
    return "format" in lowered and ("invalid" in lowered or "unsupported" in lowered)


def _json_schema(model: Type[BaseModel]) -> dict[str, Any]:
    """Derive a JSON Schema from a Pydantic model for Ollama structured output."""
    return model.model_json_schema()


def _parse_model(content: str, model: Type[T]) -> T:
    text = _strip_fences(content.strip())
    data = json.loads(text)
    return model.model_validate(data)


def _strip_fences(text: str) -> str:
    m = _FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    return text
