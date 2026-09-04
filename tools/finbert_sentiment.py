"""FinBERT financial-sentiment scorer (GPU-accelerated, optional dependency).

Loads ``ProsusAI/finbert`` and scores a batch of text into a continuous
sentiment value ``score = P(positive) - P(negative)`` in [-1, +1] plus a label
(``positive`` / ``negative`` / ``neutral``).

``torch`` and ``transformers`` are imported lazily inside the class so the rest
of the application (and the test suite) runs without them.  Install them via
``requirements-ml.txt``.

The model is run in ``eval()`` mode (no dropout), so scoring is deterministic
for a fixed model and input.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger("market_intel_agent.finbert")

_MODEL_NAME = "ProsusAI/finbert"
_MAX_LENGTH = 128


def map_predictions(
    probs: list[list[float]], id2label: dict[int, str] | dict[str, str]
) -> list[dict[str, Any]]:
    """Map softmax probability rows + an ``id2label`` to ``{score, label}``.

    Pure and testable without torch.  ``probs`` is a list of length-3 rows in
    the model's class order; ``score = P(positive) - P(negative)`` and the label
    is the argmax class name (lowercased).
    """
    label_to_idx = {str(v).lower(): int(k) for k, v in id2label.items()}
    idx_to_label = {int(k): str(v).lower() for k, v in id2label.items()}
    out: list[dict[str, Any]] = []
    for row in probs:
        pos_idx = label_to_idx.get("positive")
        neg_idx = label_to_idx.get("negative")
        p_pos = row[pos_idx] if pos_idx is not None else 0.0
        p_neg = row[neg_idx] if neg_idx is not None else 0.0
        arg = max(range(len(row)), key=lambda i: row[i])
        out.append({"score": round(p_pos - p_neg, 6), "label": idx_to_label.get(arg, "neutral")})
    return out


class FinBertSentiment:
    """Thin wrapper around ProsusAI/finbert with batched scoring."""

    def __init__(self, model_name: str = _MODEL_NAME, device: str | None = None) -> None:
        import torch  # type: ignore[import-not-found]  # noqa: PLC0415 - optional dependency
        from transformers import (  # type: ignore[import-not-found]  # noqa: PLC0415
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        self._torch = torch
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        import os  # noqa: PLC0415
        if os.getenv("PYTEST_CURRENT_TEST"):
            self._tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            self._model = AutoModelForSequenceClassification.from_pretrained(model_name, local_files_only=True)
        else:
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
                self._model = AutoModelForSequenceClassification.from_pretrained(model_name, local_files_only=True)
            except Exception:
                self._tokenizer = AutoTokenizer.from_pretrained(model_name)
                self._model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self._model.to(self._device)
        self._model.eval()

        id2label = self._model.config.id2label or {}
        labels = {int(idx): str(label).lower() for idx, label in id2label.items()}
        log.info("FinBERT loaded on %s (labels: %s)", self._device, labels)

    def score(self, text: str) -> dict[str, Any]:
        """Score a single text; returns ``{score, label}``."""
        return self.score_batch([text])[0]

    def score_batch(self, texts: list[str], batch_size: int = 64) -> list[dict[str, Any]]:
        """Score a list of texts (batched) -> ``[{score, label}, ...]``."""
        torch = self._torch
        results: list[dict[str, Any]] = []

        for start in range(0, len(texts), batch_size):
            chunk = texts[start : start + batch_size]
            encoded = self._tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=_MAX_LENGTH,
                return_tensors="pt",
            )
            encoded = {k: v.to(self._device) for k, v in encoded.items()}
            with torch.no_grad():
                logits = self._model(**encoded).logits
            probs = torch.softmax(logits, dim=-1).tolist()  # [batch, 3]
            results.extend(map_predictions(probs, self._model.config.id2label or {}))

        return results
