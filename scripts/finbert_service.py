"""Shared FinBERT sentiment microservice.

Provides fast, batched inference for financial text so all trading runners
can query a single shared model over HTTP without keeping heavy torch models
in the trader containers.

Endpoints:
- POST /score_batch   {"texts": ["headline. summary", ...]}
                      -> [{"label": "positive"|"negative"|"neutral", "score": float}, ...]
- GET  /health        -> {"status": "ok"}
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel

log = logging.getLogger("finbert_service")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

app = FastAPI(title="FinBERT Sentiment Microservice")

_MODEL_NAME = "ProsusAI/finbert"
_model = None
_tokenizer = None
_device = "cpu"
_id2label: dict[int, str] = {0: "positive", 1: "negative", 2: "neutral"}


def _load_model() -> None:
    global _model, _tokenizer, _device, _id2label
    if _model is not None:
        return

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Loading %s on %s...", _MODEL_NAME, _device)
    _tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
    _model = AutoModelForSequenceClassification.from_pretrained(_MODEL_NAME).to(_device).eval()
    if hasattr(_model.config, "id2label"):
        _id2label = {int(k): v.lower() for k, v in _model.config.id2label.items()}
    log.info("FinBERT model loaded successfully on %s", _device)


@app.on_event("startup")
def startup_event() -> None:
    _load_model()


class BatchRequest(BaseModel):
    texts: list[str]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "device": _device}


@app.post("/score_batch")
def score_batch(req: BatchRequest) -> list[dict[str, Any]]:
    _load_model()
    import torch
    import torch.nn.functional as F

    texts = [t.strip() for t in req.texts if t.strip()]
    if not texts:
        return []

    with torch.no_grad():
        inputs = _tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(_device)
        outputs = _model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1).cpu()

    results: list[dict[str, Any]] = []
    for row in probs:
        best_idx = int(row.argmax().item())
        label = _id2label.get(best_idx, "neutral")
        confidence = float(row[best_idx].item())
        results.append({"label": label, "score": round(confidence, 4)})

    return results


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
