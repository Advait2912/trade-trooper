"""Industry- and stock-scoped weight database for the tuning harness.

Maps a stock ticker to its industry and stores tuning overrides
(``TuningConfig`` + ``Settings`` knobs) in a JSON file.  The tuning harness
resolves each ticker's weights via ``resolve_overrides`` so a mixed-universe
backtest uses each stock's own config.

    db = load_weights_db("data/weights_db.json")  # {} if missing
    overrides = resolve_overrides("NVDA", db, base=preset_overrides)
    # overrides -> (Settings, TuningConfig) via scripts/tune.py

Resolution order for a ticker (later layers win):

    base (preset/--set)  <-  "default"  <-  industry  <-  "tickers"[TICKER]

A ticker not listed in ``INDUSTRY_STOCKS`` resolves to the ``"default"``
industry, which falls back to the global tuned config seeded in the DB.
Stock-specific overrides live under the reserved ``TICKER_NAMESPACE`` key.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_INDUSTRY = "default"

#: Reserved top-level key holding per-ticker overrides: db["tickers"]["NVDA"].
TICKER_NAMESPACE = "tickers"

INDUSTRY_STOCKS: dict[str, list[str]] = {
    "Technology": ["NVDA", "AAPL", "MSFT", "AVGO", "AMD", "ORCL"],
    "Financials": ["JPM", "BAC", "GS", "BLK", "V", "MA"],
    "Healthcare": ["LLY", "JNJ", "UNH", "ABBV", "MRK", "TMO"],
    "Consumer Discretionary": ["AMZN", "TSLA", "HD", "MCD", "NKE", "BKNG"],
    "Industrials": ["CAT", "GE", "HON", "RTX", "UPS", "BA"],
    "Communication Services": ["GOOGL", "META", "NFLX", "DIS", "TMUS", "CMCSA"],
    "Consumer Staples": ["WMT", "COST", "PG", "KO", "PEP", "MDLZ"],
    "Energy": ["XOM", "CVX", "COP", "SLB", "EOG", "MPC"],
    "Materials": ["LIN", "SHW", "FCX", "NUE", "NEM", "APD"],
    "Utilities": ["NEE", "DUK", "SO", "AEP", "EXC", "SRE"],
    "Real Estate": ["PLD", "AMT", "EQIX", "WELL", "SPG", "O"],
}

TICKER_TO_INDUSTRY: dict[str, str] = {}
for _industry, _tickers in INDUSTRY_STOCKS.items():
    for _t in _tickers:
        if _t in TICKER_TO_INDUSTRY:
            raise ValueError(f"Ticker {_t!r} appears in multiple industries")
        TICKER_TO_INDUSTRY[_t] = _industry


def resolve_industry(ticker: str) -> str:
    """Return the industry for a ticker, or ``DEFAULT_INDUSTRY`` if unknown."""
    return TICKER_TO_INDUSTRY.get(ticker.strip().upper(), DEFAULT_INDUSTRY)


def load_weights_db(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load the weights DB; returns {} when the file does not exist."""
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def save_weights_db(path: str | Path, data: dict[str, dict[str, Any]]) -> None:
    """Atomically write the weights DB (tmp file + replace)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(p)


def resolve_overrides(
    ticker: str,
    db: dict[str, dict[str, Any]],
    base: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge overrides for a ticker: ``base`` <- ``default`` <- industry <- ticker.

    ``base`` holds preset/``--set``/``--config`` overrides applied to every
    ticker; ``default``, the industry entry, and the stock-specific entry
    (``db["tickers"][TICKER]``) come from the weights DB.  Later layers win.
    """
    merged = dict(base or {})
    merged.update(db.get(DEFAULT_INDUSTRY) or {})
    merged.update(db.get(resolve_industry(ticker)) or {})
    ticker_entry = (db.get(TICKER_NAMESPACE) or {}).get(ticker.strip().upper())
    if ticker_entry:
        merged.update(ticker_entry)
    return merged
