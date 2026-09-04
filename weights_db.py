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

import dataclasses
import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from utils.paths import data_path

if TYPE_CHECKING:
    from tuning import TuningConfig
    from utils.config import Settings

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


def resolve_config(
    ticker: str,
    db: dict[str, dict[str, Any]] | None,
    base_settings: Settings,
    base: dict[str, Any] | None = None,
) -> tuple[Settings, TuningConfig]:
    """Return a per-ticker (Settings clone, TuningConfig) from weights_db.

    Resolution order (later wins):
        global defaults <- db["default"] <- db[industry] <- db["tickers"][TICKER]

    ``base`` (preset/``--set`` overrides) is applied first, then the DB layers.
    Falls back cleanly to default TuningConfig when db is None or empty.
    Settings is shallow-cloned so per-ticker min_confidence / min_risk_reward /
    trade_horizon_days from the DB don't bleed across tickers in the same cycle.
    """
    from tuning import TuningConfig

    overrides = resolve_overrides(ticker, db or {}, base)
    settings = dataclasses.replace(base_settings)
    tuning = TuningConfig()
    for key, value in overrides.items():
        if hasattr(tuning, key):
            setattr(tuning, key, value)
        elif hasattr(settings, key):
            setattr(settings, key, value)
    return settings, tuning


# ---------------------------------------------------------------------------
# Checkpoints (snapshots of the weights DB)
# ---------------------------------------------------------------------------
def checkpoints_dir() -> Path:
    """Directory where weights-DB snapshots are stored (``data/checkpoints``)."""
    return data_path("checkpoints")


def snapshot_weights_db(path: str | Path, label: str = "manual") -> Path:
    """Copy the live weights DB into ``data/checkpoints`` and return the copy."""
    data = load_weights_db(path)
    dest = checkpoints_dir() / f"weights_db_{datetime.now():%Y%m%d_%H%M%S}_{label}.json"
    save_weights_db(dest, data)
    return dest


def list_checkpoints() -> list[Path]:
    """Checkpoint files, newest first."""
    if not checkpoints_dir().exists():
        return []
    return sorted(checkpoints_dir().glob("weights_db_*.json"), reverse=True)


def restore_checkpoint(checkpoint_path: str | Path, live_path: str | Path) -> None:
    """Overwrite the live weights DB with a checkpoint file."""
    data = json.loads(Path(checkpoint_path).read_text(encoding="utf-8"))
    save_weights_db(live_path, data)


def delete_checkpoint(checkpoint_path: str | Path) -> None:
    Path(checkpoint_path).unlink(missing_ok=True)
