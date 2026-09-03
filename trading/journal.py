"""Phase 5 - Trade journal.

A small SQLite-backed persistence layer that records, in one place:

- every pipeline cycle (the decision that was made and its snapshot),
- every order submitted / fill,
- every closed position (realized P&L),
- periodic equity snapshots (for the equity curve).

The journal is the single source of truth for the statistics report
(``trading.stats``).  It is deliberately dependency-free (stdlib ``sqlite3``).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cycles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    ticker TEXT NOT NULL,
    decision TEXT,
    composite_bias TEXT,
    confidence REAL,
    snapshot TEXT
);
CREATE TABLE IF NOT EXISTS orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle_ts TEXT,
    client_order_id TEXT UNIQUE,
    symbol TEXT,
    side TEXT,
    qty REAL,
    type TEXT,
    limit_price REAL,
    status TEXT,
    filled_avg_price REAL,
    order_id TEXT,
    reason TEXT
);
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    opened_ts TEXT,
    closed_ts TEXT,
    ticker TEXT,
    instrument TEXT,
    option_type TEXT,
    symbol TEXT,
    quantity REAL,
    entry_price REAL,
    exit_price REAL,
    pnl REAL,
    pnl_pct REAL,
    exit_reason TEXT,
    cycle_ts TEXT
);
CREATE TABLE IF NOT EXISTS equity (
    ts TEXT PRIMARY KEY,
    equity REAL,
    cash REAL
);
"""


class TradeJournal:
    """Append-only journal backed by SQLite."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path))
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    @property
    def path(self) -> Path:
        return self._path

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------
    def record_cycle(
        self,
        ts: str,
        ticker: str,
        decision: str | None,
        composite_bias: str | None,
        confidence: float,
        snapshot: dict,
    ) -> None:
        self._conn.execute(
            "INSERT INTO cycles (ts, ticker, decision, composite_bias, confidence, snapshot) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (ts, ticker, decision, composite_bias, confidence, json.dumps(snapshot, default=str)),
        )
        self._conn.commit()

    def record_order(
        self,
        cycle_ts: str,
        client_order_id: str,
        symbol: str,
        side: str,
        qty: float,
        order_type: str,
        limit_price: float | None,
        status: str,
        filled_avg_price: float | None,
        order_id: str | None,
        reason: str = "",
    ) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO orders (cycle_ts, client_order_id, symbol, side, qty, type, "
            "limit_price, status, filled_avg_price, order_id, reason) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                cycle_ts, client_order_id, symbol, side, qty, order_type, limit_price,
                status, filled_avg_price, order_id, reason,
            ),
        )
        self._conn.commit()

    def record_trade(
        self,
        opened_ts: str,
        closed_ts: str,
        ticker: str,
        instrument: str,
        option_type: str,
        symbol: str,
        quantity: float,
        entry_price: float,
        exit_price: float,
        pnl: float,
        pnl_pct: float,
        exit_reason: str,
        cycle_ts: str,
    ) -> None:
        self._conn.execute(
            "INSERT INTO trades (opened_ts, closed_ts, ticker, instrument, option_type, symbol, "
            "quantity, entry_price, exit_price, pnl, pnl_pct, exit_reason, cycle_ts) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                opened_ts, closed_ts, ticker, instrument, option_type, symbol, quantity,
                entry_price, exit_price, pnl, pnl_pct, exit_reason, cycle_ts,
            ),
        )
        self._conn.commit()

    def record_equity(self, ts: str, equity: float, cash: float) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO equity (ts, equity, cash) VALUES (?, ?, ?)",
            (ts, equity, cash),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------
    def cycles(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT id, ts, ticker, decision, composite_bias, confidence, snapshot FROM cycles ORDER BY id"
        ).fetchall()
        out = []
        for r in rows:
            out.append(
                {
                    "id": r[0], "ts": r[1], "ticker": r[2], "decision": r[3],
                    "composite_bias": r[4], "confidence": r[5],
                    "snapshot": json.loads(r[6]) if r[6] else {},
                }
            )
        return out

    def orders(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT client_order_id, symbol, side, qty, type, status, filled_avg_price, reason "
            "FROM orders ORDER BY id"
        ).fetchall()
        return [
            {
                "client_order_id": r[0], "symbol": r[1], "side": r[2], "qty": r[3],
                "type": r[4], "status": r[5], "filled_avg_price": r[6], "reason": r[7],
            }
            for r in rows
        ]

    def trades(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT opened_ts, closed_ts, ticker, instrument, option_type, symbol, quantity, "
            "entry_price, exit_price, pnl, pnl_pct, exit_reason FROM trades ORDER BY id"
        ).fetchall()
        return [
            {
                "opened_ts": r[0], "closed_ts": r[1], "ticker": r[2], "instrument": r[3],
                "option_type": r[4], "symbol": r[5], "quantity": r[6], "entry_price": r[7],
                "exit_price": r[8], "pnl": r[9], "pnl_pct": r[10], "exit_reason": r[11],
            }
            for r in rows
        ]

    def equity_history(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT ts, equity, cash FROM equity ORDER BY ts"
        ).fetchall()
        return [{"ts": r[0], "equity": r[1], "cash": r[2]} for r in rows]

    def clear(self) -> None:
        for table in ("cycles", "orders", "trades", "equity"):
            self._conn.execute(f"DELETE FROM {table}")
        self._conn.commit()
