"""Phase 5 - Execution / evidence pipeline.

Deterministic paper-trading execution and statistics over the Phase 1-4
decision pipeline:

- ``journal``          SQLite persistence for cycles/orders/trades/equity
- ``executor``         DecisionResult -> paper orders (equity bracket / options)
- ``position_manager`` exit rules (stop/target/horizon/flip)
- ``runner``           the 30-minute market-hours loop (``main.py --trade``)
- ``backtest``         deterministic Phase 2-4 replay over historical bars
- ``stats``            realized/unrealized P&L, win rate, profit factor, etc.
"""

from trading.journal import TradeJournal

__all__ = ["TradeJournal"]
