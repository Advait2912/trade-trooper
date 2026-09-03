"""Phase 5 - Trade statistics.

``render_stats`` reads a ``TradeJournal`` and produces a human-readable report
of the genuine performance statistics:

- realized P&L, trade count, win rate, profit factor, expectancy
- average win / average loss, avg R, max drawdown (on realized equity)
- decision distribution (trades vs holds vs avoids)
- per-instrument breakdown (equity / call / put)
- open positions + latest equity snapshot
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from trading.journal import TradeJournal


def compute_stats(journal: TradeJournal) -> dict[str, Any]:
    trades = journal.trades()
    cycles = journal.cycles()

    pnl = sum(t["pnl"] for t in trades)
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] < 0]
    win_rate = len(wins) / len(trades) if trades else 0.0
    avg_win = sum(t["pnl"] for t in wins) / len(wins) if wins else 0.0
    avg_loss = sum(t["pnl"] for t in losses) / len(losses) if losses else 0.0
    gross_win = sum(t["pnl"] for t in wins)
    gross_loss = abs(sum(t["pnl"] for t in losses))
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else (float("inf") if gross_win > 0 else 0.0)
    expectancy = pnl / len(trades) if trades else 0.0

    # max drawdown on realized equity curve (cumulative P&L, starting 0)
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for t in trades:
        equity += t["pnl"]
        peak = max(peak, equity)
        max_dd = min(max_dd, equity - peak)

    decisions: dict[str, int] = {}
    for c in cycles:
        decisions[c["decision"] or "?"] = decisions.get(c["decision"] or "?", 0) + 1

    instrument: dict[str, dict[str, float]] = {}
    for t in trades:
        key = f"{t['instrument']}({t['option_type']})" if t["instrument"] == "option" else t["instrument"]
        b = instrument.setdefault(key, {"trades": 0.0, "pnl": 0.0})
        b["trades"] += 1
        b["pnl"] += t["pnl"]

    equity_hist = journal.equity_history()

    return {
        "trade_count": len(trades),
        "cycle_count": len(cycles),
        "realized_pnl": round(pnl, 2),
        "win_rate": round(win_rate, 4),
        "profit_factor": round(profit_factor, 2) if math.isfinite(profit_factor) else float("inf"),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "expectancy": round(expectancy, 2),
        "max_drawdown": round(max_dd, 2),
        "decisions": decisions,
        "by_instrument": instrument,
        "latest_equity": equity_hist[-1]["equity"] if equity_hist else None,
        "equity_points": len(equity_hist),
    }


def render_stats(journal_path: str | Path) -> str:
    path = Path(journal_path)
    if not path.exists():
        return f"No journal found at {path}. Run --trade or --backtest first."

    journal = TradeJournal(path)
    stats = compute_stats(journal)
    lines = ["=" * 48, "TRADE STATISTICS", "=" * 48]
    lines.append(f"Journal: {path}")
    lines.append("")
    lines.append(f"Cumulative P&L (realized): ${stats['realized_pnl']:,.2f}")
    lines.append(f"Trades: {stats['trade_count']}  |  Cycles: {stats['cycle_count']}")
    lines.append(f"Win rate: {stats['win_rate'] * 100:.1f}%")
    pf = stats["profit_factor"]
    lines.append(f"Profit factor: {'∞' if pf == float('inf') else round(pf, 2)}")
    lines.append(f"Avg win: ${stats['avg_win']:,.2f}  |  Avg loss: ${stats['avg_loss']:,.2f}")
    lines.append(f"Expectancy: ${stats['expectancy']:,.2f} per trade")
    lines.append(f"Max drawdown (realized): ${stats['max_drawdown']:,.2f}")
    if stats["latest_equity"] is not None:
        lines.append(f"Latest equity: ${stats['latest_equity']:,.2f} "
                     f"({stats['equity_points']} snapshots)")
    lines.append("")
    lines.append("DECISION DISTRIBUTION")
    lines.append("-" * 20)
    for d, n in sorted(stats["decisions"].items(), key=lambda x: -x[1]):
        lines.append(f"{d}: {n}")
    lines.append("")
    lines.append("BY INSTRUMENT")
    lines.append("-" * 13)
    for k, b in sorted(stats["by_instrument"].items(), key=lambda x: -x[1]["pnl"]):
        lines.append(f"{k}: {int(b['trades'])} trades, ${b['pnl']:,.2f}")
    lines.append("=" * 48)
    journal.close()
    return "\n".join(lines)
