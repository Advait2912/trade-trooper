"""Stats + backtest helper tests (synthetic journal data, no HTTP)."""

import pytest

from trading.backtest import _check_exit
from trading.journal import TradeJournal
from trading.stats import compute_stats, render_stats


class _Bar:
    def __init__(self, high, low, close, open=0.0):
        self.high = high
        self.low = low
        self.close = close
        self.open = open


def test_compute_stats_known_journal(tmp_path):
    j = TradeJournal(tmp_path / "s.db")
    j.record_trade("t0", "t1", "NVDA", "equity", "", "NVDA", 10, 100.0, 105.0, 50.0, 0.05, "target", "c1")
    j.record_trade("t0", "t2", "NVDA", "equity", "", "NVDA", 10, 100.0, 97.0, -30.0, -0.03, "stop", "c2")
    j.record_cycle("c1", "NVDA", "long_equity", "bullish", 0.6, {})
    j.record_cycle("c2", "NVDA", "long_equity", "bullish", 0.7, {})
    j.record_equity("t2", 100050.0, 100020.0)

    stats = compute_stats(j)
    assert stats["trade_count"] == 2
    assert stats["realized_pnl"] == 20.0
    assert stats["win_rate"] == 0.5
    assert stats["expectancy"] == 10.0
    assert stats["avg_win"] == 50.0
    # profit factor: 50 / 30 = 1.67
    assert stats["profit_factor"] == pytest.approx(1.67, abs=0.01)
    assert stats["max_drawdown"] == -30.0
    assert stats["latest_equity"] == 100050.0
    j.close()


def test_render_stats_no_journal(tmp_path):
    out = render_stats(tmp_path / "missing.db")
    assert "No journal found" in out


def test_render_stats_summary_lines(tmp_path):
    j = TradeJournal(tmp_path / "r.db")
    j.record_trade("t0", "t1", "AAPL", "option", "call", "AAPL260101C123", 1, 2.5, 4.0, 150.0, 0.6, "target", "c1")
    j.close()
    out = render_stats(tmp_path / "r.db")
    assert "Cumulative P&L" in out
    assert "BY INSTRUMENT" in out
    assert "option(call)" in out


def test_check_exit_long_call():
    active = {"decision": "long_call", "stop": 95.0, "target": 110.0}
    assert _check_exit(active, _Bar(high=112.0, low=105.0, close=111.0))[:2] == (False, True)
    assert _check_exit(active, _Bar(high=96.0, low=93.0, close=94.0))[:2] == (True, False)


def test_check_exit_long_put():
    active = {"decision": "long_put", "stop": 105.0, "target": 90.0}
    assert _check_exit(active, _Bar(high=106.0, low=100.0, close=101.0))[:2] == (True, False)
    assert _check_exit(active, _Bar(high=95.0, low=89.0, close=92.0))[:2] == (False, True)


def test_check_exit_hold():
    active = {"decision": "long_equity", "stop": 95.0, "target": 110.0}
    assert _check_exit(active, _Bar(high=103.0, low=100.0, close=102.0))[:2] == (False, False)
