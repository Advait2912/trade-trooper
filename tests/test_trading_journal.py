"""Trade journal, executor and position-manager unit tests."""



from schemas.decision import DecisionResult
from trading.executor import _as_equity_decision, build_client_order_id, build_order_payload
from trading.journal import TradeJournal
from trading.position_manager import decide_exit
from utils.config import Settings


# ---------------------------------------------------------------------------
# TradeJournal
# ---------------------------------------------------------------------------
def test_journal_round_trip(tmp_path):
    j = TradeJournal(tmp_path / "j.db")
    j.record_cycle("2026-01-01T00:00:00Z", "NVDA", "long_call", "bullish", 0.6,
                   {"price": 100.0, "extra": [1, 2]})
    j.record_order("t1", "NVDA-abc", "NVDA260918C00150000", "buy", 1.0, "limit",
                   2.5, "accepted", None, "ord-1", "")
    j.record_trade("2026-01-01", "2026-01-06", "NVDA", "option", "call",
                   "NVDA260918C00150000", 1.0, 2.5, 3.0, 0.5, 0.2, "target", "t1")
    j.record_equity("2026-01-06T00:00:00Z", 100000.0, 99999.0)
    j.close()

    j2 = TradeJournal(tmp_path / "j.db")
    cycles = j2.cycles()
    assert cycles[0]["decision"] == "long_call"
    assert cycles[0]["snapshot"]["price"] == 100.0
    assert j2.orders()[0]["symbol"] == "NVDA260918C00150000"
    trades = j2.trades()
    assert trades[0]["pnl"] == 0.5
    assert trades[0]["exit_reason"] == "target"
    assert j2.equity_history()[0]["equity"] == 100000.0
    j2.close()


def test_journal_clear(tmp_path):
    j = TradeJournal(tmp_path / "j.db")
    j.record_cycle("x", "NVDA", "hold", "neutral", 0.0, {})
    j.clear()
    assert j.cycles() == []
    j.close()


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------
def _settings(**kw) -> Settings:
    base = dict(alpaca_api_key="k", alpaca_api_secret="s", account_capital=100000.0,
                risk_per_trade_pct=0.01, max_position_pct=0.05, min_risk_reward=1.0,
                min_confidence=0.35, order_type="limit", limit_slippage_pct=0.0)
    base.update(kw)
    return Settings(**base)


def _equity_decision() -> DecisionResult:
    return DecisionResult(
        trade_decision="long_equity", symbol="NVDA", entry_price=100.0,
        stop_loss=95.0, take_profit=110.0, position_shares=5.0,
        confidence_score=0.6, composite_bias="bullish", agreement_score=0.7,
        rationale="test", summary="test",
    )


def _option_decision(option_type="call") -> DecisionResult:
    return DecisionResult(
        trade_decision=f"long_{option_type}", symbol="NVDA", entry_price=100.0,
        stop_loss=95.0, take_profit=110.0, option_contracts=2.0,
        premium_risk=500.0, confidence_score=0.6, composite_bias="bullish",
        agreement_score=0.7, instrument="option", option_type=option_type,
        rationale="test", summary="test",
    )


def test_client_order_id_deterministic():
    a = build_client_order_id("NVDA", "2026-01-01T00:00:00Z")
    b = build_client_order_id("NVDA", "2026-01-01T00:00:00Z")
    assert a == b
    assert a.startswith("NVDA-")


def test_executor_none_for_hold():
    assert build_order_payload(
        DecisionResult(trade_decision="hold"), _settings(), "t", {}) is None


def test_executor_equity_bracket():
    payload = build_order_payload(_equity_decision(), _settings(), "t", {})
    assert payload["qty"] == "5"
    assert payload["order_class"] == "bracket"
    assert payload["stop_loss"]["stop_price"] == "95.00"
    assert payload["take_profit"]["limit_price"] == "110.00"
    assert payload["client_order_id"] in payload["client_order_id"]


def test_executor_equity_market_no_limit():
    payload = build_order_payload(_equity_decision(), _settings(order_type="market"), "t", {})
    assert payload["type"] == "market"
    assert "limit_price" not in payload


def test_executor_option_limits_above_premium():
    options = {"call_symbol": "NVDA260918C00150000", "call_premium": 2.50}
    payload = build_order_payload(_option_decision("call"), _settings(), "t", options)
    assert payload["symbol"] == "NVDA260918C00150000"
    assert payload["qty"] == "2"
    assert payload["side"] == "buy"
    assert payload["class"] == "simple"
    # buy limit above mid by slippage (0 here -> at mid 2.50)
    assert float(payload["limit_price"]) == 2.50


def test_executor_option_no_symbol_returns_none():
    assert build_order_payload(_option_decision("call"), _settings(), "t", {}) is None


def test_executor_equity_fallback_preserves_numbers():
    eq = _as_equity_decision(_option_decision("call"))
    assert eq is None if _option_decision("call").position_shares <= 0 else eq.trade_decision == "long_equity"


# ---------------------------------------------------------------------------
# Position manager
# ---------------------------------------------------------------------------
def _now():
    from datetime import datetime, timezone

    return datetime(2026, 1, 10, tzinfo=timezone.utc)


def test_decide_exit_horizon():
    from datetime import timedelta

    pos = {"opened_at": (_now() - timedelta(days=6)).isoformat(), "mark_price": 100.0,
           "avg_entry_price": 100.0}
    out = decide_exit(pos, DecisionResult(trade_decision="hold"), horizon_days=5, now=_now())
    assert out["action"] == "close_horizon"


def test_decide_exit_flip_avoid():
    from datetime import timedelta

    pos = {"opened_at": (_now() - timedelta(days=1)).isoformat(), "mark_price": 100.0,
           "avg_entry_price": 100.0}
    out = decide_exit(pos, DecisionResult(trade_decision="avoid"), horizon_days=5, now=_now())
    assert out["action"] == "close_flip"


def test_decide_exit_stop_hit_prefers_stop_over_horizon():
    from datetime import timedelta

    pos = {"opened_at": (_now() - timedelta(days=6)).isoformat(), "mark_price": 93.0,
           "avg_entry_price": 100.0, "stop_price": 95.0, "take_profit_price": 110.0}
    out = decide_exit(pos, DecisionResult(trade_decision="avoid"), horizon_days=5, now=_now())
    assert out["action"] == "close_stop"


def test_decide_exit_hold():
    from datetime import timedelta

    pos = {"opened_at": (_now() - timedelta(days=1)).isoformat(), "mark_price": 100.0,
           "avg_entry_price": 100.0, "stop_price": 95.0, "take_profit_price": 110.0}
    out = decide_exit(pos, DecisionResult(trade_decision="hold"), horizon_days=5, now=_now())
    assert out["action"] == "hold"
