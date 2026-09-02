"""Unit tests for Phase 3 risk tools.

All tests are fully deterministic — no HTTP calls, no LLM, no Alpaca API.
"""

from __future__ import annotations

import pytest

from tools.risk_tools.greeks import calculate_greeks, parse_occ_symbol
from tools.risk_tools.max_loss import calculate_max_loss
from tools.risk_tools.position_size import calculate_position_size
from tools.risk_tools.risk_score import calculate_risk_score, risk_reward_ratio

# ===========================================================================
# parse_occ_symbol
# ===========================================================================

class TestParseOccSymbol:

    def test_parses_call(self):
        parsed = parse_occ_symbol("NVDA260918C00150000")
        assert parsed is not None
        assert parsed["root"] == "NVDA"
        assert parsed["expiry"] == "260918"
        assert parsed["type"] == "call"
        assert parsed["strike"] == 150.0

    def test_parses_put(self):
        parsed = parse_occ_symbol("SPY260620P00400000")
        assert parsed["type"] == "put"
        assert parsed["strike"] == 400.0

    def test_parses_subdollar_strike(self):
        parsed = parse_occ_symbol("ABC260918C00015000")
        assert parsed["strike"] == 15.0

    def test_invalid_symbol_returns_none(self):
        assert parse_occ_symbol("NONSENSE") is None
        assert parse_occ_symbol("") is None
        assert parse_occ_symbol("NVDA260918X00150000") is None


# ===========================================================================
# calculate_greeks
# ===========================================================================

_CHAIN = [
    {
        "symbol": "NVDA260918C00150000",
        "greeks": {"delta": 0.55, "gamma": 0.03, "theta": -0.10, "vega": 0.12, "rho": 0.02},
        "impliedVolatility": 0.35,
        "latestQuote": {"bp": 4.20, "ap": 4.40, "bs": 100, "as": 100},
    },
    {
        "symbol": "NVDA260918P00150000",
        "greeks": {"delta": -0.45, "gamma": 0.03, "theta": -0.09, "vega": 0.12, "rho": -0.02},
        "impliedVolatility": 0.36,
        "latestQuote": {"bp": 3.90, "ap": 4.10, "bs": 100, "as": 100},
    },
]


class TestCalculateGreeks:

    def test_chain_source(self):
        result = calculate_greeks(_CHAIN, 150.0, 30.0, 5)
        assert result["greeks_source"] == "alpaca_option_chain"
        assert result["iv_source"] == "market_implied"

    def test_chain_greeks_present(self):
        result = calculate_greeks(_CHAIN, 150.0, 30.0, 5)
        assert result["greeks"]["call"]["delta"] == 0.55
        assert result["greeks"]["put"]["delta"] == -0.45
        assert set(result["greeks"]["call"]) == {"delta", "gamma", "theta", "vega", "rho"}

    def test_chain_iv_percent_converted(self):
        result = calculate_greeks(_CHAIN, 150.0, 30.0, 5)
        assert result["iv_used"] == pytest.approx(35.0, abs=0.01)

    def test_chain_spread_and_implied_move(self):
        result = calculate_greeks(_CHAIN, 150.0, 30.0, 5)
        assert result["spread_pct"] > 0.0
        assert result["implied_move_pct"] > 0.0

    def test_fallback_source(self):
        result = calculate_greeks(None, 150.0, 30.0, 5)
        assert result["greeks_source"] == "black_scholes_estimated"
        assert result["iv_source"] == "estimated"

    def test_fallback_positive_premium(self):
        result = calculate_greeks([], 150.0, 30.0, 5)
        assert result["call_premium"] > 0.0
        assert result["put_premium"] > 0.0
        assert result["iv_used"] == pytest.approx(30.0, abs=0.01)

    def test_fallback_atm_delta_about_half(self):
        result = calculate_greeks([], 150.0, 30.0, 5)
        assert 0.4 < result["greeks"]["call"]["delta"] < 0.6

    def test_zero_spot_returns_zero_greeks(self):
        result = calculate_greeks(_CHAIN, 0.0, 30.0, 5)
        assert result["iv_used"] == 0.0
        assert len(result["errors"]) > 0

    def test_unparseable_chain_falls_back(self):
        bad_chain = [{"symbol": "NONSENSE", "greeks": {"delta": 1.0}}]
        result = calculate_greeks(bad_chain, 150.0, 30.0, 5)
        assert result["greeks_source"] == "black_scholes_estimated"


# ===========================================================================
# calculate_position_size
# ===========================================================================

class TestCalculatePositionSize:

    def test_positive_sizing(self):
        result = calculate_position_size(100_000, 0.01, 150.0, 147.0)
        assert result["equity_shares"] > 0
        assert result["equity_dollar_value"] > 0
        assert result["capital_at_risk"] > 0

    def test_risk_amount_matches_capital(self):
        result = calculate_position_size(
            100_000, 0.01, 150.0, 147.0, confidence=1.0, max_position_pct=1.0
        )
        assert result["capital_at_risk"] == pytest.approx(1000.0, rel=0.01)

    def test_capped_by_max_position_pct(self):
        result = calculate_position_size(
            100_000, 0.05, 150.0, 140.0, max_position_pct=0.05
        )
        assert result["equity_dollar_value"] <= 100_000 * 0.05 + 1e-6

    def test_spread_reduces_size(self):
        tight = calculate_position_size(100_000, 0.01, 150.0, 147.0, spread_pct=0.0, max_position_pct=1.0)
        wide = calculate_position_size(100_000, 0.01, 150.0, 147.0, spread_pct=0.20, max_position_pct=1.0)
        assert wide["equity_shares"] < tight["equity_shares"]

    def test_drawdown_reduces_size(self):
        low = calculate_position_size(100_000, 0.01, 150.0, 147.0, drawdown_risk="low", max_position_pct=1.0)
        extreme = calculate_position_size(100_000, 0.01, 150.0, 147.0, drawdown_risk="extreme", max_position_pct=1.0)
        assert extreme["equity_shares"] < low["equity_shares"]

    def test_option_sizing_present(self):
        result = calculate_position_size(100_000, 0.01, 150.0, 147.0, premium=4.30, delta=0.55)
        assert result["option_contracts"] > 0
        assert result["option_premium_risk"] > 0
        assert result["delta_exposure"] > 0

    def test_zero_capital(self):
        result = calculate_position_size(0.0, 0.01, 150.0, 147.0)
        assert result["equity_shares"] == 0.0
        assert len(result["errors"]) > 0

    def test_invalid_stop(self):
        result = calculate_position_size(100_000, 0.01, 150.0, 160.0)
        assert result["equity_shares"] == 0.0
        assert len(result["errors"]) > 0


# ===========================================================================
# calculate_max_loss
# ===========================================================================

class TestCalculateMaxLoss:

    def test_basic_stop_loss(self):
        result = calculate_max_loss(150.0, 145.0, 100)
        assert result["base_loss_per_share"] == pytest.approx(5.0)
        assert result["max_loss_dollars"] == pytest.approx(500.0, rel=0.01)

    def test_gap_inflates_loss(self):
        no_gap = calculate_max_loss(150.0, 145.0, 100, avg_gap_size=0.0, gap_frequency="rare")
        gappy = calculate_max_loss(150.0, 145.0, 100, avg_gap_size=2.0, gap_frequency="very_frequent")
        assert gappy["max_loss_dollars"] > no_gap["max_loss_dollars"]

    def test_tail_var_cvar(self):
        result = calculate_max_loss(150.0, 145.0, 100, var_pct=3.0, cvar_pct=5.0)
        assert result["tail_var_dollars"] == pytest.approx(450.0, rel=0.01)
        assert result["tail_cvar_dollars"] == pytest.approx(750.0, rel=0.01)

    def test_zero_position(self):
        result = calculate_max_loss(150.0, 145.0, 0.0)
        assert result["max_loss_dollars"] == 0.0


# ===========================================================================
# risk_reward_ratio
# ===========================================================================

class TestRiskRewardRatio:

    def test_two_to_one(self):
        assert risk_reward_ratio(150.0, 145.0, 160.0) == pytest.approx(2.0)

    def test_zero_risk_returns_zero(self):
        assert risk_reward_ratio(150.0, 150.0, 160.0) == 0.0

    def test_negative_risk_returns_zero(self):
        assert risk_reward_ratio(150.0, 160.0, 170.0) == 0.0


# ===========================================================================
# calculate_risk_score
# ===========================================================================

class TestCalculateRiskScore:

    def test_score_within_bounds(self):
        result = calculate_risk_score()
        assert 0.0 <= result["risk_score"] <= 100.0
        assert result["risk_level"] in {"low", "moderate", "high", "very_high"}

    def test_high_vol_raises_score(self):
        calm = calculate_risk_score(vol_regime="low")
        chaotic = calculate_risk_score(vol_regime="very_high")
        assert chaotic["risk_score"] > calm["risk_score"]

    def test_high_confidence_lowers_score(self):
        low_conf = calculate_risk_score(confidence=0.2)
        high_conf = calculate_risk_score(confidence=0.9)
        assert high_conf["risk_score"] < low_conf["risk_score"]

    def test_extreme_drawdown_raises_score(self):
        low = calculate_risk_score(drawdown_risk="low")
        extreme = calculate_risk_score(drawdown_risk="extreme")
        assert extreme["risk_score"] > low["risk_score"]

    def test_clamped_at_extremes(self):
        worst = calculate_risk_score(
            vol_regime="very_high", iv_percentile=100.0, drawdown_risk="extreme",
            gap_frequency="very_frequent", spread_pct=1.0, max_loss_pct=0.5, confidence=0.0,
        )
        assert worst["risk_score"] <= 100.0
        assert worst["risk_level"] == "very_high"
