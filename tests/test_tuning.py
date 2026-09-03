"""Tests for the externalized tuning config and the BS-priced option P&L.

No HTTP, no LLM — all deterministic.
"""

from __future__ import annotations

import math

import pytest

from agents.decision_agent import _build_candidates
from schemas.prediction import Phase1Bundle, PredictionResult
from schemas.risk import (
    EquityPosition,
    OptionPosition,
    PositionRecommendation,
    RiskResult,
)
from tools.decision_tools.ranking import calculate_opportunity_score
from tools.prediction_tools.technical import calculate_technical_indicators
from tools.risk_tools.greeks import black_scholes_price
from tools.risk_tools.risk_score import calculate_risk_score
from trading.backtest import _option_pnl
from tuning import TuningConfig
from utils.config import Settings

# ===========================================================================
# TuningConfig
# ===========================================================================

class TestTuningConfig:

    def test_defaults_match_prior_constants(self):
        t = TuningConfig()
        assert t.momentum_weights["macd"] == 0.25
        assert t.momentum_weights["stochastic"] == 0.10
        assert t.factor_weights["vol"] == 0.22
        assert t.signal_weights["prediction_signal"] == 0.30
        assert t.news_weight == 0.20
        assert t.direction_dampen == 0.50

    def test_from_overrides_applies(self):
        t = TuningConfig.from_overrides({"news_weight": 0.40, "conf_base": 0.55})
        assert t.news_weight == 0.40
        assert t.conf_base == 0.55

    def test_from_overrides_rejects_unknown_key(self):
        with pytest.raises(ValueError):
            TuningConfig.from_overrides({"nonsense_key": 1.0})

    def test_weights_sum_preserved(self):
        t = TuningConfig()
        assert sum(t.momentum_weights.values()) == pytest.approx(1.0)
        assert sum(t.signal_weights.values()) == pytest.approx(1.0)


# ===========================================================================
# Tuning threading through the tools
# ===========================================================================

def _divergent_technical() -> dict:
    """RSI oversold (bullish) but MACD bearish-strong (bearish)."""
    return {
        "calculate_rsi": {"rsi": 25.0, "signal": "oversold"},
        "calculate_macd": {"trend": "bearish", "momentum_strength": "strong"},
    }


class TestTuningThreading:

    def test_momentum_weights_override_flips_signal(self):
        base = calculate_technical_indicators(_divergent_technical())
        rsi_only = calculate_technical_indicators(
            _divergent_technical(),
            tuning=TuningConfig.from_overrides({
                "momentum_weights": {
                    "macd": 0.0, "adx": 0.0, "rsi": 1.0,
                    "bollinger": 0.0, "obv": 0.0, "stochastic": 0.0,
                },
            }),
        )
        macd_only = calculate_technical_indicators(
            _divergent_technical(),
            tuning=TuningConfig.from_overrides({
                "momentum_weights": {
                    "macd": 1.0, "adx": 0.0, "rsi": 0.0,
                    "bollinger": 0.0, "obv": 0.0, "stochastic": 0.0,
                },
            }),
        )
        assert rsi_only["momentum_score"] > 0
        assert macd_only["momentum_score"] < 0
        assert rsi_only["momentum_score"] > base["momentum_score"]
        assert macd_only["momentum_score"] < base["momentum_score"]

    def test_risk_score_weights_override(self):
        # With confidence-only weighting and high confidence, a very_high vol
        # regime should barely move the score (vol is down-weighted to 0).
        default = calculate_risk_score(vol_regime="very_high", confidence=0.9)
        conf_only = calculate_risk_score(
            vol_regime="very_high", confidence=0.9,
            tuning=TuningConfig.from_overrides({
                "factor_weights": {
                    "vol": 0.0, "drawdown": 0.0, "gap": 0.0,
                    "spread": 0.0, "max_loss": 0.0, "confidence": 1.0,
                },
                "base_weight": 0.0,
            }),
        )
        assert conf_only["risk_score"] < default["risk_score"]
        assert conf_only["risk_score"] < 25.0  # low bucket despite very_high vol


# ===========================================================================
# Black-Scholes option price
# ===========================================================================

class TestBlackScholesPrice:

    def test_put_call_parity(self):
        s, k, t, r, sig = 100.0, 100.0, 0.5, 0.05, 0.25
        c = black_scholes_price(s, k, t, r, sig, True)
        p = black_scholes_price(s, k, t, r, sig, False)
        assert c - p == pytest.approx(s - k * math.exp(-r * t), rel=1e-6)

    def test_atm_call_increases_with_spot(self):
        c1 = black_scholes_price(100.0, 100.0, 0.25, 0.05, 0.25, True)
        c2 = black_scholes_price(110.0, 100.0, 0.25, 0.05, 0.25, True)
        assert c2 > c1

    def test_invalid_inputs_return_zero(self):
        assert black_scholes_price(0.0, 100.0, 0.25, 0.05, 0.25, True) == 0.0
        assert black_scholes_price(100.0, 0.0, 0.25, 0.05, 0.25, True) == 0.0

    def test_expiry_returns_intrinsic_value(self):
        # ITM call/put at (or past) expiry -> intrinsic, not zero.
        assert black_scholes_price(110.0, 100.0, 0.0, 0.05, 0.25, True) == pytest.approx(10.0)
        assert black_scholes_price(90.0, 100.0, 0.0, 0.05, 0.25, False) == pytest.approx(10.0)

    def test_zero_vol_returns_forward_intrinsic(self):
        val = black_scholes_price(100.0, 100.0, 0.25, 0.05, 0.0, True)
        assert val == pytest.approx(100.0 - 100.0 * math.exp(-0.05 * 0.25))


# ===========================================================================
# Instrument preference
# ===========================================================================

def _risk_for_candidates() -> RiskResult:
    return RiskResult(
        greeks_source="black_scholes_estimated",
        stop_loss_level=145.0,
        take_profit_level=155.0,
        risk_reward_ratio=2.0,
        position_recommendation=PositionRecommendation(
            equity=EquityPosition(shares=10.0, dollar_value=1500.0),
            option=OptionPosition(contracts=1.0, premium_risk=430.0, delta_exposure=55.0),
        ),
        risk_metrics={"calculate_greeks": {
            "greeks": {"put": {"delta": -0.45}},
            "put_premium": 4.30,
            "spread_pct": 0.0,
        }},
    )


class TestInstrumentPreference:

    def test_defaults(self):
        t = TuningConfig()
        assert t.equity_only is False
        assert t.equity_score_boost == 0.0

    def test_equity_score_boost_raises_equity_candidate(self):
        cand = {"instrument": "equity", "option_type": "", "direction_matches": True,
                "confidence": 0.5, "r_r": 2.0}
        base = calculate_opportunity_score(cand)
        boosted = calculate_opportunity_score(
            cand, tuning=TuningConfig.from_overrides({"equity_score_boost": 10.0})
        )
        assert boosted["score"] > base["score"]

    def test_equity_only_skips_options(self):
        bundle = Phase1Bundle()
        pred = PredictionResult(confidence=0.6, composite_signal="bullish")
        risk = _risk_for_candidates()
        tuning = TuningConfig.from_overrides({"equity_only": True})
        candidates = _build_candidates(
            bundle, pred, risk, spot=150.0, composite_bias="bullish",
            settings=Settings(), errors=[], tuning=tuning,
        )
        assert candidates
        assert all(c["instrument"] == "equity" for c in candidates)

    def test_auto_keeps_options(self):
        bundle = Phase1Bundle()
        pred = PredictionResult(confidence=0.6, composite_signal="bullish")
        risk = _risk_for_candidates()
        candidates = _build_candidates(
            bundle, pred, risk, spot=150.0, composite_bias="bullish",
            settings=Settings(), errors=[], tuning=TuningConfig(),
        )
        instruments = {c["instrument"] for c in candidates}
        assert "option" in instruments


# ===========================================================================
# Backtest option P&L model
# ===========================================================================

class TestOptionPnl:

    def test_call_profits_when_spot_up(self):
        active = {
            "option_type": "call", "strike": 100.0, "iv": 0.25,
            "t_total": 5 / 365, "entry": 100.0, "entry_i": 1, "qty_contracts": 1.0,
        }
        pnl, pct = _option_pnl(active, exit_px=105.0, exit_i=3)
        assert pnl > 0
        assert pct > 0

    def test_put_profits_when_spot_down(self):
        active = {
            "option_type": "put", "strike": 100.0, "iv": 0.25,
            "t_total": 5 / 365, "entry": 100.0, "entry_i": 1, "qty_contracts": 1.0,
        }
        pnl, _ = _option_pnl(active, exit_px=95.0, exit_i=3)
        assert pnl > 0

    def test_theta_decays_a_flat_trade(self):
        # Unchanged spot over the full horizon -> a long option loses premium.
        active = {
            "option_type": "call", "strike": 100.0, "iv": 0.25,
            "t_total": 5 / 365, "entry": 100.0, "entry_i": 1, "qty_contracts": 1.0,
        }
        pnl, _ = _option_pnl(active, exit_px=100.0, exit_i=6)
        assert pnl < 0
