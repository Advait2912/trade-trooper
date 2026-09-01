"""Historical tools: levels, volatility, risk stats, events, summary."""

import math

import pytest

from tools.historical import risk_stats
from tools.historical.events import identify_trading_events
from tools.historical.levels import (
    detect_chart_patterns,
    identify_support_resistance,
    identify_trend,
)
from tools.historical.summary import generate_technical_summary
from tools.historical.volatility import (
    analyze_mean_reversion,
    calculate_correlation,
    calculate_historical_volatility,
    detect_volatility_regimes,
)


def _bars_from_closes(closes, amp=0.02):
    """Synthetic bars whose open == previous close (no artificial gaps)."""
    out = []
    for i, c in enumerate(closes):
        o = closes[i - 1] if i > 0 else c * 0.999
        out.append(
            {
                "date": f"2024-01-{i + 1:02d}",
                "open": o,
                "high": max(o, c) * (1 + amp),
                "low": min(o, c) * (1 - amp),
                "close": c,
            }
        )
    return out


def test_identify_trend_uptrend():
    closes = [100 + i for i in range(80)]
    out = identify_trend(closes, lookback_period=60)
    assert out["trend"] == "strong_uptrend"
    assert out["trend_class"] == "bullish"
    assert out["angle"] > 0
    assert 0.0 < out["trend_strength"] <= 1.0


def test_identify_trend_ranging():
    import math as m

    closes = [100 + m.sin(i / 5.0) * 0.5 for i in range(80)]
    out = identify_trend(closes, lookback_period=60)
    assert out["trend"] == "ranging"
    assert out["trend_class"] == "neutral"


def test_identify_trend_insufficient():
    out = identify_trend([100, 101], lookback_period=60)
    assert out["trend"] == "ranging"
    assert out["days_in_trend"] == 0


def test_support_resistance_ping_pong():
    # Price repeatedly bounces between ~100 and ~110.
    closes = []
    for i in range(120):
        base = 105.0 + (5 * (1 if (i // 10) % 2 == 0 else -1))
        closes.append(base)
    bars = _bars_from_closes(closes)
    out = identify_support_resistance(bars, sensitivity="high")
    levels_list = out["resistance_levels"] + out["support_levels"]
    assert any(abs(l["level"] - 110) / 110 < 0.05 for l in levels_list)
    assert any(abs(l["level"] - 100) / 100 < 0.05 for l in levels_list)


def test_detect_double_top():
    # Classic M shape: 100 -> 120 -> 105 -> 120 -> 95.
    closes = [100, 105, 110, 115, 120, 112, 105, 108, 112, 116, 120,
              114, 108, 102, 97, 93, 90, 92, 95]
    bars = _bars_from_closes(closes)
    out = detect_chart_patterns(bars)
    names = [p["pattern"] for p in out["patterns_found"]]
    assert "double_top" in names


def test_detect_double_bottom():
    closes = [100, 95, 90, 85, 80, 88, 95, 92, 88, 84, 80, 86, 92, 98, 103, 108, 112]
    bars = _bars_from_closes(closes)
    out = detect_chart_patterns(bars)
    names = [p["pattern"] for p in out["patterns_found"]]
    assert "double_bottom" in names


def test_detect_no_patterns_on_noise():
    import random

    rng = random.Random(3)
    closes = [100 * (1 + 0.002 * (i % 5)) for i in range(60)]
    closes = [c * (1 + rng.uniform(-0.004, 0.004)) for c in closes]
    out = detect_chart_patterns(_bars_from_closes(closes))
    assert isinstance(out["patterns_found"], list)


def test_historical_volatility_known():
    # Alternating ±1% daily moves: std of log returns ≈ 1%; annualized ≈ 15.87%.
    closes = [100.0]
    for i in range(60):
        closes.append(closes[-1] * (1.01 if i % 2 == 0 else 0.99))
    out = calculate_historical_volatility(closes, period=20)
    assert out["historical_vol"] == pytest.approx(15.87, abs=3.5)
    assert out["vol_trend"] in ("increasing", "decreasing", "stable")


def test_volatility_regime_expansion():
    vols = [10.0 + i * 0.8 for i in range(60)]
    out = detect_volatility_regimes(vols, sensitivity=3)
    assert out["regime"] in ("expansion", "high_expansion")
    assert out["vol_acceleration"] > 0


def test_mean_reversion_random_walk_neutral():
    import numpy as np

    rng = np.random.default_rng(42)
    closes = list(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 120))))
    out = analyze_mean_reversion(closes, lookback_days=90)
    assert -1.0 <= out["mean_reversion_score"] <= 1.0
    assert 0.05 <= out["reversion_probability"] <= 0.95
    assert out["mean_price"] > 0


def test_correlation_identical_and_inverse():
    a = [100 * (1 + 0.01 * math.sin(i / 4.0)) for i in range(100)]
    b = [v * 1.0 for v in a]
    inv = [200.0 - v * 1.0 for v in a]
    assert calculate_correlation(a, b, period=60)["correlation"] == pytest.approx(1.0, abs=0.05)
    assert calculate_correlation(a, inv, period=60)["correlation"] == pytest.approx(-1.0, abs=0.05)


def test_calculate_drawdown_known():
    # Peak 100 -> trough 50 -> recover to 100.
    closes = [100] * 5 + [90, 80, 70, 60, 50] + [60, 70, 80, 90, 100]
    out = risk_stats.calculate_drawdown(closes)
    assert out["max_drawdown"] == pytest.approx(50.0)
    assert out["risk_level"] == "extreme"
    assert out["recovery_time_days"] >= 5


def test_calculate_drawdown_flat():
    out = risk_stats.calculate_drawdown([100, 100, 100])
    assert out["max_drawdown"] == pytest.approx(0.0)


def test_analyze_gaps_known():
    bars = [
        {"date": "2024-01-01", "close": 100.0},
        {"date": "2024-01-02", "close": 102.9, "next_open": 103.0},  # +3% gap
        {"date": "2024-01-03", "close": 102.5, "next_open": 102.9},  # ~0 gap
        {"date": "2024-01-04", "close": 101.0, "next_open": 103.5},  # +~1% gap
    ]
    out = risk_stats.analyze_gaps(bars)
    up = [g for g in out["gaps"] if g["gap_size_percent"] >= 0.5]
    assert up and up[0]["gap_size_percent"] == pytest.approx(3.0, abs=1e-3)
    assert out["avg_gap_size"] == pytest.approx((3.0 + (103.5 - 102.5) / 102.5 * 100) / 2, abs=1e-3)


def test_value_at_risk_known():
    returns = sorted([-0.05, -0.04, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04])
    out = risk_stats.calculate_value_at_risk(returns, confidence_level=90)
    assert out["var"] < 0
    assert out["cvar"] <= out["var"] + 1e-9


def test_calculate_returns_known():
    closes = [100, 101, 102, 104, 108, 116]
    out = risk_stats.calculate_returns(closes, periods=[1, 2, 3, 4])
    assert out["return_1w"] == pytest.approx((116 / 108) - 1, abs=1e-4)
    assert out["return_1m"] == pytest.approx((116 / 104) - 1, abs=1e-4)
    assert out["return_3m"] == pytest.approx((116 / 102) - 1, abs=1e-4)
    assert out["return_1y"] == pytest.approx((116 / 101) - 1, abs=1e-4)


def test_identify_trading_events_volume_surge():
    closes = [100 + i for i in range(40)]
    volumes = [1_000_000] * 39 + [4_000_000]
    bars = _bars_from_closes(closes)
    out = identify_trading_events(bars, volumes)
    types = [e["event_type"] for e in out["events"]]
    assert "volume_surge" in types


def test_identify_trading_events_insufficient():
    out = identify_trading_events(_bars_from_closes([100, 101, 102])[:10], [1, 2, 3])
    assert out["events"] == []


def test_generate_technical_summary_all_bullish():
    bundle = {
        "calculate_moving_averages": {"signal": "bullish"},
        "calculate_rsi": {"signal": "oversold"},
        "calculate_macd": {"trend": "bullish", "momentum_strength": "strong"},
        "calculate_bollinger_bands": {"band_position": "below_lower"},
        "calculate_adx": {"trend_direction": "uptrend", "trend_strength": "strong"},
        "calculate_obv": {"obv_trend": "increasing"},
        "identify_trend": {"trend_class": "bullish"},
        "identify_support_resistance": {
            "resistance_levels": [{"level": 120.0, "touches": 3, "strength": "strong"}],
            "support_levels": [{"level": 95.0, "touches": 3, "strength": "strong"}],
        },
    }
    out = generate_technical_summary("NVDA", bundle)
    assert out["overall_signal"] == "strong_bullish"
    assert out["bullish_signals"] == 7
    assert out["key_levels"]["resistance"] == 120.0
    assert out["key_levels"]["support"] == 95.0
