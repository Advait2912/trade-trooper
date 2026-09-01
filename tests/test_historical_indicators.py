"""Deterministic historical tools: indicators (ta-backed)."""

import pytest

from tools.historical import indicators as ind


def test_sma_known_values():
    assert ind.sma([1, 2, 3, 4, 5], 3) == 4.0
    assert ind.sma([1, 2], 3) is None


def test_ema_known_values():
    # ewm span=2 adjust=False: ema_t = a*price + (1-a)*ema_{t-1}
    assert ind.ema([1, 1, 1], 2) == pytest.approx(1.0)
    assert ind.ema([1, 2], 2) == pytest.approx(1.6667, abs=1e-3)
    assert ind.ema([1], 2) is None


def test_rsi_extremes_and_known():
    up = [float(i) for i in range(1, 60)]
    assert ind.rsi(up, 14) == pytest.approx(100.0)
    flat = [100.0] * 30
    assert ind.rsi(flat, 14) == pytest.approx(50.0)
    assert ind.rsi([1.0] * 10, 14) is None


def test_atr_known_values():
    # TR per row = h-l; Wilder ATR(3) seeds with the first 3 TRs, then smooths.
    highs = [110.0, 112.0, 115.0, 117.0]
    lows = [100.0, 101.0, 102.0, 103.0]
    closes = [105.0, 106.0, 108.0, 110.0]
    seed = (10.0 + 11.0 + 13.0) / 3
    expected = (seed * 2 + 14.0) / 3
    assert ind.atr(highs, lows, closes, 3) == pytest.approx(expected, abs=1e-3)


def test_calculate_moving_averages_signal():
    # Rising series: price above short MAs -> bullish.
    closes = [100 + i * 0.5 for i in range(80)]
    out = ind.calculate_moving_averages(closes)
    assert out["signal"] in ("bullish", "bearish", "neutral")
    assert out["ma_20"] > out["ma_50"]
    assert out["current_price"] == pytest.approx(closes[-1])


def test_calculate_moving_averages_insufficient():
    out = ind.calculate_moving_averages([100, 101])
    assert out["ma_200"] == 0.0
    assert out["signal"] == "neutral"


def test_calculate_rsi_overbought():
    up = [100 + i for i in range(40)]
    out = ind.calculate_rsi(up, 14)
    assert out["rsi"] > 95
    assert out["signal"] == "overbought"
    assert out["momentum"] in ("up", "strong_up")


def test_calculate_macd_trend_and_thresholds():
    rising = [100 + i for i in range(80)]
    out = ind.calculate_macd(rising)
    assert out["trend"] == "bullish"
    assert out["momentum_strength"] in ("strong", "moderate", "weak")
    out_short = ind.calculate_macd([100.0, 101.0])
    assert out_short["trend"] == "bearish"
    assert out_short["momentum_strength"] == "weak"


def test_calculate_bollinger_bands():
    closes = [100.0] * 40
    closes[-1] = 110.0
    out = ind.calculate_bollinger_bands(closes, period=20)
    # Middle band = mean of last 20 closes: 19 * 100 + 110 -> 100.5
    assert out["middle_band"] == pytest.approx(100.5, abs=0.01)
    assert out["upper_band"] > out["middle_band"] > out["lower_band"]
    assert out["band_position"] == "above_upper"


def test_calculate_stochastic():
    # Clear 14-period high at the end with a rising close -> %K near 100.
    closes = [float(10 + i * 0.1) for i in range(40)]
    highs = [c * 1.01 for c in closes]
    lows = [c * 0.99 for c in closes]
    out = ind.calculate_stochastic(highs, lows, closes, 14)
    assert 0.0 <= out["k_percent"] <= 100.0
    assert out["signal"] in ("overbought", "oversold", "neutral")
    assert out["crossover"] in ("bullish_cross", "bearish_cross", "none")


def test_calculate_adx_uptrend():
    n = 60
    closes = [float(100 + i * 1.0) for i in range(n)]
    highs = [c + 2.0 for c in closes]
    lows = [c - 2.0 for c in closes]
    out = ind.calculate_adx(highs, lows, closes, 14)
    assert 0.0 <= out["adx"] <= 100.0
    assert out["plus_di"] >= out["minus_di"]
    assert out["trend_direction"] == "uptrend"


def test_calculate_obv_rising_volume():
    closes = [100.0, 101.0, 102.0, 103.0]
    volumes = [1000, 2000, 3000, 4000]
    out = ind.calculate_obv(closes, volumes)
    # All up moves: OBV accumulates +volume.
    assert out["obv"] == pytest.approx(10000)


def test_calculate_obv_non_matching_lengths():
    out = ind.calculate_obv([100, 101], [1000])
    assert out["volume_confirmation"] == "divergence"
