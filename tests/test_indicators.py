"""Deterministic indicator correctness."""

from tools.historical import indicators as ind


def test_sma():
    assert ind.sma([1, 2, 3, 4, 5], 3) == 4.0
    assert ind.sma([1, 2], 3) is None


def test_rsi_extremes():
    up = [float(i) for i in range(1, 30)]
    assert ind.rsi(up, 14) == 100.0

    down = [float(i) for i in range(30, 1, -1)]
    assert ind.rsi(down, 14) == 0.0

    assert ind.rsi([1.0, 2.0], 14) is None  # not enough data


def test_rsi_flat_is_neutral():
    flat = [10.0] * 20
    # A perfectly flat series has no gains or losses: RSI is undefined, so the
    # convention here is neutral (50.0) rather than 100.
    assert ind.rsi(flat, 14) == 50.0


def test_atr():
    highs = [11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0]
    lows = [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0]
    closes = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0]
    assert ind.atr(highs, lows, closes, 14) is not None
    assert ind.atr(highs[:3], lows[:3], closes[:3], 14) is None


def test_return_over():
    r1 = ind.return_over([100.0, 101.0], 1)
    assert r1 is not None
    assert abs(r1 - 0.01) < 1e-9

    r2 = ind.return_over([100.0, 105.0, 110.0], 2)
    assert r2 is not None
    assert abs(r2 - 0.10) < 1e-9

    assert ind.return_over([100.0], 1) is None


def test_volume_vs_average():
    vols = [1000.0] * 20
    assert ind.volume_vs_average(vols, 20) == 1.0

    vols2 = [1000.0] * 19 + [2000.0]
    ratio = ind.volume_vs_average(vols2, 20)
    assert ratio is not None
    assert abs(ratio - 2000.0 / 1050.0) < 1e-9


def test_volatility():
    closes = [float(i) for i in range(1, 30)]
    v = ind.volatility(closes, 20)
    assert v is not None and v >= 0
    assert ind.volatility([1.0, 2.0], 20) is None
