"""Market data computation, stale-data labeling, and indicator integration."""

from datetime import datetime, timedelta, timezone

from alpaca.market_data import compute_market_data, suggest_trend


def _bars(n: int = 60) -> list[dict]:
    base = datetime(2026, 1, 5, 13, 0, tzinfo=timezone.utc)  # a Monday
    price = 100.0
    out = []
    for i in range(n):
        day = base + timedelta(days=i)
        c = round(price * 1.01, 2)
        out.append(
            {
                "t": day.isoformat(),
                "o": round(price, 2),
                "h": round(c * 1.005, 2),
                "l": round(price * 0.995, 2),
                "c": c,
                "v": 1_000_000 + i,
            }
        )
        price = c
    return out


def _snapshot(bars: list[dict], as_of: str) -> dict:
    return {
        "symbol": "NVDA",
        "latestTrade": {"t": as_of, "p": bars[-1]["c"], "s": 500000},
        "dailyBar": bars[-1],
        "prevDailyBar": bars[-2],
    }


def test_indicators_computed_deterministically():
    bars = _bars(60)
    md = compute_market_data(_snapshot(bars, bars[-1]["t"]), bars, "NVDA")
    assert md.price == bars[-1]["c"]
    assert md.sma20 > 0
    assert md.sma50 > 0
    assert md.sma20 > md.sma50  # rising prices
    assert md.rsi14 == 100.0
    assert md.return_1d > 0
    assert md.volume_vs_average > 0


def test_insufficient_history_yields_zeros():
    bars = _bars(3)
    md = compute_market_data(_snapshot(bars, bars[-1]["t"]), bars, "NVDA")
    assert md.sma50 == 0.0
    assert md.rsi14 == 0.0
    assert md.atr14 == 0.0


def test_stale_data_labeled():
    bars = _bars(60)
    now = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)
    old = (now - timedelta(hours=48)).isoformat()
    md = compute_market_data(_snapshot(bars, old), bars, "NVDA", now=now)
    assert md.stale is True


def test_market_closed_labeled():
    bars = _bars(60)
    # A Saturday (2026-05-02) is a non-trading day.
    saturday = "2026-05-02T13:00:00+00:00"
    md = compute_market_data(_snapshot(bars, saturday), bars, "NVDA")
    assert md.market_closed is True


def test_suggest_trend():
    from schemas.market import MarketData

    md = MarketData(price=110.0, sma20=105.0, sma50=100.0, rsi14=60.0)
    assert suggest_trend(md).value == "bullish"

    md2 = MarketData(price=90.0, sma20=95.0, sma50=100.0, rsi14=40.0)
    assert suggest_trend(md2).value == "bearish"
