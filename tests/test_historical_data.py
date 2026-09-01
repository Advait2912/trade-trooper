"""Alpaca historical data layer tests (respx-mocked HTTP)."""

from datetime import datetime, timedelta

import httpx
import pytest
import respx

from alpaca.client import AlpacaClient
from alpaca.historical import (
    get_dividends_history,
    get_earnings_history,
    get_price_history,
    get_volatility_history,
    parse_dividend,
)
from tests.conftest import (
    CORPORATE_ACTIONS_URL_RE,
    mock_historical,
    mock_historical_error,
)


@pytest.fixture
def settings():
    from utils.config import Settings

    return Settings(alpaca_api_key="k", alpaca_api_secret="s")


@respx.mock
async def test_price_history_parses_bars_and_passes_params(settings):
    from tests.conftest import BARS_URL_RE, historical_bars

    bars = historical_bars("AAPL", n=90)
    respx.get(BARS_URL_RE).mock(return_value=httpx.Response(200, json={"bars": bars, "symbol": "AAPL"}))

    async with AlpacaClient(settings) as client:
        result = await get_price_history(client, "AAPL", days_back=30, interval="1d")

    assert len(result) == 90
    first = result[0]
    assert first.close > 0
    assert first.volume > 0
    assert first.date.endswith("+00:00")
    # Params asserted: timeframe/limit/adjustment/sort for the desired interval.
    route = respx.routes[0]
    param_calls = [c for c in route.calls]
    assert param_calls
    request = param_calls[0].request
    assert request.url.params["timeframe"] == "1Day"
    assert request.url.params["adjustment"] == "split"
    assert request.url.params["sort"] == "asc"


@respx.mock
async def test_price_history_intraday_limit_capped(settings):
    from tests.conftest import BARS_URL_RE, historical_bars

    respx.get(BARS_URL_RE).mock(
        return_value=httpx.Response(200, json={"bars": historical_bars(n=20), "symbol": "NVDA"})
    )
    async with AlpacaClient(settings) as client:
        await get_price_history(client, "NVDA", days_back=60, interval="1m")
    request = respx.routes[0].calls[0].request
    assert request.url.params["timeframe"] == "1Min"
    # 60 days x 390 bars > cap -> 10000
    assert int(request.url.params["limit"]) == 10000


@respx.mock
async def test_price_history_unsupported_interval(settings):
    async with AlpacaClient(settings) as client:
        with pytest.raises(KeyError):
            await get_price_history(client, "NVDA", interval="2h")


@respx.mock
async def test_dividends_parse_nested_payload(settings):
    mock_historical("NVDA")
    async with AlpacaClient(settings) as client:
        divs = await get_dividends_history(client, "NVDA", years_back=2)
    assert len(divs) == 3
    assert divs[0].dividend_amount == 1.25
    assert divs[0].ex_date != ""
    request = respx.routes[1].calls[0].request
    assert request.url.params["types"] == "CASH_DIVIDEND"


@respx.mock
async def test_dividends_skip_non_dividend_actions(settings):

    payload = {
        "corporate_actions": {
            "NVDA": {
                "asset-1": [
                    {"type": "stock_split", "symbol": "NVDA", "split_ratio": "2/1"},
                    {"type": "cash_dividend", "symbol": "NVDA", "amount": "0.50", "ex_date": "2024-03-01"},
                ]
            }
        }
    }
    respx.get(CORPORATE_ACTIONS_URL_RE).mock(return_value=httpx.Response(200, json=payload))
    async with AlpacaClient(settings) as client:
        divs = await get_dividends_history(client, "NVDA", years_back=1)
    assert [d.dividend_amount for d in divs] == [0.5]


@respx.mock
async def test_earnings_history_stub(settings):
    async with AlpacaClient(settings) as client:
        assert await get_earnings_history(client, "NVDA") == []


@respx.mock
async def test_volatility_history_shape_and_python_value(settings):
    # Constant daily growth -> zero volatility; the final point must have a
    # populated 20d window.
    bars = []
    price = 100.0
    for i in range(250):
        price *= 1.0004
        bars.append(
            {
                "t": (datetime(2024, 1, 1, 13, 30) + timedelta(days=i)).isoformat(),
                "o": price * 0.999,
                "h": price * 1.001,
                "l": price * 0.999,
                "c": price,
                "v": 1_000_000,
            }
        )
    from tests.conftest import BARS_URL_RE

    respx.get(BARS_URL_RE).mock(return_value=httpx.Response(200, json={"bars": bars, "symbol": "NVDA"}))
    async with AlpacaClient(settings) as client:
        vols = await get_volatility_history(client, "NVDA", days_back=100, period=20)

    assert len(vols) == 250
    last = vols[-1]
    assert last.date != ""
    assert last.realized_vol == pytest.approx(0.0, abs=0.05)
    assert last.rolling_vol_60d == pytest.approx(0.0, abs=0.05)


@respx.mock
async def test_volatility_history_insufficient_data(settings):
    from tests.conftest import BARS_URL_RE

    # One bar only.
    respx.get(BARS_URL_RE).mock(
        return_value=httpx.Response(
            200, json={"bars": [{"t": "2024-01-01T13:30:00Z", "o": 1, "h": 1, "l": 1, "c": 1, "v": 1}], "symbol": "NVDA"}
        )
    )
    async with AlpacaClient(settings) as client:
        assert await get_volatility_history(client, "NVDA") == []


@respx.mock
async def test_historical_api_error_raises(settings):
    mock_historical_error(500)
    async with AlpacaClient(settings) as client:
        from alpaca.client import AlpacaAPIError

        with pytest.raises(AlpacaAPIError):
            await get_price_history(client, "NVDA")


def test_parse_dividend_None_for_garbage():
    assert parse_dividend(None) is None
    assert parse_dividend({"type": "cash_dividend", "amount": "abc"}) is None
    assert parse_dividend({"type": "stock_split"}) is None
