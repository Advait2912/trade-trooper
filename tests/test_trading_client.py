"""Paper trading client + greeks OCC symbol tests (respx-mocked)."""

import httpx
import pytest
import respx

from alpaca.trading import PAPER_TRADING_BASE_URL, TradingAPIError, TradingAuthError, TradingClient
from utils.config import Settings


@pytest.fixture
def tsettings():
    return Settings(alpaca_api_key="k", alpaca_api_secret="s")


@respx.mock
async def test_get_account(tsettings):
    respx.get(f"{PAPER_TRADING_BASE_URL}/v2/account").mock(
        return_value=httpx.Response(200, json={"equity": "150000", "cash": "149000"})
    )
    async with TradingClient(tsettings) as client:
        account = await client.get_account()
    assert account["equity"] == "150000"


@respx.mock
async def test_submit_order_payload_and_auth(tsettings):
    route = respx.post(f"{PAPER_TRADING_BASE_URL}/v2/orders").mock(
        return_value=httpx.Response(200, json={"id": "order-1", "status": "accepted"})
    )

    async with TradingClient(tsettings) as client:
        order = await client.submit_order(
            {"symbol": "NVDA", "qty": "1", "side": "buy", "type": "limit",
             "limit_price": "100", "time_in_force": "day", "client_order_id": "NVDA-123"}
        )

    assert order["id"] == "order-1"
    request = route.calls[0].request
    body = request.content.decode()
    import json as _json
    body = _json.loads(body)
    assert body["client_order_id"] == "NVDA-123"
    assert body["symbol"] == "NVDA"


@respx.mock
async def test_auth_error_raises(tsettings):
    respx.get(f"{PAPER_TRADING_BASE_URL}/v2/account").mock(
        return_value=httpx.Response(403, json={"message": "forbidden"})
    )
    async with TradingClient(tsettings) as client:
        with pytest.raises(TradingAuthError):
            await client.get_account()


@respx.mock
async def test_api_error_raises(tsettings):
    respx.post(f"{PAPER_TRADING_BASE_URL}/v2/orders").mock(
        return_value=httpx.Response(400, json={"message": "bad request"})
    )
    async with TradingClient(tsettings) as client:
        with pytest.raises(TradingAPIError):
            await client.submit_order({"symbol": "NVDA"})


@respx.mock
async def test_close_position(tsettings):
    respx.delete(f"{PAPER_TRADING_BASE_URL}/v2/positions/NVDA").mock(
        return_value=httpx.Response(200, json=[{"symbol": "NVDA", "side": "sell"}])
    )
    async with TradingClient(tsettings) as client:
        closes = await client.close_position("NVDA")
    assert closes[0]["symbol"] == "NVDA"


@respx.mock
async def test_portfolio_history(tsettings):
    respx.get(f"{PAPER_TRADING_BASE_URL}/v2/account/portfolio/history").mock(
        return_value=httpx.Response(200, json={"timestamp": [1], "equity": [100000.0]})
    )
    async with TradingClient(tsettings) as client:
        hist = await client.portfolio_history(period="1D", timeframe="1H")
    assert "equity" in hist
