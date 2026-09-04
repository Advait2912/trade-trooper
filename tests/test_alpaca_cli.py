"""Unit tests for Alpaca CLI integration tool."""

from unittest.mock import MagicMock, patch
from tools.alpaca_cli import (
    get_alpaca_binary,
    is_available,
    ensure_profile_login,
    run_cli,
    cli_doctor,
    cli_get_account,
    cli_get_clock,
    cli_list_positions,
    cli_list_orders,
)


def test_get_alpaca_binary():
    with patch("shutil.which", return_value="/custom/bin/alpaca"):
        assert get_alpaca_binary() == "/custom/bin/alpaca"
        assert is_available() is True


def test_is_available_false():
    with patch("shutil.which", return_value=None), \
         patch("os.path.exists", return_value=False):
        assert get_alpaca_binary() is None
        assert is_available() is False


def test_ensure_profile_login_success():
    with patch("tools.alpaca_cli.get_alpaca_binary", return_value="/bin/alpaca"), \
         patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        success = ensure_profile_login("test_key", "test_sec", paper=True)
        assert success is True
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert "--paper" in args
        assert "test_key" in args


def test_run_cli_json_parsing():
    with patch("tools.alpaca_cli.get_alpaca_binary", return_value="/bin/alpaca"), \
         patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='{"account_number": "PA123", "status": "ACTIVE"}',
            stderr="",
        )
        res = run_cli(["account", "get"])
        assert res["ok"] is True
        assert res["data"]["account_number"] == "PA123"


def test_cli_doctor_check():
    with patch("tools.alpaca_cli.get_alpaca_binary", return_value="/bin/alpaca"), \
         patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Trading:  https://paper-api.alpaca.markets\n✓ trading API: connected",
            stderr="",
        )
        res = cli_doctor()
        assert res["ok"] is True
        assert res["is_paper"] is True
        assert res["is_connected"] is True


def test_cli_wrappers():
    with patch("tools.alpaca_cli.run_cli") as mock_run_cli:
        mock_run_cli.return_value = {"ok": True, "data": {}}
        cli_get_account()
        mock_run_cli.assert_called_with(["account", "get"])

        cli_get_clock()
        mock_run_cli.assert_called_with(["clock"])

        cli_list_positions()
        mock_run_cli.assert_called_with(["position", "list"])

        cli_list_orders(status="closed")
        mock_run_cli.assert_called_with(["order", "list", "--status", "closed"])
