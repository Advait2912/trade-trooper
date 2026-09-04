"""Alpaca CLI Integration Tool.

Enables Trade-Trooper to execute diagnostic preflight checks, query accounts,
inspect positions, and route orders through the official Alpaca CLI tool
(https://github.com/alpacahq/cli), fully adhering to the Alpaca Paper Trading CLI Skill.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from typing import Any

log = logging.getLogger("market_intel_agent.alpaca_cli")


def get_alpaca_binary() -> str | None:
    """Locate the official alpaca CLI binary if installed on the system."""
    found = shutil.which("alpaca")
    if found:
        return found
    user_local = os.path.expanduser("~/.local/bin/alpaca")
    if os.path.exists(user_local) and os.access(user_local, os.X_OK):
        return user_local
    usr_local = "/usr/local/bin/alpaca"
    if os.path.exists(usr_local) and os.access(usr_local, os.X_OK):
        return usr_local
    return None


def is_available() -> bool:
    """Check if the alpaca CLI is installed and executable."""
    return get_alpaca_binary() is not None


def ensure_profile_login(api_key: str, api_secret: str, paper: bool = True) -> bool:
    """Configure credentials in the Alpaca CLI profile if not already configured."""
    binary = get_alpaca_binary()
    if not binary or not api_key or not api_secret:
        return False

    cmd = [
        binary,
        "profile",
        "login",
        "--api-key",
        "--key",
        api_key.strip(),
        "--secret",
        api_secret.strip(),
        "--no-validate",
    ]
    if paper:
        cmd.append("--paper")

    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return res.returncode == 0
    except Exception as exc:
        log.warning("Alpaca CLI profile login failed: %s", exc)
        return False


def run_cli(args: list[str], timeout: int = 15) -> dict[str, Any]:
    """Execute an alpaca CLI command and return structured JSON output."""
    binary = get_alpaca_binary()
    if not binary:
        return {"ok": False, "error": "alpaca CLI binary not found"}

    cmd = [binary] + args
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if res.returncode != 0:
            return {
                "ok": False,
                "error": res.stderr.strip() or f"Command failed with code {res.returncode}",
                "exit_code": res.returncode,
            }
        stdout = res.stdout.strip()
        try:
            data = json.loads(stdout)
            return {"ok": True, "data": data, "raw": stdout}
        except json.JSONDecodeError:
            return {"ok": True, "data": stdout, "raw": stdout}
    except Exception as exc:
        log.warning("Alpaca CLI execution error: %s", exc)
        return {"ok": False, "error": str(exc)}


def cli_doctor() -> dict[str, Any]:
    """Run `alpaca doctor` to verify connectivity and validate paper trading target."""
    binary = get_alpaca_binary()
    if not binary:
        return {"ok": False, "error": "alpaca CLI binary not found"}

    try:
        res = subprocess.run([binary, "doctor"], capture_output=True, text=True, timeout=10)
        is_paper = "https://paper-api.alpaca.markets" in res.stdout
        is_connected = "trading API: connected" in res.stdout
        return {
            "ok": res.returncode == 0 and is_paper,
            "is_paper": is_paper,
            "is_connected": is_connected,
            "stdout": res.stdout.strip(),
            "stderr": res.stderr.strip(),
            "exit_code": res.returncode,
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def cli_get_account() -> dict[str, Any]:
    """Fetch account details using `alpaca account get`."""
    return run_cli(["account", "get"])


def cli_get_clock() -> dict[str, Any]:
    """Fetch market clock using `alpaca clock`."""
    return run_cli(["clock"])


def cli_list_positions() -> dict[str, Any]:
    """Fetch open positions using `alpaca position list`."""
    return run_cli(["position", "list"])


def cli_list_orders(status: str = "open") -> dict[str, Any]:
    """Fetch orders using `alpaca order list --status <status>`."""
    return run_cli(["order", "list", "--status", status])
