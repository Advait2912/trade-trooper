"""Runner control — start/stop/restart/status of the paper-trading loop.

Manages the runner subprocess via ``data/runner.pid`` and ``data/runner.log``.
All commands run relative to the repo root.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

from web.ui.data import read_env, repo_root, runner_log_file, runner_pid_file, write_env

DEFAULT_UNIVERSE = ["NVDA", "AAPL", "MSFT", "AMD", "JPM", "BAC", "V", "GS", "TSLA", "XOM", "KO"]


def _read_pid() -> int | None:
    p = runner_pid_file()
    if not p.exists():
        return None
    try:
        return int(p.read_text(encoding="utf-8").strip())
    except (ValueError, OSError):
        return None


def _alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def is_running() -> tuple[bool, int | None]:
    pid = _read_pid()
    if pid is None:
        return False, None
    return (_alive(pid), pid)


def status() -> dict:
    running, pid = is_running()
    env = read_env()
    universe = env.get("RUNNER_UNIVERSE", ",".join(DEFAULT_UNIVERSE))
    last_lines: list[str] = []
    log_file = runner_log_file()
    if log_file.exists():
        try:
            last_lines = log_file.read_text(encoding="utf-8").splitlines()[-5:]
        except OSError:
            last_lines = []
    return {
        "running": running,
        "pid": pid,
        "universe": [t for t in universe.split(",") if t],
        "started_at": env.get("RUNNER_STARTED_AT", ""),
        "last_log_lines": last_lines,
    }


def start(universe: list[str]) -> tuple[bool, str]:
    running, pid = is_running()
    if running:
        return False, f"Runner already running (PID {pid}). Stop it first or use restart."

    universe = [t.strip().upper() for t in universe if t.strip()]
    if not universe:
        return False, "No tickers selected."

    env = read_env()
    if not env.get("ALPACA_API_KEY") or not env.get("ALPACA_API_SECRET"):
        return False, "API key/secret not set. Enter them in the API Credentials tab first."
    if env.get("ALPACA_API_KEY", "").startswith("AK"):
        return False, "Live (AK…) key detected — paper trading only. Use a PK… key."

    write_env(
        {
            "TRADING_ENABLED": "true",
            "RUNNER_UNIVERSE": ",".join(universe),
            "RUNNER_STARTED_AT": datetime.now(timezone.utc).isoformat(),
        }
    )

    cmd = [sys.executable, "main.py", "--universe", ",".join(universe), "--trade"]
    log_file = runner_log_file()
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "a", encoding="utf-8") as out:
        proc = subprocess.Popen(
            cmd,
            cwd=str(repo_root()),
            stdout=out,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    runner_pid_file().write_text(str(proc.pid), encoding="utf-8")
    time.sleep(2)
    if not _alive(proc.pid):
        return False, "Runner exited immediately — check data/runner.log."
    return True, f"Runner started (PID {proc.pid}) with {len(universe)} tickers."


def stop() -> tuple[bool, str]:
    running, pid = is_running()
    if not running:
        return True, "Runner is already stopped."
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        pass
    for _ in range(10):
        if not _alive(pid):
            break
        time.sleep(0.5)
    else:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
    write_env({"TRADING_ENABLED": "false"})
    try:
        runner_pid_file().unlink()
    except OSError:
        pass
    return True, "Runner stopped."


def restart(universe: list[str]) -> tuple[bool, str]:
    stop()
    return start(universe)
