"""Wrap a CLI command as a UI job — writes ``state.json`` around execution.

Used by ``web/ui/jobs.py``: the Streamlit process spawns this script, which
runs the real command (inheriting stdout/stderr into the job log) and records
``running``/``done``/``error`` + the exit code in ``state.json`` so the UI can
poll without keeping a Process handle across reruns.

Usage:
    python scripts/job_runner.py --state <state.json> -- <cmd...>
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def _write(state: Path, **updates) -> None:
    data: dict = {}
    if state.exists():
        try:
            data = json.loads(state.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            data = {}
    data.update(updates)
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="job-runner")
    parser.add_argument("--state", required=True, help="path to state.json")
    parser.add_argument("--done-marker", default=None,
                        help="optional artifact whose existence forces status=done")
    parser.add_argument("cmd", nargs=argparse.REMAINDER, help="command to run")
    args = parser.parse_args(argv)

    state = Path(args.state)
    cmd = list(args.cmd)
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]  # argparse REMAINDER keeps the terminator; drop it
    if not cmd:
        _write(state, status="error", message="empty command")
        return 2

    _write(state, status="running")
    try:
        proc = subprocess.run(cmd)  # inherits stdout/stderr -> job.log
    except Exception as exc:  # noqa: BLE001
        _write(state, status="error", message=str(exc))
        return 1

    status = "done" if proc.returncode == 0 else "error"
    if args.done_marker and Path(args.done_marker).exists():
        status = "done"
    _write(state, status=status, exit_code=proc.returncode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
