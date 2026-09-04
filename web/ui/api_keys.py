"""API credential management for the dashboard.

Paper-only: keys are validated (must be ``PK...``), written atomically to
``.env``, the Alpaca CLI profile is re-authenticated, and the runner is
optionally restarted.
"""

from __future__ import annotations

import subprocess

import streamlit as st

from web.ui.data import env_path, read_env, write_env


def _alpaca_cli() -> str | None:
    import shutil

    return shutil.which("alpaca")


def validate_key(key: str) -> tuple[bool, str]:
    key = (key or "").strip()
    if not key:
        return False, "API key is required."
    if key.startswith("AK"):
        return False, "Live trading key (AK…) detected — this app is PAPER ONLY. Use a PK… key."
    if not key.startswith("PK"):
        return False, "Key must start with PK (Alpaca paper key)."
    return True, ""


def cli_login(api_key: str, api_secret: str) -> tuple[bool, str]:
    binary = _alpaca_cli()
    if not binary:
        return False, "alpaca CLI not installed — saved to .env but CLI login skipped."
    cmd = [
        binary,
        "profile",
        "login",
        "--api-key",
        "--key",
        api_key,
        "--secret",
        api_secret,
        "--no-validate",
        "--paper",
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        return False, "alpaca CLI login timed out."
    if res.returncode == 0:
        return True, "alpaca CLI logged in (paper)."
    return False, f"alpaca CLI login failed: {res.stderr.strip()[:200]}"


def cli_doctor() -> tuple[bool, str]:
    binary = _alpaca_cli()
    if not binary:
        return False, "alpaca CLI not installed."
    try:
        res = subprocess.run([binary, "doctor"], capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        return False, "alpaca doctor timed out."
    ok = res.returncode == 0 and "paper-api.alpaca.markets" in res.stdout
    return ok, res.stdout.strip()[-400:]


def save_credentials(api_key: str, api_secret: str, auto_restart: bool = True) -> tuple[bool, str]:
    ok, msg = validate_key(api_key)
    if not ok:
        return False, msg
    if not api_secret.strip():
        return False, "API secret is required."

    write_env({"ALPACA_API_KEY": api_key.strip(), "ALPACA_API_SECRET": api_secret.strip()})

    login_ok, login_msg = cli_login(api_key.strip(), api_secret.strip())
    messages = [f"Credentials saved to {env_path()} (0600)."]
    if login_ok:
        messages.append(login_msg)
    else:
        messages.append(f"⚠ {login_msg}")

    if auto_restart:
        from web.ui.runner_control import is_running, restart

        running, _ = is_running()
        if running:
            ok_restart, msg_restart = restart(st.session_state.get("runner_universe", ["NVDA"]))
            messages.append(f"Runner restarted: {msg_restart}")
        else:
            messages.append("Runner is stopped — start it from the Runner tab to use these keys.")

    return True, " ".join(messages)


def credentials_form() -> None:
    """Render the API credentials form (masked inputs + save button)."""
    env = read_env()
    current_key = env.get("ALPACA_API_KEY", "")
    current_secret = env.get("ALPACA_API_SECRET", "")
    has_key = bool(current_key)

    st.markdown(
        "Enter your Alpaca **paper-trading** API key and secret. "
        "They are stored in `.env` (chmod 600) and used to place orders on the paper platform."
    )
    if has_key:
        st.caption(f"Current key saved: `{current_key[:4]}…{current_key[-4:]}`")

    with st.form("api_credentials"):
        api_key = st.text_input(
            "API Key (PK…)",
            value="" if has_key else "",
            type="password",
            placeholder="PK…" if has_key else "Paste your paper API key",
            help="Alpaca paper API key. Must start with PK. Live (AK) keys are rejected.",
        )
        col1, col2 = st.columns([3, 1])
        with col1:
            api_secret = st.text_input(
                "API Secret",
                value="" if has_key else "",
                type="password",
                placeholder="Paste your secret",
            )
        with col2:
            show = st.checkbox("Show", key="show_secret")
        if show and api_secret:
            st.code(api_secret)

        submitted = st.form_submit_button("💾 Save & Connect", type="primary")

    if submitted:
        if not api_key:
            api_key = current_key
        if not api_secret:
            api_secret = current_secret
        ok, msg = save_credentials(api_key, api_secret, auto_restart=True)
        if ok:
            st.success(msg)
        else:
            st.error(msg)
        if has_key and current_key:
            doc_ok, doc_msg = cli_doctor()
            if doc_ok:
                st.success("✓ Trading API: connected to paper endpoint.")
            else:
                st.warning(f"Alpaca CLI doctor: {doc_msg}")
