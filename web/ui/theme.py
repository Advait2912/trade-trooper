"""UI theme helpers — Claude-aligned palette, KPI tiles, cards, status pills."""

from __future__ import annotations

import streamlit as st

PALETTE = {
    "bg": "#0B1220",
    "card": "#111827",
    "accent": "#22D3EE",
    "success": "#22C55E",
    "danger": "#EF4444",
    "warning": "#F59E0B",
    "muted": "#9CA3AF",
    "text": "#E5E7EB",
}

_PILL_COLORS = {
    "running": ("#22C55E", "#052E16"),
    "stopped": ("#9CA3AF", "#1F2937"),
    "error": ("#EF4444", "#450A0A"),
    "warning": ("#F59E0B", "#451A03"),
    "info": ("#22D3EE", "#083344"),
}


def inject_theme_css() -> None:
    """Apply the global theme CSS once."""
    st.markdown(
        f"""
        <style>
        :root {{
            --tt-bg: {PALETTE['bg']};
            --tt-card: {PALETTE['card']};
            --tt-accent: {PALETTE['accent']};
        }}
        .stApp {{
            background-color: {PALETTE['bg']};
        }}
        .tt-card {{
            background-color: {PALETTE['card']};
            border: 1px solid #1F2937;
            border-radius: 12px;
            padding: 16px 20px;
            margin-bottom: 16px;
        }}
        .tt-kpi {{
            background-color: {PALETTE['card']};
            border: 1px solid #1F2937;
            border-radius: 12px;
            padding: 14px 18px;
        }}
        .tt-kpi .label {{
            color: {PALETTE['muted']};
            font-size: 0.8rem;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }}
        .tt-kpi .value {{
            color: {PALETTE['text']};
            font-size: 1.6rem;
            font-weight: 700;
            font-variant-numeric: tabular-nums;
        }}
        .tt-kpi .delta-up {{ color: {PALETTE['success']}; font-size: 0.85rem; }}
        .tt-kpi .delta-down {{ color: {PALETTE['danger']}; font-size: 0.85rem; }}
        .tt-pill {{
            display: inline-block;
            padding: 3px 12px;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 600;
            letter-spacing: 0.03em;
        }}
        .tt-section-title {{
            color: {PALETTE['text']};
            font-size: 1.15rem;
            font-weight: 650;
            margin: 0 0 2px 0;
        }}
        .tt-section-sub {{
            color: {PALETTE['muted']};
            font-size: 0.85rem;
            margin: 0 0 12px 0;
        }}
        div[data-testid="stMetric"] {{
            background-color: {PALETTE['card']};
            border: 1px solid #1F2937;
            border-radius: 12px;
            padding: 12px 16px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def section(icon: str, title: str, subtitle: str | None = None) -> None:
    """Render a consistent section header."""
    st.markdown(
        f'<div class="tt-section-title">{icon} {title}</div>'
        f'<div class="tt-section-sub">{subtitle or ""}</div>',
        unsafe_allow_html=True,
    )


def card(body: str, height: int | None = None) -> None:
    """Render a card with arbitrary HTML body."""
    style = f"height: {height}px;" if height else ""
    st.markdown(f'<div class="tt-card" style="{style}">{body}</div>', unsafe_allow_html=True)


def kpi_tile(label: str, value: str, delta: float | None = None) -> None:
    """Render a KPI tile: label + big value + optional delta arrow."""
    delta_html = ""
    if delta is not None:
        cls = "delta-up" if delta >= 0 else "delta-down"
        arrow = "▲" if delta >= 0 else "▼"
        delta_html = f'<div class="{cls}">{arrow} {abs(delta):.2f}</div>'
    st.markdown(
        f'<div class="tt-kpi"><div class="label">{label}</div>'
        f'<div class="value">{value}</div>{delta_html}</div>',
        unsafe_allow_html=True,
    )


def status_pill(text: str, kind: str = "info") -> None:
    """Render a colored status pill.  kind: running|stopped|error|warning|info."""
    fg, bg = _PILL_COLORS.get(kind, _PILL_COLORS["info"])
    st.markdown(
        f'<span class="tt-pill" style="color:{fg};background-color:{bg};">{text}</span>',
        unsafe_allow_html=True,
    )
