"""Plotly charts for the dashboard — consistent dark palette."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

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

_CATEGORY_COLORS = [
    "#22D3EE", "#22C55E", "#F59E0B", "#8B5CF6", "#EC4899",
    "#60A5FA", "#34D399", "#F472B6", "#A3E635", "#FB923C", "#818CF8",
]


def _base_layout(title: str, height: int = 380, **kwargs) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title={"text": title, "font": {"color": PALETTE["text"]}},
        paper_bgcolor=PALETTE["bg"],
        plot_bgcolor=PALETTE["bg"],
        font={"color": PALETTE["text"]},
        height=height,
        margin={"l": 40, "r": 20, "t": 50, "b": 40},
        legend={"font": {"color": PALETTE["muted"]}},
        **kwargs,
    )
    fig.update_xaxes(gridcolor="#1F2937", tickfont={"color": PALETTE["muted"]})
    fig.update_yaxes(gridcolor="#1F2937", tickfont={"color": PALETTE["muted"]})
    return fig


def equity_curve(df: pd.DataFrame) -> go.Figure:
    fig = _base_layout("Account Equity Over Time")
    if df.empty or "ticker" not in df:
        return fig
    for i, (ticker, g) in enumerate(df.groupby("ticker")):
        g = g.sort_values("ts")
        fig.add_trace(
            go.Scatter(
                x=g["ts"], y=g["equity"], mode="lines", name=ticker,
                line={"color": _CATEGORY_COLORS[i % len(_CATEGORY_COLORS)], "width": 2},
            )
        )
    fig.update_layout(xaxis_title="Time", yaxis_title="Equity ($)")
    return fig


def cumulative_pnl(trades: pd.DataFrame) -> go.Figure:
    fig = _base_layout("Cumulative Realized P&L", height=320)
    if trades.empty:
        return fig
    t = trades.sort_values("closed_ts")
    cum = t["pnl"].cumsum()
    fig.add_trace(
        go.Scatter(
            x=t["closed_ts"], y=cum, mode="lines+markers",
            line={"color": PALETTE["accent"], "width": 2.5},
            fill="tozeroy", fillcolor="rgba(34,211,238,0.12)",
        )
    )
    fig.update_layout(xaxis_title="Close Date", yaxis_title="Cumulative P&L ($)")
    return fig


def drawdown_timeline(equity: pd.DataFrame) -> go.Figure:
    fig = _base_layout("Drawdown from Peak", height=320)
    if equity.empty:
        return fig
    e = equity.sort_values("ts")
    peak = e["equity"].cummax()
    dd = (e["equity"] - peak) / peak * 100.0
    fig.add_trace(
        go.Scatter(
            x=e["ts"], y=dd, mode="lines", name="Drawdown %",
            line={"color": PALETTE["danger"], "width": 1.8},
            fill="tozeroy", fillcolor="rgba(239,68,68,0.15)",
        )
    )
    fig.update_layout(xaxis_title="Time", yaxis_title="Drawdown (%)")
    return fig


def pnl_bar(trades: pd.DataFrame) -> go.Figure:
    fig = _base_layout("Realized P&L per Trade")
    if trades.empty:
        return fig
    t = trades.sort_values("closed_ts")
    colors = [PALETTE["success"] if p > 0 else PALETTE["danger"] for p in t["pnl"]]
    fig.add_trace(
        go.Bar(
            x=t["closed_ts"], y=t["pnl"], marker_color=colors,
            customdata=t[["ticker", "exit_reason"]],
            hovertemplate="%{x}<br>P&L $%{y:.2f}<br>%{customdata[0]} (%{customdata[1]})<extra></extra>",
        )
    )
    fig.update_layout(xaxis_title="Close Date", yaxis_title="P&L ($)")
    return fig


def winloss_heatmap(trades: pd.DataFrame) -> go.Figure:
    fig = _base_layout("Win/Loss Heatmap (ticker × day)", height=420)
    if trades.empty:
        return fig
    t = trades.copy()
    t["date"] = pd.to_datetime(t["closed_ts"]).dt.date
    pivot = t.pivot_table(index="ticker", columns="date", values="pnl", aggfunc="sum")
    fig.add_trace(
        go.Heatmap(
            z=pivot.values,
            x=[str(d) for d in pivot.columns],
            y=pivot.index,
            colorscale=[[0.0, PALETTE["danger"]], [0.5, "#1F2937"], [1.0, PALETTE["success"]]],
            zmid=0,
            colorbar={"title": "P&L ($)", "tickfont": {"color": PALETTE["muted"]}},
        )
    )
    fig.update_layout(xaxis_title="Day", yaxis_title="Ticker")
    return fig


def decision_donut(cycles: pd.DataFrame) -> go.Figure:
    fig = _base_layout("Decision Distribution", height=340)
    if cycles.empty or "decision" not in cycles:
        return fig
    counts = cycles["decision"].value_counts()
    fig.add_trace(
        go.Pie(
            labels=counts.index, values=counts.values, hole=0.55,
            marker={"colors": _CATEGORY_COLORS[: len(counts)]},
            textinfo="label+percent",
        )
    )
    return fig


def risk_score_timeline(cycles: pd.DataFrame) -> go.Figure:
    fig = _base_layout("Risk Score Over Time", height=340)
    if cycles.empty:
        return fig
    rows = []
    for _, row in cycles.iterrows():
        try:
            snap = row["snapshot"]
            if isinstance(snap, str):
                import json

                snap = json.loads(snap)
            score = snap.get("risk", {}).get("risk_score")
            if score is not None:
                rows.append({"ts": row["ts"], "ticker": row["ticker"], "score": score})
        except Exception:  # noqa: BLE001
            continue
    if not rows:
        return fig
    r = pd.DataFrame(rows).sort_values("ts")
    fig.add_trace(
        go.Scatter(
            x=r["ts"], y=r["score"], mode="lines+markers",
            line={"color": PALETTE["warning"], "width": 2},
            customdata=r["ticker"],
            hovertemplate="%{x}<br>%{customdata} risk %{y:.0f}/100<extra></extra>",
        )
    )
    fig.update_layout(xaxis_title="Time", yaxis_title="Risk score (0–100)")
    fig.add_hline(y=75, line_dash="dash", line_color=PALETTE["danger"], opacity=0.4)
    fig.add_hline(y=25, line_dash="dash", line_color=PALETTE["success"], opacity=0.4)
    return fig


def signal_vote_radar(cfg: dict, title: str = "Signal Weights") -> go.Figure:
    fig = _base_layout(title, height=380)
    weights = cfg.get("signal_weights", {})
    if not weights:
        return fig
    cats = list(weights.keys())
    vals = [weights.get(c, 0.0) for c in cats]
    fig.add_trace(
        go.Scatterpolar(r=vals + [vals[0]], theta=cats + [cats[0]], fill="toself",
                        line={"color": PALETTE["accent"]}, fillcolor="rgba(34,211,238,0.25)")
    )
    fig.update_layout(polar={"bgcolor": PALETTE["bg"],
                             "radialaxis": {"visible": True, "gridcolor": "#1F2937", "tickfont": {"color": PALETTE["muted"]}},
                             "angularaxis": {"gridcolor": "#1F2937", "tickfont": {"color": PALETTE["text"]}}})
    return fig


def momentum_weights_bar(cfg: dict) -> go.Figure:
    fig = _base_layout("Momentum Weights", height=300)
    weights = cfg.get("momentum_weights", {})
    if not weights:
        return fig
    names = list(weights.keys())
    vals = [weights[n] for n in names]
    fig.add_trace(go.Bar(x=names, y=vals, marker_color=PALETTE["accent"]))
    fig.update_layout(xaxis_title="Indicator", yaxis_title="Weight")
    return fig


def industry_pf_compare(stats: dict[str, dict]) -> go.Figure:
    fig = _base_layout("Industry Comparison (backtest)", height=420)
    if not stats:
        return fig
    names = list(stats.keys())
    pf = [stats[n].get("profit_factor", 0) for n in names]
    wr = [stats[n].get("win_rate", 0) * 100 for n in names]
    fig.add_trace(go.Bar(x=names, y=pf, name="Profit Factor", marker_color=PALETTE["accent"]))
    fig.add_trace(go.Bar(x=names, y=wr, name="Win Rate (%)", marker_color=PALETTE["success"]))
    fig.update_layout(barmode="group", xaxis_title="Industry", yaxis_title="Value")
    return fig


def equity_vs_pnl(equity: pd.DataFrame, trades: pd.DataFrame) -> go.Figure:
    fig = _base_layout("Equity vs Cumulative P&L", height=340)
    if not equity.empty:
        e = equity.sort_values("ts")
        fig.add_trace(go.Scatter(x=e["ts"], y=e["equity"], name="Equity",
                                 line={"color": PALETTE["text"], "width": 2}))
    if not trades.empty:
        t = trades.sort_values("closed_ts")
        fig.add_trace(go.Scatter(x=t["closed_ts"], y=t["pnl"].cumsum(), name="Cum P&L",
                                 line={"color": PALETTE["accent"], "width": 2}))
    fig.update_layout(xaxis_title="Time", yaxis_title="$")
    return fig
