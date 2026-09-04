"""Decision-trace extraction + Ollama chat helpers for the dashboard.

A cycle snapshot stored in the journal is a full ``FinalReport`` JSON dict;
``decision_trace_text`` compacts it into a readable block the chat model can
reason over, and ``chat_about_trade`` runs the Ollama chat endpoint on it.
"""

from __future__ import annotations

import json

CHAT_SYSTEM = (
    "You are the reasoning assistant for an autonomous Alpaca paper-trading agent. "
    "Given a decision trace (market context, prediction, risk, decision and the "
    "realized trade outcome), explain concisely what the agent saw, why it acted "
    "(or held), how the risk model sized/stopped the position, and whether the "
    "outcome matched the thesis. Be specific and honest; flag when the trade "
    "lost because the thesis failed or because of noise. Use plain language."
)


def _pick(snapshot: dict, *path: str, default=""):
    cur = snapshot
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    if cur is None:
        return default
    if isinstance(cur, (dict, list)):
        return json.dumps(cur, default=str)[:400]
    return str(cur)


def decision_trace_text(
    snapshot: dict,
    trade: dict | None = None,
    logs: list[str] | str | None = None,
) -> str:
    """Compact, human-readable decision trace from a cycle snapshot."""
    lines = ["=== DECISION TRACE ==="]
    lines.append(f"Ticker: {_pick(snapshot, 'ticker') or _pick(snapshot, 'historical', 'symbol')}")
    lines.append(f"Time: {_pick(snapshot, 'timestamp')}")
    lines.append("")
    d = snapshot.get("decision") or {}
    lines.append("-- Decision --")
    lines.append(f"  action: {d.get('trade_decision', '?')}")
    lines.append(f"  bias: {d.get('composite_bias', '?')}  confidence: {d.get('confidence_score', '?')}")
    lines.append(f"  entry ${d.get('entry_price') or d.get('price', '?')}  stop ${d.get('stop_loss', '?')}  "
                 f"target ${d.get('take_profit', '?')}")
    lines.append(f"  rationale: {d.get('rationale') or d.get('summary') or ''}")
    lines.append("")
    pred = snapshot.get("prediction") or {}
    lines.append("-- Prediction (Phase 2) --")
    lines.append(f"  composite_signal: {pred.get('composite_signal', '?')}  momentum: {pred.get('momentum_score', '?')}")
    lines.append(f"  price_forecast ${pred.get('price_forecast', '?')}  conf {pred.get('confidence', '?')}  "
                 f"iv_forecast {pred.get('iv_forecast', '?')}")
    lines.append("")
    risk = snapshot.get("risk") or {}
    lines.append("-- Risk (Phase 3) --")
    lines.append(f"  risk_score: {risk.get('risk_score', '?')} ({risk.get('risk_level', '?')})  "
                 f"r/r {risk.get('risk_reward_ratio', '?')}")
    lines.append(f"  stop ${risk.get('stop_loss_level', '?')}  target ${risk.get('take_profit_level', '?')}")
    lines.append("")
    mc = snapshot.get("market_context") or {}
    lines.append("-- Market --")
    lines.append(f"  price ${mc.get('price', '?')}  1d {mc.get('return_1d', '?')}  rsi {mc.get('rsi14', '?')}  "
                 f"trend {mc.get('trend', '?')}")
    news = snapshot.get("news") or {}
    if news:
        lines.append(f"  news: {news.get('headline', '')[:160]}")
    lines.append("")
    a = snapshot.get("analysis") or {}
    if a.get("summary"):
        lines.append(f"LLM narrative: {a.get('summary')}")
        lines.append("")

    if trade:
        lines.append("-- Realized trade --")
        lines.append(f"  instrument {trade.get('instrument', '?')}  qty {trade.get('quantity', '?')}")
        lines.append(f"  entry ${trade.get('entry_price', '?')}  exit ${trade.get('exit_price', '?')}  "
                     f"pnl ${trade.get('pnl', '?')} ({trade.get('pnl_pct', '?')})")
        lines.append(f"  exit_reason: {trade.get('exit_reason', '?')}")

    if logs:
        lines.append("")
        lines.append("-- Execution Logs --")
        if isinstance(logs, list):
            lines.append("\n".join(logs[-30:]))
        else:
            lines.append(str(logs)[-2500:])

    return "\n".join(lines)


def chat_about_trade(
    trace: str,
    question: str = "Explain what happened and whether the decision was sound.",
    history: list[dict] | None = None,
) -> str:
    """Run a one-shot Ollama chat over a decision trace; returns the reply."""
    import asyncio

    from agents.llm import OllamaClient, OllamaError
    from utils.config import load_settings

    settings = load_settings()

    async def _run() -> str:
        messages = list(history or [])
        if not messages or messages[-1].get("role") != "user":
            messages.append({"role": "user", "content": f"{trace}\n\nQuestion: {question}"})
        async with OllamaClient(settings) as client:
            return await client.chat(messages, system=CHAT_SYSTEM)

    try:
        return asyncio.run(_run())
    except OllamaError as exc:
        return f"⚠ Ollama error: {exc}"
    except Exception as exc:  # noqa: BLE001
        return f"⚠ Chat failed: {exc}"


def generate_backtest_analysis(
    kpis: dict,
    trades: any,
    params: dict,
    logs: list[str] | str | None = None,
    question: str = "Analyze this backtest run, summarizing performance metrics, key drivers, top trade outcomes, and what the execution logs reveal.",
) -> str:
    """Run Ollama reasoning over the full backtest performance + execution logs."""
    lines = ["=== BACKTEST RESULTS & EXECUTION LOGS ==="]
    lines.append(f"Parameters: start={params.get('start')} end={params.get('end')} weights_db={params.get('use_weights')}")
    lines.append(f"Trades Count: {kpis.get('trades_count')}")
    lines.append(f"Win Rate: {kpis.get('win_rate')}")
    lines.append(f"Profit Factor: {kpis.get('profit_factor')}")
    lines.append(f"Expectancy: {kpis.get('expectancy')}")
    lines.append(f"Max Drawdown: {kpis.get('max_drawdown')}")
    lines.append(f"Total P&L: ${kpis.get('total_pnl', 0):,.2f}")

    if hasattr(trades, "empty") and not trades.empty:
        lines.append("")
        lines.append("-- Top Winning Trades --")
        winners = trades.sort_values("pnl", ascending=False).head(3)
        for _, r in winners.iterrows():
            lines.append(f"  {r['ticker']} {r.get('instrument', 'option')} PnL: ${r['pnl']:.2f} ({r.get('pnl_pct', 0):.1f}%) exit={r.get('exit_reason', '')}")

        lines.append("-- Top Losing Trades --")
        losers = trades.sort_values("pnl", ascending=True).head(3)
        for _, r in losers.iterrows():
            lines.append(f"  {r['ticker']} {r.get('instrument', 'option')} PnL: ${r['pnl']:.2f} ({r.get('pnl_pct', 0):.1f}%) exit={r.get('exit_reason', '')}")

    if logs:
        lines.append("")
        lines.append("-- Execution Logs (Tail) --")
        if isinstance(logs, list):
            lines.append("\n".join(logs[-40:]))
        else:
            lines.append(str(logs)[-3000:])

    context = "\n".join(lines)
    return chat_about_trade(context, question=question)


def generate_live_narrative(
    snapshot: dict,
    logs: list[str] | str | None = None,
    question: str = "Explain the current market state, autonomous decision, risk parameters, and recent execution logs.",
) -> str:
    """Run Ollama synthesis over the live state + runner logs."""
    trace = decision_trace_text(snapshot, logs=logs)
    return chat_about_trade(trace, question=question)


def user_message(text: str) -> str:
    return text

