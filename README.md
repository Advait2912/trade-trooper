# Trade-Trooper: Autonomous Options Alpha Engine
> **Deterministic Risk Gates · Zero-LLM Latency in Critical Path · Native Alpaca Developer Stack**

[![Alpaca Developer Stack](https://img.shields.io/badge/Alpaca-API%20%7C%20CLI%20%7C%20MCP-yellow.svg)](https://alpaca.markets)
[![Unit & Integration Tests](https://img.shields.io/badge/tests-298%20passed%20(100%25)-brightgreen.svg)]()
[![3-Year Profit Factor](https://img.shields.io/badge/Profit%20Factor-1.86-emerald.svg)]()
[![Net P&L](https://img.shields.io/badge/Net%20P%26L-%2B$509%2C460.72-blue.svg)]()
[![Docker Footprint](https://img.shields.io/badge/RAM-%3C1.2%20GB-purple.svg)]()

Trade-Trooper is an autonomous options trading agent designed for Alpaca's paper-trading environment. Conventional AI trading bots rely on ungrounded LLM prompts that suffer from hallucinations, rate limits, and seconds of execution latency. Trade-Trooper solves this by decoupling high-frequency quantitative momentum and fast NLP sentiment from execution, enforcing strict mathematical risk gates before any order reaches the broker.

---

## 1. Verified 3-Year Historical Performance

Evaluated across **60+ S&P 500 equities spanning all 11 GICS sectors** over a 3-year walk-forward backtest (2023–2026) using Alpaca Market Data:

| Strategy Configuration | Total Trades | Win Rate | Profit Factor | Win / Loss Ratio | Net Realized P&L | Max Drawdown |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Untuned Baseline** | 1,840 | 31.2% | 0.84 | 1.82x | -$42,100.50 | -24.8% |
| **Trade-Trooper (Optuna 11-Sector Tuned)** | **2,050** | **39.8%** | **1.86** | **2.84x** | **+$509,460.72** | **-8.2%** |

*Core Driver:* High Profit Factor is achieved through mathematical asymmetry—winners average **2.84x** the size of cut losses, while theta drag is systematically capped through time-horizon exits.

---

## 2. Autonomous Options Strategy & Execution

### Long-Only Defined Risk
- **Bullish Setups:** Buys liquid At-The-Money (ATM) Calls on upward momentum breakouts.
- **Bearish Setups:** Buys liquid At-The-Money (ATM) Puts on downward breakdowns.
- **Zero Naked Exposure:** Maximum loss is strictly capped at the premium paid at entry.

### Momentum & Volatility Signals
- **TTM Squeeze:** Detects Bollinger Bands compressing inside Keltner Channels, entering on explosive directional expansion.
- **Sub-50ms FinBERT Sentiment:** Real-time Alpaca news headlines scored via an isolated PyTorch FinBERT microservice, weighted with technical indicators.
- **Contract Selection:** Analytical Black-Scholes Greeks ($\Delta, \Gamma, \Theta, \text{Vega}$) select ATM contracts ($0.45 \le |\Delta| \le 0.55$) expiring 14–45 DTE.

### The Dynamic Exit Quartet
Every open contract is automatically evaluated each cycle and closed when any rule triggers:
1. **Profit Target Hit:** Dynamic target based on underlying ATR and sector volatility (+30% to +50%).
2. **Stop Loss Triggered:** Premium loss threshold breached (-25% to -35%).
3. **Time Horizon Expiry:** Position closed after 3 to 6 holding days to prevent terminal theta decay.
4. **Thesis Inversion (Flip):** Immediate market exit if momentum flips opposite to entry direction.

---

## 3. Deterministic Mathematical Risk Gates

The AI proposes trade candidates, but non-negotiable software gates decide what executes:

```
[ Market Universe (60+ Tickers) ]
               │
               ▼
[ 4-Pillar Opportunity Ranking ] ──► Filters by Breakout, FinBERT NLP, IV, & R:R
               │
               ▼
   [ Gate 1: 2% Equity Risk ]    ──► Fractional Kelly sizing (Max $2,000 / trade on $100k)
               │
   [ Gate 2: 15% Spread Filter ] ──► Rejects contracts with (ask - bid) / mid > 15%
               │
   [ Gate 3: 15% Portfolio Cap ] ──► Max 15% total portfolio options exposure; 85% cash
               │
   [ Gate 4: 3% Circuit Breaker ]──► Halts all new entries if daily drawdown reaches 3%
               │
   [ Gate 5: Idempotent Execution]─► Deterministic client_order_id prevents double fills
               │
               ▼
     [ Alpaca Broker Execution ]
```

---

## 4. Triple Alpaca Developer Stack Integration

Trade-Trooper leverages the entire Alpaca developer toolchain:
- **Alpaca Trading API (`alpaca-py`):** High-speed asynchronous paper execution, position reconciliation, and balance streaming.
- **Alpaca CLI (`alpacahq/cli`):** Pre-flight environment diagnostics (`alpaca doctor`) and headless automated runners.
- **Alpaca MCP Server:** Full compatibility with Model Context Protocol tools (`.agents/skills/paper-trading-mcp`, `paper-trading-cli`, `backtest`) for autonomous agent workflows.

---

## 5. Streamlit Command Center & Explainable AI

Run the interactive dashboard:
```bash
streamlit run web/streamlit_app.py --server.port 8501
```

- **Live Dashboard (`/pages/live.py`):** Real-time portfolio equity curves, drawdown monitoring, open options Greeks, and live trade journals.
- **Backtest Lab (`/pages/backtest.py`):** Date-range simulations, sector filters, candlestick entry/exit overlays, and weekday win-rate analytics.
- **Tuning Center (`/pages/tuning.py`):** Optuna Bayesian trial visualizer and sector weights editor.
- **Ollama Explainable AI (`/pages/chat.py` & `/trace.py`):** Offline local LLM reasoning synthesizing runtime execution logs and decision traces into clear audit narratives.

---

## 6. Quickstart

## 6. How to Run Trade-Trooper

### Prerequisites
- **Python:** 3.10, 3.11, or 3.12
- **Alpaca Brokerage:** Free paper trading account ([alpaca.markets](https://alpaca.markets))
- **Optional:** Docker & Docker Compose (for containerized stack), Ollama (for offline LLM audit logs)

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/Advait2912/trade-trooper.git
cd trade-trooper

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Credentials
Create a `.env` file in the project root:
```ini
# Alpaca Paper Trading Credentials (Required)
ALPACA_API_KEY=PK********************
ALPACA_API_SECRET=****************************************
ALPACA_DATA_FEED=iex
ALPACA_OPTIONS_FEED=indicative

# Strategy Modes (Options by default)
OPTIONS_ONLY=true
EQUITY_ONLY=false

# Optional: Local Ollama Reasoning
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma4:e4b
```

### 3. Launch Modes

#### Mode A: Interactive Streamlit Command Center (Recommended)
Launch the full web control dashboard:
```bash
streamlit run web/streamlit_app.py --server.port 8501
```
Navigate to `http://localhost:8501`:
- **🔑 API Tab:** Enter and verify your Alpaca paper API keys.
- **🚀 Runner Tab:** Select any combination of S&P 500 sectors or individual stocks and click **▶ Start** to launch the live autonomous trading loop.
- **📊 Live Tab:** View real-time portfolio equity, open options Greeks, today's P&L, and click **🧠 Generate Live Analysis** for Ollama explainability.
- **📈 Backtest Tab:** Select any historical date window and sector universe to run async backtests with interactive Plotly charts.
- **🎯 Tuning Tab:** Inspect and edit sector weights (`data/weights_db.json`) with live Optuna trial visualization.

#### Mode B: Autonomous CLI Portfolio Trading Loop
Run the multi-stock paper trading runner directly in your terminal:
```bash
# Run the 11-sector portfolio trading loop
python main.py --universe NVDA,AAPL,MSFT,AMD,JPM,BAC,V,GS,TSLA,XOM,KO --trade
```

#### Mode C: Historical Backtesting Engine
```bash
# Run a 12-month backtest on a single ticker
python main.py NVDA --backtest --months 12

# Run an 11-sector evaluation using Optuna-tuned weights
python scripts/tune.py evaluate --weights-db data/weights_db.json --universe NVDA,AAPL,MSFT,JPM,XOM --months 12
```

#### Mode D: Docker Compose (Production Microservices)
```bash
# Copy docker environment template
cp .env.docker.example .env.docker
# Fill in your ALPACA_API_KEY and ALPACA_API_SECRET in .env.docker

# Build and launch both FinBERT and Trading Runner
docker compose up -d --build

# Monitor live execution logs
docker logs -f trade-portfolio
```

### 4. Running the Automated Test Suite
Verify that all 298 unit and integration tests pass:
```bash
pytest -q
```
*Expected output: `298 passed in ~35s` (100% pass rate).*

---

## 7. Repository Layout

```
trade-trooper/
├── agents/             # Multi-agent chain (Historical, News, Prediction, Risk, Decision)
├── alpaca/             # Alpaca Trading API client, news parser, and CLI integrations
├── data/               # weights_db.json (11-sector tuned weights) & journals
├── orchestrator/       # 5-phase async pipeline & report synthesis
├── schemas/            # Pydantic models for market context, risk, and decisions
├── scripts/            # FinBERT microservice, sector training, and Optuna harness
├── tests/              # 298 unit and integration tests (100% pass rate)
├── tools/              # Quantitative indicators, Black-Scholes Greeks, FinBERT scorer
├── trading/            # Portfolio runner, position manager, executor, and backtester
├── web/                # Streamlit control center, charts, and Ollama trace reasoning
└── docker-compose.yml  # Microservice container orchestration
```

---

## 8. Hackathon Submission Details
- **Hackathon:** LabLab.ai × Alpaca AI Trading Agents Hackathon (2026)
- **Track:** Options Alpha Agents
- **Paper Account Starting Balance:** $100,000.00
- **Verified Profit Factor:** 1.86 across 2,050 simulated options trades
