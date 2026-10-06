# ⚡ Alpha Engine Pro

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-live-red)](https://netflix-stock-prediction-h4e4qxevbfjweltuumcxeb.streamlit.app)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![SEBI Compliant](https://img.shields.io/badge/SEBI-Algo%20Compliant-green)](src/compliance_sebi.py)

> **Institutional Quant Research & Execution Platform**

An end-to-end ML + agentic-AI quant platform: multi-strategy signal generation, conformal prediction intervals, SEBI-compliant audit trail, verifiable track record, and a 4-agent Research Copilot that synthesizes RAG-backed research notes — all in a fully interactive Streamlit dashboard.

> ⚠️ **Disclaimer:** Educational / research use only. Not financial advice.

**[▶ Live App](https://netflix-stock-prediction-h4e4qxevbfjweltuumcxeb.streamlit.app)** · **[Results](RESULTS.md)** · **[Contributing](CONTRIBUTING.md)**

---

## 🆕 What's New in v2.0

- 🧩 **Multi-Strategy Framework** — `strategy_registry.py` lets you define, version, and compare named strategies (ticker universe, feature set, model config, risk params) side-by-side; no longer limited to a single `DEFAULT_TICKER=NFLX`
- 🤖 **Research Copilot 4-Agent Pipeline** — `RetrieverAgent → ToolAgent → WriterAgent → HITLRouter` with LangGraph orchestration, ChromaDB RAG, Groq LLaMA3 generation, and a human-in-the-loop gate before any `/execute` call
- ⚖️ **SEBI Compliance Module** — `compliance_sebi.py` covers 8 SEBI algo-trading norms with an evidence+remediation JSON report (audit trail, kill-switch, order-to-trade ratio, and more)
- 📊 **Verifiable Track Record Layer** — `track_record.py` logs every signal with timestamp, outcome, Sharpe/Sortino/Calmar/max-drawdown, exposed on a `/track-record` endpoint and a dedicated dashboard tab
- 🎨 **White-Label Branding System** — `branding_config.py` loads name, logo, and color theme from env/config; per-tenant strategy namespace so the same codebase can be resold to multiple clients
- 🗂️ **5 New Streamlit Tabs** — Strategy Lab, Research Copilot, Track Record, Compliance, and Observability added to the existing 10-tab dashboard
- 🧪 **RAGAS + Regime Eval Scripts** — `eval_ragas_copilot.py` scores Copilot citation faithfulness; `eval_regime_robustness.py` reports directional accuracy and RMSE across HMM-detected bull/bear/sideways regimes

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Alpha Engine Pro                             │
├─────────────────────────────────────────────────────────────────────┤
│  DATA LAYER                                                         │
│  CSV · yfinance · Alpha Vantage · Alpaca (multi-ticker)            │
│                        ↓                                            │
│  FEATURE ENGINEERING                                                │
│  51 technical indicators (RSI, MACD, BB, ATR, Stochastic, CCI …)  │
│                        ↓                                            │
│  ML MODELING                                                        │
│  ManualStackingRegressor (XGBoost + LightGBM + RF + ExtraTrees     │
│  → Ridge meta-learner)  +  Conformal Prediction CI (90% coverage) │
│                        ↓                                            │
│  RISK MANAGER                                                       │
│  ATR stop-loss · Kelly sizing · VaR/CVaR · circuit breaker         │
│                        ↓                                            │
│  STRATEGY REGISTRY                                                  │
│  Named strategies · per-strategy model version · parallel compare  │
│                        ↓                                            │
│  RESEARCH COPILOT                                                   │
│  RetrieverAgent (ChromaDB RAG)                                      │
│    → ToolAgent (Model + SHAP + Risk state)                         │
│    → WriterAgent (Groq LLaMA3 / rule-based fallback)               │
│    → HITLRouter (human gate above position-size threshold)          │
│                        ↓                                            │
│  COMPLIANCE (SEBI)   ·   TRACK RECORD   ·   OBSERVABILITY         │
│                        ↓                                            │
│  STREAMLIT UI  (15 interactive Plotly tabs)                        │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🗂️ Tabs

| # | Tab | Description |
|---|-----|-------------|
| 1 | 🕯 Market Overview | Live candlestick · MA 20/50/200 · RSI · MACD · Bollinger Bands |
| 2 | 🔮 Predict | Live data editor → next-day return + conformal interval |
| 3 | 📈 Backtesting | Equity curve · Rolling Sharpe · Drawdown |
| 4 | 📋 Paper Trade | Day-by-day simulation · cumulative PnL · pred vs actual scatter |
| 5 | 🧠 Sentiment | VADER scoring · bar + pie charts · fallback sample data |
| 6 | ⚠️ Risk | Kelly sizing · ATR stop-loss · VaR/CVaR · volatility surface |
| 7 | 🔬 Drift Monitor | PSI + KS drift detection across all 51 features |
| 8 | 🔍 Explainability | Per-model feature importance · target correlation |
| 9 | 🤖 AI Narrative | RAG narrative · sentiment gauge · CI chart · source relevance · RAGAS eval |
| 10 | 🏗 Architecture | Pipeline diagram · design decisions · limitations |
| 11 | 🧪 Strategy Lab | Define, compare, and register named strategies side-by-side |
| 12 | 🔬 Research Copilot | Chat interface · HITL gate · prediction gauge · SHAP bar · source relevance |
| 13 | 📊 Track Record | Equity curve · drawdown series · monthly PnL bar · signal log table |
| 14 | ⚖️ Compliance | SEBI check details (expandable) · OTR gauge · JSON report download |
| 15 | 📡 Observability | Agent trace JSONL viewer · latency box plots · Prometheus-style metrics |

All charts are **fully interactive Plotly** (zoom, pan, hover tooltip, PNG export).

---

## 🚀 Quick Start

```bash
git clone https://github.com/SumedhPatil1507/netflix-stock-prediction.git
cd netflix-stock-prediction

python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app/app.py
```

Open <http://localhost:8501>. The app loads `models/model.pkl` and the bundled feature cache — no retraining needed.

### Full AI + Copilot deps (already in `requirements.txt`)

```bash
pip install langchain langchain-openai langgraph langfuse \
            chromadb sentence-transformers ragas openai groq
```

All these are optional — every tab degrades gracefully with a helpful install prompt when a dep is absent.

---

## 🔑 Environment Variables

Copy `.env.example` to `.env` and fill in the values you need. Never commit `.env`.

| Name | Required | Description |
|------|----------|-------------|
| `OPENAI_API_KEY` | Optional | LLM narratives via GPT-4o-mini; rule-based fallback if absent |
| `GROQ_API_KEY` | Optional | Groq LLaMA3-8b-8192 for Research Copilot `WriterAgent`; fallback if absent |
| `LANGFUSE_PUBLIC_KEY` | Optional | Langfuse cloud trace dashboard; JSONL fallback always active |
| `LANGFUSE_SECRET_KEY` | Optional | Langfuse secret (pair with public key) |
| `LANGFUSE_HOST` | Optional | Self-hosted Langfuse endpoint (default: cloud) |
| `HITL_THRESHOLD_USD` | Optional | USD notional above which HITL sign-off is required (default: 10000) |
| `APP_NAME` | Optional | White-label app name shown in UI title and page config |
| `TENANT_ID` | Optional | Tenant namespace for per-client strategy isolation |
| `PRIMARY_COLOR` | Optional | Hex color for white-label theme (e.g. `#1DB954`) |
| `ALLOWED_TICKERS` | Optional | Comma-separated list restricting which tickers a tenant may trade |

---

## 🤖 Research Copilot Pipeline

```
User Query / Signal Trigger
        │
        ▼
  RetrieverAgent
  ┌─────────────────────────────────────────────────┐
  │  ChromaDB semantic search over:                 │
  │  · Earnings-call transcripts (10-K/10-Q chunks) │
  │  · Recent financial news                        │
  │  Returns: top-N {text, metadata, distance}      │
  └─────────────────────────────────────────────────┘
        │
        ▼
  ToolAgent
  ┌─────────────────────────────────────────────────┐
  │  Pulls live state from existing modules:        │
  │  · Model prediction + conformal interval        │
  │  · SHAP feature drivers (top-5)                 │
  │  · Risk manager: ATR stop, Kelly %, heat        │
  └─────────────────────────────────────────────────┘
        │
        ▼
  WriterAgent
  ┌─────────────────────────────────────────────────┐
  │  Groq LLaMA3-8b-8192 (or rule-based fallback)  │
  │  Synthesizes cited research note:               │
  │  "Model is bullish, CI [X–Y], primary driver   │
  │   RSI divergence. Supporting context from Q3   │
  │   earnings: [cited excerpt]."                  │
  └─────────────────────────────────────────────────┘
        │
        ▼
  HITLRouter
  ┌─────────────────────────────────────────────────┐
  │  If position_size > HITL_THRESHOLD_USD:         │
  │    → Writes pending record to hitl_pending.json │
  │    → /execute blocked until human approval      │
  │  Otherwise: signal flows through immediately    │
  └─────────────────────────────────────────────────┘
        │
        ▼
  Research Note (cited, signed-off, audit-logged)
```

---

## ⚖️ SEBI Compliance

`compliance_sebi.py` maps to SEBI circular requirements for algorithmic trading systems. Each check produces an evidence + remediation JSON entry.

| # | Check | SEBI Reference |
|---|-------|----------------|
| 1 | Audit trail (all orders logged with timestamp + strategy ID) | SEBI Circular CIR/MRD/DP/09/2012 |
| 2 | Kill-switch / circuit breaker (halt on daily loss limit) | SEBI Algo Guidelines §4 |
| 3 | Order-to-trade ratio logging (OTR threshold enforcement) | NSE/BSE co-location guidelines |
| 4 | Price band checks (no orders outside ±circuit limits) | SEBI Circular §6 |
| 5 | Latency disclosure (average execution latency logged) | SEBI Algo §7 |
| 6 | Strategy disclosure document present | SEBI Algo §8 |
| 7 | Risk parameter file version-controlled | SEBI Circular §5 |
| 8 | Annual system audit log retained ≥ 5 years | SEBI §10 |

Run a compliance check:

```python
from src.compliance_sebi import SEBICompliance
report = SEBICompliance().run_all_checks()
print(report)  # JSON with status, evidence, remediation per check
```

---

## 📊 Track Record Metrics

`track_record.py` logs every paper/live signal and computes rolling performance statistics exposed at `/track-record`.

| Metric | Description |
|--------|-------------|
| `sharpe` | Annualized Sharpe ratio (rolling 252-day) |
| `sortino` | Sortino ratio using downside deviation |
| `calmar` | Annualized return ÷ max drawdown |
| `max_drawdown` | Peak-to-trough equity drawdown (%) |
| `profit_factor` | Gross profit ÷ gross loss |
| `trailing_sharpe_63d` | 63-trading-day (≈ 1 quarter) Sharpe |
| `win_rate_pct` | Percentage of signals with positive realized return |
| `n_signals` | Total logged signals (paper + live) |
| `equity_curve` | Day-by-day cumulative PnL series (Plotly) |

---

## 🎨 White-Label Deployment

`branding_config.py` reads display settings from environment variables (or `branding.yaml`), enabling zero-code-change reskinning for each client:

```bash
export APP_NAME="Quant Edge Pro"
export PRIMARY_COLOR="#005EB8"
export TENANT_ID="client_abc"
export ALLOWED_TICKERS="RELIANCE,TCS,INFY"
streamlit run app/app.py
```

Per-tenant strategy isolation means each `TENANT_ID` gets its own namespace in `strategy_registry.json` — strategies, model versions, and track records never bleed across tenants.

---

## 🧪 Eval Scripts

### Research Copilot RAGAS Eval

```bash
python eval_ragas_copilot.py
# → outputs/eval_copilot_ragas.json
```

Evaluates `CopilotGraph` for each ticker in `["NFLX", "AAPL"]`. Reports RAGAS faithfulness score and source count. Falls back to lexical-overlap scoring when RAGAS is not installed.

### Regime Robustness Report

```bash
python eval_regime_robustness.py
# → outputs/eval_regime_robustness.json
```

Loads the trained stacking model, splits the dataset by HMM-detected regime (bull/bear/sideways), and reports directional accuracy (%) and RMSE for each regime. Use this output in the sales pitch deck to show the model generalizes across market conditions.

---

## 💼 Why Buyers Pay For This

| Feature | Why It Matters |
|---------|---------------|
| **Verifiable Track Record** | Buyers pay for demonstrated live performance — not backtest claims. Signal log with Sharpe/Sortino/Calmar from day one. |
| **SEBI Compliance Module** | The single deciding factor between a hobby script and a deployable product for Indian markets. |
| **HITL Gate** | No serious institutional buyer accepts fully autonomous order placement. Human sign-off is non-negotiable. |
| **White-Label Ready** | One codebase, multiple clients, zero code changes — `APP_NAME`, `PRIMARY_COLOR`, `TENANT_ID` override everything. |
| **Multi-Strategy Framework** | Run NFLX momentum vs. AAPL mean-reversion vs. custom universe simultaneously — each with its own model version. |
| **Research Copilot** | AI-generated, citation-backed research notes replace the analyst workflow for small trading desks. |

---

## 🛠️ Tech Stack

| Layer | Libraries |
|-------|-----------|
| Dashboard | Streamlit · Plotly |
| ML | scikit-learn · XGBoost · LightGBM · pandas · numpy · scipy |
| Uncertainty | Conformal prediction (MAPIE-style) |
| Regimes | hmmlearn (HMM bull/bear/sideways) |
| Data | yfinance · Alpha Vantage · Alpaca |
| Copilot | LangChain · LangGraph · ChromaDB · Groq · sentence-transformers |
| LLM | Groq LLaMA3-8b-8192 · OpenAI GPT-4o-mini (optional) |
| Observability | Langfuse · JSONL · Prometheus conventions |
| Evaluation | RAGAS · datasets |
| Compliance | Custom SEBI audit module |
| API | FastAPI · uvicorn |
| Testing | pytest |
| CI/CD | GitHub Actions |

---

## 📁 Project Structure

```
netflix-stock-project/
│
├── app/
│   └── app.py                        # Streamlit dashboard (15 tabs, all Plotly)
│
├── api/
│   └── main.py                       # FastAPI (/predict, /risk/*, /execute, /track-record)
│
├── src/
│   ├── data_loader.py                # Multi-source OHLCV loader (CSV/yfinance/AV/Alpaca)
│   ├── preprocessing.py              # Cleaning + outlier removal
│   ├── feature_engineering.py        # 51 technical indicators
│   ├── feature_utils.py              # Prediction-row builder
│   ├── modeling.py                   # ManualStackingRegressor (XGB+LGB+RF+ET → Ridge)
│   ├── uncertainty.py                # Conformal prediction intervals
│   ├── model_registry.py             # Versioned model saves + registry.json
│   ├── strategy_registry.py          # 🆕 Named strategy definitions + parallel compare
│   ├── backtest.py                   # Binary long/flat + Kelly backtest
│   ├── risk_manager.py               # ATR stop, Kelly, VaR/CVaR, circuit breaker
│   ├── drift.py                      # PSI + KS drift detection
│   ├── regime_detection.py           # HMM bull/bear/sideways regimes
│   ├── sentiment.py                  # VADER sentiment helpers
│   ├── paper_trade.py                # Day-by-day paper simulation
│   ├── track_record.py               # 🆕 Signal log + Sharpe/Sortino/Calmar/MDD
│   ├── track_record_seeder.py        # Seed sample track-record data
│   ├── compliance_sebi.py            # 🆕 SEBI algo-trading compliance checks
│   ├── branding_config.py            # 🆕 White-label name/color/logo from env
│   ├── explainability.py             # SHAP + feature importance
│   ├── agent_traces.py               # Langfuse tracing + JSONL fallback
│   ├── monitoring.py                 # Slack/email drift alerts
│   │
│   ├── narrator/                     # Original AI Narrative pipeline
│   │   ├── vector_store.py           # ChromaDB wrapper + hash-embedding fallback
│   │   ├── corpus.py                 # Earnings transcripts + news seeder
│   │   ├── retriever_agent.py        # Semantic retrieval agent
│   │   ├── synthesis_agent.py        # GPT / rule-based narrative generator
│   │   ├── graph.py                  # LangGraph: retrieve → synthesise
│   │   └── eval.py                   # RAGAS faithfulness + lexical fallback
│   │
│   └── copilot/                      # 🆕 Research Copilot 4-agent pipeline
│       ├── retriever.py              # CopilotRetriever (ChromaDB RAG)
│       ├── tool_agent.py             # CopilotToolAgent (model+SHAP+risk state)
│       ├── writer.py                 # CopilotWriter (Groq LLaMA3 / fallback)
│       ├── hitl_router.py            # HITLRouter (human gate)
│       └── graph.py                  # CopilotGraph (LangGraph orchestration)
│
├── main.py                           # Model training + evaluation pipeline
├── eval_narrative.py                 # Original RAGAS eval script
├── eval_ragas_copilot.py             # 🆕 RAGAS eval for Research Copilot
├── eval_regime_robustness.py         # 🆕 Regime-split backtest robustness report
│
├── models/
│   ├── model.pkl                     # Trained stacking regressor
│   └── regime_model.pkl              # Trained HMM regime detector
│
├── data/
│   ├── netflix.csv                   # Bundled OHLCV data
│   ├── transcripts/                  # Earnings call transcripts (auto-seeded)
│   ├── news/                         # Financial news articles (auto-seeded)
│   └── chroma_db/                    # ChromaDB vector store (auto-created)
│
├── outputs/                          # Plots, metrics, backtest curves, eval results
├── logs/
│   └── agent_traces.jsonl            # Per-run agent trace log
│
├── tests/                            # Pytest suite
├── .github/workflows/                # CI (test.yml) + retrain (retrain.yml)
├── branding.yaml                     # Default branding config
├── config.yaml                       # Pipeline hyperparameters
├── requirements.txt                  # Runtime deps (Streamlit Cloud)
└── requirements-dev.txt              # + test deps
```

---

## ▶ Run Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

---

## 🔄 Retrain

```bash
python main.py --source csv --ticker NFLX
```

Writes updated artefacts to `models/` and `outputs/`. The weekly GitHub Actions workflow (`retrain.yml`) runs this automatically.

---

## 📜 License

MIT — see [LICENSE](LICENSE).
