# ⚡ Alpha Engine Pro

[![Tests](https://github.com/SumedhPatil1507/netflix-stock-prediction/actions/workflows/test.yml/badge.svg)](https://github.com/SumedhPatil1507/netflix-stock-prediction/actions)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-live%20app-FF4B4B?logo=streamlit&logoColor=white)](https://netflix-stock-prediction-h4e4qxevbfjweltuumcxeb.streamlit.app)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![SEBI Compliant](https://img.shields.io/badge/SEBI-Algo%20Compliant-brightgreen)](src/compliance_sebi.py)
[![FastAPI](https://img.shields.io/badge/FastAPI-v2.0-009688?logo=fastapi)](api/main.py)

> **Institutional Quant Research & Execution Platform**
>
> Multi-strategy signal generation · Conformal prediction intervals · 4-agent Research Copilot · SEBI compliance · Verifiable track record · White-label ready

**[▶ Live App](https://netflix-stock-prediction-h4e4qxevbfjweltuumcxeb.streamlit.app)** · **[Results](RESULTS.md)** · **[Contributing](CONTRIBUTING.md)**

> ⚠️ Educational / research use only. Not financial advice.

---

## 🆕 What's New — v2.0

| # | Feature | Description |
|---|---------|-------------|
| 1 | 🧩 **Multi-Strategy Framework** | `strategy_registry.py` — define named strategies with their own ticker universe, feature set, model config, and risk params. Run and compare side-by-side. |
| 2 | 🤖 **Research Copilot** | 4-agent LangGraph pipeline: `RetrieverAgent → ToolAgent → WriterAgent → HITLRouter`. Groq LLaMA3-8b-8192 + rule-based fallback. |
| 3 | ⚖️ **SEBI Compliance Module** | 8-check audit report (audit trail, kill-switch, OTR, HITL gate, model traceability) with evidence + remediation JSON. |
| 4 | 📊 **Verifiable Track Record** | Append-only signal log with Sharpe, Sortino, Calmar, max-drawdown, profit factor. Exposed at `/track-record`. |
| 5 | 🎨 **White-Label Config** | `branding_config.py` — name, color, logo, tenant namespace from env/YAML. Zero code changes per client. |
| 6 | 🗂️ **5 New Streamlit Tabs** | Strategy Lab · Research Copilot · Track Record · Compliance · Observability |
| 7 | 🧪 **Eval Scripts** | RAGAS citation faithfulness + HMM regime robustness report — straight into the sales deck. |

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Alpha Engine Pro v2.0                             │
│                                                                             │
│  ┌──────────────┐   ┌──────────────────┐   ┌────────────────────────────┐  │
│  │  DATA LAYER  │   │  FEATURE ENGINE  │   │      ML MODELING           │  │
│  │  yfinance    │ → │  51 indicators   │ → │  XGB + LGBM + RF + ET      │  │
│  │  Alpha Vant. │   │  RSI MACD BB ATR │   │  → Ridge (OOF stacking)    │  │
│  │  Alpaca      │   │  Stoch CCI OBV   │   │  Conformal CI (90%)        │  │
│  │  CSV         │   │  HMM regimes     │   │                            │  │
│  └──────────────┘   └──────────────────┘   └────────────────────────────┘  │
│                                                         │                   │
│  ┌─────────────────────────────────────────────────────▼─────────────────┐  │
│  │                    STRATEGY REGISTRY                                  │  │
│  │  nflx_momentum · faang_diversified · tech_mean_reversion · custom    │  │
│  │  Per-strategy model version · parallel run · compare dashboard       │  │
│  └───────────────────────────────────┬────────────────────────────────┘  │
│                                      │                                   │
│  ┌───────────────────────────────────▼────────────────────────────────┐  │
│  │                   RESEARCH COPILOT (4-agent)                       │  │
│  │                                                                    │  │
│  │  RetrieverAgent ──→ ToolAgent ──→ WriterAgent ──→ HITLRouter       │  │
│  │  (ChromaDB RAG)    (Model+SHAP   (Groq LLaMA3   (human gate        │  │
│  │                     +Risk state)  /rule-based)   > $threshold)     │  │
│  └───────────────────────────────────┬────────────────────────────────┘  │
│                                      │                                   │
│  ┌───────────┐  ┌───────────────┐  ┌─▼──────────────┐  ┌─────────────┐  │
│  │  SEBI     │  │  TRACK RECORD │  │  RISK MANAGER  │  │ OBSERV-     │  │
│  │  AUDIT    │  │  Sharpe       │  │  ATR stop-loss │  │ ABILITY     │  │
│  │  8 checks │  │  Sortino      │  │  Kelly sizing  │  │ JSONL       │  │
│  │  kill-sw. │  │  Calmar       │  │  VaR/CVaR      │  │ Prometheus  │  │
│  │  OTR log  │  │  Max DD       │  │  Circuit brk.  │  │ Langfuse    │  │
│  └───────────┘  └───────────────┘  └────────────────┘  └─────────────┘  │
│                                                                           │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │              STREAMLIT DASHBOARD (15 interactive Plotly tabs)        │ │
│  │  Market · Predict · Backtest · PaperTrade · Sentiment · Risk ·       │ │
│  │  Drift · Explainability · AI Narrative · Architecture ·              │ │
│  │  Strategy Lab · Research Copilot · Track Record · Compliance · Obs   │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│                                                                           │
│  ┌──────────────────────────┐    ┌──────────────────────────────────────┐ │
│  │  FastAPI v2.0            │    │  GitHub Actions                      │ │
│  │  /predict /risk/*        │    │  test.yml  — CI on every push        │ │
│  │  /execute /track-record  │    │  retrain.yml — weekly retraining     │ │
│  │  /strategies /copilot/*  │    │                                      │ │
│  │  /hitl/* /compliance/*   │    │                                      │ │
│  └──────────────────────────┘    └──────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## 🗂 15-Tab Dashboard

| # | Tab | What you get |
|---|-----|-------------|
| 1 | 🕯 **Market Overview** | Live candlestick · MA 20/50/200 · RSI · MACD · Bollinger Bands — all Plotly interactive |
| 2 | 🔮 **Predict** | Live data editor → next-day return + 90% conformal interval → BUY/HOLD signal |
| 3 | 📈 **Backtesting** | Equity curve · rolling 63-day Sharpe · drawdown series |
| 4 | 📋 **Paper Trade** | Day-by-day simulation · cumulative PnL · predicted vs actual scatter |
| 5 | 🧠 **Sentiment** | VADER scoring on recent headlines · bar + pie charts · fallback sample data |
| 6 | ⚠️ **Risk** | Kelly sizing · ATR stop · take-profit · VaR/CVaR · volatility surface · correlation matrix |
| 7 | 🔬 **Drift Monitor** | PSI + KS test across all 51 features · feature-level drift bar chart |
| 8 | 🔍 **Explainability** | Per-model feature importance · cross-model comparison · target correlation |
| 9 | 🤖 **AI Narrative** | RAG research note · sentiment gauge · CI chart · source relevance bar · RAGAS eval |
| 10 | 🏗 **Architecture** | Full pipeline diagram · design decisions · limitations |
| 11 | 🧪 **Strategy Lab** | Define/run/compare named strategies · interactive grouped bar comparison |
| 12 | 🔬 **Research Copilot** | Chat-style research note · prediction gauge · SHAP bar · HITL approve/reject · source citations |
| 13 | 📊 **Track Record** | Equity curve · drawdown · monthly PnL heatmap · Sharpe/Sortino/Calmar KPIs |
| 14 | ⚖️ **Compliance** | 8 SEBI checks with ✅/❌/⚠️ expanders · OTR gauge · JSON report download |
| 15 | 📡 **Observability** | Agent trace JSONL · latency box plots · Prometheus-style metrics |

> All charts are **fully interactive Plotly** — zoom, pan, hover tooltips, PNG export.

---

## 🚀 Quick Start

```bash
git clone https://github.com/SumedhPatil1507/netflix-stock-prediction.git
cd netflix-stock-prediction

python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows PowerShell
# source .venv/bin/activate       # macOS / Linux

pip install -r requirements.txt
streamlit run app/app.py
```

Open <http://localhost:8501>. The app loads `models/model.pkl` and the bundled feature cache — **no retraining needed**.

### Optional AI + Copilot dependencies

```bash
pip install langchain langchain-openai langgraph langfuse \
            chromadb sentence-transformers ragas openai groq
```

Every tab degrades gracefully with a helpful install prompt when any optional dep is absent.

---

## 🔑 Environment Variables

Copy `.env.example` → `.env`. Never commit `.env`.

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `OPENAI_API_KEY` | Optional | — | GPT-4o-mini for AI Narrative; rule-based fallback if absent |
| `GROQ_API_KEY` | Optional | — | Groq LLaMA3-8b-8192 for Research Copilot WriterAgent |
| `LANGFUSE_PUBLIC_KEY` | Optional | — | Langfuse cloud traces; JSONL fallback always active |
| `LANGFUSE_SECRET_KEY` | Optional | — | Langfuse secret |
| `HITL_THRESHOLD_USD` | Optional | 10000 | USD above which HITL sign-off blocks `/execute` |
| `APP_NAME` | Optional | Alpha Engine Pro | White-label app title |
| `TENANT_ID` | Optional | default | Per-client strategy namespace |
| `PRIMARY_COLOR` | Optional | #e50914 | White-label theme hex color |
| `ALLOWED_TICKERS` | Optional | all | Comma-separated ticker allowlist per tenant |
| `ALPHA_VANTAGE_KEY` | Optional | — | Alpha Vantage intraday data |
| `ALPACA_API_KEY` | Optional | — | Alpaca paper/live execution |
| `ALPACA_SECRET_KEY` | Optional | — | Alpaca secret key |

---

## 🤖 Research Copilot Pipeline

```
User clicks "Generate Research Note"
        │
        ▼
RetrieverAgent — ChromaDB semantic search
  · Earnings-call transcripts (10-K/10-Q chunks)
  · Recent financial news headlines
  Returns: top-5 {text, metadata, distance}
        │
        ▼
ToolAgent — pulls live state from existing modules
  · Model prediction + 90% conformal interval
  · SHAP feature drivers (top-5 with direction)
  · Risk manager: ATR stop, Kelly %, portfolio heat
        │
        ▼
WriterAgent — Groq LLaMA3-8b-8192 (or rule-based fallback)
  Synthesizes a cited research note:
  "Model is bullish (+0.42%), CI [−0.3%, +1.1%].
   Primary driver: RSI divergence (+0.18 importance).
   Q3 earnings call: 'subscriber growth exceeded
   expectations by 2.1M' [Source: NFLX Q3 2024, Oct 2024]"
        │
        ▼
HITLRouter — human gate
  IF position_value > HITL_THRESHOLD_USD:
    → pending record written to outputs/hitl_pending.json
    → /execute blocked until human approval
  ELSE:
    → signal flows through automatically
        │
        ▼
  Research Note (cited · signed-off · audit-logged)
```

---

## ⚖️ SEBI Compliance — 8 Checks

| ID | Check | Status Logic |
|----|-------|-------------|
| SEBI_001 | Audit trail JSONL exists with order records | PASS if `logs/audit_trail.jsonl` has entries |
| SEBI_002 | Kill-switch JSON present with required fields | PASS if `outputs/kill_switch.json` has active/reason/activated_at |
| SEBI_003 | Order-to-trade ratio ≤ 50 | PASS if OTR ≤ 50; FAIL if > 50; WARN if no data |
| SEBI_004 | Pre-trade risk check (RiskManager) importable | PASS if `src.risk_manager.RiskManager` imports OK |
| SEBI_005 | Max position per instrument ≤ 10% | PASS if `RiskConfig.max_position_pct` ≤ 0.10 |
| SEBI_006 | HITL gate deployed and pending file accessible | PASS if `HITLRouter` imports + `hitl_pending.json` exists |
| SEBI_007 | Model version registry has ≥ 1 entry | PASS if `models/registry.json` is non-empty |
| SEBI_008 | Agent latency log exists | PASS if `logs/agent_traces.jsonl` exists |

```python
from src.compliance_sebi import SEBIComplianceChecker
report = SEBIComplianceChecker().generate_report()
# → {report_id, summary: {total, pass, fail, warn, compliance_pct},
#    checks: [...], kill_switch_status, order_to_trade_ratio}
```

---

## 📊 Track Record Metrics

| Metric | Description |
|--------|-------------|
| `sharpe` | Annualised Sharpe ratio √252 · μ/σ |
| `sortino` | Sortino — uses downside std only |
| `calmar` | Annualised return ÷ max drawdown |
| `max_drawdown` | Peak-to-trough equity drawdown (fraction) |
| `profit_factor` | Gross wins ÷ gross losses |
| `trailing_sharpe_63d` | Rolling 63-day (≈ 1 quarter) Sharpe |
| `win_rate_pct` | % of BUY signals with positive pnl_pct |
| `n_signals` | Total logged signals (paper + live) |
| `total_pnl_pct` | Cumulative P&L % across all signals |

---

## 🎨 White-Label Deployment

Re-skin the entire platform with environment variables — zero code changes:

```bash
export APP_NAME="Quant Edge Pro"
export PRIMARY_COLOR="#005EB8"
export TENANT_ID="client_abc"
export ALLOWED_TICKERS="RELIANCE,TCS,INFY"
export HITL_THRESHOLD_USD="25000"
streamlit run app/app.py
```

Or edit `branding.yaml` at the project root. Per-tenant strategy isolation ensures each `TENANT_ID` gets its own namespace in `strategy_registry.json` — strategies, models, and track records never bleed across clients.

---

## 🧪 Eval Scripts

```bash
# Research Copilot — RAGAS citation faithfulness
python eval_ragas_copilot.py
# → outputs/eval_copilot_ragas.json

# Regime robustness — directional accuracy by HMM regime
python eval_regime_robustness.py
# → outputs/eval_regime_robustness.json

# Original AI Narrative eval
python eval_narrative.py
# → outputs/evaluation_results.json
```

---

## 💼 Why Buyers Pay For This

| Feature | Why It Matters to a Buyer |
|---------|--------------------------|
| **Verifiable Track Record** | Buyers pay for demonstrated live Sharpe — not backtest claims. Signal log from day one. |
| **SEBI Compliance Module** | The single deciding factor between a hobby script and a deployable product in Indian markets. |
| **HITL Gate** | No institutional buyer accepts fully autonomous order placement. Human sign-off is non-negotiable. |
| **White-Label Ready** | One codebase · multiple clients · zero code changes — branding and tenant isolation built in. |
| **Multi-Strategy Framework** | Run NFLX momentum vs AAPL mean-reversion vs custom universe simultaneously, each with its own model version. |
| **Research Copilot** | AI-generated, citation-backed notes replace the analyst workflow for small trading desks. |
| **Conformal Intervals** | Calibrated uncertainty — not point predictions. Institutional buyers demand uncertainty quantification. |

---

## 🛠 Tech Stack

| Layer | Libraries |
|-------|-----------|
| Dashboard | Streamlit · Plotly · Pandas |
| ML | scikit-learn · XGBoost · LightGBM · numpy · scipy |
| Uncertainty | Conformal prediction (calibrated 90% CI) |
| Regimes | hmmlearn (HMM bull/bear/sideways) |
| Data | yfinance · Alpha Vantage · Alpaca Markets |
| Copilot | LangChain · LangGraph · ChromaDB · Groq · sentence-transformers |
| LLM | Groq LLaMA3-8b-8192 · OpenAI GPT-4o-mini (optional) |
| Observability | Langfuse · JSONL · Prometheus conventions |
| Evaluation | RAGAS · HuggingFace datasets |
| Compliance | Custom SEBI audit module |
| API | FastAPI · uvicorn · slowapi (rate limiting) |
| Testing | pytest (40+ tests) |
| CI/CD | GitHub Actions (test + weekly retrain) |

---

## 📁 Project Structure

```
netflix-stock-project/
├── app/app.py                        # Streamlit — 15 tabs, all Plotly interactive
├── api/main.py                       # FastAPI v2.0 — 14 endpoints
├── main.py                           # Model training + evaluation pipeline
├── branding.yaml                     # White-label defaults (override via env)
├── eval_narrative.py                 # AI Narrative RAGAS eval
├── eval_ragas_copilot.py             # Research Copilot RAGAS eval
├── eval_regime_robustness.py         # Regime-split backtest robustness
│
├── src/
│   ├── data_loader.py                # yfinance + Alpha Vantage + Alpaca + CSV
│   ├── preprocessing.py              # Cleaning + outlier removal
│   ├── feature_engineering.py        # 51 technical indicators
│   ├── feature_utils.py              # Prediction-row builder
│   ├── modeling.py                   # ManualStackingRegressor
│   ├── uncertainty.py                # Conformal prediction intervals
│   ├── model_registry.py             # Versioned model saves + registry.json
│   ├── strategy_registry.py          # ★ Named strategies + parallel compare
│   ├── backtest.py                   # Kelly + Sharpe/Sortino/Calmar backtest
│   ├── risk_manager.py               # ATR stop · Kelly · VaR/CVaR · circuit breaker
│   ├── drift.py                      # PSI + KS drift detection
│   ├── regime_detection.py           # HMM bull/bear/sideways
│   ├── sentiment.py                  # VADER news scoring
│   ├── paper_trade.py                # Day-by-day paper simulation
│   ├── track_record.py               # ★ Signal log + performance metrics
│   ├── track_record_seeder.py        # Seed sample track-record data
│   ├── compliance_sebi.py            # ★ SEBI 8-check audit module
│   ├── branding_config.py            # ★ White-label config from env/YAML
│   ├── explainability.py             # SHAP + feature importance
│   ├── agent_traces.py               # Langfuse tracing + JSONL fallback
│   ├── monitoring.py                 # Slack/email drift alerts
│   │
│   ├── narrator/                     # AI Narrative pipeline (v1)
│   │   ├── vector_store.py           # ChromaDB + hash-embedding fallback
│   │   ├── corpus.py                 # Earnings transcripts + news seeder
│   │   ├── retriever_agent.py        # Semantic retrieval
│   │   ├── synthesis_agent.py        # GPT / rule-based narrative
│   │   ├── graph.py                  # LangGraph: retrieve → synthesise
│   │   └── eval.py                   # RAGAS faithfulness + lexical fallback
│   │
│   └── copilot/                      # ★ Research Copilot 4-agent pipeline
│       ├── retriever.py              # CopilotRetriever (ChromaDB RAG)
│       ├── tool_agent.py             # CopilotToolAgent (model+SHAP+risk)
│       ├── writer.py                 # CopilotWriter (Groq / fallback)
│       ├── hitl_router.py            # HITLRouter (human gate)
│       └── graph.py                  # CopilotGraph (LangGraph)
│
├── models/
│   ├── model.pkl                     # Trained stacking regressor
│   └── regime_model.pkl              # Trained HMM regime detector
│
├── outputs/                          # Metrics, plots, backtest curves, eval results
├── data/netflix.csv                  # Bundled OHLCV data
├── logs/agent_traces.jsonl           # Per-run agent trace log
├── tests/                            # 40+ pytest unit tests
└── .github/workflows/                # CI (test.yml) + retrain (retrain.yml)
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
python main.py --source csv --ticker NFLX      # train on bundled CSV
python main.py --source yfinance --ticker AAPL  # train on live data
```

Weekly GitHub Actions workflow (`retrain.yml`) runs this automatically.

---

## 💻 VS Code — Update GitHub Commands

Open the VS Code terminal (`Ctrl+`` ` ``) and run:

```powershell
# 1. Navigate to project
cd c:\Users\Sumedh\projects\netflix-stock-project

# 2. Check what changed
git status
git diff --stat

# 3. Stage your changes
git add app/app.py src/ branding.yaml requirements.txt README.md

# 4. Commit
git commit -m "your change description"

# 5. Push to BOTH branches (keeps them in sync)
git push origin main
git push origin streamlit-streamlit-cloud-setup

# One-liner for quick updates:
git add -A; git commit -m "update"; git push origin main; git push origin streamlit-streamlit-cloud-setup
```

---

## 📜 License

MIT — see [LICENSE](LICENSE).
