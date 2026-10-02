# Alpha Engine — Netflix Stock Prediction

[![Tests](https://github.com/SumedhPatil1507/netflix-stock-prediction/actions/workflows/test.yml/badge.svg)](https://github.com/SumedhPatil1507/netflix-stock-prediction/actions)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-live-red)](https://netflix-stock-prediction-h4e4qxevbfjweltuumcxeb.streamlit.app)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end ML + agentic-AI stock analysis platform built on Streamlit. Predicts next-day NFLX returns, runs backtests, manages risk, explains model decisions, and — as of the latest release — generates **citation-backed market narratives** via a LangGraph RAG pipeline backed by ChromaDB.

> **Disclaimer:** Educational / research use only. Not investment advice.

**[▶ Live app](https://netflix-stock-prediction-h4e4qxevbfjweltuumcxeb.streamlit.app)** · **[Results](RESULTS.md)** · **[Contributing](CONTRIBUTING.md)**

---

## What's new — AI Market Narrator

The biggest addition is a **two-agent LangGraph pipeline** that produces plain-English, citation-backed explanations of every model prediction:

| Component | What it does |
|---|---|
| `RetrieverAgent` | Semantic search over ChromaDB (earnings transcripts + financial news) |
| `SynthesisAgent` | Reads prediction + conformal interval + retrieved docs → narrative |
| `NarratorGraph` | LangGraph workflow: retrieve → synthesise (or error-handle) |
| `_HashEmbeddingFunction` | Zero-dep fallback embedder — works without `sentence-transformers` |
| `AgentTracer` | Langfuse trace + local JSONL fallback in `logs/agent_traces.jsonl` |
| `NarrativeEvaluator` | RAGAS faithfulness when available; lexical-overlap fallback otherwise |
| `eval_narrative.py` | Standalone evaluation script at project root |

The AI Narrative tab renders five interactive Plotly charts: a sentiment gauge, a conformal-interval scatter, a source-relevance bar, a faithfulness gauge (on demand), and an agent trace log viewer.

---

## Ten-tab Streamlit dashboard

| Tab | Contents |
|---|---|
| 🕯 Market Overview | Live candlestick · MA20/50/200 · RSI · MACD · Bollinger Bands |
| 🔮 Predict | Live data editor → next-day return + conformal interval |
| 📈 Backtesting | Equity curve · Rolling Sharpe · Drawdown |
| 📋 Paper Trade | Day-by-day simulation · cumulative PnL · pred vs actual scatter |
| 🧠 Sentiment | VADER scoring · bar + pie charts · fallback sample data |
| ⚠️ Risk | Kelly sizing · ATR stop-loss · VaR/CVaR · volatility surface |
| 🔬 Drift Monitor | PSI + KS drift detection across all 51 features |
| 🔍 Explainability | Per-model feature importance · target correlation |
| 🤖 AI Narrative | RAG narrative · sentiment gauge · CI chart · source relevance · RAGAS eval |
| 🏗 Architecture | Pipeline diagram · design decisions · limitations |

All charts are **fully interactive Plotly** (zoom, pan, hover, download).

---

## Model

- **Target:** next-day return `%` (stationary; not price)
- **Architecture:** `ManualStackingRegressor` — XGBoost + LightGBM + Random Forest + Extra Trees → Ridge meta-learner
- **Validation:** walk-forward time-series CV (no future leakage)
- **Uncertainty:** conformal prediction intervals (90% coverage guarantee)
- **Features:** 51 technical indicators (RSI, MACD, Bollinger, ATR, Stochastic, Williams %R, CCI, …)

---

## AI Market Narrator — deep dive

### Pipeline

```
User clicks "Generate AI Narrative"
        │
        ▼
NarratorGraph.run(ticker, prediction, conformal_interval, price)
        │
        ├─ [retrieve node]
        │    RetrieverAgent → ChromaDB.query(semantic search, n=5)
        │    Returns: list of {text, metadata, distance}
        │
        ├─ [synthesise node]
        │    SynthesisAgent.generate_narrative(...)
        │    → GPT-4o-mini if OPENAI_API_KEY set
        │    → enriched fallback (cites retrieved doc titles) otherwise
        │
        └─ Result dict: narrative, citations, sentiment, sources_used
                │
                ├─ Streamlit renders 5 interactive Plotly charts
                ├─ AgentTracer logs to Langfuse + logs/agent_traces.jsonl
                └─ Optional: NarrativeEvaluator.evaluate_narrative()
                             (RAGAS faithfulness or lexical-overlap fallback)
```

### Embedding strategy

`VectorStore` tries `sentence-transformers/all-MiniLM-L6-v2` first. If the package is absent it falls back to `_HashEmbeddingFunction` — a pure-Python 128-dim word-hash embedder that requires no downloads and no `onnxruntime`. Retrieval quality is lower than a transformer but the pipeline is fully functional.

### Observability

Every agent run writes one JSONL line to `logs/agent_traces.jsonl`:

```json
{
  "timestamp": "2026-10-01T12:34:56",
  "agent_name": "narrator_graph",
  "ticker": "NFLX",
  "inputs_summary": "{'ticker': 'NFLX', 'prediction': 0.005, ...}",
  "outputs_summary": "{'success': True, 'sentiment': 'bullish', ...}",
  "duration_seconds": 1.23,
  "error": null
}
```

Langfuse is optional: set `LANGFUSE_PUBLIC_KEY` + `LANGFUSE_SECRET_KEY` to enable cloud tracing.

### RAGAS evaluation

```bash
python eval_narrative.py
```

Generates a narrative for NFLX, scores it with `NarrativeEvaluator`, prints a report, and saves to `outputs/evaluation_results.json`. Uses RAGAS faithfulness when the package is installed; falls back to lexical overlap otherwise.

---

## Quick start

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

### Optional integrations

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | LLM-powered narratives (GPT-4o-mini); falls back to rule-based if absent |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | Cloud trace dashboard; JSONL fallback always active |

Copy `.env.example` to `.env` and fill in values. Never commit `.env`.

### AI Narrator extra deps (already in `requirements.txt`)

```bash
pip install langchain langchain-openai langgraph langfuse \
            chromadb sentence-transformers ragas openai
```

The base dashboard works without these — the AI Narrative tab shows an install prompt when they are absent.

---

## Deploy to Streamlit Community Cloud

1. Push to the `streamlit-streamlit-cloud-setup` branch.
2. In [Streamlit Community Cloud](https://share.streamlit.io/) select this repo, branch `streamlit-streamlit-cloud-setup`, main file `app/app.py`.
3. Python 3.11 is auto-detected from `.python-version`.
4. Add optional secrets (`OPENAI_API_KEY`, `LANGFUSE_*`) in **Settings → Secrets**.

The model, data, and feature cache are tracked in the repo — no training step needed at deploy time.

---

## Project structure

```
app/app.py                     Streamlit dashboard (10 tabs, all Plotly)
api/main.py                    FastAPI service (/predict, /risk/*, /execute)
main.py                        Model training + evaluation pipeline
eval_narrative.py              Standalone RAGAS eval script

src/
  data_loader.py               Multi-source OHLCV loader (CSV / yfinance / AV / Alpaca)
  preprocessing.py             Cleaning, outlier removal
  feature_engineering.py       51 technical indicators
  modeling.py                  ManualStackingRegressor
  uncertainty.py               Conformal prediction intervals
  model_registry.py            Versioned model saves + registry.json
  backtest.py                  Binary long/flat + Kelly strategy
  risk_manager.py              ATR stop-loss, Kelly sizing, VaR/CVaR, circuit breaker
  drift.py                     PSI + KS drift detection
  regime_detection.py          HMM bull/bear/sideways regimes
  sentiment.py                 VADER sentiment helpers
  paper_trade.py               Day-by-day live simulation
  explainability.py            SHAP + feature importance
  agent_traces.py              Langfuse tracing + JSONL fallback
  monitoring.py                Slack/email drift alerts

  narrator/
    __init__.py
    vector_store.py            ChromaDB wrapper (_HashEmbeddingFunction fallback)
    corpus.py                  Earnings transcripts + news loader / seeder
    retriever_agent.py         Semantic retrieval agent
    synthesis_agent.py         GPT / rule-based narrative generator
    graph.py                   LangGraph workflow (retrieve → synthesise)
    eval.py                    RAGAS faithfulness + lexical-overlap fallback

models/model.pkl               Trained stacking regressor
data/netflix.csv               Bundled OHLCV data
data/chroma_db/                ChromaDB vector store (auto-created)
data/transcripts/              Earnings call transcripts (auto-seeded)
data/news/                     Financial news articles (auto-seeded)
outputs/                       Plots, metrics, backtest curves, eval results
logs/agent_traces.jsonl        Per-run agent trace log

requirements.txt               Runtime deps (Streamlit Cloud)
requirements-dev.txt           + test deps
tests/                         Pytest suite
.github/workflows/             CI (test.yml) + retrain (retrain.yml)
```

---

## Run tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

---

## Retrain

```bash
python main.py --source csv --ticker NFLX
```

Writes updated artefacts to `models/` and `outputs/`. The weekly GitHub Actions workflow (`retrain.yml`) runs this automatically.

---

## Technology stack

| Layer | Libraries |
|---|---|
| Dashboard | Streamlit · Plotly |
| ML | scikit-learn · XGBoost · LightGBM · pandas · numpy · scipy |
| Data | yfinance · Alpha Vantage · Alpaca |
| AI Narrator | LangChain · LangGraph · ChromaDB · OpenAI · sentence-transformers |
| Observability | Langfuse · JSONL |
| Evaluation | RAGAS · datasets |
| API | FastAPI · uvicorn |
| Testing | pytest |
| CI/CD | GitHub Actions |

---

## License

MIT — see [LICENSE](LICENSE).
