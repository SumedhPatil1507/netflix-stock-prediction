# Alpha Engine

[![Tests](https://github.com/SumedhPatil1507/netflix-stock-prediction/actions/workflows/test.yml/badge.svg)](https://github.com/SumedhPatil1507/netflix-stock-prediction/actions)
[![Retrain](https://github.com/SumedhPatil1507/netflix-stock-prediction/actions/workflows/retrain.yml/badge.svg)](https://github.com/SumedhPatil1507/netflix-stock-prediction/actions)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live-red)](https://netflix-stock-prediction-h4e4qxevbfjweltuumcxeb.streamlit.app)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> Production-grade ML system for stock return prediction with execution-ready risk management and broker integration.

**[Live App](https://netflix-stock-prediction-h4e4qxevbfjweltuumcxeb.streamlit.app)** | **[Results](RESULTS.md)** | **[Contributing](CONTRIBUTING.md)**

---

## What's Inside

| Layer | Implementation |
|---|---|
| **Data** | yfinance · Alpha Vantage (1min/5min) · Alpaca Markets (minute bars) · CSV fallback |
| **Features** | 51 technical indicators — lags, RSI, MACD, BB, ATR, Stochastic, CCI, OBV, regime |
| **Model** | XGB + LGBM + RF + ET → Ridge (manual stacking, OOF, walk-forward CV) |
| **Uncertainty** | Conformal prediction — 90% calibrated intervals |
| **Risk** | Configurable max position value cap (5% default) · ATR stop-loss · Kelly sizing · portfolio heat · drawdown circuit breaker |
| **Execution** | `/execute` endpoint — Alpaca paper/live + paper simulation |
| **Versioning** | Timestamped model saves + JSON registry with rollback |
| **Retraining** | GitHub Actions cron (weekly) + manual trigger |
| **Monitoring** | PSI + KS drift detection · Slack/email alerts |
| **AI Narrative** | Two-agent LangGraph RAG over recent ticker news and earnings-call transcripts · Chroma citations · Langfuse traces · RAGAS faithfulness evaluation |
| **API** | FastAPI v2.0 — rate limited, `/predict` `/risk/position` `/execute` `/registry` |
| **Dashboard** | 10-tab Streamlit — all Plotly interactive, multi-ticker |
| **Tests** | 50+ pytest unit tests across all modules |

---

## Architecture

```
Data Sources
  ├── yfinance (daily/intraday)
  ├── Alpha Vantage (1min–daily, free tier)
  └── Alpaca Markets (minute bars, paper account)
        │
  data_loader.py → preprocessing → feature_engineering (51 features)
        │
  regime_detection.py (HMM: Bull/Bear/Sideways)
        │
  ManualStackingRegressor (XGB + LGBM + RF + ET → Ridge)
        │
  uncertainty.py (conformal intervals, 90% coverage)
        │
  risk_manager.py (max position value cap, ATR stops, Kelly, heat, drawdown circuit breaker)
        │
  model_registry.py (versioned models + latest per-ticker predictions)
        │
  ├── FastAPI: /predict /risk/position /risk/matrix /execute /registry
  ├── Streamlit: 10 interactive tabs
  │     └── AI Narrative: LangGraph retrieval → grounded synthesis + citations
  └── GitHub Actions: CI + weekly retraining
```

---

## Quick Start

```bash
cp .env.example .env          # add API keys
pip install -r requirements-dev.txt

make train                    # train on CSV
make train-live               # train on live yfinance
make train-alphavantage       # train on Alpha Vantage data
make train-alpaca             # train on Alpaca minute bars

make app                      # Streamlit dashboard
make api                      # FastAPI at localhost:8000/docs
make test                     # run all tests
make paper-trade              # 90-day paper trade simulation
make registry                 # view model version history
python scripts/evaluate_narrative_faithfulness.py # RAGAS faithfulness score (after a narrative run)
```

---

## Deploy to Streamlit Community Cloud

The existing [live dashboard](https://netflix-stock-prediction-h4e4qxevbfjweltuumcxeb.streamlit.app) is served from this repository. The Community Cloud app should use:

| Setting | Value |
|---|---|
| Repository | `SumedhPatil1507/netflix-stock-prediction` |
| Branch | `main` |
| App entrypoint | `app/app.py` |
| Python | Select `3.11` in Community Cloud advanced settings (`.python-version` records the local target) |
| Dependencies | Root `requirements.txt` |

After the app is connected to this repository and branch, Community Cloud rebuilds it when new commits are pushed. To connect or check these settings, open the app in [Streamlit Community Cloud](https://share.streamlit.io/) and choose **Manage app**. See Streamlit's [deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy) and [dependency guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies).

For AI narratives, add an `OPENAI_API_KEY` in the app's **Settings → Secrets** in Community Cloud. Without it, the main dashboard remains available, but narrative generation reports that the key is required. The following optional settings enable a compatible API endpoint, model selection, and Langfuse tracing:

```toml
OPENAI_API_KEY = "your-key"
# OPENAI_API_BASE = "https://api.openai.com/v1"
MARKET_NARRATOR_MODEL = "gpt-4o-mini"
MARKET_NARRATOR_EMBEDDING_MODEL = "text-embedding-3-small"
# LANGFUSE_PUBLIC_KEY = "your-public-key"
# LANGFUSE_SECRET_KEY = "your-secret-key"
# LANGFUSE_BASE_URL = "https://cloud.langfuse.com"
```

Set secrets through the Streamlit dashboard, **not** in GitHub. See [Streamlit secrets management](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management). The app fetches recent news from Yahoo Finance; upload `.txt` or `.md` earnings-call transcripts in the AI Narrative tab. Keep original transcript files separately because local app files and the local Chroma index are deployment-instance data, not a durable shared database.

---

## Data Sources

| Source | Interval | Key Required | Notes |
|---|---|---|---|
| yfinance | daily + intraday | No | Free, rate limited |
| Alpha Vantage | 1min/5min/daily | `ALPHA_VANTAGE_KEY` | 25 req/day free |
| Alpaca Markets | 1min–1day | `ALPACA_API_KEY` + `ALPACA_SECRET_KEY` | Free paper account |
| CSV | daily | No | Offline fallback |

---

## API Endpoints

```bash
uvicorn api.main:app --reload
# Swagger: http://localhost:8000/docs
```

| Endpoint | Method | Description |
|---|---|---|
| `/predict` | POST | ML prediction + conformal interval |
| `/risk/position` | POST | Position size bounded by the configured max position value |
| `/risk/matrix` | POST | Full risk matrix |
| `/execute` | POST | Submit order to Alpaca or paper |
| `/model_info` | GET | Architecture + metrics + version |
| `/registry` | GET | All model versions |
| `/health` | GET | Status check |

---

## Risk Management

The `/risk/position` endpoint and Risk tab compute:
- **Hard position-value cap** — gross position value cannot exceed `portfolio_value × max_position_pct` (5% default); returns `HOLD` if even one share exceeds the cap
- **ATR-based stop-loss** — adapts to current volatility
- **Fixed % stop** — hard floor (default 2%)
- **Take-profit** — 2× stop distance (configurable R:R)
- **Kelly fraction** — position size proportional to edge
- **Portfolio heat** — max total open risk (default 20%)
- **Drawdown circuit breaker** — halts trading at 10% drawdown

---

## Environment Variables

```bash
# Data sources
ALPHA_VANTAGE_KEY=...
ALPACA_API_KEY=...
ALPACA_SECRET_KEY=...
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# Alerts
SLACK_WEBHOOK_URL=...
ALERT_EMAIL=...

# API
API_RATE_LIMIT=10
DEFAULT_TICKER=NFLX

# AI Market Narrator (required for narrative generation; optional Langfuse observability)
OPENAI_API_KEY=...
MARKET_NARRATOR_MODEL=gpt-4o-mini
MARKET_NARRATOR_EMBEDDING_MODEL=text-embedding-3-small
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_BASE_URL=https://cloud.langfuse.com
```

---

## Project Structure

```
src/
  data_loader.py       # yfinance + Alpha Vantage + Alpaca + CSV
  feature_utils.py     # shared feature computation
  modeling.py          # ManualStackingRegressor + conformal
  risk_manager.py      # max position value cap, ATR stop, Kelly, circuit breaker
  model_registry.py    # versioned models + latest per-ticker predictions
  market_narrator.py   # LangGraph retrieval + grounded narrative synthesis
  monitoring.py        # Slack/email alerts
  regime_detection.py  # HMM Bull/Bear/Sideways
  backtest.py          # Kelly + Sharpe/Sortino/Calmar
  drift.py             # PSI + KS drift detection
  paper_trade.py       # day-by-day live simulation
  sentiment.py         # VADER news scoring
  tuning.py            # Optuna hyperparameter search
agent_traces.py        # local JSONL and optional Langfuse agent spans
scripts/evaluate_narrative_faithfulness.py # RAGAS faithfulness evaluation
api/main.py            # FastAPI v2.0
app/app.py             # Streamlit 10-tab dashboard
tests/                 # 50+ unit tests
.github/workflows/     # CI + weekly retraining
```

---

## Tech Stack

Python · XGBoost · LightGBM · Scikit-learn · hmmlearn · Plotly · FastAPI · Streamlit · Optuna · Pytest · GitHub Actions · python-dotenv · yfinance · Alpha Vantage · Alpaca Markets
