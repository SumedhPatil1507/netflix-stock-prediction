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
| **Risk** | ATR stop-loss · Kelly sizing · portfolio heat · drawdown circuit breaker |
| **Execution** | `/execute` endpoint — Alpaca paper/live + paper simulation |
| **Versioning** | Timestamped model saves + JSON registry with rollback |
| **Retraining** | GitHub Actions cron (weekly) + manual trigger |
| **Monitoring** | PSI + KS drift detection · Slack/email alerts |
| **API** | FastAPI v2.0 — rate limited, `/predict` `/risk/position` `/execute` `/registry` |
| **Dashboard** | 9-tab Streamlit — all Plotly interactive, multi-ticker |
| **Tests** | 40+ pytest unit tests across all modules |

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
  risk_manager.py (ATR stop, Kelly, circuit breaker)
        │
  model_registry.py (versioned saves)
        │
  ├── FastAPI: /predict /risk/position /risk/matrix /execute /registry
  ├── Streamlit: 9 interactive tabs
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
```

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
| `/risk/position` | POST | Execution-ready position size |
| `/risk/matrix` | POST | Full risk matrix |
| `/execute` | POST | Submit order to Alpaca or paper |
| `/model_info` | GET | Architecture + metrics + version |
| `/registry` | GET | All model versions |
| `/health` | GET | Status check |

---

## Risk Management

The `/risk/position` endpoint and Risk tab compute:
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
```

---

## Project Structure

```
src/
  data_loader.py       # yfinance + Alpha Vantage + Alpaca + CSV
  feature_utils.py     # shared feature computation
  modeling.py          # ManualStackingRegressor + conformal
  risk_manager.py      # ATR stop, Kelly, circuit breaker
  model_registry.py    # versioned model saves
  monitoring.py        # Slack/email alerts
  regime_detection.py  # HMM Bull/Bear/Sideways
  backtest.py          # Kelly + Sharpe/Sortino/Calmar
  drift.py             # PSI + KS drift detection
  paper_trade.py       # day-by-day live simulation
  sentiment.py         # VADER news scoring
  tuning.py            # Optuna hyperparameter search
api/main.py            # FastAPI v2.0
app/app.py             # Streamlit 9-tab dashboard
tests/                 # 40+ unit tests
.github/workflows/     # CI + weekly retraining
```

---

## Tech Stack

Python · XGBoost · LightGBM · Scikit-learn · hmmlearn · Plotly · FastAPI · Streamlit · Optuna · Pytest · GitHub Actions · python-dotenv · yfinance · Alpha Vantage · Alpaca Markets
