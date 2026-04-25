# Alpha Engine

[![Tests](https://github.com/SumedhPatil1507/netflix-stock-prediction/actions/workflows/test.yml/badge.svg)](https://github.com/SumedhPatil1507/netflix-stock-prediction/actions)
[![Retrain](https://github.com/SumedhPatil1507/netflix-stock-prediction/actions/workflows/retrain.yml/badge.svg)](https://github.com/SumedhPatil1507/netflix-stock-prediction/actions)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live-red)](https://netflix-stock-prediction-h4e4qxevbfjweltuumcxeb.streamlit.app)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> Production-grade ML system for stock return prediction. Multi-ticker · Stacking ensemble · HMM regime detection · Conformal prediction intervals · Kelly backtesting · Scheduled retraining · Model versioning · Drift monitoring · FastAPI with rate limiting.

**[Launch Live App](https://netflix-stock-prediction-h4e4qxevbfjweltuumcxeb.streamlit.app)** | **[Results](RESULTS.md)** | **[Contributing](CONTRIBUTING.md)**

---

## Production Features

| Feature | Implementation |
|---|---|
| Multi-ticker support | Any Yahoo Finance symbol (NFLX, AAPL, TSLA...) |
| Model versioning | Timestamped saves + JSON registry with rollback |
| Scheduled retraining | GitHub Actions cron every Sunday 02:00 UTC |
| Rate limiting | slowapi — 10 req/min per IP on FastAPI |
| Drift monitoring | PSI + KS test with Slack/email alerting |
| Secrets management | python-dotenv + .env.example |
| Conformal intervals | 90% calibrated prediction intervals |
| Walk-forward CV | No future data leakage |
| Kelly backtesting | Sharpe/Sortino/Calmar + rolling Sharpe |
| Paper trading | Day-by-day live deployment simulation |

---

## Architecture

```
Yahoo Finance (any ticker)
        │
        ▼
  data_loader.py  ──── multi-ticker, validation, CSV fallback
        │
        ▼
  feature_engineering.py  ──── 51 technical features
        │
        ▼
  regime_detection.py  ──── HMM Bull/Bear/Sideways
        │
        ▼
  ManualStackingRegressor
  XGB + LGBM + RF + ET → Ridge
        │
        ▼
  uncertainty.py  ──── conformal prediction intervals
        │
        ▼
  model_registry.py  ──── versioned saves + registry.json
        │
        ▼
  monitoring.py  ──── Slack/email drift + retrain alerts
        │
        ├── FastAPI (/predict /health /model_info /registry)
        ├── Streamlit (9-tab live dashboard)
        └── GitHub Actions (CI + weekly retraining)
```

---

## Quick Start

```bash
# Setup
cp .env.example .env          # fill in optional secrets
pip install -r requirements-dev.txt

# Train (any ticker)
make train TICKER=NFLX        # CSV data
make train-live TICKER=AAPL   # live yfinance data

# Run
make app                       # Streamlit dashboard
make api                       # FastAPI at localhost:8000/docs

# Test
make test

# View model registry
make registry

# Paper trade simulation
make paper-trade
```

---

## Environment Variables (.env)

```bash
SLACK_WEBHOOK_URL=...    # drift/retrain alerts
SMTP_HOST=...            # email alerts
ALERT_EMAIL=...
API_RATE_LIMIT=10        # requests/min per IP
DEFAULT_TICKER=NFLX
```

---

## API

```bash
uvicorn api.main:app --reload
# Swagger UI: http://localhost:8000/docs
```

```python
import requests
resp = requests.post("http://localhost:8000/predict", json={
    "ticker": "AAPL",
    "rows": [{"open":170,"high":175,"low":168,"close":172,"volume":50000000}
             # ... 10 rows minimum
    ]
})
# Returns: predicted_return_pct, predicted_next_close, signal, confidence_interval
```

---

## Scheduled Retraining

GitHub Actions runs `main.py --source live` every Sunday at 02:00 UTC, commits the new model and cache, and sends a Slack notification. Trigger manually via Actions → Scheduled Retraining → Run workflow.

---

## Project Structure

```
├── src/
│   ├── feature_utils.py      # Shared feature computation (single source of truth)
│   ├── feature_engineering.py
│   ├── modeling.py           # ManualStackingRegressor + conformal
│   ├── model_registry.py     # Versioned model saves + registry
│   ├── monitoring.py         # Slack/email alerting
│   ├── regime_detection.py   # HMM Bull/Bear/Sideways
│   ├── uncertainty.py        # Conformal prediction intervals
│   ├── backtest.py           # Kelly sizing, Sharpe/Sortino/Calmar
│   ├── drift.py              # PSI + KS drift detection
│   ├── sentiment.py          # VADER news sentiment
│   ├── paper_trade.py        # Paper trading simulation
│   ├── tuning.py             # Optuna hyperparameter search
│   └── pipeline_config.py    # YAML config dataclasses
├── app/app.py                # Streamlit (9 tabs, Plotly, multi-ticker)
├── api/main.py               # FastAPI (rate limited, multi-ticker)
├── tests/                    # 30+ pytest unit tests
├── .github/workflows/
│   ├── test.yml              # CI on every push
│   └── retrain.yml           # Weekly scheduled retraining
├── config.yaml               # All hyperparameters
├── .env.example              # Secrets template
├── Makefile                  # One-command workflow
└── pyproject.toml            # Modern Python packaging
```

---

## Tech Stack

Python · Pandas · NumPy · Scikit-learn · XGBoost · LightGBM · hmmlearn · Statsmodels · Plotly · FastAPI · slowapi · Streamlit · Pytest · Optuna · GitHub Actions · python-dotenv
