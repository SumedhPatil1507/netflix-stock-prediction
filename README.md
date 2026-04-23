# NFLX Alpha Engine

[![Tests](https://github.com/SumedhPatil1507/netflix-stock-prediction/actions/workflows/test.yml/badge.svg)](https://github.com/SumedhPatil1507/netflix-stock-prediction/actions)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live-red)](https://netflix-stock-prediction-h4e4qxevbfjweltuumcxeb.streamlit.app)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> End-to-end ML pipeline for Netflix stock return prediction. Stacking ensemble · HMM regime detection · Conformal prediction intervals · Kelly backtesting · Live data · Interactive Plotly dashboard.

**[Launch Live App](https://netflix-stock-prediction-h4e4qxevbfjweltuumcxeb.streamlit.app)** | **[Results](RESULTS.md)**

---

## The Edge

Most stock prediction projects predict **price** — trivially correlated with itself, giving misleading R² > 0.99.

This project predicts **next-day return (%)** — stationary, genuinely hard, no free lunch. The meaningful metric is **directional accuracy** (>52% = real signal) and **Sharpe ratio** of the resulting strategy.

See [RESULTS.md](RESULTS.md) for honest performance numbers and limitations.

---

## Architecture

```
Yahoo Finance (live) ──► data_loader ──► preprocessing
                                              │
                                    feature_engineering (51 features)
                                              │
                                    regime_detection (HMM 3-state)
                                              │
                                    ManualStackingRegressor
                                    XGB + LGBM + RF + ET → Ridge
                                              │
                                    uncertainty (conformal intervals)
                                              │
                              ┌───────────────┼───────────────┐
                           backtest        FastAPI         Streamlit
                        (Kelly/Sharpe)   (/predict)     (live dashboard)
```

---

## Features (51 total)

| Category | Features |
|---|---|
| Lag prices | Lag1–Lag20 |
| Lag returns | RetLag1–RetLag5 |
| Rolling stats | RollingMean/Std 5/10, Volatility 5/20, VolRatio |
| Volume | Volume, Volume_Ratio, OBV_Ratio |
| Momentum | RSI7/14, MACD+Signal+Hist+Norm, EMA_Cross |
| Bands | BB_Width, BB_Pct, ATR_Norm |
| Oscillators | Stoch %K/%D, Williams %R, CCI |
| Price vs MAs | vs MA5/10/21/50/200 |
| Momentum ratios | Momentum5/10/20 |
| Calendar | DayOfWeek, Month, Quarter, EarningsMonth |
| Regime | Bear/Sideways/Bull probabilities (HMM) |

---

## Quick Start

```bash
# Install
pip install -r requirements.txt        # Streamlit Cloud / production
pip install -r requirements-dev.txt    # Full dev environment

# Train
make train                             # CSV data
make train-live                        # Live yfinance data

# Run app
make app                               # Streamlit dashboard

# Run API
make api                               # FastAPI at localhost:8000/docs

# Test
make test                              # pytest

# Tune hyperparameters
make tune                              # Optuna (saves to outputs/optuna_best_params.json)
```

---

## Project Structure

```
├── src/
│   ├── feature_utils.py        # Shared feature computation (single source of truth)
│   ├── feature_engineering.py  # Full pipeline feature engineering
│   ├── modeling.py             # ManualStackingRegressor + conformal
│   ├── regime_detection.py     # Gaussian HMM (Bull/Bear/Sideways)
│   ├── uncertainty.py          # Conformal prediction intervals
│   ├── backtest.py             # Kelly sizing, Sharpe/Sortino/Calmar
│   ├── drift.py                # PSI + KS drift detection
│   ├── sentiment.py            # VADER news sentiment
│   ├── tuning.py               # Optuna hyperparameter search
│   ├── pipeline_config.py      # YAML-driven config dataclasses
│   └── data_loader.py          # CSV + live yfinance loader
├── app/app.py                  # Streamlit dashboard (7 tabs, all Plotly)
├── api/main.py                 # FastAPI (/predict /health /features)
├── tests/                      # 17 pytest unit tests
├── notebooks/analysis.ipynb   # Storytelling notebook
├── config.yaml                 # All hyperparameters
├── Makefile                    # One-command workflow
├── RESULTS.md                  # Honest performance numbers
└── .github/workflows/test.yml  # CI on every push
```

---

## API Usage

```bash
uvicorn api.main:app --reload
```

```python
import requests
response = requests.post("http://localhost:8000/predict", json={
    "rows": [
        {"open": 640, "high": 650, "low": 635, "close": 645, "volume": 5000000}
        # ... 10 rows minimum
    ]
})
print(response.json())
# {"predicted_return_pct": 0.12, "predicted_next_close": 645.77,
#  "last_close": 645.0, "signal": "BUY"}
```

---

## Tech Stack

Python · Pandas · NumPy · Scikit-learn · XGBoost · LightGBM · hmmlearn · SHAP · Statsmodels · Plotly · FastAPI · Streamlit · Pytest · Optuna · GitHub Actions

---

## What This Project Demonstrates

| Component | What It Does |
|---|---|
| Manual stacking ensemble | XGB + LGBM + RF + ET → Ridge meta-learner via OOF predictions |
| Walk-forward CV | Time-series aware validation — no future data leakage |
| Conformal prediction | Calibrated 90% prediction intervals with mathematical coverage guarantee |
| HMM regime detection | Identifies Bull/Bear/Sideways market states as model features |
| Kelly criterion backtesting | Position sizing proportional to edge, not binary long/flat |
| FastAPI serving | Production REST endpoint with Swagger docs |
| Drift detection | PSI + KS test monitoring for distribution shift |
| Paper trading simulation | Day-by-day live deployment test with prediction vs actual log |
| SHAP explainability | Feature-level contribution for every prediction |
| Optuna tuning | Automated hyperparameter search beyond defaults |
| GitHub Actions CI | Automated pytest on every push |
| YAML config system | All hyperparameters in one place, no hardcoded values |

---

## Dashboard Tabs

- **Market Overview** — Live candlestick chart, RSI, MACD, Bollinger Bands (Plotly interactive)
- **Predict** — Auto-fills with live NFLX data, returns predicted return + 90% conformal interval
- **Backtesting** — Equity curve, rolling Sharpe, drawdown for binary and Kelly strategies
- **Paper Trade** — Simulates model running day-by-day on live data, logs each prediction vs actual
- **Sentiment** — VADER-scored Netflix news headlines with daily sentiment chart
- **Risk** — VaR/CVaR at configurable confidence, volatility surface, correlation matrix
- **Drift Monitor** — PSI + KS test on all 51 features comparing training vs recent data
- **Architecture** — System design, design decisions, and honest limitations
