# Netflix Stock Prediction

End-to-end ML pipeline predicting next-day Netflix stock returns using a stacking ensemble, HMM regime detection, conformal prediction intervals, and a full backtesting engine.

**Live App:** [Launch on Streamlit](https://netflix-stock-prediction-h4e4qxevbfjweltuumcxeb.streamlit.app)

---

## What makes this research-grade

- **HMM Regime Detection** — Gaussian Hidden Markov Model identifies Bull/Bear/Sideways market states. Regime posterior probabilities fed as features to the model
- **Conformal Prediction** — calibrated 90% prediction intervals with coverage guarantees (not just point estimates)
- **Manual Stacking Ensemble** — XGBoost + LightGBM + RandomForest + ExtraTrees → Ridge meta-learner via OOF
- **Advanced Backtesting** — Kelly-sized positions, Sharpe/Sortino/Calmar ratios, rolling Sharpe, transaction costs
- **YAML Config** — all hyperparameters in `config.yaml`, no hardcoded values
- **FastAPI endpoint** — `/predict` REST API for production serving
- **GitHub Actions CI** — pytest runs on every push
- **SHAP explainability** — feature-level contribution for every prediction

---

## Project Structure

```
netflix-stock-project/
├── src/
│   ├── data_loader.py          # CSV + live yfinance loader
│   ├── preprocessing.py
│   ├── feature_engineering.py  # 51 technical features
│   ├── regime_detection.py     # HMM Bull/Bear/Sideways
│   ├── modeling.py             # ManualStackingRegressor + conformal
│   ├── uncertainty.py          # Conformal prediction intervals
│   ├── backtest.py             # Kelly sizing, Sharpe, Sortino, Calmar
│   ├── explainability.py       # SHAP via XGB sub-model
│   ├── pipeline_config.py      # YAML-driven config dataclasses
│   ├── visualization.py        # 14 plot types
│   ├── forecasting.py          # ARIMA with CI
│   └── utils.py                # Logging, experiment log CSV
├── api/
│   └── main.py                 # FastAPI /predict /health /features
├── app/
│   └── app.py                  # Streamlit (Predict, EDA, Charts, Backtest, About)
├── tests/
│   ├── test_features.py        # 12 unit tests
│   └── test_regime_uncertainty.py  # 5 unit tests
├── notebooks/
│   └── analysis.ipynb          # Full storytelling notebook
├── .github/workflows/test.yml  # CI: pytest on push
├── .pre-commit-config.yaml     # black + isort + flake8
├── config.yaml                 # Hyperparameter overrides
├── main.py                     # Pipeline runner
└── requirements.txt
```

---

## Pipeline

### Features (51 total)
| Category | Features |
|---|---|
| Lag prices | Lag1–Lag20 |
| Lag returns | RetLag1–RetLag5 |
| Rolling stats | RollingMean/Std 5/10 |
| Returns | Return, LogReturn, Volatility, VolRatio |
| Volume | Volume, Volume_Ratio, OBV_Ratio |
| Momentum | RSI7/14, MACD, Stoch %K/%D, Williams %R, CCI |
| Bands | BB_Width, BB_Pct, ATR_Norm |
| Price vs MAs | vs MA5/10/21/50/200 |
| Momentum ratios | Momentum5/10/20 |
| Calendar | DayOfWeek, Month, Quarter, EarningsMonth |
| Regime | Regime label, Bear/Side/Bull probabilities |

### Model
- Level-0: XGBoost + LightGBM + RandomForest + ExtraTrees
- Level-1: Ridge meta-learner (trained on OOF predictions)
- Scaler: RobustScaler
- Validation: 5-fold walk-forward CV
- Target: next-day return (%) — stationary

### Conformal Prediction
- Split conformal (inductive) with 90% coverage guarantee
- Calibrated on held-out 10% of training set
- Reports interval width and empirical coverage on test set

### Backtesting
- Binary long/flat strategy + Kelly-sized variant
- Transaction cost: 0.1% per trade
- Metrics: Total Return, Ann Return, Sharpe, Sortino, Calmar, Max Drawdown, Win Rate
- Rolling 63-day Sharpe to detect strategy decay

---

## Running Locally

```bash
pip install -r requirements.txt

# Train model + generate all outputs
python main.py

# With live data from Yahoo Finance
python main.py --source live

# Generate default config.yaml to customise
python main.py --save-config

# Run tests
pytest tests/ -v

# Start FastAPI server
uvicorn api.main:app --reload
# Swagger UI at http://localhost:8000/docs

# Launch Streamlit app
streamlit run app/app.py
```

---

## Tech Stack
Python · Pandas · NumPy · Scikit-learn · XGBoost · LightGBM · hmmlearn · SHAP · Statsmodels · FastAPI · Streamlit · Pytest · GitHub Actions
