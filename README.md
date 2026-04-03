# 📈 Netflix Stock Prediction

## Overview
End-to-end ML pipeline that predicts the **next-day closing price** of Netflix stock using historical OHLCV data. Includes data preprocessing, advanced feature engineering, ensemble modeling, SHAP explainability, and a Streamlit web app.

**Live App:** [Launch on Streamlit](https://netflix-stock-prediction-h4e4qxevbfjweltuumcxeb.streamlit.app)

---

## Problem Statement
Predict the next-day closing price of Netflix stock from historical price and volume data, without data leakage.

---

## Dataset
- Source: Historical Netflix stock data (yFinance / CSV)
- Columns: Date, Open, High, Low, Close, Volume, Stock Splits
- Time Period: May 2002 – present (~5,900+ trading days)

---

## Pipeline

### 1. Data Loading & Preprocessing
- Tab-separated CSV ingestion
- Date parsing, deduplication, chronological sorting

### 2. Feature Engineering (29 features)
| Category | Features |
|---|---|
| Lag prices | Lag1, Lag2, Lag3, Lag5, Lag10 |
| Rolling stats | RollingMean_5/10, RollingStd_5/10 |
| Returns | Return (%), LogReturn, Volatility (20d) |
| Volume | Volume, Volume_Ratio |
| Momentum | RSI (14), MACD, MACD_Signal, MACD_Hist |
| Volatility bands | BB_Width, BB_Pct, ATR (14) |
| Price position | Range, RangePct, Price_vs_MA50, Price_vs_MA200 |
| Moving averages | MA7, MA21 |
| Calendar | DayOfWeek, Month |

### 3. Modeling
- **Model:** `VotingRegressor` — GradientBoosting (500 trees) + RandomForest (300 trees)
- **Scaling:** `RobustScaler` inside a `Pipeline` (no leakage)
- **Validation:** 5-fold walk-forward (time-series) cross-validation
- **Metrics:** RMSE, MAE, R², CV-RMSE, CV-R², Directional Accuracy

### 4. Explainability
- SHAP TreeExplainer on the GBR sub-model
- Summary dot plot + bar chart saved to `outputs/`

### 5. Forecasting
- ARIMA(5,1,0) on monthly-resampled close prices
- 5-year projection with 95% confidence interval

### 6. Visualizations (18 plots)
Close distribution, returns histogram, yearly bar/box/violin, pairplot, scatter, correlation heatmap, moving averages, Bollinger Bands, RSI, MACD, time series, ARIMA forecast, actual vs predicted, residuals, feature importance, walk-forward CV scores, volatility regime

---

## Results

| Metric | Value |
|---|---|
| Model | GBR + RF Ensemble |
| CV R² (mean) | improved vs baseline ~0.23 |
| Directional Accuracy | tracked per run |
| SHAP | ✅ integrated |

> Stock prediction is inherently noisy. Walk-forward CV gives a realistic estimate of out-of-sample performance.

---

## Project Structure
```
netflix-stock-project/
├── app/
│   ├── app.py               # Streamlit app (4 tabs: Predict, EDA, Charts, About)
│   └── STREAMLIT_URL.md     # Live deployment URL
├── data/
│   └── netflix.csv
├── models/
│   └── model.pkl            # Saved ensemble pipeline
├── outputs/                 # All generated plots + metrics.json
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── eda.py
│   ├── feature_engineering.py
│   ├── modeling.py          # Ensemble + walk-forward CV
│   ├── visualization.py     # 18 plots
│   ├── forecasting.py       # ARIMA with CI
│   ├── explainability.py    # SHAP analysis
│   ├── utils.py
│   └── test_model.py
├── .streamlit/
│   └── config.toml          # Netflix-themed dark UI
├── streamlit_app.py         # Streamlit Cloud entry point
├── main.py                  # Full pipeline runner
└── requirements.txt
```

---

## Running Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Train model + generate all outputs
python main.py

# Launch Streamlit app
streamlit run streamlit_app.py
```

---

## Streamlit Cloud Deployment
1. Push repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Set **Main file path** to `streamlit_app.py`
4. Deploy — the app reads `models/model.pkl` from the repo

> Make sure `models/model.pkl` is committed (or run `python main.py` in a setup step).

---

## Tech Stack
Python · Pandas · NumPy · Scikit-learn · Matplotlib · Seaborn · Statsmodels · SHAP · Streamlit · Joblib
