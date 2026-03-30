# 📈 Netflix Stock Prediction

## 🚀 Overview
This project is an end-to-end **Data Science & Machine Learning pipeline** that analyzes and predicts the next-day movement of Netflix stock using historical data.

It includes data preprocessing, feature engineering, visualization, model training, and forecasting — built with a modular and production-ready structure.

---

## 🎯 Problem Statement
Predict the **next-day stock behavior** of Netflix using historical price and volume data.

---

## 📊 Dataset
- Source: Historical Netflix stock data (yFinance)
- Features:
  - Open, High, Low, Close prices
  - Volume
  - Stock splits
- Time Period: ~2002 to 2026

---

## ⚙️ Project Pipeline

### 1. Data Processing
- Date parsing and indexing
- Handling missing values
- Data validation checks

### 2. Feature Engineering
- Lag features (Lag1, Lag2, Lag3, Lag5, Lag10)
- Rolling statistics (moving averages, standard deviation)
- Volatility estimation
- Price range calculation

### 3. Visualization
- Histograms and distributions
- Bar plots, box plots, violin plots
- Correlation heatmap
- Time series trends
- Feature importance plots
- Forecast visualization

### 4. Modeling
- Model: **Random Forest Regressor**
- Time-series aware train-test split
- No data leakage (future data not used in training)

### 5. Forecasting
- ARIMA-based time series forecasting
- 5-year projection of stock trends

---

## 📈 Results

| Metric | Value |
|------|--------|
| Model | Random Forest |
| R² Score | ~0.23 |
| RMSE | Moderate |
| MAE | Moderate |

> ⚠️ Note: Stock prediction is inherently noisy. An R² of ~0.23 is realistic and indicates meaningful predictive signal without data leakage.

---

## 🧠 Key Learnings

- Avoiding **data leakage** is critical in time-series modeling
- Stock prices are highly noisy and difficult to predict
- Feature engineering (lags, rolling stats) significantly improves performance
- Real-world ML models often have lower but more reliable accuracy

---

## 🛠️ Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Statsmodels (ARIMA)
- Joblib (model persistence)

---

## 📁 Project Structure
netflix-stock-project/
│
├── data/
├── src/
│ ├── data_loader.py
│ ├── preprocessing.py
│ ├── feature_engineering.py
│ ├── modeling.py
│ ├── visualization.py
│ ├── forecasting.py
│ ├── utils.py
│
├── models/
├── outputs/
├── main.py
├── test_model.py
└── README.md

