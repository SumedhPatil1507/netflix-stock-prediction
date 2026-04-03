"""
Quick smoke-test: load the saved model and run a single prediction.
All feature values are illustrative — replace with real data as needed.
"""
import joblib
import pandas as pd
from src.modeling import FEATURES

model = joblib.load("models/model.pkl")

sample_values = {
    'Lag1': 650.0, 'Lag2': 645.0, 'Lag3': 640.0, 'Lag5': 635.0,
    'Lag10': 620.0, 'Lag20': 600.0,
    'RetLag1': 0.8, 'RetLag2': -0.3, 'RetLag3': 0.5, 'RetLag5': 1.2,
    'RollingMean_5': 648.0, 'RollingMean_10': 642.0,
    'RollingStd_5': 4.5, 'RollingStd_10': 6.0,
    'Return': 0.8, 'LogReturn': 0.008, 'Volatility': 1.2, 'Volatility_5': 0.9,
    'Volume': 5_000_000, 'Volume_Ratio': 1.05, 'Volume_Ratio20': 0.98, 'OBV_Ratio': 1.01,
    'RSI': 55.0, 'RSI7': 58.0,
    'MACD': 2.5, 'MACD_Signal': 2.0, 'MACD_Hist': 0.5, 'MACD_Norm': 0.004,
    'EMA_Cross': 1.2,
    'BB_Width': 0.06, 'BB_Pct': 0.65,
    'ATR_Norm': 0.018, 'Range_Norm': 0.023, 'RangePct': 2.3,
    'Stoch_K': 62.0, 'Stoch_D': 58.0, 'Williams_R': -38.0, 'CCI': 45.0,
    'Price_vs_MA5': 0.003, 'Price_vs_MA10': 0.008, 'Price_vs_MA21': 0.015,
    'Price_vs_MA50': 0.02, 'Price_vs_MA200': 0.08,
    'Momentum5': 0.012, 'Momentum10': 0.025, 'Momentum20': 0.04,
    'DayOfWeek': 2, 'Month': 4, 'Quarter': 2,
}

sample = pd.DataFrame([sample_values])[FEATURES]

# Model predicts next-day return (%)
pred_return = model.predict(sample)[0]
last_close  = sample_values['Lag1']
pred_price  = last_close * (1 + pred_return / 100)

print(f"📈 Predicted Next-Day Return : {pred_return:+.4f}%")
print(f"📈 Predicted Next Close      : ${pred_price:.2f}")
