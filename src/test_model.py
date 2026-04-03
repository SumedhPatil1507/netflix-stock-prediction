"""
Quick smoke-test: load the saved model and run a single prediction.
All feature values are illustrative — replace with real data as needed.
"""
import joblib
import pandas as pd
from src.modeling import FEATURES

model = joblib.load("models/model.pkl")

# Build a sample row that matches the training feature set
sample_values = {
    'Lag1': 650.0, 'Lag2': 645.0, 'Lag3': 640.0, 'Lag5': 635.0, 'Lag10': 620.0,
    'RollingMean_5': 648.0, 'RollingMean_10': 642.0,
    'RollingStd_5': 4.5, 'RollingStd_10': 6.0,
    'Return': 0.8, 'LogReturn': 0.008, 'Volatility': 1.2,
    'Volume': 5_000_000, 'Volume_Ratio': 1.05,
    'RSI': 55.0, 'MACD': 2.5, 'MACD_Signal': 2.0, 'MACD_Hist': 0.5,
    'BB_Width': 0.06, 'BB_Pct': 0.65,
    'ATR': 12.0, 'Range': 15.0, 'RangePct': 2.3,
    'Price_vs_MA50': 0.02, 'Price_vs_MA200': 0.08,
    'MA7': 647.0, 'MA21': 640.0,
    'DayOfWeek': 2, 'Month': 4,
}

sample = pd.DataFrame([sample_values])[FEATURES]

pred_price = model.predict(sample)[0]
last_close = sample_values['Lag1']
pred_return = (pred_price - last_close) / last_close * 100

print(f"📈 Predicted Next Close : {pred_price:.2f}")
print(f"📈 Implied Return       : {pred_return:+.2f}%")
