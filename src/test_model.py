import joblib
import pandas as pd

model = joblib.load("models/model.pkl")

columns = [
    'Lag1','Lag2','Lag3','Lag5','Lag10',
    'RollingMean_5','RollingMean_10',
    'RollingStd_5',
    'Return','Volatility',
    'Volume','Range'
]

# Example input (replace with real data later)
sample = pd.DataFrame([[100,99,98,97,95,
                        100,101,
                        1.5,
                        0.01,0.02,
                        1000000,5]],
                      columns=columns)

# Predict return
pred_return = model.predict(sample)[0]

# Convert to price (assuming last close = 100)
last_close = 100
pred_price = last_close * (1 + pred_return)

print("📈 Predicted Return:", pred_return)
print("📈 Predicted Next Price:", pred_price)