import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

MODEL_PATH = "models/model.pkl"

def train_model(df):
    features = ['Open','High','Low','Close','Volume','MA7','MA21','Return']

    df = df.dropna()

    X = df[features]
    y = df['Close'].shift(-1)

    X = X[:-1]
    y = y[:-1]

    split = int(len(X) * 0.8)

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    results = {
        "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
        "MAE": mean_absolute_error(y_test, preds),
        "R2": r2_score(y_test, preds)
    }

    return model, results, X_test

def save_model(model):
    joblib.dump(model, MODEL_PATH)

def load_model():
    return joblib.load(MODEL_PATH)