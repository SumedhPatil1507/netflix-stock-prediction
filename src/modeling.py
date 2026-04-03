import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

MODEL_PATH = "models/model.pkl"

FEATURES = [
    'Lag1', 'Lag2', 'Lag3', 'Lag5', 'Lag10',
    'RollingMean_5', 'RollingMean_10', 'RollingStd_5', 'RollingStd_10',
    'Return', 'LogReturn', 'Volatility',
    'Volume', 'Volume_Ratio',
    'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
    'BB_Width', 'BB_Pct',
    'ATR', 'Range', 'RangePct',
    'Price_vs_MA50', 'Price_vs_MA200',
    'MA7', 'MA21',
    'DayOfWeek', 'Month',
]


def _build_pipeline():
    gbr = GradientBoostingRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        min_samples_leaf=10,
        random_state=42,
    )
    rfr = RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )
    ensemble = VotingRegressor([('gbr', gbr), ('rfr', rfr)])
    return Pipeline([('scaler', RobustScaler()), ('model', ensemble)])


def train_model(df):
    df = df.copy().dropna(subset=FEATURES)

    X = df[FEATURES]
    y = df['Close'].shift(-1).loc[X.index]

    # drop last row (no target)
    X = X.iloc[:-1]
    y = y.iloc[:-1].dropna()
    X = X.loc[y.index]

    # ── Walk-forward cross-validation ─────────────────────────────────────────
    tscv = TimeSeriesSplit(n_splits=5)
    cv_rmse, cv_r2 = [], []

    for train_idx, val_idx in tscv.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        pipe = _build_pipeline()
        pipe.fit(X_tr, y_tr)
        preds = pipe.predict(X_val)
        cv_rmse.append(np.sqrt(mean_squared_error(y_val, preds)))
        cv_r2.append(r2_score(y_val, preds))

    print(f"  CV RMSE: {np.mean(cv_rmse):.4f} ± {np.std(cv_rmse):.4f}")
    print(f"  CV R²  : {np.mean(cv_r2):.4f} ± {np.std(cv_r2):.4f}")

    # ── Final model on 80 % train / 20 % test ────────────────────────────────
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    final_pipe = _build_pipeline()
    final_pipe.fit(X_train, y_train)
    preds = final_pipe.predict(X_test)

    results = {
        "RMSE":    float(np.sqrt(mean_squared_error(y_test, preds))),
        "MAE":     float(mean_absolute_error(y_test, preds)),
        "R2":      float(r2_score(y_test, preds)),
        "CV_RMSE": float(np.mean(cv_rmse)),
        "CV_R2":   float(np.mean(cv_r2)),
        "Dir_Acc": float(_directional_accuracy(y_test, preds)),
    }

    return final_pipe, results, X_test, y_test, preds


def _directional_accuracy(y_true, y_pred):
    """% of times the predicted direction matches actual direction."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    actual_dir = np.sign(np.diff(y_true))
    pred_dir   = np.sign(y_pred[1:] - y_true[:-1])
    return np.mean(actual_dir == pred_dir) * 100


def save_model(model):
    joblib.dump(model, MODEL_PATH)


def load_model():
    return joblib.load(MODEL_PATH)
