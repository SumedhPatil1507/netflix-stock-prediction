import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

MODEL_PATH = "models/model.pkl"

FEATURES = [
    'Lag1', 'Lag2', 'Lag3', 'Lag5', 'Lag10', 'Lag20',
    'RetLag1', 'RetLag2', 'RetLag3', 'RetLag5',
    'RollingMean_5', 'RollingMean_10', 'RollingStd_5', 'RollingStd_10',
    'Return', 'LogReturn', 'Volatility', 'Volatility_5',
    'Volume', 'Volume_Ratio', 'Volume_Ratio20', 'OBV_Ratio',
    'RSI', 'RSI7',
    'MACD', 'MACD_Signal', 'MACD_Hist', 'MACD_Norm',
    'EMA_Cross',
    'BB_Width', 'BB_Pct',
    'ATR_Norm', 'Range_Norm', 'RangePct',
    'Stoch_K', 'Stoch_D', 'Williams_R', 'CCI',
    'Price_vs_MA5', 'Price_vs_MA10', 'Price_vs_MA21',
    'Price_vs_MA50', 'Price_vs_MA200',
    'Momentum5', 'Momentum10', 'Momentum20',
    'DayOfWeek', 'Month', 'Quarter',
]


# ── Base learners ─────────────────────────────────────────────────────────────
def _base_learners():
    return [
        ('xgb', XGBRegressor(
            n_estimators=600, learning_rate=0.03, max_depth=5,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, n_jobs=-1, verbosity=0,
        )),
        ('lgbm', LGBMRegressor(
            n_estimators=600, learning_rate=0.03, max_depth=5,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, n_jobs=-1, verbose=-1,
        )),
        ('rf', RandomForestRegressor(
            n_estimators=400, max_depth=10, min_samples_leaf=3,
            max_features=0.6, random_state=42, n_jobs=-1,
        )),
        ('et', ExtraTreesRegressor(
            n_estimators=400, max_depth=10, min_samples_leaf=3,
            max_features=0.6, random_state=42, n_jobs=-1,
        )),
    ]


class ManualStackingRegressor:
    """
    Manual 2-level stacking:
      Level-0: XGB, LGBM, RF, ExtraTrees  (trained on scaled features)
      Level-1: Ridge meta-learner          (trained on OOF predictions)
    Avoids sklearn StackingRegressor's is_regressor() validator entirely.
    """

    def __init__(self, n_splits=3):
        self.n_splits  = n_splits
        self.scaler    = RobustScaler()
        self.learners  = _base_learners()
        self.meta      = Ridge(alpha=1.0)
        self.fitted_learners_ = []

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        X_scaled = self.scaler.fit_transform(X)
        tscv     = TimeSeriesSplit(n_splits=self.n_splits)
        oof      = np.zeros((len(X), len(self.learners)))

        # ── OOF predictions for meta-learner ──────────────────────────────────
        for j, (name, est) in enumerate(self.learners):
            for tr_idx, val_idx in tscv.split(X_scaled):
                import copy
                clone = copy.deepcopy(est)
                clone.fit(X_scaled[tr_idx], y[tr_idx])
                oof[val_idx, j] = clone.predict(X_scaled[val_idx])

        # ── Train meta-learner on OOF ─────────────────────────────────────────
        self.meta.fit(oof, y)

        # ── Refit all base learners on full training data ─────────────────────
        self.fitted_learners_ = []
        for name, est in self.learners:
            import copy
            clone = copy.deepcopy(est)
            clone.fit(X_scaled, y)
            self.fitted_learners_.append((name, clone))

        return self

    def predict(self, X):
        X_scaled = self.scaler.transform(np.array(X))
        preds    = np.column_stack([
            est.predict(X_scaled) for _, est in self.fitted_learners_
        ])
        return self.meta.predict(preds)

    @property
    def feature_importances_(self):
        """Average feature importances from tree-based base learners."""
        imps, count = None, 0
        for _, est in self.fitted_learners_:
            if hasattr(est, 'feature_importances_'):
                fi = est.feature_importances_
                imps = fi if imps is None else imps + fi
                count += 1
        return imps / count if count > 0 else None


def _build_model():
    return ManualStackingRegressor(n_splits=3)


def train_model(df):
    df = df.copy().dropna(subset=FEATURES)

    X          = df[FEATURES]
    next_close = df['Close'].shift(-1).loc[X.index]
    curr_close = df['Close'].loc[X.index]
    y_return   = (next_close - curr_close) / curr_close * 100
    y_close    = next_close

    # Drop last row (no target)
    X          = X.iloc[:-1]
    y_return   = y_return.iloc[:-1].dropna()
    y_close    = y_close.iloc[:-1].dropna()
    X          = X.loc[y_return.index]
    curr_close = curr_close.iloc[:-1].loc[y_return.index]

    # ── Walk-forward CV ───────────────────────────────────────────────────────
    tscv = TimeSeriesSplit(n_splits=5)
    cv_rmse, cv_r2 = [], []

    print("  Running walk-forward CV...")
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X), 1):
        m = _build_model()
        m.fit(X.iloc[tr_idx], y_return.iloc[tr_idx])
        p = m.predict(X.iloc[val_idx])
        cv_rmse.append(float(np.sqrt(mean_squared_error(y_return.iloc[val_idx], p))))
        cv_r2.append(float(r2_score(y_return.iloc[val_idx], p)))
        print(f"    Fold {fold}: RMSE={cv_rmse[-1]:.4f}  R²={cv_r2[-1]:.4f}")

    print(f"  CV RMSE : {np.mean(cv_rmse):.4f} ± {np.std(cv_rmse):.4f}")
    print(f"  CV R²   : {np.mean(cv_r2):.4f} ± {np.std(cv_r2):.4f}")

    # ── Final model on 80/20 split ────────────────────────────────────────────
    split      = int(len(X) * 0.8)
    X_train, X_test     = X.iloc[:split],          X.iloc[split:]
    y_tr_ret, y_te_ret  = y_return.iloc[:split],   y_return.iloc[split:]
    curr_test           = curr_close.iloc[split:]
    y_te_close          = y_close.iloc[split:]

    final_model = _build_model()
    final_model.fit(X_train, y_tr_ret)

    pred_returns = final_model.predict(X_test)
    pred_prices  = curr_test.values * (1 + pred_returns / 100)

    results = {
        "RMSE":    float(np.sqrt(mean_squared_error(y_te_close, pred_prices))),
        "MAE":     float(mean_absolute_error(y_te_close, pred_prices)),
        "R2":      float(r2_score(y_te_close, pred_prices)),
        "Ret_R2":  float(r2_score(y_te_ret, pred_returns)),
        "CV_RMSE": float(np.mean(cv_rmse)),
        "CV_R2":   float(np.mean(cv_r2)),
        "Dir_Acc": float(_directional_accuracy(y_te_ret.values, pred_returns)),
    }

    return final_model, results, X_test, y_te_close, pred_prices


def _directional_accuracy(y_true_ret, pred_ret):
    return float(np.mean(np.sign(y_true_ret) == np.sign(pred_ret)) * 100)


def save_model(model):
    joblib.dump(model, MODEL_PATH)


def load_model():
    return joblib.load(MODEL_PATH)
