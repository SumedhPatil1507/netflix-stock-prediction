import copy
import logging
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

logger = logging.getLogger(__name__)

MODEL_PATH = "models/model.pkl"

FEATURES = [
    # Lag prices
    'Lag1', 'Lag2', 'Lag3', 'Lag5', 'Lag10', 'Lag20',
    # Lag returns
    'RetLag1', 'RetLag2', 'RetLag3', 'RetLag5',
    # Rolling stats
    'RollingMean_5', 'RollingMean_10', 'RollingStd_5', 'RollingStd_10',
    # Returns & volatility
    'Return', 'LogReturn', 'Volatility', 'Volatility_5', 'VolRatio_5_20',
    # Volume
    'Volume', 'Volume_Ratio', 'Volume_Ratio20', 'OBV_Ratio',
    # Momentum indicators
    'RSI', 'RSI7',
    'MACD', 'MACD_Signal', 'MACD_Hist', 'MACD_Norm',
    'EMA_Cross',
    # Bollinger
    'BB_Width', 'BB_Pct',
    # ATR / range
    'ATR_Norm', 'Range_Norm', 'RangePct',
    # Stochastic / Williams / CCI
    'Stoch_K', 'Stoch_D', 'Williams_R', 'CCI',
    # Price vs MAs
    'Price_vs_MA5', 'Price_vs_MA10', 'Price_vs_MA21',
    'Price_vs_MA50', 'Price_vs_MA200',
    # Momentum ratios
    'Momentum5', 'Momentum10', 'Momentum20',
    # Calendar
    'DayOfWeek', 'Month', 'Quarter', 'EarningsMonth',
    # Regime (added if regime detection ran)
    'Regime', 'Regime_Bear', 'Regime_Side', 'Regime_Bull',
]

# Regime features are optional — drop if not present
REGIME_FEATURES = ['Regime', 'Regime_Bear', 'Regime_Side', 'Regime_Bull']


def get_active_features(df) -> list:
    """Return FEATURES list filtered to columns that exist in df."""
    return [f for f in FEATURES if f in df.columns]


def _base_learners(cfg=None):
    xe = 600 if cfg is None else cfg.model.xgb_n_estimators
    le = 600 if cfg is None else cfg.model.lgbm_n_estimators
    re = 400 if cfg is None else cfg.model.rf_n_estimators
    ee = 400 if cfg is None else cfg.model.et_n_estimators

    return [
        ('xgb', XGBRegressor(
            n_estimators=xe, learning_rate=0.03, max_depth=5,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, n_jobs=-1, verbosity=0,
        )),
        ('lgbm', LGBMRegressor(
            n_estimators=le, learning_rate=0.03, max_depth=5,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, n_jobs=-1, verbose=-1,
        )),
        ('rf', RandomForestRegressor(
            n_estimators=re, max_depth=10, min_samples_leaf=3,
            max_features=0.6, random_state=42, n_jobs=-1,
        )),
        ('et', ExtraTreesRegressor(
            n_estimators=ee, max_depth=10, min_samples_leaf=3,
            max_features=0.6, random_state=42, n_jobs=-1,
        )),
    ]


class ManualStackingRegressor:
    """
    2-level stacking:
      Level-0 : XGB, LGBM, RandomForest, ExtraTrees
      Level-1 : Ridge meta-learner trained on OOF predictions
    RobustScaler applied before all learners.
    """

    def __init__(self, n_splits: int = 3, cfg=None):
        self.n_splits         = n_splits
        self.scaler           = RobustScaler()
        self.learners         = _base_learners(cfg)
        self.meta             = Ridge(alpha=1.0 if cfg is None else cfg.model.ridge_alpha)
        self.fitted_learners_ = []

    def fit(self, X, y):
        if hasattr(X, 'columns'):
            self.feature_names_ = list(X.columns)
        X = np.array(X)
        y = np.array(y)
        X_scaled = self.scaler.fit_transform(X)
        tscv     = TimeSeriesSplit(n_splits=self.n_splits)
        oof      = np.zeros((len(X), len(self.learners)))

        for j, (name, est) in enumerate(self.learners):
            for tr_idx, val_idx in tscv.split(X_scaled):
                cl = copy.deepcopy(est)
                cl.fit(X_scaled[tr_idx], y[tr_idx])
                oof[val_idx, j] = cl.predict(X_scaled[val_idx])

        self.meta.fit(oof, y)

        self.fitted_learners_ = []
        for name, est in self.learners:
            cl = copy.deepcopy(est)
            cl.fit(X_scaled, y)
            self.fitted_learners_.append((name, cl))

        return self

    def predict(self, X):
        # If model knows its training features, select/reorder to match exactly
        if hasattr(self, 'feature_names_') and hasattr(X, 'columns'):
            # Add any missing columns as 0, drop any extra columns
            for col in self.feature_names_:
                if col not in X.columns:
                    X = X.copy()
                    X[col] = 0.0
            X = X[self.feature_names_]
        X_scaled = self.scaler.transform(np.array(X))
        preds    = np.column_stack([
            est.predict(X_scaled) for _, est in self.fitted_learners_
        ])
        return self.meta.predict(preds)

    @property
    def feature_importances_(self):
        imps, count = None, 0
        for _, est in self.fitted_learners_:
            if hasattr(est, 'feature_importances_'):
                fi   = est.feature_importances_
                imps = fi if imps is None else imps + fi
                count += 1
        return imps / count if count > 0 else None


def _build_model(cfg=None):
    n = 3 if cfg is None else cfg.model.n_splits_oof
    return ManualStackingRegressor(n_splits=n, cfg=cfg)


def train_model(df, cfg=None):
    active_features = get_active_features(df)
    df = df.copy().dropna(subset=active_features)

    X          = df[active_features]
    next_close = df['Close'].shift(-1).loc[X.index]
    curr_close = df['Close'].loc[X.index]
    y_return   = (next_close - curr_close) / curr_close * 100
    y_close    = next_close

    X          = X.iloc[:-1]
    y_return   = y_return.iloc[:-1].dropna()
    y_close    = y_close.iloc[:-1].dropna()
    X          = X.loc[y_return.index]
    curr_close = curr_close.iloc[:-1].loc[y_return.index]

    n_cv   = 5 if cfg is None else cfg.model.n_splits_cv
    ratio  = 0.8 if cfg is None else cfg.model.train_ratio
    tscv   = TimeSeriesSplit(n_splits=n_cv)
    cv_rmse, cv_r2 = [], []

    logger.info("Running walk-forward CV...")
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X), 1):
        m = _build_model(cfg)
        m.fit(X.iloc[tr_idx], y_return.iloc[tr_idx])
        p = m.predict(X.iloc[val_idx])
        cv_rmse.append(float(np.sqrt(mean_squared_error(y_return.iloc[val_idx], p))))
        cv_r2.append(float(r2_score(y_return.iloc[val_idx], p)))
        logger.info(f"  Fold {fold}: RMSE={cv_rmse[-1]:.4f}  R2={cv_r2[-1]:.4f}")

    logger.info(f"CV RMSE: {np.mean(cv_rmse):.4f} +/- {np.std(cv_rmse):.4f}")
    logger.info(f"CV R2  : {np.mean(cv_r2):.4f} +/- {np.std(cv_r2):.4f}")

    split      = int(len(X) * ratio)
    X_train, X_test    = X.iloc[:split],        X.iloc[split:]
    y_tr_ret, y_te_ret = y_return.iloc[:split], y_return.iloc[split:]
    curr_test          = curr_close.iloc[split:]
    y_te_close         = y_close.iloc[split:]

    # ── Conformal calibration on last 10% of train ────────────────────────────
    from src.uncertainty import ConformalPredictor
    cal_split  = int(len(X_train) * 0.9)
    X_tr2      = X_train.iloc[:cal_split]
    y_tr2      = y_tr_ret.iloc[:cal_split]
    X_cal      = X_train.iloc[cal_split:]
    y_cal      = y_tr_ret.iloc[cal_split:]

    final_model = _build_model(cfg)
    final_model.fit(X_tr2, y_tr2)

    alpha = 0.1 if cfg is None else cfg.conformal.alpha
    cp = ConformalPredictor(alpha=alpha)
    cp.calibrate(final_model, X_cal.values, y_cal.values)

    # Refit on full train for final predictions
    final_model_full = _build_model(cfg)
    final_model_full.fit(X_train, y_tr_ret)
    final_model_full.conformal_     = cp
    final_model_full.feature_names_ = list(X_train.columns)  # exact features used at train time

    pred_returns = final_model_full.predict(X_test)
    pred_prices  = curr_test.values * (1 + pred_returns / 100)

    # Conformal intervals on test set (in return space)
    lo_ret, hi_ret = cp.predict_interval(X_test.values)
    coverage       = cp.coverage(X_test.values, y_te_ret.values)
    logger.info(f"Conformal coverage: {coverage:.3f} (target >= {1-alpha:.2f})")
    logger.info(f"Interval width: {cp.interval_width:.4f}%")

    results = {
        "RMSE":         float(np.sqrt(mean_squared_error(y_te_close, pred_prices))),
        "MAE":          float(mean_absolute_error(y_te_close, pred_prices)),
        "R2":           float(r2_score(y_te_close, pred_prices)),
        "Ret_R2":       float(r2_score(y_te_ret, pred_returns)),
        "CV_RMSE":      float(np.mean(cv_rmse)),
        "CV_R2":        float(np.mean(cv_r2)),
        "Dir_Acc":      float(_directional_accuracy(y_te_ret.values, pred_returns)),
        "CP_Coverage":  float(coverage),
        "CP_Width":     float(cp.interval_width),
    }

    return final_model_full, results, X_test, y_te_close, pred_prices


def _directional_accuracy(y_true_ret, pred_ret):
    return float(np.mean(np.sign(y_true_ret) == np.sign(pred_ret)) * 100)


def save_model(model):
    joblib.dump(model, MODEL_PATH)
    logger.info(f"Model saved -> {MODEL_PATH}")


def load_model():
    return joblib.load(MODEL_PATH)
