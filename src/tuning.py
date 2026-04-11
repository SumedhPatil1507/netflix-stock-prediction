"""
Optuna hyperparameter tuning for XGBoost base learner.
Run standalone: python -m src.tuning
Results saved to outputs/optuna_best_params.json
"""
import json
import logging
import numpy as np
import os

logger = logging.getLogger(__name__)
PARAMS_PATH = "outputs/optuna_best_params.json"


def tune_xgb(X_train, y_train, n_trials: int = 50, timeout: int = 300) -> dict:
    """
    Optimise XGBRegressor hyperparameters using Optuna.
    Uses TimeSeriesSplit CV to avoid data leakage.

    Parameters
    ----------
    n_trials : number of Optuna trials (default 50)
    timeout  : max seconds to run (default 5 min)

    Returns best params dict.
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        logger.warning("optuna not installed — skipping tuning")
        return {}

    from xgboost import XGBRegressor
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_squared_error
    from sklearn.preprocessing import RobustScaler

    scaler   = RobustScaler()
    X_scaled = scaler.fit_transform(X_train)
    tscv     = TimeSeriesSplit(n_splits=3)

    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 200, 1000),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "max_depth":        trial.suggest_int("max_depth", 3, 7),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "random_state": 42, "verbosity": 0, "n_jobs": -1,
        }
        scores = []
        for tr, val in tscv.split(X_scaled):
            m = XGBRegressor(**params)
            m.fit(X_scaled[tr], np.array(y_train)[tr])
            p = m.predict(X_scaled[val])
            scores.append(np.sqrt(mean_squared_error(np.array(y_train)[val], p)))
        return float(np.mean(scores))

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

    best = study.best_params
    os.makedirs("outputs", exist_ok=True)
    with open(PARAMS_PATH, "w") as f:
        json.dump(best, f, indent=2)

    logger.info(f"Best XGB params: {best}")
    logger.info(f"Best CV RMSE: {study.best_value:.4f}")
    return best


def load_best_params() -> dict:
    """Load previously tuned params, or return empty dict."""
    if os.path.exists(PARAMS_PATH):
        with open(PARAMS_PATH) as f:
            return json.load(f)
    return {}
