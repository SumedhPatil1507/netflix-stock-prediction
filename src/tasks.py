"""
Celery Task Orchestration Layer.
Completely isolates heavy calculations (feature engineering, model training) out-of-band.
Supports local execution fallbacks if Celery is not running.
"""
from __future__ import annotations
import os
import logging
from celery import Celery

logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Initialize Celery app
celery_app = Celery(
    "alpha_tasks",
    broker=REDIS_URL,
    backend=REDIS_URL
)

# Optional configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_always_eager=os.getenv("CELERY_ALWAYS_EAGER", "False").lower() in ("true", "1", "t")
)

@celery_app.task(name="tasks.calculate_features_task")
def calculate_features_task(ticker: str = "NFLX", interval: str = "daily", days_back: int = 730) -> bool:
    """Out-of-band task to fetch raw time-series, compute features, and cache them in Redis."""
    logger.info(f"Asynchronous Feature Calculation started for {ticker} (interval={interval}).")
    try:
        from src.data_loader import load_data
        from src.preprocessing import preprocess_data
        from src.feature_engineering import create_features
        from src.features_redis import cache_features
        from src.time_series_db import save_ohlcv_data
        
        # Load raw data (this will query external APIs like Alpaca/AlphaVantage and/or TimescaleDB)
        logger.info(f"Loading data for {ticker}...")
        df_raw = load_data(source="alpaca", ticker=ticker, interval=interval, days_back=days_back)
        
        if df_raw.empty:
            logger.error(f"No raw data loaded for {ticker}. Task aborted.")
            return False
            
        # Persist raw data to TimescaleDB
        save_ohlcv_data(df_raw, ticker)
        
        # Preprocess and engineer features
        logger.info(f"Preprocessing data for {ticker}...")
        df_preprocessed = preprocess_data(df_raw)
        
        logger.info(f"Calculating features for {ticker}...")
        df_features = create_features(df_preprocessed)
        
        # Cache features in Redis for fast API/Streamlit lookup
        logger.info(f"Caching features to Redis for {ticker}...")
        cache_features(df_features, ticker)
        
        logger.info(f"Asynchronous Feature Calculation completed successfully for {ticker}.")
        return True
    except Exception as e:
        logger.error(f"Error in calculate_features_task for {ticker}: {e}", exc_info=True)
        return False

@celery_app.task(name="tasks.retrain_model_task")
def retrain_model_task(ticker: str = "NFLX", config_path: str = "config.yaml") -> dict | None:
    """Out-of-band task to run the complete model retraining pipeline."""
    logger.info(f"Asynchronous Model Retraining started for {ticker} using config {config_path}.")
    try:
        from src.pipeline_config import load_config
        from src.utils import create_output_folder, save_metrics, log_experiment
        from src.data_loader import load_data
        from src.preprocessing import preprocess_data
        from src.feature_engineering import create_features
        from src.modeling import train_model, get_active_features
        from src.model_registry import save_versioned_model, get_latest_version
        from src.monitoring import alert_drift, alert_retrain_complete
        from src.features_redis import cache_features
        from src.time_series_db import save_ohlcv_data
        
        cfg = load_config(config_path)
        create_output_folder()
        
        # 1. Load, persist raw, preprocess, compute features
        df_raw = load_data(source="alpaca", ticker=ticker, days_back=1000)
        if df_raw.empty:
            raise ValueError(f"No raw data retrieved for retraining ticker {ticker}.")
            
        save_ohlcv_data(df_raw, ticker)
        df_prep = preprocess_data(df_raw)
        df_feat = create_features(df_prep)
        
        # Cache features in Redis
        cache_features(df_feat, ticker)
        
        # 2. Train model
        active_features = get_active_features(df_feat)
        model, results, X_test, y_test, preds = train_model(df_feat, cfg=cfg)
        
        # 3. Save to secure object store model registry
        versioned_path = save_versioned_model(model, results, ticker=ticker)
        save_metrics(results)
        
        # 4. Log experiment
        log_experiment(
            params={
                "source": "database_retrain",
                "ticker": ticker,
                "n_features": len(active_features),
                "n_rows": len(df_feat),
                "model": "ManualStacking(XGB+LGBM+RF+ET->Ridge)",
                "regime_enabled": cfg.regime.enabled
            },
            metrics=results
        )
        
        # 5. Drift check
        try:
            from src.drift import detect_drift
            split = int(len(df_feat) * 0.8)
            dr = detect_drift(df_feat.iloc[:split], df_feat.iloc[split:], active_features)
            if dr["overall_drift"]:
                alert_drift(len(dr["drifted_features"]), len(active_features), ticker)
        except Exception as de:
            logger.warning(f"Drift check skipped in task: {de}")
            
        # 6. Send alert
        latest_ver = get_latest_version() or "unknown"
        alert_retrain_complete(ticker, results, latest_ver)
        
        logger.info(f"Asynchronous Model Retraining completed successfully. New version: {latest_ver}")
        return results
    except Exception as e:
        logger.error(f"Error in retrain_model_task for {ticker}: {e}", exc_info=True)
        return None
