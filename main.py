from __future__ import annotations
import os
import argparse
import numpy as np
import pandas as pd

from src.utils import setup_logging, create_output_folder, save_metrics, log_experiment
from src.pipeline_config import load_config, save_default_config
from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.feature_engineering import create_features
from src.modeling import train_model, save_model, get_active_features
from src.backtest import run_backtest


def main(source: str = "csv", config_path: str = "config.yaml") -> None:
    cfg    = load_config(config_path)
    logger = setup_logging(cfg.log_level)
    logger.info("Starting Netflix Stock Prediction Pipeline")

    create_output_folder()

    # ── Load & preprocess ─────────────────────────────────────────────────────
    logger.info(f"Loading data (source={source})...")
    try:
        df = load_data(source=source)
        df = preprocess_data(df)
        logger.info(f"Data loaded: {len(df):,} rows, {df.index.min().date()} to {df.index.max().date()}")
    except Exception as e:
        logger.error(f"Data loading failed: {e}")
        raise

    # ── Feature engineering ───────────────────────────────────────────────────
    logger.info("Creating features...")
    try:
        df = create_features(df)
    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        raise

    # ── Regime detection ──────────────────────────────────────────────────────
    if cfg.regime.enabled:
        logger.info("Fitting HMM regime detector...")
        try:
            from src.regime_detection import fit_and_add_regimes
            df, regime_detector = fit_and_add_regimes(df)
            regime_detector.save()
            logger.info(f"Regime distribution: {df['Regime'].value_counts().to_dict()}")
        except Exception as e:
            logger.warning(f"Regime detection skipped: {e}")

    active_features = get_active_features(df)
    logger.info(f"Active features: {len(active_features)} | Rows: {len(df):,}")

    # ── Model training ────────────────────────────────────────────────────────
    logger.info("Training stacking ensemble (XGB + LGBM + RF + ET -> Ridge)...")
    try:
        model, results, X_test, y_test, preds = train_model(df, cfg=cfg)
        logger.info("Model results:")
        for k, v in results.items():
            logger.info(f"  {k}: {v:.4f}")
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        raise

    # ── Save model & metrics ──────────────────────────────────────────────────
    save_model(model)
    save_metrics(results)

    # ── Cache features for fast Streamlit load ────────────────────────────────
    try:
        cache_path = os.path.join("outputs", "features_cache.parquet")
        # Ensure all columns are float64 before saving — prevents string dtype issues
        df_cache = df.copy()
        for col in df_cache.columns:
            df_cache[col] = pd.to_numeric(df_cache[col], errors="coerce")
        df_cache.to_parquet(cache_path)
        logger.info(f"Feature cache saved -> {cache_path}")
    except Exception as e:
        logger.warning(f"Feature cache save failed: {e}")

    # ── Experiment log ────────────────────────────────────────────────────────
    log_experiment(
        params={"source": source, "n_features": len(active_features),
                "n_rows": len(df), "model": "ManualStacking(XGB+LGBM+RF+ET->Ridge)",
                "regime_enabled": cfg.regime.enabled},
        metrics=results,
    )

    # ── Backtest ──────────────────────────────────────────────────────────────
    logger.info("Running backtest...")
    try:
        y_test_ret = (y_test.values - X_test["Lag1"].values) / X_test["Lag1"].values * 100
        pred_ret   = preds - X_test["Lag1"].values
        bt = run_backtest(y_test_ret, pred_ret,
                          transaction_cost=cfg.backtest.transaction_cost,
                          rf_annual=cfg.backtest.risk_free_rate)
        for k, v in bt["metrics"].items():
            logger.info(f"  {k}: {v}")
        bt["curves"].to_csv("outputs/backtest_curves.csv")
        pd.Series(bt["rolling_sharpe"]).to_csv("outputs/rolling_sharpe.csv", index=False)
    except Exception as e:
        logger.warning(f"Backtest failed: {e}")

    # ── Visualizations ────────────────────────────────────────────────────────
    logger.info("Generating visualizations...")
    try:
        from src.visualization import run_all_visualizations
        run_all_visualizations(df, model=model, y_test=y_test,
                               preds=preds, rmse=results["RMSE"], r2=results["R2"])
    except Exception as e:
        logger.warning(f"Visualizations skipped: {e}")

    # ── SHAP ──────────────────────────────────────────────────────────────────
    logger.info("Running SHAP analysis...")
    try:
        from src.explainability import shap_analysis
        shap_analysis(model, X_test)
    except Exception as e:
        logger.warning(f"SHAP skipped: {e}")

    # ── Drift detection ───────────────────────────────────────────────────────
    logger.info("Running drift detection...")
    try:
        from src.drift import detect_drift, drift_summary_df
        split    = int(len(df) * 0.8)
        dr       = detect_drift(df.iloc[:split], df.iloc[split:], active_features)
        drift_df = drift_summary_df(dr)
        drift_df.to_csv("outputs/drift_report.csv", index=False)
        logger.info(f"Drift: {len(dr['drifted_features'])} features drifted")
        if dr["overall_drift"]:
            logger.warning("Significant overall drift — consider retraining soon")
    except Exception as e:
        logger.warning(f"Drift detection skipped: {e}")

    # ── ARIMA forecast ────────────────────────────────────────────────────────
    logger.info("Running ARIMA forecast...")
    try:
        from src.forecasting import arima_forecast
        forecast = arima_forecast(df["Close"])
        logger.info(f"Forecast (next 10 months):\n{forecast.head(10)}")
    except Exception as e:
        logger.warning(f"ARIMA forecast skipped: {e}")

    logger.info("Pipeline completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Netflix Stock Prediction Pipeline")
    parser.add_argument("--source", choices=["csv", "live"], default="csv")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--save-config", action="store_true",
                        help="Write default config.yaml and exit")
    args = parser.parse_args()

    if args.save_config:
        save_default_config(args.config)
    else:
        main(source=args.source, config_path=args.config)
