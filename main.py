import argparse
from src.utils import setup_logging, create_output_folder, save_metrics, log_experiment
from src.pipeline_config import load_config, save_default_config
from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.eda import basic_eda
from src.feature_engineering import create_features
from src.regime_detection import fit_and_add_regimes
from src.visualization import run_all_visualizations
from src.modeling import train_model, save_model, get_active_features
from src.forecasting import arima_forecast
from src.explainability import shap_analysis
from src.backtest import run_backtest


def main(source: str = "csv", config_path: str = "config.yaml"):
    cfg    = load_config(config_path)
    logger = setup_logging(cfg.log_level)
    logger.info("Starting Netflix Stock Prediction Pipeline")

    create_output_folder()

    # ── Load & preprocess ─────────────────────────────────────────────────────
    logger.info(f"Loading data (source={source})...")
    df = load_data(source=source)
    df = preprocess_data(df)

    # ── EDA ───────────────────────────────────────────────────────────────────
    logger.info("Running EDA...")
    basic_eda(df)

    # ── Feature engineering ───────────────────────────────────────────────────
    logger.info("Creating features...")
    df = create_features(df)

    # ── Regime detection ──────────────────────────────────────────────────────
    if cfg.regime.enabled:
        logger.info("Fitting HMM regime detector...")
        try:
            df, regime_detector = fit_and_add_regimes(df)
            regime_detector.save()
            regime_counts = df['Regime'].value_counts().to_dict()
            logger.info(f"Regime distribution: {regime_counts}")
        except Exception as e:
            logger.warning(f"Regime detection skipped: {e}")

    active_features = get_active_features(df)
    logger.info(f"Active features: {len(active_features)} | Rows: {len(df):,}")

    # ── Model training ────────────────────────────────────────────────────────
    logger.info("Training stacking ensemble (XGB + LGBM + RF + ET -> Ridge)...")
    model, results, X_test, y_test, preds = train_model(df, cfg=cfg)

    logger.info("Model results:")
    for k, v in results.items():
        logger.info(f"  {k}: {v:.4f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    save_model(model)
    save_metrics(results)
    log_experiment(
        params={"source": source, "n_features": len(active_features),
                "n_rows": len(df), "model": "ManualStacking(XGB+LGBM+RF+ET->Ridge)",
                "regime_enabled": cfg.regime.enabled},
        metrics=results,
    )

    # ── Backtest ──────────────────────────────────────────────────────────────
    logger.info("Running backtest...")
    y_test_ret = (y_test.values - X_test["Lag1"].values) / X_test["Lag1"].values * 100
    pred_ret   = preds - X_test["Lag1"].values
    bt = run_backtest(y_test_ret, pred_ret,
                      transaction_cost=cfg.backtest.transaction_cost,
                      rf_annual=cfg.backtest.risk_free_rate)
    for k, v in bt["metrics"].items():
        logger.info(f"  {k}: {v}")
    bt["curves"].to_csv("outputs/backtest_curves.csv")

    import numpy as np
    import pandas as pd
    pd.Series(bt["rolling_sharpe"]).to_csv("outputs/rolling_sharpe.csv", index=False)

    # ── Visualizations ────────────────────────────────────────────────────────
    logger.info("Generating visualizations...")
    run_all_visualizations(df, model=model, y_test=y_test,
                           preds=preds, rmse=results["RMSE"], r2=results["R2"])

    # ── SHAP ──────────────────────────────────────────────────────────────────
    logger.info("Running SHAP analysis...")
    try:
        shap_analysis(model, X_test)
    except Exception as e:
        logger.warning(f"SHAP skipped: {e}")

    # ── ARIMA forecast ────────────────────────────────────────────────────────
    logger.info("Running ARIMA forecast...")
    forecast = arima_forecast(df["Close"])
    logger.info(f"Forecast (next 10 months):\n{forecast.head(10)}")

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
