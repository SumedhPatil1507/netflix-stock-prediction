import argparse
from src.utils import setup_logging, create_output_folder, save_metrics, log_experiment
from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.eda import basic_eda
from src.feature_engineering import create_features
from src.visualization import run_all_visualizations
from src.modeling import train_model, save_model, FEATURES
from src.forecasting import arima_forecast
from src.explainability import shap_analysis
from src.backtest import run_backtest


def main(source: str = "csv"):
    logger = setup_logging()
    logger.info("🚀 Starting Netflix Stock Analysis Pipeline")

    create_output_folder()

    # ── Load & preprocess ─────────────────────────────────────────────────────
    logger.info(f"Loading data (source={source})...")
    df = load_data(source=source)

    logger.info("Preprocessing...")
    df = preprocess_data(df)

    # ── EDA ───────────────────────────────────────────────────────────────────
    logger.info("Running EDA...")
    basic_eda(df)

    # ── Feature engineering ───────────────────────────────────────────────────
    logger.info("Creating features...")
    df = create_features(df)
    logger.info(f"Features: {len(FEATURES)} | Rows: {len(df):,}")

    # ── Model training ────────────────────────────────────────────────────────
    logger.info("Training stacking ensemble (XGB + LGBM + RF + ET → Ridge)...")
    model, results, X_test, y_test, preds = train_model(df)

    logger.info("Model results:")
    for k, v in results.items():
        logger.info(f"  {k}: {v:.4f}")

    # ── Save model & metrics ──────────────────────────────────────────────────
    save_model(model)
    save_metrics(results)

    # ── Log experiment ────────────────────────────────────────────────────────
    log_experiment(
        params={"source": source, "n_features": len(FEATURES), "n_rows": len(df),
                "model": "ManualStacking(XGB+LGBM+RF+ET->Ridge)"},
        metrics=results,
    )

    # ── Backtest ──────────────────────────────────────────────────────────────
    logger.info("Running backtest...")
    y_test_ret = (y_test.values - X_test["Lag1"].values) / X_test["Lag1"].values * 100
    bt = run_backtest(y_test_ret, preds - X_test["Lag1"].values)
    logger.info("Backtest metrics:")
    for k, v in bt["metrics"].items():
        logger.info(f"  {k}: {v}")
    bt["curves"].to_csv("outputs/backtest_curves.csv")

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

    logger.info("✅ Pipeline completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Netflix Stock Prediction Pipeline")
    parser.add_argument("--source", choices=["csv", "live"], default="csv",
                        help="Data source: 'csv' (default) or 'live' (yfinance)")
    args = parser.parse_args()
    main(source=args.source)
