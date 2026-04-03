from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.eda import basic_eda
from src.feature_engineering import create_features
from src.visualization import run_all_visualizations
from src.modeling import train_model, save_model, FEATURES
from src.forecasting import arima_forecast
from src.explainability import shap_analysis
from src.utils import create_output_folder, save_metrics


def main():
    print("🚀 Starting Netflix Stock Analysis Pipeline...\n")

    create_output_folder()

    # ── Load & preprocess ─────────────────────────────────────────────────────
    print("📥 Loading data...")
    df = load_data()

    print("🧹 Preprocessing data...")
    df = preprocess_data(df)

    # ── EDA ───────────────────────────────────────────────────────────────────
    print("📊 Running EDA...")
    basic_eda(df)

    # ── Feature engineering ───────────────────────────────────────────────────
    print("⚙️  Creating features...")
    df = create_features(df)
    print(f"   Features created: {len(FEATURES)} | Rows after dropna: {len(df):,}")

    # ── Model training ────────────────────────────────────────────────────────
    print("\n🤖 Training stacking ensemble (XGB + LGBM + RF + ET → Ridge)...")
    model, results, X_test, y_test, preds = train_model(df)

    print("\n📊 Model Results:")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    save_model(model)
    save_metrics(results)

    # ── Visualizations ────────────────────────────────────────────────────────
    print("\n📈 Generating visualizations...")
    run_all_visualizations(
        df,
        model=model,
        y_test=y_test,
        preds=preds,
        rmse=results['RMSE'],
        r2=results['R2'],
    )

    # ── SHAP ──────────────────────────────────────────────────────────────────
    print("🔍 Running SHAP analysis...")
    try:
        shap_analysis(model, X_test)
    except Exception as e:
        print(f"  SHAP skipped: {e}")

    # ── ARIMA forecast ────────────────────────────────────────────────────────
    print("\n🔮 Running ARIMA forecast...")
    forecast = arima_forecast(df['Close'])
    print("📅 Forecast (next 10 months):")
    print(forecast.head(10))

    print("\n✅ Pipeline completed successfully!")


if __name__ == "__main__":
    main()
