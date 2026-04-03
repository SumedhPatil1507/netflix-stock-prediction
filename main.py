from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.eda import basic_eda
from src.feature_engineering import create_features
from src.visualization import run_all_visualizations
from src.modeling import train_model, save_model, FEATURES
from src.forecasting import arima_forecast
from src.explainability import shap_analysis
from src.utils import create_output_folder, save_metrics
from sklearn.model_selection import TimeSeriesSplit
import numpy as np


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

    # ── Model training ────────────────────────────────────────────────────────
    print("🤖 Training model (walk-forward CV + ensemble)...")
    model, results, X_test, y_test, preds = train_model(df)

    print("\n📊 Model Results:")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")

    # ── Save model & metrics ──────────────────────────────────────────────────
    save_model(model)
    save_metrics(results)

    # ── Collect CV fold scores for plotting ───────────────────────────────────
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_squared_error, r2_score
    X_all = df[FEATURES].iloc[:-1]
    y_all = df['Close'].shift(-1).loc[X_all.index].iloc[:-1].dropna()
    X_all = X_all.loc[y_all.index]

    tscv = TimeSeriesSplit(n_splits=5)
    cv_rmse_list, cv_r2_list = [], []
    for tr, val in tscv.split(X_all):
        from src.modeling import _build_pipeline
        p = _build_pipeline()
        p.fit(X_all.iloc[tr], y_all.iloc[tr])
        pr = p.predict(X_all.iloc[val])
        cv_rmse_list.append(float(np.sqrt(mean_squared_error(y_all.iloc[val], pr))))
        cv_r2_list.append(float(r2_score(y_all.iloc[val], pr)))

    # ── Visualizations ────────────────────────────────────────────────────────
    print("📈 Generating visualizations...")
    run_all_visualizations(
        df,
        model=model,
        y_test=y_test,
        preds=preds,
        rmse=results['RMSE'],
        r2=results['R2'],
        cv_rmse_list=cv_rmse_list,
        cv_r2_list=cv_r2_list,
    )

    # ── SHAP explainability ───────────────────────────────────────────────────
    print("🔍 Running SHAP analysis...")
    try:
        shap_analysis(model, X_test)
    except Exception as e:
        print(f"  SHAP skipped: {e}")

    # ── ARIMA forecast ────────────────────────────────────────────────────────
    print("\n🔮 Running ARIMA forecast...")
    forecast = arima_forecast(df['Close'])
    print("📅 Forecast (next values):")
    print(forecast.head(10))

    print("\n✅ Pipeline completed successfully!")


if __name__ == "__main__":
    main()
