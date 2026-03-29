from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.eda import basic_eda
from src.feature_engineering import create_features
from src.visualization import run_all_visualizations
from src.modeling import train_model, save_model
from src.forecasting import arima_forecast
from src.utils import create_output_folder, save_metrics

def main():
    print("🚀 Starting Netflix Stock Analysis Pipeline...\n")

    # --------------------------------------------------
    # Create output folders
    # --------------------------------------------------
    create_output_folder()

    # --------------------------------------------------
    # Load Data
    # --------------------------------------------------
    print("📥 Loading data...")
    df = load_data()

    # --------------------------------------------------
    # Preprocess Data
    # --------------------------------------------------
    print("🧹 Preprocessing data...")
    df = preprocess_data(df)

    # --------------------------------------------------
    # EDA
    # --------------------------------------------------
    print("📊 Running EDA...")
    basic_eda(df)

    # --------------------------------------------------
    # Feature Engineering
    # --------------------------------------------------
    print("⚙️ Creating features...")
    df = create_features(df)

    # --------------------------------------------------
    # Visualization (ALL plots)
    # --------------------------------------------------
    print("📈 Generating visualizations...")
    run_all_visualizations(df)

    # --------------------------------------------------
    # Model Training
    # --------------------------------------------------
    print("🤖 Training model...")
    model, results, X_test = train_model(df)

    print("\n📊 Model Results:")
    for key, value in results.items():
        print(f"{key}: {value:.4f}")

    # --------------------------------------------------
    # Save Model
    # --------------------------------------------------
    save_model(model)

    # --------------------------------------------------
    # Save Metrics
    # --------------------------------------------------
    save_metrics(results)

    # --------------------------------------------------
    # Forecast
    # --------------------------------------------------
    print("\n🔮 Running forecast...")
    forecast = arima_forecast(df['Close'])

    print("\n📅 Forecast (next values):")
    print(forecast.head())

    print("\n✅ Pipeline completed successfully!")


# --------------------------------------------------
# ENTRY POINT
# --------------------------------------------------
if __name__ == "__main__":
    main()