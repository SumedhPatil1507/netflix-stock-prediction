import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from statsmodels.tsa.arima.model import ARIMA

sns.set_style("whitegrid")

# --------------------------------------------------
# 1. HISTOGRAMS
# --------------------------------------------------
def plot_histograms(df):
    plt.figure()
    sns.histplot(df['Close'], bins=60, kde=True)
    plt.title("Close Price Distribution")
    plt.savefig("outputs/hist_close.png")
    plt.close()

    df['Return'] = df['Close'].pct_change() * 100
    plt.figure()
    sns.histplot(df['Return'].dropna(), bins=100, kde=True)
    plt.title("Returns Distribution")
    plt.savefig("outputs/hist_returns.png")
    plt.close()


# --------------------------------------------------
# 2. BARPLOTS
# --------------------------------------------------
def plot_barplots(df):
    df['Year'] = df.index.year
    yearly = df.groupby('Year')['Close'].mean()

    plt.figure()
    yearly.plot(kind='bar')
    plt.title("Avg Close Price per Year")
    plt.savefig("outputs/bar_year.png")
    plt.close()


# --------------------------------------------------
# 3. BOXPLOTS
# --------------------------------------------------
def plot_boxplots(df):
    df['Year'] = df.index.year

    plt.figure(figsize=(12,6))
    sns.boxplot(x='Year', y='Close', data=df)
    plt.xticks(rotation=60)
    plt.title("Boxplot Close by Year")
    plt.savefig("outputs/boxplot.png")
    plt.close()


# --------------------------------------------------
# 4. VIOLIN PLOTS
# --------------------------------------------------
def plot_violin(df):
    df['Return'] = df['Close'].pct_change() * 100
    df['Year'] = df.index.year

    plt.figure(figsize=(12,6))
    sns.violinplot(x='Year', y='Return', data=df)
    plt.xticks(rotation=60)
    plt.title("Violin Plot Returns")
    plt.savefig("outputs/violin.png")
    plt.close()


# --------------------------------------------------
# 5. PAIRPLOTS
# --------------------------------------------------
def plot_pairplot(df):
    df['Return'] = df['Close'].pct_change()
    cols = ['Close','Volume','Return']

    sns.pairplot(df[cols].dropna())
    plt.savefig("outputs/pairplot.png")
    plt.close()


# --------------------------------------------------
# 6. SCATTERPLOTS
# --------------------------------------------------
def plot_scatter(df):
    df['Return'] = df['Close'].pct_change()

    plt.figure()
    sns.scatterplot(x=df['Volume'], y=df['Close'])
    plt.title("Volume vs Close")
    plt.savefig("outputs/scatter_volume_close.png")
    plt.close()


# --------------------------------------------------
# 7. CORRELATION HEATMAP
# --------------------------------------------------
def plot_heatmap(df):
    df['Return'] = df['Close'].pct_change()
    corr = df[['Open','High','Low','Close','Volume','Return']].dropna().corr()

    plt.figure(figsize=(10,8))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title("Correlation Heatmap")
    plt.savefig("outputs/heatmap.png")
    plt.close()


# --------------------------------------------------
# 8. FEATURE ENGINEERING VISUALS
# --------------------------------------------------
def plot_feature_engineering(df):
    df['MA50'] = df['Close'].rolling(50).mean()
    df['MA200'] = df['Close'].rolling(200).mean()

    plt.figure(figsize=(12,6))
    plt.plot(df['Close'], label='Close')
    plt.plot(df['MA50'], label='MA50')
    plt.plot(df['MA200'], label='MA200')
    plt.legend()
    plt.title("Moving Averages")
    plt.savefig("outputs/moving_avg.png")
    plt.close()


# --------------------------------------------------
# 9. TIME SERIES
# --------------------------------------------------
def plot_time_series(df):
    plt.figure(figsize=(12,6))
    plt.plot(df.index, df['Close'])
    plt.title("Time Series Close Price")
    plt.savefig("outputs/time_series.png")
    plt.close()


# --------------------------------------------------
# 10. 5-YEAR FORECAST
# --------------------------------------------------
def plot_forecast(df):
    series = df['Close'].resample('ME').mean()

    model = ARIMA(series, order=(5,1,0))
    fitted = model.fit()

    forecast = fitted.forecast(60)

    plt.figure(figsize=(12,6))
    plt.plot(series, label='Historical')
    plt.plot(forecast, label='Forecast', linestyle='--')
    plt.legend()
    plt.title("5-Year Forecast")
    plt.savefig("outputs/forecast.png")
    plt.close()


# --------------------------------------------------
# 11. ML MODEL COMPARISON
# --------------------------------------------------
def plot_model_comparison(df):
    df = df.dropna()

    df['Target'] = df['Close'].shift(-1)
    df = df.dropna()

    X = df[['Open','High','Low','Close','Volume']]
    y = df['Target']

    split = int(len(df)*0.8)

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    plt.figure()
    plt.plot(y_test.values, label="Actual")
    plt.plot(preds, label="Predicted")
    plt.legend()
    plt.title(f"Model Prediction (RMSE={rmse:.2f})")
    plt.savefig("outputs/model_pred.png")
    plt.close()

    return model, X_test


# --------------------------------------------------
# 12. FEATURE IMPORTANCE
# --------------------------------------------------
def plot_feature_importance(model, X):
    importances = model.feature_importances_

    plt.figure()
    sns.barplot(x=importances, y=X.columns)
    plt.title("Feature Importance")
    plt.savefig("outputs/feature_importance.png")
    plt.close()


# --------------------------------------------------
# MASTER FUNCTION
# --------------------------------------------------
def run_all_visualizations(df):
    plot_histograms(df)
    plot_barplots(df)
    plot_boxplots(df)
    plot_violin(df)
    plot_pairplot(df)
    plot_scatter(df)
    plot_heatmap(df)
    plot_feature_engineering(df)
    plot_time_series(df)
    plot_forecast(df)

    model, X_test = plot_model_comparison(df)
    plot_feature_importance(model, X_test)