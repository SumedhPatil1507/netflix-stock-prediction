import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np
import pandas as pd

from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import TimeSeriesSplit

sns.set_style("whitegrid")
sns.set_palette("husl")

OUT = "outputs"


# ── helpers ───────────────────────────────────────────────────────────────────
def _save(name):
    plt.tight_layout()
    plt.savefig(f"{OUT}/{name}", dpi=120)
    plt.close()


# ── 1. Close price distribution ───────────────────────────────────────────────
def plot_histograms(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(df['Close'], bins=60, kde=True, ax=axes[0])
    axes[0].set_title("Close Price Distribution")

    ret = df['Close'].pct_change().dropna() * 100
    sns.histplot(ret, bins=100, kde=True, ax=axes[1], color='coral')
    axes[1].set_title("Daily Returns Distribution (%)")
    _save("hist_close_returns.png")


# ── 2. Yearly average bar ─────────────────────────────────────────────────────
def plot_barplots(df):
    yearly = df['Close'].resample('YE').mean()
    plt.figure(figsize=(14, 5))
    yearly.plot(kind='bar', color='steelblue')
    plt.title("Average Close Price per Year")
    plt.xticks(rotation=45)
    _save("bar_year.png")


# ── 3. Box plot by year ───────────────────────────────────────────────────────
def plot_boxplots(df):
    d = df.copy()
    d['Year'] = d.index.year
    plt.figure(figsize=(16, 6))
    sns.boxplot(x='Year', y='Close', data=d)
    plt.xticks(rotation=60)
    plt.title("Close Price Distribution by Year")
    _save("boxplot.png")


# ── 4. Violin plot ────────────────────────────────────────────────────────────
def plot_violin(df):
    d = df.copy()
    d['Return'] = d['Close'].pct_change() * 100
    d['Year']   = d.index.year
    plt.figure(figsize=(16, 6))
    sns.violinplot(x='Year', y='Return', data=d)
    plt.xticks(rotation=60)
    plt.title("Return Distribution by Year")
    _save("violin.png")


# ── 5. Pair plot ──────────────────────────────────────────────────────────────
def plot_pairplot(df):
    cols = ['Close', 'Volume', 'Return', 'Volatility', 'RSI']
    available = [c for c in cols if c in df.columns]
    sns.pairplot(df[available].dropna(), diag_kind='kde', plot_kws={'alpha': 0.3})
    _save("pairplot.png")


# ── 6. Scatter: Volume vs Close ───────────────────────────────────────────────
def plot_scatter(df):
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=df['Volume'], y=df['Close'], alpha=0.3)
    plt.title("Volume vs Close Price")
    _save("scatter_volume_close.png")


# ── 7. Correlation heatmap ────────────────────────────────────────────────────
def plot_heatmap(df):
    cols = ['Open', 'High', 'Low', 'Close', 'Volume',
            'Return', 'RSI', 'MACD', 'ATR', 'Volatility', 'BB_Width']
    available = [c for c in cols if c in df.columns]
    corr = df[available].dropna().corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
    plt.title("Feature Correlation Heatmap")
    _save("heatmap.png")


# ── 8. Moving averages ────────────────────────────────────────────────────────
def plot_moving_averages(df):
    plt.figure(figsize=(14, 6))
    plt.plot(df.index, df['Close'], label='Close', alpha=0.6, linewidth=1)
    for ma, color in [('MA7', 'orange'), ('MA21', 'green'), ('MA50', 'red'), ('MA200', 'purple')]:
        if ma in df.columns:
            plt.plot(df.index, df[ma], label=ma, color=color, linewidth=1.2)
    plt.legend()
    plt.title("Close Price with Moving Averages")
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    _save("moving_avg.png")


# ── 9. Bollinger Bands ────────────────────────────────────────────────────────
def plot_bollinger_bands(df):
    recent = df.tail(252)  # last ~1 year
    plt.figure(figsize=(14, 6))
    plt.plot(recent.index, recent['Close'], label='Close', linewidth=1.5)
    if 'BB_Upper' in df.columns:
        plt.plot(recent.index, recent['BB_Upper'], '--', label='Upper Band', color='red', alpha=0.7)
        plt.plot(recent.index, recent['BB_Lower'], '--', label='Lower Band', color='green', alpha=0.7)
        plt.fill_between(recent.index, recent['BB_Lower'], recent['BB_Upper'], alpha=0.1, color='blue')
    plt.legend()
    plt.title("Bollinger Bands (Last 1 Year)")
    _save("bollinger_bands.png")


# ── 10. RSI ───────────────────────────────────────────────────────────────────
def plot_rsi(df):
    if 'RSI' not in df.columns:
        return
    recent = df.tail(252)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    ax1.plot(recent.index, recent['Close'])
    ax1.set_title("Close Price")
    ax2.plot(recent.index, recent['RSI'], color='purple')
    ax2.axhline(70, color='red', linestyle='--', alpha=0.7, label='Overbought (70)')
    ax2.axhline(30, color='green', linestyle='--', alpha=0.7, label='Oversold (30)')
    ax2.set_title("RSI (14)")
    ax2.legend()
    _save("rsi.png")


# ── 11. MACD ──────────────────────────────────────────────────────────────────
def plot_macd(df):
    if 'MACD' not in df.columns:
        return
    recent = df.tail(252)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    ax1.plot(recent.index, recent['Close'])
    ax1.set_title("Close Price")
    ax2.plot(recent.index, recent['MACD'], label='MACD', color='blue')
    ax2.plot(recent.index, recent['MACD_Signal'], label='Signal', color='orange')
    colors = ['green' if v >= 0 else 'red' for v in recent['MACD_Hist']]
    ax2.bar(recent.index, recent['MACD_Hist'], color=colors, alpha=0.5, label='Histogram')
    ax2.legend()
    ax2.set_title("MACD")
    _save("macd.png")


# ── 12. Time series ───────────────────────────────────────────────────────────
def plot_time_series(df):
    plt.figure(figsize=(14, 6))
    plt.plot(df.index, df['Close'], linewidth=1)
    plt.title("Netflix Close Price — Full History")
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    _save("time_series.png")


# ── 13. 5-year ARIMA forecast ─────────────────────────────────────────────────
def plot_forecast(df):
    series = df['Close'].resample('ME').mean().dropna()
    model  = ARIMA(series, order=(5, 1, 0))
    fitted = model.fit()
    fc     = fitted.get_forecast(60)
    mean   = fc.predicted_mean
    ci     = fc.conf_int()

    plt.figure(figsize=(14, 6))
    plt.plot(series, label='Historical')
    plt.plot(mean, label='Forecast', linestyle='--', color='orange')
    plt.fill_between(ci.index, ci.iloc[:, 0], ci.iloc[:, 1], alpha=0.2, color='orange', label='95% CI')
    plt.legend()
    plt.title("5-Year ARIMA Forecast with Confidence Interval")
    _save("forecast.png")


# ── 14. Actual vs Predicted ───────────────────────────────────────────────────
def plot_predictions(y_test, preds, rmse, r2):
    plt.figure(figsize=(14, 6))
    plt.plot(np.array(y_test), label='Actual', linewidth=1.5)
    plt.plot(preds, label='Predicted', linestyle='--', linewidth=1.5, alpha=0.8)
    plt.legend()
    plt.title(f"Actual vs Predicted  |  RMSE={rmse:.2f}  R²={r2:.4f}")
    _save("model_pred.png")


# ── 15. Residuals ─────────────────────────────────────────────────────────────
def plot_residuals(y_test, preds):
    residuals = np.array(y_test) - preds
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(residuals, alpha=0.7)
    axes[0].axhline(0, color='red', linestyle='--')
    axes[0].set_title("Residuals over Time")
    sns.histplot(residuals, bins=60, kde=True, ax=axes[1])
    axes[1].set_title("Residual Distribution")
    _save("residuals.png")


# ── 16. Feature importance ────────────────────────────────────────────────────
def plot_feature_importance(model, feature_names, top_n=20):
    """Works with ManualStackingRegressor — averages importances from tree base models."""
    try:
        importances = model.feature_importances_
        if importances is None:
            return
        importances = importances[:len(feature_names)]
        idx = np.argsort(importances)[-top_n:]
        plt.figure(figsize=(10, 8))
        sns.barplot(x=importances[idx], y=np.array(feature_names)[idx], orient='h')
        plt.title(f"Top {top_n} Feature Importances (avg of tree models)")
        _save("feature_importance.png")
    except Exception:
        pass


# ── 18. Volatility regime ─────────────────────────────────────────────────────
def plot_volatility(df):
    if 'Volatility' not in df.columns:
        return
    plt.figure(figsize=(14, 5))
    plt.plot(df.index, df['Volatility'], color='darkorange', linewidth=1)
    plt.title("20-Day Rolling Volatility")
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    _save("volatility.png")


# ── master ────────────────────────────────────────────────────────────────────
def run_all_visualizations(df, model=None, y_test=None, preds=None,
                            rmse=None, r2=None):
    plot_histograms(df)
    plot_barplots(df)
    plot_boxplots(df)
    plot_violin(df)
    plot_pairplot(df)
    plot_scatter(df)
    plot_heatmap(df)
    plot_moving_averages(df)
    plot_bollinger_bands(df)
    plot_rsi(df)
    plot_macd(df)
    plot_time_series(df)
    plot_forecast(df)
    plot_volatility(df)

    if y_test is not None and preds is not None:
        plot_predictions(y_test, preds, rmse, r2)
        plot_residuals(y_test, preds)

    if model is not None:
        from src.modeling import FEATURES
        plot_feature_importance(model, FEATURES)
