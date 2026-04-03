import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.modeling import FEATURES
from src.feature_engineering import create_features
from src.preprocessing import preprocess_data
from src.data_loader import load_data

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Netflix Stock Predictor",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Netflix Stock Prediction App")
st.caption("Ensemble model (GradientBoosting + RandomForest) with 29 engineered features")

# ── Load model ────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "model.pkl")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def get_data():
    df = load_data()
    df = preprocess_data(df)
    df = create_features(df)
    return df

try:
    model = load_model()
except Exception as e:
    st.error(f"Model not found. Run `python main.py` first to train and save the model.\n\n{e}")
    st.stop()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_pred, tab_eda, tab_charts, tab_about = st.tabs(
    ["🔮 Predict", "📊 EDA", "📉 Charts", "ℹ️ About"]
)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ═══════════════════════════════════════════════════════════════════════════════
with tab_pred:
    st.subheader("Enter the last 10 trading days of OHLCV data")
    st.info("Fill in the table below (most recent row = last row). The app computes all 29 features automatically.")

    default_data = {
        "Open":   [600, 605, 610, 615, 620, 625, 630, 635, 640, 645],
        "High":   [610, 615, 620, 625, 630, 635, 640, 645, 650, 655],
        "Low":    [595, 600, 605, 610, 615, 620, 625, 630, 635, 640],
        "Close":  [605, 610, 615, 620, 625, 630, 635, 640, 645, 650],
        "Volume": [5_000_000] * 10,
    }

    edited = st.data_editor(
        pd.DataFrame(default_data),
        num_rows="fixed",
        use_container_width=True,
        key="ohlcv_input",
    )

    if st.button("🔮 Predict Next Close Price", type="primary"):
        try:
            df_in = edited.copy().astype(float)

            # ── Compute features ──────────────────────────────────────────────
            df_in['Return']    = df_in['Close'].pct_change() * 100
            df_in['LogReturn'] = np.log(df_in['Close'] / df_in['Close'].shift(1))
            df_in['RangePct']  = (df_in['High'] - df_in['Low']) / df_in['Close'].shift(1) * 100

            for lag in [1, 2, 3, 5, 10]:
                df_in[f'Lag{lag}'] = df_in['Close'].shift(lag)

            for w in [7, 21]:
                df_in[f'MA{w}'] = df_in['Close'].rolling(w, min_periods=1).mean()

            df_in['RollingMean_5']  = df_in['Close'].rolling(5, min_periods=1).mean()
            df_in['RollingMean_10'] = df_in['Close'].rolling(10, min_periods=1).mean()
            df_in['RollingStd_5']   = df_in['Close'].rolling(5, min_periods=1).std().fillna(0)
            df_in['RollingStd_10']  = df_in['Close'].rolling(10, min_periods=1).std().fillna(0)
            df_in['Volatility']     = df_in['Return'].rolling(10, min_periods=1).std().fillna(0)

            df_in['Volume_MA10']  = df_in['Volume'].rolling(10, min_periods=1).mean()
            df_in['Volume_Ratio'] = df_in['Volume'] / df_in['Volume_MA10'].replace(0, np.nan)

            delta = df_in['Close'].diff()
            gain  = delta.clip(lower=0).rolling(14, min_periods=1).mean()
            loss  = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean()
            rs    = gain / loss.replace(0, np.nan)
            df_in['RSI'] = (100 - (100 / (1 + rs))).fillna(50)

            ema12 = df_in['Close'].ewm(span=12, adjust=False).mean()
            ema26 = df_in['Close'].ewm(span=26, adjust=False).mean()
            df_in['MACD']        = ema12 - ema26
            df_in['MACD_Signal'] = df_in['MACD'].ewm(span=9, adjust=False).mean()
            df_in['MACD_Hist']   = df_in['MACD'] - df_in['MACD_Signal']

            bb_mid         = df_in['Close'].rolling(20, min_periods=1).mean()
            bb_std         = df_in['Close'].rolling(20, min_periods=1).std().fillna(0)
            df_in['BB_Upper'] = bb_mid + 2 * bb_std
            df_in['BB_Lower'] = bb_mid - 2 * bb_std
            denom             = (df_in['BB_Upper'] - df_in['BB_Lower']).replace(0, np.nan)
            df_in['BB_Width'] = denom / bb_mid
            df_in['BB_Pct']   = (df_in['Close'] - df_in['BB_Lower']) / denom

            hl  = df_in['High'] - df_in['Low']
            hc  = (df_in['High'] - df_in['Close'].shift(1)).abs()
            lc  = (df_in['Low']  - df_in['Close'].shift(1)).abs()
            tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
            df_in['ATR'] = tr.rolling(14, min_periods=1).mean()

            df_in['Range']          = df_in['High'] - df_in['Low']
            df_in['Price_vs_MA50']  = df_in['Close'] / df_in['Close'].rolling(50, min_periods=1).mean() - 1
            df_in['Price_vs_MA200'] = df_in['Close'] / df_in['Close'].rolling(200, min_periods=1).mean() - 1
            df_in['DayOfWeek']      = 2
            df_in['Month']          = pd.Timestamp.now().month

            df_in = df_in.fillna(method='bfill').fillna(0)

            row  = df_in[FEATURES].iloc[[-1]]
            pred = model.predict(row)[0]
            last = df_in['Close'].iloc[-1]
            ret  = (pred - last) / last * 100

            # ── Results ───────────────────────────────────────────────────────
            c1, c2, c3 = st.columns(3)
            c1.metric("Last Close",            f"${last:.2f}")
            c2.metric("Predicted Next Close",  f"${pred:.2f}")
            c3.metric("Implied Return",         f"{ret:+.2f}%",
                      delta=f"{ret:+.2f}%", delta_color="normal")

            # ── Mini chart ────────────────────────────────────────────────────
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(range(len(df_in)), df_in['Close'], marker='o', label='Input Close')
            ax.axhline(pred, color='orange', linestyle='--', label=f'Predicted: ${pred:.2f}')
            ax.set_title("Input Window + Prediction")
            ax.legend()
            st.pyplot(fig)
            plt.close()

        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.exception(e)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — EDA
# ═══════════════════════════════════════════════════════════════════════════════
with tab_eda:
    st.subheader("Exploratory Data Analysis")

    try:
        df = get_data()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Rows",  f"{len(df):,}")
        c2.metric("Date Range",  f"{df.index.min().year} – {df.index.max().year}")
        c3.metric("Max Close",   f"${df['Close'].max():.2f}")
        c4.metric("Avg Volume",  f"{df['Volume'].mean():,.0f}")

        st.markdown("#### Statistical Summary")
        st.dataframe(df[['Open','High','Low','Close','Volume','Return','RSI','Volatility']].describe().round(4))

        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(df.index, df['Close'], linewidth=0.8)
            ax.set_title("Close Price — Full History")
            st.pyplot(fig)
            plt.close()

        with col2:
            fig, ax = plt.subplots(figsize=(6, 4))
            ret = df['Close'].pct_change().dropna() * 100
            ax.hist(ret, bins=100, color='coral', edgecolor='none')
            ax.set_title("Daily Returns Distribution (%)")
            st.pyplot(fig)
            plt.close()

        col3, col4 = st.columns(2)

        with col3:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(df.index, df['Volatility'], color='darkorange', linewidth=0.8)
            ax.set_title("20-Day Rolling Volatility")
            st.pyplot(fig)
            plt.close()

        with col4:
            import seaborn as sns
            cols = ['Close','Volume','Return','RSI','MACD','Volatility','ATR']
            corr = df[cols].dropna().corr()
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax, linewidths=0.5)
            ax.set_title("Correlation Heatmap")
            st.pyplot(fig)
            plt.close()

    except Exception as e:
        st.warning(f"Could not load data for EDA: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — CHARTS (saved outputs)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_charts:
    st.subheader("Generated Analysis Charts")
    st.caption("Run `python main.py` to regenerate all charts.")

    OUTPUTS = os.path.join(os.path.dirname(__file__), "..", "outputs")

    chart_map = {
        "Moving Averages":       "moving_avg.png",
        "Bollinger Bands":       "bollinger_bands.png",
        "RSI":                   "rsi.png",
        "MACD":                  "macd.png",
        "Actual vs Predicted":   "model_pred.png",
        "Residuals":             "residuals.png",
        "Feature Importance":    "feature_importance.png",
        "SHAP Summary":          "shap_summary.png",
        "SHAP Bar":              "shap_bar.png",
        "5-Year Forecast":       "forecast.png",
        "CV Fold Scores":        "cv_scores.png",
        "Volatility":            "volatility.png",
        "Heatmap":               "heatmap.png",
        "Histogram":             "hist_close_returns.png",
    }

    keys = list(chart_map.keys())
    for i in range(0, len(keys), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i + j < len(keys):
                name = keys[i + j]
                path = os.path.join(OUTPUTS, chart_map[name])
                with col:
                    if os.path.exists(path):
                        st.image(path, caption=name, use_column_width=True)
                    else:
                        st.info(f"{name} — not generated yet")

    # Metrics JSON
    metrics_path = os.path.join(OUTPUTS, "metrics.json")
    if os.path.exists(metrics_path):
        import json
        with open(metrics_path) as f:
            metrics = json.load(f)
        st.markdown("#### Model Metrics")
        mc = st.columns(len(metrics))
        for col, (k, v) in zip(mc, metrics.items()):
            col.metric(k, f"{v:.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ABOUT
# ═══════════════════════════════════════════════════════════════════════════════
with tab_about:
    st.markdown("""
### About This App

This app predicts the **next-day closing price** of Netflix stock using an ensemble ML model.

**Model:** VotingRegressor (GradientBoosting + RandomForest) with RobustScaler  
**Validation:** 5-fold walk-forward cross-validation  
**Features (29 total):**
- Lag prices (Lag1–Lag10)
- Rolling mean/std (5, 10 windows)
- Returns, log returns, volatility
- RSI (14), MACD + signal + histogram
- Bollinger Bands (width, %B)
- ATR (14), price range
- Price vs MA50/MA200
- Volume ratio
- Calendar features (day of week, month)

**Data:** Netflix historical OHLCV from 2002 to present  
**Source code:** [GitHub](https://github.com)
    """)
