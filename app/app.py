import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys

# Resolve repo root regardless of where Streamlit launches from
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(REPO_ROOT)
sys.path.insert(0, REPO_ROOT)

from src.modeling import get_active_features

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Netflix Stock Predictor",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Netflix Stock Prediction App")
st.caption("Stacking ensemble (XGB + LGBM + RF + ET → Ridge) · 51 features · Walk-forward CV · Conformal Intervals")

# ── Load model ────────────────────────────────────────────────────────────────
MODEL_PATH  = os.path.join(REPO_ROOT, "models", "model.pkl")
CACHE_PATH  = os.path.join(REPO_ROOT, "outputs", "features_cache.parquet")

@st.cache_resource(show_spinner="Loading model...")
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data(show_spinner="Loading data...")
def get_data(source="live"):
    # Try live first, fall back to CSV parquet cache, then raw CSV
    if source == "live":
        try:
            from src.data_loader import load_data
            from src.preprocessing import preprocess_data
            from src.feature_engineering import create_features
            df = load_data(source="live")
            df = preprocess_data(df)
            df = create_features(df)
            return df
        except Exception:
            pass  # fall through to cache
    # Fast path: parquet cache
    if os.path.exists(CACHE_PATH):
        return pd.read_parquet(CACHE_PATH)
    # Slow path: raw CSV
    from src.data_loader import load_data
    from src.preprocessing import preprocess_data
    from src.feature_engineering import create_features
    df = load_data(source="csv")
    df = preprocess_data(df)
    df = create_features(df)
    return df

try:
    model = load_model()
except Exception as e:
    st.error(f"Model not found. Run `python main.py` first.\n\n{e}")
    st.stop()

FEATURES = get_active_features(get_data())
# Always use the model's own feature list if available (prevents version mismatch)
if hasattr(model, 'feature_names_'):
    FEATURES = model.feature_names_

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    data_source = st.radio(
        "Data source",
        ["live (yfinance)", "csv"],
        index=0,
        help="'live' fetches latest NFLX data from Yahoo Finance (recommended)"
    )
    source_key = "live" if "live" in data_source else "csv"
    if source_key == "live":
        st.success("Using live NFLX data")
    else:
        st.info("Using static CSV data")
    st.markdown("---")
    st.markdown("**Model:** XGB + LGBM + RF + ET → Ridge")
    st.markdown("**Features:** 51 technical indicators")
    st.markdown("**Target:** Next-day return (%)")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_pred, tab_eda, tab_charts, tab_backtest, tab_drift, tab_about = st.tabs(
    ["🔮 Predict", "📊 EDA", "📉 Charts", "📈 Backtest", "🔬 Drift Monitor", "ℹ️ About"]
)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ═══════════════════════════════════════════════════════════════════════════════
with tab_pred:
    st.subheader("Enter the last 10 trading days of OHLCV data")
    st.info("Table auto-filled with real NFLX data from Yahoo Finance. Edit any values or use your own.")

    # ── Default data: try live NFLX, fall back to placeholder ────────────────
    @st.cache_data(ttl=3600, show_spinner=False)
    def _get_default_ohlcv():
        try:
            import yfinance as yf
            df_live = yf.Ticker("NFLX").history(period="20d")
            df_live = df_live[['Open','High','Low','Close','Volume']].dropna().tail(10)
            df_live.columns = ['Open','High','Low','Close','Volume']
            df_live = df_live.round(2)
            return df_live.reset_index(drop=True).to_dict('list')
        except Exception:
            return {
                "Open":   [600, 605, 610, 615, 620, 625, 630, 635, 640, 645],
                "High":   [610, 615, 620, 625, 630, 635, 640, 645, 650, 655],
                "Low":    [595, 600, 605, 610, 615, 620, 625, 630, 635, 640],
                "Close":  [605, 610, 615, 620, 625, 630, 635, 640, 645, 650],
                "Volume": [5_000_000] * 10,
            }

    default_data = _get_default_ohlcv()

    edited = st.data_editor(
        pd.DataFrame(default_data),
        num_rows="fixed",
        use_container_width=True,
        key="ohlcv_input",
    )

    if st.button("🔮 Predict Next Close Price", type="primary"):
        try:
            df_in = edited.copy().astype(float)

            # ── Compute all features (mirrors feature_engineering.py) ─────────
            df_in['Return']    = df_in['Close'].pct_change() * 100
            df_in['LogReturn'] = np.log(df_in['Close'] / df_in['Close'].shift(1))
            df_in['RangePct']  = (df_in['High'] - df_in['Low']) / df_in['Close'].shift(1) * 100

            for lag in [1, 2, 3, 5, 10, 20]:
                df_in[f'Lag{lag}'] = df_in['Close'].shift(lag)
            for lag in [1, 2, 3, 5]:
                df_in[f'RetLag{lag}'] = df_in['Return'].shift(lag)

            for w in [5, 7, 10, 21, 50, 200]:
                df_in[f'MA{w}'] = df_in['Close'].rolling(w, min_periods=1).mean()

            df_in['EMA9']      = df_in['Close'].ewm(span=9,  adjust=False).mean()
            df_in['EMA21']     = df_in['Close'].ewm(span=21, adjust=False).mean()
            df_in['EMA_Cross'] = df_in['EMA9'] - df_in['EMA21']

            df_in['RollingMean_5']  = df_in['Close'].rolling(5, min_periods=1).mean()
            df_in['RollingMean_10'] = df_in['Close'].rolling(10, min_periods=1).mean()
            df_in['RollingStd_5']   = df_in['Close'].rolling(5, min_periods=1).std().fillna(0)
            df_in['RollingStd_10']  = df_in['Close'].rolling(10, min_periods=1).std().fillna(0)
            df_in['Volatility']     = df_in['Return'].rolling(20, min_periods=1).std().fillna(0)
            df_in['Volatility_5']   = df_in['Return'].rolling(5,  min_periods=1).std().fillna(0)

            for col in ['MA5','MA10','MA21','MA50','MA200']:
                key = f'Price_vs_{col}'
                df_in[key] = df_in['Close'] / df_in[col].replace(0, np.nan) - 1

            df_in['Volume_MA10']    = df_in['Volume'].rolling(10, min_periods=1).mean()
            df_in['Volume_MA20']    = df_in['Volume'].rolling(20, min_periods=1).mean()
            df_in['Volume_Ratio']   = df_in['Volume'] / df_in['Volume_MA10'].replace(0, np.nan)
            df_in['Volume_Ratio20'] = df_in['Volume'] / df_in['Volume_MA20'].replace(0, np.nan)

            obv = (np.sign(df_in['Close'].diff()) * df_in['Volume']).fillna(0).cumsum()
            df_in['OBV']       = obv
            df_in['OBV_MA10']  = df_in['OBV'].rolling(10, min_periods=1).mean()
            df_in['OBV_Ratio'] = df_in['OBV'] / df_in['OBV_MA10'].replace(0, np.nan)

            for period in [7, 14]:
                delta = df_in['Close'].diff()
                gain  = delta.clip(lower=0).rolling(period, min_periods=1).mean()
                loss  = (-delta.clip(upper=0)).rolling(period, min_periods=1).mean()
                rs    = gain / loss.replace(0, np.nan)
                df_in[f'RSI{period}'] = (100 - (100 / (1 + rs))).fillna(50)
            df_in['RSI'] = df_in['RSI14']

            ema12 = df_in['Close'].ewm(span=12, adjust=False).mean()
            ema26 = df_in['Close'].ewm(span=26, adjust=False).mean()
            df_in['MACD']        = ema12 - ema26
            df_in['MACD_Signal'] = df_in['MACD'].ewm(span=9, adjust=False).mean()
            df_in['MACD_Hist']   = df_in['MACD'] - df_in['MACD_Signal']
            df_in['MACD_Norm']   = df_in['MACD'] / df_in['Close'].replace(0, np.nan)

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
            df_in['ATR']      = tr.rolling(14, min_periods=1).mean()
            df_in['ATR_Norm'] = df_in['ATR'] / df_in['Close'].replace(0, np.nan)

            low14  = df_in['Low'].rolling(14, min_periods=1).min()
            high14 = df_in['High'].rolling(14, min_periods=1).max()
            rng14  = (high14 - low14).replace(0, np.nan)
            df_in['Stoch_K']   = 100 * (df_in['Close'] - low14) / rng14
            df_in['Stoch_D']   = df_in['Stoch_K'].rolling(3, min_periods=1).mean()
            df_in['Williams_R']= -100 * (high14 - df_in['Close']) / rng14

            tp     = (df_in['High'] + df_in['Low'] + df_in['Close']) / 3
            tp_ma  = tp.rolling(20, min_periods=1).mean()
            tp_std = tp.rolling(20, min_periods=1).std().replace(0, np.nan)
            df_in['CCI'] = (tp - tp_ma) / (0.015 * tp_std)

            df_in['Range']      = df_in['High'] - df_in['Low']
            df_in['Range_Norm'] = df_in['Range'] / df_in['Close'].replace(0, np.nan)
            df_in['Momentum5']  = df_in['Close'] / df_in['Close'].shift(5).replace(0, np.nan)  - 1
            df_in['Momentum10'] = df_in['Close'] / df_in['Close'].shift(10).replace(0, np.nan) - 1
            df_in['Momentum20'] = df_in['Close'] / df_in['Close'].shift(20).replace(0, np.nan) - 1

            df_in['DayOfWeek'] = 2
            df_in['Month']     = pd.Timestamp.now().month
            df_in['Quarter']   = (pd.Timestamp.now().month - 1) // 3 + 1
            df_in['EarningsMonth'] = int(pd.Timestamp.now().month in [1, 4, 7, 10])
            df_in['VolRatio_5_20'] = df_in['Volatility_5'] / df_in['Volatility'].replace(0, np.nan)

            # Regime features — default to Sideways (1) if not available
            df_in['Regime']      = 1
            df_in['Regime_Bear'] = 0.0
            df_in['Regime_Side'] = 1.0
            df_in['Regime_Bull'] = 0.0

            df_in = df_in.ffill().fillna(0)

            # Use exact features the model was trained on (stored in model.feature_names_)
            # Fall back to filtering FEATURES against what's available in df_in
            if hasattr(model, 'feature_names_'):
                train_features = model.feature_names_
            else:
                train_features = FEATURES

            # Ensure all training features exist in df_in (fill missing with 0)
            for f in train_features:
                if f not in df_in.columns:
                    df_in[f] = 0.0

            row         = df_in[train_features].iloc[[-1]]   # DataFrame, keeps column names
            last        = df_in['Close'].iloc[-1]
            pred_return = model.predict(row)[0]
            pred        = last * (1 + pred_return / 100)
            ret         = pred_return

            # ── Results ───────────────────────────────────────────────────────
            c1, c2, c3 = st.columns(3)
            c1.metric("Last Close",            f"${last:.2f}")
            c2.metric("Predicted Next Close",  f"${pred:.2f}")
            c3.metric("Implied Return",         f"{ret:+.2f}%",
                      delta=f"{ret:+.2f}%", delta_color="normal")

            # ── Conformal interval ────────────────────────────────────────────
            if hasattr(model, 'conformal_'):
                cp = model.conformal_
                lo_r, hi_r = cp.predict_interval(row)
                lo_p = last * (1 + lo_r[0] / 100)
                hi_p = last * (1 + hi_r[0] / 100)
                st.info(f"90% Prediction Interval: **${lo_p:.2f}** to **${hi_p:.2f}**  "
                        f"(return: {lo_r[0]:+.2f}% to {hi_r[0]:+.2f}%)")

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
        df = get_data(source_key)

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
# TAB 3 — CHARTS (live, generated from data — no pre-generated PNGs needed)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_charts:
    st.subheader("Interactive Technical Analysis Charts")

    try:
        df = get_data(source_key)

        chart_choice = st.selectbox("Select chart", [
            "Price + Moving Averages",
            "Bollinger Bands (Last 1 Year)",
            "RSI (14)",
            "MACD",
            "Volume & OBV",
            "Volatility Regime",
            "Yearly Returns Heatmap",
            "Drawdown",
            "Return Distribution",
            "Correlation Heatmap",
        ])

        if chart_choice == "Price + Moving Averages":
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(df.index, df['Close'], label='Close', linewidth=1, alpha=0.8)
            for ma, color in [('MA7','orange'),('MA21','green'),('MA50','red'),('MA200','purple')]:
                if ma in df.columns:
                    ax.plot(df.index, df[ma], label=ma, linewidth=1, color=color)
            ax.set_title("Close Price with Moving Averages")
            ax.legend()
            st.pyplot(fig); plt.close()

        elif chart_choice == "Bollinger Bands (Last 1 Year)":
            recent = df.tail(252)
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(recent.index, recent['Close'], label='Close', linewidth=1.5)
            ax.plot(recent.index, recent['BB_Upper'], '--', color='red',   alpha=0.7, label='Upper Band')
            ax.plot(recent.index, recent['BB_Lower'], '--', color='green', alpha=0.7, label='Lower Band')
            ax.fill_between(recent.index, recent['BB_Lower'], recent['BB_Upper'], alpha=0.08, color='blue')
            ax.set_title("Bollinger Bands — Last 1 Year")
            ax.legend()
            st.pyplot(fig); plt.close()

        elif chart_choice == "RSI (14)":
            recent = df.tail(365)
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
            ax1.plot(recent.index, recent['Close'], linewidth=1)
            ax1.set_title("Close Price")
            ax2.plot(recent.index, recent['RSI'], color='purple', linewidth=1)
            ax2.axhline(70, color='red',   linestyle='--', alpha=0.7, label='Overbought (70)')
            ax2.axhline(30, color='green', linestyle='--', alpha=0.7, label='Oversold (30)')
            ax2.axhline(50, color='gray',  linestyle=':',  alpha=0.5)
            ax2.set_ylim(0, 100)
            ax2.set_title("RSI (14)")
            ax2.legend()
            st.pyplot(fig); plt.close()

        elif chart_choice == "MACD":
            recent = df.tail(365)
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
            ax1.plot(recent.index, recent['Close'], linewidth=1)
            ax1.set_title("Close Price")
            ax2.plot(recent.index, recent['MACD'],        label='MACD',   color='blue',   linewidth=1)
            ax2.plot(recent.index, recent['MACD_Signal'], label='Signal', color='orange', linewidth=1)
            colors = ['green' if v >= 0 else 'red' for v in recent['MACD_Hist']]
            ax2.bar(recent.index, recent['MACD_Hist'], color=colors, alpha=0.4, label='Histogram')
            ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)
            ax2.set_title("MACD")
            ax2.legend()
            st.pyplot(fig); plt.close()

        elif chart_choice == "Volume & OBV":
            recent = df.tail(365)
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
            ax1.bar(recent.index, recent['Volume'], color='steelblue', alpha=0.6, width=1)
            ax1.plot(recent.index, recent['Volume_MA10'], color='orange', linewidth=1.5, label='MA10')
            ax1.set_title("Volume")
            ax1.legend()
            ax2.plot(recent.index, recent['OBV'], color='teal', linewidth=1)
            ax2.set_title("On-Balance Volume (OBV)")
            st.pyplot(fig); plt.close()

        elif chart_choice == "Volatility Regime":
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
            ax1.plot(df.index, df['Close'], linewidth=0.8)
            ax1.set_title("Close Price")
            ax2.plot(df.index, df['Volatility'], color='darkorange', linewidth=0.8)
            ax2.fill_between(df.index, df['Volatility'], alpha=0.3, color='darkorange')
            ax2.set_title("20-Day Rolling Volatility (%)")
            st.pyplot(fig); plt.close()

        elif chart_choice == "Yearly Returns Heatmap":
            import seaborn as sns
            df_yr = df.copy()
            df_yr['Year']  = df_yr.index.year
            df_yr['Month'] = df_yr.index.month
            monthly_ret = df_yr.groupby(['Year','Month'])['Return'].mean().unstack()
            fig, ax = plt.subplots(figsize=(14, 8))
            sns.heatmap(monthly_ret, cmap='RdYlGn', center=0, annot=False,
                        linewidths=0.3, ax=ax, cbar_kws={'label': 'Avg Daily Return (%)'})
            ax.set_title("Monthly Average Daily Return Heatmap")
            ax.set_xlabel("Month"); ax.set_ylabel("Year")
            st.pyplot(fig); plt.close()

        elif chart_choice == "Drawdown":
            roll_max = df['Close'].cummax()
            drawdown = (df['Close'] - roll_max) / roll_max * 100
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
            ax1.plot(df.index, df['Close'], linewidth=0.8)
            ax1.set_title("Close Price")
            ax2.fill_between(df.index, drawdown, 0, color='red', alpha=0.4)
            ax2.set_title("Drawdown from All-Time High (%)")
            st.pyplot(fig); plt.close()

        elif chart_choice == "Return Distribution":
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            ret = df['Return'].dropna()
            axes[0].hist(ret, bins=120, color='steelblue', edgecolor='none', density=True)
            axes[0].set_title("Daily Return Distribution")
            axes[0].set_xlabel("Return (%)")
            import seaborn as sns
            sns.boxplot(y=ret, ax=axes[1], color='coral')
            axes[1].set_title("Return Boxplot")
            st.pyplot(fig); plt.close()

        elif chart_choice == "Correlation Heatmap":
            import seaborn as sns
            cols = ['Return','RSI','MACD_Norm','BB_Pct','ATR_Norm',
                    'Volatility','Stoch_K','Williams_R','CCI','Momentum5']
            available = [c for c in cols if c in df.columns]
            corr = df[available].dropna().corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
                        linewidths=0.5, ax=ax)
            ax.set_title("Feature Correlation Heatmap")
            st.pyplot(fig); plt.close()

        # ── Metrics from saved JSON ───────────────────────────────────────────
        metrics_path = os.path.join(REPO_ROOT, "outputs", "metrics.json")
        if os.path.exists(metrics_path):
            import json
            with open(metrics_path) as f:
                metrics = json.load(f)
            st.markdown("---")
            st.markdown("#### Model Metrics")
            mc = st.columns(len(metrics))
            for col, (k, v) in zip(mc, metrics.items()):
                col.metric(k, f"{v:.4f}")

    except Exception as e:
        st.warning(f"Could not load data for charts: {e}")
        st.exception(e)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — BACKTEST
# ═══════════════════════════════════════════════════════════════════════════════
with tab_backtest:
    st.subheader("Strategy Backtesting")
    st.caption("Simulates: go long when model predicts positive return, flat otherwise. Includes transaction costs.")

    bt_path = os.path.join(REPO_ROOT, "outputs", "backtest_curves.csv")
    metrics_path = os.path.join(REPO_ROOT, "outputs", "metrics.json")

    if os.path.exists(bt_path):
        curves = pd.read_csv(bt_path, index_col=0)
        rs_path = os.path.join(REPO_ROOT, "outputs", "rolling_sharpe.csv")

        # ── Equity curve ──────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(curves.index, curves["Strategy"],   label="Binary Strategy", linewidth=1.5)
        if "Kelly" in curves.columns:
            ax.plot(curves.index, curves["Kelly"], label="Kelly Strategy", linewidth=1.2, linestyle="-.")
        ax.plot(curves.index, curves["BuyAndHold"], label="Buy & Hold", linewidth=1.5, linestyle="--")
        ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5)
        ax.set_title("Strategy vs Buy & Hold — Equity Curve")
        ax.set_ylabel("Portfolio Value (start = 1.0)")
        ax.legend()
        st.pyplot(fig); plt.close()

        # ── Rolling Sharpe ────────────────────────────────────────────────────
        if os.path.exists(rs_path):
            rs = pd.read_csv(rs_path).squeeze()
            fig, ax = plt.subplots(figsize=(12, 3))
            ax.plot(rs.values, color="purple", linewidth=1)
            ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
            ax.axhline(1, color="green", linestyle=":", alpha=0.5, label="Sharpe=1")
            ax.set_title("Rolling 63-Day Sharpe Ratio")
            ax.legend()
            st.pyplot(fig); plt.close()

        # ── Metrics ───────────────────────────────────────────────────────────
        if os.path.exists(metrics_path):
            import json
            with open(metrics_path) as f:
                ml_metrics = json.load(f)
            st.markdown("#### ML Model Metrics")
            mc = st.columns(min(len(ml_metrics), 5))
            for col, (k, v) in zip(mc, list(ml_metrics.items())[:5]):
                col.metric(k, f"{v:.4f}")

    else:
        st.info("No backtest data found. Run `python main.py` to generate it.")
        st.markdown("""
        **What the backtest does:**
        - Uses model predictions on the held-out test set (last 20% of data)
        - Goes **long** when predicted return > 0%, stays **flat** otherwise
        - Applies 0.1% transaction cost per trade
        - Compares against simple buy-and-hold benchmark
        - Reports: Total Return, Annualised Return, Sharpe Ratio, Sortino Ratio, Max Drawdown
        """)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — DRIFT MONITOR
# ═══════════════════════════════════════════════════════════════════════════════
with tab_drift:
    st.subheader("Model Drift Monitor")
    st.caption("Compares recent data distribution against training data using PSI and KS test.")

    try:
        df_full = get_data(source_key)
        split   = int(len(df_full) * 0.8)
        df_train = df_full.iloc[:split]
        df_recent = df_full.iloc[split:]

        from src.drift import detect_drift, drift_summary_df
        drift_result = detect_drift(df_train, df_recent, FEATURES)
        drift_df     = drift_summary_df(drift_result)

        # ── Summary banner ────────────────────────────────────────────────────
        n_drifted = len(drift_result["drifted_features"])
        if drift_result["overall_drift"]:
            st.error(f"Significant drift detected in {n_drifted} features — consider retraining.")
        elif n_drifted > 0:
            st.warning(f"Moderate drift in {n_drifted} features — monitor closely.")
        else:
            st.success("No significant drift detected. Model is stable.")

        c1, c2, c3 = st.columns(3)
        c1.metric("Features Checked", len(drift_df))
        c2.metric("Drifted Features", n_drifted)
        c3.metric("PSI Threshold", drift_result["psi_threshold"])

        # ── PSI bar chart ─────────────────────────────────────────────────────
        top20 = drift_df.head(20)
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ["red" if d else "steelblue" for d in top20["Drifted"]]
        ax.barh(top20["Feature"], top20["PSI"], color=colors)
        ax.axvline(0.1, color="orange", linestyle="--", alpha=0.7, label="Moderate (0.1)")
        ax.axvline(0.2, color="red",    linestyle="--", alpha=0.7, label="Significant (0.2)")
        ax.set_title("Top 20 Features by PSI (Population Stability Index)")
        ax.legend()
        st.pyplot(fig); plt.close()

        # ── Full table ────────────────────────────────────────────────────────
        st.markdown("#### Full Drift Report")
        st.dataframe(
            drift_df.style.applymap(
                lambda v: "background-color: #ffcccc" if v is True else "",
                subset=["Drifted"]
            ),
            use_container_width=True,
        )

    except Exception as e:
        st.warning(f"Drift monitor unavailable: {e}")
        st.info("Install scipy: `pip install scipy`")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — ABOUT
# ═══════════════════════════════════════════════════════════════════════════════
with tab_about:
    st.markdown("""
### About This App

Predicts the **next-day return (%)** of Netflix stock using a manual stacking ensemble, then converts it back to a price.

**Model:** Manual Stacking — XGBoost + LightGBM + RandomForest + ExtraTrees → Ridge meta-learner  
**Validation:** 5-fold walk-forward (time-series) cross-validation  
**Target:** Next-day return (%) — stationary, no price-level bias  
**Features (51 total):** Lag prices/returns, RSI (7 & 14), MACD, Bollinger Bands, ATR, Stochastic %K/%D, Williams %R, CCI, OBV, EMA crossover, momentum ratios, price vs MAs, volatility, regime probabilities, earnings flag

**Conformal Prediction:** 90% calibrated prediction intervals — not just a point estimate  
**Drift Monitor:** PSI + KS test on all features to detect distribution shift  
**Sentiment:** VADER-scored news headlines via yfinance (local dev)  
**Backtesting:** Kelly criterion sizing, Sharpe/Sortino/Calmar, rolling Sharpe  
**Hyperparameter Tuning:** Optuna (run `python -m src.tuning` locally)

**Key metric:** Directional Accuracy (`Dir_Acc`) — above 52% is meaningful signal.

**Data:** Netflix historical OHLCV — May 2002 to present  
**API:** `uvicorn api.main:app --reload` → Swagger at `http://localhost:8000/docs`  
**Tests:** `pytest tests/ -v`  
**Source:** [GitHub — SumedhPatil1507/netflix-stock-prediction](https://github.com/SumedhPatil1507/netflix-stock-prediction)

---
> This is a research/learning project. Do not use for real trading decisions.
    """)
