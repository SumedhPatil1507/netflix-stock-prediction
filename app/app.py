import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os, sys, json
from pathlib import Path

try:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from src.branding_config import get_branding, TenantIsolation
    _brand = get_branding()
    APP_NAME = _brand.app_name
    PRIMARY_COLOR = _brand.primary_color
except Exception:
    APP_NAME = "Alpha Engine Pro"
    PRIMARY_COLOR = "#e50914"

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(REPO_ROOT)
sys.path.insert(0, REPO_ROOT)

from src.modeling import get_active_features
from src.feature_utils import build_prediction_row

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=APP_NAME,
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.metric-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid #e50914;
    border-radius: 10px;
    padding: 15px;
    text-align: center;
}
.hero-title {
    font-size: 2.5rem;
    font-weight: 800;
    background: linear-gradient(90deg, #e50914, #ff6b6b);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
</style>
""", unsafe_allow_html=True)

MODEL_PATH = os.path.join(REPO_ROOT, "models", "model.pkl")
CACHE_PATH = os.path.join(REPO_ROOT, "outputs", "features_cache.parquet")

# ── Data & model loaders ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data(ttl=7200, show_spinner="Fetching live data...")
def load_live_ohlcv(ticker_sym: str = "NFLX", period: str = "2y") -> pd.DataFrame:
    try:
        import yfinance as yf
        df = yf.Ticker(ticker_sym).history(period=period)
        if hasattr(df.index.dtype, "tz") and df.index.dtype.tz is not None:
            df.index = df.index.tz_localize(None)
        return df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    except Exception:
        return None

@st.cache_data(show_spinner="Computing features...")
def get_featured_data():
    if os.path.exists(CACHE_PATH):
        return pd.read_parquet(CACHE_PATH)
    from src.data_loader import load_data
    from src.preprocessing import preprocess_data
    from src.feature_engineering import create_features
    df = load_data(source="csv")
    df = preprocess_data(df)
    return create_features(df)

try:
    model = load_model()
except Exception as e:
    st.error(f"Model not found. Run `python main.py` first.\n\n{e}")
    st.stop()

# ── Sidebar (must come before any ticker-dependent data loads) ────────────────
with st.sidebar:
    st.markdown("## Alpha Engine")
    st.markdown("---")
    ticker = st.text_input("Ticker", value="NFLX",
                            help="Any valid Yahoo Finance ticker (NFLX, AAPL, TSLA...)").upper()
    period = st.selectbox("Chart period", ["6mo","1y","2y","5y","max"], index=2)
    st.markdown("---")
    st.markdown("**Model:** XGB + LGBM + RF + ET → Ridge")
    st.markdown("**Validation:** Walk-forward CV")
    st.markdown("**Target:** Next-day return (%)")
    st.markdown("**Features:** 51 technical indicators")
    st.markdown("---")
    st.markdown("[![Tests](https://github.com/SumedhPatil1507/netflix-stock-prediction/actions/workflows/test.yml/badge.svg)](https://github.com/SumedhPatil1507/netflix-stock-prediction/actions)")
    st.markdown("[GitHub Repo](https://github.com/SumedhPatil1507/netflix-stock-prediction)")

df_feat   = get_featured_data()
FEATURES  = model.feature_names_ if hasattr(model, "feature_names_") else get_active_features(df_feat)
df_live   = load_live_ohlcv(ticker, "2y")
df_source = df_live if df_live is not None else df_feat[["Open","High","Low","Close","Volume"]]

# ── Hero header ───────────────────────────────────────────────────────────────
st.markdown(f'<p class="hero-title">{APP_NAME}</p>', unsafe_allow_html=True)
st.caption(f"Real-time ML prediction · {ticker} · Backtesting · Sentiment · Risk · Drift Monitor")

# ── KPI row ───────────────────────────────────────────────────────────────────
metrics_path = os.path.join(REPO_ROOT, "outputs", "metrics.json")
if os.path.exists(metrics_path):
    with open(metrics_path) as f:
        m = json.load(f)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Directional Acc", f"{m.get('Dir_Acc', 0):.1f}%", help="% correct up/down predictions")
    c2.metric("CV R²", f"{m.get('CV_R2', 0):.4f}", help="Walk-forward cross-validation R²")
    c3.metric("CP Coverage", f"{m.get('CP_Coverage', 0):.1%}", help="Conformal prediction interval coverage")
    c4.metric("CP Width", f"{m.get('CP_Width', 0):.2f}%", help="90% prediction interval width")
    c5.metric("CV RMSE", f"{m.get('CV_RMSE', 0):.4f}", help="Walk-forward CV RMSE on returns")

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "🕯 Market Overview",
    "🔮 Predict",
    "📈 Backtesting",
    "📋 Paper Trade",
    "🧠 Sentiment",
    "⚠️ Risk",
    "🔬 Drift Monitor",
    "🔍 Explainability",
    "🤖 AI Narrative",
    "🏗 Architecture",
    "🧪 Strategy Lab",
    "🔬 Research Copilot",
    "📊 Track Record",
    "⚖️ Compliance",
    "📡 Observability",
])
(tab_market, tab_pred, tab_bt, tab_paper, tab_sent, tab_risk,
 tab_drift, tab_shap, tab_narrative, tab_arch,
 tab_strategy, tab_copilot, tab_track, tab_compliance, tab_obs) = tabs

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — MARKET OVERVIEW (Candlestick + indicators)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_market:
    st.subheader("Live Market Overview")

    @st.cache_data(ttl=7200)
    def _get_period_data(ticker_sym: str, p: str):
        try:
            import yfinance as yf
            df = yf.Ticker(ticker_sym).history(period=p)
            if hasattr(df.index.dtype, "tz") and df.index.dtype.tz is not None:
                df.index = df.index.tz_localize(None)
            return df[["Open","High","Low","Close","Volume"]].dropna()
        except Exception:
            return df_source

    df_p = _get_period_data(ticker, period)

    # ── Candlestick + Volume ──────────────────────────────────────────────────
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.75, 0.25], vertical_spacing=0.03)

    fig.add_trace(go.Candlestick(
        x=df_p.index, open=df_p["Open"], high=df_p["High"],
        low=df_p["Low"], close=df_p["Close"],
        name="NFLX", increasing_line_color="#00c853",
        decreasing_line_color="#e50914",
    ), row=1, col=1)

    # Moving averages
    for w, color in [(20,"#ffd700"),(50,"#00bcd4"),(200,"#ff9800")]:
        ma = df_p["Close"].rolling(w).mean()
        fig.add_trace(go.Scatter(x=df_p.index, y=ma, name=f"MA{w}",
                                  line=dict(color=color, width=1)), row=1, col=1)

    # Volume bars
    colors = ["#00c853" if c >= o else "#e50914"
              for c, o in zip(df_p["Close"], df_p["Open"])]
    fig.add_trace(go.Bar(x=df_p.index, y=df_p["Volume"], name="Volume",
                          marker_color=colors, opacity=0.6), row=2, col=1)

    fig.update_layout(
        template="plotly_dark", height=600,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.02),
        margin=dict(l=0, r=0, t=30, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── RSI + MACD ────────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        delta = df_p["Close"].diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rsi   = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=df_p.index, y=rsi, name="RSI",
                                      line=dict(color="#9c27b0", width=1.5)))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.6)
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.6)
        fig_rsi.add_hrect(y0=30, y1=70, fillcolor="gray", opacity=0.05)
        fig_rsi.update_layout(template="plotly_dark", height=250,
                               title="RSI (14)", margin=dict(l=0,r=0,t=30,b=0))
        st.plotly_chart(fig_rsi, use_container_width=True)

    with col2:
        ema12 = df_p["Close"].ewm(span=12, adjust=False).mean()
        ema26 = df_p["Close"].ewm(span=26, adjust=False).mean()
        macd  = ema12 - ema26
        sig   = macd.ewm(span=9, adjust=False).mean()
        hist  = macd - sig

        fig_macd = make_subplots(rows=1, cols=1)
        fig_macd.add_trace(go.Scatter(x=df_p.index, y=macd, name="MACD",
                                       line=dict(color="#2196f3", width=1.5)))
        fig_macd.add_trace(go.Scatter(x=df_p.index, y=sig, name="Signal",
                                       line=dict(color="#ff9800", width=1.5)))
        fig_macd.add_trace(go.Bar(x=df_p.index, y=hist, name="Histogram",
                                   marker_color=["#00c853" if v >= 0 else "#e50914" for v in hist],
                                   opacity=0.6))
        fig_macd.update_layout(template="plotly_dark", height=250,
                                title="MACD", margin=dict(l=0,r=0,t=30,b=0))
        st.plotly_chart(fig_macd, use_container_width=True)

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    bb_mid = df_p["Close"].rolling(20).mean()
    bb_std = df_p["Close"].rolling(20).std()
    bb_up  = bb_mid + 2 * bb_std
    bb_lo  = bb_mid - 2 * bb_std

    fig_bb = go.Figure()
    fig_bb.add_trace(go.Scatter(x=df_p.index, y=bb_up, name="Upper",
                                 line=dict(color="red", dash="dash", width=1)))
    fig_bb.add_trace(go.Scatter(x=df_p.index, y=bb_lo, name="Lower",
                                 line=dict(color="green", dash="dash", width=1),
                                 fill="tonexty", fillcolor="rgba(128,128,128,0.1)"))
    fig_bb.add_trace(go.Scatter(x=df_p.index, y=df_p["Close"], name="Close",
                                 line=dict(color="white", width=1.5)))
    fig_bb.add_trace(go.Scatter(x=df_p.index, y=bb_mid, name="MA20",
                                 line=dict(color="#ffd700", width=1, dash="dot")))
    fig_bb.update_layout(template="plotly_dark", height=350,
                          title="Bollinger Bands", margin=dict(l=0,r=0,t=30,b=0))
    st.plotly_chart(fig_bb, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PREDICT
# ═══════════════════════════════════════════════════════════════════════════════
with tab_pred:
    st.subheader("Next-Day Return Prediction")
    st.caption("Auto-filled with live NFLX data. Edit any row or use your own values.")

    @st.cache_data(ttl=7200, show_spinner=False)
    def _live_input(ticker_sym: str):
        try:
            import yfinance as yf
            df = yf.Ticker(ticker_sym).history(period="20d")
            if hasattr(df.index.dtype, "tz") and df.index.dtype.tz is not None:
                df.index = df.index.tz_localize(None)
            df = df[["Open","High","Low","Close","Volume"]].dropna().tail(10).round(2)
            return df.reset_index(drop=True).to_dict("list")
        except Exception:
            return {"Open":[600,605,610,615,620,625,630,635,640,645],
                    "High":[610,615,620,625,630,635,640,645,650,655],
                    "Low": [595,600,605,610,615,620,625,630,635,640],
                    "Close":[605,610,615,620,625,630,635,640,645,650],
                    "Volume":[5_000_000]*10}

    edited = st.data_editor(pd.DataFrame(_live_input(ticker)), num_rows="fixed",
                             use_container_width=True, key="pred_input")

    if st.button("Predict Next Close", type="primary"):
        try:
            d    = build_prediction_row(edited.copy(), model)
            last = edited["Close"].iloc[-1]
            pred_ret = float(model.predict(d)[0])
            pred_px  = last * (1 + pred_ret / 100)

            # Results
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Last Close",   f"${last:.2f}")
            c2.metric("Predicted",    f"${pred_px:.2f}")
            c3.metric("Return",       f"{pred_ret:+.3f}%",
                      delta=f"{pred_ret:+.3f}%", delta_color="normal")
            signal = "BUY" if pred_ret > 0 else "HOLD"
            c4.metric("Signal", signal)

            # Conformal interval
            if hasattr(model, "conformal_"):
                cp = model.conformal_
                lo_r, hi_r = cp.predict_interval(d)
                lo_p = last * (1 + lo_r[0] / 100)
                hi_p = last * (1 + hi_r[0] / 100)
                st.info(f"90% Prediction Interval: **${lo_p:.2f}** — **${hi_p:.2f}**  "
                        f"(return: {lo_r[0]:+.2f}% to {hi_r[0]:+.2f}%)")

            # Interactive mini chart
            fig_pred = go.Figure()
            fig_pred.add_trace(go.Scatter(
                x=list(range(len(edited))), y=edited["Close"],
                mode="lines+markers", name="Input",
                line=dict(color="#2196f3", width=2)))
            fig_pred.add_hline(y=pred_px, line_dash="dash",
                                line_color="#e50914",
                                annotation_text=f"Predicted: ${pred_px:.2f}")
            fig_pred.update_layout(template="plotly_dark", height=300,
                                    title="Input Window + Prediction",
                                    margin=dict(l=0,r=0,t=40,b=0))
            st.plotly_chart(fig_pred, use_container_width=True)

        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.exception(e)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — BACKTESTING
# ═══════════════════════════════════════════════════════════════════════════════
with tab_bt:
    st.subheader("Strategy Backtesting Engine")
    st.caption("Binary long/flat + Kelly-sized strategy vs Buy & Hold. Includes transaction costs.")

    bt_path = os.path.join(REPO_ROOT, "outputs", "backtest_curves.csv")
    rs_path = os.path.join(REPO_ROOT, "outputs", "rolling_sharpe.csv")

    if os.path.exists(bt_path):
        curves = pd.read_csv(bt_path, index_col=0)

        # ── Equity curve ──────────────────────────────────────────────────────
        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(y=curves["Strategy"], name="Binary Strategy",
                                     line=dict(color="#e50914", width=2)))
        if "Kelly" in curves.columns:
            fig_eq.add_trace(go.Scatter(y=curves["Kelly"], name="Kelly Strategy",
                                         line=dict(color="#ffd700", width=1.5, dash="dot")))
        fig_eq.add_trace(go.Scatter(y=curves["BuyAndHold"], name="Buy & Hold",
                                     line=dict(color="#9e9e9e", width=1.5, dash="dash")))
        fig_eq.add_hline(y=1.0, line_dash="dot", line_color="white", opacity=0.3)
        fig_eq.update_layout(template="plotly_dark", height=400,
                              title="Equity Curve (starting value = 1.0)",
                              yaxis_title="Portfolio Value",
                              margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig_eq, use_container_width=True)

        # ── Rolling Sharpe ────────────────────────────────────────────────────
        if os.path.exists(rs_path):
            rs = pd.read_csv(rs_path).squeeze()
            fig_rs = go.Figure()
            fig_rs.add_trace(go.Scatter(y=rs.values, name="Rolling Sharpe",
                                         line=dict(color="#9c27b0", width=1.5),
                                         fill="tozeroy",
                                         fillcolor="rgba(156,39,176,0.15)"))
            fig_rs.add_hline(y=0, line_color="white", opacity=0.3)
            fig_rs.add_hline(y=1, line_dash="dash", line_color="#00c853",
                              annotation_text="Sharpe = 1", opacity=0.6)
            fig_rs.update_layout(template="plotly_dark", height=250,
                                  title="Rolling 63-Day Sharpe Ratio",
                                  margin=dict(l=0,r=0,t=40,b=0))
            st.plotly_chart(fig_rs, use_container_width=True)

        # ── Drawdown ──────────────────────────────────────────────────────────
        strat = curves["Strategy"].values
        roll_max = np.maximum.accumulate(strat)
        dd = (strat - roll_max) / roll_max * 100
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(y=dd, name="Drawdown",
                                     fill="tozeroy", fillcolor="rgba(229,9,20,0.3)",
                                     line=dict(color="#e50914", width=1)))
        fig_dd.update_layout(template="plotly_dark", height=200,
                              title="Strategy Drawdown (%)",
                              margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig_dd, use_container_width=True)

        # ── Metrics ───────────────────────────────────────────────────────────
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                ml = json.load(f)
            st.markdown("#### Model Performance Metrics")
            cols = st.columns(min(len(ml), 5))
            for col, (k, v) in zip(cols, list(ml.items())[:5]):
                col.metric(k, f"{v:.4f}")
    else:
        st.info("Run `python main.py` to generate backtest results.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — PAPER TRADE
# ═══════════════════════════════════════════════════════════════════════════════
with tab_paper:
    st.subheader("Paper Trading Simulation")
    st.caption("Runs the model day-by-day on live data, logging each prediction vs actual — simulating real deployment.")

    PAPER_LOG = os.path.join(REPO_ROOT, "outputs", "paper_trade_log.csv")

    c_btn, c_days = st.columns([3, 1])
    with c_days:
        sim_days = st.number_input("Days to simulate", 10, 365, 90, 10)
    with c_btn:
        run_sim = st.button("Run Paper Trade Simulation", type="primary")

    if run_sim:
        with st.spinner(f"Simulating {sim_days} trading days..."):
            try:
                from src.paper_trade import run_paper_trade, paper_trade_summary
                log_df  = run_paper_trade(days=int(sim_days))
                summary = paper_trade_summary(log_df)
                st.session_state["paper_log"]     = log_df
                st.session_state["paper_summary"] = summary
            except Exception as e:
                st.error(f"Simulation failed: {e}")
                st.exception(e)

    # Load from session or saved CSV
    _log = st.session_state.get("paper_log",
           pd.read_csv(PAPER_LOG) if os.path.exists(PAPER_LOG) else None)
    _sum = st.session_state.get("paper_summary", {})

    if _log is not None and not _log.empty:
        if not _sum:
            from src.paper_trade import paper_trade_summary
            _sum = paper_trade_summary(_log)

        # KPIs
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Days Simulated", _sum.get("days_simulated", 0))
        k2.metric("Dir Accuracy",   f"{_sum.get('dir_accuracy_pct', 0):.1f}%")
        k3.metric("Trades Taken",   _sum.get("n_trades", 0))
        k4.metric("Win Rate",       f"{_sum.get('win_rate_pct', 0):.1f}%")
        k5.metric("Total PnL",      f"{_sum.get('total_pnl_pct', 0):+.2f}%")

        # Cumulative PnL
        _log["cum_pnl"] = _log["pnl_pct"].cumsum()
        fig_pnl = go.Figure()
        fig_pnl.add_trace(go.Scatter(
            x=list(range(len(_log))), y=_log["cum_pnl"],
            name="Cumulative PnL (%)",
            line=dict(color="#00c853", width=2),
            fill="tozeroy", fillcolor="rgba(0,200,83,0.1)"))
        fig_pnl.add_hline(y=0, line_color="white", opacity=0.3)
        fig_pnl.update_layout(template="plotly_dark", height=350,
                               title="Cumulative Paper Trade PnL (%)",
                               yaxis_title="PnL (%)",
                               margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_pnl, use_container_width=True)

        # Predicted vs Actual scatter
        fig_sc = px.scatter(
            _log, x="actual_return", y="pred_return",
            color="correct",
            color_discrete_map={True: "#00c853", False: "#e50914"},
            title="Predicted vs Actual Return (%)",
            labels={"actual_return": "Actual Return (%)",
                    "pred_return": "Predicted Return (%)"},
            template="plotly_dark", height=350,
            hover_data=["date", "signal", "direction"])
        fig_sc.add_hline(y=0, line_color="white", opacity=0.2)
        fig_sc.add_vline(x=0, line_color="white", opacity=0.2)
        fig_sc.update_layout(margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_sc, use_container_width=True)

        # Daily log table
        st.markdown("#### Daily Trade Log")
        st.dataframe(
            _log[["date","prev_close","next_close","pred_return",
                  "actual_return","signal","direction","correct","pnl_pct"]]
            .sort_values("date", ascending=False),
            use_container_width=True,
        )
    else:
        st.info("Click 'Run Paper Trade Simulation' to simulate the model on live data.")
        st.markdown("""
        **What this does:**
        - Fetches live NFLX data from Yahoo Finance
        - For each day in the simulation window, feeds the model the preceding history
        - Records: predicted return, actual return, signal (BUY/HOLD), correct/wrong
        - Shows cumulative PnL and directional accuracy over time
        - This is the closest thing to a live deployment test without real money
        """)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — SENTIMENT
# ═══════════════════════════════════════════════════════════════════════════════
with tab_sent:
    st.subheader("News Sentiment Analysis")
    st.caption("VADER sentiment scoring on Netflix headlines via Yahoo Finance. No API key required.")

    @st.cache_data(ttl=3600, show_spinner="Fetching news sentiment...")
    def _get_sentiment():
        try:
            import yfinance as yf
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            sia   = SentimentIntensityAnalyzer()
            news  = yf.Ticker("NFLX").news or []
            
            # If no news from yfinance, use sample data
            if not news:
                st.info("Live news unavailable from Yahoo Finance. Using sample data for demonstration.")
                import time
                now = time.time()
                sample_news = [
                    {"title": "Netflix stock surges on strong subscriber growth", "providerPublishTime": int(now - 86400)},
                    {"title": "Netflix faces increasing competition in streaming market", "providerPublishTime": int(now - 172800)},
                    {"title": "Netflix reports better than expected earnings", "providerPublishTime": int(now - 259200)},
                    {"title": "Netflix content strategy drives international expansion", "providerPublishTime": int(now - 345600)},
                    {"title": "Wall Street remains bullish on Netflix despite valuation concerns", "providerPublishTime": int(now - 432000)},
                    {"title": "Netflix advertising business shows promise", "providerPublishTime": int(now - 518400)},
                    {"title": "Netflix original content continues to drive engagement", "providerPublishTime": int(now - 604800)},
                ]
                news = sample_news
            
            rows  = []
            for item in news:
                ts    = pd.Timestamp(item.get("providerPublishTime", 0), unit="s")
                title = item.get("title", "")
                score = sia.polarity_scores(title)["compound"]
                rows.append({"date": ts, "title": title, "score": score,
                              "sentiment": "Positive" if score > 0.05
                              else ("Negative" if score < -0.05 else "Neutral")})
            return pd.DataFrame(rows)
        except ImportError:
            # Handle missing vaderSentiment gracefully
            return pd.DataFrame(columns=["date","title","score","sentiment"])
        except Exception as e:
            return pd.DataFrame(columns=["date","title","score","sentiment"])

    df_sent = _get_sentiment()

    if df_sent.empty:
        st.warning("Sentiment data unavailable. vaderSentiment may not be installed. Install with: `pip install vaderSentiment`")
    else:
        avg = df_sent["score"].mean()
        pos = (df_sent["sentiment"] == "Positive").sum()
        neg = (df_sent["sentiment"] == "Negative").sum()
        neu = (df_sent["sentiment"] == "Neutral").sum()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Avg Sentiment", f"{avg:+.3f}",
                  delta="Bullish" if avg > 0 else "Bearish",
                  delta_color="normal" if avg > 0 else "inverse")
        c2.metric("Positive", pos)
        c3.metric("Neutral",  neu)
        c4.metric("Negative", neg)

        # Sentiment bar chart
        fig_sent = px.bar(df_sent, x="date", y="score", color="sentiment",
                           color_discrete_map={"Positive":"#00c853",
                                               "Neutral":"#ffd700",
                                               "Negative":"#e50914"},
                           title="News Sentiment Scores",
                           template="plotly_dark", height=350)
        fig_sent.add_hline(y=0, line_color="white", opacity=0.3)
        fig_sent.update_layout(margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig_sent, use_container_width=True)

        # Pie chart
        fig_pie = px.pie(values=[pos, neu, neg],
                          names=["Positive","Neutral","Negative"],
                          color_discrete_sequence=["#00c853","#ffd700","#e50914"],
                          title="Sentiment Distribution",
                          template="plotly_dark", height=300)
        st.plotly_chart(fig_pie, use_container_width=True)

        # Headlines table
        st.markdown("#### Recent Headlines")
        st.dataframe(
            df_sent[["date","title","score","sentiment"]].sort_values("date", ascending=False),
            use_container_width=True,
        )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — RISK MANAGEMENT (Position Sizing + VaR + Execution Matrix)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_risk:
    st.subheader("Risk & Position Management")
    st.caption("Execution-ready position sizing with stop-loss, take-profit, Kelly fraction, VaR/CVaR, and circuit breaker.")

    # ── Portfolio settings ────────────────────────────────────────────────────
    with st.expander("Portfolio Settings", expanded=True):
        rc1, rc2, rc3, rc4 = st.columns(4)
        portfolio_val   = rc1.number_input("Portfolio ($)", 10_000, 10_000_000, 100_000, 10_000)
        max_pos_pct     = rc2.slider("Max Position %", 1, 20, 5) / 100
        max_heat_pct    = rc3.slider("Max Portfolio Heat %", 5, 50, 20) / 100
        halt_dd_pct     = rc4.slider("Circuit Breaker DD %", 5, 30, 10) / 100

    # ── Live prediction for risk calc ─────────────────────────────────────────
    st.markdown("#### Position Sizing Calculator")
    pr1, pr2, pr3, pr4 = st.columns(4)
    input_price    = pr1.number_input("Current Price ($)", 1.0, 10000.0, 650.0, 1.0)
    input_atr      = pr2.number_input("ATR (14-period)", 0.1, 500.0, 15.0, 0.5)
    input_pred_ret = pr3.number_input("Predicted Return (%)", -10.0, 10.0, 0.5, 0.1)
    input_win_rate = pr4.slider("Historical Win Rate %", 40, 65, 52) / 100

    if st.button("Compute Position", type="primary"):
        try:
            from src.risk_manager import RiskManager, RiskConfig
            cfg = RiskConfig(
                portfolio_value    = portfolio_val,
                max_position_pct   = max_pos_pct,
                max_portfolio_heat = max_heat_pct,
                max_drawdown_halt  = halt_dd_pct,
            )
            rm    = RiskManager(cfg)
            order = rm.compute_position(
                ticker      = ticker,
                pred_return = input_pred_ret,
                last_price  = input_price,
                atr         = input_atr,
                win_rate    = input_win_rate,
            )
            matrix = rm.risk_matrix(input_pred_ret, input_price, input_atr, portfolio_val)

            # Signal banner
            if order.signal == "BUY":
                st.success(f"Signal: **BUY** — {order.shares} shares @ ${order.entry_price:.2f}")
            elif order.signal == "HALT":
                st.error("Circuit breaker triggered — trading halted")
            else:
                st.warning(f"Signal: **{order.signal}** — {order.notes}")

            # Position metrics
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric("Shares",        order.shares)
            m2.metric("Position ($)",  f"${order.position_value:,.0f}")
            m3.metric("Stop Loss",     f"${order.stop_loss:.2f}")
            m4.metric("Take Profit",   f"${order.take_profit:.2f}")
            m5.metric("Risk/Trade",    f"${order.risk_per_trade:,.0f}")
            m6.metric("Kelly Frac",    f"{order.kelly_fraction:.3f}")

            # Risk matrix table
            st.markdown("#### Risk Matrix")
            matrix_df = pd.DataFrame([matrix]).T.reset_index()
            matrix_df.columns = ["Parameter", "Value"]
            st.dataframe(matrix_df, use_container_width=True, hide_index=True)

            # Risk/Reward chart
            prices = np.linspace(input_price * 0.85, input_price * 1.15, 100)
            pnl    = (prices - input_price) * order.shares
            fig_rr = go.Figure()
            fig_rr.add_trace(go.Scatter(x=prices, y=pnl, mode="lines",
                                         line=dict(color="#2196f3", width=2), name="P&L"))
            fig_rr.add_hline(y=0, line_color="white", opacity=0.3)
            fig_rr.add_vline(x=order.stop_loss,   line_dash="dash", line_color="#e50914",
                              annotation_text="Stop Loss")
            fig_rr.add_vline(x=order.take_profit, line_dash="dash", line_color="#00c853",
                              annotation_text="Take Profit")
            fig_rr.add_vline(x=input_price,       line_dash="dot",  line_color="white",
                              annotation_text="Entry")
            fig_rr.update_layout(template="plotly_dark", height=350,
                                  title="P&L vs Price (Risk/Reward Diagram)",
                                  xaxis_title="Price ($)", yaxis_title="P&L ($)",
                                  margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig_rr, use_container_width=True)

        except Exception as e:
            st.error(f"Risk calculation error: {e}")

    st.markdown("---")

    # ── VaR / CVaR section ────────────────────────────────────────────────────
    st.markdown("#### Portfolio Risk Metrics (Historical)")
    if "Return" in df_feat.columns:
        ret  = df_feat["Return"].dropna() / 100
        conf = st.slider("Confidence Level", 0.90, 0.99, 0.95, 0.01)
        var  = float(np.percentile(ret, (1 - conf) * 100))
        cvar = float(ret[ret <= var].mean())

        v1, v2, v3, v4 = st.columns(4)
        v1.metric(f"VaR ({conf:.0%})",  f"{var:.3%}")
        v2.metric(f"CVaR ({conf:.0%})", f"{cvar:.3%}")
        v3.metric("Ann. Volatility",    f"{ret.std() * np.sqrt(252):.2%}")
        v4.metric("Hist. Sharpe",
                  f"{ret.mean() / ret.std() * np.sqrt(252):.3f}" if ret.std() > 0 else "N/A")

        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(x=ret * 100, nbinsx=120,
                                         marker_color="#2196f3", opacity=0.7))
        fig_dist.add_vline(x=var * 100,  line_dash="dash", line_color="#e50914",
                            annotation_text=f"VaR {conf:.0%}")
        fig_dist.add_vline(x=cvar * 100, line_dash="dash", line_color="#ff9800",
                            annotation_text=f"CVaR {conf:.0%}")
        fig_dist.update_layout(template="plotly_dark", height=300,
                                title="Return Distribution with VaR/CVaR",
                                xaxis_title="Daily Return (%)",
                                margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_dist, use_container_width=True)

    # ── Volatility surface ────────────────────────────────────────────────────
    if "Return" in df_feat.columns:
        ret = df_feat["Return"].dropna() / 100
        vol_20  = ret.rolling(20).std()  * np.sqrt(252) * 100
        vol_60  = ret.rolling(60).std()  * np.sqrt(252) * 100
        vol_120 = ret.rolling(120).std() * np.sqrt(252) * 100
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Scatter(x=df_feat.index, y=vol_20,  name="20d",
                                      line=dict(color="#e50914", width=1.5)))
        fig_vol.add_trace(go.Scatter(x=df_feat.index, y=vol_60,  name="60d",
                                      line=dict(color="#ffd700", width=1.5)))
        fig_vol.add_trace(go.Scatter(x=df_feat.index, y=vol_120, name="120d",
                                      line=dict(color="#00bcd4", width=1.5)))
        fig_vol.update_layout(template="plotly_dark", height=300,
                               title="Annualised Volatility Surface",
                               yaxis_title="Volatility (%)",
                               margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_vol, use_container_width=True)

    # ── Correlation matrix ────────────────────────────────────────────────────
    corr_cols = ["Return","RSI","MACD_Norm","BB_Pct","ATR_Norm",
                 "Volatility","Stoch_K","Williams_R","CCI","Momentum5"]
    avail = [c for c in corr_cols if c in df_feat.columns]
    if avail:
        corr = df_feat[avail].dropna().corr()
        fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                              zmin=-1, zmax=1, title="Feature Correlation Matrix",
                              template="plotly_dark", height=450)
        fig_corr.update_layout(margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_corr, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — DRIFT MONITOR
# ═══════════════════════════════════════════════════════════════════════════════
with tab_drift:
    st.subheader("Model Drift Monitor")
    st.caption("PSI + KS test comparing training vs recent data distribution.")

    try:
        split    = int(len(df_feat) * 0.8)
        df_train = df_feat.iloc[:split]
        df_rec   = df_feat.iloc[split:]

        from src.drift import detect_drift, drift_summary_df
        dr  = detect_drift(df_train, df_rec, FEATURES)
        ddf = drift_summary_df(dr)

        n_drift = len(dr["drifted_features"])
        if dr["overall_drift"]:
            st.error(f"Significant drift in {n_drift} features — consider retraining.")
        elif n_drift > 0:
            st.warning(f"Moderate drift in {n_drift} features.")
        else:
            st.success("No significant drift. Model is stable.")

        c1, c2, c3 = st.columns(3)
        c1.metric("Features Checked", len(ddf))
        c2.metric("Drifted",          n_drift)
        c3.metric("PSI Threshold",    dr["psi_threshold"])

        top20 = ddf.head(20)
        fig_psi = px.bar(top20, x="PSI", y="Feature", orientation="h",
                          color="Drifted",
                          color_discrete_map={True:"#e50914", False:"#2196f3"},
                          title="Top 20 Features by PSI",
                          template="plotly_dark", height=500)
        fig_psi.add_vline(x=0.1, line_dash="dash", line_color="#ffd700",
                           annotation_text="Moderate")
        fig_psi.add_vline(x=0.2, line_dash="dash", line_color="#e50914",
                           annotation_text="Significant")
        fig_psi.update_layout(margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig_psi, use_container_width=True)

        st.dataframe(ddf, use_container_width=True)

    except Exception as e:
        st.warning(f"Drift monitor error: {e}")
        st.info("Install scipy: `pip install scipy`")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 8 — EXPLAINABILITY (Feature Importance — interactive Plotly)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_shap:
    st.subheader("Model Explainability")
    st.caption("Feature importance from all 4 base learners + correlation analysis.")

    @st.cache_data(show_spinner="Computing feature importance...")
    def _compute_importance():
        """Average feature importances from all tree-based base learners."""
        try:
            feat_cols = model.feature_names_ if hasattr(model, "feature_names_") else FEATURES
            imps, count = None, 0
            for name, est in model.fitted_learners_:
                if hasattr(est, "feature_importances_"):
                    fi = np.array(est.feature_importances_[:len(feat_cols)], dtype=np.float64)
                    imps = fi if imps is None else imps + fi
                    count += 1
            if imps is None or count == 0:
                return None, None, "No feature importances available"
            return imps / count, feat_cols, None
        except Exception as e:
            return None, None, str(e)

    imps, feat_cols, err = _compute_importance()

    if imps is None:
        st.warning(f"Feature importance unavailable: {err}")
    else:
        n_top = st.slider("Top N features", 5, min(40, len(feat_cols)), 20)

        # ── 1. Avg feature importance bar ─────────────────────────────────────
        idx       = np.argsort(imps)[-n_top:]
        top_feats = np.array(feat_cols)[idx]
        top_vals  = imps[idx]

        fig_fi = go.Figure(go.Bar(
            x=top_vals, y=top_feats, orientation="h",
            marker=dict(color=top_vals, colorscale="Reds",
                        showscale=True, colorbar=dict(title="Importance")),
        ))
        fig_fi.update_layout(
            template="plotly_dark", height=max(400, n_top * 22),
            title=f"Top {n_top} Features — Avg Importance (XGB + LGBM + RF + ET)",
            xaxis_title="Feature Importance (higher = more influential)",
            margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig_fi, use_container_width=True)

        # ── 2. Per-model importance comparison ────────────────────────────────
        st.markdown("#### Per-Model Importance Comparison")
        model_imps = {}
        for name, est in model.fitted_learners_:
            if hasattr(est, "feature_importances_"):
                fi = np.array(est.feature_importances_[:len(feat_cols)], dtype=np.float64)
                model_imps[name] = fi

        if model_imps:
            top_feat_list = list(top_feats)
            fig_comp = go.Figure()
            colors = {"xgb": "#e50914", "lgbm": "#ffd700",
                      "rf": "#00c853", "et": "#00bcd4"}
            for mname, mfi in model_imps.items():
                vals = [mfi[list(feat_cols).index(f)] if f in feat_cols else 0
                        for f in top_feat_list]
                fig_comp.add_trace(go.Bar(
                    name=mname.upper(), x=vals, y=top_feat_list,
                    orientation="h",
                    marker_color=colors.get(mname, "#9e9e9e"),
                    opacity=0.8,
                ))
            fig_comp.update_layout(
                template="plotly_dark", barmode="group",
                height=max(400, n_top * 28),
                title="Feature Importance by Model",
                xaxis_title="Importance",
                margin=dict(l=0, r=0, t=40, b=0),
            )
            st.plotly_chart(fig_comp, use_container_width=True)

        # ── 3. Feature correlation with target ────────────────────────────────
        st.markdown("#### Feature Correlation with Next-Day Return")
        feat_df = get_featured_data()
        if "Return" in feat_df.columns:
            feat_df["NextReturn"] = feat_df["Return"].shift(-1)
            avail = [f for f in top_feats if f in feat_df.columns]
            corr  = feat_df[avail + ["NextReturn"]].dropna() \
                        .corr()["NextReturn"].drop("NextReturn").reindex(avail)

            fig_corr = go.Figure(go.Bar(
                x=corr.values, y=corr.index,
                orientation="h",
                marker_color=["#00c853" if v > 0 else "#e50914" for v in corr.values],
            ))
            fig_corr.add_vline(x=0, line_color="white", opacity=0.3)
            fig_corr.update_layout(
                template="plotly_dark", height=max(350, n_top * 22),
                title="Pearson Correlation of Top Features with Next-Day Return",
                xaxis_title="Correlation coefficient",
                margin=dict(l=0, r=0, t=40, b=0),
            )
            st.plotly_chart(fig_corr, use_container_width=True)

        st.caption(
            "Feature importance = average gain across all splits in each tree model. "
            "Correlation shows linear relationship with next-day return — "
            "low correlation doesn't mean a feature is useless (non-linear effects)."
        )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 9 — AI NARRATIVE
# ═══════════════════════════════════════════════════════════════════════════════
with tab_narrative:
    st.subheader("AI Market Narrator")
    st.caption("Agentic RAG system that explains model predictions with citation-backed narratives from earnings calls and financial news.")
    
    # Check if narrator dependencies are available
    narrator_available = False
    try:
        from src.narrator import NarratorGraph, VectorStore, CorpusManager
        from src.agent_traces import get_tracer
        narrator_available = True
    except ImportError as e:
        st.warning(f"""
        **AI Narrator dependencies not fully installed.**
        
        To enable the AI Narrator features, install the additional dependencies:
        ```bash
        pip install langchain langchain-openai langgraph langfuse chromadb sentence-transformers ragas openai
        ```
        
        Error: {str(e)}
        
        The rest of the dashboard will continue to work normally.
        """)
    
    if narrator_available:
        # Initialize narrator components
        @st.cache_resource(show_spinner="Loading AI Narrator components...")
        def load_narrator_components():
            try:
                vector_store = VectorStore()
                corpus_manager = CorpusManager()
                narrator_graph = NarratorGraph(
                    vector_store=vector_store,
                    corpus_manager=corpus_manager
                )
                tracer = get_tracer()
                
                # Initialize vector store with sample data if empty
                stats = vector_store.get_collection_stats()
                if stats["document_count"] == 0:
                    narrator_graph.initialize_vector_store("NFLX")
                
                return narrator_graph, vector_store, tracer
            except Exception as e:
                st.error(f"Error loading narrator components: {e}")
                return None, None, None
        
        narrator_graph, vector_store, tracer = load_narrator_components()
        
        if narrator_graph is None:
            st.warning("AI Narrator components could not be loaded. Please check the error above.")
        else:
            # Show vector store stats
            stats = vector_store.get_collection_stats()
            c1, c2, c3 = st.columns(3)
            c1.metric("Documents in Vector Store", stats["document_count"])
            c2.metric("Collection Name", stats["collection_name"])
            c3.metric("Embedding Model", stats["embedding_model"])
            
            st.markdown("---")
            
            # Generate narrative section
            st.markdown("#### Generate Market Narrative")
            
            # Get current price from live data
            current_price = df_source["Close"].iloc[-1] if not df_source.empty else 650.0
            
            # Input prediction parameters
            col1, col2, col3 = st.columns(3)
            with col1:
                input_prediction = st.number_input(
                    "Predicted Return (%)",
                    value=0.5,
                    min_value=-10.0,
                    max_value=10.0,
                    step=0.1,
                    help="Model's predicted next-day return"
                )
            with col2:
                input_ci_lower = st.number_input(
                    "CI Lower Bound (%)",
                    value=-0.5,
                    min_value=-10.0,
                    max_value=10.0,
                    step=0.1,
                    help="Lower bound of conformal prediction interval"
                )
            with col3:
                input_ci_upper = st.number_input(
                    "CI Upper Bound (%)",
                    value=1.5,
                    min_value=-10.0,
                    max_value=10.0,
                    step=0.1,
                    help="Upper bound of conformal prediction interval"
                )
            
            current_price = st.number_input(
                "Current Price ($)",
                value=current_price,
                min_value=1.0,
                max_value=10000.0,
                step=1.0
            )
            
            if st.button("Generate AI Narrative", type="primary"):
                with st.spinner("Generating narrative with RAG pipeline..."):
                    try:
                        # Run the narrator workflow
                        result = narrator_graph.run(
                            ticker=ticker,
                            prediction=input_prediction / 100,  # Convert to decimal
                            conformal_interval=(input_ci_lower / 100, input_ci_upper / 100),
                            current_price=current_price
                        )
                        
                        if result["success"]:
                            narrative   = result["narrative"] or ""
                            prediction  = result["prediction"]          # decimal
                            ci_lower, ci_upper = result["conformal_interval"]
                            retrieved_docs = result.get("retrieved_documents") or []
                            sources_used = result.get("sources_used", len(retrieved_docs))

                            # ── a. Sentiment Gauge ─────────────────────────────
                            st.markdown("### Sentiment & Confidence")
                            col_g1, col_g2 = st.columns(2)

                            with col_g1:
                                sentiment_value = max(-1.0, min(1.0, prediction * 100))
                                fig_gauge = go.Figure(go.Indicator(
                                    mode="gauge+number",
                                    value=sentiment_value,
                                    title={"text": "Sentiment Score"},
                                    gauge={
                                        "axis": {"range": [-1, 1]},
                                        "bar": {"color": "darkblue"},
                                        "steps": [
                                            {"range": [-1, 0], "color": "rgba(255,80,80,0.3)"},
                                            {"range": [0, 1],  "color": "rgba(80,200,80,0.3)"}
                                        ],
                                        "threshold": {
                                            "line": {"color": "black", "width": 3},
                                            "thickness": 0.75,
                                            "value": sentiment_value
                                        }
                                    }
                                ))
                                fig_gauge.update_layout(height=300, margin=dict(t=40, b=10, l=10, r=10))
                                st.plotly_chart(fig_gauge, use_container_width=True)

                            # ── b. Confidence Interval chart ───────────────────
                            with col_g2:
                                pred_pct   = prediction * 100
                                lower_pct  = ci_lower * 100
                                upper_pct  = ci_upper * 100
                                fig_ci = go.Figure(go.Scatter(
                                    x=[pred_pct],
                                    y=[0],
                                    mode="markers",
                                    marker=dict(size=14, color="royalblue"),
                                    error_x=dict(
                                        type="data",
                                        symmetric=False,
                                        minus=abs(pred_pct - lower_pct),
                                        plus=abs(upper_pct - pred_pct),
                                        visible=True,
                                        color="royalblue",
                                        thickness=3,
                                        width=8
                                    ),
                                    name="Prediction ± CI"
                                ))
                                fig_ci.update_layout(
                                    title="Conformal Prediction Interval",
                                    xaxis_title="Predicted Return (%)",
                                    yaxis=dict(showticklabels=False, zeroline=False),
                                    height=300,
                                    margin=dict(t=40, b=40, l=10, r=10)
                                )
                                st.plotly_chart(fig_ci, use_container_width=True)

                            # ── c. Source Relevance Bar chart ──────────────────
                            if retrieved_docs:
                                st.markdown("### Source Relevance")
                                relevances = [
                                    1 - doc["distance"] if "distance" in doc and doc["distance"] is not None
                                    else 0.8
                                    for doc in retrieved_docs
                                ]
                                titles = [
                                    (doc.get("metadata", {}).get("title", f"Doc {i+1}") or f"Doc {i+1}")[:50]
                                    for i, doc in enumerate(retrieved_docs)
                                ]
                                fig_bar = go.Figure(go.Bar(
                                    x=relevances,
                                    y=titles,
                                    orientation="h",
                                    marker=dict(
                                        color=relevances,
                                        colorscale="Greens",
                                        showscale=True,
                                        cmin=0,
                                        cmax=1
                                    )
                                ))
                                fig_bar.update_layout(
                                    title="Retrieved Document Relevance Scores",
                                    xaxis_title="Relevance Score (1 - distance)",
                                    yaxis_title="Document",
                                    height=max(250, 50 * len(retrieved_docs)),
                                    margin=dict(t=40, b=40, l=10, r=10)
                                )
                                st.plotly_chart(fig_bar, use_container_width=True)

                            # ── d. Narrative text ──────────────────────────────
                            st.markdown("### Generated Market Narrative")
                            st.markdown(narrative)

                            # ── Metadata row ───────────────────────────────────
                            st.markdown("---")
                            m1, m2, m3, m4 = st.columns(4)
                            m1.metric("Sentiment", result["sentiment"].title() if result.get("sentiment") else "N/A")
                            m2.metric("Prediction", f"{input_prediction:+.2f}%")
                            m3.metric("Sources Used", sources_used)
                            query_str = result.get("query") or ""
                            m4.metric("Query", query_str[:30] + "..." if len(query_str) > 30 else query_str)

                            # ── Citations ──────────────────────────────────────
                            if result.get("citations"):
                                st.markdown("#### Source Citations")
                                for i, citation in enumerate(result["citations"], 1):
                                    with st.expander(f"Citation {i}: {citation['title']}"):
                                        st.markdown(f"**Source:** {citation['source']}")
                                        st.markdown(f"**Date:** {citation['date']}")
                                        if citation.get("url"):
                                            st.markdown(f"**URL:** {citation['url']}")

                            # ── Retrieved Documents ────────────────────────────
                            if retrieved_docs:
                                st.markdown("#### Retrieved Documents")
                                for i, doc in enumerate(retrieved_docs, 1):
                                    with st.expander(f"Document {i}: {doc['metadata'].get('title', 'Unknown')}"):
                                        st.markdown(f"**Source:** {doc['metadata'].get('source', 'Unknown')}")
                                        st.markdown(f"**Date:** {doc['metadata'].get('date', 'Unknown')}")
                                        st.markdown(f"**Content:** {doc['text']}")
                                        if doc.get("distance") is not None:
                                            st.metric("Relevance Score", f"{1 - doc['distance']:.3f}")

                            # ── Log the run ────────────────────────────────────
                            if tracer:
                                tracer.log_graph_run(
                                    workflow_name="narrator_graph",
                                    inputs={
                                        "ticker": ticker,
                                        "prediction": input_prediction,
                                        "conformal_interval": (input_ci_lower, input_ci_upper)
                                    },
                                    outputs=result
                                )

                            # ── e. Faithfulness Evaluation button ──────────────
                            st.markdown("---")
                            if st.button("Run Faithfulness Evaluation"):
                                with st.spinner("Running faithfulness evaluation..."):
                                    try:
                                        from src.narrator.eval import NarrativeEvaluator
                                        evaluator = NarrativeEvaluator()
                                        query_for_eval = result.get("query") or f"{ticker} stock analysis"
                                        eval_result = evaluator.evaluate_narrative(
                                            narrative, retrieved_docs, query_for_eval
                                        )
                                        faith_score = eval_result.get("scores", {}).get("faithfulness", 0.0)
                                        if isinstance(faith_score, str):
                                            faith_score = 0.0
                                        st.metric("Faithfulness Score", f"{faith_score:.3f}")
                                        fig_faith = go.Figure(go.Indicator(
                                            mode="gauge+number",
                                            value=float(faith_score),
                                            title={"text": "Faithfulness"},
                                            gauge={
                                                "axis": {"range": [0, 1]},
                                                "bar": {"color": "steelblue"},
                                                "steps": [
                                                    {"range": [0, 0.4], "color": "rgba(255,80,80,0.3)"},
                                                    {"range": [0.4, 0.7], "color": "rgba(255,200,80,0.3)"},
                                                    {"range": [0.7, 1.0], "color": "rgba(80,200,80,0.3)"}
                                                ]
                                            }
                                        ))
                                        fig_faith.update_layout(height=280, margin=dict(t=40, b=10, l=10, r=10))
                                        st.plotly_chart(fig_faith, use_container_width=True)
                                        is_fallback = eval_result.get("fallback", False)
                                        method = eval_result.get("scores", {}).get("method", "ragas")
                                        st.caption(f"Evaluation method: {'lexical overlap (fallback)' if is_fallback else method}")
                                    except Exception as eval_err:
                                        st.error(f"Evaluation error: {eval_err}")

                            # ── f. Agent Trace Log Viewer ──────────────────────
                            JSONL_LOG = Path(REPO_ROOT) / "logs" / "agent_traces.jsonl"
                            with st.expander("View Agent Trace Log"):
                                if JSONL_LOG.exists():
                                    try:
                                        lines = JSONL_LOG.read_text(encoding="utf-8").splitlines()
                                        last_10 = lines[-10:]
                                        records = []
                                        for ln in last_10:
                                            try:
                                                records.append(json.loads(ln))
                                            except Exception:
                                                pass
                                        if records:
                                            import pandas as _pd
                                            st.dataframe(_pd.DataFrame(records), use_container_width=True)
                                        else:
                                            st.info("Trace log is empty.")
                                    except Exception as log_err:
                                        st.error(f"Could not read trace log: {log_err}")
                                else:
                                    st.info("No trace log yet. Run a narrative generation first.")

                        else:
                            st.error(f"Error generating narrative: {result.get('error', 'Unknown error')}")
                    
                    except Exception as e:
                        st.error(f"Error in narrative generation: {e}")
                        st.exception(e)
            
            st.markdown("---")
            
            # Vector store management
            st.markdown("#### Vector Store Management")
            
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("Reinitialize Vector Store"):
                    with st.spinner("Reinitializing vector store..."):
                        try:
                            narrator_graph.initialize_vector_store(ticker)
                            st.success("Vector store reinitialized successfully!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error reinitializing: {e}")
            
            with col_b:
                if st.button("Clear Vector Store"):
                    with st.spinner("Clearing vector store..."):
                        try:
                            vector_store.clear_collection()
                            st.success("Vector store cleared successfully!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error clearing: {e}")
            
            # Evaluation section
            st.markdown("---")
            st.markdown("#### Narrative Evaluation")
            
            st.markdown("""
            **RAGAS-based Evaluation**: The system includes RAGAS metrics to evaluate narrative faithfulness 
            against retrieved sources. This ensures the AI narratives are grounded in the actual retrieved documents.
            
            **Metrics Available:**
            - **Faithfulness**: Measures how well the narrative aligns with retrieved context
            - **Answer Relevancy**: Measures how relevant the narrative is to the original query
            - **Context Precision**: Measures the relevance of retrieved documents
            
            To run evaluation, use the evaluation script in `src/narrator/eval.py`.
            """)
            
            # System explanation
            st.markdown("---")
            st.markdown("#### How It Works")
            
            st.markdown("""
            **AI Market Narrator Pipeline:**
            
            1. **Retriever Agent**: Uses ChromaDB vector store to find relevant earnings call transcripts 
               and financial news articles based on the prediction context.
            
            2. **Synthesis Agent**: Reads the model prediction, conformal interval, and retrieved documents 
               to generate a plain-English narrative explaining the bullish/bearish stance.
            
            3. **Citation System**: Automatically extracts and formats citations from the retrieved documents 
               to provide transparency and source attribution.
            
            4. **Observability**: Every agent run is logged to Langfuse for monitoring and debugging.
            
            5. **Evaluation**: RAGAS-based evaluation ensures narrative faithfulness against retrieved sources.
            
            **Technologies Used:**
            - **LangGraph**: Multi-agent workflow orchestration
            - **ChromaDB**: Vector database for semantic search
            - **LangChain**: LLM integration and agent framework
            - **Langfuse**: Observability and tracing
            - **RAGAS**: Evaluation metrics for RAG systems
            """)
    else:
        # Show information about what the AI Narrator would do
        st.markdown("---")
        st.markdown("#### AI Market Narrator Features")
        
        st.markdown("""
        The AI Market Narrator provides:
        
        - **Agentic RAG System**: Multi-agent workflow with retriever and synthesis agents
        - **ChromaDB Vector Store**: Semantic search over earnings call transcripts and financial news
        - **Citation System**: Automatic source attribution for all narrative claims
        - **Langfuse Observability**: Comprehensive logging and tracing of all agent runs
        - **RAGAS Evaluation**: Faithfulness scoring to ensure narratives are grounded in retrieved sources
        
        **To enable these features, install the additional dependencies:**
        ```bash
        pip install langchain langchain-openai langgraph langfuse chromadb sentence-transformers ragas openai
        ```
        
        **Required Environment Variables:**
        - `OPENAI_API_KEY`: Your OpenAI API key for LLM-powered narrative generation
        - `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY`: Optional, for observability
        """)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 10 — ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════════════
with tab_arch:
    st.subheader("System Architecture & Edge")

    st.markdown("""
### The Edge

Most stock prediction projects predict **price** (trivially correlated with itself).
This project predicts **next-day return (%)** — a stationary, genuinely hard target.

The directional accuracy metric (>52% = real signal) is what matters, not R².

---

### Full Pipeline Architecture

```
Data Sources (multi-source)
  ├── yfinance        — daily/intraday, free
  ├── Alpha Vantage   — daily + 1min/5min REST, free tier
  └── Alpaca Markets  — minute bars, free paper account
        │
        ▼
  data_loader.py  ──── validation, fallback chain
        │
        ▼
  feature_engineering.py  ──── 51 technical features
        │
        ▼
  regime_detection.py  ──── HMM Bull/Bear/Sideways
        │
        ▼
  ManualStackingRegressor
  XGB + LGBM + RF + ET → Ridge (OOF stacking)
        │
        ▼
  uncertainty.py  ──── conformal prediction intervals (90%)
        │
        ▼
  risk_manager.py  ──── ATR stop-loss, Kelly sizing,
        │                circuit breaker, portfolio heat
        ▼
  model_registry.py  ──── versioned saves + registry.json
        │
        ▼
  monitoring.py  ──── Slack/email drift + retrain alerts
        │
        ├── FastAPI v2.0
        │     /predict        — ML prediction + CI
        │     /risk/position  — execution-ready position size
        │     /risk/matrix    — full risk matrix
        │     /execute        — broker integration (Alpaca/paper)
        │     /model_info     — version + metrics
        │     /registry       — model version history
        │
        ├── Streamlit (9 tabs, all Plotly interactive)
        │     Market Overview · Predict · Backtesting
        │     Paper Trade · Sentiment · Risk Management
        │     Drift Monitor · Explainability · Architecture
        │
        └── GitHub Actions
              test.yml    — CI on every push
              retrain.yml — weekly scheduled retraining
```

---

### Design Decisions

| Choice | Reason |
|---|---|
| Return target (not price) | Stationary; avoids spurious R² from autocorrelation |
| Manual stacking (not sklearn) | Avoids is_regressor() validator bug with XGB/LGBM |
| Walk-forward CV | Only valid CV for time-series; no future leakage |
| Conformal prediction | Calibrated intervals with mathematical coverage guarantee |
| HMM regime detection | Market dynamics differ across regimes |
| ATR-based stop-loss | Adapts to current volatility; tighter in calm markets |
| Kelly criterion | Bet proportional to edge; maximises long-run growth |
| Model versioning | Rollback capability; track performance over time |
| Multi-source data | Fallback chain ensures reliability; intraday capability |

---

### Honest Limitations

- No earnings surprise signal (biggest NFLX driver — ±15% moves)
- Technical indicators are correlated — ~10 independent signals, not 51
- 15-min delayed Yahoo Finance data — not suitable for HFT
- Model trained on 2002–2026; pre-streaming era data may not generalise
- Alpaca execution is paper-only by default — live trading requires explicit config

---

### Tests
[![Tests](https://github.com/SumedhPatil1507/netflix-stock-prediction/actions/workflows/test.yml/badge.svg)](https://github.com/SumedhPatil1507/netflix-stock-prediction/actions)

Run locally: `pytest tests/ -v`
    """)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 11 — STRATEGY LAB
# ═══════════════════════════════════════════════════════════════════════════════
with tab_strategy:
    st.subheader("🧪 Strategy Lab")
    st.caption("Define, run, and compare multi-ticker strategies side by side.")
    try:
        from src.strategy_registry import StrategyRegistry, StrategyConfig
        _reg = StrategyRegistry()
        _strategy_names = _reg.list_strategies()
        col_run, col_compare = st.columns(2)
        with col_run:
            st.markdown("#### Run a Strategy")
            sel_strategy = st.selectbox("Select strategy", _strategy_names, key="lab_sel")
            if sel_strategy:
                cfg = _reg.get(sel_strategy)
                st.caption(cfg.description)
                st.write(f"**Tickers:** {', '.join(cfg.tickers)}")
                st.write(f"**Feature set:** {len(cfg.feature_set)} features")
                src_sel = st.selectbox("Data source", ["csv", "yfinance"], key="lab_src")
                if st.button("▶ Run Strategy", type="primary"):
                    with st.spinner(f"Running {sel_strategy}..."):
                        try:
                            result = _reg.run_strategy(sel_strategy, source=src_sel)
                            if "error" in result:
                                st.error(result["error"])
                            else:
                                metrics = result.get("metrics", {})
                                c1, c2, c3, c4 = st.columns(4)
                                c1.metric("Ticker", result.get("ticker", "?"))
                                c2.metric("CV R2", f"{metrics.get('CV_R2', metrics.get('R2', 0)):.4f}")
                                c3.metric("Dir Acc", f"{metrics.get('Dir_Acc', 0):.1f}%")
                                c4.metric("CV RMSE", f"{metrics.get('CV_RMSE', metrics.get('RMSE', 0)):.4f}")
                        except Exception as e:
                            st.error(f"Strategy run failed: {e}")
        with col_compare:
            st.markdown("#### Compare Strategies")
            compare_sel = st.multiselect("Strategies to compare", _strategy_names, default=_strategy_names[:2], key="lab_cmp")
            if st.button("Compare", type="secondary") and compare_sel:
                with st.spinner("Comparing strategies..."):
                    try:
                        compare_results = _reg.compare_strategies(compare_sel, source="csv")
                        rows = []
                        for sname, res in compare_results.items():
                            if "error" not in res:
                                m = res.get("metrics", {})
                                rows.append({"Strategy": sname, "Ticker": res.get("ticker", "?"), "R2": m.get("CV_R2", m.get("R2", 0)), "Dir Acc %": m.get("Dir_Acc", 0), "RMSE": m.get("CV_RMSE", m.get("RMSE", 0))})
                        if rows:
                            df_cmp = pd.DataFrame(rows)
                            fig_cmp = go.Figure()
                            for col_metric in ["R2", "Dir Acc %"]:
                                fig_cmp.add_trace(go.Bar(name=col_metric, x=df_cmp["Strategy"], y=df_cmp[col_metric]))
                            fig_cmp.update_layout(template="plotly_dark", barmode="group", height=350, title="Strategy Comparison", margin=dict(l=0,r=0,t=40,b=0))
                            st.plotly_chart(fig_cmp, use_container_width=True)
                            st.dataframe(df_cmp, use_container_width=True)
                    except Exception as e:
                        st.error(f"Comparison failed: {e}")
        st.markdown("---")
        st.markdown("#### Register New Strategy")
        with st.expander("Define a new strategy"):
            new_name = st.text_input("Strategy name", placeholder="my_strategy")
            new_tickers = st.text_input("Tickers (comma-separated)", value="NFLX,AAPL")
            new_fset = st.selectbox("Feature set", ["full", "momentum", "mean_reversion", "volume"])
            new_desc = st.text_area("Description", height=80)
            if st.button("Register Strategy") and new_name:
                try:
                    from src.strategy_registry import _MOMENTUM_FEATURES, _MEAN_REVERSION_FEATURES, _FULL_FEATURES
                    fset_map = {"full": _FULL_FEATURES, "momentum": _MOMENTUM_FEATURES, "mean_reversion": _MEAN_REVERSION_FEATURES, "volume": _MOMENTUM_FEATURES}
                    new_cfg = StrategyConfig(name=new_name, tickers=[t.strip() for t in new_tickers.split(",") if t.strip()], feature_set=fset_map[new_fset], description=new_desc)
                    _reg.register(new_cfg)
                    st.success(f"Strategy '{new_name}' registered!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Registration failed: {e}")
    except Exception as e:
        st.warning(f"Strategy Lab unavailable: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 12 — RESEARCH COPILOT
# ═══════════════════════════════════════════════════════════════════════════════
with tab_copilot:
    st.subheader("🔬 Research Copilot")
    st.caption("AI-powered research notes with HITL approval gate and citation-backed analysis.")
    try:
        from src.copilot import CopilotGraph
        from src.copilot.hitl_router import HITLRouter
        cop_col1, cop_col2, cop_col3 = st.columns(3)
        cop_ticker = cop_col1.text_input("Ticker", value=ticker, key="cop_ticker").upper()
        cop_price = cop_col2.number_input("Current Price ($)", value=float(df_source["Close"].iloc[-1]) if not df_source.empty else 650.0, key="cop_price")
        cop_query = cop_col3.text_input("Custom query (optional)", placeholder="Why is the model bullish?", key="cop_query")
        cop_pos_val = st.number_input("Proposed Position Value ($)", value=5000.0, step=1000.0, key="cop_pos")
        if st.button("Generate Research Note", type="primary", key="cop_btn"):
            with st.spinner("Running Copilot pipeline (retrieve -> tool -> write -> HITL)..."):
                try:
                    _copilot = CopilotGraph(ticker=cop_ticker)
                    cop_result = _copilot.run(ticker=cop_ticker, query=cop_query or None, position_value=cop_pos_val)
                    st.session_state["cop_result"] = cop_result
                except Exception as e:
                    st.error(f"Copilot error: {e}")
        if "cop_result" in st.session_state:
            cop_result = st.session_state["cop_result"]
            tool_state = cop_result.get("tool_state", {})
            hitl = cop_result.get("hitl_result", {})
            if hitl.get("requires_hitl"):
                st.warning(f"HITL Required - Position value exceeds threshold. Signal ID: {hitl.get('signal_id','?')}")
                hcol1, hcol2 = st.columns(2)
                if hcol1.button("Approve Signal", key="hitl_approve"):
                    HITLRouter().approve(hitl.get("signal_id",""), approver=st.session_state.get("username","analyst"))
                    st.success("Signal approved!")
                if hcol2.button("Reject Signal", key="hitl_reject"):
                    HITLRouter().reject(hitl.get("signal_id",""), reason="Manual rejection")
                    st.error("Signal rejected.")
            else:
                st.success("Signal auto-approved (below HITL threshold)")
            pred_ret = tool_state.get("pred_return", 0.0) or 0.0
            ci_l = tool_state.get("ci_lower", pred_ret - 0.5)
            ci_u = tool_state.get("ci_upper", pred_ret + 0.5)
            gc1, gc2 = st.columns(2)
            with gc1:
                fig_gauge = go.Figure(go.Indicator(mode="gauge+number", value=float(pred_ret)*100, title={"text": "Predicted Return (%)"}, gauge={"axis": {"range": [-3, 3]}, "bar": {"color": "#00c853" if pred_ret >= 0 else "#e50914"}, "steps": [{"range": [-3, 0], "color": "rgba(229,9,20,0.2)"}, {"range": [0, 3], "color": "rgba(0,200,83,0.2)"}]}))
                fig_gauge.update_layout(height=280, margin=dict(t=40,b=10,l=10,r=10))
                st.plotly_chart(fig_gauge, use_container_width=True)
            with gc2:
                fig_ci = go.Figure(go.Scatter(x=[float(pred_ret)*100], y=[0], mode="markers", marker=dict(size=16, color="#2196f3"), error_x=dict(type="data", symmetric=False, minus=abs(float(pred_ret)-float(ci_l))*100, plus=abs(float(ci_u)-float(pred_ret))*100, visible=True, color="#2196f3", thickness=4, width=10), name="Prediction +/- CI"))
                fig_ci.update_layout(title="Conformal Prediction Interval", xaxis_title="Return (%)", yaxis=dict(showticklabels=False), height=280, margin=dict(t=40,b=40,l=10,r=10))
                st.plotly_chart(fig_ci, use_container_width=True)
            rm1, rm2, rm3, rm4, rm5 = st.columns(5)
            rm1.metric("Signal", tool_state.get("signal", "N/A"))
            rm2.metric("Stop Loss", f"${tool_state.get('stop_loss', 0):.2f}" if tool_state.get("stop_loss") else "N/A")
            rm3.metric("Take Profit", f"${tool_state.get('take_profit', 0):.2f}" if tool_state.get("take_profit") else "N/A")
            rm4.metric("Kelly Frac", f"{tool_state.get('kelly_fraction', 0):.3f}")
            rm5.metric("Position ($)", f"${cop_pos_val:,.0f}")
            shap_drivers = tool_state.get("shap_drivers", [])
            if shap_drivers:
                st.markdown("#### Top Feature Drivers")
                feats = [d.get("feature","?") for d in shap_drivers]
                imps = [d.get("importance", 0) for d in shap_drivers]
                colors = ["#00c853" if d.get("direction","") == "positive" else "#e50914" for d in shap_drivers]
                fig_shap_c = go.Figure(go.Bar(x=imps, y=feats, orientation="h", marker_color=colors))
                fig_shap_c.update_layout(template="plotly_dark", height=250, title="SHAP Feature Importance", margin=dict(l=0,r=0,t=40,b=0))
                st.plotly_chart(fig_shap_c, use_container_width=True)
            st.markdown("### Research Note")
            st.markdown(cop_result.get("note", "No note generated."))
            docs = cop_result.get("retrieval", {}).get("documents", [])
            if docs:
                st.markdown(f"**Sources used:** {len(docs)}")
                titles = [d.get("metadata",{}).get("title","Doc")[:40] for d in docs]
                dists = [1-(d.get("distance",0.5) or 0.5) for d in docs]
                fig_src = go.Figure(go.Bar(x=dists, y=titles, orientation="h", marker_color="#ffd700"))
                fig_src.update_layout(template="plotly_dark", height=200, title="Source Relevance", margin=dict(l=0,r=0,t=30,b=0))
                st.plotly_chart(fig_src, use_container_width=True)
                with st.expander("View source documents"):
                    for i, d in enumerate(docs, 1):
                        st.markdown(f"**[{i}] {d.get('metadata',{}).get('title','?')}** - {d.get('metadata',{}).get('date','?')}")
                        st.markdown(f"> {d.get('text','')[:200]}...")
            method = "Groq LLaMA3" if "groq" in str(cop_result.get("note","")).lower() else "Rule-based fallback"
            st.caption(f"Generation method: {method} · Sources: {cop_result.get('sources_used',0)}")
    except Exception as e:
        st.warning(f"Research Copilot unavailable: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 13 — TRACK RECORD
# ═══════════════════════════════════════════════════════════════════════════════
with tab_track:
    st.subheader("📊 Track Record")
    st.caption("Verifiable, timestamped signal log with institutional-grade performance metrics.")
    try:
        from src.track_record import TrackRecord
        from src.strategy_registry import StrategyRegistry
        tr_strategies = StrategyRegistry().list_strategies()
        tr_sel = st.selectbox("Strategy", tr_strategies, key="tr_sel")
        tr = TrackRecord(strategy_name=tr_sel)
        tr_df = tr.load()
        if tr_df.empty:
            st.info("No track record data yet. Run: python src/track_record_seeder.py")
            if st.button("Seed Track Record Data"):
                with st.spinner("Seeding..."):
                    try:
                        from src.track_record_seeder import seed_all
                        seed_all()
                        st.success("Seeded!")
                        st.rerun()
                    except Exception as seed_e:
                        st.error(f"Seed error: {seed_e}")
        else:
            metrics = tr.compute_metrics()
            equity = tr.get_equity_curve()
            dd = tr.get_drawdown_series()
            k1,k2,k3,k4,k5,k6,k7 = st.columns(7)
            k1.metric("Signals", metrics["n_signals"])
            k2.metric("Win Rate", f"{metrics['win_rate_pct']:.1f}%")
            k3.metric("Sharpe", f"{metrics['sharpe']:.2f}")
            k4.metric("Sortino", f"{metrics['sortino']:.2f}")
            k5.metric("Calmar", f"{metrics['calmar']:.2f}")
            k6.metric("Max DD", f"{metrics['max_drawdown']*100:.1f}%")
            k7.metric("Profit Fac.", f"{metrics['profit_factor']:.2f}")
            fig_eq = go.Figure(go.Scatter(y=equity.values, mode="lines", line=dict(color="#00c853", width=2), fill="tozeroy", fillcolor="rgba(0,200,83,0.1)", name="Cumulative PnL %"))
            fig_eq.add_hline(y=0, line_color="white", opacity=0.3)
            fig_eq.update_layout(template="plotly_dark", height=300, title="Equity Curve (Cumulative PnL %)", margin=dict(l=0,r=0,t=40,b=0))
            st.plotly_chart(fig_eq, use_container_width=True)
            fig_dd = go.Figure(go.Scatter(y=dd.values*100, mode="lines", fill="tozeroy", fillcolor="rgba(229,9,20,0.3)", line=dict(color="#e50914", width=1), name="Drawdown %"))
            fig_dd.update_layout(template="plotly_dark", height=200, title="Drawdown (%)", margin=dict(l=0,r=0,t=40,b=0))
            st.plotly_chart(fig_dd, use_container_width=True)
            if "date" in tr_df.columns and len(tr_df) >= 10:
                try:
                    tr_df["date"] = pd.to_datetime(tr_df["date"], errors="coerce")
                    tr_df["month"] = tr_df["date"].dt.to_period("M").astype(str)
                    monthly = tr_df.groupby("month")["pnl_pct"].sum().reset_index()
                    fig_monthly = px.bar(monthly, x="month", y="pnl_pct", color="pnl_pct", color_continuous_scale=["#e50914","#333","#00c853"], title="Monthly PnL (%)", template="plotly_dark", height=250)
                    fig_monthly.update_layout(margin=dict(l=0,r=0,t=40,b=0))
                    st.plotly_chart(fig_monthly, use_container_width=True)
                except Exception:
                    pass
            st.markdown("#### Signal Log")
            st.dataframe(tr_df.tail(100).sort_values("date", ascending=False) if "date" in tr_df.columns else tr_df.tail(100), use_container_width=True)
    except Exception as e:
        st.warning(f"Track Record unavailable: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 14 — COMPLIANCE
# ═══════════════════════════════════════════════════════════════════════════════
with tab_compliance:
    st.subheader("SEBI Compliance Dashboard")
    st.caption("Algorithmic trading disclosure norms - SEBI circular SEBI/HO/MRD/MRD-DP/P/CIR/2021/1064.")
    try:
        from src.compliance_sebi import SEBIComplianceChecker
        if st.button("Run Compliance Check", type="primary", key="comp_btn"):
            with st.spinner("Running SEBI compliance checks..."):
                try:
                    checker = SEBIComplianceChecker()
                    report = checker.generate_report()
                    st.session_state["comp_report"] = report
                except Exception as e:
                    st.error(f"Compliance check failed: {e}")
        if "comp_report" in st.session_state:
            report = st.session_state["comp_report"]
            summary = report.get("summary", {})
            cc1,cc2,cc3,cc4,cc5 = st.columns(5)
            cc1.metric("Total Checks", summary.get("total", 0))
            cc2.metric("Pass", summary.get("pass", 0))
            cc3.metric("Fail", summary.get("fail", 0))
            cc4.metric("Warn", summary.get("warn", 0))
            cc5.metric("Compliance %", f"{summary.get('compliance_pct', 0):.1f}%")
            gs1, gs2 = st.columns(2)
            with gs1:
                ks = report.get("kill_switch_status", "UNKNOWN")
                st.metric("Kill Switch", ks, delta="ARMED" if ks=="ACTIVE" else "CHECK CONFIG", delta_color="normal" if ks=="ACTIVE" else "inverse")
            with gs2:
                otr = report.get("order_to_trade_ratio", 0.0)
                fig_otr = go.Figure(go.Indicator(mode="gauge+number", value=float(otr), title={"text": "Order-to-Trade Ratio"}, gauge={"axis": {"range": [0, 10]}, "bar": {"color": "#2196f3"}, "steps": [{"range": [0, 5], "color": "rgba(0,200,83,0.3)"}, {"range": [5, 10], "color": "rgba(229,9,20,0.3)"}], "threshold": {"line": {"color":"red","width":3}, "thickness":0.75, "value":5}}))
                fig_otr.update_layout(height=260, margin=dict(t=40,b=10,l=10,r=10))
                st.plotly_chart(fig_otr, use_container_width=True)
            st.markdown("#### Check Details")
            checks = report.get("checks", [])
            status_icon = {"PASS": "✅", "FAIL": "❌", "WARN": "⚠️"}
            for chk in checks:
                status = chk.get("status", "WARN")
                icon = status_icon.get(status, "❓")
                check_id  = chk.get("check_id", chk.get("id", "?"))
                desc      = chk.get("description", chk.get("requirement", chk.get("category", "")))
                with st.expander(f"{icon} [{check_id}] {desc[:70]}"):
                    st.markdown(f"**Status:** {status}")
                    st.markdown(f"**Evidence:** {chk.get('evidence','N/A')}")
                    if chk.get("remediation") and chk["remediation"] != "No action required.":
                        st.info(f"**Remediation:** {chk['remediation']}")
            st.download_button("Download Compliance Report (JSON)", data=json.dumps(report, indent=2), file_name=f"sebi_compliance_{report.get('report_id','report')}.json", mime="application/json")
    except Exception as e:
        st.warning(f"Compliance module unavailable: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 15 — OBSERVABILITY
# ═══════════════════════════════════════════════════════════════════════════════
with tab_obs:
    st.subheader("📡 Observability")
    st.caption("Agent trace logs, latency percentiles, and Prometheus-style metrics.")
    TRACE_LOG = Path(REPO_ROOT) / "logs" / "agent_traces.jsonl"
    try:
        if TRACE_LOG.exists():
            trace_lines = TRACE_LOG.read_text(encoding="utf-8").strip().splitlines()
            trace_records = []
            for ln in trace_lines:
                try:
                    trace_records.append(json.loads(ln))
                except Exception:
                    pass
            if trace_records:
                df_tr = pd.DataFrame(trace_records)
                df_tr["timestamp"] = pd.to_datetime(df_tr.get("timestamp",""), errors="coerce")
                df_tr["duration"] = pd.to_numeric(df_tr.get("duration_seconds", df_tr.get("duration",0)), errors="coerce").fillna(0)
                ob1,ob2,ob3 = st.columns(3)
                ob1.metric("Total Runs", len(df_tr))
                ob2.metric("Error Rate", f"{(df_tr.get('error','').notna() & (df_tr.get('error','')!='')).mean()*100:.1f}%" if "error" in df_tr else "0%")
                ob3.metric("Avg Latency", f"{df_tr['duration'].mean():.2f}s")
                if "agent_name" in df_tr.columns:
                    freq = df_tr["agent_name"].value_counts().reset_index()
                    freq.columns = ["agent","count"]
                    fig_freq = go.Figure(go.Bar(x=freq["agent"], y=freq["count"], marker_color="#2196f3"))
                    fig_freq.update_layout(template="plotly_dark", height=280, title="Agent Run Frequency", margin=dict(l=0,r=0,t=40,b=0))
                    st.plotly_chart(fig_freq, use_container_width=True)
                if df_tr["duration"].max() > 0:
                    fig_lat = go.Figure()
                    if "agent_name" in df_tr.columns:
                        for agent_n in df_tr["agent_name"].unique():
                            subset = df_tr[df_tr["agent_name"]==agent_n]["duration"]
                            fig_lat.add_trace(go.Box(y=subset, name=agent_n, boxpoints="outliers"))
                    else:
                        fig_lat.add_trace(go.Box(y=df_tr["duration"], name="All agents"))
                    fig_lat.update_layout(template="plotly_dark", height=300, title="Latency Distribution (seconds)", margin=dict(l=0,r=0,t=40,b=0))
                    st.plotly_chart(fig_lat, use_container_width=True)
                total_runs = len(df_tr)
                agent_counts = df_tr["agent_name"].value_counts().to_dict() if "agent_name" in df_tr else {}
                p99 = float(df_tr["duration"].quantile(0.99)) if not df_tr["duration"].empty else 0
                p50 = float(df_tr["duration"].quantile(0.50)) if not df_tr["duration"].empty else 0
                prom_lines = [
                    "# HELP alphaengine_copilot_runs_total Total agent runs",
                    f"alphaengine_copilot_runs_total {total_runs}",
                    "# HELP alphaengine_copilot_latency_p50_seconds p50 latency",
                    f"alphaengine_copilot_latency_p50_seconds {p50:.4f}",
                    "# HELP alphaengine_copilot_latency_p99_seconds p99 latency",
                    f"alphaengine_copilot_latency_p99_seconds {p99:.4f}",
                ]
                for agent_n2, count in agent_counts.items():
                    prom_lines.append(f'alphaengine_copilot_agent_runs_total{{agent="{agent_n2}"}} {count}')
                st.markdown("#### Prometheus Metrics")
                st.code("\n".join(prom_lines), language="text")
                st.markdown("#### Recent Traces")
                display_cols = [c for c in ["timestamp","agent_name","ticker","duration","error"] if c in df_tr.columns]
                st.dataframe(df_tr[display_cols].tail(20).iloc[::-1], use_container_width=True)
            else:
                st.info("Trace log is empty. Generate traces via the AI Narrative or Research Copilot tabs.")
        else:
            st.info("No trace log found at logs/agent_traces.jsonl.")
    except Exception as e:
        st.warning(f"Observability tab error: {e}")

