import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os, sys, json

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
    page_title="NFLX Alpha Engine",
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
st.markdown('<p class="hero-title">Alpha Engine</p>', unsafe_allow_html=True)
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
    "🏗 Architecture",
])
tab_market, tab_pred, tab_bt, tab_paper, tab_sent, tab_risk, tab_drift, tab_shap, tab_arch = tabs

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
            rows  = []
            for item in news:
                ts    = pd.Timestamp(item.get("providerPublishTime", 0), unit="s")
                title = item.get("title", "")
                score = sia.polarity_scores(title)["compound"]
                rows.append({"date": ts, "title": title, "score": score,
                              "sentiment": "Positive" if score > 0.05
                              else ("Negative" if score < -0.05 else "Neutral")})
            return pd.DataFrame(rows)
        except Exception as e:
            return pd.DataFrame(columns=["date","title","score","sentiment"])

    df_sent = _get_sentiment()

    if df_sent.empty:
        st.warning("Sentiment data unavailable. Install vaderSentiment: `pip install vaderSentiment`")
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
# TAB 5 — RISK MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════
with tab_risk:
    st.subheader("Risk Management Tools")

    df_r = df_feat[["Close","Return","Volatility","ATR_Norm"]].dropna().copy() \
           if "Return" in df_feat.columns else None

    if df_r is None:
        st.warning("Feature data not available.")
    else:
        ret = df_r["Return"].dropna() / 100

        # ── VaR & CVaR ────────────────────────────────────────────────────────
        conf = st.slider("Confidence Level", 0.90, 0.99, 0.95, 0.01)
        var  = float(np.percentile(ret, (1 - conf) * 100))
        cvar = float(ret[ret <= var].mean())

        c1, c2, c3, c4 = st.columns(4)
        c1.metric(f"VaR ({conf:.0%})",  f"{var:.3%}", help="Value at Risk")
        c2.metric(f"CVaR ({conf:.0%})", f"{cvar:.3%}", help="Conditional VaR (Expected Shortfall)")
        c3.metric("Ann. Volatility",    f"{ret.std() * np.sqrt(252):.2%}")
        c4.metric("Sharpe (hist)",
                  f"{(ret.mean() / ret.std() * np.sqrt(252)):.3f}" if ret.std() > 0 else "N/A")

        # ── Return distribution with VaR ──────────────────────────────────────
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(x=ret * 100, nbinsx=120, name="Returns",
                                         marker_color="#2196f3", opacity=0.7))
        fig_dist.add_vline(x=var * 100, line_dash="dash", line_color="#e50914",
                            annotation_text=f"VaR {conf:.0%}", annotation_position="top right")
        fig_dist.add_vline(x=cvar * 100, line_dash="dash", line_color="#ff9800",
                            annotation_text=f"CVaR {conf:.0%}", annotation_position="top left")
        fig_dist.update_layout(template="plotly_dark", height=350,
                                title="Return Distribution with VaR / CVaR",
                                xaxis_title="Daily Return (%)",
                                margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig_dist, use_container_width=True)

        # ── Volatility surface (rolling vol over time) ────────────────────────
        vol_20  = ret.rolling(20).std()  * np.sqrt(252) * 100
        vol_60  = ret.rolling(60).std()  * np.sqrt(252) * 100
        vol_120 = ret.rolling(120).std() * np.sqrt(252) * 100

        fig_vol = go.Figure()
        fig_vol.add_trace(go.Scatter(x=df_r.index, y=vol_20,  name="20d Vol",
                                      line=dict(color="#e50914", width=1.5)))
        fig_vol.add_trace(go.Scatter(x=df_r.index, y=vol_60,  name="60d Vol",
                                      line=dict(color="#ffd700", width=1.5)))
        fig_vol.add_trace(go.Scatter(x=df_r.index, y=vol_120, name="120d Vol",
                                      line=dict(color="#00bcd4", width=1.5)))
        fig_vol.update_layout(template="plotly_dark", height=350,
                               title="Annualised Volatility Surface",
                               yaxis_title="Volatility (%)",
                               margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig_vol, use_container_width=True)

        # ── Correlation matrix ────────────────────────────────────────────────
        corr_cols = ["Return","RSI","MACD_Norm","BB_Pct","ATR_Norm",
                     "Volatility","Stoch_K","Williams_R","CCI","Momentum5"]
        avail = [c for c in corr_cols if c in df_feat.columns]
        corr  = df_feat[avail].dropna().corr()

        fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                              zmin=-1, zmax=1, title="Feature Correlation Matrix",
                              template="plotly_dark", height=500)
        fig_corr.update_layout(margin=dict(l=0,r=0,t=40,b=0))
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
# TAB 9 — ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════════════
with tab_arch:
    st.subheader("System Architecture & Edge")

    st.markdown("""
### The Edge

Most stock prediction projects predict **price** (trivially correlated with itself).
This project predicts **next-day return (%)** — a stationary, genuinely hard target.

The directional accuracy metric (>52% = real signal) is what matters, not R².

---

### Pipeline Architecture

```
Yahoo Finance (live)
        │
        ▼
  data_loader.py  ──── yfinance / CSV fallback
        │
        ▼
  preprocessing.py ─── date parsing, dedup, sort
        │
        ▼
  feature_engineering.py
        │  51 features: lags, RSI, MACD, BB, ATR,
        │  Stochastic, Williams %R, CCI, OBV,
        │  EMA cross, momentum, regime probs
        ▼
  regime_detection.py ── Gaussian HMM (3 states)
        │                 Bull / Sideways / Bear
        ▼
  modeling.py
        │  ManualStackingRegressor
        │  Level-0: XGB + LGBM + RF + ExtraTrees
        │  Level-1: Ridge meta-learner (OOF)
        │  RobustScaler inside fit/predict
        ▼
  uncertainty.py ── Conformal prediction intervals
        │            90% coverage guarantee
        ▼
  backtest.py ── Binary + Kelly strategies
        │         Sharpe / Sortino / Calmar
        ▼
  drift.py ── PSI + KS test on all features
        │
        ▼
  FastAPI (/predict, /health, /features)
        │
        ▼
  Streamlit Cloud (this app)
```

---

### Why This Stack

| Choice | Reason |
|---|---|
| XGB + LGBM + RF + ET stacking | Diversity reduces variance; OOF prevents leakage |
| Return target (not price) | Stationary; avoids spurious R² from autocorrelation |
| Walk-forward CV | Only valid CV for time-series; no future leakage |
| Conformal prediction | Calibrated intervals with mathematical coverage guarantee |
| HMM regime detection | Market dynamics differ across regimes; separate signal |
| Kelly sizing | Bet proportional to edge; maximises long-run growth |
| VADER sentiment | Orthogonal signal — price data alone misses news events |

---

### Limitations (honest)

- No earnings surprise signal (biggest driver of NFLX moves)
- Technical indicators are correlated — ~10 independent signals, not 51
- Model trained on 2002–2026; regime changes may not generalise
- 15-min delayed Yahoo Finance data — not suitable for intraday

---

### Tests Badge
[![Tests](https://github.com/SumedhPatil1507/netflix-stock-prediction/actions/workflows/test.yml/badge.svg)](https://github.com/SumedhPatil1507/netflix-stock-prediction/actions)

Run locally: `pytest tests/ -v`
    """)
