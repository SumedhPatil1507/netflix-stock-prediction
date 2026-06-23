"""
Page 2 — Prediction Engine
Pure presentation layer: submits OHLCV rows to /predict → renders results.
Zero src.* imports.
"""
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import app.api_client as api

st.set_page_config(page_title="Prediction Engine · Alpha Engine",
                   page_icon="🤖", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
[data-testid="stSidebar"] {
    background:linear-gradient(180deg,#0d1117 0%,#161b22 100%);
    border-right:1px solid #30363d;
}
[data-testid="stSidebar"] * { color:#c9d1d9 !important; }
[data-testid="metric-container"] {
    background:linear-gradient(135deg,#161b22,#1c2333);
    border:1px solid #30363d;border-radius:12px;padding:16px;
}
.signal-buy  { color:#3fb950;font-weight:700;font-size:2rem; }
.signal-hold { color:#d29922;font-weight:700;font-size:2rem; }
.signal-sell { color:#f85149;font-weight:700;font-size:2rem; }
.ci-box {
    background:linear-gradient(135deg,#0d1117,#1c2333);
    border:1px solid #30363d;border-radius:12px;
    padding:20px;margin:12px 0;
}
.predict-btn > button {
    background:linear-gradient(135deg,#1a237e,#3949ab) !important;
    color:#fff !important;border:none !important;
    border-radius:8px !important;font-weight:600 !important;
    padding:10px 28px !important;font-size:1rem !important;
    transition:all 0.2s ease !important;
}
.predict-btn > button:hover {
    background:linear-gradient(135deg,#283593,#5c6bc0) !important;
    transform:translateY(-1px) !important;
    box-shadow:0 4px 12px rgba(57,73,171,0.4) !important;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 Prediction Engine")
    st.markdown("---")
    ticker = st.selectbox("Ticker", ["NFLX","AAPL","TSLA","GOOGL","MSFT","AMZN","META"], index=0)
    n_rows = st.slider("History rows (n)", min_value=10, max_value=50, value=15, step=1)
    st.markdown("---")
    st.info("The model needs at least **10 OHLCV rows** (most recent last) to predict next-day return.")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("## 🤖 Prediction Engine")
st.markdown("Edit the OHLCV table below (or auto-fill from live data), then click **Run Prediction**.")

# ── Auto-fill from live data ──────────────────────────────────────────────────
col_fill, col_clear = st.columns([1, 5])
with col_fill:
    auto_fill = st.button("📥 Auto-fill from Live Data")

if auto_fill or "live_df" not in st.session_state:
    with st.spinner(f"Fetching last {n_rows} bars for {ticker}…"):
        raw = api.get_live_input(ticker)
    # raw is dict of lists
    st.session_state["live_df"] = pd.DataFrame({
        "Open":   raw.get("Open",   [600.0]*n_rows)[:n_rows],
        "High":   raw.get("High",   [610.0]*n_rows)[:n_rows],
        "Low":    raw.get("Low",    [595.0]*n_rows)[:n_rows],
        "Close":  raw.get("Close",  [605.0]*n_rows)[:n_rows],
        "Volume": raw.get("Volume", [5_000_000]*n_rows)[:n_rows],
    })

# ── Editable OHLCV Table ──────────────────────────────────────────────────────
st.markdown("#### 📋 Input OHLCV (editable)")
edited_df = st.data_editor(
    st.session_state["live_df"],
    use_container_width=True,
    num_rows="fixed",
    key="ohlcv_editor",
    column_config={
        "Open":   st.column_config.NumberColumn("Open ($)",   format="$%.2f"),
        "High":   st.column_config.NumberColumn("High ($)",   format="$%.2f"),
        "Low":    st.column_config.NumberColumn("Low ($)",    format="$%.2f"),
        "Close":  st.column_config.NumberColumn("Close ($)",  format="$%.2f"),
        "Volume": st.column_config.NumberColumn("Volume",     format="%d"),
    },
)

# ── Predict Button ────────────────────────────────────────────────────────────
st.markdown("")
_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    st.markdown('<div class="predict-btn">', unsafe_allow_html=True)
    run_pred = st.button("🚀 Run Prediction", use_container_width=True, key="run_pred_btn")
    st.markdown('</div>', unsafe_allow_html=True)

if run_pred:
    if len(edited_df) < 10:
        st.error("Need at least 10 rows. Add more data or increase history rows.")
        st.stop()

    rows_payload = edited_df[["Open","High","Low","Close","Volume"]].rename(columns={
        "Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"
    }).to_dict("records")

    with st.spinner("🧠 Running stacking ensemble prediction…"):
        result = api.predict(rows_payload, ticker)

    if "error" in result:
        st.error(f"Prediction failed: {result.get('detail', result['error'])}")
    else:
        st.markdown("---")
        st.markdown("### 📊 Prediction Results")

        pred_ret   = result["predicted_return_pct"]
        pred_price = result["predicted_next_close"]
        last_close = result["last_close"]
        signal     = result["signal"]
        ci         = result.get("confidence_interval")

        # Signal badge
        signal_class = f"signal-{signal.lower()}"
        signal_emoji = "🟢 BUY" if signal=="BUY" else "🟡 HOLD" if signal=="HOLD" else "🔴 SELL"
        st.markdown(f'<div class="{signal_class}">{signal_emoji}</div>', unsafe_allow_html=True)
        st.markdown("")

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Predicted Return",     f"{pred_ret:+.4f}%",
                  "Bullish" if pred_ret>0 else "Bearish")
        r2.metric("Predicted Next Close", f"${pred_price:,.2f}",
                  f"{pred_price - last_close:+.2f}")
        r3.metric("Last Close",           f"${last_close:,.2f}")
        r4.metric("Signal",               signal)

        # Confidence Interval
        if ci:
            st.markdown("#### 📐 90% Conformal Prediction Interval")
            lo_ret  = ci["lower_return_pct"]
            hi_ret  = ci["upper_return_pct"]
            lo_px   = ci["lower_price"]
            hi_px   = ci["upper_price"]
            cov     = ci.get("coverage","90%")

            fig_ci = go.Figure()
            # CI bar
            fig_ci.add_trace(go.Bar(
                x=["Lower","Point Estimate","Upper"],
                y=[lo_px, pred_price, hi_px],
                marker_color=["rgba(248,81,73,0.7)","rgba(88,166,255,0.9)","rgba(63,185,80,0.7)"],
                text=[f"${lo_px:,.2f}",f"${pred_price:,.2f}",f"${hi_px:,.2f}"],
                textposition="outside",
                textfont=dict(color="#c9d1d9"),
            ))
            fig_ci.add_hline(y=last_close, line=dict(color="#d29922",dash="dash",width=1.5),
                             annotation_text="Last Close", annotation_font_color="#d29922")
            fig_ci.update_layout(
                height=320, paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(22,27,34,0.8)",
                font=dict(family="Inter",color="#c9d1d9"),
                showlegend=False,
                margin=dict(l=0,r=0,t=10,b=0),
                yaxis=dict(title="Price ($)",gridcolor="rgba(48,54,61,0.5)"),
            )
            st.plotly_chart(fig_ci, use_container_width=True)

            ci1,ci2,ci3,ci4 = st.columns(4)
            ci1.metric("Lower Return",  f"{lo_ret:+.4f}%")
            ci2.metric("Upper Return",  f"{hi_ret:+.4f}%")
            ci3.metric("Lower Price",   f"${lo_px:,.2f}")
            ci4.metric("Upper Price",   f"${hi_px:,.2f}")
            st.caption(f"Coverage guarantee: {cov} of true values fall inside this interval.")
        else:
            st.info("No conformal interval attached — retrain model with `python main.py` to enable.")

        # Input waterfall chart
        st.markdown("#### 📈 Input Price History")
        fig_hist = go.Figure(go.Scatter(
            x=list(range(len(edited_df))), y=edited_df["Close"].tolist(),
            mode="lines+markers",
            line=dict(color="#58a6ff",width=2),
            marker=dict(size=6,color="#58a6ff"),
            fill="tozeroy", fillcolor="rgba(88,166,255,0.08)",
            name="Close",
        ))
        fig_hist.add_trace(go.Scatter(
            x=[len(edited_df)], y=[pred_price],
            mode="markers",
            marker=dict(size=14, color="#3fb950" if pred_ret>0 else "#f85149",
                        symbol="star", line=dict(color="#fff",width=1)),
            name="Prediction",
        ))
        fig_hist.update_layout(
            height=280, paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(22,27,34,0.8)",
            font=dict(family="Inter",color="#c9d1d9"),
            margin=dict(l=0,r=0,t=0,b=0),
            yaxis=dict(gridcolor="rgba(48,54,61,0.5)"),
        )
        st.plotly_chart(fig_hist, use_container_width=True)
else:
    st.info("👆 Click **Auto-fill from Live Data** then **Run Prediction** to get started.")
