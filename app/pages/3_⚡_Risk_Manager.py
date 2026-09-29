"""
Page 3 — Risk Manager
Pure presentation layer: sends params to /risk/position and /risk/matrix → renders results.
Zero src.* imports.
"""
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import app.api_client as api

st.set_page_config(page_title="Risk Manager · Alpha Engine",
                   page_icon="⚡", layout="wide")

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
.risk-card {
    background:linear-gradient(135deg,#0d1117,#1c2333);
    border:1px solid #30363d;border-radius:12px;padding:20px;margin:8px 0;
}
.halt-badge {
    background:rgba(248,81,73,0.15);color:#f85149;
    border:1px solid #f85149;border-radius:8px;
    padding:8px 16px;font-weight:600;
}
.ok-badge {
    background:rgba(63,185,80,0.15);color:#3fb950;
    border:1px solid #3fb950;border-radius:8px;
    padding:8px 16px;font-weight:600;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ Risk Manager")
    st.markdown("---")
    st.markdown("**Position Parameters**")
    ticker    = st.selectbox("Ticker", ["NFLX","AAPL","TSLA","GOOGL","MSFT","AMZN","META"])
    pred_ret  = st.number_input("Predicted Return (%)", value=1.25, step=0.1, format="%.4f")
    last_price= st.number_input("Current Price ($)", value=650.0, step=1.0, format="%.2f")
    atr       = st.number_input("14-period ATR ($)", value=15.0, step=0.5, format="%.2f")
    st.markdown("---")
    st.markdown("**Portfolio Settings**")
    portfolio = st.number_input("Portfolio Value ($)", value=100_000.0, step=1000.0, format="%.0f")
    win_rate  = st.slider("Historical Win Rate", 0.30, 0.80, 0.52, 0.01)
    avg_win   = st.number_input("Avg Win (%)", value=1.5, step=0.1, format="%.2f")
    avg_loss  = st.number_input("Avg Loss (%)", value=1.0, step=0.1, format="%.2f")
    st.markdown("---")
    st.markdown("**Risk Limits**")
    max_pos   = st.slider("Max Position (%)", 1, 20, 5) / 100
    max_dd    = st.slider("Max Drawdown Halt (%)", 5, 30, 10) / 100
    st.markdown("---")
    calc_risk = st.button("⚡ Calculate Risk", use_container_width=True, key="calc_risk_btn")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("## ⚡ Risk Manager")
st.markdown("ATR-based stop-loss · Kelly position sizing · Drawdown circuit breaker · R:R ratio enforcement")

if not calc_risk:
    st.info("👈 Set your parameters in the sidebar and click **Calculate Risk** to compute position sizing.")
    st.stop()

# ── Fetch Position ────────────────────────────────────────────────────────────
payload = {
    "ticker":           ticker,
    "pred_return_pct":  pred_ret,
    "last_price":       last_price,
    "atr":              atr,
    "portfolio_value":  portfolio,
    "win_rate":         win_rate,
    "avg_win_pct":      avg_win,
    "avg_loss_pct":     avg_loss,
    "max_position_pct": max_pos,
    "max_drawdown_halt":max_dd,
}

with st.spinner("Computing position sizing…"):
    pos  = api.compute_risk_position(payload)
    mat  = api.compute_risk_matrix(payload)

if "error" in pos:
    st.error(f"Risk calc failed: {pos.get('detail', pos['error'])}")
    st.stop()

# ── Signal & Circuit Breaker ──────────────────────────────────────────────────
signal   = pos.get("signal","HOLD")
is_halt  = mat.get("is_halted", False)
cur_dd   = mat.get("current_drawdown","0%")

badge_html = (
    '<div class="halt-badge">🛑 CIRCUIT BREAKER TRIGGERED</div>' if is_halt
    else f'<div class="ok-badge">✅ Trading Active — Drawdown: {cur_dd}</div>'
)
st.markdown(badge_html, unsafe_allow_html=True)

sig_color = {"BUY":"#3fb950","HOLD":"#d29922","SELL":"#f85149","HALT":"#f85149"}
_sig_col  = sig_color.get(signal, "#c9d1d9")
st.markdown(
    f"<h2 style='color:{_sig_col};margin-top:12px;'>"
    f"Signal: {signal}</h2>",
    unsafe_allow_html=True
)

# ── Position KPIs ─────────────────────────────────────────────────────────────
st.markdown("### 📐 Position Sizing")
k1,k2,k3,k4,k5 = st.columns(5)
k1.metric("Shares",          pos.get("shares",0))
k2.metric("Position Value",  f"${pos.get('position_value',0):,.2f}")
k3.metric("Risk Per Trade",  f"${pos.get('risk_per_trade',0):,.2f}")
k4.metric("Risk %",          f"{pos.get('risk_pct',0):.3f}%")
k5.metric("Kelly Fraction",  f"{pos.get('kelly_fraction',0):.4f}")

# ── Price Levels ──────────────────────────────────────────────────────────────
st.markdown("### 🎯 Price Levels")
p1,p2,p3,p4 = st.columns(4)
p1.metric("Entry Price",   f"${pos.get('entry_price',0):,.2f}")
p2.metric("Stop Loss",     f"${pos.get('stop_loss',0):,.2f}",
          f"-{(last_price - pos.get('stop_loss',0)):,.2f}")
p3.metric("Take Profit",   f"${pos.get('take_profit',0):,.2f}",
          f"+{(pos.get('take_profit',0) - last_price):,.2f}")
p4.metric("R:R Ratio",     mat.get("risk_reward_ratio",0))

# ── Price Level Waterfall ─────────────────────────────────────────────────────
entry  = pos.get("entry_price",last_price)
sl     = pos.get("stop_loss",0)
tp     = pos.get("take_profit",0)

fig_levels = go.Figure()
y_all = sorted([sl, entry, tp])
fig_levels.add_shape(type="line",x0=0,x1=1,y0=sl,y1=sl,
                     line=dict(color="#f85149",width=2,dash="dash"),xref="paper")
fig_levels.add_shape(type="line",x0=0,x1=1,y0=entry,y1=entry,
                     line=dict(color="#58a6ff",width=2),xref="paper")
fig_levels.add_shape(type="line",x0=0,x1=1,y0=tp,y1=tp,
                     line=dict(color="#3fb950",width=2,dash="dash"),xref="paper")

fig_levels.add_annotation(x=0.02,y=sl,xref="paper",text=f"Stop ${sl:,.2f}",
                           font=dict(color="#f85149"),showarrow=False,xanchor="left")
fig_levels.add_annotation(x=0.02,y=entry,xref="paper",text=f"Entry ${entry:,.2f}",
                           font=dict(color="#58a6ff"),showarrow=False,xanchor="left")
fig_levels.add_annotation(x=0.02,y=tp,xref="paper",text=f"Target ${tp:,.2f}",
                           font=dict(color="#3fb950"),showarrow=False,xanchor="left")

fig_levels.add_hrect(y0=sl,y1=entry,fillcolor="rgba(248,81,73,0.07)",line_width=0)
fig_levels.add_hrect(y0=entry,y1=tp,fillcolor="rgba(63,185,80,0.07)",line_width=0)

fig_levels.update_layout(
    height=300, paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(22,27,34,0.8)",
    font=dict(family="Inter",color="#c9d1d9"),
    margin=dict(l=0,r=0,t=10,b=0),
    xaxis=dict(showticklabels=False,showgrid=False),
    yaxis=dict(title="Price ($)",gridcolor="rgba(48,54,61,0.5)"),
    showlegend=False,
)
st.plotly_chart(fig_levels, use_container_width=True)

# ── Full Risk Matrix Table ────────────────────────────────────────────────────
st.markdown("### 📊 Full Risk Matrix")
mat_display = {k: str(v) for k,v in mat.items()}
df_mat = pd.DataFrame(list(mat_display.items()), columns=["Parameter","Value"])
st.dataframe(df_mat, use_container_width=True, hide_index=True)

if pos.get("notes"):
    st.caption(f"ℹ️ {pos['notes']}")

# ── Execute Trade ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🚀 Execute Trade")

col_broker, col_type, col_limit = st.columns(3)
with col_broker: broker = st.selectbox("Broker", ["paper","alpaca"], index=0, key="exec_broker")
with col_type:   otype  = st.selectbox("Order Type", ["market","limit"], index=0, key="exec_type")
with col_limit:
    limit_px = st.number_input("Limit Price ($)", value=float(entry), format="%.2f",
                                disabled=(otype=="market"), key="exec_limit")

shares_to_exec = pos.get("shares", 0)
exec_btn = st.button(
    f"⚡ Execute {signal} — {shares_to_exec} shares via {broker.upper()}",
    disabled=(signal in ("HOLD","HALT") or shares_to_exec == 0),
    key="exec_trade_btn",
)

if exec_btn:
    exec_payload = {
        "ticker":      ticker,
        "shares":      shares_to_exec,
        "side":        "buy" if signal=="BUY" else "sell",
        "order_type":  otype,
        "limit_price": limit_px if otype=="limit" else None,
        "broker":      broker,
    }
    with st.spinner(f"Submitting order to {broker.upper()}…"):
        exec_result = api.execute_trade(exec_payload)

    if "error" in exec_result:
        st.error(f"Execution failed: {exec_result.get('detail',exec_result['error'])}")
    else:
        st.success(f"✅ Order {exec_result.get('status','submitted')}: "
                   f"{exec_result.get('side','').upper()} {exec_result.get('shares')} "
                   f"{exec_result.get('ticker')} via {exec_result.get('broker','').upper()}")
        if exec_result.get("order_id"):
            st.caption(f"Order ID: `{exec_result['order_id']}`")
