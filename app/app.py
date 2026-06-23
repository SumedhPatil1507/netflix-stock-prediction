"""
Alpha Engine — Streamlit Dashboard (Pure Presentation Layer)

NO src.* imports. NO direct model/broker calls.
All data comes from the FastAPI backend via app/api_client.py.
Heavy tasks use async job polling (Celery + Redis via FastAPI).
"""
import time
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(__file__))
from api_client import (
    health, get_ohlcv, get_indicators, get_live_input,
    predict, get_model_info, get_sentiment, get_drift_report,
    get_feature_importance, compute_risk_position, compute_risk_matrix,
    execute_trade, submit_backtest, submit_paper_trade,
    submit_drift_check, get_task_result, poll_until_done,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Alpha Engine", page_icon="📈",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.hero-title{font-size:2.2rem;font-weight:800;
  background:linear-gradient(90deg,#e50914,#ff6b6b);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent}
</style>""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Alpha Engine")
    st.markdown("---")
    ticker = st.text_input("Ticker", value="NFLX",
                            help="Any Yahoo Finance ticker").upper()
    period = st.selectbox("Chart period", ["6mo","1y","2y","5y","max"], index=2)
    st.markdown("---")
    api_status = health()
    if "error" in api_status:
        st.error(f"API offline: {api_status['error']}")
        st.info("Start backend: `make api`")
    else:
        st.success(f"API v{api_status.get('version','?')} online")
    st.markdown("---")
    st.markdown("**Model:** XGB+LGBM+RF+ET → Ridge")
    st.markdown("**Target:** Next-day return (%)")
    st.markdown("[GitHub](https://github.com/SumedhPatil1507/netflix-stock-prediction)")

# ── Hero & KPIs ───────────────────────────────────────────────────────────────
st.markdown('<p class="hero-title">Alpha Engine</p>', unsafe_allow_html=True)
st.caption(f"Presentation layer — all data via FastAPI backend · {ticker}")

@st.cache_data(ttl=300)
def _model_info(): return get_model_info()

info = _model_info()
if "latest_metrics" in info:
    m  = info["latest_metrics"]
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Dir Acc",    f"{m.get('Dir_Acc',0):.1f}%")
    c2.metric("CV R²",      f"{m.get('CV_R2',0):.4f}")
    c3.metric("CP Coverage",f"{m.get('CP_Coverage',0):.1%}")
    c4.metric("CP Width",   f"{m.get('CP_Width',0):.2f}%")
    c5.metric("CV RMSE",    f"{m.get('CV_RMSE',0):.4f}")

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "🕯 Market","🔮 Predict","📈 Backtest","📋 Paper Trade",
    "🧠 Sentiment","⚠️ Risk","🔬 Drift","🔍 Explainability","🏗 Architecture"
])
tab_mkt,tab_pred,tab_bt,tab_paper,tab_sent,tab_risk,tab_drift,tab_xp,tab_arch = tabs

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — MARKET OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
with tab_mkt:
    st.subheader(f"Market Overview — {ticker}")

    @st.cache_data(ttl=7200)
    def _ohlcv(t, p): return get_ohlcv(t, p)

    data = _ohlcv(ticker, period)
    if "error" in data:
        st.error(data["error"])
    else:
        dates = data["dates"]
        fig = make_subplots(rows=2,cols=1,shared_xaxes=True,
                            row_heights=[0.75,0.25],vertical_spacing=0.03)
        fig.add_trace(go.Candlestick(x=dates,open=data["open"],high=data["high"],
            low=data["low"],close=data["close"],name=ticker,
            increasing_line_color="#00c853",decreasing_line_color="#e50914"),row=1,col=1)
        for w,col in [(20,"#ffd700"),(50,"#00bcd4"),(200,"#ff9800")]:
            closes = pd.Series(data["close"])
            ma = closes.rolling(w).mean()
            fig.add_trace(go.Scatter(x=dates,y=ma,name=f"MA{w}",
                line=dict(color=col,width=1)),row=1,col=1)
        colors = ["#00c853" if c>=o else "#e50914"
                  for c,o in zip(data["close"],data["open"])]
        fig.add_trace(go.Bar(x=dates,y=data["volume"],name="Vol",
            marker_color=colors,opacity=0.6),row=2,col=1)
        fig.update_layout(template="plotly_dark",height=550,
            xaxis_rangeslider_visible=False,margin=dict(l=0,r=0,t=20,b=0))
        st.plotly_chart(fig,use_container_width=True)

    @st.cache_data(ttl=7200)
    def _ind(t): return get_indicators(t)

    ind = _ind(ticker)
    if "error" not in ind:
        col1,col2 = st.columns(2)
        with col1:
            fig_r = go.Figure()
            fig_r.add_trace(go.Scatter(x=ind["dates"],y=ind["rsi"],
                name="RSI",line=dict(color="#9c27b0",width=1.5)))
            fig_r.add_hline(y=70,line_dash="dash",line_color="red",opacity=0.6)
            fig_r.add_hline(y=30,line_dash="dash",line_color="green",opacity=0.6)
            fig_r.update_layout(template="plotly_dark",height=240,
                title="RSI (14)",margin=dict(l=0,r=0,t=30,b=0))
            st.plotly_chart(fig_r,use_container_width=True)
        with col2:
            fig_m = go.Figure()
            fig_m.add_trace(go.Scatter(x=ind["dates"],y=ind["macd"],
                name="MACD",line=dict(color="#2196f3",width=1.5)))
            fig_m.add_trace(go.Scatter(x=ind["dates"],y=ind["macd_sig"],
                name="Signal",line=dict(color="#ff9800",width=1.5)))
            colors_h = ["#00c853" if v>=0 else "#e50914" for v in ind["macd_hist"]]
            fig_m.add_trace(go.Bar(x=ind["dates"],y=ind["macd_hist"],
                marker_color=colors_h,opacity=0.5,name="Hist"))
            fig_m.update_layout(template="plotly_dark",height=240,
                title="MACD",margin=dict(l=0,r=0,t=30,b=0))
            st.plotly_chart(fig_m,use_container_width=True)

        fig_bb = go.Figure()
        fig_bb.add_trace(go.Scatter(x=ind["dates"],y=ind["bb_upper"],name="Upper",
            line=dict(color="red",dash="dash",width=1)))
        fig_bb.add_trace(go.Scatter(x=ind["dates"],y=ind["bb_lower"],name="Lower",
            line=dict(color="green",dash="dash",width=1),
            fill="tonexty",fillcolor="rgba(128,128,128,0.1)"))
        fig_bb.add_trace(go.Scatter(x=ind["dates"],y=ind["close"],name="Close",
            line=dict(color="white",width=1.5)))
        fig_bb.update_layout(template="plotly_dark",height=300,
            title="Bollinger Bands",margin=dict(l=0,r=0,t=30,b=0))
        st.plotly_chart(fig_bb,use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PREDICT
# ═══════════════════════════════════════════════════════════════════════════════
with tab_pred:
    st.subheader("Next-Day Return Prediction")
    st.caption("Data auto-loaded from backend. No local model imports.")

    @st.cache_data(ttl=7200, show_spinner=False)
    def _live(t): return get_live_input(t)

    raw = _live(ticker)
    if "error" in raw:
        raw = {"Open":[600]*10,"High":[610]*10,"Low":[595]*10,
               "Close":[605]*10,"Volume":[5_000_000]*10}

    edited = st.data_editor(pd.DataFrame(raw), num_rows="fixed",
                             use_container_width=True, key="pred_input")

    if st.button("Predict Next Close", type="primary"):
        rows = edited.rename(columns=str.lower).to_dict("records")
        result = predict(rows, ticker)
        if "error" in result:
            st.error(result["error"])
        else:
            last = result["last_close"]
            pred = result["predicted_next_close"]
            ret  = result["predicted_return_pct"]
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Last Close",  f"${last:.2f}")
            c2.metric("Predicted",   f"${pred:.2f}")
            c3.metric("Return",      f"{ret:+.3f}%",
                      delta=f"{ret:+.3f}%",delta_color="normal")
            c4.metric("Signal",      result.get("signal","—"))

            ci = result.get("confidence_interval")
            if ci:
                st.info(f"90% CI: **${ci['lower_price']:.2f}** — **${ci['upper_price']:.2f}**  "
                        f"({ci['lower_return_pct']:+.2f}% to {ci['upper_return_pct']:+.2f}%)")

            closes = edited["Close"].tolist()
            fig_p = go.Figure()
            fig_p.add_trace(go.Scatter(x=list(range(len(closes))),y=closes,
                mode="lines+markers",name="Input",line=dict(color="#2196f3",width=2)))
            fig_p.add_hline(y=pred,line_dash="dash",line_color="#e50914",
                annotation_text=f"Predicted: ${pred:.2f}")
            fig_p.update_layout(template="plotly_dark",height=280,
                title="Input Window + Prediction",margin=dict(l=0,r=0,t=40,b=0))
            st.plotly_chart(fig_p,use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — BACKTEST (async Celery job with polling)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_bt:
    st.subheader("Strategy Backtesting")
    st.caption("Heavy calculation offloaded to Celery worker. Polls every 2s.")

    bt_days = st.slider("Days to simulate", 30, 365, 90, 10, key="bt_days")
    if st.button("Run Backtest", type="primary", key="run_bt"):
        job = submit_backtest(ticker, bt_days)
        if "error" in job:
            st.error(f"Could not submit: {job['error']}")
        else:
            job_id = job["job_id"]
            status_box = st.empty()
            result = poll_until_done(job_id, status_box, timeout_s=180)

            if result:
                bt  = result.get("bt_metrics", {})
                m1,m2,m3,m4,m5 = st.columns(5)
                m1.metric("Total Return",   f"{bt.get('Strategy_Total_Return_%',0):+.2f}%")
                m2.metric("Sharpe",         f"{bt.get('Strategy_Sharpe',0):.3f}")
                m3.metric("Sortino",        f"{bt.get('Strategy_Sortino',0):.3f}")
                m4.metric("Max Drawdown",   f"{bt.get('Strategy_MaxDrawdown_%',0):.2f}%")
                m5.metric("Kelly Fraction", f"{bt.get('Kelly_Fraction',0):.4f}")

                curves = result.get("curves", {})
                if curves:
                    fig_eq = go.Figure()
                    fig_eq.add_trace(go.Scatter(y=curves.get("Strategy",[]),
                        name="Strategy",line=dict(color="#e50914",width=2)))
                    fig_eq.add_trace(go.Scatter(y=curves.get("Kelly",[]),
                        name="Kelly",line=dict(color="#ffd700",width=1.5,dash="dot")))
                    fig_eq.add_trace(go.Scatter(y=curves.get("BuyAndHold",[]),
                        name="Buy&Hold",line=dict(color="#9e9e9e",width=1.5,dash="dash")))
                    fig_eq.update_layout(template="plotly_dark",height=380,
                        title="Equity Curve",yaxis_title="Portfolio Value",
                        margin=dict(l=0,r=0,t=40,b=0))
                    st.plotly_chart(fig_eq,use_container_width=True)

                rs = result.get("rolling_sharpe", [])
                if rs:
                    fig_rs = go.Figure()
                    fig_rs.add_trace(go.Scatter(y=rs,name="Rolling Sharpe",
                        line=dict(color="#9c27b0",width=1.5),
                        fill="tozeroy",fillcolor="rgba(156,39,176,0.12)"))
                    fig_rs.add_hline(y=0,line_color="white",opacity=0.2)
                    fig_rs.update_layout(template="plotly_dark",height=200,
                        title="Rolling 63-Day Sharpe",margin=dict(l=0,r=0,t=30,b=0))
                    st.plotly_chart(fig_rs,use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — PAPER TRADE (async Celery job)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_paper:
    st.subheader("Paper Trading Simulation")
    st.caption("Day-by-day simulation offloaded to Celery worker.")

    pt_days = st.number_input("Days", 10, 365, 90, 10, key="pt_days")
    if st.button("Run Paper Trade", type="primary", key="run_pt"):
        job = submit_paper_trade(ticker, int(pt_days))
        if "error" in job:
            st.error(job["error"])
        else:
            status_box = st.empty()
            result = poll_until_done(job["job_id"], status_box, timeout_s=180)
            if result:
                s = result.get("summary", {})
                k1,k2,k3,k4,k5 = st.columns(5)
                k1.metric("Days",       s.get("days_simulated",0))
                k2.metric("Dir Acc",    f"{s.get('dir_accuracy_pct',0):.1f}%")
                k3.metric("Trades",     s.get("n_trades",0))
                k4.metric("Win Rate",   f"{s.get('win_rate_pct',0):.1f}%")
                k5.metric("Total PnL",  f"{s.get('total_pnl_pct',0):+.2f}%")

                log = result.get("log", [])
                if log:
                    df_log = pd.DataFrame(log)
                    df_log["cum_pnl"] = df_log["pnl_pct"].cumsum()
                    fig_pnl = go.Figure()
                    fig_pnl.add_trace(go.Scatter(y=df_log["cum_pnl"],
                        name="Cum PnL",line=dict(color="#00c853",width=2),
                        fill="tozeroy",fillcolor="rgba(0,200,83,0.1)"))
                    fig_pnl.update_layout(template="plotly_dark",height=300,
                        title="Cumulative PnL (%)",margin=dict(l=0,r=0,t=40,b=0))
                    st.plotly_chart(fig_pnl,use_container_width=True)

                    fig_sc = px.scatter(df_log,x="actual_return",y="pred_return",
                        color="correct",
                        color_discrete_map={True:"#00c853",False:"#e50914"},
                        title="Predicted vs Actual Return (%)",
                        template="plotly_dark",height=320)
                    fig_sc.update_layout(margin=dict(l=0,r=0,t=40,b=0))
                    st.plotly_chart(fig_sc,use_container_width=True)
                    st.dataframe(df_log.sort_values("date",ascending=False),
                                 use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — SENTIMENT
# ═══════════════════════════════════════════════════════════════════════════════
with tab_sent:
    st.subheader("News Sentiment")

    @st.cache_data(ttl=3600)
    def _sent(t): return get_sentiment(t)

    s_data = _sent(ticker)
    if "error" in s_data:
        st.warning(s_data["error"])
    else:
        items = s_data.get("items", [])
        avg   = s_data.get("avg_score", 0)
        pos   = sum(1 for i in items if i["sentiment"]=="Positive")
        neg   = sum(1 for i in items if i["sentiment"]=="Negative")
        neu   = len(items) - pos - neg
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Avg Score", f"{avg:+.3f}",
                  delta="Bullish" if avg>0 else "Bearish",
                  delta_color="normal" if avg>0 else "inverse")
        c2.metric("Positive",pos); c3.metric("Neutral",neu); c4.metric("Negative",neg)

        if items:
            df_s = pd.DataFrame(items)
            fig_s = px.bar(df_s,x="date",y="score",color="sentiment",
                color_discrete_map={"Positive":"#00c853","Neutral":"#ffd700","Negative":"#e50914"},
                template="plotly_dark",height=320,title="Sentiment Scores")
            st.plotly_chart(fig_s,use_container_width=True)
            st.dataframe(df_s[["date","title","score","sentiment"]]
                         .sort_values("date",ascending=False),use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — RISK MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════
with tab_risk:
    st.subheader("Risk & Position Management")

    with st.expander("Portfolio Settings", expanded=True):
        rc1,rc2,rc3,rc4 = st.columns(4)
        pv   = rc1.number_input("Portfolio ($)",10_000,10_000_000,100_000,10_000)
        mxp  = rc2.slider("Max Position %",1,20,5)/100
        mxh  = rc3.slider("Max Heat %",5,50,20)/100
        halt = rc4.slider("Circuit Breaker %",5,30,10)/100

    pr1,pr2,pr3,pr4 = st.columns(4)
    inp_price  = pr1.number_input("Price ($)",1.0,10000.0,650.0,1.0)
    inp_atr    = pr2.number_input("ATR",0.1,500.0,15.0,0.5)
    inp_ret    = pr3.number_input("Predicted Return (%)",-10.0,10.0,0.5,0.1)
    inp_wr     = pr4.slider("Win Rate %",40,65,52)/100

    if st.button("Compute Position", type="primary"):
        payload = {"ticker":ticker,"pred_return_pct":inp_ret,"last_price":inp_price,
                   "atr":inp_atr,"portfolio_value":pv,"win_rate":inp_wr,
                   "max_position_pct":mxp,"max_drawdown_halt":halt}
        order  = compute_risk_position(payload)
        matrix = compute_risk_matrix(payload)

        if "error" in order:
            st.error(order["error"])
        else:
            sig = order.get("signal","—")
            if sig=="BUY":
                st.success(f"Signal: BUY — {order['shares']} shares @ ${order['entry_price']:.2f}")
            elif sig=="HALT":
                st.error("Circuit breaker triggered")
            else:
                st.warning(f"Signal: {sig} — {order.get('notes','')}")

            m1,m2,m3,m4,m5,m6 = st.columns(6)
            m1.metric("Shares",       order.get("shares",0))
            m2.metric("Position ($)", f"${order.get('position_value',0):,.0f}")
            m3.metric("Stop Loss",    f"${order.get('stop_loss',0):.2f}")
            m4.metric("Take Profit",  f"${order.get('take_profit',0):.2f}")
            m5.metric("Risk/Trade",   f"${order.get('risk_per_trade',0):,.0f}")
            m6.metric("Kelly",        f"{order.get('kelly_fraction',0):.3f}")

            if matrix and "error" not in matrix:
                mdf = pd.DataFrame([matrix]).T.reset_index()
                mdf.columns = ["Parameter","Value"]
                st.dataframe(mdf,use_container_width=True,hide_index=True)

            prices = [inp_price*(0.85+0.002*i) for i in range(76)]
            pnl    = [(p-inp_price)*order.get("shares",0) for p in prices]
            fig_rr = go.Figure()
            fig_rr.add_trace(go.Scatter(x=prices,y=pnl,mode="lines",
                line=dict(color="#2196f3",width=2),name="P&L"))
            sl = order.get("stop_loss",0); tp = order.get("take_profit",0)
            if sl: fig_rr.add_vline(x=sl,line_dash="dash",line_color="#e50914",
                annotation_text="Stop")
            if tp: fig_rr.add_vline(x=tp,line_dash="dash",line_color="#00c853",
                annotation_text="TP")
            fig_rr.add_hline(y=0,line_color="white",opacity=0.3)
            fig_rr.update_layout(template="plotly_dark",height=320,
                title="P&L Diagram",xaxis_title="Price ($)",yaxis_title="P&L ($)",
                margin=dict(l=0,r=0,t=40,b=0))
            st.plotly_chart(fig_rr,use_container_width=True)

    st.markdown("---")
    st.markdown("#### Execute Trade (Paper / Alpaca)")
    ec1,ec2,ec3,ec4 = st.columns(4)
    ex_shares = ec1.number_input("Shares",1,10000,10)
    ex_side   = ec2.selectbox("Side",["buy","sell"])
    ex_type   = ec3.selectbox("Order Type",["market","limit"])
    ex_broker = ec4.selectbox("Broker",["paper","alpaca"])
    ex_limit  = None
    if ex_type=="limit":
        ex_limit = st.number_input("Limit Price",1.0,10000.0,inp_price,0.01)
    if st.button("Execute", type="primary"):
        resp = execute_trade({"ticker":ticker,"shares":int(ex_shares),
                              "side":ex_side,"order_type":ex_type,
                              "limit_price":ex_limit,"broker":ex_broker})
        if "error" in resp:
            st.error(resp["error"])
        else:
            st.success(f"Order {resp.get('status','submitted')} — "
                       f"{resp.get('side','')} {resp.get('shares','')} {resp.get('ticker','')}"
                       f" via {resp.get('broker','')}")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 7 — DRIFT (async Celery job)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_drift:
    st.subheader("Model Drift Monitor")

    col_drift, _ = st.columns([1,3])
    run_drift = col_drift.button("Run Drift Check", type="primary")

    # Try cached result first
    @st.cache_data(ttl=1800)
    def _drift(t): return get_drift_report(t)
    cached = _drift(ticker)

    if run_drift:
        job = submit_drift_check(ticker)
        if "error" in job:
            st.error(job["error"])
        else:
            sb = st.empty()
            cached = poll_until_done(job["job_id"], sb, timeout_s=120)

    if cached and "error" not in cached:
        n = cached.get("n_drifted", 0)
        if cached.get("overall_drift"):
            st.error(f"Significant drift in {n} features")
        elif n>0:
            st.warning(f"Moderate drift in {n} features")
        else:
            st.success("No significant drift")

        tbl = cached.get("table", [])
        if tbl:
            df_d = pd.DataFrame(tbl).sort_values("PSI", ascending=False)
            top  = df_d.head(20)
            fig_psi = px.bar(top,x="PSI",y="Feature",orientation="h",
                color="Drifted",
                color_discrete_map={True:"#e50914",False:"#2196f3"},
                template="plotly_dark",height=480,title="Top 20 Features by PSI")
            fig_psi.add_vline(x=0.1,line_dash="dash",line_color="#ffd700",
                annotation_text="Moderate")
            fig_psi.add_vline(x=0.2,line_dash="dash",line_color="#e50914",
                annotation_text="Significant")
            st.plotly_chart(fig_psi,use_container_width=True)
            st.dataframe(df_d,use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 8 — EXPLAINABILITY
# ═══════════════════════════════════════════════════════════════════════════════
with tab_xp:
    st.subheader("Model Explainability")

    @st.cache_data(ttl=3600)
    def _fi(): return get_feature_importance()

    fi = _fi()
    if "error" in fi:
        st.warning(fi["error"])
    else:
        feats = fi.get("features",[])
        imps  = fi.get("importances",[])
        n_top = st.slider("Top N features",5,min(40,len(feats)),20)
        fig_fi = go.Figure(go.Bar(
            x=imps[:n_top],y=feats[:n_top],orientation="h",
            marker=dict(color=imps[:n_top],colorscale="Reds",
                showscale=True,colorbar=dict(title="Importance"))))
        fig_fi.update_layout(template="plotly_dark",
            height=max(380,n_top*22),
            title=f"Top {n_top} Feature Importances",
            xaxis_title="Importance",margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig_fi,use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 9 — ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════════════
with tab_arch:
    st.subheader("Architecture & Edge")
    st.markdown("""
### Decoupled Architecture

```
Streamlit (pure presentation)
    │  httpx HTTP requests only
    ▼
FastAPI (v2.0 — headless backend)
    ├── /predict          — ML inference
    ├── /market/*         — OHLCV + indicators
    ├── /risk/position    — position sizing
    ├── /execute          — broker integration
    ├── /sentiment        — VADER news
    ├── /drift            — PSI + KS
    ├── /explainability/* — feature importance
    └── /api/v1/tasks/*   — async job routing
            │
        Redis (broker + result backend)
            │
        Celery Workers
            ├── run_backtest_task
            ├── run_paper_trade_task
            └── run_drift_task
```

### Why this architecture

- Streamlit never blocks on CPU-heavy tasks — Celery workers handle them
- Model loading happens once at FastAPI startup, not on every Streamlit interaction
- FastAPI can be scaled horizontally; Streamlit is just a UI skin
- Redis result backend lets any client (Streamlit, mobile, CLI) poll job status

### Running locally

```bash
# Terminal 1 — backend
make api

# Terminal 2 — Celery worker
celery -A worker.celery_app worker --loglevel=info

# Terminal 3 — Streamlit
make app
```

Set `API_BASE_URL=http://localhost:8000` in `.env` or Streamlit secrets.
    """)
