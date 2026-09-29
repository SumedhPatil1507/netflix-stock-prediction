"""
Page 4 — Backtest Lab
Async heavy job: submits backtest to Celery via FastAPI, polls every 2s.
Zero src.* imports.
"""
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import time
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import app.api_client as api

st.set_page_config(page_title="Backtest Lab · Alpha Engine",
                   page_icon="🔬", layout="wide")

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
.job-pending { color:#d29922; font-weight:600; font-size:0.9rem; }
.job-success { color:#3fb950; font-weight:600; font-size:0.9rem; }
.job-failure { color:#f85149; font-weight:600; font-size:0.9rem; }
.polling-box {
    background:linear-gradient(135deg,#161b22,#0d1117);
    border:1px solid #3949ab;border-radius:12px;
    padding:20px;text-align:center;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 Backtest Lab")
    st.markdown("---")
    ticker    = st.selectbox("Ticker", ["NFLX","AAPL","TSLA","GOOGL","MSFT","AMZN","META"])
    days      = st.slider("Simulation Days", 30, 365, 90, step=10)
    timeout_s = st.number_input("Max Wait (seconds)", value=300, step=30, min_value=60)
    st.markdown("---")
    st.info(
        "Backtests are CPU-bound.\n\n"
        "The job is dispatched to a **Celery worker** and you receive a `job_id` instantly. "
        "Streamlit polls every **2 seconds** until done."
    )
    run_bt = st.button("🔬 Run Backtest", use_container_width=True, key="run_bt_btn")
    run_pt = st.button("📄 Run Paper Trade", use_container_width=True, key="run_pt_btn")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("## 🔬 Backtest Lab")
st.markdown(
    "Strategies: **Binary signal** (long/flat) · **Kelly-sized** · **Buy-and-hold benchmark**  \n"
    "Metrics: Total Return · Annualised Return · Sharpe · Sortino · Calmar · Max Drawdown · Win Rate"
)

# ── Helper: Poll with Spinner ─────────────────────────────────────────────────
def poll_job(job_id: str, label: str, timeout: int = 300) -> dict | None:
    """
    Polls /api/v1/tasks/{job_id} every 2 seconds.
    Updates a Streamlit placeholder with live status.
    Returns the result dict on SUCCESS, None on FAILURE or timeout.
    """
    placeholder = st.empty()
    start = time.time()

    while True:
        elapsed = int(time.time() - start)
        if elapsed > timeout:
            placeholder.warning(f"⏱️ {label} timed out after {timeout}s. Try again.")
            return None

        resp  = api.get_task_result(job_id)
        state = resp.get("state", "PENDING")
        meta  = resp.get("meta", {})
        step  = meta.get("step", "") if isinstance(meta, dict) else ""
        pct   = meta.get("pct", "") if isinstance(meta, dict) else ""
        pct_s = f" ({pct}%)" if pct != "" else ""

        if state == "SUCCESS":
            placeholder.success(f"✅ {label} completed in {elapsed}s")
            return resp.get("result", {})

        if state == "FAILURE":
            err = resp.get("error", "Unknown error")
            placeholder.error(f"❌ {label} failed: {err}")
            return None

        if state in ("PROGRESS", "STARTED"):
            placeholder.markdown(
                f'<div class="polling-box">'
                f'<div style="font-size:1.5rem;">⚙️</div>'
                f'<div class="job-pending">{label} · {state}{pct_s}</div>'
                f'<div style="color:#8b949e;font-size:0.85rem;margin-top:4px;">'
                f'Step: {step} · {elapsed}s elapsed · polling every 2s…</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            placeholder.markdown(
                f'<div class="polling-box">'
                f'<div style="font-size:1.5rem;">🕐</div>'
                f'<div class="job-pending">{label} · {state} · {elapsed}s elapsed…</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        time.sleep(2.0)


# ── Render Backtest Result ────────────────────────────────────────────────────
def render_backtest(result: dict):
    summary  = result.get("summary", {})
    metrics  = result.get("bt_metrics", {})
    curves   = result.get("curves", {})
    roll_sh  = result.get("rolling_sharpe", [])
    log      = result.get("log", [])

    # ── KPI strip ──────────────────────────────────────────────────────────
    st.markdown("### 📊 Strategy Performance")
    m1,m2,m3,m4,m5,m6 = st.columns(6)
    m1.metric("Total Return",      f"{metrics.get('Strategy_Total_Return_%',0):+.2f}%")
    m2.metric("Ann. Return",       f"{metrics.get('Strategy_Ann_Return_%',0):+.2f}%")
    m3.metric("Sharpe",            f"{metrics.get('Strategy_Sharpe',0):.3f}")
    m4.metric("Sortino",           f"{metrics.get('Strategy_Sortino',0):.3f}")
    m5.metric("Max Drawdown",      f"{metrics.get('Strategy_MaxDrawdown_%',0):.2f}%")
    m6.metric("Win Rate",          f"{metrics.get('Strategy_WinRate_%',0):.1f}%")

    k1,k2,k3,k4,k5,k6 = st.columns(6)
    k1.metric("Kelly Fraction",    f"{metrics.get('Kelly_Fraction',0):.4f}")
    k2.metric("Kelly Return",      f"{metrics.get('Kelly_Total_Return_%',0):+.2f}%")
    k3.metric("Kelly Sharpe",      f"{metrics.get('Kelly_Sharpe',0):.3f}")
    k4.metric("B&H Return",        f"{metrics.get('BuyHold_Total_Return_%',0):+.2f}%")
    k5.metric("B&H Sharpe",        f"{metrics.get('BuyHold_Sharpe',0):.3f}")
    k6.metric("N Trades",          metrics.get("N_Trades",0))

    # ── Equity curves ──────────────────────────────────────────────────────
    st.markdown("### 📈 Equity Curves")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.65,0.35], vertical_spacing=0.04,
                        subplot_titles=("Equity Curves (start=1.0)","Rolling 63-Day Sharpe"))

    n = len(curves.get("Strategy",[]))
    x = list(range(n))

    fig.add_trace(go.Scatter(x=x,y=curves.get("Strategy",[]),name="Strategy",
        line=dict(color="#58a6ff",width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=x,y=curves.get("Kelly",[]),name="Kelly",
        line=dict(color="#3fb950",width=2,dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=x,y=curves.get("BuyAndHold",[]),name="Buy & Hold",
        line=dict(color="#d29922",width=1.5,dash="dash")), row=1, col=1)

    sh_colors = ["rgba(63,185,80,0.6)" if v>=0 else "rgba(248,81,73,0.6)" for v in roll_sh]
    fig.add_trace(go.Bar(x=list(range(len(roll_sh))),y=roll_sh,
        marker_color=sh_colors,name="Rolling Sharpe",showlegend=True), row=2, col=1)
    fig.add_hline(y=0,line=dict(color="#8b949e",width=1),row=2,col=1)

    fig.update_layout(
        height=520, paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(22,27,34,0.8)",
        font=dict(family="Inter",color="#c9d1d9"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=0,r=0,t=30,b=0),
        yaxis=dict(gridcolor="rgba(48,54,61,0.5)"),
        yaxis2=dict(gridcolor="rgba(48,54,61,0.5)"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Paper trade summary ────────────────────────────────────────────────
    if summary:
        st.markdown("### 📋 Paper Trade Summary")
        s1,s2,s3,s4,s5 = st.columns(5)
        s1.metric("Days Simulated",    summary.get("days_simulated",0))
        s2.metric("Directional Acc",   f"{summary.get('dir_accuracy_pct',0):.1f}%")
        s3.metric("N Trades",          summary.get("n_trades",0))
        s4.metric("Win Rate",          f"{summary.get('win_rate_pct',0):.1f}%")
        s5.metric("Total PnL",         f"{summary.get('total_pnl_pct',0):+.2f}%")

    # ── Trade log ─────────────────────────────────────────────────────────
    if log:
        st.markdown("### 📝 Trade Log")
        df_log = pd.DataFrame(log)
        if "correct" in df_log.columns:
            df_log["correct"] = df_log["correct"].map({True:"✅",False:"❌"})
        st.dataframe(df_log, use_container_width=True, hide_index=True)


# ── Main Logic ────────────────────────────────────────────────────────────────
if run_bt:
    st.markdown("---")
    st.markdown(f"### ⏳ Backtest: {ticker} · {days} days")

    with st.spinner("Submitting backtest to Celery worker…"):
        sub = api.submit_backtest(ticker, days)

    if "error" in sub:
        st.error(f"Submission failed: {sub.get('detail', sub['error'])}")
        st.stop()

    job_id = sub["job_id"]
    st.markdown(f'<span style="color:#8b949e;font-size:0.85rem;">Job ID: `{job_id}`</span>',
                unsafe_allow_html=True)

    result = poll_job(job_id, f"Backtest ({ticker}, {days}d)", timeout_s)
    if result:
        render_backtest(result)

elif run_pt:
    st.markdown("---")
    st.markdown(f"### ⏳ Paper Trade: {ticker} · {days} days")

    with st.spinner("Submitting paper trade simulation…"):
        sub = api.submit_paper_trade(ticker, days)

    if "error" in sub:
        st.error(f"Submission failed: {sub.get('detail', sub['error'])}")
        st.stop()

    job_id = sub["job_id"]
    st.markdown(f'<span style="color:#8b949e;font-size:0.85rem;">Job ID: `{job_id}`</span>',
                unsafe_allow_html=True)

    result = poll_job(job_id, f"Paper Trade ({ticker}, {days}d)", timeout_s)
    if result:
        summary = result.get("summary", {})
        log     = result.get("log", [])

        st.markdown("### 📋 Paper Trade Summary")
        s1,s2,s3,s4,s5 = st.columns(5)
        s1.metric("Days Simulated",  summary.get("days_simulated",0))
        s2.metric("Directional Acc", f"{summary.get('dir_accuracy_pct',0):.1f}%")
        s3.metric("N Trades",        summary.get("n_trades",0))
        s4.metric("Win Rate",        f"{summary.get('win_rate_pct',0):.1f}%")
        s5.metric("Total PnL",       f"{summary.get('total_pnl_pct',0):+.2f}%")

        if log:
            df_log = pd.DataFrame(log)
            # Cumulative PnL chart
            if "cum_pnl" in df_log.columns:
                fig_pnl = go.Figure(go.Scatter(
                    x=df_log.get("date", list(range(len(df_log)))),
                    y=df_log["cum_pnl"],
                    mode="lines", name="Cumulative PnL %",
                    line=dict(color="#3fb950" if df_log["cum_pnl"].iloc[-1]>=0 else "#f85149",
                              width=2.5),
                    fill="tozeroy",
                    fillcolor="rgba(63,185,80,0.08)" if df_log["cum_pnl"].iloc[-1]>=0
                              else "rgba(248,81,73,0.08)",
                ))
                fig_pnl.add_hline(y=0,line=dict(color="#8b949e",width=1))
                fig_pnl.update_layout(
                    height=280, paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(22,27,34,0.8)",
                    font=dict(family="Inter",color="#c9d1d9"),
                    margin=dict(l=0,r=0,t=0,b=0),
                    yaxis=dict(title="Cum PnL (%)",gridcolor="rgba(48,54,61,0.5)"),
                )
                st.plotly_chart(fig_pnl, use_container_width=True)

            if "correct" in df_log.columns:
                df_log["correct"] = df_log["correct"].map({True:"✅",False:"❌"})
            st.dataframe(df_log, use_container_width=True, hide_index=True)
else:
    st.info(
        "👈 Click **Run Backtest** to run a full strategy simulation, or "
        "**Run Paper Trade** for a day-by-day live trading simulation.\n\n"
        "These are CPU-heavy jobs dispatched to a **Celery worker** — "
        "Streamlit will poll every 2 seconds and render results when done."
    )
