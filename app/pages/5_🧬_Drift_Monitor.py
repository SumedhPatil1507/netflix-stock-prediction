"""
Page 5 — Drift Monitor
Async heavy job: submits drift check to Celery, polls every 2s.
Zero src.* imports.
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import time
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import app.api_client as api

st.set_page_config(page_title="Drift Monitor · Alpha Engine",
                   page_icon="🧬", layout="wide")

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
.drift-alert {
    background:rgba(248,81,73,0.12);color:#f85149;
    border:1px solid #f85149;border-radius:10px;
    padding:12px 20px;font-weight:600;
}
.drift-ok {
    background:rgba(63,185,80,0.12);color:#3fb950;
    border:1px solid #3fb950;border-radius:10px;
    padding:12px 20px;font-weight:600;
}
.polling-box {
    background:linear-gradient(135deg,#161b22,#0d1117);
    border:1px solid #3949ab;border-radius:12px;
    padding:20px;text-align:center;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧬 Drift Monitor")
    st.markdown("---")
    ticker    = st.selectbox("Ticker", ["NFLX","AAPL","TSLA","GOOGL","MSFT","AMZN","META"])
    timeout_s = st.number_input("Max Wait (seconds)", value=180, min_value=30)
    st.markdown("---")
    st.info(
        "**PSI < 0.1** — No significant change  \n"
        "**PSI 0.1–0.2** — Monitor closely  \n"
        "**PSI > 0.2** — Significant drift → retrain"
    )
    run_drift = st.button("🧬 Run Drift Check", use_container_width=True, key="run_drift_btn")
    st.markdown("---")
    st.info("Fast drift check uses cached `features_cache.parquet`. Run `python main.py` first to populate it.")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("## 🧬 Drift Monitor")
st.markdown(
    "Detects distribution shift between **training** and **live** feature distributions.  \n"
    "Methods: **Population Stability Index (PSI)** + **Kolmogorov-Smirnov test**  \n"
    "Threshold: PSI > 0.2 → significant drift detected → consider retraining."
)

# ── Polling helper ────────────────────────────────────────────────────────────
def poll_job(job_id: str, label: str, timeout: int = 180) -> dict | None:
    placeholder = st.empty()
    start = time.time()

    while True:
        elapsed = int(time.time() - start)
        if elapsed > timeout:
            placeholder.warning(f"⏱️ {label} timed out after {timeout}s.")
            return None

        resp  = api.get_task_result(job_id)
        state = resp.get("state", "PENDING")
        meta  = resp.get("meta", {})
        step  = meta.get("step","") if isinstance(meta, dict) else ""
        pct   = meta.get("pct","") if isinstance(meta, dict) else ""
        pct_s = f" ({pct}%)" if pct != "" else ""

        if state == "SUCCESS":
            placeholder.success(f"✅ {label} completed in {elapsed}s")
            return resp.get("result", {})
        if state == "FAILURE":
            err = resp.get("error","Unknown error")
            placeholder.error(f"❌ {label} failed: {err}")
            return None

        placeholder.markdown(
            f'<div class="polling-box">'
            f'<div style="font-size:1.5rem;">🔬</div>'
            f'<div style="color:#d29922;font-weight:600;">{label} · {state}{pct_s}</div>'
            f'<div style="color:#8b949e;font-size:0.85rem;margin-top:4px;">'
            f'Step: {step} · {elapsed}s elapsed · polling every 2s…</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        time.sleep(2.0)


if run_drift:
    st.markdown("---")
    st.markdown(f"### ⏳ Drift Check: {ticker}")

    with st.spinner("Submitting drift check to Celery worker…"):
        sub = api.submit_drift_check(ticker)

    if "error" in sub:
        st.error(f"Submission failed: {sub.get('detail', sub['error'])}")
        st.stop()

    job_id = sub["job_id"]
    st.markdown(f'<span style="color:#8b949e;font-size:0.85rem;">Job ID: `{job_id}`</span>',
                unsafe_allow_html=True)

    result = poll_job(job_id, f"Drift Check ({ticker})", timeout_s)

    if result:
        overall    = result.get("overall_drift", False)
        n_drifted  = result.get("n_drifted", 0)
        drifted    = result.get("drifted_features", [])
        table      = result.get("table", [])

        # ── Overall status ─────────────────────────────────────────────────
        st.markdown("---")
        if overall:
            st.markdown(
                f'<div class="drift-alert">🚨 SIGNIFICANT DRIFT DETECTED — '
                f'{n_drifted} features have PSI > 0.2 — Consider retraining!</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="drift-ok">✅ No Significant Drift — '
                f'{n_drifted} features flagged — Model appears stable.</div>',
                unsafe_allow_html=True,
            )
        st.markdown("")

        # ── KPIs ───────────────────────────────────────────────────────────
        d1,d2,d3 = st.columns(3)
        d1.metric("Overall Drift",    "⚠️ YES" if overall else "✅ NO")
        d2.metric("Drifted Features", n_drifted)
        d3.metric("PSI Threshold",    "0.20")

        # ── PSI Heatmap / Bar Chart ─────────────────────────────────────────
        if table:
            df_drift = pd.DataFrame(table)
            df_drift = df_drift.sort_values("PSI", ascending=False).head(30)

            tab1, tab2 = st.tabs(["📊 PSI Bar Chart", "📋 Full Feature Table"])

            with tab1:
                bar_colors = [
                    "#f85149" if psi > 0.2 else "#d29922" if psi > 0.1 else "#3fb950"
                    for psi in df_drift["PSI"]
                ]
                fig_bar = go.Figure(go.Bar(
                    y=df_drift["Feature"], x=df_drift["PSI"],
                    orientation="h",
                    marker_color=bar_colors,
                    text=df_drift["PSI"].round(4),
                    textposition="outside",
                    textfont=dict(color="#c9d1d9", size=11),
                ))
                fig_bar.add_vline(x=0.1, line=dict(color="#d29922",dash="dash",width=1.5),
                                  annotation_text="Monitor (0.1)",
                                  annotation_font_color="#d29922")
                fig_bar.add_vline(x=0.2, line=dict(color="#f85149",dash="dash",width=1.5),
                                  annotation_text="Retrain (0.2)",
                                  annotation_font_color="#f85149")
                fig_bar.update_layout(
                    height=max(350, len(df_drift)*22),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(22,27,34,0.8)",
                    font=dict(family="Inter", color="#c9d1d9"),
                    margin=dict(l=0,r=80,t=10,b=0),
                    xaxis=dict(title="PSI Score", gridcolor="rgba(48,54,61,0.5)"),
                    yaxis=dict(autorange="reversed"),
                    showlegend=False,
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            with tab2:
                df_display = df_drift.copy()
                df_display["Drifted"] = df_display["Drifted"].map({True:"🚨 YES", False:"✅ NO"})
                st.dataframe(
                    df_display.rename(columns={
                        "Feature":"Feature","PSI":"PSI Score",
                        "KS_Stat":"KS Statistic","KS_Pval":"KS p-value","Drifted":"Drifted?"
                    }),
                    use_container_width=True, hide_index=True,
                )

        # ── Drifted features list ──────────────────────────────────────────
        if drifted:
            st.markdown("### 🚨 Drifted Features")
            cols = st.columns(min(4, len(drifted)))
            for i, feat in enumerate(drifted):
                cols[i % len(cols)].markdown(
                    f'<span style="background:rgba(248,81,73,0.15);color:#f85149;'
                    f'border:1px solid #f85149;border-radius:6px;padding:4px 10px;'
                    f'font-size:0.85rem;font-weight:600;">{feat}</span>',
                    unsafe_allow_html=True,
                )

else:
    st.info(
        "👈 Click **Run Drift Check** to detect feature distribution shift.  \n\n"
        "The drift analysis is offloaded to a **Celery worker** and "
        "Streamlit polls every 2 seconds until results are ready.  \n\n"
        "**Prerequisite:** Run `python main.py` first to generate `outputs/features_cache.parquet`."
    )
