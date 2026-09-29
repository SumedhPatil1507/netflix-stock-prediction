"""
Page 6 — Model Registry & Feature Importance
Pure presentation layer. Zero src.* imports.
"""
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import app.api_client as api

st.set_page_config(page_title="Model Registry · Alpha Engine",
                   page_icon="⚙️", layout="wide")

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
.model-card {
    background:linear-gradient(135deg,#0d1117,#1a237e22);
    border:1px solid #3949ab;border-radius:14px;
    padding:24px;margin-bottom:16px;
}
.arch-badge {
    display:inline-block;
    background:rgba(88,166,255,0.12);color:#58a6ff;
    border:1px solid #58a6ff;border-radius:6px;
    padding:4px 12px;font-size:0.85rem;margin:4px 2px;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Model Registry")
    st.markdown("---")
    top_n  = st.slider("Top N Features", 10, 51, 25)
    refresh = st.button("🔄 Refresh")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("## ⚙️ Model Registry & Feature Importance")
st.markdown("Model architecture, training metadata, and ensemble feature importances.")

# ── Fetch data ────────────────────────────────────────────────────────────────
with st.spinner("Loading model info…"):
    info = api.get_model_info()
with st.spinner("Loading feature importances…"):
    fi   = api.get_feature_importance()

# ── Model Card ────────────────────────────────────────────────────────────────
st.markdown("### 🏗️ Model Architecture")

arch_parts = ["XGBoost","LightGBM","Random Forest","Extra Trees","→ Ridge Meta-Learner"]
badges = " ".join(f'<span class="arch-badge">{p}</span>' for p in arch_parts)
st.markdown(
    f'<div class="model-card">'
    f'<div style="font-size:1.1rem;font-weight:600;color:#c9d1d9;margin-bottom:12px;">'
    f'ManualStackingRegressor</div>'
    f'{badges}'
    f'<div style="color:#8b949e;font-size:0.85rem;margin-top:14px;">'
    f'Target: next-day return (%) · Scaler: RobustScaler · OOF stacking (3-fold) · '
    f'Walk-forward CV (5-fold) · Conformal prediction (90% coverage)</div>'
    f'</div>',
    unsafe_allow_html=True,
)

# ── Model KPIs ────────────────────────────────────────────────────────────────
if "error" not in info:
    col1,col2,col3,col4 = st.columns(4)
    col1.metric("Model Loaded",        "✅ Yes" if info.get("model_loaded") else "❌ No")
    col2.metric("Features",            info.get("n_features","?"))
    col3.metric("Trained Features",    info.get("trained_feature_count","?"))
    col4.metric("Trained At",          info.get("model_trained_at","unknown"))

    if info.get("conformal_alpha") is not None:
        ca1,ca2,ca3 = st.columns(3)
        ca1.metric("Conformal Alpha",  f"{info['conformal_alpha']:.2f}")
        ca2.metric("Conformal Width",  f"{info.get('conformal_width',0):.4f}%")
        ca3.metric("Coverage Target",  f"{(1-info['conformal_alpha'])*100:.0f}%")

    # Latest metrics
    metrics = info.get("latest_metrics", {})
    if metrics:
        st.markdown("### 📊 Latest Evaluation Metrics")
        m1,m2,m3,m4,m5,m6 = st.columns(6)
        m1.metric("RMSE",           f"{metrics.get('RMSE',0):.4f}")
        m2.metric("MAE",            f"{metrics.get('MAE',0):.4f}")
        m3.metric("R²",             f"{metrics.get('R2',0):.4f}")
        m4.metric("Return R²",      f"{metrics.get('Ret_R2',0):.4f}")
        m5.metric("Dir. Accuracy",  f"{metrics.get('Dir_Acc',0):.1f}%")
        m6.metric("CV RMSE",        f"{metrics.get('CV_RMSE',0):.4f}")

    # Version info
    if info.get("total_versions"):
        st.markdown("---")
        vi1,vi2 = st.columns(2)
        vi1.metric("Registry Versions", info.get("total_versions","?"))
        vi2.metric("Latest Version",    info.get("latest_version","?"))
else:
    st.error(f"Model info unavailable: {info['error']}")

# ── Feature Importance ────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🎯 Feature Importances (Ensemble Average)")

if "error" in fi:
    st.warning(f"Feature importances unavailable: {fi.get('error','Model not loaded?')}")
elif not fi.get("features"):
    st.info("No feature importances available — load a trained model first.")
else:
    features    = fi["features"][:top_n]
    importances = fi["importances"][:top_n]

    # Normalise to 0-100%
    total = sum(importances)
    pcts  = [v/total*100 if total>0 else 0 for v in importances]

    # Colour gradient: top features are brightest blue
    n = len(features)
    colors = [f"rgba(88,166,255,{max(0.3, 1.0 - i/n*0.7):.2f})" for i in range(n)]

    fig_fi = go.Figure(go.Bar(
        y=features[::-1],
        x=pcts[::-1],
        orientation="h",
        marker_color=colors[::-1],
        text=[f"{p:.2f}%" for p in pcts[::-1]],
        textposition="outside",
        textfont=dict(color="#c9d1d9",size=11),
    ))
    fig_fi.update_layout(
        height=max(400, n*24),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(22,27,34,0.8)",
        font=dict(family="Inter",color="#c9d1d9"),
        margin=dict(l=0,r=80,t=10,b=0),
        xaxis=dict(title="Relative Importance (%)",gridcolor="rgba(48,54,61,0.5)"),
        yaxis=dict(autorange="reversed"),
        showlegend=False,
    )
    st.plotly_chart(fig_fi, use_container_width=True)

    # Top-N table
    df_fi = pd.DataFrame({
        "Rank":       list(range(1, len(features)+1)),
        "Feature":    features,
        "Importance": [round(v,6) for v in importances],
        "Weight (%)": [round(p,4) for p in pcts],
    })
    st.dataframe(df_fi, use_container_width=True, hide_index=True)

    # Cumulative importance
    cumulative = []
    s = 0
    for p in pcts:
        s += p
        cumulative.append(round(s,2))

    fig_cum = go.Figure()
    fig_cum.add_trace(go.Scatter(
        x=list(range(1, len(features)+1)), y=cumulative,
        mode="lines+markers",
        line=dict(color="#58a6ff",width=2),
        fill="tozeroy", fillcolor="rgba(88,166,255,0.08)",
        name="Cumulative %",
    ))
    fig_cum.add_hline(y=80, line=dict(color="#d29922",dash="dash",width=1.5),
                      annotation_text="80% threshold",
                      annotation_font_color="#d29922")
    fig_cum.update_layout(
        height=250,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(22,27,34,0.8)",
        font=dict(family="Inter",color="#c9d1d9"),
        margin=dict(l=0,r=0,t=10,b=0),
        xaxis=dict(title=f"Top-N Features",gridcolor="rgba(48,54,61,0.5)"),
        yaxis=dict(title="Cumulative Importance (%)",gridcolor="rgba(48,54,61,0.5)"),
        showlegend=False,
    )
    st.plotly_chart(fig_cum, use_container_width=True)

    # 80% coverage
    n80 = next((i+1 for i,c in enumerate(cumulative) if c >= 80), len(features))
    st.caption(f"ℹ️ Top **{n80}** features account for **≥80%** of ensemble importance.")

# ── Full Feature List ─────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("📋 All Model Features"):
    feats_resp = api._get("/features")
    if "features" in feats_resp:
        all_feats = feats_resp["features"]
        cols = st.columns(4)
        for i, f in enumerate(all_feats):
            cols[i % 4].markdown(
                f'<span style="color:#8b949e;font-size:0.85rem;font-family:monospace;">'
                f'{i+1:02d}. {f}</span>',
                unsafe_allow_html=True,
            )
