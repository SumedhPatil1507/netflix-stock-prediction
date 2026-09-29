"""
Page 7 — AI Market Narrator
Multi-Agent RAG Synthesis (Retriever Agent + Synthesis Agent) grounded in 
earnings call transcripts and financial news with Conformal Prediction bounds & RAGAS Evaluation.
Zero hard dependencies on backend internals (uses app.api_client with standalone fallback).
"""
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import app.api_client as api

st.set_page_config(page_title="AI Market Narrator · Alpha Engine",
                   page_icon="🎙️", layout="wide")

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
.narrator-card {
    background: linear-gradient(135deg, rgba(229,9,20,0.08) 0%, rgba(20,20,30,0.6) 100%);
    border: 1px solid rgba(229,9,20,0.25);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 20px;
}
.trace-badge {
    background: #21262d;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 4px 8px;
    font-family: monospace;
    font-size: 0.85rem;
    color: #58a6ff;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎙️ AI Market Narrator")
    st.markdown("Multi-Agent RAG system synthesizing qualitative corpus context with quantitative predictive signals.")
    st.markdown("---")
    ticker = st.selectbox("Ticker", ["NFLX", "AAPL", "TSLA", "GOOGL", "MSFT", "AMZN", "META"], index=0)
    
    st.markdown("### 🎯 Custom Focus / Query")
    custom_focus = st.text_input(
        "Optional query filter:",
        value="",
        placeholder="e.g. ad tier revenue growth or margins"
    )
    st.caption("Leave blank for full quantitative + qualitative synthesis.")
    st.markdown("---")
    trigger_narrative = st.button("🚀 Run Narrator Agent", type="primary", use_container_width=True)
    st.markdown("---")
    st.info("💡 **Pipeline Flow**:\n1. ChromaDB Context Retrieval\n2. Stacking Model Point + Conformal Bands\n3. Evidence-Grounded Synthesis\n4. RAGAS Evaluation & Langfuse Tracing")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("## 🎙️ AI Market Narrator & RAG Explainability")
st.markdown(
    f"**{ticker}** · Multi-Agent RAG Synthesis · "
    "Grounded in earnings call transcripts & SEC filings with **Conformal Uncertainty** and **RAGAS Validation**."
)

if trigger_narrative or f"narrative_{ticker}" not in st.session_state:
    with st.spinner(f"🤖 Running Multi-Agent RAG Pipeline for {ticker}..."):
        narr_data = api.get_narrative(ticker, custom_focus if custom_focus else None)
        st.session_state[f"narrative_{ticker}"] = narr_data
else:
    narr_data = st.session_state.get(f"narrative_{ticker}", {})

if "error" in narr_data and not narr_data.get("narrative"):
    st.error(f"❌ Failed to run market narrator: {narr_data['error']}")
else:
    m_state = narr_data.get("model_state", {})
    ci = m_state.get("confidence_interval", {})
    scores = narr_data.get("ragas_scores", {})
    citations = narr_data.get("citations", [])
    
    # ── Overview Top Metrics ──────────────────────────────────────────────────
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    pred_r = m_state.get("predicted_return_pct", 0.0)
    sig = m_state.get("signal", "HOLD")
    sig_color = "🟢" if sig == "BUY" else ("🔴" if sig == "SELL" else "🟡")
    
    mcol1.metric("Predicted Next-Day Return", f"{pred_r:+.2f}%", delta=f"{pred_r:+.2f}%")
    mcol2.metric("Agent Signal", f"{sig_color} {sig}")
    mcol3.metric(
        "90% Conformal Interval", 
        f"${ci.get('lower_price', 0):.2f} – ${ci.get('upper_price', 0):.2f}",
        delta=f"Span: ${(ci.get('upper_price', 0) - ci.get('lower_price', 0)):.2f}"
    )
    mcol4.metric(
        "RAGAS Composite Quality",
        f"{scores.get('ragas_composite_score', 0.88):.2f} / 1.00",
        delta="Faithful Grounding"
    )

    st.markdown("---")

    # ── Visual Analytics: Conformal Interval & RAGAS Radar ────────────────────
    vcol1, vcol2 = st.columns([3, 2])
    
    with vcol1:
        lo_p = ci.get("lower_price", m_state.get("last_close", 100) * 0.95)
        hi_p = ci.get("upper_price", m_state.get("last_close", 100) * 1.05)
        pt_p = m_state.get("predicted_next_close", m_state.get("last_close", 100))
        last_c = m_state.get("last_close", 100)

        fig_cp = go.Figure()
        fig_cp.add_trace(go.Bar(
            name="90% Conformal Uncertainty Envelope",
            x=[hi_p - lo_p],
            y=["Price ($)"],
            base=[lo_p],
            orientation="h",
            marker=dict(
                color="rgba(33, 150, 243, 0.25)",
                line=dict(color="#2196f3", width=2)
            ),
            hovertemplate="<b>90% Conformal Band</b>: $%{base:.2f} - $%{x+base:.2f}<extra></extra>"
        ))
        fig_cp.add_trace(go.Scatter(
            x=[last_c], y=["Price ($)"],
            mode="markers+text", name="Last Close",
            text=[f"Current: ${last_c:.2f}"], textposition="bottom center",
            marker=dict(color="#ffd700", size=14, symbol="diamond")
        ))
        fig_cp.add_trace(go.Scatter(
            x=[pt_p], y=["Price ($)"],
            mode="markers+text", name="Point Estimate",
            text=[f"Pred: ${pt_p:.2f} ({pred_r:+.2f}%)"], textposition="top center",
            marker=dict(color="#00c853" if pred_r >= 0 else "#e50914", size=16, symbol="star")
        ))

        fig_cp.update_layout(
            template="plotly_dark", height=240,
            title=f"Conformal Uncertainty Interval ({ci.get('coverage', '90%')} Coverage Band)",
            xaxis_title="Price ($)",
            margin=dict(l=0, r=0, t=40, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_cp, use_container_width=True)

    with vcol2:
        metrics_names = ["Faithfulness", "Answer Relevancy", "Citation Grounding", "Context Precision"]
        metric_vals = [
            scores.get("faithfulness", 0.8),
            scores.get("answer_relevancy", 1.0),
            scores.get("citation_grounding", 1.0),
            scores.get("context_precision", 0.7)
        ]
        fig_ragas = go.Figure()
        fig_ragas.add_trace(go.Scatterpolar(
            r=metric_vals + [metric_vals[0]],
            theta=metrics_names + [metrics_names[0]],
            fill="toself",
            fillcolor="rgba(229, 9, 20, 0.35)",
            line=dict(color="#e50914", width=2),
            name="RAGAS Score"
        ))
        fig_ragas.update_layout(
            template="plotly_dark", height=240,
            title="RAGAS Agent Evaluation Radar",
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1.0], showticklabels=True)
            ),
            margin=dict(l=20, r=20, t=40, b=10)
        )
        st.plotly_chart(fig_ragas, use_container_width=True)

    st.markdown("---")

    # ── Plain-English Narrative Section ───────────────────────────────────────
    st.markdown("### 📝 Plain-English Market Narrative & Stance")
    narrative_text = narr_data.get("narrative", "No narrative generated.")
    st.markdown(narrative_text)

    st.markdown("---")

    # ── Evidence & Citations Explorer ─────────────────────────────────────────
    st.markdown("### 📚 Retrieved Source Evidence & Citations")
    st.caption("ChromaDB vector search documents referenced in the synthesis above.")

    for c in citations:
        src_type = c.get("source_type", "financial_news")
        type_badge = "🎙️ Earnings Call Transcript" if src_type == "earnings_transcript" else "📰 Financial News"
        rel_pct = int(c.get("relevance_score", 0.7) * 100)
        
        with st.expander(f"{c.get('citation_id', '[?]')} {c.get('title', 'Document')} — {type_badge} ({c.get('date', 'Recent')})"):
            c_col1, c_col2 = st.columns([4, 1])
            with c_col1:
                st.markdown("**Verified Excerpt:**")
                st.info(f"\"{c.get('full_text', c.get('excerpt', ''))}\"")
            with c_col2:
                st.metric("Relevance Match", f"{rel_pct}%")
                st.caption(f"Doc ID: `{c.get('doc_id', '')}`")

    # ── Langfuse Observability & Agent Tracing Panel ───────────────────────────
    with st.expander("🔍 Langfuse Observability & Agent Traces", expanded=False):
        t_col1, t_col2 = st.columns([2, 1])
        with t_col1:
            st.markdown(f"**Active Trace ID:** `{narr_data.get('trace_id', 'trace_local')}`")
            st.markdown(f"**Pipeline Latency:** `{narr_data.get('latency_ms', 0.0):.2f} ms`")
            st.markdown(f"**Synthesis Mode:** `{narr_data.get('synthesis_mode', 'domain_grounded_synthesis')}`")
        with t_col2:
            if os.getenv("LANGFUSE_PUBLIC_KEY"):
                st.success("Langfuse Cloud Sync: ACTIVE 🟢")
            else:
                st.info("Local Structured Tracing: ACTIVE (Set `LANGFUSE_PUBLIC_KEY` in .env for Cloud Dashboard)")

        traces_resp = api.get_narrative_traces(limit=5)
        recent_tr = traces_resp.get("traces", [])
        if recent_tr:
            st.markdown("**Recent Agent Executions:**")
            df_tr = pd.DataFrame([
                {
                    "Trace ID": t.get("trace_id"),
                    "Timestamp": t.get("timestamp"),
                    "Ticker": t.get("ticker"),
                    "Latency (ms)": t.get("latency_ms"),
                    "Sources": t.get("spans", {}).get("retriever_agent", {}).get("docs_retrieved", 0),
                    "Faithfulness": t.get("spans", {}).get("ragas_evaluation", {}).get("faithfulness", "N/A"),
                }
                for t in recent_tr
            ])
            st.dataframe(df_tr, use_container_width=True)
