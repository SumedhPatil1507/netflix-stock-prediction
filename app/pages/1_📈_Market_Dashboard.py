"""
Page 1 — Market Dashboard
Pure presentation layer: all data fetched via api_client → FastAPI.
Zero src.* imports.
"""
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import app.api_client as api

st.set_page_config(page_title="Market Dashboard · Alpha Engine",
                   page_icon="📈", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#0d1117 0%,#161b22 100%);
    border-right:1px solid #30363d;
}
[data-testid="stSidebar"] * { color:#c9d1d9 !important; }
[data-testid="metric-container"] {
    background:linear-gradient(135deg,#161b22,#1c2333);
    border:1px solid #30363d;border-radius:12px;padding:16px;
}
.tab-header {
    font-size:1.3rem;font-weight:600;color:#c9d1d9;
    border-bottom:2px solid #58a6ff;padding-bottom:8px;margin-bottom:16px;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 Market Dashboard")
    st.markdown("---")
    ticker = st.selectbox("Ticker", ["NFLX","AAPL","TSLA","GOOGL","MSFT","AMZN","META"], index=0)
    period = st.selectbox("Period", ["1mo","3mo","6mo","1y","2y"], index=3)
    st.markdown("---")
    refresh = st.button("🔄 Refresh Data")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("## 📈 Market Dashboard")
st.markdown(f"**{ticker}** · Technical indicators computed server-side · "
            "Data source: Alpaca → Alpha Vantage → TimescaleDB")

# ── Fetch OHLCV ──────────────────────────────────────────────────────────────
with st.spinner(f"Fetching {ticker} OHLCV…"):
    ohlcv = api.get_ohlcv(ticker, period)

if "error" in ohlcv:
    st.error(f"❌ OHLCV fetch failed: {ohlcv['error']}")
    st.stop()

dates  = ohlcv["dates"]
opens  = ohlcv["open"]
highs  = ohlcv["high"]
lows   = ohlcv["low"]
closes = ohlcv["close"]
vols   = ohlcv["volume"]

# ── Price KPIs ────────────────────────────────────────────────────────────────
last   = closes[-1]
prev   = closes[-2] if len(closes) > 1 else last
change = last - prev
pct    = change / prev * 100 if prev else 0

c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("Last Close",   f"${last:,.2f}",    f"{change:+.2f} ({pct:+.2f}%)")
c2.metric("Period High",  f"${max(highs):,.2f}")
c3.metric("Period Low",   f"${min(lows):,.2f}")
c4.metric("Avg Volume",   f"{int(sum(vols)/len(vols)):,}")
c5.metric("Trading Days", f"{len(dates):,}")

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["🕯️ Candlestick", "📉 RSI / MACD", "📊 Bollinger Bands", "💬 Sentiment"])

# ── Tab 1: Candlestick ────────────────────────────────────────────────────────
with tab1:
    closes_s = pd.Series(closes)
    ma20 = closes_s.rolling(20).mean().tolist()
    ma50 = closes_s.rolling(50).mean().tolist()
    ma200= closes_s.rolling(200).mean().tolist()

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.75, 0.25], vertical_spacing=0.02)

    fig.add_trace(go.Candlestick(
        x=dates, open=opens, high=highs, low=lows, close=closes, name=ticker,
        increasing_line_color="#3fb950", decreasing_line_color="#f85149",
        increasing_fillcolor="rgba(63,185,80,0.3)",
        decreasing_fillcolor="rgba(248,81,73,0.3)",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates,y=ma20,name="MA20",
        line=dict(color="#58a6ff",width=1.2,dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates,y=ma50,name="MA50",
        line=dict(color="#d29922",width=1.2,dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates,y=ma200,name="MA200",
        line=dict(color="#bc8cff",width=1.2,dash="dot")), row=1, col=1)

    vol_colors = ["rgba(63,185,80,0.5)" if c>=o else "rgba(248,81,73,0.5)"
                  for c,o in zip(closes,opens)]
    fig.add_trace(go.Bar(x=dates,y=vols,marker_color=vol_colors,
                         name="Volume",showlegend=False), row=2, col=1)

    fig.update_layout(height=580, paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(22,27,34,0.8)",
                      font=dict(family="Inter",color="#c9d1d9"),
                      xaxis_rangeslider_visible=False,
                      legend=dict(bgcolor="rgba(0,0,0,0)"),
                      margin=dict(l=0,r=0,t=10,b=0),
                      yaxis=dict(gridcolor="rgba(48,54,61,0.5)"),
                      yaxis2=dict(showgrid=False))
    st.plotly_chart(fig, use_container_width=True)

# ── Tab 2: RSI / MACD ────────────────────────────────────────────────────────
with tab2:
    with st.spinner("Computing RSI & MACD…"):
        ind = api.get_indicators(ticker, period)

    if "error" in ind:
        st.error(ind["error"])
    else:
        fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                             row_heights=[0.5,0.5], vertical_spacing=0.04,
                             subplot_titles=("RSI (14)", "MACD (12/26/9)"))

        # RSI
        rsi = ind["rsi"]
        fig2.add_trace(go.Scatter(x=ind["dates"],y=rsi,name="RSI",
            line=dict(color="#58a6ff",width=2)), row=1, col=1)
        fig2.add_hline(y=70, line=dict(color="#f85149",dash="dash",width=1), row=1, col=1)
        fig2.add_hline(y=30, line=dict(color="#3fb950",dash="dash",width=1), row=1, col=1)
        fig2.add_hrect(y0=70,y1=100,fillcolor="rgba(248,81,73,0.05)",
                       line_width=0, row=1, col=1)
        fig2.add_hrect(y0=0,y1=30,fillcolor="rgba(63,185,80,0.05)",
                       line_width=0, row=1, col=1)

        # MACD
        macd_hist = ind["macd_hist"]
        bar_colors = ["rgba(63,185,80,0.7)" if v>=0 else "rgba(248,81,73,0.7)"
                      for v in macd_hist]
        fig2.add_trace(go.Bar(x=ind["dates"],y=macd_hist,name="Histogram",
            marker_color=bar_colors,showlegend=True), row=2, col=1)
        fig2.add_trace(go.Scatter(x=ind["dates"],y=ind["macd"],name="MACD",
            line=dict(color="#58a6ff",width=1.5)), row=2, col=1)
        fig2.add_trace(go.Scatter(x=ind["dates"],y=ind["macd_sig"],name="Signal",
            line=dict(color="#d29922",width=1.5,dash="dot")), row=2, col=1)

        fig2.update_layout(height=520,paper_bgcolor="rgba(0,0,0,0)",
                           plot_bgcolor="rgba(22,27,34,0.8)",
                           font=dict(family="Inter",color="#c9d1d9"),
                           legend=dict(bgcolor="rgba(0,0,0,0)"),
                           margin=dict(l=0,r=0,t=30,b=0),
                           yaxis=dict(gridcolor="rgba(48,54,61,0.5)"),
                           yaxis2=dict(gridcolor="rgba(48,54,61,0.5)"))
        st.plotly_chart(fig2, use_container_width=True)

        # Current values
        r1,r2,r3 = st.columns(3)
        r1.metric("Current RSI",     f"{rsi[-1]:.2f}",
                  "Overbought" if rsi[-1]>70 else "Oversold" if rsi[-1]<30 else "Neutral")
        r2.metric("MACD",            f"{ind['macd'][-1]:.4f}")
        r3.metric("MACD Histogram",  f"{macd_hist[-1]:+.4f}")

# ── Tab 3: Bollinger Bands ────────────────────────────────────────────────────
with tab3:
    if "error" not in ind:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=ind["dates"],y=ind["close"],name="Close",
            line=dict(color="#c9d1d9",width=2)))
        fig3.add_trace(go.Scatter(x=ind["dates"],y=ind["bb_upper"],name="BB Upper",
            line=dict(color="#58a6ff",width=1,dash="dash")))
        fig3.add_trace(go.Scatter(x=ind["dates"],y=ind["bb_mid"],name="BB Mid (MA20)",
            line=dict(color="#d29922",width=1.5)))
        fig3.add_trace(go.Scatter(x=ind["dates"],y=ind["bb_lower"],name="BB Lower",
            line=dict(color="#58a6ff",width=1,dash="dash"),
            fill="tonexty", fillcolor="rgba(88,166,255,0.07)"))

        fig3.update_layout(height=460,paper_bgcolor="rgba(0,0,0,0)",
                           plot_bgcolor="rgba(22,27,34,0.8)",
                           font=dict(family="Inter",color="#c9d1d9"),
                           legend=dict(bgcolor="rgba(0,0,0,0)"),
                           margin=dict(l=0,r=0,t=10,b=0),
                           yaxis=dict(gridcolor="rgba(48,54,61,0.5)"))
        st.plotly_chart(fig3, use_container_width=True)

        b1,b2,b3 = st.columns(3)
        b1.metric("BB Upper",  f"${ind['bb_upper'][-1]:,.2f}")
        b2.metric("BB Mid",    f"${ind['bb_mid'][-1]:,.2f}")
        b3.metric("BB Lower",  f"${ind['bb_lower'][-1]:,.2f}")
    else:
        st.error("Indicators unavailable")

# ── Tab 4: Sentiment ─────────────────────────────────────────────────────────
with tab4:
    with st.spinner("Fetching sentiment…"):
        sent = api.get_sentiment(ticker)

    if "error" in sent:
        st.error(sent["error"])
    else:
        items    = sent.get("items", [])
        avg      = sent.get("avg_score", 0)
        source   = sent.get("source", "unknown")

        ss1, ss2, ss3 = st.columns(3)
        ss1.metric("Avg Sentiment",    f"{avg:+.4f}",
                   "📈 Bullish" if avg>0.05 else "📉 Bearish" if avg<-0.05 else "➡️ Neutral")
        ss2.metric("Headlines",        len(items))
        ss3.metric("Source",           source)

        if items:
            df_sent = pd.DataFrame(items)
            df_sent["score"] = df_sent["score"].round(4)

            # Sentiment distribution donut
            counts = df_sent["sentiment"].value_counts()
            color_map = {"Positive":"#3fb950","Neutral":"#d29922","Negative":"#f85149"}
            fig_s = go.Figure(go.Pie(
                labels=counts.index, values=counts.values,
                hole=0.55,
                marker_colors=[color_map.get(l,"#8b949e") for l in counts.index],
            ))
            fig_s.update_layout(height=280,paper_bgcolor="rgba(0,0,0,0)",
                                 font=dict(family="Inter",color="#c9d1d9"),
                                 showlegend=True,
                                 margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig_s, use_container_width=True)

            st.dataframe(
                df_sent[["date","sentiment","score","title"]].rename(columns={
                    "date":"Date","sentiment":"Sentiment","score":"Score","title":"Headline"
                }),
                use_container_width=True, hide_index=True,
            )
        else:
            st.info("No news items. Set ALPHA_VANTAGE_KEY in .env for live sentiment data.")
