# Alpha Engine Pro

<p align="center">
  <strong>Interactive quantitative research for equity signals, risk, and execution.</strong><br>
  A Streamlit and Plotly dashboard backed by a time-series ML pipeline and FastAPI service.
</p>

<p align="center">
  <a href="https://netflix-stock-prediction-h4e4qxevbfjweltuumcxeb.streamlit.app">Open the live dashboard</a>
  &nbsp;·&nbsp;
  <a href="RESULTS.md">Model results</a>
  &nbsp;·&nbsp;
  <a href="CONTRIBUTING.md">Contributing</a>
</p>

<p align="center">
  <img alt="Python 3.11+" src="https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white">
  <img alt="Streamlit" src="https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white">
  <img alt="Plotly" src="https://img.shields.io/badge/Charts-Plotly-3F4F75?logo=plotly&logoColor=white">
  <img alt="FastAPI" src="https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white">
  <a href="https://github.com/SumedhPatil1507/netflix-stock-prediction/actions/workflows/test.yml"><img alt="Tests" src="https://github.com/SumedhPatil1507/netflix-stock-prediction/actions/workflows/test.yml/badge.svg"></a>
  <a href="LICENSE"><img alt="MIT License" src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
</p>

> **Research software only.** Predictions and backtests are experimental and are not investment advice or a promise of future performance. Review data quality, costs, and execution assumptions before relying on any output.

---

## Dashboard

Run the app locally with `streamlit run app/app.py`. The dark, responsive dashboard uses interactive Plotly charts: hover for point values, zoom and pan across time, toggle series in the legend, and download chart images from the Plotly toolbar.

| Workspace | Interactive views |
|---|---|
| Market | Candlesticks, volume, moving averages, RSI, MACD, and Bollinger Bands |
| Predict | Editable OHLCV window, next-close estimate, signal, and conformal interval |
| Backtest | Chronological holdout, equity, drawdown, position exposure, and rolling Sharpe; configure costs and risk limits |
| Paper trading | Day-by-day simulation, cumulative PnL, and predicted-versus-realized returns |
| Risk & drift | Position sizing, stop levels, VaR/CVaR, correlations, volatility, and feature drift |
| Explainability | Feature importance and model comparisons |
| Research & operations | Copilot research, strategy lab, track record, compliance, and observability |

The Backtesting tab runs the current model on the final 20% of the cached chronological feature data. It exposes commission, bid-ask spread, annual volatility target, drawdown breaker, and risk-free rate controls. Its interactive report includes Sharpe, Sortino, Calmar, maximum drawdown and duration, 95% expected shortfall, and profit factor.

## Start the Streamlit app

Requirements: Python 3.11 or newer. From the repository root:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate

python -m pip install -r requirements.txt
python -m streamlit run app/app.py
```

Open <http://localhost:8501>. The repository includes a model and cached features for the dashboard. Market history uses Yahoo Finance when available and falls back to the local dataset. Use the ticker and period controls in the sidebar to explore the data.

To retrain the model and regenerate cached outputs:

```bash
python main.py --source csv --ticker NFLX
```

## Run the API

The API provides prediction, risk, and execution routes, plus bearer-authenticated SSE streams. Configure secrets in an untracked `.env` file; start Redis for Pub/Sub features.

```bash
Copy-Item .env.example .env  # Windows PowerShell; edit the values before starting
python -m uvicorn api.main:app --reload
```

Set a strong `API_JWT_SECRET` (at least 32 characters) and `API_USERS_JSON` with PBKDF2 password hashes and roles (`trader`, `risk_analyst`, or `admin`). The API also accepts externally issued HS256 or RS256/384/512 JWTs when the matching issuer, audience, and key settings are configured. Never commit `.env`, broker credentials, JWT secrets, or production keys.

Obtain a local access token from `/oauth/token` using form fields `username` and `password`, then send `Authorization: Bearer <token>`. Prediction streams use:

```text
GET /api/v1/stream/predictions/{symbol}
GET /api/v1/stream/risk
```

Publish market updates as JSON to Redis channel `alpha:stream:{SYMBOL}`. Include a `rows` array of OHLCV objects to request a new model prediction and conformal interval. Risk position calculations publish risk metrics and circuit-breaker events on `alpha:risk:events`.

## Point-in-time and streaming features

- `feature_store.yaml` and `src/feature_store.py` define Feast stock metrics with event and creation timestamps for historical point-in-time retrieval.
- `src/data_loader.py` includes a backward, availability-aware join and Feast historical retrieval helper to prevent future or late-arriving values from leaking into backtests.
- `src/streaming/kafka_consumer.py` consumes normalized Kafka events and writes OFI, VWAP, spread, and micro-slippage features to Redis.
- The Kafka producer/market-data bridge must normalize its event format. Alpaca's stock websocket provides trades and best quotes; full market depth requires a data source that supplies L2 snapshots.

## Project map

```text
app/app.py                    Streamlit research dashboard
api/main.py                   FastAPI inference, RBAC, SSE, and Redis events
src/backtest.py               Event-driven execution and risk simulation
src/data_loader.py            Market data and point-in-time feature joins
src/feature_utils.py          Technical and L2 microstructure features
src/feature_store.py          Feast entities and feature views
src/streaming/kafka_consumer.py  Kafka-to-Redis online feature consumer
feature_store.yaml            Feast repository configuration
tests/                        Unit and API tests
```

## Development

```bash
python -m pytest tests/ -q
```

The Streamlit dashboard and FastAPI are separate processes. Redis and Kafka are only needed for online streaming and Pub/Sub features; the local dashboard and offline backtest can run without them.

## License

MIT. See [LICENSE](LICENSE).
