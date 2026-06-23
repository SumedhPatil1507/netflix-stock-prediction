# 📊 Alpha Engine — Netflix Stock AI

[![Tests](https://github.com/SumedhPatil1507/netflix-stock-prediction/actions/workflows/test.yml/badge.svg)](https://github.com/SumedhPatil1507/netflix-stock-prediction/actions)
[![Retrain](https://github.com/SumedhPatil1507/netflix-stock-prediction/actions/workflows/retrain.yml/badge.svg)](https://github.com/SumedhPatil1507/netflix-stock-prediction/actions)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![Celery](https://img.shields.io/badge/Celery-5.3-37814A?logo=celery)](https://docs.celeryq.dev)
[![Redis](https://img.shields.io/badge/Redis-7-DC382D?logo=redis)](https://redis.io)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker)](https://docs.docker.com/compose/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Production-grade quantitative trading system** built on a fully decoupled microservices architecture.  
> Streamlit acts as a **pure presentation layer** — every piece of data is fetched from a headless FastAPI backend.  
> CPU-heavy jobs (backtests, conformal intervals) are offloaded to **Celery workers** via **Redis**.

🚀 **[Live Streamlit App](https://netflix-stock-prediction-h4e4qxevbfjweltuumcxeb.streamlit.app)**

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        PRESENTATION LAYER                        │
│  Streamlit (app/app.py + app/pages/)                            │
│  • Zero src.* imports • Pure HTTP via httpx • 2s polling        │
└───────────────────────────┬─────────────────────────────────────┘
                            │ HTTP (httpx.AsyncClient)
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                         BACKEND LAYER                            │
│  FastAPI (api/main.py)  ·  uvicorn[uvloop + httptools]          │
│  • /predict  • /risk/*  • /market/*  • /sentiment               │
│  • /api/v1/tasks/{backtest,drift,conformal} → job_id instantly  │
└──────────┬──────────────────────────────────┬───────────────────┘
           │ Celery tasks                      │ SQL
           ▼                                   ▼
┌─────────────────────┐           ┌────────────────────────────┐
│  Celery Worker      │           │  TimescaleDB / SQLite       │
│  worker/tasks.py    │           │  src/time_series_db.py      │
│  • run_backtest     │           │  OHLCV market data cache    │
│  • run_paper_trade  │           └────────────────────────────┘
│  • run_drift_check  │
│  • run_conformal    │  ←── broker: Redis 7
└─────────────────────┘
           │
           ▼
┌─────────────────────┐
│  Flower Monitor     │
│  :5555              │
└─────────────────────┘
```

---

## 🧠 ML Model

| Component | Detail |
|---|---|
| Architecture | **ManualStackingRegressor** (2-level stacking) |
| Level-0 learners | XGBoost · LightGBM · RandomForest · ExtraTrees |
| Level-1 meta-learner | Ridge regression on OOF predictions |
| Target variable | Next-day return (%) |
| Uncertainty | **Conformal Prediction** — 90% calibrated intervals |
| Validation | Walk-forward time-series CV (5-fold) |
| Feature set | 51 technical indicators (RSI, MACD, Bollinger, ATR, regime, etc.) |

### Model Performance (Test Set — Last 20% of Data)

| Metric | Value | Notes |
|---|---|---|
| Directional Accuracy | **51.3%** | > 52% = tradeable edge |
| Conformal Coverage | **91.5%** | Target 90% ✅ |
| CP Interval Width | **±8.0%** | 90% of true returns within band |
| CV RMSE | **3.12%** | Walk-forward RMSE on daily returns |
| CV R² (returns) | ~0.00 | Expected — returns are near-random walk |

> **Why R² ≈ 0 is correct**: Predicting price gives R² > 0.99 (trivially — tomorrow ≈ today). Predicting *returns* gives R² ≈ 0 — this is the honest result. The meaningful metrics are directional accuracy and strategy Sharpe ratio.

---

## 🗂️ Project Structure

```
alpha-engine/
├── api/
│   └── main.py              # FastAPI — all endpoints, Celery routing
├── app/
│   ├── api_client.py        # httpx HTTP client (only file allowed to call backend)
│   ├── app.py               # Streamlit home dashboard
│   └── pages/
│       ├── 1_📈_Market_Dashboard.py    # OHLCV · RSI/MACD · Bollinger · Sentiment
│       ├── 2_🤖_Prediction_Engine.py   # Editable OHLCV → /predict → CI chart
│       ├── 3_⚡_Risk_Manager.py        # Kelly sizing · stop-loss · trade execution
│       ├── 4_🔬_Backtest_Lab.py        # Async Celery backtest + polling spinner
│       ├── 5_🧬_Drift_Monitor.py       # Async Celery drift check + PSI heatmap
│       └── 6_⚙️_Model_Registry.py     # Feature importance · model card
├── src/
│   ├── feature_utils.py     # 51 technical features — single source of truth
│   ├── modeling.py          # ManualStackingRegressor + conformal calibration
│   ├── risk_manager.py      # Kelly fraction · ATR stop-loss · circuit breaker
│   ├── backtest.py          # Strategy simulation (binary + Kelly + B&H)
│   ├── paper_trade.py       # Day-by-day live data simulation
│   ├── drift.py             # PSI + KS feature drift detection
│   ├── data_loader.py       # Alpaca → Alpha Vantage → TimescaleDB/SQLite
│   ├── time_series_db.py    # TimescaleDB/SQLite OHLCV persistence
│   └── uncertainty.py       # Split conformal prediction (inductive CP)
├── worker/
│   ├── celery_app.py        # Celery factory (Redis broker + backend)
│   └── tasks.py             # 4 CPU-bound Celery tasks
├── docker-compose.yml       # 6-service orchestration
├── Dockerfile               # Multi-stage Python 3.11 image
├── .env.example             # All environment variables documented
├── requirements.txt         # Dependencies (yfinance removed)
└── main.py                  # Training pipeline entrypoint
```

---

## 🚀 Quick Start

### Option A — Docker (recommended, all services auto-configured)

```bash
# 1. Clone
git clone https://github.com/SumedhPatil1507/netflix-stock-prediction.git
cd netflix-stock-prediction

# 2. Configure environment
cp .env.example .env
# Edit .env — set ALPACA_API_KEY + ALPACA_SECRET_KEY (free paper account)
# Optionally set ALPHA_VANTAGE_KEY for news sentiment

# 3. Train the model first (required once)
pip install -r requirements.txt
python main.py

# 4. Start all 6 services
docker-compose up --build -d

# 5. Open
# Streamlit:  http://localhost:8501
# API docs:   http://localhost:8000/docs
# Flower:     http://localhost:5555  (admin:alphaflower)
```

### Option B — Local development

```bash
# Terminal 1 — FastAPI
uvicorn api.main:app --reload --port 8000

# Terminal 2 — Celery worker
celery -A worker.celery_app worker --loglevel=info --concurrency=2

# Terminal 3 — Streamlit
streamlit run app/app.py --server.port 8501
```

> **Prerequisites for local:** Redis must be running (`redis-server` or Docker: `docker run -p 6379:6379 redis:7-alpine`)

---

## 🔑 Environment Variables

Copy `.env.example` → `.env` and fill in your values:

```bash
# Data Sources (authenticated pipeline — yfinance removed)
ALPACA_API_KEY=your_key          # Free paper account → minute bars
ALPACA_SECRET_KEY=your_secret
ALPACA_BASE_URL=https://paper-api.alpaca.markets

ALPHA_VANTAGE_KEY=your_key       # Free tier: 25 req/day · news sentiment

# Database
DB_HOST=localhost
DB_NAME=alphaengine
DB_USER=alphauser
DB_PASSWORD=changeme

# Infrastructure
REDIS_URL=redis://localhost:6379
API_BASE_URL=http://localhost:8000

# Flower monitor
FLOWER_USER=admin
FLOWER_PASSWORD=alphaflower
```

---

## 📡 API Reference

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | API status + model loaded flag |
| POST | `/predict` | Run stacking ensemble prediction (sync) |
| GET | `/market/ohlcv` | OHLCV bars (Alpaca/AV/DB) |
| GET | `/market/indicators` | RSI · MACD · Bollinger |
| GET | `/market/live_input` | Last N bars for prediction tab |
| GET | `/sentiment` | News sentiment via Alpha Vantage |
| POST | `/risk/position` | Kelly position sizing |
| POST | `/risk/matrix` | Full risk matrix |
| POST | `/execute` | Trade execution (Alpaca / paper) |
| POST | `/api/v1/tasks/backtest` | Submit 90-day backtest → `{job_id}` |
| POST | `/api/v1/tasks/paper_trade` | Submit paper trade → `{job_id}` |
| POST | `/api/v1/tasks/drift` | Submit drift check → `{job_id}` |
| POST | `/api/v1/tasks/conformal` | Submit 10k conformal intervals → `{job_id}` |
| GET | `/api/v1/tasks/{job_id}` | Poll task status (Streamlit polls every 2s) |
| GET | `/explainability/importance` | Ensemble feature importances |
| GET | `/model_info` | Model metadata + latest metrics |

Interactive docs: `http://localhost:8000/docs`

---

## ⚡ Async Task Flow (CPU-Bound Jobs)

```
Streamlit                   FastAPI                     Celery Worker
──────────                  ───────                     ─────────────
[Run Backtest] ──POST──▶   /api/v1/tasks/backtest
                        ◀── {job_id: "abc123"}
                            (returns instantly)
      ↓ poll every 2s
[GET /api/v1/tasks/abc123]                         ── run_backtest_task()
                        ◀── {state: "PROGRESS",        (fetches data,
                              meta: {step, pct}}         runs simulation,
      ↓ poll                                             updates state)
[GET /api/v1/tasks/abc123]
                        ◀── {state: "SUCCESS",
                              result: {...}}
      ↓
[Render charts & metrics]
```

---

## 🐳 Docker Services

| Service | Image | Port | Purpose |
|---|---|---|---|
| `timescaledb` | `timescale/timescaledb:latest-pg16` | 5432 | Time-series OHLCV storage |
| `redis` | `redis:7-alpine` | 6379 | Celery broker + result backend |
| `api` | `alpha-engine:latest` | 8000 | FastAPI + uvicorn (2 workers) |
| `worker` | `alpha-engine:latest` | — | Celery worker (2 concurrency) |
| `flower` | `mher/flower:2.0` | 5555 | Task monitor UI |
| `streamlit` | `alpha-engine:latest` | 8501 | Streamlit presentation layer |

All services share the `alpha_net` bridge network with proper health checks and dependency ordering.

---

## 🛠️ Development

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/ -v

# Run pre-commit hooks
pre-commit run --all-files

# Retrain model
python main.py

# Run a 90-day backtest locally (without Docker)
celery -A worker.celery_app call worker.tasks.run_backtest_task \
  --kwargs '{"ticker":"NFLX","days":90}'
```

---

## 📦 Data Pipeline

**yfinance has been removed.** The data pipeline is:

```
Request → src.data_loader.load_data()
            ├─ 1st: TimescaleDB / SQLite cache (fastest)
            ├─ 2nd: Alpaca Markets REST API (free paper account)
            └─ 3rd: Alpha Vantage REST API (25 req/day free tier)
```

Data is automatically cached in TimescaleDB (or SQLite fallback) after the first fetch.

---

## 📈 Streamlit Pages

| Page | Route | Key Feature |
|---|---|---|
| 🏠 Home Dashboard | `/` | Live candlestick + MA overlays + model KPIs |
| 📈 Market Dashboard | Page 1 | RSI/MACD/Bollinger tabs + sentiment donut |
| 🤖 Prediction Engine | Page 2 | Editable OHLCV → conformal interval chart |
| ⚡ Risk Manager | Page 3 | Kelly sizing · price levels · trade execution |
| 🔬 Backtest Lab | Page 4 | Async job · equity curves · rolling Sharpe |
| 🧬 Drift Monitor | Page 5 | Async job · PSI heatmap · KS test table |
| ⚙️ Model Registry | Page 6 | Feature importance · cumulative curve · model card |

---

## 📄 License

[MIT](LICENSE) © 2024 Sumedh Patil

**[Results](RESULTS.md)** | **[Contributing](CONTRIBUTING.md)**

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Streamlit (Pure Presentation Layer)                    │
│  app/app.py  ──  app/api_client.py                      │
│  Zero src.* imports. All data via httpx to FastAPI.     │
└───────────────────┬─────────────────────────────────────┘
                    │ HTTP (httpx)
┌───────────────────▼─────────────────────────────────────┐
│  FastAPI v2.0  (api/main.py)                            │
│  /predict  /market/*  /risk/*  /execute                 │
│  /sentiment  /drift  /explainability/*                  │
│  /api/v1/tasks/*  (job routing → Celery)                │
└──────┬─────────────────────────┬───────────────────────┘
       │ publish task            │ result store
┌──────▼────────┐      ┌────────▼────────────────────────┐
│  Redis        │      │  Celery Workers (worker/)        │
│  (broker +    │      │  run_backtest_task               │
│   results)    │      │  run_paper_trade_task            │
└───────────────┘      │  run_drift_task                  │
                       └─────────────────────────────────┘
```

---

## What's Inside

| Layer | Implementation |
|---|---|
| **Presentation** | Streamlit — zero ML imports, pure httpx calls |
| **Backend** | FastAPI v2.0 — async, rate limited, versioned |
| **Task Queue** | Celery + Redis — heavy CPU work offloaded |
| **Data** | yfinance · Alpha Vantage · Alpaca Markets · CSV |
| **Features** | 51 technical indicators + HMM regime |
| **Model** | XGB + LGBM + RF + ET → Ridge (manual stacking) |
| **Uncertainty** | Conformal prediction — 90% calibrated intervals |
| **Risk** | ATR stop · Kelly sizing · circuit breaker |
| **Execution** | `/execute` — Alpaca paper/live + simulation |
| **Versioning** | Timestamped model saves + JSON registry |
| **Retraining** | GitHub Actions cron weekly + manual trigger |
| **Monitoring** | PSI + KS drift · Slack/email alerts |
| **Tests** | 40+ pytest unit tests across all modules |

---

## Running Locally

```bash
# 1. Setup
cp .env.example .env
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
pip install -r requirements.txt

# 2. Train model
make train

# 3. Start services (3 terminals)
make api          # FastAPI at http://localhost:8000
make worker       # Celery worker
make app          # Streamlit at http://localhost:8501

# 4. Or use Docker
make up           # docker compose up --build
```

---

## Streamlit Cloud Deployment

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Set **Main file path**: `app/app.py`
4. Add secret in Streamlit Cloud settings:
   ```toml
   [general]
   API_BASE_URL = "https://your-fastapi-backend.railway.app"
   ```
5. Deploy FastAPI + Celery separately on [Railway](https://railway.app) or [Render](https://render.com)

---

## FastAPI Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/predict` | POST | ML prediction + conformal interval |
| `/market/ohlcv` | GET | OHLCV bars for any ticker |
| `/market/indicators` | GET | RSI, MACD, Bollinger |
| `/market/live_input` | GET | Last N rows for predict tab |
| `/risk/position` | POST | Execution-ready position size |
| `/risk/matrix` | POST | Full risk matrix |
| `/execute` | POST | Submit order (Alpaca/paper) |
| `/sentiment` | GET | VADER news sentiment |
| `/drift` | GET | PSI + KS drift report |
| `/explainability/importance` | GET | Feature importances |
| `/api/v1/tasks/backtest` | POST | Submit async backtest job |
| `/api/v1/tasks/paper_trade` | POST | Submit async paper trade job |
| `/api/v1/tasks/drift` | POST | Submit async drift check |
| `/api/v1/tasks/{job_id}` | GET | Poll job status |
| `/model_info` | GET | Version + metrics |
| `/health` | GET | Status check |

---

## Async Task Flow (Celery)

When Streamlit submits a heavy job:
1. FastAPI receives request → calls `task.apply_async()` → returns `{job_id}` instantly
2. Celery worker picks up the task from Redis queue
3. Streamlit polls `/api/v1/tasks/{job_id}` every 2 seconds
4. On `SUCCESS`, result is rendered; on `FAILURE`, error is shown

---

## Environment Variables

```bash
# .env / Streamlit secrets
API_BASE_URL=http://localhost:8000   # FastAPI backend URL
REDIS_URL=redis://localhost:6379
ALPHA_VANTAGE_KEY=...
ALPACA_API_KEY=...
ALPACA_SECRET_KEY=...
ALPACA_BASE_URL=https://paper-api.alpaca.markets
SLACK_WEBHOOK_URL=...
DEFAULT_TICKER=NFLX
```

---

## Tech Stack

Python · XGBoost · LightGBM · Scikit-learn · hmmlearn · FastAPI · Celery · Redis · Streamlit · Plotly · httpx · Pytest · GitHub Actions · Docker · python-dotenv

