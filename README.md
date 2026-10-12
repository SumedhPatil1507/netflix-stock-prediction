# Alpha Engine Pro

<p align="center">
  <strong>Interactive quantitative research for equity signals, risk, and execution.</strong><br>
  A Next.js streaming cockpit backed by explainable recommendations, adaptive HLS playback, and a FastAPI service.
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
  <img alt="Next.js 14" src="https://img.shields.io/badge/UI-Next.js%2014-black?logo=next.js&logoColor=white">
  <img alt="Plotly" src="https://img.shields.io/badge/Charts-Plotly-3F4F75?logo=plotly&logoColor=white">
  <img alt="FastAPI" src="https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white">
  <a href="https://github.com/SumedhPatil1507/netflix-stock-prediction/actions/workflows/test.yml"><img alt="Tests" src="https://github.com/SumedhPatil1507/netflix-stock-prediction/actions/workflows/test.yml/badge.svg"></a>
  <a href="LICENSE"><img alt="MIT License" src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
</p>

> **Research software only.** Predictions and backtests are experimental and are not investment advice or a promise of future performance. Review data quality, costs, and execution assumptions before relying on any output.

---

## Production web cockpit

The primary interface is a Next.js 14 App Router static export with Tailwind CSS, Shadcn-style UI primitives, and HLS.js. It provides an adaptive video player, bitrate and playback telemetry, live vector-similarity explanations, conversational catalog search, and a bearer-authenticated recommendation SSE feed.

Configure `API_JWT_SECRET` and `API_USERS_JSON` in an untracked `.env`, then start the production services:

```bash
docker compose up --build
```

Open <http://localhost:3000> or <http://localhost:3001>. Prometheus scrapes the API at <http://localhost:8000/metrics> and its UI is available at <http://localhost:9090>. The Next.js App Router is statically exported and served by Nginx; Next.js is only present in the build stage.

The legacy Streamlit research prototype is reachable from `/legacy`. Start its optional service with `docker compose --profile legacy up --build`.

## Legacy Streamlit research dashboard

Run the legacy prototype locally with `streamlit run app/app.py`. It uses a coordinated dark theme and responsive, interactive Plotly charts. Use hover for detailed values, drag to zoom, scroll to zoom, pan across time, toggle series in the legend, and export high-resolution PNGs from each chart toolbar.

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

## Start the legacy Streamlit app

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

### Deploy on Streamlit Community Cloud

1. Push the desired commit to GitHub and merge it into the branch you want to deploy. This repository's `main` branch requires a pull request, passing `Tests` status check, and verified commit signatures; direct pushes to `main` are rejected.
2. In Streamlit Community Cloud, create or open the app and select repository `SumedhPatil1507/netflix-stock-prediction`.
3. Set the branch to `main` and the main file path to `app/app.py`.
4. Deploy. The app installs its dashboard packages from the root `requirements.txt`; the checked-in model and feature cache provide the initial offline fallback.

When updating an existing app, confirm its configured repository, branch, and main file path first. After a merge to the configured branch, Streamlit Cloud redeploys the app. If it remains on an old build, use **Manage app → Reboot** or **Clear cache** and inspect the app logs for a dependency or startup error.

#### Publish dashboard changes from VS Code (PowerShell)

`main` requires verified signatures on every commit in the pull request. If your feature branch contains older unsigned commits, create a signed snapshot branch from `origin/main` so the PR contains one signed commit. First configure a GitHub-verified signing key under **GitHub → Settings → SSH and GPG keys**; GitHub must list the public key as a signing key. For an SSH key at the default path, configure Git with:

```powershell
git config --global gpg.format ssh
git config --global user.signingkey "$env:USERPROFILE\.ssh\id_ed25519.pub"
git config --global commit.gpgsign true
```

Then commit the edits on the feature branch, copy its complete tracked tree to a fresh branch based on `origin/main`, and create the signed snapshot:

```powershell
git status
git switch publish/dashboard-updates
git pull --ff-only origin publish/dashboard-updates
git add app/app.py src/backtest.py README.md
git commit -S -m "Polish interactive dashboard charts and docs"
git fetch origin
git switch -c publish/dashboard-updates-pr origin/main
git checkout publish/dashboard-updates -- .
git add -A
git commit -S -m "Publish interactive dashboard updates"
git push -u origin publish/dashboard-updates-pr
```

Open a pull request from `publish/dashboard-updates-pr` to `main` on GitHub. Resolve any conflicts in the PR and wait for the **Tests** check. After merge, Streamlit Cloud deploys from `main` if its app settings point to this repository, `main`, and `app/app.py`. Check **Manage app → Logs** if deployment fails. Keep the original feature branch as a backup; the snapshot branch is the one to propose for merge.

Plotly charts are interactive: hover for values, drag to zoom, scroll to zoom, pan, toggle traces from the legend, and export PNGs from the mode bar. Backtest charts use the same dark dashboard palette and toolbar controls.

To retrain the model and regenerate cached outputs:

```bash
python main.py --source csv --ticker NFLX
```

## Run the API

The API provides prediction, risk, and execution routes, plus bearer-authenticated SSE streams. Configure secrets in an untracked `.env` file; start Redis for Pub/Sub features.

```bash
python -m pip install -r requirements-api.txt
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

### Catalog ingestion services (optional)

The repository also contains an isolated streaming-catalog ingestion slice. It is separate from the equity research dashboard. Run Kafka and Redis, install `requirements-catalog.txt`, and install FFmpeg on the API host to enable video transcoding. Configure `KAFKA_BOOTSTRAP_SERVERS`; configure `QDRANT_URL` (and optionally `QDRANT_API_KEY`, `QDRANT_COLLECTION`, and `EMBEDDING_MODEL`) to index metadata and subtitles. Configure `CATALOG_UPLOAD_DIR` to a persistent media volume and set `CATALOG_MAX_UPLOAD_BYTES` to match that volume and proxy limits.

- `POST /api/v1/interactions` accepts validated play, pause, seek, abandon, hover, thumbs-up, and thumbs-down events and publishes them to `netflix.user.interactions.v1` (admin JWT required). Clients should emit `abandon` after five minutes and `hover` when focus ends without a click.
- `python -m src.streaming.interaction_consumer` consumes with manual commits, sends invalid records to `netflix.telemetry.dlq`, and writes valid events to the Redis stream `netflix:user:interactions`.
- `POST /api/v1/catalog/upload` accepts MP4, MKV, MOV, JSON, SRT, and VTT files (admin JWT required), returns an asset ID, and runs processing in a FastAPI background task. Poll `GET /api/v1/catalog/assets/{asset_id}` for status. FFmpeg creates source-compatible 4K/1080p/720p renditions and a 360p fallback. Text uploads are parsed and indexed in Qdrant when configured.
- `web/` is the primary Next.js 14 App Router cockpit. For local development run `cd web; npm install; npm run dev`; set `NEXT_PUBLIC_API_BASE_URL` to the FastAPI origin. The bearer token stays in page memory; uploads require an admin token, while recommendations accept trader or admin tokens.
- `src/recommendations/two_tower.py` trains user and title towers from implicit play/pause/feedback history with pairwise BPR loss. It writes normalized item embeddings into the `netflix_recommendations` Qdrant collection and sentence embeddings into `netflix_catalog` for conversational retrieval.
- `src/recommendations/engine.py` gets at most 500 cosine ANN candidates, then `ranking.py` ranks them with an XGBoost relevance/completion multitask model and device, time, session, recency, completion, and genre signals. The response includes measured candidate retrieval/ranking milliseconds; the 15/30 ms targets must be benchmarked on your Qdrant deployment and catalog size.
- `src/copilot/search_agent.py` uses a LangGraph retrieval/generation flow over Qdrant and validates the JSON result with Guardrails AI plus a catalog-ID allowlist. Returned title names are rendered from retrieved catalog records.
- `POST /api/v1/recommendations/{user_id}` returns an explainable personalized row. `GET /api/v1/recommendations/stream` sends the initial row and refreshes over SSE as Kafka interaction events update the Redis profile. `POST /api/v1/search/recommendations` accepts `{ "query": "dark sci-fi under 90 minutes" }` and optional bounded `history` messages. These routes require an authenticated trader or admin token; users can only request their own personalized feed.

Train and publish a model after the catalog and interaction history have been exported to JSON arrays or JSONL files:

```bash
python -m src.recommendations.train --titles data/catalog/titles.jsonl --events data/catalog/interactions.jsonl
```

Title records need `content_id`, `title`, and preferably `description`, `genres`, `duration_seconds`, and `popularity`. Interaction records need `user_id`, `content_id`, and `event_type` (`play`, `pause`, `thumbs_up`, `thumbs_down`, or `abandon`); `occurred_at`, `position_seconds`, `device_type`, and `session_duration_seconds` improve ranking. The training command writes model artifacts under `models/recommendations/` (ignored by Git) and indexes the catalog into Qdrant. Run model training as an offline job; do not train inside the API request path.

The Streamlit app remains a separate research prototype and does not render catalog media. Streamlit Cloud follows the repository branch configured in its app settings; the checked-in deployment instructions currently specify `main`.

Install the online feature services with `python -m pip install -r requirements-streaming.txt`. Feast/Kafka/Redis are separate from the Streamlit dashboard install so hosted charts do not need those services.

## Project map

```text
app/app.py                    Streamlit research dashboard
api/main.py                   FastAPI inference, RBAC, SSE, and Redis events
src/backtest.py               Event-driven execution and risk simulation
src/data_loader.py            Market data and point-in-time feature joins
src/feature_utils.py          Technical and L2 microstructure features
src/feature_store.py          Feast entities and feature views
src/streaming/kafka_consumer.py  Kafka-to-Redis online feature consumer
src/streaming/kafka_producer.py  Versioned user interaction event producer
src/streaming/interaction_consumer.py  Manual-commit telemetry consumer and DLQ
src/catalog/asset_pipeline.py   HLS transcoding and metadata/subtitle ingestion
src/recommendations/            Two-tower retrieval and contextual ranking
src/copilot/search_agent.py     Grounded LangGraph catalog search
web/                            Next.js 14 production streaming cockpit
feature_store.yaml            Feast repository configuration
tests/                        Unit and API tests
```

## Development

```bash
python -m pytest tests/ -q
python eval/rec_eval.py --sessions 10000 --k 10 --seed 17 --output eval/rec_eval_results.json
cd web; npm ci; npm run build; npm audit --omit=dev
```

See [the performance whitepaper](docs/PERFORMANCE_WHITEPAPER.md) for the synthetic evaluation method and Prometheus/OpenTelemetry metric definitions.

The dashboard, API, and streaming worker use separate dependency groups. The root `requirements.txt` is sized for Streamlit Cloud and offline analytics. Install `requirements-api.txt` for FastAPI, SSE, JWT, and Redis Pub/Sub; install `requirements-streaming.txt` for Feast, Kafka, and the Redis online feature path. Redis and Kafka services are only needed for online features.

## License

MIT. See [LICENSE](LICENSE).
