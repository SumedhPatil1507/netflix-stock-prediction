TICKER ?= NFLX
SOURCE ?= csv

# ── Training ──────────────────────────────────────────────────────────────────
train:
	python main.py --source csv --ticker $(TICKER)

train-live:
	python main.py --source live --ticker $(TICKER)

train-alphavantage:
	python main.py --source alphavantage --ticker $(TICKER)

train-alpaca:
	python main.py --source alpaca --ticker $(TICKER)

# ── Services (run each in a separate terminal) ────────────────────────────────
api:
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

worker:
	celery -A worker.celery_app worker --loglevel=info --concurrency=2

worker-beat:
	celery -A worker.celery_app beat --loglevel=info

app:
	streamlit run app/app.py

# ── Full stack (Docker) ───────────────────────────────────────────────────────
up:
	docker compose up --build

down:
	docker compose down

# ── Dev tools ─────────────────────────────────────────────────────────────────
tune:
	python -m src.tuning

paper-trade:
	python -m src.paper_trade --days 90

test:
	pytest tests/ -v

config:
	python main.py --save-config

registry:
	python -c "from src.model_registry import get_registry; import json; print(json.dumps(get_registry(), indent=2))"

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; \
	find . -name "*.pyc" -delete 2>/dev/null; echo "Cleaned."

.PHONY: train train-live train-alphavantage train-alpaca api worker worker-beat app up down tune paper-trade test config registry clean
