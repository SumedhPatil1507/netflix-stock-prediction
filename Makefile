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

# ── Tools ─────────────────────────────────────────────────────────────────────
tune:
	python -m src.tuning

paper-trade:
	python -m src.paper_trade --days 90

# ── Dev ───────────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v

app:
	streamlit run app/app.py

api:
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

config:
	python main.py --save-config

registry:
	python -c "from src.model_registry import get_registry; import json; print(json.dumps(get_registry(), indent=2))"

# ── Maintenance ───────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; \
	find . -name "*.pyc" -delete 2>/dev/null; \
	echo "Cleaned."

.PHONY: train train-live train-alphavantage train-alpaca tune paper-trade test app api config registry clean
