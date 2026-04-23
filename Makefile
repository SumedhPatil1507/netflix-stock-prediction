train:
	python main.py

train-live:
	python main.py --source live

tune:
	python -m src.tuning

paper-trade:
	python -m src.paper_trade --days 90

test:
	pytest tests/ -v

app:
	streamlit run app/app.py

api:
	uvicorn api.main:app --reload

config:
	python main.py --save-config

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; \
	find . -name "*.pyc" -delete 2>/dev/null; \
	echo "Cleaned."

.PHONY: train train-live tune paper-trade test app api config clean
