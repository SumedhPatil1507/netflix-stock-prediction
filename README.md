# Alpha Engine — Netflix Stock Prediction

[![Tests](https://github.com/SumedhPatil1507/netflix-stock-prediction/actions/workflows/test.yml/badge.svg)](https://github.com/SumedhPatil1507/netflix-stock-prediction/actions)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-app-red)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An interactive Streamlit dashboard for exploring Netflix (NFLX) market data, next-day return estimates, historical backtests, sentiment, risk metrics, model explainability, and drift. The repository includes a trained model, sample data, and a feature cache, so the dashboard can start without first training a model.

> **Disclaimer:** This project is for educational and research purposes only. It is not investment advice and does not guarantee future performance. Market data may be delayed or unavailable.

**[Open the deployed app](https://netflix-stock-prediction-h4e4qxevbfjweltuumcxeb.streamlit.app)** · **[Results](RESULTS.md)** · **[Contributing](CONTRIBUTING.md)**

## Dashboard and model

The dashboard is a nine-tab Streamlit app with interactive Plotly charts. It uses a stacking regressor (XGBoost, LightGBM, Random Forest, and Extra Trees with a Ridge meta-model) and engineered technical indicators. It includes market overview, next-day prediction, backtesting, paper-trading simulation, sentiment, risk, drift monitoring, explainability, and architecture views.

The repository also includes a separate FastAPI service (`api/main.py`), model training pipeline (`main.py`), tests, and GitHub Actions workflows. The Streamlit dashboard is launched independently from the API.

## Run the Streamlit app locally

Use Python 3.11 (the version selected by `.python-version`):

```bash
git clone https://github.com/SumedhPatil1507/netflix-stock-prediction.git
cd netflix-stock-prediction

python3.11 -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

streamlit run app/app.py
```

Open the local URL printed by Streamlit, normally <http://localhost:8501>. The app loads `models/model.pkl` and uses the checked-in feature cache and CSV data as fallbacks. Live chart data is fetched from Yahoo Finance when available; an internet connection is needed for live quotes. No API key is required for the basic dashboard.

You can also launch it with `make app` after installing the dependencies. To run the separate API locally, install the development dependencies and use:

```bash
python -m pip install -r requirements-dev.txt
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

The API docs are then at <http://127.0.0.1:8000/docs>.

## Deploy to Streamlit Community Cloud

1. Push this repository to GitHub.
2. In [Streamlit Community Cloud](https://share.streamlit.io/), create an app and select `SumedhPatil1507/netflix-stock-prediction`.
3. Select branch `main` and set **Main file path** to `app/app.py`.
4. Use Python 3.11 (the repository includes `.python-version`) and deploy. Streamlit Cloud installs the root `requirements.txt` automatically.
5. After subsequent commits are pushed to the selected branch, Streamlit Cloud redeploys the app.

The trained model and data/cache files are tracked in the repository, so deployment does not require a separate training step. `scikit-learn` is pinned to the version used to serialize the checked-in model. Optional integrations may need credentials: keep secrets out of source control and configure them in the app's Streamlit Cloud **Settings → Secrets** only if you enable those integrations. `.env` is ignored by Git.

## Update your changes on GitHub

After editing files locally, review the changes and push them to the selected branch:

```bash
git status
git diff
git add README.md app requirements.txt .streamlit .github  # adjust this list to your changes
git commit -m "Update Streamlit app and documentation"
git push origin main
```

To stage every changed and newly added file instead, use `git add -A` in place of the targeted `git add` command. Do not commit `.env`, credentials, or other secrets.

## Install and run tests

```bash
python -m pip install -r requirements-dev.txt
pytest -q
```

The GitHub Actions test workflow runs on pushes and pull requests targeting `main`.

## Optional training and configuration

The basic dashboard uses the model already in `models/model.pkl`; retraining is not needed just to launch it. To retrain from the included CSV, install `requirements-dev.txt` and run:

```bash
python main.py --source csv --ticker NFLX
```

Other data providers and alerting features can require API credentials. See `.env.example` for the supported variable names. Never commit real credentials. Training writes updated artifacts under `models/` and `outputs/`.

## Project structure

```text
app/app.py                 Streamlit dashboard (Community Cloud entrypoint)
api/main.py                FastAPI service
main.py                    Training and evaluation pipeline
src/                       Data, feature, modeling, risk, and monitoring modules
models/model.pkl           Trained model used by the dashboard
data/netflix.csv           Bundled sample market data
outputs/features_cache.parquet  Cached engineered features
requirements.txt           Runtime dependencies for Streamlit
requirements-dev.txt       Runtime plus test and development dependencies
tests/                     Pytest suite
.github/workflows/         CI and scheduled retraining workflows
```

## Main technologies

Python 3.11 · Streamlit · Plotly · pandas · scikit-learn · XGBoost · LightGBM · yfinance · FastAPI · pytest

## License

MIT — see [LICENSE](LICENSE).
