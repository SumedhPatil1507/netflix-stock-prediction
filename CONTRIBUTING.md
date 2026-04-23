# Contributing

## Setup

```bash
git clone https://github.com/SumedhPatil1507/netflix-stock-prediction.git
cd netflix-stock-prediction
pip install -r requirements-dev.txt
pre-commit install
```

## Running the pipeline

```bash
make train          # train on CSV data
make train-live     # train on live yfinance data
make test           # run pytest
make app            # launch Streamlit
make api            # launch FastAPI
make paper-trade    # run 90-day paper trade simulation
make tune           # Optuna hyperparameter search
```

## Project structure

All source modules are in `src/`. Each file has a single responsibility:

| File | Responsibility |
|---|---|
| `feature_utils.py` | Shared feature computation — edit this, not the duplicates |
| `feature_engineering.py` | Full pipeline feature engineering (calls feature_utils) |
| `modeling.py` | ManualStackingRegressor, train_model, FEATURES list |
| `regime_detection.py` | HMM regime detection |
| `uncertainty.py` | Conformal prediction intervals |
| `backtest.py` | Backtesting engine (Kelly, Sharpe, Sortino) |
| `drift.py` | PSI + KS drift detection |
| `sentiment.py` | VADER news sentiment |
| `paper_trade.py` | Paper trading simulation |
| `tuning.py` | Optuna hyperparameter search |
| `pipeline_config.py` | YAML config dataclasses |

## Adding a new feature

1. Add the computation to `src/feature_utils.py` → `compute_features_from_ohlcv()`
2. Add the column name to `FEATURES` in `src/modeling.py`
3. Add a unit test in `tests/test_features.py`
4. Retrain: `make train`

## Code style

- Type hints on all function signatures
- Docstrings on all public functions
- `black` formatting (enforced by pre-commit)
- No `print()` — use `logging.getLogger(__name__)`

## Pull requests

- One feature or fix per PR
- All tests must pass: `make test`
- Update `RESULTS.md` if model metrics change
