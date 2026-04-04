"""Unit tests for feature engineering and preprocessing."""
import pytest
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.feature_engineering import create_features
from src.preprocessing import preprocess_data
from src.backtest import run_backtest


# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture
def raw_df():
    """Minimal raw OHLCV dataframe with 300 rows."""
    np.random.seed(42)
    dates  = pd.date_range("2020-01-01", periods=300, freq="B")
    close  = 100 + np.cumsum(np.random.randn(300) * 2)
    df = pd.DataFrame({
        "Date":   dates.strftime("%d-%m-%Y"),
        "Open":   close * 0.99,
        "High":   close * 1.01,
        "Low":    close * 0.98,
        "Close":  close,
        "Volume": np.random.randint(1_000_000, 10_000_000, 300).astype(float),
        "Stock Splits": 0,
    })
    return df


@pytest.fixture
def processed_df(raw_df):
    return preprocess_data(raw_df)


@pytest.fixture
def featured_df(processed_df):
    return create_features(processed_df)


# ── Preprocessing tests ───────────────────────────────────────────────────────
def test_preprocess_sets_date_index(processed_df):
    assert isinstance(processed_df.index, pd.DatetimeIndex)


def test_preprocess_no_duplicate_dates(processed_df):
    assert processed_df.index.duplicated().sum() == 0


def test_preprocess_sorted(processed_df):
    assert processed_df.index.is_monotonic_increasing


# ── Feature engineering tests ─────────────────────────────────────────────────
def test_features_no_nulls(featured_df):
    assert featured_df.isnull().sum().sum() == 0, "Feature df should have no NaNs after dropna"


def test_lag_features_exist(featured_df):
    for lag in [1, 2, 3, 5, 10, 20]:
        assert f"Lag{lag}" in featured_df.columns


def test_rsi_bounds(featured_df):
    assert featured_df["RSI"].between(0, 100).all(), "RSI must be in [0, 100]"
    assert featured_df["RSI7"].between(0, 100).all()


def test_bollinger_band_ordering(featured_df):
    assert (featured_df["BB_Upper"] >= featured_df["BB_Lower"]).all()


def test_atr_positive(featured_df):
    assert (featured_df["ATR"] >= 0).all()


def test_return_column_exists(featured_df):
    assert "Return" in featured_df.columns


def test_momentum_columns(featured_df):
    for col in ["Momentum5", "Momentum10", "Momentum20"]:
        assert col in featured_df.columns


# ── Backtest tests ────────────────────────────────────────────────────────────
def test_backtest_returns_expected_keys():
    np.random.seed(0)
    actual = pd.Series(np.random.randn(200) * 1.5)
    pred   = np.random.randn(200) * 1.5
    result = run_backtest(actual, pred)
    for key in ["metrics", "curves"]:
        assert key in result


def test_backtest_curves_length():
    np.random.seed(1)
    actual = pd.Series(np.random.randn(100) * 1.0)
    pred   = np.random.randn(100) * 1.0
    result = run_backtest(actual, pred)
    assert len(result["curves"]) == 100


def test_backtest_equity_starts_near_one():
    np.random.seed(2)
    actual = pd.Series(np.random.randn(100) * 0.5)
    pred   = np.random.randn(100) * 0.5
    result = run_backtest(actual, pred)
    assert abs(result["curves"]["Strategy"].iloc[0] - 1.0) < 0.05
