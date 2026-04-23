"""Unit tests for ManualStackingRegressor."""
import pytest
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.modeling import ManualStackingRegressor, _directional_accuracy, FEATURES


@pytest.fixture
def small_dataset():
    """200-row synthetic OHLCV-like feature dataset."""
    np.random.seed(99)
    n = 200
    X = pd.DataFrame(np.random.randn(n, 10),
                     columns=[f"feat_{i}" for i in range(10)])
    y = pd.Series(np.random.randn(n) * 2)
    return X, y


def test_stacking_fit_predict(small_dataset):
    X, y = small_dataset
    model = ManualStackingRegressor(n_splits=2)
    model.fit(X.iloc[:150], y.iloc[:150])
    preds = model.predict(X.iloc[150:])
    assert len(preds) == 50
    assert not np.any(np.isnan(preds))


def test_feature_names_stored(small_dataset):
    X, y = small_dataset
    model = ManualStackingRegressor(n_splits=2)
    model.fit(X, y)
    assert hasattr(model, "feature_names_")
    assert model.feature_names_ == list(X.columns)


def test_predict_handles_extra_columns(small_dataset):
    X, y = small_dataset
    model = ManualStackingRegressor(n_splits=2)
    model.fit(X, y)
    # Add extra column — model should ignore it
    X_extra = X.copy()
    X_extra["extra_col"] = 99.0
    preds = model.predict(X_extra)
    assert len(preds) == len(X_extra)


def test_predict_handles_missing_columns(small_dataset):
    X, y = small_dataset
    model = ManualStackingRegressor(n_splits=2)
    model.fit(X, y)
    # Remove a column — model should fill with 0
    X_missing = X.drop(columns=["feat_0"])
    preds = model.predict(X_missing)
    assert len(preds) == len(X_missing)


def test_feature_importances(small_dataset):
    X, y = small_dataset
    model = ManualStackingRegressor(n_splits=2)
    model.fit(X, y)
    fi = model.feature_importances_
    assert fi is not None
    assert len(fi) == X.shape[1]
    assert np.all(fi >= 0)


def test_directional_accuracy_perfect():
    y = np.array([1.0, -1.0, 2.0, -0.5])
    p = np.array([0.5, -0.3, 1.0, -0.1])
    assert _directional_accuracy(y, p) == 100.0


def test_directional_accuracy_random():
    np.random.seed(0)
    y = np.random.randn(1000)
    p = np.random.randn(1000)
    acc = _directional_accuracy(y, p)
    assert 40 < acc < 60  # should be near 50% for random


def test_features_list_no_duplicates():
    assert len(FEATURES) == len(set(FEATURES))


def test_features_list_not_empty():
    assert len(FEATURES) > 0
