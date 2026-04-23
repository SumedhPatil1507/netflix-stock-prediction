"""Unit tests for drift detection."""
import pytest
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.drift import _psi, detect_drift, drift_summary_df


def test_psi_identical_distributions():
    np.random.seed(0)
    x = np.random.randn(1000)
    psi = _psi(x, x)
    assert psi < 0.01  # identical distributions = near-zero PSI


def test_psi_different_distributions():
    np.random.seed(0)
    x = np.random.randn(1000)
    y = np.random.randn(1000) + 5  # shifted by 5 std
    psi = _psi(x, y)
    assert psi > 0.2  # significant drift


def test_psi_non_negative():
    np.random.seed(1)
    x = np.random.randn(500)
    y = np.random.randn(500) * 2
    assert _psi(x, y) >= 0


def test_detect_drift_returns_expected_keys():
    np.random.seed(2)
    n = 300
    df_train = pd.DataFrame({"feat_a": np.random.randn(n),
                              "feat_b": np.random.randn(n)})
    df_live  = pd.DataFrame({"feat_a": np.random.randn(n),
                              "feat_b": np.random.randn(n)})
    result = detect_drift(df_train, df_live, ["feat_a", "feat_b"])
    assert "per_feature" in result
    assert "drifted_features" in result
    assert "overall_drift" in result


def test_detect_drift_no_drift_on_same_data():
    np.random.seed(3)
    n = 500
    df = pd.DataFrame({"x": np.random.randn(n), "y": np.random.randn(n)})
    result = detect_drift(df, df, ["x", "y"])
    assert not result["overall_drift"]


def test_drift_summary_df_shape():
    np.random.seed(4)
    n = 300
    df_train = pd.DataFrame({"a": np.random.randn(n), "b": np.random.randn(n)})
    df_live  = pd.DataFrame({"a": np.random.randn(n) + 3, "b": np.random.randn(n)})
    result = detect_drift(df_train, df_live, ["a", "b"])
    df_out = drift_summary_df(result)
    assert "Feature" in df_out.columns
    assert "PSI" in df_out.columns
    assert "Drifted" in df_out.columns
    assert len(df_out) == 2
