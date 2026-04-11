"""Tests for regime detection and conformal prediction."""
import pytest
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.uncertainty import ConformalPredictor


# ── Conformal prediction tests ────────────────────────────────────────────────
class _DummyModel:
    """Predicts mean of training y."""
    def __init__(self): self._mean = 0.0
    def fit(self, X, y): self._mean = float(np.mean(y)); return self
    def predict(self, X): return np.full(len(X), self._mean)


def test_conformal_coverage():
    np.random.seed(42)
    X = np.random.randn(500, 5)
    y = np.random.randn(500) * 2

    model = _DummyModel()
    model.fit(X[:300], y[:300])

    cp = ConformalPredictor(alpha=0.1)
    cp.calibrate(model, X[300:400], y[300:400])

    coverage = cp.coverage(X[400:], y[400:])
    # Should be >= 0.90 with high probability
    assert coverage >= 0.85, f"Coverage {coverage:.3f} too low"


def test_conformal_interval_width_positive():
    np.random.seed(0)
    X = np.random.randn(200, 3)
    y = np.random.randn(200)
    model = _DummyModel()
    model.fit(X[:150], y[:150])
    cp = ConformalPredictor(alpha=0.2)
    cp.calibrate(model, X[150:], y[150:])
    assert cp.interval_width > 0


def test_conformal_requires_calibration():
    cp = ConformalPredictor()
    with pytest.raises(RuntimeError):
        cp.predict_interval(np.random.randn(10, 3))


# ── Regime detection tests ────────────────────────────────────────────────────
def test_regime_labels_valid():
    try:
        from src.regime_detection import RegimeDetector
    except ImportError:
        pytest.skip("hmmlearn not installed")

    np.random.seed(7)
    log_ret = np.random.randn(500) * 0.01
    det = RegimeDetector(n_states=3, n_iter=50)
    det.fit(log_ret)
    labels = det.predict(log_ret)
    assert set(labels).issubset({0, 1, 2})


def test_regime_proba_sums_to_one():
    try:
        from src.regime_detection import RegimeDetector
    except ImportError:
        pytest.skip("hmmlearn not installed")

    np.random.seed(8)
    log_ret = np.random.randn(300) * 0.01
    det = RegimeDetector(n_states=3, n_iter=50)
    det.fit(log_ret)
    proba = det.predict_proba(log_ret)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)
