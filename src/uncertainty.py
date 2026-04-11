"""
Conformal Prediction wrapper.
Produces calibrated prediction intervals instead of point estimates.
Coverage guarantee: the true value falls inside the interval
at least (1 - alpha)% of the time on exchangeable data.
"""
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ConformalPredictor:
    """
    Split conformal prediction (inductive conformal prediction).

    Usage
    -----
    cp = ConformalPredictor(alpha=0.1)   # 90% coverage
    cp.calibrate(model, X_cal, y_cal)
    lo, hi = cp.predict_interval(X_new)
    """

    def __init__(self, alpha: float = 0.1):
        self.alpha      = alpha
        self._quantile  = None
        self._model     = None

    def calibrate(self, model, X_cal: np.ndarray, y_cal: np.ndarray):
        """
        Compute nonconformity scores on a held-out calibration set.
        Nonconformity score = |y_true - y_pred|  (absolute residual).
        """
        self._model = model
        preds       = model.predict(X_cal)
        scores      = np.abs(np.array(y_cal) - preds)

        # Adjusted quantile for finite-sample coverage
        n = len(scores)
        level = np.ceil((n + 1) * (1 - self.alpha)) / n
        level = min(level, 1.0)
        self._quantile = float(np.quantile(scores, level))
        logger.info(f"Conformal quantile (alpha={self.alpha}): {self._quantile:.4f}")
        return self

    def predict_interval(self, X) -> tuple:
        """Returns (lower, upper) arrays for each sample in X."""
        if self._model is None or self._quantile is None:
            raise RuntimeError("Call calibrate() first.")
        preds = self._model.predict(X)
        lower = preds - self._quantile
        upper = preds + self._quantile
        return lower, upper

    def coverage(self, X, y_true) -> float:
        """Empirical coverage on a test set (should be >= 1 - alpha)."""
        lo, hi = self.predict_interval(X)
        y = np.array(y_true)
        return float(np.mean((y >= lo) & (y <= hi)))

    @property
    def interval_width(self) -> float:
        return 2 * self._quantile if self._quantile else float("nan")
