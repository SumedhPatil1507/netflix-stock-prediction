"""
Market regime detection using Hidden Markov Model.
Identifies Bull / Bear / Sideways regimes from return series.
"""
import logging
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

REGIME_MODEL_PATH = "models/regime_model.pkl"
REGIME_NAMES = {0: "Bear", 1: "Sideways", 2: "Bull"}


class RegimeDetector:
    """
    Fits a Gaussian HMM on log-returns to identify market regimes.
    States are sorted by mean return so labels are consistent:
      0 = Bear (lowest mean return)
      1 = Sideways
      2 = Bull  (highest mean return)
    """

    def __init__(self, n_states: int = 3, n_iter: int = 200):
        self.n_states = n_states
        self.n_iter   = n_iter
        self._model   = None
        self._state_map: dict = {}

    def fit(self, log_returns: np.ndarray) -> "RegimeDetector":
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError:
            logger.warning("hmmlearn not installed — regime detection disabled")
            return self

        X = log_returns.reshape(-1, 1)
        model = GaussianHMM(
            n_components=self.n_states,
            covariance_type="full",
            n_iter=self.n_iter,
            random_state=42,
        )
        model.fit(X)
        self._model = model

        # Sort states by mean return so 0=Bear, 1=Sideways, 2=Bull
        means = model.means_.flatten()
        order = np.argsort(means)
        self._state_map = {old: new for new, old in enumerate(order)}
        logger.info(f"HMM fitted | state means: {means[order].round(5)}")
        return self

    def predict(self, log_returns: np.ndarray) -> np.ndarray:
        if self._model is None:
            return np.ones(len(log_returns), dtype=int)  # default Sideways
        X      = log_returns.reshape(-1, 1)
        raw    = self._model.predict(X)
        mapped = np.array([self._state_map.get(s, 1) for s in raw])
        return mapped

    def predict_proba(self, log_returns: np.ndarray) -> np.ndarray:
        """Returns (n, n_states) posterior probabilities."""
        if self._model is None:
            n = len(log_returns)
            p = np.zeros((n, self.n_states))
            p[:, 1] = 1.0
            return p
        X = log_returns.reshape(-1, 1)
        return self._model.predict_proba(X)

    def save(self):
        joblib.dump(self, REGIME_MODEL_PATH)
        logger.info(f"Regime model saved -> {REGIME_MODEL_PATH}")

    @staticmethod
    def load() -> "RegimeDetector":
        return joblib.load(REGIME_MODEL_PATH)


def add_regime_features(df: pd.DataFrame, detector: RegimeDetector) -> pd.DataFrame:
    """
    Adds regime label + posterior probabilities as features.
    Must be called after create_features() so LogReturn exists.
    """
    log_ret = df["LogReturn"].fillna(0).values
    regimes = detector.predict(log_ret)
    proba   = detector.predict_proba(log_ret)

    df = df.copy()
    df["Regime"]       = regimes
    df["Regime_Bear"]  = proba[:, 0]
    df["Regime_Side"]  = proba[:, 1]
    df["Regime_Bull"]  = proba[:, 2]
    return df


def fit_and_add_regimes(df: pd.DataFrame) -> tuple:
    """Convenience: fit detector, add features, return (df, detector)."""
    detector = RegimeDetector(n_states=3)
    log_ret  = df["LogReturn"].fillna(0).values
    detector.fit(log_ret)
    df = add_regime_features(df, detector)
    return df, detector
