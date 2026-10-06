"""
CopilotToolAgent — pulls model state, SHAP drivers, and risk state
from the existing Alpha Engine Pro modules.

All public methods return a dict and never raise; errors are caught,
logged, and an empty dict (or list) is returned instead.
"""
from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


class CopilotToolAgent:
    """
    Aggregates model predictions, SHAP importance, and risk state
    into a single structured dict for the CopilotWriter.

    Parameters
    ----------
    model_path : Optional path to a model pickle.  If *None* the agent
                 uses :func:`src.model_registry.load_latest_model`.
    """

    def __init__(self, model_path: str | None = None) -> None:
        self._model_path = model_path
        self._model: Any = None  # lazy-loaded

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_model_state(
        self,
        ticker: str = "NFLX",
        source: str = "csv",
    ) -> dict[str, Any]:
        """
        Load the latest model and run one-step prediction on *ticker*.

        Returns
        -------
        dict with keys:
            ``pred_return`` (float %), ``signal`` (BUY/HOLD/SELL),
            ``last_price`` (float), ``ticker`` (str), ``model_version`` (str|None),
            ``conformal_lo`` (float), ``conformal_hi`` (float).
        Returns ``{}`` on any failure.
        """
        try:
            from src.model_registry import load_latest_model, get_latest_version
            from src.data_loader import load_data
            from src.preprocessing import preprocess_data
            from src.feature_utils import build_prediction_row

            # Load model
            if self._model is None:
                if self._model_path and os.path.exists(self._model_path):
                    import joblib
                    self._model = joblib.load(self._model_path)
                else:
                    self._model = load_latest_model()

            model = self._model
            version = get_latest_version()

            # Load data
            df_raw = load_data(source=source, ticker=ticker)
            df = preprocess_data(df_raw)

            # Build single prediction row
            row = build_prediction_row(df, model)
            pred_return = float(model.predict(row)[0])
            last_price = float(df["Close"].iloc[-1])
            signal = "BUY" if pred_return > 0 else ("SELL" if pred_return < 0 else "HOLD")

            # Conformal interval
            lo, hi = 0.0, 0.0
            cp = getattr(model, "conformal_", None)
            if cp is not None:
                try:
                    lo_arr, hi_arr = cp.predict_interval(row.values)
                    lo = float(lo_arr[0])
                    hi = float(hi_arr[0])
                except Exception as exc:
                    logger.debug("Conformal interval failed: %s", exc)

            return {
                "pred_return": round(pred_return, 6),
                "signal": signal,
                "last_price": round(last_price, 4),
                "ticker": ticker,
                "model_version": version,
                "conformal_lo": round(lo, 6),
                "conformal_hi": round(hi, 6),
            }

        except Exception as exc:
            logger.error("get_model_state failed: %s", exc)
            return {}

    def get_shap_drivers(self, n: int = 5) -> list[dict[str, Any]]:
        """
        Return the top-*n* feature importances from the loaded model.

        Reads ``model.feature_importances_`` (available on the stacking
        ensemble via its property).  Falls back to an empty list on any
        failure (e.g. model not yet loaded, no feature importances).

        Returns
        -------
        list[dict] — each dict has ``feature`` and ``importance`` keys,
        sorted descending by importance.  Returns ``[]`` on failure.
        """
        try:
            model = self._model
            if model is None:
                from src.model_registry import load_latest_model
                model = load_latest_model()
                self._model = model

            fi = getattr(model, "feature_importances_", None)
            names = getattr(model, "feature_names_", None)

            if fi is None or names is None:
                logger.debug("No feature_importances_ or feature_names_ on model.")
                return []

            paired = sorted(
                zip(names, fi), key=lambda x: x[1], reverse=True
            )
            return [
                {"feature": feat, "importance": round(float(imp), 6)}
                for feat, imp in paired[:n]
            ]

        except Exception as exc:
            logger.error("get_shap_drivers failed: %s", exc)
            return []

    def get_risk_state(
        self,
        ticker: str,
        pred_return: float,
        last_price: float,
        atr: float = 5.0,
    ) -> dict[str, Any]:
        """
        Compute position sizing and risk metrics using RiskManager.

        Parameters
        ----------
        ticker      : Stock symbol.
        pred_return : Model predicted next-day return (%).
        last_price  : Current closing price.
        atr         : Average True Range (defaults to 5.0 if unavailable).

        Returns
        -------
        dict with keys:
            ``stop_loss``, ``take_profit``, ``kelly_fraction``,
            ``portfolio_heat``, ``is_halted``.
        Returns ``{}`` on failure.
        """
        try:
            from src.risk_manager import RiskManager

            rm = RiskManager()
            order = rm.compute_position(
                ticker=ticker,
                pred_return=pred_return,
                last_price=last_price,
                atr=atr,
            )
            matrix = rm.risk_matrix(
                pred_return=pred_return,
                last_price=last_price,
                atr=atr,
            )

            return {
                "stop_loss": order.stop_loss,
                "take_profit": order.take_profit,
                "kelly_fraction": order.kelly_fraction,
                "portfolio_heat": matrix.get("current_drawdown", "0.00%"),
                "is_halted": rm.is_halted,
                "signal": order.signal,
                "shares": order.shares,
                "position_value": order.position_value,
            }

        except Exception as exc:
            logger.error("get_risk_state failed: %s", exc)
            return {}
