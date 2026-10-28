"""
Strategy Registry — Alpha Engine Pro multi-strategy framework.

Defines StrategyConfig (a dataclass describing a named strategy) and
StrategyRegistry (a persistent registry that lets users define, store,
and run/compare multiple strategies side-by-side, each backed by its
own model version in the model registry).

Usage
-----
    from src.strategy_registry import StrategyRegistry, StrategyConfig

    reg = StrategyRegistry()
    print(reg.list_strategies())

    cfg = StrategyConfig(
        name="my_strategy",
        tickers=["AAPL"],
        feature_set=["RSI", "MACD", "Lag1"],
        model_config={},
        risk_params={},
    )
    reg.register(cfg)
    metrics = reg.run_strategy("my_strategy", source="csv")
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ── Default feature subsets ───────────────────────────────────────────────────
# These reference column names from modeling.FEATURES.

_MOMENTUM_FEATURES = [
    "Lag1", "Lag2", "Lag3", "Lag5",
    "RSI", "MACD", "MACD_Signal", "MACD_Hist", "MACD_Norm",
    "EMA_Cross",
    "BB_Width",
    "Momentum5", "Momentum10",
    "Regime", "Regime_Bear", "Regime_Side", "Regime_Bull",
]

_MEAN_REVERSION_FEATURES = [
    "BB_Width", "BB_Pct",
    "RSI", "RSI7",
    "Williams_R", "CCI",
    "Stoch_K", "Stoch_D",
]

# Full feature list mirrors modeling.FEATURES
_FULL_FEATURES = [
    "Lag1", "Lag2", "Lag3", "Lag5", "Lag10", "Lag20",
    "RetLag1", "RetLag2", "RetLag3", "RetLag5",
    "RollingMean_5", "RollingMean_10", "RollingStd_5", "RollingStd_10",
    "Return", "LogReturn", "Volatility", "Volatility_5", "VolRatio_5_20",
    "Volume", "Volume_Ratio", "Volume_Ratio20", "OBV_Ratio",
    "RSI", "RSI7",
    "MACD", "MACD_Signal", "MACD_Hist", "MACD_Norm",
    "EMA_Cross",
    "BB_Width", "BB_Pct",
    "ATR_Norm", "Range_Norm", "RangePct",
    "Stoch_K", "Stoch_D", "Williams_R", "CCI",
    "Price_vs_MA5", "Price_vs_MA10", "Price_vs_MA21",
    "Price_vs_MA50", "Price_vs_MA200",
    "Momentum5", "Momentum10", "Momentum20",
    "DayOfWeek", "Month", "Quarter", "EarningsMonth",
    "Regime", "Regime_Bear", "Regime_Side", "Regime_Bull",
]

# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class StrategyConfig:
    """
    Immutable description of a named trading strategy.

    Attributes
    ----------
    name        : Unique strategy identifier.
    tickers     : List of stock symbols this strategy trades.
    feature_set : Subset of modeling.FEATURES column names used for this strategy.
    model_config: Dict of kwargs forwarded to the model builder (e.g.
                  xgb_n_estimators, lgbm_n_estimators, etc.).
    risk_params : Dict of kwargs forwarded to RiskConfig (e.g.
                  max_position_pct, kelly_fraction, atr_stop_multiplier).
    tenant_id   : Multi-tenant namespace — strategies are isolated per tenant.
    description : Human-readable description for the UI/sales deck.
    """

    name: str
    tickers: list[str]
    feature_set: list[str]
    model_config: dict[str, Any] = field(default_factory=dict)
    risk_params: dict[str, Any] = field(default_factory=dict)
    tenant_id: str = "default"
    description: str = ""


# ── Default strategies ────────────────────────────────────────────────────────

def _default_strategies() -> list[StrategyConfig]:
    """Return the three built-in default strategies."""
    return [
        StrategyConfig(
            name="nflx_momentum",
            tickers=["NFLX"],
            feature_set=_MOMENTUM_FEATURES,
            model_config={},
            risk_params={},
            tenant_id="default",
            description=(
                "Single-ticker NFLX momentum strategy using lagged prices, "
                "RSI/MACD oscillators, EMA cross, and HMM regime features."
            ),
        ),
        StrategyConfig(
            name="faang_diversified",
            tickers=["NFLX", "AAPL", "GOOGL", "META", "AMZN"],
            feature_set=_FULL_FEATURES,
            model_config={},
            risk_params={},
            tenant_id="default",
            description=(
                "Diversified FAANG portfolio using the full feature set. "
                "Run independently per ticker, results aggregated."
            ),
        ),
        StrategyConfig(
            name="tech_mean_reversion",
            tickers=["NFLX", "MSFT", "TSLA"],
            feature_set=_MEAN_REVERSION_FEATURES,
            model_config={},
            risk_params={},
            tenant_id="default",
            description=(
                "Mean-reversion strategy for NFLX/MSFT/TSLA using Bollinger, "
                "RSI, Williams %R, CCI, and Stochastic oscillators."
            ),
        ),
    ]


# ── Registry ──────────────────────────────────────────────────────────────────

_REGISTRY_PATH = "outputs/strategy_registry.json"


class StrategyRegistry:
    """
    Persistent strategy store backed by ``outputs/strategy_registry.json``.

    On construction the registry is loaded from disk (or initialised with
    the three built-in defaults if the file does not exist).

    Thread-safety: single-process reads/writes are safe; concurrent writes
    are not guarded — sufficient for the interactive research context.
    """

    def __init__(self, registry_path: str = _REGISTRY_PATH) -> None:
        self._path = registry_path
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        self._store: dict[str, StrategyConfig] = {}

        if os.path.exists(self._path):
            self._load()
        else:
            for cfg in _default_strategies():
                self._store[cfg.name] = cfg
            self._save()
            logger.info("Strategy registry initialised with 3 default strategies.")

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self) -> None:
        """Load registry from JSON on disk."""
        try:
            with open(self._path, "r", encoding="utf-8") as fh:
                raw: dict[str, Any] = json.load(fh)
            for name, data in raw.items():
                self._store[name] = StrategyConfig(**data)
            logger.info(
                "Strategy registry loaded: %d strategies from %s",
                len(self._store),
                self._path,
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to load strategy registry (%s) — using defaults.", exc)
            for cfg in _default_strategies():
                self._store[cfg.name] = cfg

    def _save(self) -> None:
        """Persist registry to JSON on disk."""
        try:
            serialisable = {name: asdict(cfg) for name, cfg in self._store.items()}
            with open(self._path, "w", encoding="utf-8") as fh:
                json.dump(serialisable, fh, indent=2)
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to save strategy registry: %s", exc)

    # ── Public API ────────────────────────────────────────────────────────────

    def register(self, cfg: StrategyConfig) -> None:
        """
        Register (or overwrite) a strategy.

        Parameters
        ----------
        cfg : StrategyConfig
            Strategy configuration to persist.
        """
        self._store[cfg.name] = cfg
        self._save()
        logger.info("Strategy registered: %s (tenant=%s)", cfg.name, cfg.tenant_id)

    def get(self, name: str) -> StrategyConfig:
        """
        Retrieve a strategy by name.

        Raises
        ------
        KeyError if the strategy does not exist.
        """
        if name not in self._store:
            raise KeyError(f"Strategy '{name}' not found in registry.")
        return self._store[name]

    def list_strategies(self) -> list[str]:
        """Return an ordered list of all registered strategy names."""
        return list(self._store.keys())

    def run_strategy(self, name: str, source: str = "csv") -> dict:
        """
        Run a single strategy: load data for the first ticker, engineer
        features, train the stacking model, and return evaluation metrics.

        Only the *first* ticker in ``cfg.tickers`` is used for training
        (multi-ticker parallel execution is deferred to a batch runner).

        Parameters
        ----------
        name   : Registered strategy name.
        source : Data source passed to ``data_loader.load_data``.

        Returns
        -------
        dict with keys: strategy, ticker, metrics (RMSE, MAE, R2, …).
        """
        try:
            from src.data_loader import load_data
            from src.preprocessing import preprocess_data
            from src.feature_engineering import create_features
            from src.modeling import train_model, FEATURES

            cfg = self.get(name)
            ticker = cfg.tickers[0]
            logger.info("Running strategy '%s' on ticker %s …", name, ticker)

            df_raw = load_data(source=source, ticker=ticker)
            df = preprocess_data(df_raw)
            df = create_features(df)

            # Restrict to the strategy's feature_set (intersected with what's
            # actually available in the dataframe so Regime_* columns are optional).
            available = [f for f in cfg.feature_set if f in df.columns]
            if not available:
                logger.warning(
                    "No requested features found in data for strategy '%s'.", name
                )
                available = [f for f in FEATURES if f in df.columns]

            # Temporarily narrow the dataframe to available features + Close
            df_subset = df[[c for c in df.columns if c in available or c == "Close"]]

            _model, metrics, _Xtest, _ytest, _preds = train_model(df_subset)

            result = {"strategy": name, "ticker": ticker, "metrics": metrics}
            logger.info("Strategy '%s' complete. R2=%.4f", name, metrics.get("R2", float("nan")))
            return result

        except Exception as exc:
            logger.exception("run_strategy failed for '%s': %s", name, exc)
            return {"strategy": name, "error": str(exc)}

    def compare_strategies(self, names: list[str], source: str = "csv") -> dict:
        """
        Run each strategy in *names* and return side-by-side metrics.

        Parameters
        ----------
        names  : List of strategy names to compare.
        source : Data source passed to each ``run_strategy`` call.

        Returns
        -------
        dict mapping strategy name → result dict from ``run_strategy``.
        """
        results: dict[str, dict] = {}
        for name in names:
            results[name] = self.run_strategy(name, source=source)
        return results
