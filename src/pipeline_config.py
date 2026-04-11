"""
Central configuration — all hyperparameters and paths in one place.
Override via config.yaml or environment variables.
"""
import os
import yaml
import logging
from dataclasses import dataclass, field, asdict
from typing import List

logger = logging.getLogger(__name__)

CONFIG_PATH = "config.yaml"


@dataclass
class DataConfig:
    source: str = "csv"
    csv_path: str = "data/netflix.csv"
    ticker: str = "NFLX"
    yf_period: str = "max"


@dataclass
class FeatureConfig:
    lag_periods: List[int] = field(default_factory=lambda: [1, 2, 3, 5, 10, 20])
    ret_lag_periods: List[int] = field(default_factory=lambda: [1, 2, 3, 5])
    ma_windows: List[int] = field(default_factory=lambda: [5, 7, 10, 21, 50, 200])
    rsi_periods: List[int] = field(default_factory=lambda: [7, 14])
    bb_window: int = 20
    atr_window: int = 14
    stoch_window: int = 14
    cci_window: int = 20
    volatility_window: int = 20


@dataclass
class ModelConfig:
    n_splits_oof: int = 3
    n_splits_cv: int = 5
    train_ratio: float = 0.8
    xgb_n_estimators: int = 600
    xgb_learning_rate: float = 0.03
    xgb_max_depth: int = 5
    xgb_subsample: float = 0.8
    xgb_colsample: float = 0.8
    lgbm_n_estimators: int = 600
    lgbm_learning_rate: float = 0.03
    lgbm_max_depth: int = 5
    rf_n_estimators: int = 400
    rf_max_depth: int = 10
    et_n_estimators: int = 400
    et_max_depth: int = 10
    ridge_alpha: float = 1.0


@dataclass
class BacktestConfig:
    transaction_cost: float = 0.001
    risk_free_rate: float = 0.05


@dataclass
class RegimeConfig:
    n_states: int = 3
    n_iter: int = 200
    enabled: bool = True


@dataclass
class ConformalConfig:
    alpha: float = 0.1       # 90% coverage
    cal_ratio: float = 0.1   # fraction of train set used for calibration
    enabled: bool = True


@dataclass
class PipelineConfig:
    data:      DataConfig      = field(default_factory=DataConfig)
    features:  FeatureConfig   = field(default_factory=FeatureConfig)
    model:     ModelConfig     = field(default_factory=ModelConfig)
    backtest:  BacktestConfig  = field(default_factory=BacktestConfig)
    regime:    RegimeConfig    = field(default_factory=RegimeConfig)
    conformal: ConformalConfig = field(default_factory=ConformalConfig)
    model_path: str = "models/model.pkl"
    output_dir: str = "outputs"
    log_level:  str = "INFO"


def load_config(path: str = CONFIG_PATH) -> PipelineConfig:
    cfg = PipelineConfig()
    if os.path.exists(path):
        with open(path, "r") as f:
            overrides = yaml.safe_load(f) or {}
        _apply_overrides(cfg, overrides)
        logger.info(f"Config loaded from {path}")
    else:
        logger.info("No config.yaml found — using defaults")
    return cfg


def save_default_config(path: str = CONFIG_PATH):
    """Write default config to YAML so users can edit it."""
    import dataclasses

    def _to_dict(obj):
        if dataclasses.is_dataclass(obj):
            return {k: _to_dict(v) for k, v in dataclasses.asdict(obj).items()}
        return obj

    with open(path, "w") as f:
        yaml.dump(_to_dict(PipelineConfig()), f, default_flow_style=False)
    logger.info(f"Default config written to {path}")


def _apply_overrides(cfg: PipelineConfig, overrides: dict):
    for section, values in overrides.items():
        if hasattr(cfg, section) and isinstance(values, dict):
            sub = getattr(cfg, section)
            for k, v in values.items():
                if hasattr(sub, k):
                    setattr(sub, k, v)
        elif hasattr(cfg, section):
            setattr(cfg, section, values)
