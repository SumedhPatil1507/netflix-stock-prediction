"""Execution cost and institutional backtest metric checks."""
import numpy as np
import pandas as pd

from src.backtest import almgren_chriss_cost, run_backtest
from api.main import RiskRequest, _risk_json


def test_almgren_chriss_cost_increases_with_spread_and_participation():
    low = almgren_chriss_cost(0.1, spread_bps=2, daily_volatility=0.02, participation=0.01)
    high = almgren_chriss_cost(0.1, spread_bps=10, daily_volatility=0.02, participation=0.5)
    assert high > low > 0


def test_backtest_exposes_institutional_risk_metrics():
    actual = pd.Series(np.array([0.8, -1.2, 0.4, -0.7, 1.0, -0.5] * 15))
    predicted = np.array([0.6, -0.5, 0.3, -0.2, 0.7, -0.4] * 15)
    result = run_backtest(actual, predicted, spread_bps=np.linspace(1, 8, len(actual)))
    metrics = result["metrics"]
    for key in ("Strategy_Sharpe", "Strategy_Sortino", "Strategy_Calmar",
                "Strategy_Expected_Shortfall_95_%", "Strategy_Max_Drawdown_Duration_Days"):
        assert key in metrics
    assert len(result["execution_costs"]) == len(actual)


def test_risk_contract_has_exact_keys_and_next_steps():
    request = RiskRequest(pred_return_pct=0.8, last_price=650, atr=12)
    result = _risk_json({
        "signal": "BUY", "kelly_fraction": 0.12, "shares": 18,
        "stop_loss": 625, "shap_drivers": [{"feature": "RSI14", "importance": 0.3}],
        "confidence_interval": {"lower_return_pct": -0.2, "upper_return_pct": 1.4},
    }, request)
    assert list(result) == ["verdict", "reasoning", "recommendation", "next_steps"]
    assert "RSI14" in result["reasoning"]
    assert "-0.2000% to 1.4000%" in result["reasoning"]
    assert result["next_steps"] == [
        "Step 1: Submit order to Alpaca paper/live gateway",
        "Step 2: Log trade execution to audit ledger",
        "Step 3: Monitor PSI/CSI feature drift",
    ]
