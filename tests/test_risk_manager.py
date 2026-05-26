"""Unit tests for RiskManager and PositionOrder."""
import pytest
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.risk_manager import RiskManager, RiskConfig, PositionOrder


@pytest.fixture
def rm():
    cfg = RiskConfig(
        portfolio_value    = 100_000,
        max_position_pct   = 0.05,
        max_portfolio_heat = 0.20,
        max_drawdown_halt  = 0.10,
        atr_stop_multiplier= 2.0,
        fixed_stop_pct     = 0.02,
        take_profit_ratio  = 2.0,
    )
    return RiskManager(cfg)


def test_buy_signal_positive_return(rm):
    order = rm.compute_position("NFLX", pred_return=0.5, last_price=650.0, atr=10.0)
    assert order.signal == "BUY"
    assert order.shares > 0


def test_hold_signal_zero_return(rm):
    order = rm.compute_position("NFLX", pred_return=0.0, last_price=650.0, atr=10.0)
    assert order.signal == "HOLD"
    assert order.shares == 0


def test_hold_signal_negative_return(rm):
    order = rm.compute_position("NFLX", pred_return=-0.5, last_price=650.0, atr=10.0)
    assert order.signal == "HOLD"


def test_stop_loss_below_entry(rm):
    order = rm.compute_position("NFLX", pred_return=1.0, last_price=650.0, atr=10.0)
    assert order.stop_loss < order.entry_price


def test_take_profit_above_entry(rm):
    order = rm.compute_position("NFLX", pred_return=1.0, last_price=650.0, atr=10.0)
    assert order.take_profit > order.entry_price


def test_risk_reward_ratio(rm):
    order = rm.compute_position("NFLX", pred_return=1.0, last_price=650.0, atr=10.0)
    stop_dist = order.entry_price - order.stop_loss
    tp_dist   = order.take_profit - order.entry_price
    if stop_dist > 0:
        ratio = tp_dist / stop_dist
        assert abs(ratio - rm.cfg.take_profit_ratio) < 0.01


def test_position_value_within_limit(rm):
    order = rm.compute_position("NFLX", pred_return=1.0, last_price=650.0, atr=10.0)
    max_val = rm.cfg.portfolio_value * rm.cfg.max_position_pct
    assert order.position_value <= max_val * 1.01  # 1% tolerance


def test_circuit_breaker_halts_trading(rm):
    rm.update_portfolio_value(85_000)  # 15% drawdown > 10% halt
    order = rm.compute_position("NFLX", pred_return=1.0, last_price=650.0, atr=10.0)
    assert order.signal == "HALT"
    assert order.shares == 0


def test_to_dict_has_required_keys(rm):
    order = rm.compute_position("NFLX", pred_return=1.0, last_price=650.0, atr=10.0)
    d = order.to_dict()
    for key in ["ticker", "signal", "shares", "entry_price", "stop_loss",
                "take_profit", "position_value", "risk_per_trade", "kelly_fraction"]:
        assert key in d


def test_risk_matrix_keys(rm):
    matrix = rm.risk_matrix(pred_return=0.5, last_price=650.0, atr=10.0)
    for key in ["entry_price", "stop_loss_used", "take_profit",
                "risk_reward_ratio", "is_halted", "predicted_return"]:
        assert key in matrix


def test_current_drawdown_zero_initially(rm):
    assert rm.current_drawdown == 0.0


def test_drawdown_calculation(rm):
    rm.update_portfolio_value(90_000)
    assert abs(rm.current_drawdown - 0.10) < 0.001
