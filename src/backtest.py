"""Event-driven, daily-bar backtesting with causal sizing and execution costs."""
from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
TRADING_DAYS = 252


def _annualised_return(equity: np.ndarray, n_days: int) -> float:
    if n_days <= 0 or equity[-1] <= 0:
        return 0.0
    return float((equity[-1] ** (TRADING_DAYS / n_days) - 1) * 100)


def _sharpe(returns: np.ndarray, rf: float = 0.0) -> float:
    excess = returns - rf / TRADING_DAYS
    std = excess.std(ddof=1) if len(excess) > 1 else 0.0
    return float(excess.mean() / std * np.sqrt(TRADING_DAYS)) if std > 0 else 0.0


def _sortino(returns: np.ndarray, rf: float = 0.0) -> float:
    excess = returns - rf / TRADING_DAYS
    downside_deviation = np.sqrt(np.mean(np.minimum(excess, 0.0) ** 2))
    return float(excess.mean() / downside_deviation * np.sqrt(TRADING_DAYS)) if downside_deviation else 0.0


def _drawdown(equity: np.ndarray) -> np.ndarray:
    peaks = np.maximum.accumulate(np.r_[1.0, equity])[1:]
    return equity / peaks - 1.0


def _max_drawdown_duration(drawdown: np.ndarray) -> int:
    longest = current = 0
    for value in drawdown:
        current = current + 1 if value < 0 else 0
        longest = max(longest, current)
    return longest


def _profit_factor(returns: np.ndarray) -> float:
    gains = returns[returns > 0].sum()
    losses = -returns[returns < 0].sum()
    return float(gains / losses) if losses > 0 else (float("inf") if gains > 0 else 0.0)


def _max_drawdown(equity: np.ndarray) -> float:
    return float(_drawdown(equity).min() * 100) if len(equity) else 0.0


def _rolling_sharpe(returns: np.ndarray, window: int = 63, rf: float = 0.0) -> np.ndarray:
    excess = pd.Series(returns - rf / TRADING_DAYS)
    return (excess.rolling(window, min_periods=2).mean() /
            excess.rolling(window, min_periods=2).std() * np.sqrt(TRADING_DAYS)).fillna(0).to_numpy()


def _as_array(values: Iterable[float] | float | None, n: int, default: float) -> np.ndarray:
    if values is None:
        return np.full(n, default, dtype=float)
    if np.isscalar(values):
        return np.full(n, float(values), dtype=float)
    result = np.asarray(values, dtype=float)
    if len(result) != n:
        raise ValueError("Optional market data must be aligned with returns")
    return result


def almgren_chriss_cost(turnover: float, *, spread_bps: float, daily_volatility: float,
                        participation: float, quadratic_impact: float = 0.10,
                        sqrt_impact: float = 0.10) -> float:
    """Return one-way execution cost as a fraction of traded notional.

    Spread is variable per event. Impact combines a volatility-scaled square-root
    term and a quadratic participation term, following the temporary-impact form
    used in Almgren-Chriss style execution models.
    """
    p = max(0.0, float(participation))
    spread_cost = max(0.0, float(spread_bps)) / 20_000.0
    impact = max(0.0, sqrt_impact) * max(0.0, daily_volatility) * np.sqrt(p)
    impact += max(0.0, quadratic_impact) * p * p
    return max(0.0, float(turnover)) * (spread_cost + impact)


def _metrics(returns: np.ndarray, equity: np.ndarray, rf: float) -> dict:
    dd = _drawdown(equity)
    ann = _annualised_return(equity, len(returns))
    tail = returns[returns <= np.quantile(returns, 0.05)] if len(returns) else np.array([])
    es = float(tail.mean() * 100) if len(tail) else 0.0
    return {
        "Total_Return_%": round(float((equity[-1] - 1) * 100), 2) if len(equity) else 0.0,
        "Ann_Return_%": round(ann, 2),
        "Sharpe": round(_sharpe(returns, rf), 3),
        "Sortino": round(_sortino(returns, rf), 3),
        "Calmar": round(ann / abs(dd.min() * 100), 3) if len(dd) and dd.min() < 0 else 0.0,
        "MaxDrawdown_%": round(float(dd.min() * 100), 2) if len(dd) else 0.0,
        "Max_Drawdown_Duration_Days": _max_drawdown_duration(dd),
        "Expected_Shortfall_95_%": round(es, 3),
        "Profit_Factor": round(_profit_factor(returns), 3),
    }


def run_backtest(
    y_true_returns: pd.Series,
    pred_returns: np.ndarray,
    transaction_cost: float = 0.001,
    rf_annual: float = 0.05,
    use_kelly: bool = True,
    *,
    prices: Iterable[float] | None = None,
    volumes: Iterable[float] | None = None,
    spread_bps: Iterable[float] | float | None = None,
    volatility_target: float = 0.15,
    max_drawdown: float = 0.20,
    fractional_kelly: float = 0.5,
    max_leverage: float = 1.0,
    impact_quadratic: float = 0.01,
    impact_sqrt: float = 0.01,
    sizing_window: int = 63,
    breaker_cooldown_bars: int = 20,
) -> dict:
    """Run a sequential daily-bar simulation.

    Returns are supplied in percent. At each step, position sizing uses only
    prior realized returns for volatility and Kelly estimates. The supplied
    prediction determines direction; costs are charged on target-position
    changes and include variable spread plus market impact.
    """
    actual_pct = np.asarray(y_true_returns, dtype=float).reshape(-1)
    pred_pct = np.asarray(pred_returns, dtype=float).reshape(-1)
    if len(actual_pct) != len(pred_pct):
        raise ValueError("Actual and predicted returns must have equal length")
    n = len(actual_pct)
    if n == 0:
        raise ValueError("Backtest requires at least one observation")
    if not np.isfinite(actual_pct).all() or not np.isfinite(pred_pct).all():
        raise ValueError("Returns must be finite")
    if volatility_target <= 0 or not 0 < max_drawdown < 1 or not 0 <= fractional_kelly <= 1:
        raise ValueError("Invalid volatility target, drawdown limit, or fractional Kelly")

    actual = actual_pct / 100.0
    pred = pred_pct / 100.0
    px = _as_array(prices, n, 1.0)
    vol = _as_array(volumes, n, np.inf)
    spreads = _as_array(spread_bps, n, transaction_cost * 10_000.0)
    strategy_returns = np.zeros(n)
    kelly_returns = np.zeros(n)
    positions = np.zeros(n)
    kelly_positions = np.zeros(n)
    costs = np.zeros(n)
    peak_equity = 1.0
    breaker_latched = False
    breaker_triggered = False
    breaker_bars_remaining = 0
    prior_strategy: list[float] = []
    prior_signal = 0.0

    for i in range(n):
        history = np.asarray(prior_strategy[-sizing_window:], dtype=float)
        hist_vol = float(history.std(ddof=1)) if len(history) > 1 else 0.01
        hist_vol = max(hist_vol, 1e-4)
        # Dynamic fractional Kelly: expected edge is the current forecast, while
        # variance is estimated only from outcomes observed before this event.
        kelly = fractional_kelly * max(0.0, abs(pred[i]) / (hist_vol * hist_vol))
        kelly = min(kelly, max_leverage)
        vol_cap = min(max_leverage, volatility_target / (hist_vol * np.sqrt(TRADING_DAYS)))
        raw_size = min(kelly, vol_cap) if use_kelly else vol_cap
        desired = np.sign(pred[i]) * raw_size

        if breaker_latched:
            desired = 0.0
        prior_equity = float(np.prod(1.0 + np.asarray(strategy_returns[:i]))) if i else 1.0
        drawdown = prior_equity / peak_equity - 1.0
        if not breaker_latched and drawdown <= -max_drawdown:
            breaker_latched = True
            breaker_triggered = True
            breaker_bars_remaining = max(1, int(breaker_cooldown_bars))
            desired = 0.0

        turnover = abs(desired - prior_signal)
        if np.isfinite(vol[i]) and vol[i] > 0 and px[i] > 0:
            dollar_volume = vol[i] * px[i]
            participation = turnover * prior_equity / max(dollar_volume, 1e-12)
        else:
            participation = turnover
        cost = turnover * max(transaction_cost, 0.0) + almgren_chriss_cost(
            turnover, spread_bps=spreads[i], daily_volatility=hist_vol,
            participation=participation, quadratic_impact=impact_quadratic,
            sqrt_impact=impact_sqrt,
        )
        gross = desired * actual[i]
        strategy_returns[i] = gross - cost
        positions[i] = desired
        costs[i] = cost

        # Kelly diagnostic sleeve retains the same risk controls but uses the
        # fractional Kelly allocation before volatility scaling.
        kelly_size = 0.0 if breaker_latched else np.sign(pred[i]) * min(kelly, vol_cap)
        kelly_turnover = abs(kelly_size - (kelly_positions[i - 1] if i else 0.0))
        kelly_returns[i] = kelly_size * actual[i] - kelly_turnover * max(transaction_cost, 0.0)
        kelly_positions[i] = kelly_size
        prior_signal = desired
        current_equity = prior_equity * max(1e-12, 1.0 + strategy_returns[i])
        peak_equity = max(peak_equity, current_equity)
        if breaker_latched:
            breaker_bars_remaining -= 1
            if breaker_bars_remaining <= 0:
                breaker_latched = False
        prior_strategy.append(strategy_returns[i])

    bh_returns = actual
    strategy_eq = np.cumprod(1.0 + strategy_returns)
    kelly_eq = np.cumprod(1.0 + kelly_returns)
    bh_eq = np.cumprod(1.0 + bh_returns)
    strategy_metrics = _metrics(strategy_returns, strategy_eq, rf_annual)
    kelly_metrics = _metrics(kelly_returns, kelly_eq, rf_annual)
    bh_metrics = _metrics(bh_returns, bh_eq, rf_annual)

    # Preserve established metric names consumed by the existing UI and scripts.
    metrics = {
        **{f"Strategy_{key}": value for key, value in strategy_metrics.items()},
        **{f"Kelly_{key}": value for key, value in kelly_metrics.items()},
        "BuyHold_Total_Return_%": bh_metrics["Total_Return_%"],
        "BuyHold_Ann_Return_%": bh_metrics["Ann_Return_%"],
        "BuyHold_Sharpe": bh_metrics["Sharpe"],
        "BuyHold_MaxDrawdown_%": bh_metrics["MaxDrawdown_%"],
        "N_Trades": int(np.count_nonzero(np.abs(np.diff(positions, prepend=0)) > 1e-12)),
        "Average_Exposure_%": round(float(np.mean(np.abs(positions)) * 100), 2),
        "Volatility_Target_%": round(volatility_target * 100, 2),
        "Circuit_Breaker_Triggered": breaker_triggered,
    }
    curves = pd.DataFrame({"Strategy": strategy_eq, "Kelly": kelly_eq, "BuyAndHold": bh_eq})
    curves["Strategy_Drawdown"] = _drawdown(strategy_eq)
    curves["Position"] = positions
    curves["Execution_Cost"] = costs
    rolling_sh = _rolling_sharpe(strategy_returns, window=63, rf=rf_annual)
    logger.info("Backtest completed: Sharpe %.3f, max drawdown %.2f%%",
                strategy_metrics["Sharpe"], strategy_metrics["MaxDrawdown_%"])
    return {"metrics": metrics, "curves": curves, "rolling_sharpe": rolling_sh,
            "returns": pd.Series(strategy_returns, index=getattr(y_true_returns, "index", None), name="Strategy"),
            "positions": pd.Series(positions), "execution_costs": pd.Series(costs)}


def create_backtest_dashboard(result: dict):
    """Build an interactive Plotly equity, drawdown, exposure and Sharpe dashboard."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:
        raise RuntimeError("Install plotly to create the backtest dashboard") from exc
    curves = result["curves"]
    figure = make_subplots(rows=3, cols=1, shared_xaxes=True,
                           specs=[[{}], [{}], [{"secondary_y": True}]],
                           vertical_spacing=0.08,
                           subplot_titles=("Equity Curves", "Drawdown", "Exposure / Rolling Sharpe"))
    for name in ("Strategy", "Kelly", "BuyAndHold"):
        figure.add_trace(go.Scatter(y=curves[name], name=name, mode="lines"), row=1, col=1)
    figure.add_trace(go.Scatter(y=curves["Strategy_Drawdown"] * 100, name="Drawdown %",
                                fill="tozeroy", line={"color": "firebrick"}), row=2, col=1)
    figure.add_trace(go.Scatter(y=curves["Position"] * 100, name="Position %"), row=3, col=1, secondary_y=False)
    figure.add_trace(go.Scatter(y=result["rolling_sharpe"], name="Rolling Sharpe"), row=3, col=1, secondary_y=True)
    figure.update_layout(template="plotly_white", height=850, hovermode="x unified",
                         title="Event-Driven Backtest Analytics")
    return figure


def render_backtest_dashboard(result: dict, streamlit_module=None) -> None:
    """Render report metrics and Plotly dashboard in Streamlit."""
    if streamlit_module is None:
        import streamlit as streamlit_module
    metrics = result["metrics"]
    key_metrics = ["Strategy_Sharpe", "Strategy_Sortino", "Strategy_Calmar",
                   "Strategy_MaxDrawdown_%", "Strategy_Max_Drawdown_Duration_Days",
                   "Strategy_Expected_Shortfall_95_%", "Strategy_Profit_Factor"]
    columns = streamlit_module.columns(len(key_metrics))
    for column, key in zip(columns, key_metrics):
        column.metric(key.replace("Strategy_", "").replace("_", " "), metrics[key])
    streamlit_module.plotly_chart(create_backtest_dashboard(result), use_container_width=True)
