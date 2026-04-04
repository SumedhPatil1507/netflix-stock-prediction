"""
Simple backtesting engine.
Strategy: go long if model predicts positive return, stay flat otherwise.
Compares against buy-and-hold benchmark.
"""
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def run_backtest(y_true_returns: pd.Series, pred_returns: np.ndarray,
                 transaction_cost: float = 0.001) -> dict:
    """
    Parameters
    ----------
    y_true_returns  : actual next-day returns (%) aligned with predictions
    pred_returns    : model's predicted next-day returns (%)
    transaction_cost: one-way cost as fraction of trade value (default 0.1%)

    Returns
    -------
    dict with strategy metrics and equity curves
    """
    actual = np.array(y_true_returns) / 100   # convert % → fraction
    pred   = np.array(pred_returns)

    # Signal: 1 = long, 0 = flat  (no shorting)
    signal = (pred > 0).astype(float)

    # Detect trades (signal changes) to apply transaction cost
    trades      = np.abs(np.diff(signal, prepend=0))
    cost_series = trades * transaction_cost

    # Strategy daily return
    strat_returns = signal * actual - cost_series

    # Buy-and-hold daily return
    bh_returns = actual

    # Equity curves (cumulative)
    strat_equity = (1 + strat_returns).cumprod()
    bh_equity    = (1 + bh_returns).cumprod()

    n_days = len(actual)
    n_years = n_days / 252

    def _annualised_return(equity):
        return (equity[-1] ** (1 / n_years) - 1) * 100

    def _sharpe(rets, rf=0.0):
        excess = rets - rf / 252
        return (excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 0 else 0.0

    def _max_drawdown(equity):
        roll_max = np.maximum.accumulate(equity)
        dd = (equity - roll_max) / roll_max
        return dd.min() * 100

    def _sortino(rets, rf=0.0):
        excess    = rets - rf / 252
        downside  = excess[excess < 0].std()
        return (excess.mean() / downside * np.sqrt(252)) if downside > 0 else 0.0

    metrics = {
        "Strategy_Total_Return_%":    round((strat_equity[-1] - 1) * 100, 2),
        "BuyHold_Total_Return_%":     round((bh_equity[-1] - 1) * 100, 2),
        "Strategy_Ann_Return_%":      round(_annualised_return(strat_equity), 2),
        "BuyHold_Ann_Return_%":       round(_annualised_return(bh_equity), 2),
        "Strategy_Sharpe":            round(_sharpe(strat_returns), 3),
        "BuyHold_Sharpe":             round(_sharpe(bh_returns), 3),
        "Strategy_Sortino":           round(_sortino(strat_returns), 3),
        "Strategy_MaxDrawdown_%":     round(_max_drawdown(strat_equity), 2),
        "BuyHold_MaxDrawdown_%":      round(_max_drawdown(bh_equity), 2),
        "N_Trades":                   int(trades.sum()),
        "Win_Rate_%":                 round(np.mean((signal * actual) > 0) * 100, 2),
    }

    curves = pd.DataFrame({
        "Strategy": strat_equity,
        "BuyAndHold": bh_equity,
    })

    logger.info("Backtest complete")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v}")

    return {"metrics": metrics, "curves": curves}
