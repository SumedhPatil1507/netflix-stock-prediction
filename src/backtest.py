"""
Advanced backtesting engine.
Strategies:
  - Binary long/flat based on model signal
  - Kelly-sized positions (bet proportional to edge)
Benchmarks: Buy-and-hold NFLX, SPY (if available via yfinance)
Metrics: Total/Ann return, Sharpe, Sortino, Calmar, Max Drawdown,
         Win Rate, Rolling Sharpe, Kelly fraction
"""
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Core metrics ──────────────────────────────────────────────────────────────
def _annualised_return(equity: np.ndarray, n_days: int) -> float:
    n_years = n_days / 252
    return (equity[-1] ** (1 / n_years) - 1) * 100 if n_years > 0 else 0.0


def _sharpe(rets: np.ndarray, rf: float = 0.0) -> float:
    excess = rets - rf / 252
    return float(excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 0 else 0.0


def _sortino(rets: np.ndarray, rf: float = 0.0) -> float:
    excess   = rets - rf / 252
    downside = excess[excess < 0].std()
    return float(excess.mean() / downside * np.sqrt(252)) if downside > 0 else 0.0


def _max_drawdown(equity: np.ndarray) -> float:
    roll_max = np.maximum.accumulate(equity)
    dd = (equity - roll_max) / roll_max
    return float(dd.min() * 100)


def _calmar(ann_return: float, max_dd: float) -> float:
    return ann_return / abs(max_dd) if max_dd != 0 else 0.0


def _kelly_fraction(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """Full Kelly fraction. Cap at 0.25 for safety."""
    if avg_loss == 0:
        return 0.0
    b = avg_win / avg_loss
    f = (b * win_rate - (1 - win_rate)) / b
    return float(np.clip(f, 0, 0.25))


def _rolling_sharpe(rets: np.ndarray, window: int = 63) -> np.ndarray:
    s = pd.Series(rets)
    roll_mean = s.rolling(window).mean()
    roll_std  = s.rolling(window).std()
    return (roll_mean / roll_std * np.sqrt(252)).fillna(0).values


# ── Main backtest ─────────────────────────────────────────────────────────────
def run_backtest(
    y_true_returns: pd.Series,
    pred_returns: np.ndarray,
    transaction_cost: float = 0.001,
    rf_annual: float = 0.05,
    use_kelly: bool = True,
) -> dict:
    """
    Parameters
    ----------
    y_true_returns  : actual next-day returns (%) aligned with predictions
    pred_returns    : model predicted next-day returns (%)
    transaction_cost: one-way cost as fraction (default 0.1%)
    rf_annual       : annual risk-free rate for Sharpe/Sortino
    use_kelly       : if True, also compute Kelly-sized strategy

    Returns
    -------
    dict: metrics, curves (DataFrame), rolling_sharpe (array)
    """
    actual = np.array(y_true_returns) / 100
    pred   = np.array(pred_returns)
    n      = len(actual)

    # ── Binary signal strategy ────────────────────────────────────────────────
    signal = (pred > 0).astype(float)
    trades = np.abs(np.diff(signal, prepend=0))
    costs  = trades * transaction_cost
    strat_rets = signal * actual - costs

    # ── Kelly-sized strategy ──────────────────────────────────────────────────
    wins     = actual[signal == 1]
    avg_win  = wins[wins > 0].mean() if (wins > 0).any() else 0.0
    avg_loss = abs(wins[wins < 0].mean()) if (wins < 0).any() else 1e-6
    win_rate = float(np.mean(wins > 0)) if len(wins) > 0 else 0.5
    kf       = _kelly_fraction(win_rate, avg_win, avg_loss)

    kelly_signal = np.clip(pred / (np.std(pred) + 1e-8), 0, 1) * kf
    kelly_rets   = kelly_signal * actual - trades * transaction_cost

    # ── Buy-and-hold ──────────────────────────────────────────────────────────
    bh_rets = actual

    # ── Equity curves ─────────────────────────────────────────────────────────
    strat_eq  = (1 + strat_rets).cumprod()
    kelly_eq  = (1 + kelly_rets).cumprod()
    bh_eq     = (1 + bh_rets).cumprod()

    # ── Metrics ───────────────────────────────────────────────────────────────
    strat_ann  = _annualised_return(strat_eq, n)
    strat_mdd  = _max_drawdown(strat_eq)
    bh_ann     = _annualised_return(bh_eq, n)
    bh_mdd     = _max_drawdown(bh_eq)

    metrics = {
        # Binary strategy
        "Strategy_Total_Return_%":  round((strat_eq[-1] - 1) * 100, 2),
        "Strategy_Ann_Return_%":    round(strat_ann, 2),
        "Strategy_Sharpe":          round(_sharpe(strat_rets, rf_annual), 3),
        "Strategy_Sortino":         round(_sortino(strat_rets, rf_annual), 3),
        "Strategy_Calmar":          round(_calmar(strat_ann, strat_mdd), 3),
        "Strategy_MaxDrawdown_%":   round(strat_mdd, 2),
        "Strategy_WinRate_%":       round(float(np.mean((signal * actual) > 0) * 100), 2),
        "N_Trades":                 int(trades.sum()),
        # Kelly strategy
        "Kelly_Fraction":           round(kf, 4),
        "Kelly_Total_Return_%":     round((kelly_eq[-1] - 1) * 100, 2),
        "Kelly_Sharpe":             round(_sharpe(kelly_rets, rf_annual), 3),
        # Buy-and-hold benchmark
        "BuyHold_Total_Return_%":   round((bh_eq[-1] - 1) * 100, 2),
        "BuyHold_Ann_Return_%":     round(bh_ann, 2),
        "BuyHold_Sharpe":           round(_sharpe(bh_rets, rf_annual), 3),
        "BuyHold_MaxDrawdown_%":    round(bh_mdd, 2),
    }

    curves = pd.DataFrame({
        "Strategy":   strat_eq,
        "Kelly":      kelly_eq,
        "BuyAndHold": bh_eq,
    })

    rolling_sh = _rolling_sharpe(strat_rets, window=63)

    logger.info("Backtest complete")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v}")

    return {"metrics": metrics, "curves": curves, "rolling_sharpe": rolling_sh}
