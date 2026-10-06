"""
Track Record — Alpha Engine Pro verifiable performance log.

Provides an append-only JSONL log of every paper/live signal, together
with a suite of realised performance metrics that form the evidence layer
for a freelance sales pitch.

Usage
-----
    from src.track_record import TrackRecord

    tr = TrackRecord("nflx_momentum")
    tr.log_signal(
        date="2024-06-01",
        ticker="NFLX",
        pred_return=0.8,
        actual_return=0.6,
        signal="BUY",
        pnl_pct=0.6,
    )
    metrics = tr.compute_metrics()
    print(metrics)
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_LOG_PATH = "outputs/track_record.jsonl"

# ── Small helpers ─────────────────────────────────────────────────────────────


def _annualised_return(daily_pct_series: pd.Series, trading_days: int = 252) -> float:
    """Compound annualised return from a daily-pct series (values in %, not fractions)."""
    if daily_pct_series.empty:
        return 0.0
    daily_frac = daily_pct_series / 100.0
    cum = (1 + daily_frac).prod()
    n = len(daily_frac)
    if n == 0 or cum <= 0:
        return 0.0
    return float((cum ** (trading_days / n) - 1) * 100)


def _max_drawdown(equity: pd.Series) -> float:
    """Peak-to-trough drawdown as a positive fraction (e.g. 0.15 = 15 %)."""
    if equity.empty or equity.max() == 0:
        return 0.0
    rolling_peak = equity.cummax()
    drawdown = (equity - rolling_peak) / rolling_peak
    return float(abs(drawdown.min()))


def _empty_metrics() -> dict:
    return {
        "n_signals": 0,
        "annualised_return_pct": 0.0,
        "sharpe": 0.0,
        "sortino": 0.0,
        "calmar": 0.0,
        "max_drawdown": 0.0,
        "profit_factor": 0.0,
        "trailing_sharpe_63d": 0.0,
        "win_rate_pct": 0.0,
        "total_pnl_pct": 0.0,
    }


# ── Core class ────────────────────────────────────────────────────────────────


class TrackRecord:
    """
    Append-only JSONL log of trading signals with realised outcomes,
    plus a suite of portfolio-level performance metrics.

    Parameters
    ----------
    strategy_name : Name of the strategy being tracked.
    log_path      : Path to the JSONL log file (default: outputs/track_record.jsonl).
    """

    def __init__(
        self,
        strategy_name: str,
        log_path: str = _DEFAULT_LOG_PATH,
    ) -> None:
        self.strategy_name = strategy_name
        self.log_path = log_path
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)

    # ── Write path ────────────────────────────────────────────────────────────

    def log_signal(
        self,
        date: Any,
        ticker: str,
        pred_return: float,
        actual_return: float,
        signal: str,
        pnl_pct: float,
        strategy_name: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        """
        Append one signal record to the JSONL log.

        Parameters
        ----------
        date          : Trade date (str, date, or datetime — stored as str).
        ticker        : Stock symbol.
        pred_return   : Model's predicted next-day return (%).
        actual_return : Realised next-day return (%).
        signal        : "BUY" | "HOLD" | "SELL".
        pnl_pct       : P&L for this signal period (%).
        strategy_name : Override the instance strategy name if supplied.
        metadata      : Any extra key/value pairs to attach to the record.
        """
        record: dict[str, Any] = {
            "strategy": strategy_name or self.strategy_name,
            "date": str(date),
            "ticker": ticker,
            "pred_return": float(pred_return),
            "actual_return": float(actual_return),
            "signal": signal,
            "pnl_pct": float(pnl_pct),
        }
        if metadata:
            record["metadata"] = metadata

        try:
            with open(self.log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
        except Exception as exc:  # pragma: no cover
            logger.error("TrackRecord.log_signal failed: %s", exc)

    # ── Read path ─────────────────────────────────────────────────────────────

    def load(self) -> pd.DataFrame:
        """
        Load the full JSONL log into a DataFrame.

        Filters to rows matching ``self.strategy_name`` so multiple
        strategies can share a single log file.

        Returns an empty DataFrame (not raises) if the file is missing or empty.
        """
        if not os.path.exists(self.log_path):
            return pd.DataFrame()

        rows: list[dict] = []
        try:
            with open(self.log_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.warning("Skipping malformed JSONL line.")
        except Exception as exc:  # pragma: no cover
            logger.error("TrackRecord.load failed: %s", exc)
            return pd.DataFrame()

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        # Filter to this strategy
        if "strategy" in df.columns:
            df = df[df["strategy"] == self.strategy_name]
        return df.reset_index(drop=True)

    # ── Analytics ─────────────────────────────────────────────────────────────

    def get_equity_curve(self) -> pd.Series:
        """
        Cumulative sum of ``pnl_pct`` — i.e. additive P&L equity curve.

        Returns an empty Series if there are no records.
        """
        df = self.load()
        if df.empty or "pnl_pct" not in df.columns:
            return pd.Series(dtype=float)
        return df["pnl_pct"].cumsum().reset_index(drop=True)

    def get_drawdown_series(self) -> pd.Series:
        """
        Rolling drawdown series: (rolling_peak − current) / rolling_peak.

        Values are non-negative fractions.  Returns empty Series if no data.
        """
        equity = self.get_equity_curve()
        if equity.empty:
            return pd.Series(dtype=float)
        # Shift to start from 100 so early negatives don't break peak calc
        base = 100.0 + equity
        rolling_peak = base.cummax()
        dd = (rolling_peak - base) / rolling_peak.where(rolling_peak != 0, other=1.0)
        return dd

    def compute_metrics(self, window_days: int = 252) -> dict:
        """
        Compute annualised performance metrics over the trailing ``window_days``.

        Metrics returned
        ----------------
        n_signals              : number of log entries considered
        annualised_return_pct  : compound annual growth (%)
        sharpe                 : annualised Sharpe (sqrt(252)·μ/σ of daily pnl_pct)
        sortino                : annualised Sortino (sqrt(252)·μ/σ_downside)
        calmar                 : annualised_return / max_drawdown
        max_drawdown           : peak-to-trough drawdown (fraction, e.g. 0.12)
        profit_factor          : sum_wins / |sum_losses|  (∞ → 999.0 if no losses)
        trailing_sharpe_63d    : Sharpe over the last 63 trading days
        win_rate_pct           : % of BUY signals with positive pnl_pct
        total_pnl_pct          : sum of all pnl_pct values

        Returns zeros dict gracefully on empty log.
        """
        df = self.load()
        if df.empty or "pnl_pct" not in df.columns:
            return _empty_metrics()

        # Use only the trailing window
        df = df.tail(window_days)
        if df.empty:
            return _empty_metrics()

        pnl = df["pnl_pct"].astype(float)
        n = len(pnl)
        ann_factor = float(np.sqrt(252))

        # Sharpe
        mu = pnl.mean()
        sigma = pnl.std(ddof=1) if n > 1 else 0.0
        sharpe = float(ann_factor * mu / sigma) if sigma > 0 else 0.0

        # Sortino
        downside = pnl[pnl < 0]
        downside_std = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
        sortino = float(ann_factor * mu / downside_std) if downside_std > 0 else 0.0

        # Max drawdown via equity curve
        equity = (100.0 + pnl.cumsum())
        mdd = _max_drawdown(equity)

        # Calmar
        ann_ret = _annualised_return(pnl)
        calmar = float(ann_ret / (mdd * 100)) if mdd > 0 else 0.0

        # Profit factor
        wins = pnl[pnl > 0].sum()
        losses = abs(pnl[pnl < 0].sum())
        profit_factor = float(wins / losses) if losses > 0 else (999.0 if wins > 0 else 0.0)

        # Trailing 63-day Sharpe
        tail63 = pnl.tail(63)
        mu63 = tail63.mean()
        sigma63 = tail63.std(ddof=1) if len(tail63) > 1 else 0.0
        trailing_sharpe = float(ann_factor * mu63 / sigma63) if sigma63 > 0 else 0.0

        # Win rate (BUY signals only)
        buy_mask = df.get("signal", pd.Series(dtype=str)) == "BUY"
        buy_rows = df[buy_mask]
        win_rate = (
            float((buy_rows["pnl_pct"] > 0).mean() * 100) if not buy_rows.empty else 0.0
        )

        return {
            "n_signals": n,
            "annualised_return_pct": round(ann_ret, 4),
            "sharpe": round(sharpe, 4),
            "sortino": round(sortino, 4),
            "calmar": round(calmar, 4),
            "max_drawdown": round(mdd, 4),
            "profit_factor": round(profit_factor, 4),
            "trailing_sharpe_63d": round(trailing_sharpe, 4),
            "win_rate_pct": round(win_rate, 2),
            "total_pnl_pct": round(float(pnl.sum()), 4),
        }
