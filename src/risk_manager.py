"""
Risk & Position Management Layer.
Converts model signals into execution-ready position sizes
with hard risk controls.

Replaces simple Kelly fraction with a full risk matrix:
  - Max position size (% of portfolio)
  - Stop-loss levels (ATR-based and fixed %)
  - Take-profit levels
  - Portfolio heat (total open risk)
  - Drawdown circuit breaker
  - Volatility-adjusted sizing
"""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class RiskConfig:
    """All risk parameters in one place."""
    portfolio_value:     float = 100_000.0   # total capital ($)
    max_position_pct:    float = 0.05        # max 5% per position
    max_portfolio_heat:  float = 0.20        # max 20% total open risk
    kelly_fraction:      float = 0.25        # max Kelly cap
    atr_stop_multiplier: float = 2.0         # stop = entry - 2×ATR
    fixed_stop_pct:      float = 0.02        # hard stop at 2% loss
    take_profit_ratio:   float = 2.0         # TP = 2× stop distance (R:R)
    max_drawdown_halt:   float = 0.10        # halt trading at 10% drawdown
    min_confidence:      float = 0.0         # min predicted return to trade
    commission_per_share: float = 0.005      # $0.005/share (IBKR-like)


@dataclass
class PositionOrder:
    """Execution-ready position order."""
    ticker:          str
    signal:          str          # BUY | HOLD | SELL
    shares:          int
    entry_price:     float
    stop_loss:       float
    take_profit:     float
    position_value:  float
    risk_per_trade:  float        # $ at risk
    risk_pct:        float        # % of portfolio at risk
    kelly_fraction:  float
    confidence_interval: Optional[dict] = None
    notes:           str = ""

    def to_dict(self) -> dict:
        return {
            "ticker":           self.ticker,
            "signal":           self.signal,
            "shares":           self.shares,
            "entry_price":      round(self.entry_price, 2),
            "stop_loss":        round(self.stop_loss, 2),
            "take_profit":      round(self.take_profit, 2),
            "position_value":   round(self.position_value, 2),
            "risk_per_trade":   round(self.risk_per_trade, 2),
            "risk_pct":         round(self.risk_pct * 100, 3),
            "kelly_fraction":   round(self.kelly_fraction, 4),
            "confidence_interval": self.confidence_interval,
            "notes":            self.notes,
        }


class RiskManager:
    """
    Converts model predictions into execution-ready orders
    with full risk controls.
    """

    def __init__(self, config: Optional[RiskConfig] = None):
        self.cfg = config or RiskConfig()
        self._open_risk: float = 0.0   # current portfolio heat
        self._peak_value: float = self.cfg.portfolio_value
        self._current_value: float = self.cfg.portfolio_value

    def update_portfolio_value(self, value: float) -> None:
        """Call after each trade to update drawdown tracking."""
        self._current_value = value
        self._peak_value    = max(self._peak_value, value)

    @property
    def current_drawdown(self) -> float:
        """Current drawdown from peak as fraction."""
        if self._peak_value == 0:
            return 0.0
        return (self._peak_value - self._current_value) / self._peak_value

    @property
    def is_halted(self) -> bool:
        """True if drawdown circuit breaker is triggered."""
        return self.current_drawdown >= self.cfg.max_drawdown_halt

    def compute_position(
        self,
        ticker:       str,
        pred_return:  float,
        last_price:   float,
        atr:          float,
        win_rate:     float = 0.52,
        avg_win_pct:  float = 1.5,
        avg_loss_pct: float = 1.0,
        ci:           Optional[dict] = None,
    ) -> PositionOrder:
        """
        Compute a full position order from model output.

        Parameters
        ----------
        ticker       : stock symbol
        pred_return  : model's predicted next-day return (%)
        last_price   : current price
        atr          : Average True Range (for stop placement)
        win_rate     : historical directional accuracy (fraction)
        avg_win_pct  : average winning trade return (%)
        avg_loss_pct : average losing trade return (%)
        ci           : conformal prediction interval dict
        """
        # ── Circuit breaker ───────────────────────────────────────────────────
        if self.is_halted:
            return PositionOrder(
                ticker=ticker, signal="HALT", shares=0,
                entry_price=last_price, stop_loss=0, take_profit=0,
                position_value=0, risk_per_trade=0, risk_pct=0,
                kelly_fraction=0, notes="Drawdown circuit breaker triggered",
            )

        # ── Signal ────────────────────────────────────────────────────────────
        if pred_return <= self.cfg.min_confidence:
            return PositionOrder(
                ticker=ticker, signal="HOLD", shares=0,
                entry_price=last_price, stop_loss=0, take_profit=0,
                position_value=0, risk_per_trade=0, risk_pct=0,
                kelly_fraction=0, notes="Predicted return below minimum confidence",
            )

        # ── Kelly fraction ────────────────────────────────────────────────────
        b  = avg_win_pct / avg_loss_pct if avg_loss_pct > 0 else 1.0
        kf = (b * win_rate - (1 - win_rate)) / b
        kf = float(np.clip(kf, 0, self.cfg.kelly_fraction))

        # ── Stop loss (ATR-based, with fixed % floor) ─────────────────────────
        atr_stop   = last_price - self.cfg.atr_stop_multiplier * atr
        fixed_stop = last_price * (1 - self.cfg.fixed_stop_pct)
        stop_loss  = max(atr_stop, fixed_stop)   # tighter of the two

        # ── Take profit (R:R ratio) ───────────────────────────────────────────
        stop_dist   = last_price - stop_loss
        take_profit = last_price + stop_dist * self.cfg.take_profit_ratio

        # ── Position sizing ───────────────────────────────────────────────────
        # Risk per share = distance to stop
        risk_per_share = max(last_price - stop_loss, 0.01)

        # Max $ risk per trade = portfolio × max_position_pct × kelly
        max_risk_dollars = self.cfg.portfolio_value * self.cfg.max_position_pct * kf

        # Check portfolio heat
        remaining_heat = (self.cfg.max_portfolio_heat * self.cfg.portfolio_value
                          - self._open_risk)
        max_risk_dollars = min(max_risk_dollars, remaining_heat)

        if max_risk_dollars <= 0:
            return PositionOrder(
                ticker=ticker, signal="HOLD", shares=0,
                entry_price=last_price, stop_loss=stop_loss, take_profit=take_profit,
                position_value=0, risk_per_trade=0, risk_pct=0,
                kelly_fraction=kf, notes="Portfolio heat limit reached",
            )

        shares = int(max_risk_dollars / risk_per_share)
        if shares < 1:
            shares = 1

        position_value = shares * last_price
        risk_per_trade = shares * risk_per_share
        risk_pct       = risk_per_trade / self.cfg.portfolio_value

        # Commission
        commission = shares * self.cfg.commission_per_share
        notes = f"Commission est: ${commission:.2f} | ATR: {atr:.2f}"

        return PositionOrder(
            ticker=ticker, signal="BUY",
            shares=shares,
            entry_price=last_price,
            stop_loss=round(stop_loss, 2),
            take_profit=round(take_profit, 2),
            position_value=round(position_value, 2),
            risk_per_trade=round(risk_per_trade, 2),
            risk_pct=round(risk_pct, 4),
            kelly_fraction=round(kf, 4),
            confidence_interval=ci,
            notes=notes,
        )

    def risk_matrix(
        self,
        pred_return: float,
        last_price:  float,
        atr:         float,
        portfolio_value: Optional[float] = None,
    ) -> dict:
        """
        Return a full risk matrix for display in the dashboard.
        """
        pv = portfolio_value or self.cfg.portfolio_value
        stop_atr   = last_price - self.cfg.atr_stop_multiplier * atr
        stop_fixed = last_price * (1 - self.cfg.fixed_stop_pct)
        stop       = max(stop_atr, stop_fixed)
        tp         = last_price + (last_price - stop) * self.cfg.take_profit_ratio

        return {
            "entry_price":       round(last_price, 2),
            "stop_loss_atr":     round(stop_atr, 2),
            "stop_loss_fixed":   round(stop_fixed, 2),
            "stop_loss_used":    round(stop, 2),
            "take_profit":       round(tp, 2),
            "risk_reward_ratio": round(self.cfg.take_profit_ratio, 1),
            "max_position_pct":  f"{self.cfg.max_position_pct*100:.0f}%",
            "max_portfolio_heat":f"{self.cfg.max_portfolio_heat*100:.0f}%",
            "current_drawdown":  f"{self.current_drawdown*100:.2f}%",
            "circuit_breaker":   f"{self.cfg.max_drawdown_halt*100:.0f}%",
            "is_halted":         self.is_halted,
            "predicted_return":  f"{pred_return:+.3f}%",
        }
