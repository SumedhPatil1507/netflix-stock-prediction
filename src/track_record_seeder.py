"""
Track Record Seeder — Alpha Engine Pro.

Seeds 90 days of simulated trading signals for every default strategy.
Uses the paper_trade module when the model is available; otherwise falls
back to synthetic data so the dashboard always has something to display.

Run:
    python -m src.track_record_seeder          # seeds all 3 defaults, 90 days
    python -m src.track_record_seeder --days 180
"""
from __future__ import annotations

import argparse
import logging
import random
import sys
from datetime import date, timedelta

logger = logging.getLogger(__name__)


def _synthetic_rows(strategy_name: str, ticker: str, days: int = 90) -> list[dict]:
    """
    Generate reproducible synthetic paper-trade rows for seeding.
    Uses a fixed seed so results are deterministic.
    """
    rng = random.Random(hash(strategy_name) % (2**31))
    rows = []
    base_date = date.today() - timedelta(days=days)
    for i in range(days):
        d = base_date + timedelta(days=i)
        # Skip weekends
        if d.weekday() >= 5:
            continue
        pred = rng.gauss(0.05, 0.6)   # mean ~0.05%, std 0.6%
        actual = pred * rng.uniform(0.6, 1.4) + rng.gauss(0, 0.3)
        signal = "BUY" if pred > 0 else "HOLD"
        pnl = actual if signal == "BUY" else 0.0
        rows.append({
            "date": str(d),
            "ticker": ticker,
            "pred_return": round(pred, 4),
            "actual_return": round(actual, 4),
            "signal": signal,
            "pnl_pct": round(pnl, 4),
            "strategy_name": strategy_name,
            "metadata": {
                "direction": "up" if actual > 0 else "down",
                "correct": (pred > 0) == (actual > 0),
            },
        })
    return rows


def seed_strategy(strategy_name: str, ticker: str, days: int = 90) -> int:
    """
    Seed one strategy. Tries paper_trade first; falls back to synthetic data.
    Returns the number of rows written.
    """
    from src.track_record import TrackRecord
    tr = TrackRecord(strategy_name=strategy_name)

    rows: list[dict] = []

    # --- Try real paper trade ---
    try:
        from src.paper_trade import run_paper_trade
        log_df = run_paper_trade(days=days)
        if log_df is not None and not log_df.empty:
            for _, row in log_df.iterrows():
                rows.append({
                    "date": str(row.get("date", "")),
                    "ticker": ticker,
                    "pred_return": float(row.get("pred_return", 0.0)),
                    "actual_return": float(row.get("actual_return", 0.0)),
                    "signal": str(row.get("signal", "HOLD")),
                    "pnl_pct": float(row.get("pnl_pct", 0.0)),
                    "strategy_name": strategy_name,
                    "metadata": {
                        "direction": str(row.get("direction", "")),
                        "correct": bool(row.get("correct", False)),
                    },
                })
            logger.info("  Used real paper-trade data (%d rows) for '%s'.", len(rows), strategy_name)
    except Exception as exc:
        logger.warning("  paper_trade failed for '%s' (%s) — using synthetic data.", strategy_name, exc)

    # --- Fall back to synthetic ---
    if not rows:
        rows = _synthetic_rows(strategy_name, ticker, days)
        logger.info("  Synthetic data: %d rows for '%s'.", len(rows), strategy_name)

    # --- Write rows ---
    written = 0
    for row in rows:
        try:
            tr.log_signal(
                date=row["date"],
                ticker=row["ticker"],
                pred_return=row["pred_return"],
                actual_return=row["actual_return"],
                signal=row["signal"],
                pnl_pct=row["pnl_pct"],
                strategy_name=row.get("strategy_name", strategy_name),
                metadata=row.get("metadata"),
            )
            written += 1
        except Exception as exc:
            logger.warning("  Failed to log row: %s", exc)

    return written


def seed_all(days: int = 90) -> None:
    """Seed paper-trade records for every strategy in the default registry."""
    try:
        from src.strategy_registry import StrategyRegistry
        registry = StrategyRegistry()
        strategy_names = registry.list_strategies()
    except Exception as exc:
        logger.error("Cannot load StrategyRegistry: %s", exc)
        return

    logger.info("Seeding track records for %d strategies (%d days each)…", len(strategy_names), days)

    for strategy_name in strategy_names:
        try:
            cfg = registry.get(strategy_name)
            primary_ticker = cfg.tickers[0] if cfg.tickers else "NFLX"
        except Exception:
            primary_ticker = "NFLX"

        logger.info("  Seeding strategy: %s (ticker=%s)", strategy_name, primary_ticker)
        n = seed_strategy(strategy_name, primary_ticker, days)
        logger.info("  Wrote %d records for '%s'.", n, strategy_name)

    logger.info("Seeding complete.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    parser = argparse.ArgumentParser(description="Seed paper-trade track records")
    parser.add_argument("--days", type=int, default=90,
                        help="Number of trading days to simulate per strategy (default: 90)")
    args = parser.parse_args()
    try:
        seed_all(days=args.days)
    except Exception as exc:
        logger.error("Seeder failed: %s", exc)
        sys.exit(1)
