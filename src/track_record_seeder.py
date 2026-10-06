"""
Track Record Seeder — Alpha Engine Pro.

Seeds 90 days of simulated paper-trade signals for every default strategy
in the StrategyRegistry, writing each row into the shared
``outputs/track_record.jsonl`` log via TrackRecord.

Run once after training the model to populate the track-record dashboard:

    python -m src.track_record_seeder          # seeds all 3 defaults, 90 days
    python -m src.track_record_seeder --days 180
"""
from __future__ import annotations

import argparse
import logging
import sys

logger = logging.getLogger(__name__)


def seed_all(days: int = 90) -> None:
    """
    Seed paper-trade records for every strategy in the default registry.

    For each strategy the seeder:
      1. Calls ``paper_trade.run_paper_trade(days=days)`` (uses the trained model).
      2. Iterates the resulting DataFrame rows.
      3. Writes each row to ``TrackRecord`` for that strategy name.

    If ``run_paper_trade`` fails (e.g. model not trained yet) the strategy is
    skipped with a warning — the seeder never raises.

    Parameters
    ----------
    days : Number of trailing trading days to simulate per strategy.
    """
    try:
        from src.strategy_registry import StrategyRegistry
    except Exception as exc:  # pragma: no cover
        logger.error("Cannot import StrategyRegistry: %s", exc)
        return

    try:
        from src.track_record import TrackRecord
    except Exception as exc:  # pragma: no cover
        logger.error("Cannot import TrackRecord: %s", exc)
        return

    try:
        from src.paper_trade import run_paper_trade
    except Exception as exc:  # pragma: no cover
        logger.error("Cannot import run_paper_trade: %s", exc)
        return

    try:
        registry = StrategyRegistry()
        strategy_names = registry.list_strategies()
    except Exception as exc:  # pragma: no cover
        logger.error("Failed to load StrategyRegistry: %s", exc)
        return

    logger.info("Seeding track records for %d strategies (%d days each) …", len(strategy_names), days)

    for strategy_name in strategy_names:
        logger.info("  Seeding strategy: %s", strategy_name)
        tr = TrackRecord(strategy_name=strategy_name)

        try:
            log_df = run_paper_trade(days=days)
        except Exception as exc:
            logger.warning(
                "  run_paper_trade failed for '%s' (model not trained?): %s — skipping.",
                strategy_name,
                exc,
            )
            continue

        if log_df is None or log_df.empty:
            logger.warning("  No paper-trade rows returned for '%s' — skipping.", strategy_name)
            continue

        try:
            cfg = registry.get(strategy_name)
            primary_ticker = cfg.tickers[0] if cfg.tickers else "UNKNOWN"
        except Exception:
            primary_ticker = "UNKNOWN"

        row_count = 0
        for _, row in log_df.iterrows():
            try:
                tr.log_signal(
                    date=row.get("date", ""),
                    ticker=primary_ticker,
                    pred_return=float(row.get("pred_return", 0.0)),
                    actual_return=float(row.get("actual_return", 0.0)),
                    signal=str(row.get("signal", "HOLD")),
                    pnl_pct=float(row.get("pnl_pct", 0.0)),
                    strategy_name=strategy_name,
                    metadata={
                        "direction": str(row.get("direction", "")),
                        "correct": bool(row.get("correct", False)),
                    },
                )
                row_count += 1
            except Exception as exc:
                logger.warning("  Failed to log row: %s", exc)
                continue

        logger.info("  Seeded %d records for strategy '%s'.", row_count, strategy_name)

    logger.info("Seeding complete.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    parser = argparse.ArgumentParser(description="Seed paper-trade track records")
    parser.add_argument(
        "--days",
        type=int,
        default=90,
        help="Number of trading days to simulate per strategy (default: 90)",
    )
    args = parser.parse_args()

    try:
        seed_all(days=args.days)
    except Exception as exc:
        logger.error("Seeder encountered an unexpected error: %s", exc)
        sys.exit(1)
