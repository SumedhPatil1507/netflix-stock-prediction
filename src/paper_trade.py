"""
Paper Trading Simulation.
Runs the trained model on the last N days of live data and logs
each prediction vs actual outcome — simulating real deployment.

Usage:
    python -m src.paper_trade          # last 90 days
    python -m src.paper_trade --days 180
"""
from __future__ import annotations
import argparse
import logging
import os
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
LOG_PATH = "outputs/paper_trade_log.csv"


def run_paper_trade(days: int = 90) -> pd.DataFrame:
    """
    Simulate running the model daily for the last `days` trading days.

    For each day t:
      - Input: OHLCV rows up to and including day t
      - Prediction: model's predicted return for day t+1
      - Actual: realised return on day t+1
      - Signal: BUY if predicted > 0, else HOLD
      - Outcome: CORRECT if signal direction matches actual direction

    Returns a DataFrame with one row per trading day.
    """
    from src.modeling import load_model
    from src.data_loader import load_data
    from src.preprocessing import preprocess_data
    from src.feature_utils import build_prediction_row

    logger.info(f"Loading model and data for {days}-day paper trade...")
    model = load_model()

    df_raw = load_data(source="live")
    df     = preprocess_data(df_raw)

    # Need at least 250 rows of history for features to stabilise
    min_history = 250
    if len(df) < min_history + days:
        logger.warning("Not enough data for requested days. Using available data.")
        days = max(1, len(df) - min_history)

    records = []
    total   = len(df)

    for i in range(total - days - 1, total - 1):
        # Window: all data up to and including day i
        window = df.iloc[max(0, i - 249): i + 1][["Open", "High", "Low", "Close", "Volume"]]

        try:
            row      = build_prediction_row(window, model)
            pred_ret = float(model.predict(row)[0])
        except Exception as e:
            logger.warning(f"Prediction failed on index {i}: {e}")
            continue

        curr_close = float(df["Close"].iloc[i])
        next_close = float(df["Close"].iloc[i + 1])
        actual_ret = (next_close - curr_close) / curr_close * 100

        signal    = "BUY"  if pred_ret  > 0 else "HOLD"
        direction = "UP"   if actual_ret > 0 else "DOWN"
        correct   = (pred_ret > 0) == (actual_ret > 0)

        records.append({
            "date":         df.index[i + 1].date(),
            "prev_close":   round(curr_close, 2),
            "next_close":   round(next_close, 2),
            "pred_return":  round(pred_ret, 4),
            "actual_return":round(actual_ret, 4),
            "signal":       signal,
            "direction":    direction,
            "correct":      correct,
            "pnl_pct":      round(actual_ret if signal == "BUY" else 0.0, 4),
        })

    log_df = pd.DataFrame(records)

    if log_df.empty:
        logger.warning("No paper trade records generated.")
        return log_df

    # ── Summary stats ─────────────────────────────────────────────────────────
    dir_acc   = log_df["correct"].mean() * 100
    total_pnl = log_df["pnl_pct"].sum()
    n_trades  = (log_df["signal"] == "BUY").sum()
    win_rate  = log_df.loc[log_df["signal"] == "BUY", "correct"].mean() * 100

    logger.info(f"Paper Trade Summary ({days} days):")
    logger.info(f"  Directional Accuracy : {dir_acc:.1f}%")
    logger.info(f"  Total PnL (%)        : {total_pnl:+.2f}%")
    logger.info(f"  Trades Taken         : {n_trades}")
    logger.info(f"  Win Rate on Trades   : {win_rate:.1f}%")

    os.makedirs("outputs", exist_ok=True)
    log_df.to_csv(LOG_PATH, index=False)
    logger.info(f"Log saved -> {LOG_PATH}")

    return log_df


def paper_trade_summary(log_df: pd.DataFrame) -> dict:
    """Return summary metrics from a paper trade log DataFrame."""
    if log_df.empty:
        return {}
    buy_rows = log_df[log_df["signal"] == "BUY"]
    return {
        "days_simulated":   len(log_df),
        "dir_accuracy_pct": round(log_df["correct"].mean() * 100, 2),
        "n_trades":         int((log_df["signal"] == "BUY").sum()),
        "win_rate_pct":     round(buy_rows["correct"].mean() * 100, 2) if len(buy_rows) else 0.0,
        "total_pnl_pct":    round(log_df["pnl_pct"].sum(), 2),
        "avg_pnl_per_trade":round(buy_rows["pnl_pct"].mean(), 4) if len(buy_rows) else 0.0,
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="Paper trading simulation")
    parser.add_argument("--days", type=int, default=90,
                        help="Number of trading days to simulate (default: 90)")
    args = parser.parse_args()

    log = run_paper_trade(days=args.days)
    if not log.empty:
        summary = paper_trade_summary(log)
        print("\n=== Paper Trade Summary ===")
        for k, v in summary.items():
            print(f"  {k}: {v}")
        print(f"\nFull log saved to {LOG_PATH}")
