"""
Celery tasks — CPU-bound work offloaded from FastAPI.
Each task returns a serialisable dict that Streamlit can render.

Tasks:
  run_backtest_task    — 90-day strategy backtest with Kelly sizing
  run_paper_trade_task — day-by-day paper trading simulation
  run_drift_task       — PSI + KS feature drift detection
  run_conformal_task   — batch conformal prediction intervals (up to 10,000)
"""
from __future__ import annotations
import logging
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from worker.celery_app import celery

logger = logging.getLogger(__name__)


@celery.task(bind=True, name="run_backtest")
def run_backtest_task(self, ticker: str = "NFLX", days: int = 90) -> dict:
    """Run Kelly backtesting simulation for N days."""
    self.update_state(state="PROGRESS", meta={"step": "loading model", "pct": 5})
    try:
        import numpy as np
        import pandas as pd
        from src.paper_trade import run_paper_trade, paper_trade_summary
        from src.backtest import run_backtest

        self.update_state(state="PROGRESS", meta={"step": "fetching data", "pct": 20})
        log_df = run_paper_trade(days=days)

        self.update_state(state="PROGRESS", meta={"step": "running backtest", "pct": 70})
        summary = paper_trade_summary(log_df)

        y_ret  = log_df["actual_return"].values
        p_ret  = log_df["pred_return"].values
        bt     = run_backtest(pd.Series(y_ret), p_ret)

        self.update_state(state="PROGRESS", meta={"step": "serialising results", "pct": 95})
        return {
            "ticker":        ticker,
            "days":          days,
            "summary":       summary,
            "bt_metrics":    bt["metrics"],
            "curves":        bt["curves"].to_dict("list"),
            "rolling_sharpe":bt["rolling_sharpe"].tolist(),
            "log":           log_df[[
                "date", "prev_close", "next_close",
                "pred_return", "actual_return",
                "signal", "direction", "correct", "pnl_pct"
            ]].to_dict("records"),
        }
    except Exception as e:
        self.update_state(state="FAILURE", meta={"error": str(e)})
        raise


@celery.task(bind=True, name="run_paper_trade")
def run_paper_trade_task(self, ticker: str = "NFLX", days: int = 90) -> dict:
    """Day-by-day paper trading simulation."""
    self.update_state(state="PROGRESS", meta={"step": "simulating", "pct": 10})
    try:
        from src.paper_trade import run_paper_trade, paper_trade_summary
        log_df  = run_paper_trade(days=days)
        summary = paper_trade_summary(log_df)
        log_df["cum_pnl"] = log_df["pnl_pct"].cumsum()
        return {
            "ticker":  ticker,
            "days":    days,
            "summary": summary,
            "log":     log_df.to_dict("records"),
        }
    except Exception as e:
        self.update_state(state="FAILURE", meta={"error": str(e)})
        raise


@celery.task(bind=True, name="run_drift_check")
def run_drift_task(self, ticker: str = "NFLX") -> dict:
    """PSI + KS drift detection on full feature set."""
    self.update_state(state="PROGRESS", meta={"step": "loading features", "pct": 10})
    try:
        import pandas as pd
        from src.drift import detect_drift, drift_summary_df
        from src.modeling import FEATURES

        cache = os.path.join(ROOT, "outputs", "features_cache.parquet")
        df    = pd.read_parquet(cache)
        split = int(len(df) * 0.8)

        self.update_state(state="PROGRESS", meta={"step": "computing PSI + KS", "pct": 60})
        dr  = detect_drift(df.iloc[:split], df.iloc[split:], FEATURES)
        ddf = drift_summary_df(dr)

        return {
            "ticker":           ticker,
            "overall_drift":    dr["overall_drift"],
            "n_drifted":        len(dr["drifted_features"]),
            "drifted_features": dr["drifted_features"],
            "table":            ddf.to_dict("records"),
        }
    except Exception as e:
        self.update_state(state="FAILURE", meta={"error": str(e)})
        raise


@celery.task(bind=True, name="run_conformal")
def run_conformal_task(
    self,
    ticker: str = "NFLX",
    n_intervals: int = 10_000,
    window_days: int = 252,
) -> dict:
    """
    Batch conformal prediction interval computation.

    For each of the last `n_intervals` data points (or however many are
    available), uses the trained model's attached ConformalPredictor to compute
    the 90% prediction interval. Returns lower/upper bounds + coverage stats.

    This is CPU-bound because it runs model.predict() thousands of times.
    """
    self.update_state(state="PROGRESS", meta={"step": "loading model & data", "pct": 5})
    try:
        import numpy as np
        import pandas as pd
        import joblib
        from src.feature_utils import compute_features_from_ohlcv
        from src.modeling import FEATURES, load_model
        from src.data_loader import load_data

        # ── Load model ──────────────────────────────────────────────────────
        model = load_model()
        if not hasattr(model, "conformal_"):
            return {
                "error":   "Model has no attached ConformalPredictor. Retrain with main.py.",
                "ticker":  ticker,
                "n_computed": 0,
            }

        cp = model.conformal_
        self.update_state(state="PROGRESS", meta={"step": "fetching market data", "pct": 15})

        # ── Load historical data ─────────────────────────────────────────────
        df_raw = load_data(source="database", ticker=ticker,
                           days_back=max(window_days + n_intervals + 100, 1500))
        if df_raw.empty:
            return {"error": f"No data for {ticker}", "n_computed": 0}

        # ── Compute features for full series ─────────────────────────────────
        self.update_state(state="PROGRESS", meta={"step": "computing features", "pct": 30})
        feat_df = compute_features_from_ohlcv(
            df_raw[["Open", "High", "Low", "Close", "Volume"]]
        )

        train_feats = getattr(model, "feature_names_", FEATURES)
        for f in train_feats:
            if f not in feat_df.columns:
                feat_df[f] = 0.0

        X_all = feat_df[train_feats].dropna()
        # Use at most the last n_intervals rows
        X_all = X_all.tail(n_intervals)
        actual_n = len(X_all)

        # ── Batch prediction in chunks of 500 (progress reporting) ──────────
        self.update_state(state="PROGRESS",
                          meta={"step": f"computing {actual_n:,} intervals", "pct": 40})

        chunk_size = 500
        lowers, uppers, preds = [], [], []
        for i in range(0, actual_n, chunk_size):
            chunk = X_all.iloc[i: i + chunk_size]
            lo, hi = cp.predict_interval(chunk.values)
            pred   = model.predict(chunk)
            lowers.extend(lo.tolist())
            uppers.extend(hi.tolist())
            preds.extend(pred.tolist())
            pct = 40 + int((i / actual_n) * 50)
            self.update_state(state="PROGRESS",
                              meta={"step": f"computed {i+len(chunk):,}/{actual_n:,}", "pct": pct})

        dates = [str(d.date()) for d in X_all.index]

        # ── Coverage stats ───────────────────────────────────────────────────
        self.update_state(state="PROGRESS", meta={"step": "computing coverage stats", "pct": 95})
        widths    = [u - l for u, l in zip(uppers, lowers)]
        avg_width = float(np.mean(widths))

        return {
            "ticker":       ticker,
            "n_computed":   actual_n,
            "window_days":  window_days,
            "alpha":        cp.alpha,
            "coverage_pct": f"{(1 - cp.alpha) * 100:.0f}%",
            "avg_width":    round(avg_width, 4),
            "dates":        dates,
            "predictions":  [round(p, 4) for p in preds],
            "lower_bounds": [round(l, 4) for l in lowers],
            "upper_bounds": [round(u, 4) for u in uppers],
            "interval_widths": [round(w, 4) for w in widths],
        }
    except Exception as e:
        self.update_state(state="FAILURE", meta={"error": str(e)})
        raise
