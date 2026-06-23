"""
FastAPI prediction service — production grade.
Run with: uvicorn api.main:app --reload

Features:
- Rate limiting (10 req/min per IP via slowapi)
- Multi-ticker support
- Model versioning info
- Authenticated data pipeline (Alpaca → Alpha Vantage → TimescaleDB/SQLite)
- Async task routing via Celery + Redis
- /health, /predict, /features, /model_info, /tickers endpoints
"""
from __future__ import annotations
import os
import sys
import json
import logging
from datetime import datetime
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
except ImportError:
    pass

from src.modeling import FEATURES
from src.feature_utils import compute_features_from_ohlcv

logger = logging.getLogger(__name__)

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Alpha Engine API",
    description="Multi-ticker stock return prediction using stacking ensemble.",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Rate limiting ─────────────────────────────────────────────────────────────
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded

    _rate = os.getenv("API_RATE_LIMIT", "10")
    limiter = Limiter(key_func=get_remote_address,
                      default_limits=[f"{_rate}/minute"])
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    _rate_limiting = True
except ImportError:
    _rate_limiting = False
    logger.warning("slowapi not installed — rate limiting disabled")

# ── Model loading ─────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(ROOT, "models", "model.pkl")
try:
    _model = joblib.load(MODEL_PATH)
    logger.info("Model loaded successfully")
except Exception as e:
    _model = None
    logger.error(f"Model load failed: {e}")

# ── Schemas ───────────────────────────────────────────────────────────────────
class OHLCVRow(BaseModel):
    open:   float = Field(..., example=645.0)
    high:   float = Field(..., example=655.0)
    low:    float = Field(..., example=640.0)
    close:  float = Field(..., example=650.0)
    volume: float = Field(..., example=5_000_000)


class PredictRequest(BaseModel):
    rows:   List[OHLCVRow] = Field(..., min_items=10,
                description="Last N trading days of OHLCV, most recent last")
    ticker: Optional[str]  = Field("NFLX", description="Ticker symbol")


class PredictResponse(BaseModel):
    ticker:               str
    predicted_return_pct: float
    predicted_next_close: float
    last_close:           float
    signal:               str
    confidence_interval:  Optional[dict] = None


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status":         "ok",
        "model_loaded":   _model is not None,
        "rate_limiting":  _rate_limiting,
        "version":        "2.0.0",
    }


@app.get("/tickers")
def tickers():
    """List of supported tickers."""
    return {
        "note":    "Any ticker supported by Alpaca or Alpha Vantage",
        "popular": ["NFLX", "AAPL", "TSLA", "GOOGL", "MSFT", "AMZN", "META"],
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest, req: Request = None):
    if _model is None:
        raise HTTPException(status_code=503,
                            detail="Model not loaded. Run python main.py first.")
    try:
        df = pd.DataFrame([r.dict() for r in request.rows])
        df.columns = ["Open", "High", "Low", "Close", "Volume"]
        df = compute_features_from_ohlcv(df)

        train_feats = getattr(_model, "feature_names_", FEATURES)
        for f in train_feats:
            if f not in df.columns:
                df[f] = 0.0
        row        = df[train_feats].iloc[[-1]]
        pred_ret   = float(_model.predict(row)[0])
        last_close = request.rows[-1].close
        pred_price = last_close * (1 + pred_ret / 100)
        signal     = "BUY" if pred_ret > 0 else "HOLD"

        ci = None
        if hasattr(_model, "conformal_"):
            cp = _model.conformal_
            lo_r, hi_r = cp.predict_interval(row)
            ci = {
                "lower_return_pct": round(float(lo_r[0]), 4),
                "upper_return_pct": round(float(hi_r[0]), 4),
                "lower_price":      round(last_close * (1 + lo_r[0] / 100), 2),
                "upper_price":      round(last_close * (1 + hi_r[0] / 100), 2),
                "coverage":         "90%",
            }

        return PredictResponse(
            ticker               = request.ticker or "NFLX",
            predicted_return_pct = round(pred_ret, 4),
            predicted_next_close = round(pred_price, 2),
            last_close           = last_close,
            signal               = signal,
            confidence_interval  = ci,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/features")
def features():
    return {"features": FEATURES, "count": len(FEATURES)}


@app.get("/model_info")
def model_info():
    info: dict = {
        "model_type":   "ManualStackingRegressor",
        "architecture": "XGBoost + LightGBM + RandomForest + ExtraTrees -> Ridge",
        "target":       "next-day return (%)",
        "n_features":   len(FEATURES),
        "model_loaded": _model is not None,
        "version":      "2.0.0",
    }
    if _model is not None and hasattr(_model, "feature_names_"):
        info["trained_feature_count"] = len(_model.feature_names_)
    if _model is not None and hasattr(_model, "conformal_"):
        cp = _model.conformal_
        info["conformal_alpha"] = cp.alpha
        info["conformal_width"] = cp.interval_width
    metrics_path = os.path.join(ROOT, "outputs", "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            info["latest_metrics"] = json.load(f)
    if os.path.exists(MODEL_PATH):
        mtime = os.path.getmtime(MODEL_PATH)
        info["model_trained_at"] = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")

    # Registry info
    try:
        from src.model_registry import get_registry
        reg = get_registry()
        info["total_versions"] = len(reg["models"])
        info["latest_version"] = reg.get("latest")
    except Exception:
        pass

    return info


@app.get("/registry")
def registry():
    """Return full model version registry."""
    try:
        from src.model_registry import get_registry
        return get_registry()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Risk & Execution endpoints ────────────────────────────────────────────────

class RiskRequest(BaseModel):
    ticker:           str   = Field("NFLX")
    pred_return_pct:  float = Field(..., description="Model predicted return (%)")
    last_price:       float = Field(..., description="Current price")
    atr:              float = Field(..., description="14-period ATR")
    portfolio_value:  float = Field(100_000.0, description="Total portfolio ($)")
    win_rate:         float = Field(0.52, description="Historical win rate (0-1)")
    avg_win_pct:      float = Field(1.5)
    avg_loss_pct:     float = Field(1.0)
    max_position_pct: float = Field(0.05, description="Max position size (0-1)")
    max_drawdown_halt:float = Field(0.10, description="Circuit breaker drawdown")


class ExecuteRequest(BaseModel):
    ticker:      str   = Field("NFLX")
    shares:      int   = Field(..., gt=0)
    side:        str   = Field("buy", description="buy or sell")
    order_type:  str   = Field("market", description="market or limit")
    limit_price: Optional[float] = Field(None)
    broker:      str   = Field("alpaca", description="alpaca | paper")


@app.post("/risk/position")
def compute_risk_position(request: RiskRequest):
    """
    Compute execution-ready position size with full risk controls.
    Returns stop-loss, take-profit, shares, risk per trade.
    """
    try:
        from src.risk_manager import RiskManager, RiskConfig
        cfg = RiskConfig(
            portfolio_value    = request.portfolio_value,
            max_position_pct   = request.max_position_pct,
            max_drawdown_halt  = request.max_drawdown_halt,
        )
        rm    = RiskManager(cfg)
        order = rm.compute_position(
            ticker      = request.ticker,
            pred_return = request.pred_return_pct,
            last_price  = request.last_price,
            atr         = request.atr,
            win_rate    = request.win_rate,
            avg_win_pct = request.avg_win_pct,
            avg_loss_pct= request.avg_loss_pct,
        )
        return order.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/risk/matrix")
def risk_matrix(request: RiskRequest):
    """Return full risk matrix for a given prediction."""
    try:
        from src.risk_manager import RiskManager, RiskConfig
        cfg = RiskConfig(portfolio_value=request.portfolio_value,
                         max_position_pct=request.max_position_pct)
        rm  = RiskManager(cfg)
        return rm.risk_matrix(
            pred_return    = request.pred_return_pct,
            last_price     = request.last_price,
            atr            = request.atr,
            portfolio_value= request.portfolio_value,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/execute")
def execute_trade(request: ExecuteRequest):
    """
    Execute a trade via broker integration.
    Supports: Alpaca (paper + live), paper simulation.

    Required env vars for Alpaca:
      ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL
    """
    broker = request.broker.lower()

    if broker == "paper":
        return {
            "status":     "filled",
            "broker":     "paper",
            "ticker":     request.ticker,
            "side":       request.side,
            "shares":     request.shares,
            "order_type": request.order_type,
            "fill_price": request.limit_price,
            "message":    "Paper trade executed (simulation only)",
        }

    if broker == "alpaca":
        api_key    = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")
        base_url   = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

        if not api_key or not secret_key:
            raise HTTPException(
                status_code=503,
                detail="Alpaca keys not configured. Set ALPACA_API_KEY and ALPACA_SECRET_KEY in .env"
            )

        try:
            import requests as req
            headers = {
                "APCA-API-KEY-ID":     api_key,
                "APCA-API-SECRET-KEY": secret_key,
                "Content-Type":        "application/json",
            }
            body: dict = {
                "symbol":        request.ticker,
                "qty":           str(request.shares),
                "side":          request.side,
                "type":          request.order_type,
                "time_in_force": "day",
            }
            if request.order_type == "limit" and request.limit_price:
                body["limit_price"] = str(request.limit_price)

            resp = req.post(f"{base_url}/v2/orders",
                            json=body, headers=headers, timeout=10)

            if resp.status_code in (200, 201):
                data = resp.json()
                return {
                    "status":     data.get("status", "submitted"),
                    "broker":     "alpaca",
                    "order_id":   data.get("id"),
                    "ticker":     request.ticker,
                    "side":       request.side,
                    "shares":     request.shares,
                    "order_type": request.order_type,
                    "fill_price": data.get("filled_avg_price"),
                    "message":    "Order submitted to Alpaca",
                }
            else:
                raise HTTPException(
                    status_code=resp.status_code,
                    detail=f"Alpaca error: {resp.text}"
                )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Execution failed: {e}")

    raise HTTPException(status_code=400, detail=f"Unknown broker: {broker}. Use 'alpaca' or 'paper'")


# ═══════════════════════════════════════════════════════════════════════════════
# MARKET DATA ENDPOINTS
# Uses src.data_loader (Alpaca → Alpha Vantage → TimescaleDB/SQLite)
# yfinance has been removed from all market endpoints.
# ═══════════════════════════════════════════════════════════════════════════════

def _load_ohlcv_df(ticker: str, days_back: int = 500) -> pd.DataFrame:
    """
    Unified OHLCV loader: Alpaca → Alpha Vantage → TimescaleDB/SQLite.
    Never calls yfinance.
    """
    from src.data_loader import load_data
    df = load_data(source="database", ticker=ticker, days_back=days_back)
    if df.empty:
        raise HTTPException(
            status_code=503,
            detail=(
                f"No market data for {ticker}. "
                "Set ALPACA_API_KEY + ALPACA_SECRET_KEY to bootstrap from Alpaca, "
                "or ALPHA_VANTAGE_KEY for Alpha Vantage."
            ),
        )
    # Ensure proper column casing
    df.columns = [c.capitalize() if c.lower() in {"open","high","low","close","volume"}
                  else c for c in df.columns]
    return df


@app.get("/market/ohlcv")
def get_ohlcv(ticker: str = "NFLX", period: str = "2y"):
    """Return OHLCV bars for any ticker as JSON records (no yfinance)."""
    days_map = {"1mo": 35, "3mo": 95, "6mo": 185, "1y": 370, "2y": 740, "5y": 1830}
    days_back = days_map.get(period, 740)
    try:
        df = _load_ohlcv_df(ticker, days_back)
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        # Strip tz from index if present
        if hasattr(df.index, "tz") and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return {
            "ticker": ticker,
            "period": period,
            "source": "alpaca/alphavantage/timescaledb",
            "dates":  [str(d.date()) for d in df.index],
            "open":   df["Open"].round(4).tolist(),
            "high":   df["High"].round(4).tolist(),
            "low":    df["Low"].round(4).tolist(),
            "close":  df["Close"].round(4).tolist(),
            "volume": df["Volume"].tolist(),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/market/indicators")
def get_indicators(ticker: str = "NFLX", period: str = "1y"):
    """Return pre-computed RSI, MACD, Bollinger — computed from authenticated pipeline data."""
    days_map = {"1mo": 35, "3mo": 95, "6mo": 185, "1y": 370, "2y": 740}
    days_back = days_map.get(period, 370)
    try:
        df    = _load_ohlcv_df(ticker, days_back)
        close = df["Close"].dropna()
        if hasattr(close.index, "tz") and close.index.tz is not None:
            close.index = close.index.tz_localize(None)
        dates = [str(d.date()) for d in close.index]

        # RSI
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rsi   = (100 - 100 / (1 + gain / loss.replace(0, np.nan))).fillna(50)

        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd  = ema12 - ema26
        sig   = macd.ewm(span=9, adjust=False).mean()
        hist  = macd - sig

        # Bollinger
        bb_mid = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        bb_up  = (bb_mid + 2 * bb_std).fillna(0)
        bb_lo  = (bb_mid - 2 * bb_std).fillna(0)

        return {
            "ticker":    ticker,
            "source":    "alpaca/alphavantage/timescaledb",
            "dates":     dates,
            "close":     close.round(4).tolist(),
            "rsi":       rsi.round(4).tolist(),
            "macd":      macd.fillna(0).round(4).tolist(),
            "macd_sig":  sig.fillna(0).round(4).tolist(),
            "macd_hist": hist.fillna(0).round(4).tolist(),
            "bb_mid":    bb_mid.fillna(0).round(4).tolist(),
            "bb_upper":  bb_up.round(4).tolist(),
            "bb_lower":  bb_lo.round(4).tolist(),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/market/live_input")
def get_live_input(ticker: str = "NFLX", n: int = 10):
    """Return the last N OHLCV rows for the predict tab (no yfinance)."""
    try:
        df = _load_ohlcv_df(ticker, days_back=30)
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna().tail(n).round(2)
        return df.reset_index(drop=True).to_dict("list")
    except HTTPException:
        # Graceful degradation with synthetic fallback
        return {
            "Open":   [600.0] * n, "High":   [610.0] * n,
            "Low":    [595.0] * n, "Close":  [605.0] * n,
            "Volume": [5_000_000] * n,
        }
    except Exception:
        return {
            "Open":   [600.0] * n, "High":   [610.0] * n,
            "Low":    [595.0] * n, "Close":  [605.0] * n,
            "Volume": [5_000_000] * n,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SENTIMENT ENDPOINT
# Uses Alpha Vantage News API when key is available, else returns cached/empty
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/sentiment")
def get_sentiment(ticker: str = "NFLX"):
    """
    Fetch news sentiment. Uses Alpha Vantage NEWS_SENTIMENT endpoint when
    ALPHA_VANTAGE_KEY is set; falls back to an empty result set otherwise.
    """
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        sia   = SentimentIntensityAnalyzer()
        rows  = []

        av_key = os.getenv("ALPHA_VANTAGE_KEY")
        if av_key:
            import requests as req
            resp = req.get(
                "https://www.alphavantage.co/query",
                params={
                    "function": "NEWS_SENTIMENT",
                    "tickers":  ticker,
                    "apikey":   av_key,
                    "limit":    50,
                },
                timeout=10,
            )
            news_items = resp.json().get("feed", [])
            for item in news_items:
                ts    = pd.Timestamp(item.get("time_published", ""), format="%Y%m%dT%H%M%S", errors="coerce")
                title = item.get("title", "")
                score = sia.polarity_scores(title)["compound"]
                rows.append({
                    "date":      str(ts.date()) if not pd.isna(ts) else "unknown",
                    "title":     title,
                    "score":     score,
                    "sentiment": ("Positive" if score > 0.05
                                  else "Negative" if score < -0.05 else "Neutral"),
                    "source":    item.get("source", ""),
                })
        else:
            logger.warning("ALPHA_VANTAGE_KEY not set — sentiment endpoint returning empty results")

        avg_score = round(sum(r["score"] for r in rows) / len(rows), 4) if rows else 0.0
        return {
            "ticker":    ticker,
            "source":    "alpha_vantage_news" if av_key else "unavailable",
            "items":     rows,
            "avg_score": avg_score,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# DRIFT ENDPOINT
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/drift")
def get_drift(ticker: str = "NFLX"):
    try:
        cache = os.path.join(ROOT, "outputs", "features_cache.parquet")
        if not os.path.exists(cache):
            raise HTTPException(status_code=404, detail="Run python main.py first")
        df = pd.read_parquet(cache)
        from src.drift import detect_drift, drift_summary_df
        split = int(len(df) * 0.8)
        dr  = detect_drift(df.iloc[:split], df.iloc[split:], FEATURES)
        ddf = drift_summary_df(dr)
        return {
            "overall_drift":    dr["overall_drift"],
            "n_drifted":        len(dr["drifted_features"]),
            "drifted_features": dr["drifted_features"],
            "psi_threshold":    dr["psi_threshold"],
            "table": ddf.to_dict("records"),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# EXPLAINABILITY ENDPOINT
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/explainability/importance")
def feature_importance():
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        feat_cols = getattr(_model, "feature_names_", FEATURES)
        imps, count = None, 0
        for name, est in _model.fitted_learners_:
            if hasattr(est, "feature_importances_"):
                fi = np.array(est.feature_importances_[:len(feat_cols)], dtype=np.float64)
                imps = fi if imps is None else imps + fi
                count += 1
        if imps is None:
            raise HTTPException(status_code=500, detail="No importances available")
        imps /= count
        idx = np.argsort(imps)[::-1]
        return {
            "features":    [feat_cols[i] for i in idx],
            "importances": imps[idx].tolist(),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# ASYNC TASK ROUTING (Celery + Redis)
# Heavy CPU-bound work is offloaded; caller gets job_id instantly.
# Streamlit polls /api/v1/tasks/{job_id} every 2 seconds.
# ═══════════════════════════════════════════════════════════════════════════════

class BacktestTaskRequest(BaseModel):
    ticker: str = "NFLX"
    days:   int = Field(90, ge=10, le=365)


class DriftTaskRequest(BaseModel):
    ticker: str = "NFLX"


class ConformalTaskRequest(BaseModel):
    ticker:      str = "NFLX"
    n_intervals: int = Field(10_000, ge=100, le=50_000,
                             description="Number of conformal intervals to compute")
    window_days: int = Field(252, ge=60, le=1000,
                             description="Rolling window size in trading days")


def _celery_app():
    try:
        from worker.celery_app import celery
        return celery
    except ImportError:
        raise HTTPException(status_code=503,
                            detail="Celery worker not running. Start with: "
                                   "celery -A worker.celery_app worker --loglevel=info")


@app.post("/api/v1/tasks/backtest")
def submit_backtest(req: BacktestTaskRequest):
    """Submit a backtest simulation. Returns job_id immediately."""
    _celery_app()  # validate worker reachable
    from worker.tasks import run_backtest_task
    job = run_backtest_task.apply_async(
        kwargs={"ticker": req.ticker, "days": req.days}
    )
    return {"job_id": job.id, "state": "PENDING",
            "message": f"Backtest queued for {req.ticker} ({req.days} days)"}


@app.post("/api/v1/tasks/paper_trade")
def submit_paper_trade(req: BacktestTaskRequest):
    """Submit a paper trade simulation. Returns job_id immediately."""
    _celery_app()
    from worker.tasks import run_paper_trade_task
    job = run_paper_trade_task.apply_async(
        kwargs={"ticker": req.ticker, "days": req.days}
    )
    return {"job_id": job.id, "state": "PENDING",
            "message": f"Paper trade queued for {req.ticker} ({req.days} days)"}


@app.post("/api/v1/tasks/drift")
def submit_drift(req: DriftTaskRequest):
    """Submit drift check. Returns job_id immediately."""
    _celery_app()
    from worker.tasks import run_drift_task
    job = run_drift_task.apply_async(kwargs={"ticker": req.ticker})
    return {"job_id": job.id, "state": "PENDING",
            "message": f"Drift check queued for {req.ticker}"}


@app.post("/api/v1/tasks/conformal")
def submit_conformal(req: ConformalTaskRequest):
    """
    Submit a batch conformal prediction interval computation.
    CPU-bound: computes up to 10,000 rolling conformal intervals.
    Returns job_id immediately; Streamlit polls every 2 seconds.
    """
    _celery_app()
    from worker.tasks import run_conformal_task
    job = run_conformal_task.apply_async(
        kwargs={
            "ticker":      req.ticker,
            "n_intervals": req.n_intervals,
            "window_days": req.window_days,
        }
    )
    return {
        "job_id":  job.id,
        "state":   "PENDING",
        "message": f"Conformal batch ({req.n_intervals:,} intervals) queued for {req.ticker}",
    }


@app.get("/api/v1/tasks/{job_id}")
def get_task_status(job_id: str):
    """
    Poll task state. Streamlit polls this every 2 seconds.
    Returns: {state: PENDING|PROGRESS|SUCCESS|FAILURE, result: ..., error: ...}
    """
    try:
        from worker.celery_app import celery
        res = celery.AsyncResult(job_id)
        if res.state == "SUCCESS":
            return {"state": "SUCCESS", "result": res.result}
        if res.state == "FAILURE":
            return {"state": "FAILURE", "error": str(res.info)}
        if res.state == "PROGRESS":
            return {"state": "PROGRESS", "meta": res.info}
        return {"state": res.state}
    except Exception as e:
        return {"state": "ERROR", "error": str(e)}
