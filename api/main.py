"""
FastAPI prediction service — production grade.
Run with: uvicorn api.main:app --reload

Features:
- Rate limiting (10 req/min per IP via slowapi)
- Multi-ticker support
- Model versioning info
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
    """List of supported tickers (any valid yfinance symbol)."""
    return {
        "note":    "Any valid Yahoo Finance ticker is supported via live data",
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
