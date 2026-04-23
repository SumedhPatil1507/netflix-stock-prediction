"""
FastAPI prediction service.
Run with: uvicorn api.main:app --reload
"""
import os
import sys
import logging
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.modeling import FEATURES
from src.feature_utils import compute_features_from_ohlcv

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Netflix Stock Prediction API",
    description="Predicts next-day return (%) and price for NFLX using a stacking ensemble.",
    version="1.0.0",
)

# ── Load model once at startup ────────────────────────────────────────────────
MODEL_PATH = os.path.join(ROOT, "models", "model.pkl")
try:
    _model = joblib.load(MODEL_PATH)
    logger.info("Model loaded successfully")
except Exception as e:
    _model = None
    logger.error(f"Model load failed: {e}")


class OHLCVRow(BaseModel):
    open:   float = Field(..., example=645.0)
    high:   float = Field(..., example=655.0)
    low:    float = Field(..., example=640.0)
    close:  float = Field(..., example=650.0)
    volume: float = Field(..., example=5000000)


class PredictRequest(BaseModel):
    rows: List[OHLCVRow] = Field(..., min_items=10,
        description="Last N trading days of OHLCV data, most recent last")


class PredictResponse(BaseModel):
    predicted_return_pct: float
    predicted_next_close: float
    last_close:           float
    signal:               str   # "BUY" | "HOLD"


def _compute_features(rows: List[OHLCVRow]) -> pd.DataFrame:
    df = pd.DataFrame([r.dict() for r in rows])
    df.columns = ["Open", "High", "Low", "Close", "Volume"]
    return compute_features_from_ohlcv(df)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run python main.py first.")

    try:
        df = _compute_features(request.rows)
        row         = df[FEATURES].iloc[[-1]]
        pred_return = float(_model.predict(row)[0])
        last_close  = request.rows[-1].close
        pred_price  = last_close * (1 + pred_return / 100)
        signal      = "BUY" if pred_return > 0 else "HOLD"

        return PredictResponse(
            predicted_return_pct  = round(pred_return, 4),
            predicted_next_close  = round(pred_price, 2),
            last_close            = last_close,
            signal                = signal,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/features")
def features():
    return {"features": FEATURES, "count": len(FEATURES)}
